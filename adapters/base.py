"""ModelAdapter — the uniform contract that hides each model's idiosyncrasies.

Every model (small specialist or big foundation model) is wrapped in an adapter
that owns three things the rest of the framework refuses to know about:

  1. **preprocess** — turn the canonical raw epoch ``(N, C_native, 1000)`` @ 250 Hz
     into whatever tensor this model eats (EA+45ch pad for MIRepNet, resample +
     patchify for CBraMod/LaBraM, identity for the small CNNs).
  2. **build** — instantiate the nn.Module, loading pretrained weights.
  3. **forward** — return a uniform ``(feat[B,D], logits[B,C])`` regardless of the
     model's native return signature.

With those, the base class provides a generic ``finetune`` loop and an ``export``
that dumps standardized artifacts. Subclasses override only what differs.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core import artifacts


class ModelAdapter(ABC):
    name = 'base'

    def __init__(self, device='cpu', **cfg):
        self.device = torch.device(device)
        self.cfg = cfg

    # --- model-specific surface (override these) ---------------------------
    @abstractmethod
    def preprocess(self, X_raw):
        """(N, C_native, 1000) float32 tensor -> model-input tensor."""

    @abstractmethod
    def build(self, num_classes):
        """Instantiate + load weights; return an nn.Module on self.device."""

    @abstractmethod
    def forward(self, model, x):
        """Run model on a preprocessed batch -> (feat[B,D], logits[B,C])."""

    # --- generic training / export (override only if needed) ---------------
    def finetune(self, model, X_tr, y_tr, num_classes):
        """Plain CE finetune on the calibration split. Returns the model.

        Hyperparameters come from ``self.cfg`` (epochs, lr, weight_decay,
        batch_size); sensible defaults match the MIRepNet kappa-track scripts.
        """
        epochs = self.cfg.get('epochs', 50)
        lr = self.cfg.get('lr', 1e-3)
        wd = self.cfg.get('weight_decay', 1e-4)
        bs = self.cfg.get('batch_size', 32)

        Xp = self.preprocess(X_tr)
        y = torch.as_tensor(y_tr, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xp, y), batch_size=bs, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                _, logits = self.forward(model, xb)
                loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return model

    @torch.no_grad()
    def infer(self, model, X):
        """Batched inference -> (feats (N,D), logits (N,C)) as numpy arrays."""
        bs = self.cfg.get('batch_size', 32)
        Xp = self.preprocess(X)
        model.eval()
        feats, logits = [], []
        for i in range(0, len(Xp), bs):
            f, lg = self.forward(model, Xp[i:i + bs].to(self.device))
            feats.append(f.cpu().numpy())
            logits.append(lg.cpu().numpy())
        return np.concatenate(feats, 0), np.concatenate(logits, 0)

    def export(self, model, X, y, dataset, subject, seed, split):
        """Run inference and persist a standardized artifact."""
        feats, logits = self.infer(model, X)
        return artifacts.save(dataset, self.name, subject, seed, split,
                              logits=logits, feats=feats, y=y)

    @torch.no_grad()
    def mc_uncertainty(self, model, X, K=20):
        """MC-dropout uncertainty on X: K stochastic forward passes with dropout
        ON (rest of the net in eval). Returns per-sample ``(pred_entropy, bald)``
        as numpy (N,). ``pred_entropy`` = entropy of the MC-averaged predictive
        distribution (total uncertainty); ``bald`` = pred_entropy − E_k[entropy]
        (mutual information = epistemic uncertainty, nonzero only where dropout
        actually perturbs the prediction). For overconfident EEG teachers that
        memorize the train split, BALD is the信号 that still flags "not sure"."""
        bs = self.cfg.get('batch_size', 32)
        Xp = self.preprocess(X)
        model.eval()
        for m in model.modules():                # re-enable dropout only
            if isinstance(m, nn.Dropout):
                m.train()
        N = len(Xp)
        p_sum = None                             # sum_k softmax  (N, C)
        ent_sum = np.zeros(N, dtype=np.float64)  # sum_k H(p_k)
        for _ in range(K):
            probs = []
            for i in range(0, N, bs):
                _, lg = self.forward(model, Xp[i:i + bs].to(self.device))
                probs.append(torch.softmax(lg, dim=1).cpu().numpy())
            p = np.concatenate(probs, 0)         # (N, C)
            p_sum = p if p_sum is None else p_sum + p
            ent_sum += -(p * np.log(p + 1e-12)).sum(1)
        p_bar = p_sum / K
        pred_entropy = -(p_bar * np.log(p_bar + 1e-12)).sum(1)
        expected_entropy = ent_sum / K
        bald = pred_entropy - expected_entropy
        return pred_entropy.astype(np.float32), bald.astype(np.float32)
