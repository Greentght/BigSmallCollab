"""R1 — learned per-sample router/gate over two cached models' predictions.

The D0 headroom map showed a large oracle gap (+9~17%) that plain averaging and
max-confidence routing fail to capture (even hurt for the weak teacher). R1 tests
whether a *learned* gate can close it. The gate is a tiny MLP on logit-derived
features of the big + small model; it emits a per-sample weight ``alpha in [0,1]``
and the mixture ``alpha * p_big + (1-alpha) * p_small`` is trained end-to-end by
cross-entropy on the true label.

Leakage-free evaluation (nested subject-LOSO on cached artifacts, NO base retrain):
under LOSO each subject appears as the held-out test subject in exactly one fold,
where the base models never saw them — so the pooled per-fold *test* artifacts are
out-of-sample base predictions for every subject. To route subject ``t`` we train
the gate on all *other* subjects' test artifacts and apply it to ``t``. No base
model is retrained; everything reuses the D0 ``.npz`` logits.

Decision variants reported:
  soft   : mixture ``alpha*p_big + (1-alpha)*p_small`` (the trained objective)
  hard   : pick the single model favoured by the gate (``alpha >= 0.5``)
  cond   : conditional computation — default to the small model, only "wake" the
           big model when the gate says so; sweep the threshold to trace the
           wake-rate vs accuracy Pareto (report accuracy at each wake budget).
"""
import numpy as np
import torch
import torch.nn as nn


def _softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def gate_features(big_logits, small_logits):
    """Per-sample features for the gate: both logit vectors + confidence / entropy
    / energy summaries and their big−small differences. Returns (N, F) float32."""
    bp, sp = _softmax_np(big_logits), _softmax_np(small_logits)
    bconf, sconf = bp.max(1, keepdims=True), sp.max(1, keepdims=True)
    bent = -(bp * np.log(bp + 1e-12)).sum(1, keepdims=True)
    sent = -(sp * np.log(sp + 1e-12)).sum(1, keepdims=True)
    ben = np.log(np.exp(big_logits - big_logits.max(1, keepdims=True)).sum(1, keepdims=True) + 1e-12)
    sen = np.log(np.exp(small_logits - small_logits.max(1, keepdims=True)).sum(1, keepdims=True) + 1e-12)
    feats = np.concatenate([big_logits, small_logits, bconf, sconf, bent, sent,
                            bconf - sconf, bent - sent, ben - sen], axis=1)
    return feats.astype(np.float32)


class LogitGate(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)  # alpha in (0,1)


def train_gate(feats_tr, pbig_tr, psmall_tr, y_tr, seed=0, epochs=300, lr=1e-3,
               wd=1e-4, hidden=32, device='cpu'):
    """Train the gate to minimise CE of the mixture on the training pool.

    Standardises features (stats from train), returns (gate, mu, sd)."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = feats_tr.mean(0), feats_tr.std(0) + 1e-6
    Xt = torch.as_tensor((feats_tr - mu) / sd, dtype=torch.float32, device=device)
    pb = torch.as_tensor(pbig_tr, dtype=torch.float32, device=device)
    ps = torch.as_tensor(psmall_tr, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    gate = LogitGate(Xt.shape[1], hidden).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=wd)
    gate.train()
    for _ in range(epochs):
        a = gate(Xt).unsqueeze(1)                       # (N,1)
        mix = a * pb + (1 - a) * ps                     # (N,C)
        loss = nn.functional.nll_loss(torch.log(mix + 1e-12), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return gate, mu, sd


@torch.no_grad()
def gate_alpha(gate, feats, mu, sd, device='cpu'):
    gate.eval()
    X = torch.as_tensor((feats - mu) / sd, dtype=torch.float32, device=device)
    return gate(X).cpu().numpy()                        # (N,)


def apply_variants(alpha, pbig, psmall, y):
    """Return dict of per-sample predictions for soft / hard variants + arrays for
    the conditional-computation Pareto (small-default, wake big by highest alpha)."""
    soft = (alpha[:, None] * pbig + (1 - alpha[:, None]) * psmall).argmax(1)
    hard = np.where(alpha >= 0.5, pbig.argmax(1), psmall.argmax(1))
    return dict(soft=soft, hard=hard, alpha=alpha,
                pred_big=pbig.argmax(1), pred_small=psmall.argmax(1), y=y)
