"""Adapter for the *final settled* tuned CBraMod teacher (runs in `cbramod` env).

This is the "CAR-only 高分版" teacher (PROGRESS.md 2026-07-13): official CBraMod
``all_patch_reps`` 3-layer head + CAR-only normalization (``norm=car`` = subtract
cross-channel mean, **no ÷scale division**) + zero-phase ``filtfilt`` (butter
order-4) band-pass. Removing the ÷scale division was the decisive fix that lifted
BAC to beat the EEGFMBench record on 14001_4c / 2015001. Aligned with EEGFMBench
on norm(car) / filter / target_fs / backbone / split; the big official head is
retained on purpose (EEGFMBench uses a simpler task_head).

Per-dataset tuned hyperparameters for the ``|0.7`` (70%-train, = distillation
calibration split) are embedded below from the final CAR-only sweep
(``results/cbramod_native/tuned_caronly/``), NOT the older ÷scale configs.

The canonical raw epoch ``(N, C_native, 1000)`` @ 250 Hz is fed through: CAR ->
band-pass (+notch) -> resample 200 Hz -> reshape into ``(N, ch, seconds, 200)``
patches. ``forward`` returns ``(feat_200, logits)`` where ``feat`` is the 200-d
penultimate activation (for feature-align KD).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset

from core import paths
from core.preproc import bandpass as _bandpass, notch as _notch, SRC_FS, DST_FS
from .base import ModelAdapter

# band tag -> (l_freq, h_freq, notch_freq)
_BANDS = {
    'b50': (0.3, 50.0, None),
    'b75n60': (0.3, 75.0, 60.0),
}

# Final CAR-only tuned config, |0.7 split (PROGRESS.md 2026-07-13, verified
# against results/cbramod_native/tuned_caronly/*_tp0.7.csv). scale_divisor=1
# (CAR-only), norm=car, filtfilt order-4. Keyed by framework dataset name.
_CARONLY = {
    'BNCI2014004':   dict(lr=1e-3, epochs=20, dropout=0.1, weight_decay=0.01,
                          band='b50'),
    'BNCI2014001-4': dict(lr=1e-3, epochs=50, dropout=0.1, weight_decay=0.05,
                          band='b75n60'),
    'BNCI2014001':   dict(lr=5e-4, epochs=50, dropout=0.5, weight_decay=0.05,
                          band='b75n60'),  # 14001_2c
}


class _CBraModNative(nn.Module):
    """Pretrained backbone + official all_patch_reps 3-layer MLP head."""

    def __init__(self, num_classes, n_ch, n_patch, dropout, pretrain):
        super().__init__()
        from einops.layers.torch import Rearrange
        from backbones.cbramod.cbramod import CBraMod
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30,
                                n_layer=12, nhead=8)
        if pretrain:
            self.backbone.load_state_dict(torch.load(pretrain, map_location='cpu'))
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(n_ch * n_patch * 200, 800),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(800, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        h = self.backbone(x)
        feat = self.classifier[:7](h)        # 200-d penultimate
        return feat, self.classifier[7](feat)


class CBraModNativeAdapter(ModelAdapter):
    name = 'cbramod_native'

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        tuned = _CARONLY.get(cfg.get('dataset_name'))
        for k in ('lr', 'epochs', 'dropout', 'weight_decay', 'band'):
            if tuned and k not in cfg:
                self.cfg[k] = tuned[k]

    def _seconds(self, T):
        return int(round(T / SRC_FS))

    def preprocess(self, X_raw):
        """(N, C, 1000)@250Hz -> (N, ch, seconds, 200). CAR-only: subtract
        cross-channel mean, band-pass, resample 200Hz, NO ÷scale division."""
        x = np.asarray(X_raw, dtype=np.float32)
        band = self.cfg.get('band', 'b50')
        l_freq, h_freq, notch = _BANDS[band]
        seconds = self._seconds(x.shape[2])
        x = x - x.mean(axis=1, keepdims=True)            # CAR
        x = _bandpass(x, SRC_FS, l_freq, h_freq)
        x = _notch(x, SRC_FS, notch)
        x = resample(x, seconds * DST_FS, axis=-1)       # ->200Hz (CAR-only: no /scale)
        x = np.ascontiguousarray(x, dtype=np.float32)
        n, ch, _ = x.shape
        x = x.reshape(n, ch, seconds, DST_FS)
        return torch.as_tensor(x, dtype=torch.float32)

    def build(self, num_classes):
        pretrain = self.cfg.get('pretrain') or paths.weight_path('cbramod')
        n_ch = self.cfg['in_channels']
        n_patch = self._seconds(self.cfg.get('samples', 1000))
        model = _CBraModNative(num_classes, n_ch=n_ch, n_patch=n_patch,
                               dropout=float(self.cfg.get('dropout', 0.1)),
                               pretrain=pretrain)
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)

    def finetune(self, model, X_tr, y_tr, num_classes):
        """Native training loop: AdamW + warmup-cosine, grad clip 1.0, label
        smoothing 0 — matches cbramod_native_adapt.run_subject."""
        epochs = int(self.cfg.get('epochs', 20))
        lr = float(self.cfg.get('lr', 1e-3))
        wd = float(self.cfg.get('weight_decay', 0.01))
        bs = int(self.cfg.get('batch_size', 16))
        warmup, min_lr, clip = 5, 1e-6, 1.0

        Xp = self.preprocess(X_tr)
        y = torch.as_tensor(y_tr, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xp, y), batch_size=bs, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)

        min_factor = min_lr / lr

        def lr_factor(e):
            step = e + 1
            if warmup > 0 and step <= warmup:
                return max(step / warmup, min_factor)
            denom = max(1, epochs - warmup)
            prog = (step - warmup) / denom
            cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
            return min_factor + (1.0 - min_factor) * cos

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                _, logits = self.forward(model, xb)
                loss = crit(logits, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            sched.step()
        return model
