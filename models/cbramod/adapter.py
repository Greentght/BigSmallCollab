"""Adapter for the *final settled* CBraMod teacher — the 45ch channel-template
pipeline (PROGRESS.md 2026-07-01 全量终表, port of ``MIRepNet/cbramod_template.py``).

Pipeline per set (the paper's unified preprocessing for all baselines):
EA (per-set whitening, transductive) -> inverse-distance pad to the 45-ch
template -> 250 Hz, then truncate 1000 samples, resample 800 @200 Hz,
``/ scale`` (from config), reshape ``(N, 45, 4, 200)``. Official CBraMod
``all_patch_reps`` 3-layer head. The canonical training recipe lives in
``configs/models/cbramod.yaml``; adapter code only consumes and validates it.

Reported (3seed, 80/20 single-session): 14001-2 77.78, 14001-4 62.07,
004 74.38, AlexMI 66.15, 15001 71.11 — all at/above the paper except
004 (−3.01). The earlier CAR-only native pipeline is superseded (still
available as an ablation knob in ``experiments/bigmodel/cbramod_adapt.py
--pipeline native``).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset

import config
from models.base import ModelAdapter


class _CBraModModel(nn.Module):
    """Pretrained backbone + official all_patch_reps 3-layer MLP head."""

    def __init__(self, num_classes, n_ch=45, n_patch=4, dropout=0.1, pretrain=None):
        super().__init__()
        from einops.layers.torch import Rearrange
        from .cbramod import CBraMod
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


class CBraModAdapter(ModelAdapter):
    name = 'cbramod'

    REQUIRED_MODEL_CONFIG = (
        'lr', 'epochs', 'weight_decay', 'batch_size',
        'dropout', 'scale', 'label_smoothing',
    )

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        missing = [k for k in self.REQUIRED_MODEL_CONFIG if k not in self.cfg]
        if missing:
            raise KeyError(
                'CBraModAdapter missing model config keys: '
                f'{", ".join(missing)}. Load configs/models/cbramod.yaml via '
                "config.load_model_config('cbramod') and pass it to get_adapter()."
            )
        from data.channels import (
            use_channels_names, BNCI2014001_chn_names,
            BNCI2014004_chn_names, BNCI2015001_chn_names, AlexMI_chn_names)
        from data.preproc import EA, pad_missing_channels_diff
        self._EA = EA
        self._pad = pad_missing_channels_diff
        self._template = use_channels_names
        self._src_channels = {
            'BNCI2014001': BNCI2014001_chn_names,
            'BNCI2014001-4': BNCI2014001_chn_names,
            'BNCI2014004': BNCI2014004_chn_names,
            'BNCI2015001': BNCI2015001_chn_names,
            'AlexMI': AlexMI_chn_names,
        }

    def preprocess(self, X_raw):
        """(N, C, 1000)@250Hz -> (N, 45, 4, 200). EA per-set -> 45ch template ->
        truncate 1000 -> resample 800@200Hz -> /scale."""
        # LOSO passes data already EA'd per-subject + padded to 45ch; skip to
        # avoid re-whitening the mixed multi-subject set with one covariance.
        if self.cfg.get('skip_preprocess'):
            return torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        ds = self.cfg['dataset_name']
        x = np.asarray(X_raw, dtype=np.float32)
        x = self._EA(x).astype(np.float32)                       # per-set whitening
        x = self._pad(x, self._template, self._src_channels[ds])  # -> 45 ch
        x = x[:, :, :1000]
        x = resample(x, 800, axis=-1)                            # -> 200 Hz
        x = (x / float(self.cfg['scale'])).astype(np.float32)
        n, ch, _ = x.shape
        return torch.as_tensor(x.reshape(n, ch, 4, 200), dtype=torch.float32)

    def ea_pad_per_subject(self, X_raw, subj_ids):
        """EA per subject-group + 45ch pad, then concatenate — so each subject is
        whitened by its own reference covariance (LOSO teacher preprocessing)."""
        ds = self.cfg['dataset_name']
        X = np.asarray(X_raw, dtype=np.float32)
        subj_ids = np.asarray(subj_ids)
        out = None
        for s in np.unique(subj_ids):
            m = subj_ids == s
            xs = self._pad(self._EA(X[m]).astype('float32'),
                           self._template, self._src_channels[ds])
            if out is None:
                out = np.empty((len(X), xs.shape[1], xs.shape[2]), dtype=np.float32)
            out[m] = xs
        return out

    def build(self, num_classes):
        pretrain = self.cfg.get('pretrain') or config.weight_path('cbramod')
        model = _CBraModModel(num_classes, dropout=float(self.cfg['dropout']),
                              pretrain=pretrain)
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)

    def finetune(self, model, X_tr, y_tr, num_classes):
        """45ch-template recipe: config-driven AdamW + cosine, no clip/warmup."""
        epochs = int(self.cfg['epochs'])
        lr = float(self.cfg['lr'])
        wd = float(self.cfg['weight_decay'])
        bs = int(self.cfg['batch_size'])
        ls = float(self.cfg['label_smoothing'])

        Xp = self.preprocess(X_tr)
        y = torch.as_tensor(y_tr, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xp, y), batch_size=bs, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        cr = nn.CrossEntropyLoss(label_smoothing=ls)
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                cr(model(xb)[1], yb).backward()
                opt.step()
            sch.step()
        return model


ADAPTER = CBraModAdapter
