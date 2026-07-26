"""Adapter for CBraMod (foundation model, runs in the `cbramod` conda env).

Input contract: ``(B, ch, patch_num, 200)`` at 200 Hz. The canonical epoch is
``(N, C_native, 1000)`` @ 250 Hz (= 4 s), so preprocess resamples 250->200
(1000->800 samples), optionally scales µV, and reshapes into 4 patches of 200.

CBraMod's positional encoding is convolutional, so the backbone accepts arbitrary
channel counts / patch counts — no need to match the pretrain montage. We attach
a channel-agnostic head: average-pool the patch representations to a 200-d vector
(the penultimate ``feat``) and a linear classifier on top (``logits``).

``cfg``: ``pretrain`` (path to pretrained_weights.pth), ``scale`` (divisor for µV,
default 100, matching CBraMod's convention), ``patch_num`` (default 4).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample as scipy_resample

import paths
from data.preproc import SRC_FS, DST_FS
from models.base import ModelAdapter

PATCH = 200        # samples per 1-s patch at DST_FS (model input format)


class _CBraModClassifier(nn.Module):
    """Pretrained CBraMod backbone + channel-agnostic avg-pool head."""

    def __init__(self, backbone, num_classes, d_model=200):
        super().__init__()
        self.backbone = backbone
        self.backbone.proj_out = nn.Identity()
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feats = self.backbone(x)                 # (B, ch, patch_num, d_model)
        feat = feats.mean(dim=(1, 2))            # (B, d_model) penultimate
        return feat, self.head(feat)


class CBraModAdapter(ModelAdapter):
    name = 'cbramod'

    def preprocess(self, X_raw):
        x = np.asarray(X_raw, dtype=np.float32)
        scale = self.cfg.get('scale', 100.0)
        x = x / scale
        n_dst = int(round(x.shape[2] * DST_FS / SRC_FS))   # 1000 -> 800
        x = scipy_resample(x, n_dst, axis=2).astype(np.float32)
        patch_num = self.cfg.get('patch_num', n_dst // PATCH)
        assert patch_num * PATCH == x.shape[2], (
            f'resampled length {x.shape[2]} not divisible into {PATCH}-pt patches')
        N, C, _ = x.shape
        x = x.reshape(N, C, patch_num, PATCH)
        return torch.as_tensor(x, dtype=torch.float32)

    def build(self, num_classes):
        from .cbramod import CBraMod
        backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                           dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        pretrain = self.cfg.get('pretrain') or paths.weight_path('cbramod')
        backbone.load_state_dict(torch.load(pretrain, map_location='cpu'))
        model = _CBraModClassifier(backbone, num_classes, d_model=200)
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)
