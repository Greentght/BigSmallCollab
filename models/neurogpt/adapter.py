"""NeuroGPT adapter for canonical four-second EEG epochs at 250 Hz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
from models.base import ModelAdapter
from .neurogpt import NeuroGPTClassifier, load_pretrained_encoder


class NeuroGPTAdapter(ModelAdapter):
    name = "neurogpt"

    def __init__(self, device="cpu", **cfg):
        super().__init__(device=device, **cfg)
        self.pretrained_report = None
        self.input_channels = int(self.cfg.get("input_channels", 0))

    def preprocess(self, X_raw):
        """Per-trial, per-channel z-score; no statistics cross trial boundaries."""
        x = np.asarray(X_raw, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"NeuroGPT expects (N,C,T), got {x.shape}")
        if x.shape[-1] != 1000:
            raise ValueError(
                f"NeuroGPT expects the canonical 1000-point, 4 s window; got {x.shape[-1]}")
        mean = x.mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        std = x.std(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        x = (x - mean) / np.maximum(std, np.float32(1e-6))
        return torch.as_tensor(x, dtype=torch.float32)

    def build(self, num_classes):
        if self.input_channels < 1:
            raise ValueError("NeuroGPTAdapter requires input_channels from the source manifest")
        channel_matrix = None
        if self.input_channels == 22:
            matrix_path = Path(self.cfg.get(
                "channel_matrix",
                "/data1/llx/pre_weight/neurogpt/tMatrix_value.npy",
            ))
            if not matrix_path.is_file():
                raise FileNotFoundError(
                    f"official NeuroGPT 22-channel montage matrix is missing: {matrix_path}")
            channel_matrix = np.load(matrix_path, allow_pickle=False)
            if channel_matrix.shape != (22, 22) or not np.isfinite(channel_matrix).all():
                raise ValueError(f"invalid NeuroGPT montage matrix at {matrix_path}")
            if not self.cfg.get("channel_order_verified", False):
                raise ValueError(
                    "22-channel NeuroGPT input requires a verified source channel order")
        model = NeuroGPTClassifier(
            num_classes=num_classes,
            input_channels=self.input_channels,
            fixed_channel_matrix=channel_matrix,
            head_dropout=float(self.cfg.get("head_dropout", 0.5)),
        )
        weight = self.cfg.get("pretrain") or config.weight_path(self.name)
        self.pretrained_report = load_pretrained_encoder(model, weight)
        self.pretrained_report.update({
            "checkpoint": str(weight),
            "channel_mapping": ("official_fixed_22x22_matrix" if channel_matrix is not None
                                else "trainable_C_native_to_22_conv_maxnorm1"),
            "input_channels": self.input_channels,
        })
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)


ADAPTER = NeuroGPTAdapter
