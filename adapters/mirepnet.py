"""Adapter for MIRepNet (the pretrained MI foundation model / current teacher).

Owns MIRepNet's signature preprocessing: per-set Euclidean Alignment whitening
followed by inverse-distance channel padding to the 45-channel template. This
mirrors ``process_and_replace_loader`` but applies the two transforms directly to
an array (EA is transductive / per-set, matching ``load_subject_data``). The
backbone ``mlm_mask`` returns ``(pooled_feature, logits)`` natively.

``cfg`` must carry ``dataset_name`` (selects the source channel list) and may
carry ``emb_size`` / ``depth`` / ``pretrain`` (path to MIRepNet.pth).
"""
import os

import torch

from core import paths
from .base import ModelAdapter


class MIRepNetAdapter(ModelAdapter):
    name = 'mirepnet'

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        paths.add_repo('mirepnet')
        from utils.utils import EA, pad_missing_channels_diff
        from utils.channel_list import (
            use_channels_names, BNCI2014001_chn_names,
            BNCI2014004_chn_names)
        self._EA = EA
        self._pad = pad_missing_channels_diff
        self._template = use_channels_names
        self._src_channels = {
            'BNCI2014001': BNCI2014001_chn_names,
            'BNCI2014001-4': BNCI2014001_chn_names,
            'BNCI2014004': BNCI2014004_chn_names,
        }

    def preprocess(self, X_raw):
        ds = self.cfg['dataset_name']
        x = torch.as_tensor(X_raw, dtype=torch.float32).numpy()
        x = self._EA(x).astype('float32')                      # per-set whitening
        x = self._pad(x, self._template, self._src_channels[ds])  # -> 45 ch
        return torch.as_tensor(x, dtype=torch.float32)

    def build(self, num_classes):
        from model.mlm import mlm_mask
        pretrain = self.cfg.get('pretrain')
        if pretrain is None:
            pretrain = os.path.join(paths.repo_path('mirepnet'),
                                    'weight', 'MIRepNet.pth')
        model = mlm_mask(
            emb_size=self.cfg.get('emb_size', 256),
            depth=self.cfg.get('depth', 6),
            n_classes=num_classes, pretrainmode=False, pretrain=pretrain)
        return model.to(self.device)

    def forward(self, model, x):
        pooled, logits = model(x)
        return pooled, logits
