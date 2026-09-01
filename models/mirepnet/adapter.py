"""Adapter for MIRepNet (the pretrained MI foundation model / current teacher).

Owns MIRepNet's signature preprocessing: per-set Euclidean Alignment whitening
followed by inverse-distance channel padding to the 45-channel template. This
mirrors ``process_and_replace_loader`` but applies the two transforms directly to
an array (EA is transductive / per-set, matching ``load_subject_data``). The
backbone ``mlm_mask`` returns ``(pooled_feature, logits)`` natively.

``cfg`` must carry ``dataset_name`` (selects the source channel list) and may
carry ``emb_size`` / ``depth`` / ``pretrain`` (path to MIRepNet.pth).
"""

import numpy as np
import torch

import config
from models.base import ModelAdapter


class MIRepNetAdapter(ModelAdapter):
    name = 'mirepnet'

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        # Data-layer preprocessing is framework-owned; only the pretrained backbone
        # (build()) is still loaded from the MIRepNet repo (see phase-2 vendoring).
        from data.preproc import EA, pad_missing_channels_diff
        from data.channels import (
            use_channels_names, BNCI2014001_chn_names,
            BNCI2014004_chn_names, BNCI2015001_chn_names, AlexMI_chn_names)
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
        # LOSO passes data already EA'd per-subject + padded to 45ch; skip to
        # avoid re-whitening the mixed multi-subject set with one covariance.
        if self.cfg.get('skip_preprocess'):
            return torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        ds = self.cfg['dataset_name']
        x = torch.as_tensor(X_raw, dtype=torch.float32).numpy()
        x = self._EA(x).astype('float32')                      # per-set whitening
        x = self._pad(x, self._template, self._src_channels[ds])  # -> 45 ch
        return torch.as_tensor(x, dtype=torch.float32)

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
        from .mlm import mlm_mask
        pretrain = self.cfg.get('pretrain') or config.weight_path('mirepnet')
        model = mlm_mask(
            emb_size=self.cfg.get('emb_size', 256),
            depth=self.cfg.get('depth', 6),
            n_classes=num_classes, pretrainmode=False, pretrain=pretrain)
        return model.to(self.device)

    def forward(self, model, x):
        pooled, logits = model(x)
        return pooled, logits


ADAPTER = MIRepNetAdapter
