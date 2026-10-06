"""Benchmark-style CBraMod adapter.

Pipeline per set:
truncate raw 4s @250 Hz -> resample to 200 Hz -> optional band-pass/notch ->
optional EA -> ``/ scale`` -> reshape ``(N, C_native, 4, 200)``. The backbone is
pretrained CBraMod with ``proj_out`` removed; the downstream classifier is the
benchmark loader's flatten readout plus ``Dropout + Linear`` task head.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample as scipy_resample
from torch.utils.data import DataLoader, TensorDataset

try:
    from mne.filter import resample as mne_resample
except ModuleNotFoundError:  # keep the adapter importable outside the full EEG env
    mne_resample = None

import config
from models.base import ModelAdapter


class _CBraModModel(nn.Module):
    """Pretrained backbone + flatten readout + Dropout/Linear task head."""

    def __init__(self, num_classes, n_ch, n_patch=4, dropout=0.1,
                 pretrain=None, feature_head='flatten'):
        super().__init__()
        from .cbramod import CBraMod
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30,
                                n_layer=12, nhead=8)
        if pretrain:
            self.backbone.load_state_dict(torch.load(pretrain, map_location='cpu'))
        self.backbone.proj_out = nn.Identity()
        self.feature_head = str(feature_head)
        if self.feature_head == 'flatten':
            self.flatten = nn.Flatten(start_dim=1)
            classifier_dim = n_ch * n_patch * 200
        elif self.feature_head == 'mean_pool_200':
            # Compatibility branch for the earlier diagnostic only.  This is
            # not the official task head; the corrected pilot uses
            # ``original_mlp_200`` below.
            self.flatten = None
            classifier_dim = 200
        elif self.feature_head == 'original_mlp_200':
            # Preserve the original CBraMod task head: the feature exposed to
            # the distillation runner is the 200-D penultimate activation
            # immediately before the original final Linear(200, C).
            self.flatten = nn.Flatten(start_dim=1)
            classifier_dim = n_ch * n_patch * 200
            self.feature_mlp = nn.Sequential(
                nn.Linear(classifier_dim, 800),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(800, 200),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(200, num_classes)
            return
        else:
            raise ValueError(
                f'Unsupported CBraMod feature_head={self.feature_head!r}; '
                "expected 'flatten', 'mean_pool_200', or 'original_mlp_200'.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        if self.feature_head == 'original_mlp_200':
            feat = self.feature_mlp(self.flatten(h))
            return feat, self.classifier(feat)
        if self.feature_head == 'flatten':
            feat = self.flatten(h)
        else:
            if h.ndim != 4 or h.shape[-1] != 200:
                raise ValueError(f'CBraMod pooled head expected [B,C,P,200], got {tuple(h.shape)}')
            feat = h.mean(dim=(1, 2))
        return feat, self.classifier(self.dropout(feat))


class CBraModAdapter(ModelAdapter):
    name = 'cbramod'

    REQUIRED_MODEL_CONFIG = (
        'lr', 'epochs', 'weight_decay', 'batch_size',
        'dropout', 'scale', 'label_smoothing',
        'target_fs', 'norm_method', 'apply_EA', 'warmup_epochs', 'min_lr',
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
        from data.preproc import DST_FS, SRC_FS, EA, bandpass, notch
        self._EA = EA
        self._bandpass = bandpass
        self._notch = notch
        self._src_fs = SRC_FS
        self._dst_fs = DST_FS

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ('0', 'false', 'no', 'off')
        return bool(value)

    def _target_fs(self):
        target_fs = int(self.cfg.get('target_fs', self._dst_fs))
        if target_fs != self._dst_fs:
            raise ValueError(
                'CBraModAdapter currently supports target_fs=200 only because '
                'the pretrained patch projection and classifier use 200-sample '
                f'patches; got target_fs={target_fs}.')
        return target_fs

    def _resample_to_target(self, x, target_fs):
        if mne_resample is not None:
            return mne_resample(
                np.asarray(x, dtype=np.float64),
                down=(self._src_fs / target_fs),
                axis=-1)
        out_samples = int(round(x.shape[-1] * target_fs / self._src_fs))
        return scipy_resample(x, out_samples, axis=-1)

    def _apply_frequency_filters(self, x, fs):
        l_freq = self.cfg.get('l_freq')
        h_freq = self.cfg.get('h_freq')
        if l_freq is not None or h_freq is not None:
            low = 0.0 if l_freq is None else float(l_freq)
            high = fs / 2.0 if h_freq is None else float(h_freq)
            x = self._bandpass(x, fs, low, high)

        notch_freq = self.cfg.get('notch_freq')
        if notch_freq is not None:
            x = self._notch(x, fs, float(notch_freq))
        return np.asarray(x, dtype=np.float32)

    def _normalize(self, x):
        method = self.cfg.get('norm_method')
        if method is None or str(method).lower() == 'none':
            return np.asarray(x, dtype=np.float32)
        method = str(method).lower()
        if method == 'car':
            return (x - np.mean(x, axis=1, keepdims=True)).astype(np.float32)
        raise ValueError(f'Unsupported CBraMod norm_method: {self.cfg.get("norm_method")}')

    def _preprocess_3d(self, X_raw, apply_ea=None):
        """Native-channel 4s pipeline: (N, C, 1000)@250Hz -> (N, C, 4, 200)."""
        target_fs = self._target_fs()
        x = np.asarray(X_raw, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f'CBraModAdapter expects 3D raw input, got {x.shape}')
        x = x[:, :, :1000]
        x = self._resample_to_target(x, target_fs)
        x = self._apply_frequency_filters(x, target_fs)
        if apply_ea is None:
            apply_ea = self._as_bool(self.cfg.get('apply_EA', False))
        if apply_ea:
            x = self._EA(x).astype(np.float32)
        x = self._normalize(x)
        x = (x / float(self.cfg['scale'])).astype(np.float32)
        n, ch, _ = x.shape
        n_patch = x.shape[-1] // 200
        if x.shape[-1] % 200 != 0 or n_patch != 4:
            raise ValueError(
                'CBraModAdapter expects 4 one-second 200 Hz patches after '
                f'preprocessing; got shape {x.shape}.')
        return x.reshape(n, ch, n_patch, 200)

    def preprocess(self, X_raw):
        """(N, C_native, 1000)@250Hz -> (N, C_native, 4, 200)."""
        # Some LOSO/distillation paths pass tensors already preprocessed through
        # ea_pad_per_subject; in that case the adapter should only tensorize them.
        if self.cfg.get('skip_preprocess'):
            return torch.as_tensor(np.asarray(X_raw), dtype=torch.float32)
        return torch.as_tensor(self._preprocess_3d(X_raw), dtype=torch.float32)

    def ea_pad_per_subject(self, X_raw, subj_ids):
        """Compatibility hook for LOSO scripts: native CBraMod preprocessing per subject."""
        X = np.asarray(X_raw, dtype=np.float32)
        if X.ndim == 4:
            return X
        subj_ids = np.asarray(subj_ids)
        out = None
        apply_ea = self._as_bool(self.cfg.get('apply_EA', False))
        for s in np.unique(subj_ids):
            m = subj_ids == s
            xs = self._preprocess_3d(X[m], apply_ea=apply_ea)
            if out is None:
                out = np.empty((len(X),) + xs.shape[1:], dtype=np.float32)
            out[m] = xs
        return out

    def build(self, num_classes):
        pretrain = self.cfg.get('pretrain') or config.weight_path('cbramod')
        n_ch = int(self.cfg.get('in_channels', 3))
        model = _CBraModModel(num_classes, dropout=float(self.cfg['dropout']),
                              n_ch=n_ch, n_patch=4, pretrain=pretrain,
                              feature_head=self.cfg.get('feature_head', 'flatten'))
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)

    @staticmethod
    def _cosine_schedule(base_value, final_value, epochs, niter_per_ep,
                         warmup_epochs=0, start_warmup_value=0.0):
        """Benchmark cosine schedule: warmup + cosine values indexed by step."""
        warmup_schedule = np.array([])
        warmup_iters = int(warmup_epochs) * int(niter_per_ep)
        if warmup_epochs > 0 and warmup_iters > 0:
            warmup_schedule = np.linspace(
                start_warmup_value, base_value, warmup_iters)

        cosine_iters = np.arange(int(epochs) * int(niter_per_ep) - warmup_iters)
        if len(cosine_iters) > 0:
            cosine_schedule = np.array([
                final_value + 0.5 * (base_value - final_value)
                * (1 + np.cos(np.pi * i / len(cosine_iters)))
                for i in cosine_iters
            ])
        else:
            cosine_schedule = np.array([])
        schedule = np.concatenate((warmup_schedule, cosine_schedule))
        expected = int(epochs) * int(niter_per_ep)
        if len(schedule) != expected:
            raise RuntimeError(
                f'CBraMod LR schedule length {len(schedule)} != {expected}')
        return schedule

    @staticmethod
    def _apply_lr_schedule(opt, lr_schedule_values, global_step):
        step_idx = min(global_step, len(lr_schedule_values) - 1)
        for group in opt.param_groups:
            group['lr'] = (
                lr_schedule_values[step_idx] * group.get('lr_scale', 1.0))

    def finetune(self, model, X_tr, y_tr, num_classes):
        """Benchmark-style native-channel recipe: AdamW + step-indexed LR schedule."""
        epochs = int(self.cfg['epochs'])
        lr = float(self.cfg['lr'])
        wd = float(self.cfg['weight_decay'])
        bs = int(self.cfg['batch_size'])
        ls = float(self.cfg['label_smoothing'])
        warmup_epochs = max(0, min(int(self.cfg['warmup_epochs']), epochs))
        min_lr = float(self.cfg['min_lr'])
        if min_lr < 0.0:
            raise ValueError(f'min_lr must be non-negative, got {min_lr}')
        if min_lr > lr:
            raise ValueError(f'min_lr must be <= lr, got min_lr={min_lr}, lr={lr}')

        Xp = self.preprocess(X_tr)
        y = torch.as_tensor(y_tr, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xp, y), batch_size=bs, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)
        lr_schedule_values = self._cosine_schedule(
            lr, min_lr, epochs, len(loader), warmup_epochs=warmup_epochs)
        cr = nn.CrossEntropyLoss(label_smoothing=ls)
        global_step = 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                cr(model(xb)[1], yb).backward()
                opt.step()
                global_step += 1
            self._apply_lr_schedule(opt, lr_schedule_values, global_step)
        return model


ADAPTER = CBraModAdapter
