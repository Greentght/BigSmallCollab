"""Official CodeBrain EEGSSM backbone with a project-sized MI classifier.

The required encoder files from the authors' Apache-2.0 repository are preserved
under ``upstream/CodeBrain-main`` and hash-checked at import time. One upstream
device-specific mask allocation is adapted at runtime so the official residual
block works on the selected device.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset

from models.base import ModelAdapter


OFFICIAL_REPO = 'https://github.com/jingyingma01/CodeBrain'
OFFICIAL_REVISION = '22d350caf68246d2fda4f630ef837420db3fb130'
OFFICIAL_WEIGHT_URL = (
    'https://huggingface.co/YjMajy/CodeBrain/resolve/main/CodeBrain.pth'
)
OFFICIAL_WEIGHT_REVISION = 'bef08d2fdb1759685371cc635aad21ce59163689'
OFFICIAL_WEIGHT_SHA256 = (
    'd9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352'
)
DEFAULT_REPO = Path(__file__).resolve().parent / 'upstream' / 'CodeBrain-main'
DEFAULT_WEIGHT = Path(__file__).resolve().parents[2] / 'weights' / 'codebrain.pth'
OFFICIAL_SOURCE_SHA256 = {
    'Models/SSSM.py': 'e2fe5f7364907129507f3a9e946df9f9e44e10c3511efdcf772fab46d6fa278f',
    'Models/SGConv.py': '94f12f897ab4e8784a45d64fb65664184c872e6928159a6759fda2920c488be8',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
}


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def codebrain_repo_path() -> Path:
    path = Path(os.environ.get('CODEBRAIN_REPO', str(DEFAULT_REPO))).expanduser().resolve()
    if not all((path / relpath).is_file() for relpath in OFFICIAL_SOURCE_SHA256):
        raise FileNotFoundError(
            f'Official CodeBrain source not found at {path}; set CODEBRAIN_REPO '
            f'to a checkout of {OFFICIAL_REPO} ({OFFICIAL_REVISION}).'
        )
    for relpath, expected in OFFICIAL_SOURCE_SHA256.items():
        actual = sha256_file(path / relpath)
        if actual != expected:
            raise ValueError(
                f'Official CodeBrain source hash mismatch for {path / relpath}: '
                f'expected {expected}, got {actual}'
            )
    return path


def import_official_sssm():
    """Import the author's SSSM and make its local attention mask device-safe."""
    repo = codebrain_repo_path()
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        from Models.SSSM import Residual_block, SSSM
    except Exception as exc:  # noqa: BLE001 - provide the actionable source/deps
        raise ImportError(
            'Could not import official CodeBrain Models.SSSM. Install the '
            'authors\' requirements in the active model environment and ensure '
            'CODEBRAIN_REPO points at the official checkout.'
        ) from exc

    # Upstream Residual_block.forward creates its local attention mask with
    # ``.cuda()``. Preserve the exact computation, but remove that device-bound
    # call so CPU and non-default CUDA devices also work.
    def _generate_local_window_mask(self, seq_len, window_size):
        if int(window_size) % 2 != 1:
            raise ValueError('CodeBrain attention window_size must be odd')
        parameter = next(self.parameters())
        mask = torch.full(
            (int(seq_len), int(seq_len)), float('-inf'),
            device=parameter.device,
        )
        half_window = int(window_size) // 2
        for index in range(int(seq_len)):
            start = max(0, index - half_window)
            end = min(int(seq_len), index + half_window + 1)
            mask[index, start:end] = 0
        return mask

    Residual_block.generate_local_window_mask = _generate_local_window_mask

    def _residual_block_forward(self, input_data):
        # Mirrors Models/SSSM.py's official Residual_block.forward, except the
        # attention mask is passed through on its current device.
        from einops import rearrange

        x, original = input_data
        h = x
        batch, channels, length = x.shape
        x = self.sn(x)
        if channels != self.res_channels:
            raise AssertionError(
                f'expected {self.res_channels} residual channels, got {channels}'
            )
        h = h + original.view(batch, self.res_channels, length)
        h = self.gelu(self.conv_layer(h))
        h_t, _ = self.S41(h)
        h_s = rearrange(h_t, 'b c l -> b l c')
        mask = self.generate_local_window_mask(length, 1)
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=mask)
        h_s = rearrange(h_s, 'b l c -> b c l')
        h = h_t + h_s
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(
            h[:, self.res_channels:, :]
        )
        residual = self.res_conv(out)
        if x.shape != residual.shape:
            raise AssertionError(
                f'residual shape {tuple(residual.shape)} != input {tuple(x.shape)}'
            )
        skip = self.skip_conv(out)
        return (x + residual) * np.sqrt(0.5), skip

    Residual_block.forward = _residual_block_forward
    return SSSM


def build_backbone(dropout: float = 0.3) -> nn.Module:
    """Build the same 8-layer encoder dimensions used by official SHU-MI."""
    SSSM = import_official_sssm()
    return SSSM(
        in_channels=200,
        res_channels=200,
        skip_channels=200,
        out_channels=200,
        num_res_layers=8,
        diffusion_step_embed_dim_in=200,
        diffusion_step_embed_dim_mid=200,
        diffusion_step_embed_dim_out=200,
        s4_lmax=570,
        s4_d_state=64,
        s4_dropout=float(dropout),
        s4_bidirectional=True,
        s4_layernorm=True,
        codebook_size_t=4096,
        codebook_size_f=4096,
        if_codebook=False,
    )


def _read_state_dict(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    # The official file is a plain OrderedDict. weights_only=True avoids pickle
    # object deserialization and is compatible with the public state dict.
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state:
        raise TypeError(f'Expected a non-empty state dict in {path}')
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise TypeError('CodeBrain checkpoint must be a tensor-only state dict')
    return state


def load_codebrain_backbone(
    backbone: nn.Module,
    checkpoint: str | os.PathLike | None = None,
    *,
    verify_sha256: bool = True,
) -> dict:
    """Load public encoder weights strictly and return an auditable report."""
    path = Path(checkpoint or os.environ.get('CODEBRAIN_WEIGHT', DEFAULT_WEIGHT))
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'CodeBrain weights not found at {path}; set CODEBRAIN_WEIGHT.'
        )
    digest = sha256_file(path)
    if verify_sha256 and digest != OFFICIAL_WEIGHT_SHA256:
        raise ValueError(
            f'CodeBrain checkpoint hash mismatch: expected '
            f'{OFFICIAL_WEIGHT_SHA256}, got {digest} ({path})'
        )

    state = _read_state_dict(path)
    prefix_removed = None
    if all(key.startswith('module.') for key in state):
        state = {key[len('module.'):]: value for key, value in state.items()}
        prefix_removed = 'module.'
    elif any(key.startswith('module.') for key in state):
        raise ValueError('Mixed module.-prefixed and unprefixed checkpoint keys')

    expected_state = backbone.state_dict()
    missing_keys = sorted(set(expected_state) - set(state))
    unexpected_keys = sorted(set(state) - set(expected_state))
    shape_mismatches = [
        {
            'key': key,
            'checkpoint_shape': list(state[key].shape),
            'expected_shape': list(expected_state[key].shape),
        }
        for key in sorted(set(expected_state).intersection(state))
        if tuple(state[key].shape) != tuple(expected_state[key].shape)
    ]
    report = {
        'source_url': OFFICIAL_WEIGHT_URL,
        'source_weight_revision': OFFICIAL_WEIGHT_REVISION,
        'source_repo': OFFICIAL_REPO,
        'source_revision': OFFICIAL_REVISION,
        'source_code_sha256': OFFICIAL_SOURCE_SHA256,
        'checkpoint_path': str(path),
        'checkpoint_bytes': path.stat().st_size,
        'checkpoint_sha256': digest,
        'strict': True,
        'prefix_removed': prefix_removed,
        'loaded_tensor_count': len(state),
        'expected_tensor_count': len(expected_state),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'shape_mismatches': shape_mismatches,
        'encoder_loaded_without_task_head': True,
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            'Strict CodeBrain state-dict compatibility failed before loading: '
            + str(report)
        )
    incompat = backbone.load_state_dict(state, strict=True)
    report['missing_keys'] = list(incompat.missing_keys)
    report['unexpected_keys'] = list(incompat.unexpected_keys)
    if report['missing_keys'] or report['unexpected_keys']:
        raise RuntimeError(f'Strict CodeBrain load returned incompatible keys: {report}')
    return report


class CodeBrainClassifier(nn.Module):
    """CodeBrain encoder plus the paper-style three-linear-layer MI head."""

    def __init__(self, n_channels: int, num_classes: int, dropout: float = 0.3,
                 backbone: nn.Module | None = None):
        super().__init__()
        self.n_channels = int(n_channels)
        self.num_classes = int(num_classes)
        self.backbone = backbone if backbone is not None else build_backbone(dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.n_channels * 4 * 200, 800),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(800, 200),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(200, self.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[2:]) != (4, 200):
            raise ValueError(
                f'CodeBrain expects [B,C,4,200] input, got {tuple(x.shape)}'
            )
        if int(x.shape[1]) != self.n_channels:
            raise ValueError(
                f'Expected {self.n_channels} native channels, got {int(x.shape[1])}'
            )
        features = self.backbone(x)
        # Upstream SSSM returns ``x.squeeze()``; restore a dropped singleton
        # batch dimension before applying the project head.
        if features.ndim == 3 and int(x.shape[0]) == 1:
            features = features.unsqueeze(0)
        expected = (int(x.shape[0]), self.n_channels, 4, 200)
        if tuple(features.shape) != expected:
            raise RuntimeError(
                f'CodeBrain encoder returned {tuple(features.shape)}, expected {expected}'
            )
        hidden = self.feature_mlp(self.flatten(features))
        logits = self.classifier(hidden)
        return hidden, logits


class CodeBrainInputAdapter:
    """Deterministic native-channel 250 Hz to 200 Hz patch conversion."""

    def __init__(self, scale_divisor: float = 100.0):
        if float(scale_divisor) <= 0:
            raise ValueError('scale_divisor must be positive')
        self.scale_divisor = float(scale_divisor)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(X_raw, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f'Expected raw [N,C,T] epochs, got {x.shape}')
        if x.shape[-1] != 1000:
            raise ValueError(
                f'Expected exactly 4 seconds at 250 Hz (1000 samples), got {x.shape[-1]}'
            )
        x = resample_poly(x, up=4, down=5, axis=-1)
        if x.shape[-1] != 800:
            raise RuntimeError(f'250->200 Hz resampling produced {x.shape[-1]} samples')
        x = (x / self.scale_divisor).astype(np.float32, copy=False)
        x = x.reshape(x.shape[0], x.shape[1], 4, 200)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite value after CodeBrain input conversion')
        return x

    def config(self) -> dict:
        return {
            'source_rate_hz': 250,
            'target_rate_hz': 200,
            'resampler': 'scipy.signal.resample_poly(up=4, down=5, axis=-1)',
            'window_seconds': 4,
            'patch_seconds': 1,
            'samples_per_patch': 200,
            'patch_count': 4,
            'channel_policy': 'preserve native names, count, and order; no pad/drop',
            'filtering': 'none',
            'euclidean_alignment': False,
            'source_unit': 'microvolt (µV)',
            'source_unit_evidence': (
                'Repository PROGRESS.md:1135 records X.npy as µV-scale (001 std '
                'about 4.7 µV); current run records each train split distribution. '
                'Official SHU downstream preprocessing also divides by 100.'
            ),
            'scale_divisor': self.scale_divisor,
            'fit_statistics': 'none; fixed physical-unit conversion only',
        }


class CodeBrainAdapter(ModelAdapter):
    """Project registry adapter; the dedicated runner adds audit artifacts."""

    name = 'codebrain'

    def __init__(self, device='cpu', **cfg):
        super().__init__(device=device, **cfg)
        self.input_adapter = CodeBrainInputAdapter(
            scale_divisor=float(self.cfg.get('scale_divisor', 100.0))
        )
        self.load_report = None

    def preprocess(self, X_raw):
        return torch.from_numpy(self.input_adapter.transform(X_raw))

    def build(self, num_classes):
        import config

        backbone = build_backbone(dropout=float(self.cfg.get('dropout', 0.3)))
        checkpoint = self.cfg.get('pretrain') or config.weight_path('codebrain')
        self.load_report = load_codebrain_backbone(backbone, checkpoint)
        model = CodeBrainClassifier(
            n_channels=int(self.cfg.get('in_channels')),
            num_classes=int(num_classes),
            dropout=float(self.cfg.get('dropout', 0.3)),
            backbone=backbone,
        )
        return model.to(self.device)

    def forward(self, model, x):
        return model(x)

    def finetune(self, model, X_tr, y_tr, num_classes):
        epochs = int(self.cfg.get('epochs', 20))
        batch_size = int(self.cfg.get('batch_size', 64))
        loader = DataLoader(
            TensorDataset(
                self.preprocess(X_tr),
                torch.as_tensor(y_tr, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(self.cfg.get('lr', 5e-5)),
            weight_decay=float(self.cfg.get('weight_decay', 5e-3)),
            eps=1e-8,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs * len(loader)),
            eta_min=float(self.cfg.get('min_lr', 1e-6)),
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        max_grad_norm = float(self.cfg.get('max_grad_norm', 5.0))
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                _hidden, logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
        return model


ADAPTER = CodeBrainAdapter
