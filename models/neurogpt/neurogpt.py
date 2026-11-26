"""NeuroGPT EEG-Conformer encoder and author-style downstream head.

The module follows the encoder-only downstream path in NeuroGPT. It keeps the
pretrained EEG-Conformer parameter names under ``encoder`` so the published
checkpoint can be validated and loaded without importing the benchmark repo.
The channel adapter and downstream classifier are task-specific.

Reference: Cui et al., Neuro-GPT (ISBI 2024),
https://github.com/wenhui0206/NeuroGPT . The attention/convolution blocks follow
the EEG-Conformer architecture by Song et al. (BSD-3-Clause implementation).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _PatchEmbedding(nn.Module):
    def __init__(self, n_filters_time=40, filter_time_length=25,
                 n_channels=22, pool_time_length=75,
                 stride_avg_pool=15, drop_prob=0.5):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(n_filters_time),
            nn.ELU(),
            nn.AvgPool2d((1, pool_time_length), stride=(1, stride_avg_pool)),
            nn.Dropout(p=drop_prob),
        )
        # The author's final 1x1 convolution keeps this state-dict index.
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection[0](x)
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError(f"expected collapsed channel axis, got {tuple(x.shape)}")
        return x.squeeze(2).transpose(1, 2).contiguous()


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=40, num_heads=10, dropout=0.5):
        super().__init__()
        if emb_size % num_heads:
            raise ValueError("embedding size must be divisible by number of heads")
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, tokens, _ = x.shape
        head_dim = self.emb_size // self.num_heads
        q = self.queries(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        k = self.keys(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        v = self.values(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        energy = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        att = self.att_drop(torch.softmax(energy / (self.emb_size ** 0.5), dim=-1))
        out = torch.matmul(att, v).transpose(1, 2).reshape(batch, tokens, self.emb_size)
        return self.projection(out)


class _ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size=40, expansion=4, drop_p=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size), nn.GELU(),
            nn.Dropout(drop_p), nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=40, att_heads=10, att_drop=0.5,
                 forward_expansion=4):
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, att_heads, att_drop),
                nn.Dropout(att_drop),
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(emb_size, forward_expansion, att_drop),
                nn.Dropout(att_drop),
            )),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, att_depth=6, emb_size=40, att_heads=10, att_drop=0.5):
        super().__init__(*[
            _TransformerEncoderBlock(emb_size, att_heads, att_drop)
            for _ in range(att_depth)
        ])


class NeuroGPTEncoder(nn.Module):
    """The pretrained EEG-Conformer trunk (27 x 40 tokens per 2 s chunk)."""
    def __init__(self, n_chans=22, n_times=500, n_filters_time=40,
                 filter_time_length=25, pool_time_length=75,
                 pool_time_stride=15, drop_prob=0.5, att_depth=6,
                 att_heads=10, att_drop_prob=0.5):
        super().__init__()
        if n_chans != 22 or n_times != 500:
            raise ValueError("published NeuroGPT encoder expects 22 channels x 500 samples")
        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = _TransformerEncoder(
            att_depth=att_depth, emb_size=n_filters_time,
            att_heads=att_heads, att_drop=att_drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (22, 500):
            raise ValueError(f"encoder expects (B,22,500), got {tuple(x.shape)}")
        return self.transformer(self.patch_embedding(x.unsqueeze(1)))


class _ChannelProjection(nn.Conv1d):
    """Benchmark-style C-native -> 22 channel map with per-output max norm."""
    def __init__(self, n_channels: int, max_norm=1.0):
        super().__init__(n_channels, 22, kernel_size=1)
        self.max_norm = float(max_norm)

    def forward(self, x: Tensor) -> Tensor:
        # Functional clipping avoids the benchmark's .data mutation during
        # diagnostic/evaluation forwards while preserving its max-norm rule.
        weight = torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.conv1d(x, weight, self.bias)


class NeuroGPTClassifier(nn.Module):
    """Two-chunk encoder-only NeuroGPT classifier.

    ``features`` are the 2160-D concatenated encoder tokens. The author's
    downstream head maps 2160 -> 256 -> 32 -> classes; the returned logits are
    pre-softmax for the project's standard CE/KD contract.
    """
    def __init__(self, num_classes: int, input_channels: int,
                 fixed_channel_matrix: np.ndarray | None = None,
                 head_dropout: float = 0.5):
        super().__init__()
        self.input_channels = int(input_channels)
        self.encoder = NeuroGPTEncoder()
        if fixed_channel_matrix is not None:
            matrix = torch.as_tensor(fixed_channel_matrix, dtype=torch.float32)
            if matrix.shape != (22, 22):
                raise ValueError(f"fixed channel matrix must be 22x22, got {tuple(matrix.shape)}")
            self.register_buffer("fixed_channel_matrix", matrix)
            self.channel_projection = None
        else:
            self.register_buffer("fixed_channel_matrix", None)
            self.channel_projection = _ChannelProjection(self.input_channels)
        self.fc = nn.Sequential(
            nn.Linear(2160, 256), nn.ELU(), nn.Dropout(head_dropout),
            nn.Linear(256, 32), nn.ELU(),
        )
        self.final_layer = nn.Linear(32, int(num_classes))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[2] != 1000:
            raise ValueError(
                f"classifier expects (B,{self.input_channels},1000), got {tuple(x.shape)}")
        if self.fixed_channel_matrix is not None:
            x = torch.einsum("oc,bct->bot", self.fixed_channel_matrix, x)
        else:
            x = self.channel_projection(x)
        batch = x.shape[0]
        chunks = x.reshape(batch, 22, 2, 500).permute(0, 2, 1, 3)
        tokens = self.encoder(chunks.reshape(batch * 2, 22, 500))
        chunk_features = tokens.flatten(start_dim=1)
        features = chunk_features.reshape(batch, 2, 1080).reshape(batch, 2160)
        hidden = self.fc(features)
        return features, self.final_layer(hidden)


def _checkpoint_state(checkpoint) -> tuple[dict[str, Tensor], list[str]]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            child = checkpoint.get(key)
            if isinstance(child, Mapping):
                checkpoint = child
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("NeuroGPT checkpoint does not contain a state mapping")
    state = {}
    discarded_task_head = []
    for original_key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        key = str(original_key)
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("encoder."):
            key = key[len("encoder."):]
            if key.startswith(("fc.", "final_layer.")):
                discarded_task_head.append(key)
            else:
                state[key] = value
        elif key.startswith(("patch_embedding.", "transformer.")):
            state[key] = value
    if not state:
        raise ValueError("NeuroGPT checkpoint contains no encoder.* parameters")
    return state, discarded_task_head


def load_pretrained_encoder(model: NeuroGPTClassifier, checkpoint_path: str | Path) -> dict:
    """Load all and only the published encoder trunk, failing on partial loads."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"official NeuroGPT checkpoint is missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, discarded_task_head = _checkpoint_state(checkpoint)
    expected = model.encoder.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        key for key in set(expected) & set(state)
        if expected[key].shape != state[key].shape
    )
    if missing or unexpected or shape_mismatch:
        details = {
            "missing": missing[:20], "unexpected": unexpected[:20],
            "shape_mismatch": {
                key: {"checkpoint": list(state[key].shape),
                      "model": list(expected[key].shape)}
                for key in shape_mismatch[:20]
            },
        }
        raise ValueError(f"NeuroGPT encoder checkpoint coverage failed: {details}")
    model.encoder.load_state_dict(state, strict=True)
    return {"loaded_encoder_tensors": len(state), "missing": [],
            "unexpected": [], "discarded_pretrained_task_head": discarded_task_head}
