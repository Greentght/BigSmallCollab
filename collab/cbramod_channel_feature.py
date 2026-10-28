"""CBraMod channel-wise feature targets for one shared student projection."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


PATCHES_PER_CHANNEL = 4
FEATURE_WIDTH = 200


def mean_pool_temporal_patches(features, channels: int) -> np.ndarray:
    """Convert flattened CBraMod backbone outputs to ``[N, C, 200]`` targets.

    CBraMod emits ``[N, C, 4, 200]`` before the task head. Historical caches
    flattened that tensor as ``[N, C*4*200]``. Only the four temporal patch
    vectors are averaged; channel identity remains intact.
    """
    array = np.asarray(features)
    channels = int(channels)
    if channels < 1 or array.ndim != 2:
        raise ValueError("features must be a 2-D array and channels must be positive")
    expected = channels * PATCHES_PER_CHANNEL * FEATURE_WIDTH
    if array.shape[1] != expected:
        raise ValueError(
            f"expected flattened CBraMod width {expected} for {channels} channels, "
            f"got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("CBraMod features contain NaN or Inf")
    return array.reshape(len(array), channels, PATCHES_PER_CHANNEL,
                         FEATURE_WIDTH).mean(axis=2, dtype=np.float32)


def channel_cosine_loss(student_projected: torch.Tensor,
                        teacher_channels: torch.Tensor) -> torch.Tensor:
    """Compare one shared projected student vector against every EEG channel.

    ``student_projected`` is ``[B, 200]`` and ``teacher_channels`` is
    ``[B, C, 200]``. The same student vector is broadcast to each channel; the
    cosine distance is averaged over both batch and channel dimensions.
    """
    if student_projected.ndim != 2:
        raise ValueError("projected student feature must have shape [B, 200]")
    if teacher_channels.ndim != 3:
        raise ValueError("teacher channel features must have shape [B, C, 200]")
    if (student_projected.shape[0] != teacher_channels.shape[0]
            or student_projected.shape[1] != teacher_channels.shape[2]):
        raise ValueError(
            f"student/teacher feature shape mismatch: {tuple(student_projected.shape)} "
            f"vs {tuple(teacher_channels.shape)}")
    expanded = student_projected.unsqueeze(1).expand_as(teacher_channels)
    return (1.0 - F.cosine_similarity(expanded, teacher_channels,
                                      dim=-1, eps=1e-8)).mean()
