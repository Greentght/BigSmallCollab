"""Multiple-Kernel Maximum Mean Discrepancy (MK-MMD), verbatim from the SDDA /
dalib implementation (https://github.com/Dingkun0817/SDDA, utils/loss.py).

Distribution-level feature alignment: unlike per-sample cosine alignment, MK-MMD
matches the *marginal distribution* of two feature sets (z_s vs z_t) over a batch
without requiring sample-to-sample correspondence. Used here to pull IFNet's
penultimate feature distribution toward MIRepNet's (teacher), at the encoder/head
junction.

Usage:
    mkmmd = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)], linear=True)
    loss = mkmmd(z_s, z_t)   # z_s, z_t: (batch, F), SAME batch size & dim
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None,
                 track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(
            batch_size, self.index_matrix, self.linear).to(z_s.device)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss
