"""Channel-conditioned diffusion utilities for CBraMod -> IFNet distillation."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from collab.mcc_diffkd import sinusoidal_time_embedding


class ChannelEpsilonPredictor(nn.Module):
    """Predict noise for one of CBraMod's three channel-token distributions."""

    def __init__(self, dimensions: int, hidden_dimensions: int = 128,
                 time_dimensions: int = 32, channel_dimensions: int = 8,
                 channels: int = 3):
        super().__init__()
        self.dimensions = int(dimensions)
        self.time_dimensions = int(time_dimensions)
        self.channel_dimensions = int(channel_dimensions)
        self.channels = int(channels)
        if self.dimensions < 1 or self.channels != 3:
            raise ValueError("CBraMod prior requires a positive dimension and exactly 3 channels")
        self.channel_embedding = nn.Embedding(self.channels, self.channel_dimensions)
        self.network = nn.Sequential(
            nn.Linear(self.dimensions + self.time_dimensions + self.channel_dimensions,
                      int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), self.dimensions),
        )

    def forward(self, noisy_features: torch.Tensor, timesteps: torch.Tensor,
                channel_ids: torch.Tensor) -> torch.Tensor:
        if noisy_features.ndim != 2 or noisy_features.shape[1] != self.dimensions:
            raise ValueError("channel prior expects [batch, feature_dimension]")
        if timesteps.shape != noisy_features.shape[:1] or channel_ids.shape != timesteps.shape:
            raise ValueError("timestep/channel id shapes must match feature batch")
        if torch.any((channel_ids < 0) | (channel_ids >= self.channels)):
            raise ValueError("channel IDs must be in [0, 2]")
        time = sinusoidal_time_embedding(timesteps, self.time_dimensions)
        channel = self.channel_embedding(channel_ids.to(torch.long))
        return self.network(torch.cat((noisy_features, time, channel), dim=1))


def student_channel_views(features: torch.Tensor, projection: torch.Tensor,
                          epsilon: float = 1e-5) -> torch.Tensor:
    """Map IFNet's one 512-D vector to three fixed, distinct d-D views."""
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("IFNet native feature tap must be [batch, 512]")
    if projection.ndim != 3 or projection.shape[0] != 3 or projection.shape[2] != 512:
        raise ValueError("CBraMod projection must be [3, d, 512]")
    normalized = F.layer_norm(features, (512,), weight=None, bias=None,
                              eps=float(epsilon))
    return torch.einsum("bq,cdq->bcd", normalized, projection)


def channel_ddim_denoise(features: torch.Tensor, noise: torch.Tensor,
                         predictor: ChannelEpsilonPredictor,
                         alpha: torch.Tensor, sigma: torch.Tensor,
                         timesteps: tuple[int, ...] = (250, 200, 150, 100, 50, 0)) -> torch.Tensor:
    """Five-step deterministic DDIM, retaining gradients to all input views."""
    if timesteps != (250, 200, 150, 100, 50, 0):
        raise ValueError("pilot DDIM timestep sequence is fixed")
    if features.ndim != 3 or features.shape[1] != 3 or noise.shape != features.shape:
        raise ValueError("features/noise must both have shape [batch, 3, d]")
    batch, channels, dimension = features.shape
    z = alpha[250] * features + sigma[250] * noise
    channel_ids = torch.arange(channels, device=features.device).repeat(batch)
    flat = z.reshape(batch * channels, dimension)
    for current, following in zip(timesteps[:-1], timesteps[1:]):
        t = torch.full((batch * channels,), int(current), dtype=torch.long,
                       device=features.device)
        epsilon_hat = predictor(flat, t, channel_ids)
        clean = (flat - sigma[current] * epsilon_hat) / alpha[current]
        flat = alpha[following] * clean + sigma[following] * epsilon_hat
    return flat.reshape(batch, channels, dimension)


def channel_ddpm_schedule(steps: int, beta_start: float, beta_end: float,
                          device: torch.device):
    if steps < 1 or not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("invalid DDPM schedule")
    beta = torch.linspace(float(beta_start), float(beta_end), int(steps),
                          dtype=torch.float32, device=device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    alpha = torch.cat((torch.ones(1, device=device), torch.sqrt(alpha_bar)))
    sigma = torch.cat((torch.zeros(1, device=device), torch.sqrt(1.0 - alpha_bar)))
    return beta, alpha, sigma
