"""Core losses and fixed vector diffusion used by the MCC-DiffKD LOSO pilot."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def minimum_class_confusion(logits: torch.Tensor, temperature: float = 2.0,
                            entropy_epsilon: float = 1e-5,
                            row_epsilon: float = 1e-8):
    """Return the MCC scalar and detached diagnostics for one minibatch."""
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("MCC expects logits with shape [batch, classes>=2]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("MCC temperature must be finite and positive")
    probabilities = torch.softmax(logits / float(temperature), dim=1)
    entropy = -(probabilities * torch.log(probabilities + float(entropy_epsilon))).sum(dim=1)
    confidence_weight = 1.0 + torch.exp(-entropy.detach())
    weights = logits.shape[0] * confidence_weight / confidence_weight.sum().clamp_min(1e-12)
    confusion = probabilities.transpose(0, 1) @ (weights[:, None] * probabilities)
    row_mass = confusion.sum(dim=1)
    normalized = confusion / row_mass.clamp_min(float(row_epsilon))[:, None]
    loss = (normalized.sum() - normalized.diagonal().sum()) / logits.shape[1]
    if not torch.isfinite(loss):
        raise FloatingPointError("MCC loss is not finite")
    return loss, {
        "probabilities": probabilities.detach(),
        "entropy": entropy.detach(),
        "weights": weights.detach(),
        "confusion": confusion.detach(),
        "normalized_confusion": normalized.detach(),
        "row_mass": row_mass.detach(),
        "clamped_rows": (row_mass.detach() < float(row_epsilon)).sum(),
    }


def sinusoidal_time_embedding(timesteps: torch.Tensor, dimensions: int = 32) -> torch.Tensor:
    if dimensions < 2 or dimensions % 2:
        raise ValueError("sinusoidal embedding dimension must be a positive even number")
    half = dimensions // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device,
                                          dtype=torch.float32) / max(half - 1, 1))
    phase = timesteps.to(torch.float32).reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class EpsilonPredictor(nn.Module):
    """Unconditional epsilon MLP over a low-dimensional teacher feature vector."""

    def __init__(self, dimensions: int, hidden_dimensions: int = 128,
                 time_dimensions: int = 32):
        super().__init__()
        self.dimensions = int(dimensions)
        self.time_dimensions = int(time_dimensions)
        self.network = nn.Sequential(
            nn.Linear(self.dimensions + self.time_dimensions, int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), self.dimensions),
        )

    def forward(self, noisy_features: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_features.ndim != 2 or noisy_features.shape[1] != self.dimensions:
            raise ValueError("epsilon predictor input shape does not match its feature dimension")
        time = sinusoidal_time_embedding(timesteps, self.time_dimensions)
        return self.network(torch.cat((noisy_features, time), dim=1))


def ddpm_schedule(steps: int, beta_start: float, beta_end: float,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps < 1 or not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("invalid DDPM schedule")
    betas = torch.linspace(float(beta_start), float(beta_end), int(steps),
                            dtype=torch.float32, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    alpha = torch.cat((torch.ones(1, device=device), torch.sqrt(alpha_bar)))
    sigma = torch.cat((torch.zeros(1, device=device), torch.sqrt(1.0 - alpha_bar)))
    return betas, alpha, sigma


def fixed_student_projection(dimensions: int, seed: int) -> np.ndarray:
    """CPU float64 Gaussian thin-QR projection with canonical R signs."""
    if dimensions < 1 or dimensions > 512:
        raise ValueError("projection dimension must be between 1 and 512")
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((512, int(dimensions)), dtype=np.float64)
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    projection = np.ascontiguousarray(q.T, dtype=np.float64)
    gram = projection @ projection.T
    if not np.allclose(gram, np.eye(dimensions), rtol=1e-10, atol=1e-10):
        raise RuntimeError("fixed student projection failed the orthonormality check")
    centered = projection @ (np.eye(512) - np.ones((512, 512)) / 512.0)
    if np.linalg.matrix_rank(centered, tol=1e-10) != dimensions:
        raise RuntimeError("fixed student projection loses rank after LayerNorm centering")
    return projection


def student_features_to_vector(features: torch.Tensor,
                               projection: torch.Tensor,
                               epsilon: float = 1e-5) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("IFNet native feature tap must have shape [batch, 512]")
    normalized = F.layer_norm(features, (512,), weight=None, bias=None, eps=float(epsilon))
    return F.linear(normalized, projection)


def ddim_denoise(features: torch.Tensor, noise: torch.Tensor,
                 predictor: EpsilonPredictor, alpha: torch.Tensor,
                 sigma: torch.Tensor,
                 timesteps: tuple[int, ...] = (250, 200, 150, 100, 50, 0)) -> torch.Tensor:
    """Five-step deterministic DDIM map with gradient through every step."""
    if tuple(timesteps) != (250, 200, 150, 100, 50, 0):
        raise ValueError("V1 DDIM timestep sequence is fixed")
    if features.shape != noise.shape or features.ndim != 2:
        raise ValueError("DDIM features and initial noise must have matching [B,d] shapes")
    z = alpha[250] * features + sigma[250] * noise
    batch = len(features)
    for current, following in zip(timesteps[:-1], timesteps[1:]):
        step = torch.full((batch,), int(current), dtype=torch.long, device=features.device)
        epsilon_hat = predictor(z, step)
        clean = (z - sigma[current] * epsilon_hat) / alpha[current]
        z = alpha[following] * clean + sigma[following] * epsilon_hat
    return z
