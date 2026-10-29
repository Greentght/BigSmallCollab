"""Low-dimensional, training-only Student feature priors for LOSO pilots."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_pca_whitener(features: np.ndarray, max_dimensions: int = 32,
                     rank_relative_threshold: float = 1e-5,
                     eigenvalue_floor_relative: float = 1e-4) -> dict:
    """Fit the specified float64 PCA whitening transform on D_train only."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('PCA features must be a finite [N,D] matrix')
    if len(values) < 2:
        raise ValueError('PCA requires at least two training trials')
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('PCA effective rank is zero')
    rank = int(np.sum(eigenvalues > maximum * float(rank_relative_threshold)))
    if rank == 0:
        raise ValueError('PCA effective rank is zero under the configured threshold')
    dimension = min(int(max_dimensions), values.shape[1], rank)
    kept_values = eigenvalues[:dimension]
    components = eigenvectors[:, :dimension]
    scales = np.sqrt(kept_values + float(eigenvalue_floor_relative) * maximum)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('PCA whitening scales are invalid')
    return {
        'mean': mean.astype(np.float64),
        'components': components.astype(np.float64),
        'eigenvalues': kept_values.astype(np.float64),
        'scales': scales.astype(np.float64),
        'effective_rank': rank,
        'dimension': dimension,
        'total_variance': float(eigenvalues.sum()),
        'retained_variance': float(kept_values.sum()),
        'retained_variance_fraction': float(kept_values.sum() / eigenvalues.sum()),
        'maximum_eigenvalue': maximum,
    }


def apply_pca_whitener(features: np.ndarray, transform: dict) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    transformed = ((values - transform['mean']) @ transform['components']
                   / transform['scales'])
    if not np.isfinite(transformed).all():
        raise FloatingPointError('whitened features contain NaN or Inf')
    return transformed.astype(np.float32)


def fit_ridge_projection(teacher_features: np.ndarray,
                         student_features: np.ndarray,
                         rho: float = 0.01,
                         standard_deviation_floor: float = 1e-6) -> dict:
    """Fit fixed teacher standardization and float64 ridge map with intercept."""
    teacher = np.asarray(teacher_features, dtype=np.float64)
    student = np.asarray(student_features, dtype=np.float64)
    if (teacher.ndim != 2 or student.ndim != 2 or len(teacher) != len(student)
            or len(teacher) < 2 or not np.isfinite(teacher).all()
            or not np.isfinite(student).all()):
        raise ValueError('ridge inputs must be finite, paired [N,D] matrices')
    mean = teacher.mean(axis=0)
    std = np.maximum(teacher.std(axis=0, ddof=0), float(standard_deviation_floor))
    standardized = (teacher - mean) / std
    x_mean = standardized.mean(axis=0)
    y_mean = student.mean(axis=0)
    x = standardized - x_mean
    y = student - y_mean
    gram = (x.T @ x) / len(x)
    cross = (x.T @ y) / len(x)
    coefficient = np.linalg.solve(gram + float(rho) * np.eye(gram.shape[0]), cross)
    bias = y_mean - x_mean @ coefficient
    weight = coefficient.T
    prediction = standardized @ weight.T + bias
    residual = student - prediction
    mse = float(np.mean(residual ** 2))
    target_variance = float(np.mean((student - student.mean(axis=0)) ** 2))
    r2 = float(1.0 - mse / target_variance) if target_variance > 0 else float('nan')
    singular_values = np.linalg.svd(weight, compute_uv=False)
    threshold = singular_values.max(initial=0.0) * 1e-10
    effective_rank = int(np.sum(singular_values > threshold)) if threshold > 0 else 0
    return {
        'teacher_mean': mean.astype(np.float64),
        'teacher_std': std.astype(np.float64),
        'weight': weight.astype(np.float64),
        'bias': bias.astype(np.float64),
        'rho': float(rho),
        'train_mse': mse,
        'train_r2': r2,
        'effective_rank': effective_rank,
        'singular_values': singular_values.astype(np.float64),
    }


def apply_ridge_projection(teacher_features: np.ndarray, projection: dict) -> np.ndarray:
    values = np.asarray(teacher_features, dtype=np.float64)
    standardized = (values - projection['teacher_mean']) / projection['teacher_std']
    result = standardized @ projection['weight'].T + projection['bias']
    if not np.isfinite(result).all():
        raise FloatingPointError('projected teacher features contain NaN or Inf')
    return result.astype(np.float32)


class ConditionalNoisePredictor(nn.Module):
    """The V1 class-conditioned epsilon MLP."""
    def __init__(self, feature_dimension: int, classes: int = 2,
                 hidden_dimensions: int = 128, time_dimensions: int = 32,
                 class_dimensions: int = 8):
        super().__init__()
        if time_dimensions % 2:
            raise ValueError('sinusoidal time embedding dimension must be even')
        self.time_dimensions = int(time_dimensions)
        self.class_embedding = nn.Embedding(int(classes), int(class_dimensions))
        self.network = nn.Sequential(
            nn.Linear(int(feature_dimension) + self.time_dimensions + int(class_dimensions),
                      int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), int(hidden_dimensions)),
            nn.SiLU(),
            nn.Linear(int(hidden_dimensions), int(feature_dimension)),
        )

    def time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.time_dimensions // 2
        scales = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device,
                                              dtype=torch.float32) / max(half - 1, 1))
        angles = timesteps.to(torch.float32).reshape(-1, 1) * scales.reshape(1, -1)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    def forward(self, noisy_features: torch.Tensor, timesteps: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        t = self.time_embedding(timesteps)
        y = self.class_embedding(labels.to(torch.long))
        return self.network(torch.cat((noisy_features, t, y), dim=1))


def linear_ddpm_schedule(steps: int, beta_start: float,
                         beta_end: float, device: torch.device) -> tuple[torch.Tensor,
                                                                         torch.Tensor,
                                                                         torch.Tensor]:
    betas = torch.linspace(float(beta_start), float(beta_end), int(steps),
                           dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)


def gaussian_epsilon(noisy: torch.Tensor, timesteps: torch.Tensor,
                     class_labels: torch.Tensor, means: torch.Tensor,
                     variances: torch.Tensor, alpha: torch.Tensor,
                     sigma: torch.Tensor) -> torch.Tensor:
    """Analytic epsilon predictor for class-conditional diagonal Gaussians."""
    index = timesteps.to(torch.long) - 1
    a = alpha[index].reshape(-1, 1)
    s = sigma[index].reshape(-1, 1)
    mu = means[class_labels.to(torch.long)]
    var = variances[class_labels.to(torch.long)]
    return s * (noisy - a * mu) / (a.square() * var + s.square())


def sds_direction(predicted_epsilon: torch.Tensor, sampled_epsilon: torch.Tensor,
                  timesteps: torch.Tensor, alpha: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
    index = timesteps.to(torch.long) - 1
    a = alpha[index].reshape(-1, 1)
    s = sigma[index].reshape(-1, 1)
    return (a / s) * (predicted_epsilon - sampled_epsilon)


def sds_surrogate(projected_features: torch.Tensor,
                  detached_direction: torch.Tensor) -> torch.Tensor:
    """Scalar whose feature gradient is g/(B*d), exactly as specified."""
    if projected_features.shape != detached_direction.shape:
        raise ValueError('SDS feature and direction shapes differ')
    return (projected_features * detached_direction.detach()).mean()


def gaussian_prior_fit(features: np.ndarray, labels: np.ndarray,
                       classes: int = 2, variance_floor: float = 0.01) -> dict:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError('Gaussian prior features and labels do not align')
    means, variances, counts = [], [], []
    for class_id in range(int(classes)):
        part = values[targets == class_id]
        if not len(part):
            raise ValueError(f'Gaussian prior has no D_train samples for class {class_id}')
        means.append(part.mean(axis=0))
        variances.append(np.maximum(part.var(axis=0, ddof=0), float(variance_floor)))
        counts.append(len(part))
    return {'means': np.asarray(means, dtype=np.float64),
            'variances': np.asarray(variances, dtype=np.float64),
            'counts': np.asarray(counts, dtype=np.int64),
            'variance_floor': float(variance_floor)}
