"""Controllers and per-trial logits-KD losses for the EEG utility pilot."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SampleUtilityMLP(nn.Module):
    """Map detached Teacher/Student signals and progress to a bounded value."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64),
                 output_min: float = 0.05, output_max: float = 0.95):
        super().__init__()
        if input_dim <= 0 or len(hidden_dims) != 2:
            raise ValueError('expected a positive input_dim and two hidden dimensions')
        if not 0.0 < output_min < output_max < 1.0:
            raise ValueError('controller bounds must satisfy 0 < min < max < 1')
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.SiLU(),
            nn.Linear(hidden_dims[1], 1),
        )
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.requires_grad:
            raise ValueError('sample utility inputs must be detached from the Student')
        score = self.net(features).squeeze(-1)
        return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(score)


def build_controller_input(teacher_features: torch.Tensor,
                           reference_student_features: torch.Tensor,
                           teacher_probabilities: torch.Tensor,
                           current_student_probabilities: torch.Tensor,
                           progress: float | torch.Tensor) -> torch.Tensor:
    """Build the common state vector; the returned tensor is always detached."""
    arrays = (teacher_features, reference_student_features,
              teacher_probabilities, current_student_probabilities)
    if any(t.ndim != 2 for t in arrays):
        raise ValueError('features and probabilities must have shape (batch, dimension)')
    count = arrays[0].shape[0]
    if any(t.shape[0] != count for t in arrays):
        raise ValueError('controller input arrays have different batch lengths')
    device, dtype = arrays[0].device, arrays[0].dtype
    normalized = [F.normalize(t.detach(), p=2, dim=1, eps=1e-8) for t in arrays[:2]]
    detached = [normalized[0], normalized[1],
                arrays[2].detach().to(device=device, dtype=dtype),
                arrays[3].detach().to(device=device, dtype=dtype)]
    p = torch.as_tensor(progress, device=device, dtype=dtype)
    if p.ndim == 0:
        p = p.expand(count)
    if p.shape != (count,):
        raise ValueError(f'progress must be scalar or shape ({count},), got {tuple(p.shape)}')
    result = torch.cat([*detached, p[:, None]], dim=1)
    if not torch.isfinite(result).all():
        raise ValueError('non-finite controller input')
    return result.detach()


def per_trial_logits_kd(student_logits: torch.Tensor,
                        teacher_logits: torch.Tensor,
                        temperature: float = 2.0) -> torch.Tensor:
    """Return tau² KL(teacher || student) for every row, without batch reduction."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError('teacher and student logits must have the same (B,C) shape')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1) * temperature ** 2


def student_loss(student_logits: torch.Tensor, labels: torch.Tensor,
                 teacher_logits: torch.Tensor, weights: torch.Tensor | None,
                 *, lam_kd: float = 0.5, temperature: float = 2.0) -> tuple[torch.Tensor, dict]:
    """CE for all rows plus weighted KD divided by the actual batch length."""
    ce_rows = F.cross_entropy(student_logits, labels, reduction='none')
    ce = ce_rows.mean()
    kd_rows = per_trial_logits_kd(student_logits, teacher_logits, temperature)
    if weights is None:
        weights = torch.ones_like(kd_rows)
    weights = weights.to(device=kd_rows.device, dtype=kd_rows.dtype)
    if weights.shape != kd_rows.shape:
        raise ValueError('one KD weight is required for each trial')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('KD weights must be finite and nonnegative')
    kd = (weights * kd_rows).sum() / len(labels)
    total = ce + float(lam_kd) * kd
    return total, {'ce': ce, 'kd': kd, 'ce_rows': ce_rows, 'kd_rows': kd_rows}


def sample_bernoulli_actions(probabilities: torch.Tensor,
                             generator: torch.Generator) -> torch.Tensor:
    """Sample binary KD gates from a dedicated controller RNG stream."""
    return torch.bernoulli(probabilities.detach(), generator=generator)


def bernoulli_policy_loss(probabilities: torch.Tensor, actions: torch.Tensor,
                          reward: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Sum score-function log probabilities with a detached scalar advantage."""
    if probabilities.shape != actions.shape:
        raise ValueError('one sampled action is required for each probability')
    advantage = (reward.detach() - baseline.detach())
    log_prob = (actions * torch.log(probabilities)
                + (1.0 - actions) * torch.log1p(-probabilities))
    return -advantage * log_prob.sum()
