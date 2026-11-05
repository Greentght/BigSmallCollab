"""Small, framework-independent helpers for ensemble-to-student distillation."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F


def mean_probabilities(*probabilities: np.ndarray) -> np.ndarray:
    """Average T=1 probabilities in float64, then return float32."""
    if len(probabilities) < 2:
        raise ValueError("at least two probability arrays are required")
    arrays = [np.asarray(value, dtype=np.float64) for value in probabilities]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("probability arrays must share an [N, C] shape")
    for value in arrays:
        if (not np.isfinite(value).all() or np.any(value < 0)
                or np.any(value > 1)
                or not np.allclose(value.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
            raise ValueError("invalid probability array")
    result = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)
    return result.astype(np.float32)


def categorical_kl(student_logits: torch.Tensor,
                   teacher_probabilities: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student), sum over classes and mean over actual batch."""
    if student_logits.ndim != 2 or teacher_probabilities.shape != student_logits.shape:
        raise ValueError("logits and teacher probabilities must have matching [B, C] shapes")
    if not torch.isfinite(student_logits).all():
        raise FloatingPointError("student logits contain NaN/Inf")
    if (not torch.isfinite(teacher_probabilities).all()
            or torch.any(teacher_probabilities < 0)):
        raise FloatingPointError("teacher probabilities must be finite and non-negative")
    log_student = F.log_softmax(student_logits, dim=1)
    return F.kl_div(log_student, teacher_probabilities, reduction="batchmean")


def student_objective(student_logits: torch.Tensor, labels: torch.Tensor, *,
                      teacher_probabilities: torch.Tensor | None,
                      ce_weight: float, lambda_kd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared CE/KD dispatch; lambda zero never reads or validates a teacher target."""
    ce = F.cross_entropy(student_logits, labels)
    kd = student_logits.new_zeros(())
    if float(lambda_kd) != 0.0:
        if teacher_probabilities is None:
            raise ValueError("a non-zero KD weight requires teacher probabilities")
        kd = categorical_kl(student_logits, teacher_probabilities)
    return float(ce_weight) * ce + float(lambda_kd) * kd, ce, kd


def sha256_uids(uids: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(uids, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def classification_metrics(y: np.ndarray, probabilities: np.ndarray,
                           *, prediction: np.ndarray | None = None) -> dict:
    """Return subject-level metrics; optional hard predictions support E4."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix)

    labels = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or len(probs) != len(labels):
        raise ValueError("probabilities must have shape [N, C] and align with labels")
    if prediction is None:
        pred = np.argmax(probs, axis=1).astype(np.int64)
    else:
        pred = np.asarray(prediction, dtype=np.int64)
        if pred.shape != labels.shape:
            raise ValueError("hard predictions do not align with labels")
    matrix = confusion_matrix(labels, pred, labels=list(range(probs.shape[1])))
    row = {
        "n_trials": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "kappa": float(cohen_kappa_score(labels, pred)),
        "predicted_class_count": int(np.unique(pred).size),
        "collapsed": bool(np.unique(pred).size < probs.shape[1]),
    }
    for class_id in range(probs.shape[1]):
        row[f"predicted_class_{class_id}_count"] = int(np.sum(pred == class_id))
        denom = int(np.sum(labels == class_id))
        row[f"recall_class_{class_id}"] = (
            float(matrix[class_id, class_id] / denom) if denom else float("nan"))
    if prediction is None:
        clipped = np.maximum(probs, 1e-12)
        row["cross_entropy"] = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())
        entropy = -(clipped * np.log(clipped)).sum(axis=1)
        row["predictive_entropy"] = float(entropy.mean())
    else:
        row["cross_entropy"] = float("nan")
        row["predictive_entropy"] = float("nan")
    return row


def paired_bootstrap(differences: Iterable[float], *, draws: int = 10000,
                     seed: int = 20261105,
                     index_table: np.ndarray | None = None) -> tuple[float, float, float, np.ndarray]:
    """Subject bootstrap CI for a paired vector, returning its index table."""
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be a non-empty finite vector")
    if index_table is None:
        rng = np.random.default_rng(int(seed))
        index_table = rng.integers(0, len(values), size=(int(draws), len(values)))
    else:
        index_table = np.asarray(index_table, dtype=np.int64)
        if index_table.ndim != 2 or index_table.shape[1] != len(values):
            raise ValueError("bootstrap index table has the wrong subject dimension")
    samples = values[index_table].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), index_table
