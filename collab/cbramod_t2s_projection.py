"""Fixed CBraMod full-feature projection utilities for T2S LOSO."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


TEACHER_FEATURE_SHAPE = (3, 4, 200)
TEACHER_DIMENSION = 2400
STUDENT_DIMENSION = 512


def derive_seed(stage: str, seed: int, fold: int,
                experiment_id: str = "CBRAMOD_T2S_FULL_FEATURE_LOSO_V1") -> int:
    payload = f"{experiment_id}/{stage}/{int(seed)}/{int(fold)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def fixed_teacher_projection(seed: int) -> np.ndarray:
    """Return a [512,2400] row-orthogonal projection from float64 thin QR."""
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((TEACHER_DIMENSION, STUDENT_DIMENSION), dtype=np.float64)
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    projection = np.ascontiguousarray((q * signs[None, :]).T, dtype=np.float64)
    if projection.shape != (STUDENT_DIMENSION, TEACHER_DIMENSION):
        raise RuntimeError(f"unexpected teacher projection shape {projection.shape}")
    if not np.allclose(projection @ projection.T, np.eye(STUDENT_DIMENSION),
                       rtol=1e-10, atol=1e-10):
        raise RuntimeError("teacher projection failed row-orthogonality check")
    return projection


def layer_norm_no_affine(values: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != STUDENT_DIMENSION:
        raise ValueError("auxiliary features must have shape [batch,512]")
    return F.layer_norm(values, (STUDENT_DIMENSION,), weight=None, bias=None,
                        eps=float(epsilon))


def project_teacher_features(features: np.ndarray, projection: np.ndarray,
                             device: torch.device,
                             batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Project flattened full teacher features and return pre/post-LN arrays."""
    values = np.asarray(features, dtype=np.float32)
    matrix = np.asarray(projection, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != TEACHER_DIMENSION:
        raise ValueError("teacher features must be [N,2400]")
    if matrix.shape != (STUDENT_DIMENSION, TEACHER_DIMENSION):
        raise ValueError("projection must be [512,2400]")
    projected, normalized = [], []
    p = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            x = torch.as_tensor(values[start:start + batch_size],
                                dtype=torch.float32, device=device)
            y = x @ p.T
            projected.append(y.cpu().numpy())
            normalized.append(layer_norm_no_affine(y).cpu().numpy())
    return np.concatenate(projected), np.concatenate(normalized)


def projection_readout_diagnostic(features: np.ndarray, logits: np.ndarray,
                                 weight: np.ndarray, bias: np.ndarray,
                                 projection: np.ndarray, labels: np.ndarray) -> dict:
    """Measure original linear readout information lost by the fixed projection."""
    f = np.asarray(features, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    p = np.asarray(projection, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != TEACHER_DIMENSION:
        raise ValueError("features must be [N,2400]")
    if z.shape != (len(f), 2) or w.shape != (2, TEACHER_DIMENSION) or b.shape != (2,):
        raise ValueError("teacher classifier readout shapes do not match the protocol")
    if p.shape != (STUDENT_DIMENSION, TEACHER_DIMENSION):
        raise ValueError("projection must be [512,2400]")
    projector = p.T @ p
    residual_weight = w @ (np.eye(TEACHER_DIMENSION) - projector)
    projected_logits = (f @ projector) @ w.T + b
    pred, projected_pred = z.argmax(axis=1), projected_logits.argmax(axis=1)
    from sklearn.metrics import balanced_accuracy_score
    return {
        "linear_readout_relative_weight_residual": float(
            np.linalg.norm(residual_weight) / max(np.linalg.norm(w), 1e-15)),
        "logit_mae": float(np.mean(np.abs(z - projected_logits))),
        "logit_rmse": float(np.sqrt(np.mean((z - projected_logits) ** 2))),
        "prediction_agreement": float(np.mean(pred == projected_pred)),
        "original_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "projected_balanced_accuracy": float(balanced_accuracy_score(y, projected_pred)),
        "projected_logits": projected_logits.astype(np.float32),
    }
