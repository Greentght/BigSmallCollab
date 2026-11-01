"""Fixed teacher-to-student feature coordinates and verified native feature taps.

An expansion has orthonormal columns, a reduction has orthonormal rows.  Thus
MIRepNet's 256-D native classifier input can be mapped to a larger student
space without falsely claiming that a [d_student,256] matrix has orthonormal
rows.  LayerNorm is confined to the auxiliary supervision branch.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_ID = 'MIREPNET_T2S_ALLSETS_LOSO_V1'


def derive_seed(stage: str, seed: int, fold: int, dataset: str = '',
                student: str = '', experiment_id: str = EXPERIMENT_ID) -> int:
    identity = [str(experiment_id), str(stage), int(seed), int(fold),
                str(dataset), str(student)]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**32)


def fixed_teacher_projection(teacher_dimension: int, student_dimension: int,
                             seed: int) -> np.ndarray:
    """Construct a deterministic float64 partial isometry P[d_student,d_teacher]."""
    dt, ds = int(teacher_dimension), int(student_dimension)
    if dt < 1 or ds < 1:
        raise ValueError('teacher and student feature dimensions must be positive')
    rng = np.random.default_rng(int(seed))
    tall = rng.standard_normal((max(dt, ds), min(dt, ds)), dtype=np.float64)
    q, r = np.linalg.qr(tall, mode='reduced')
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    projection = np.ascontiguousarray(q.T if ds <= dt else q, dtype=np.float64)
    gram = projection @ projection.T if ds <= dt else projection.T @ projection
    if not np.allclose(gram, np.eye(min(dt, ds)), rtol=1e-10, atol=1e-10):
        raise RuntimeError('fixed projection failed its orientation-specific orthogonality check')
    return projection


def projection_properties(projection: np.ndarray) -> dict:
    p = np.asarray(projection, dtype=np.float64)
    if p.ndim != 2 or min(p.shape) < 1 or not np.isfinite(p).all():
        raise ValueError('projection must be a finite nonempty [d_student,d_teacher] matrix')
    ds, dt = p.shape
    gram = p @ p.T if ds <= dt else p.T @ p
    error = float(np.max(np.abs(gram - np.eye(min(dt, ds)))))
    if error > 1e-8:
        raise ValueError(f'projection is not a partial isometry; max Gram error={error}')
    return {'shape': [ds, dt], 'teacher_dimension': dt, 'student_dimension': ds,
            'orthogonality': 'rows' if ds <= dt else 'columns',
            'gram_max_abs_error': error,
            'retains_all_teacher_coordinates_before_layer_norm': ds >= dt,
            'trainable': False}


def projection_tensor(projection: np.ndarray, device='cpu',
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    projection_properties(projection)
    return torch.as_tensor(np.asarray(projection), dtype=dtype,
                           device=device).detach().requires_grad_(False)


def layer_norm_no_affine(features: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError('native trial features must have shape [batch,dimension]')
    if not np.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError('LayerNorm epsilon must be finite and positive')
    return F.layer_norm(features, (features.shape[1],), weight=None, bias=None,
                        eps=float(epsilon))


def project_teacher_tensor(features: torch.Tensor, projection: torch.Tensor,
                           epsilon: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or projection.ndim != 2 or features.shape[1] != projection.shape[1]:
        raise ValueError('teacher features and projection have incompatible [B,d_teacher]/[d_student,d_teacher] shapes')
    if projection.requires_grad or isinstance(projection, nn.Parameter):
        raise ValueError('teacher projection must be fixed, without trainable parameters')
    projected = F.linear(features, projection)
    return projected, layer_norm_no_affine(projected, epsilon)


def project_teacher_features(features: np.ndarray, projection: np.ndarray,
                             device='cpu', batch_size: int = 128,
                             epsilon: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    p = projection_tensor(projection, device)
    if values.ndim != 2 or values.shape[1] != p.shape[1] or len(values) < 1:
        raise ValueError('teacher features must be nonempty [N,d_teacher]')
    if not np.isfinite(values).all() or int(batch_size) < 1:
        raise ValueError('teacher features must be finite and batch_size positive')
    before, after = [], []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            x = torch.as_tensor(values[start:start + int(batch_size)], device=device)
            projected, normalized = project_teacher_tensor(x, p, epsilon)
            before.append(projected.cpu().numpy())
            after.append(normalized.cpu().numpy())
    return np.concatenate(before), np.concatenate(after)


def _native_head(model: nn.Module, model_name: str) -> tuple[nn.Module, str]:
    name = str(model_name).lower()
    if name == 'mirepnet':
        head, path = model.clshead, 'clshead'
    elif name == 'ifnet':
        head, path = model.fc, 'fc'
    elif name == 'eegnet':
        head, path = model.fc1, 'fc1'
    elif name == 'adfcnn':
        head, path = model.classifier[0], 'classifier.0'
    else:
        raise ValueError(f'unsupported native classifier feature tap: {model_name}')
    if not isinstance(head, (nn.Linear, nn.Conv2d)):
        raise TypeError(f'{model_name}.{path} is not a supported native linear classifier')
    return head, path


def forward_native(adapter, model: nn.Module, x: torch.Tensor,
                   return_metadata: bool = False):
    """One native forward; verify the adapter feature equals the actual head input.

    ADFCNN's native classifier consumes a [B,C,1,T] map.  Reshaping that map
    preserves all values and is equivalent to its full-map classification
    convolution.  No new pooling is performed for any architecture.
    """
    name = str(adapter.name).lower()
    head, path = _native_head(model, name)
    captured = []

    def capture(module, inputs):
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError('native head input is not one tensor')
        captured.append(inputs[0])

    handle = head.register_forward_pre_hook(capture)
    try:
        returned_features, logits = adapter.forward(model, x)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f'{name} classifier was invoked {len(captured)} times in one forward')
    native = captured[0]
    flat = native.reshape(native.shape[0], -1)
    if (returned_features.ndim != 2 or returned_features.shape != flat.shape
            or not torch.allclose(returned_features.detach(), flat.detach(), rtol=1e-6, atol=1e-7)):
        raise RuntimeError(f'{name} adapter feature differs from the actual native classifier input')
    if logits.ndim != 2 or logits.shape[0] != flat.shape[0]:
        raise RuntimeError(f'{name} logits must have shape [batch,classes]')
    meta = {'model': name, 'module_path': path,
            'native_head_input_shape': list(native.shape[1:]),
            'native_trial_dimension': int(flat.shape[1]),
            'pooling_added': False, 'student_forwards_per_batch': 1,
            'returned_output': 'log_probabilities' if name == 'adfcnn' else 'logits'}
    return (flat, logits, meta) if return_metadata else (flat, logits)


native_forward = forward_native


def classifier_readout(model: nn.Module, model_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the native linear readout, before ADFCNN's LogSoftmax if applicable."""
    head, path = _native_head(model, model_name)
    if isinstance(head, nn.Conv2d):
        if (head.groups != 1 or head.stride != (1, 1) or head.padding != (0, 0)
                or head.dilation != (1, 1)):
            raise RuntimeError(f'{path} is not a plain full-map classification convolution')
        weight = head.weight.reshape(head.out_channels, -1)
    else:
        weight = head.weight
    bias = head.bias if head.bias is not None else weight.new_zeros(weight.shape[0])
    return weight, bias


def projection_readout_diagnostic(features: np.ndarray, logits: np.ndarray,
                                 weight: np.ndarray, bias: np.ndarray,
                                 projection: np.ndarray,
                                 labels: np.ndarray | None = None) -> dict:
    """Measure native teacher decision information retained before auxiliary LN."""
    f, z, w, b, p = (np.asarray(value, dtype=np.float64)
                      for value in (features, logits, weight, bias, projection))
    properties = projection_properties(p)
    if (f.ndim != 2 or f.shape[1] != p.shape[1] or w.ndim != 2
            or w.shape[1] != f.shape[1] or b.shape != (w.shape[0],)
            or z.shape != (len(f), w.shape[0])):
        raise ValueError('teacher features, readout, logits and projection shapes differ')
    # Both orientations are partial isometries, so P^T is their pseudoinverse.
    teacher_subspace_projector = p.T @ p
    residual_weight = w @ (np.eye(p.shape[1]) - teacher_subspace_projector)
    recovered_logits = (f @ teacher_subspace_projector) @ w.T + b
    result = {'before_auxiliary_layer_norm': True,
        'projection_orthogonality': properties['orthogonality'],
        'linear_readout_relative_weight_residual': float(
            np.linalg.norm(residual_weight) / max(np.linalg.norm(w), 1e-15)),
        'logit_mae': float(np.mean(np.abs(z - recovered_logits))),
        'logit_rmse': float(np.sqrt(np.mean((z - recovered_logits) ** 2))),
        'prediction_agreement': float(np.mean(z.argmax(axis=1) == recovered_logits.argmax(axis=1))),
        'native_linear_readout_max_abs_error': float(np.max(np.abs(z - (f @ w.T + b)))),
        'projected_logits': recovered_logits.astype(np.float32)}
    if labels is not None:
        from sklearn.metrics import balanced_accuracy_score
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (len(f),):
            raise ValueError('labels do not match feature trials')
        result.update(original_balanced_accuracy=float(balanced_accuracy_score(y, z.argmax(axis=1))),
                      projected_balanced_accuracy=float(balanced_accuracy_score(y, recovered_logits.argmax(axis=1))))
    return result
