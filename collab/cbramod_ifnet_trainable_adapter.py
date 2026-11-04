"""Trainable full-feature CBraMod-to-IFNet adapter for the V2 LOSO pilot."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


TEACHER_DIMENSION = 2400
STUDENT_DIMENSION = 512


class TeacherToStudentAdapter(nn.Module):
    """Map CBraMod's flattened three-channel feature to IFNet's 512-D space."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(TEACHER_DIMENSION, STUDENT_DIMENSION, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(STUDENT_DIMENSION, STUDENT_DIMENSION, bias=True)

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        if teacher_features.ndim != 2 or teacher_features.shape[1] != TEACHER_DIMENSION:
            raise ValueError("teacher features must have shape [batch,2400]")
        output = self.fc2(self.activation(self.fc1(teacher_features)))
        return F.layer_norm(output, (STUDENT_DIMENSION,), weight=None, bias=None,
                            eps=1e-5)


def derive_adapter_seed(experiment_id: str, seed: int, fold: int) -> int:
    payload = f"{experiment_id}/adapter-init/{int(seed)}/{int(fold)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def initialize_adapter(seed: int) -> tuple[TeacherToStudentAdapter, str]:
    """Use PyTorch default Linear initialization without consuming global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        module = TeacherToStudentAdapter()
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return module, digest.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
