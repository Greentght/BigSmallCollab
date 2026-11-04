"""Student-side shared-space adapter for the CBraMod -> IFNet V3 pilot."""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F


DIMENSION = 512


class StudentToSharedAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIMENSION, DIMENSION, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(DIMENSION, DIMENSION, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != DIMENSION:
            raise ValueError("student features must have shape [batch,512]")
        values = self.fc2(self.activation(self.fc1(features)))
        return F.layer_norm(values, (DIMENSION,), weight=None, bias=None, eps=1e-5)


def derive_student_adapter_seed(experiment_id: str, seed: int, fold: int) -> int:
    payload = f"{experiment_id}/student-adapter-init/{int(seed)}/{int(fold)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def initialize_student_adapter(seed: int) -> tuple[StudentToSharedAdapter, str]:
    """Use default Linear initialization without consuming the training RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        module = StudentToSharedAdapter()
    return module, state_hash(module.state_dict())


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
