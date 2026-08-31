"""Unified full-stack seeding for reproducible training.

Single canonical implementation — random/numpy/torch/cuda + cudnn determinism.
Kept behavior-identical to the historical ``collab.distill._set_seed`` so
same-seed reruns stay bit-identical.

Note: ``torch.cuda.manual_seed_all`` touches every visible device — in multi-GPU
worker processes, set ``CUDA_VISIBLE_DEVICES`` (or ``--gpu``) *before* calling this
so other cards are not disturbed (OOM lesson, PROGRESS 07-28).
"""
import random

import numpy as np
import torch


def set_seed(seed):
    """Full-stack seeding (init + dropout + dataloader shuffle + sampler).
    cudnn set deterministic to curb GPU nondeterminism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
