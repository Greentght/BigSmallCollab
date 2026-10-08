"""Small, dependency-light helpers for the fixed seven-source LOSO split."""
from __future__ import annotations

import numpy as np


def fold_masks(subjects, fold: int):
    """Return the fixed target/next-subject feedback/seven-source masks."""
    subjects = np.asarray(subjects, dtype=np.int64)
    if not 0 <= int(fold) < 9:
        raise ValueError(f'fold must be in [0,8], got {fold}')
    feedback = (int(fold) + 1) % 9
    return feedback, {'train': (subjects != fold) & (subjects != feedback),
                      'feedback': subjects == feedback,
                      'test': subjects == fold}


def batchwise_shuffle(values, batch_lengths, rng):
    """Shuffle values only within each original training batch."""
    values = np.asarray(values)
    if sum(batch_lengths) != len(values):
        raise ValueError('batch lengths do not cover the replay values')
    output, start = [], 0
    for length in batch_lengths:
        stop = start + int(length)
        output.append(values[start:stop][rng.permutation(int(length))])
        start = stop
    return np.concatenate(output) if output else values.copy()
