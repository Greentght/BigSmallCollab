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


def align_replay_by_uid(replay_uids, expected_uids, **replay_arrays):
    """Reorder replay arrays into the requested UID order, rejecting ambiguity."""
    replay_uids = np.asarray(replay_uids, dtype=np.int64)
    expected_uids = np.asarray(expected_uids, dtype=np.int64)
    if replay_uids.ndim != 2 or expected_uids.ndim != 2:
        raise ValueError('replay and expected UIDs must be two-dimensional')
    replay_keys = [tuple(uid) for uid in replay_uids.tolist()]
    expected_keys = [tuple(uid) for uid in expected_uids.tolist()]
    if (len(set(replay_keys)) != len(replay_keys)
            or len(set(expected_keys)) != len(expected_keys)):
        raise ValueError('replay and expected UIDs must be unique')
    if len(replay_keys) != len(expected_keys) or set(replay_keys) != set(expected_keys):
        raise ValueError('replay UID set differs from expected UID set')
    replay_position = {uid: index for index, uid in enumerate(replay_keys)}
    order = np.asarray([replay_position[uid] for uid in expected_keys], dtype=np.int64)
    aligned = {}
    for name, values in replay_arrays.items():
        values = np.asarray(values)
        if values.ndim == 0 or len(values) != len(replay_keys):
            raise ValueError(f'replay array {name!r} does not match the UID count')
        aligned[name] = values[order]
    return aligned
