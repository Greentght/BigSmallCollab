"""Standardized per-model prediction artifacts — the cross-env hub interface.

Each model, run in its own conda env, exports one ``.npz`` per
``(dataset, subject, seed, split)`` containing aligned arrays:

    logits : (N, C)  float32   pre-softmax outputs
    feats  : (N, D)  float32   penultimate features (for feature-align KD)
    y      : (N,)    int64     ground-truth labels (alignment checksum)
    sample_uid : (N, 2) int64  optional (subject_id, trial_idx) sample key
    split_policy : str         optional split policy name

The hub (``collab/*``) consumes these without ever importing the model, which is
what lets MIRepNet / CBraMod / LaBraM live in incompatible environments. ``y`` is
stored so consumers can assert label alignment across models before fusing; new
fusion code should also require ``sample_uid`` for true sample-level alignment.
"""
import os

import numpy as np

ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'results', 'artifacts')


def artifact_path(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    return os.path.join(root, dataset, model, f'{subject}_{seed}_{split}.npz')


def save(dataset, model, subject, seed, split, logits, feats, y,
         root=ARTIFACT_ROOT, sample_uid=None, split_policy=None):
    """Write one artifact (creates parent dirs). Arrays are coerced to f32/i64."""
    path = artifact_path(dataset, model, subject, seed, split, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logits_arr = np.asarray(logits, dtype=np.float32)
    feats_arr = np.asarray(feats, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    if len(logits_arr) != len(y_arr):
        raise ValueError(
            f'logits length {len(logits_arr)} != y length {len(y_arr)}')
    if len(feats_arr) != len(y_arr):
        raise ValueError(
            f'feats length {len(feats_arr)} != y length {len(y_arr)}')
    payload = {'logits': logits_arr, 'feats': feats_arr, 'y': y_arr}
    if sample_uid is not None:
        uid_arr = np.asarray(sample_uid, dtype=np.int64)
        if uid_arr.ndim != 2 or uid_arr.shape[1] != 2:
            raise ValueError(
                f'sample_uid must have shape (N, 2), got {uid_arr.shape}')
        if len(uid_arr) != len(y_arr):
            raise ValueError(
                f'sample_uid length {len(uid_arr)} != y length {len(y_arr)}')
        payload['sample_uid'] = uid_arr
    if split_policy is not None:
        payload['split_policy'] = np.asarray(str(split_policy))
    np.savez(path, **payload)
    return path


def exists(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    return os.path.exists(artifact_path(dataset, model, subject, seed, split, root))


def load(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    """Load one artifact as a dict with logits, feats, y, and optional metadata."""
    path = artifact_path(dataset, model, subject, seed, split, root)
    if not os.path.exists(path):
        raise FileNotFoundError(f'missing artifact: {path}')
    d = np.load(path)
    out = {'logits': d['logits'], 'feats': d['feats'], 'y': d['y']}
    if 'sample_uid' in d.files:
        out['sample_uid'] = d['sample_uid']
    if 'split_policy' in d.files:
        out['split_policy'] = str(d['split_policy'].item())
    return out


def load_aligned(dataset, models, subject, seed, split, root=ARTIFACT_ROOT,
                 require_uid=False, require_split_policy=False):
    """Load several models' artifacts, asserting row-for-row alignment.

    Labels are always compared. When ``require_uid`` is true, every artifact must
    also contain an identical ``sample_uid`` array, which is the strict
    sample-level guard. ``require_split_policy`` similarly requires equal split
    policy metadata.
    """
    per_model, ref_y, ref_uid, ref_policy = {}, None, None, None
    ref_model = models[0]
    for m in models:
        d = load(dataset, m, subject, seed, split, root)
        if ref_y is None:
            ref_y = d['y']
            ref_uid = d.get('sample_uid')
            ref_policy = d.get('split_policy')
            if require_uid and ref_uid is None:
                raise ValueError(
                    f'missing sample_uid: {m} for {dataset} S{subject} '
                    f'seed{seed} {split}; regenerate artifacts')
            if require_split_policy and ref_policy is None:
                raise ValueError(
                    f'missing split_policy: {m} for {dataset} S{subject} '
                    f'seed{seed} {split}; regenerate artifacts')
        else:
            if not np.array_equal(ref_y, d['y']):
                raise ValueError(
                    f'label misalignment: {m} disagrees with {ref_model} for '
                    f'{dataset} S{subject} seed{seed} {split}')
            uid = d.get('sample_uid')
            policy = d.get('split_policy')
            if require_uid:
                if uid is None:
                    raise ValueError(
                        f'missing sample_uid: {m} for {dataset} S{subject} '
                        f'seed{seed} {split}; regenerate artifacts')
                if not np.array_equal(ref_uid, uid):
                    raise ValueError(
                        f'sample_uid misalignment: {m} disagrees with '
                        f'{ref_model} for {dataset} S{subject} seed{seed} '
                        f'{split}')
            if require_split_policy:
                if policy is None:
                    raise ValueError(
                        f'missing split_policy: {m} for {dataset} S{subject} '
                        f'seed{seed} {split}; regenerate artifacts')
                if policy != ref_policy:
                    raise ValueError(
                        f'split_policy mismatch: {m}={policy!r} disagrees '
                        f'with {ref_model}={ref_policy!r} for {dataset} '
                        f'S{subject} seed{seed} {split}')
        entry = {'logits': d['logits'], 'feats': d['feats']}
        if 'sample_uid' in d:
            entry['sample_uid'] = d['sample_uid']
        if 'split_policy' in d:
            entry['split_policy'] = d['split_policy']
        per_model[m] = entry
    return per_model, ref_y
