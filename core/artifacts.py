"""Standardized per-model prediction artifacts — the cross-env hub interface.

Each model, run in its own conda env, exports one ``.npz`` per
``(dataset, subject, seed, split)`` containing aligned arrays:

    logits : (N, C)  float32   pre-softmax outputs
    feats  : (N, D)  float32   penultimate features (for feature-align KD)
    y      : (N,)    int64     ground-truth labels (alignment checksum)

The hub (``collab/*``) consumes these without ever importing the model, which is
what lets MIRepNet / CBraMod / LaBraM live in incompatible environments. ``y`` is
stored so consumers can assert label alignment across models before fusing.
"""
import os

import numpy as np

ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'results', 'artifacts')


def artifact_path(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    return os.path.join(root, dataset, model, f'{subject}_{seed}_{split}.npz')


def save(dataset, model, subject, seed, split, logits, feats, y,
         root=ARTIFACT_ROOT):
    """Write one artifact (creates parent dirs). Arrays are coerced to f32/i64."""
    path = artifact_path(dataset, model, subject, seed, split, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        logits=np.asarray(logits, dtype=np.float32),
        feats=np.asarray(feats, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
    )
    return path


def exists(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    return os.path.exists(artifact_path(dataset, model, subject, seed, split, root))


def load(dataset, model, subject, seed, split, root=ARTIFACT_ROOT):
    """Load one artifact as a dict {logits, feats, y}."""
    path = artifact_path(dataset, model, subject, seed, split, root)
    if not os.path.exists(path):
        raise FileNotFoundError(f'missing artifact: {path}')
    d = np.load(path)
    return {'logits': d['logits'], 'feats': d['feats'], 'y': d['y']}


def load_aligned(dataset, models, subject, seed, split, root=ARTIFACT_ROOT):
    """Load several models' artifacts, asserting their labels align row-for-row.

    Returns ``(per_model: dict[name -> {logits,feats}], y)``. Raises if any
    model's stored ``y`` disagrees — the guard against silent mis-alignment.
    """
    per_model, ref_y = {}, None
    for m in models:
        d = load(dataset, m, subject, seed, split, root)
        if ref_y is None:
            ref_y = d['y']
        elif not np.array_equal(ref_y, d['y']):
            raise ValueError(
                f'label misalignment: {m} disagrees with {models[0]} for '
                f'{dataset} S{subject} seed{seed} {split}')
        per_model[m] = {'logits': d['logits'], 'feats': d['feats']}
    return per_model, ref_y
