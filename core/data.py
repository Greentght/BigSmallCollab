"""Canonical, model-agnostic data source.

Every model in the framework must see the *same samples in the same order* for a
given ``(dataset, subject, seed, split)`` — otherwise per-sample ensemble / KD
alignment across cached artifacts silently breaks. This module is the single
authority for that contract.

It uses the framework-vendored ``EEGDataset`` (``core.eeg_dataset``, a byte-for-byte
copy of MIRepNet's, which yields raw ``(N, C_native, 1000)`` @ 250 Hz — the EA
whitening + 45-ch padding live in ``core.preproc``, applied *later* by the MIRepNet
adapter, not here) and MIRepNet's exact stratified split helper
``split_indices_with_val_ratio`` (reimplemented below). So the splits produced here
are bit-identical to the ones used by ``train_fusion.load_subject_data`` and
``run_align_combo.py`` — see verify step in the plan.

Output is raw numpy ``X (N, C_native, 1000) float32`` + ``y (N,) int64``; each
adapter applies its own preprocessing on top.
"""
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split

# Native channel count / class count per wired-up dataset.
DATASET_INFO = {
    'BNCI2014004': dict(num_classes=2, channels=3, sample_rate=250),
    'BNCI2014001-4': dict(num_classes=4, channels=22, sample_rate=250),
    'BNCI2014001': dict(num_classes=2, channels=22, sample_rate=250),
}

_imported = False


def _ensure_imports():
    """Bind the framework-vendored ``EEGDataset`` (once).

    Lives in ``core.eeg_dataset`` (a byte-for-byte copy of MIRepNet's ``dataset.py``
    with its channel import repointed to ``core.channels``). Imported lazily so the
    mne dependency is only pulled when data is actually loaded. The stratified split
    helper is reimplemented below.
    """
    global _imported, EEGDataset
    if _imported:
        return
    from core.eeg_dataset import EEGDataset as _EEGDataset
    EEGDataset = _EEGDataset
    _imported = True


def split_indices_with_val_ratio(all_indices, labels, val_split, seed):
    """Stratified calibration/test split — bit-identical to MIRepNet's helper
    (``utils/utils.py``). ``val_split`` is the TEST fraction."""
    n = len(all_indices)
    if n == 0:
        return [], []
    if val_split is None:
        val_split = 0.7
    if val_split <= 0.0:
        return list(all_indices), []
    if val_split >= 1.0:
        return [], list(all_indices)
    return train_test_split(all_indices, test_size=val_split,
                            random_state=seed, stratify=labels)


def load_subject_raw(dataset_name, subject):
    """Raw downstream session for one subject: (X (N,C,1000) f32, y (N,) i64)."""
    _ensure_imports()
    args = SimpleNamespace(dataset_name=dataset_name, sub=[subject],
                           data_mode='session3')
    ds = EEGDataset(args=args)
    X = np.asarray(ds.X, dtype=np.float32)
    y = np.asarray(ds.y, dtype=np.int64)
    return X, y


def loso_split(dataset_name, test_subject, num_subjects=None):
    """Leave-One-Subject-Out fold: all subjects except ``test_subject`` form the
    train set, ``test_subject`` is the test set. Returns
    ``(X_tr, y_tr, subj_tr, X_te, y_te)`` where ``subj_tr`` is the per-trial
    subject id (for subject-balanced sampling / per-subject EA). Raw
    ``(N, C_native, 1000)`` @ 250 Hz; each adapter preprocesses on top (the
    MIRepNet teacher must EA per subject-group — see export_teacher_loso).

    No randomness: the fold is fully determined by ``test_subject``.
    """
    _ensure_imports()
    if num_subjects is None:
        num_subjects = {'BNCI2014004': 9, 'BNCI2014001-4': 9,
                        'BNCI2014001': 9}[dataset_name]
    Xtr, ytr, subj = [], [], []
    for s in range(num_subjects):
        X, y = load_subject_raw(dataset_name, s)
        if s == test_subject:
            Xte, yte = X, y
        else:
            Xtr.append(X); ytr.append(y)
            subj.append(np.full(len(y), s, dtype=np.int64))
    return (np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(subj),
            Xte, yte)


def subject_split(dataset_name, subject, val_split=0.3, seed=666):
    """Deterministic per-subject calibration/test split.

    Returns ``(X_tr, y_tr, X_te, y_te)`` as numpy arrays. ``val_split`` is the
    TEST fraction (MIRepNet convention: 0.3 -> 70% calib / 30% test). The split
    indices are reproducible from ``seed`` and shared by every adapter, which is
    what guarantees cross-model sample alignment.
    """
    _ensure_imports()
    X, y = load_subject_raw(dataset_name, subject)
    idx = list(range(len(y)))
    idx_tr, idx_te = split_indices_with_val_ratio(idx, y, val_split, seed)
    return X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]
