"""Canonical, model-agnostic data source.

Every model in the framework must see the *same samples in the same order* for a
given ``(dataset, subject, seed, split)`` — otherwise per-sample ensemble / KD
alignment across cached artifacts silently breaks. This module is the single
authority for that contract.

It uses the framework-vendored ``EEGDataset`` as the single source of raw
``(N, C_native, 1000)`` @ 250 Hz samples. By default, each dataset keeps its
own canonical source session name: BNCI2014001 and BNCI2014001-4 use
``sessionT``; BNCI2014004 uses ``session3``. The EA whitening + channel
padding live in ``data.preproc`` and are applied later by each adapter.

Output is raw numpy ``X (N, C_native, 1000) float32`` + ``y (N,) int64``; each
adapter applies its own preprocessing on top.
"""
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split

import config

_imported = False


Cell = namedtuple('Cell', 'unit seed X_tr y_tr X_te y_te subj_ids')
_PROTOCOL_ALIASES = {'within': 'fewshot'}
FEWSHOT_SPLIT_POLICY = 'fewshot_stratified_random'
ORDERED_FEWSHOT_SPLIT_POLICY = 'fewshot_ordered_by_label'
LOSO_SPLIT_POLICY = 'loso'


def canonical_protocol(protocol):
    """Return the public protocol name; 'within' is a legacy alias."""
    p = str(protocol).lower()
    return _PROTOCOL_ALIASES.get(p, p)


def _fewshot_val_split(val_split, train_percentage=None):
    if train_percentage is None:
        return val_split
    train_percentage = float(train_percentage)
    if train_percentage <= 0.0 or train_percentage >= 1.0:
        raise ValueError('train_percentage must be in (0, 1)')
    return 1.0 - train_percentage


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
    from data.eeg_dataset import EEGDataset as _EEGDataset
    EEGDataset = _EEGDataset
    _imported = True


def sample_uids(subject, trial_indices):
    """Build stable sample keys as (subject_id, trial_idx_within_subject)."""
    trial_indices = np.asarray(trial_indices, dtype=np.int64)
    subject_ids = np.full(len(trial_indices), int(subject), dtype=np.int64)
    return np.column_stack((subject_ids, trial_indices))


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


def split_indices_by_label_ordered(all_indices, labels, train_percentage):
    """Benchmark-style few-shot split: first train_percentage per class.

    ``all_indices`` and ``labels`` must be in raw trial order. Returned train/test
    indices also preserve that original order, matching boolean-mask selection.
    """
    train_percentage = float(train_percentage)
    if train_percentage <= 0.0 or train_percentage >= 1.0:
        raise ValueError('train_percentage must be in (0, 1)')
    all_indices = np.asarray(all_indices)
    labels = np.asarray(labels)
    train_mask = np.zeros(len(all_indices), dtype=bool)
    test_mask = np.zeros(len(all_indices), dtype=bool)
    for label in np.unique(labels):
        label_positions = np.where(labels == label)[0]
        train_size = max(1, int(len(label_positions) * train_percentage))
        train_mask[label_positions[:train_size]] = True
        test_mask[label_positions[train_size:]] = True
    return list(all_indices[train_mask]), list(all_indices[test_mask])


def subject_split_ordered_fewshot(dataset_name, subject, train_percentage=0.3,
                                  data_mode=None, return_uid=False):
    """Per-subject classification few-shot split in benchmark order.

    For each class, the first ``train_percentage`` trials in the original loaded
    order become calibration data; the rest become test data. No split RNG is used.
    """
    _ensure_imports()
    X, y = load_subject_raw(dataset_name, subject, data_mode=data_mode)
    idx = np.arange(len(y))
    idx_tr, idx_te = split_indices_by_label_ordered(
        idx, y, train_percentage=train_percentage)
    if return_uid:
        return (X[idx_tr], y[idx_tr], X[idx_te], y[idx_te],
                sample_uids(subject, idx_tr), sample_uids(subject, idx_te))
    return X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]


def load_subject_raw(dataset_name, subject, data_mode=None):
    """Raw default source-named session for one subject.

    Returns ``(X (N,C,1000) float32, y (N,) int64)``. Pass an explicit
    ``data_mode`` only when reproducing a legacy split.
    """
    _ensure_imports()
    kwargs = {'dataset_name': dataset_name, 'sub': [subject]}
    if data_mode is not None:
        kwargs['data_mode'] = data_mode
    args = SimpleNamespace(**kwargs)
    ds = EEGDataset(args=args)
    X = np.asarray(ds.X, dtype=np.float32)
    y = np.asarray(ds.y, dtype=np.int64)
    return X, y


def loso_split(dataset_name, test_subject, num_subjects=None, data_mode=None,
               return_uid=False):
    """Leave-One-Subject-Out fold: all subjects except ``test_subject`` form the
    train set, ``test_subject`` is the test set. Returns
    ``(X_tr, y_tr, subj_tr, X_te, y_te)`` where ``subj_tr`` is the per-trial
    subject id (for subject-balanced sampling / per-subject EA). Raw
    ``(N, C_native, 1000)`` @ 250 Hz; each adapter preprocesses on top (the
    MIRepNet teacher must EA per subject-group — see finetune_teacher_loso).

    No randomness: the fold is fully determined by ``test_subject``.
    """
    _ensure_imports()
    if num_subjects is None:
        num_subjects = config.load_dataset_config(dataset_name)['num_subjects']
    Xtr, ytr, subj, uid_tr = [], [], [], []
    for s in range(num_subjects):
        X, y = load_subject_raw(dataset_name, s, data_mode=data_mode)
        uid = sample_uids(s, np.arange(len(y)))
        if s == test_subject:
            Xte, yte, uid_te = X, y, uid
        else:
            Xtr.append(X); ytr.append(y)
            subj.append(np.full(len(y), s, dtype=np.int64))
            uid_tr.append(uid)
    out = (np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(subj),
           Xte, yte)
    if return_uid:
        out = out + (np.concatenate(uid_tr), uid_te)
    return out


def subject_split(dataset_name, subject, val_split=0.3, seed=666,
                  data_mode=None, return_uid=False):
    """Deterministic per-subject calibration/test split.

    Returns ``(X_tr, y_tr, X_te, y_te)`` as numpy arrays. ``val_split`` is the
    TEST fraction (MIRepNet convention: 0.3 -> 70% calib / 30% test). The split
    indices are reproducible from ``seed`` and shared by every adapter, which is
    what guarantees cross-model sample alignment.
    """
    _ensure_imports()
    X, y = load_subject_raw(dataset_name, subject, data_mode=data_mode)
    idx = list(range(len(y)))
    idx_tr, idx_te = split_indices_with_val_ratio(idx, y, val_split, seed)
    if return_uid:
        return (X[idx_tr], y[idx_tr], X[idx_te], y[idx_te],
                sample_uids(subject, idx_tr), sample_uids(subject, idx_te))
    return X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]



def fewshot_cells(dataset, subjects, seeds, val_split=None, train_percentage=None):
    """Within-subject few-shot cells.

    ``unit`` is the subject id. ``val_split`` is the TEST fraction; pass
    ``train_percentage`` to use benchmark-style TRAIN fraction naming.
    """
    effective_val_split = _fewshot_val_split(val_split, train_percentage)
    for seed in seeds:
        for subj in subjects:
            X_tr, y_tr, X_te, y_te = subject_split(
                dataset, subj, val_split=effective_val_split, seed=seed)
            yield Cell(subj, seed, X_tr, y_tr, X_te, y_te, None)


# Compatibility name for older experiment configs/imports.
within_cells = fewshot_cells


def loso_cells(dataset, folds, seeds, num_subjects=None):
    """Leave-one-subject-out cells; ``unit`` is the held-out subject/fold id."""
    for seed in seeds:
        for fold in folds:
            X_tr, y_tr, subj_ids, X_te, y_te = loso_split(
                dataset, fold, num_subjects=num_subjects)
            yield Cell(fold, seed, X_tr, y_tr, X_te, y_te, subj_ids)


def iter_cells(protocol, dataset, units, seeds, val_split=None, num_subjects=None,
               train_percentage=None):
    """Dispatch to the protocol's cell generator.

    Public protocols are ``fewshot`` and ``loso``. ``within`` remains accepted as
    a legacy alias for ``fewshot`` so old configs keep working.
    """
    protocol = canonical_protocol(protocol)
    if protocol == 'fewshot':
        return fewshot_cells(dataset, units, seeds, val_split, train_percentage)
    if protocol == 'loso':
        return loso_cells(dataset, units, seeds, num_subjects)
    raise ValueError(f'unknown protocol {protocol!r} (fewshot | loso)')


# Compatibility name for older imports.
get_cells = iter_cells
