"""Protocol cell generators — turn a (dataset, subjects, seeds) spec into the
per-cell data the runner trains on, using the canonical ``core.data`` splits so
every experiment sees identical samples/order (the cross-model alignment contract).

A *cell* is one training run's data:
    Cell(unit, seed, X_tr, y_tr, X_te, y_te, subj_ids)
where ``unit`` is the pairing unit for stats (subject id within-subject, held-out
fold id for LOSO) and ``subj_ids`` is the per-train-trial subject id (LOSO only,
for subject-balanced sampling; ``None`` within-subject).

Teacher artifacts are keyed by ``(dataset, teacher, unit, seed, 'train')`` — the
same ``unit`` — so a cached teacher lines up with each cell's train split.
"""
from collections import namedtuple

import data

Cell = namedtuple('Cell', 'unit seed X_tr y_tr X_te y_te subj_ids')


def within_cells(dataset, subjects, seeds, val_split):
    """Within-subject: per subject, stratified calib/test split (seed-reproducible).
    ``unit`` = subject id; teacher artifact key uses the same subject id."""
    for seed in seeds:
        for subj in subjects:
            X_tr, y_tr, X_te, y_te = data.subject_split(
                dataset, subj, val_split=val_split, seed=seed)
            yield Cell(subj, seed, X_tr, y_tr, X_te, y_te, None)


def loso_cells(dataset, folds, seeds, num_subjects=None):
    """Leave-One-Subject-Out: each fold holds out one subject as test, the rest are
    train. ``unit`` = held-out subject id. The fold is deterministic (no seed effect
    on the split); seeds still vary student training. ``subj_ids`` carries the
    per-trial train subject id for subject-balanced sampling."""
    for seed in seeds:
        for fold in folds:
            X_tr, y_tr, subj_ids, X_te, y_te = data.loso_split(
                dataset, fold, num_subjects=num_subjects)
            yield Cell(fold, seed, X_tr, y_tr, X_te, y_te, subj_ids)


def get_cells(protocol, dataset, units, seeds, val_split, num_subjects=None):
    """Dispatch to the protocol's cell generator. ``units`` = subjects (within) or
    held-out folds (loso)."""
    if protocol == 'within':
        return within_cells(dataset, units, seeds, val_split)
    if protocol == 'loso':
        return loso_cells(dataset, units, seeds, num_subjects)
    raise ValueError(f'unknown protocol {protocol!r} (within | loso)')
