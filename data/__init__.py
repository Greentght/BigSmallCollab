"""Data layer: canonical loading, splitting, and preprocessing (one place).

  - ``eeg_dataset`` : raw npy loader (vendored, byte-faithful).
  - ``split``       : subject / LOSO splits — the cross-model alignment contract.
  - ``preproc``     : EA whitening, channel padding, band-pass / notch, fs constants.
  - ``channels``    : montage names + scalp positions.

The split API is re-exported here so callers do ``import data; data.subject_split(...)``.
"""
from .split import (
    Cell, canonical_protocol, fewshot_cells, get_cells, iter_cells, loso_cells,
    subject_split, loso_split, load_subject_raw, split_indices_with_val_ratio,
    within_cells,
)

__all__ = [
    'Cell', 'canonical_protocol', 'fewshot_cells', 'get_cells', 'iter_cells',
    'loso_cells', 'subject_split', 'loso_split', 'load_subject_raw',
    'split_indices_with_val_ratio', 'within_cells',
]
