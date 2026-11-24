"""Read the verified shared AlexMI broadband export for LaBraM experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.channels import AlexMI_chn_names

SOURCE_ROOT = Path('/data1/llx/AlexMI/wide_fre_band')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_alexmi_broadband():
    """Return verified right-hand/feet trials in the shared few-shot contract."""
    manifest_path = SOURCE_ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name in ('X.npy', 'labels.npy', 'trials.csv'):
        record = manifest.get('files', {}).get(name)
        if not isinstance(record, dict) or record.get('sha256') != _sha256(SOURCE_ROOT / name):
            raise ValueError(f'AlexMI broadband manifest verification failed: {name}')
    x = np.load(SOURCE_ROOT / 'X.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(SOURCE_ROOT / 'labels.npy', allow_pickle=False).astype(str)
    trials = pd.read_csv(SOURCE_ROOT / 'trials.csv', dtype={'trial_uid': str})
    if tuple(x.shape) != (480, 16, 1537) or len(labels) != len(x) or len(trials) != len(x):
        raise ValueError('AlexMI broadband source shape or trial count does not match manifest')
    if not np.array_equal(trials['source_row'].to_numpy(), np.arange(len(x))):
        raise ValueError('AlexMI broadband trial rows are out of order')
    if not np.array_equal(labels, trials['class_name'].astype(str).to_numpy()):
        raise ValueError('AlexMI broadband labels do not match trial metadata')
    if trials['trial_uid'].duplicated().any():
        raise ValueError('AlexMI broadband trial UIDs must be unique')
    channels = manifest.get('channels')
    if channels is not None and [str(x).upper() for x in channels] != list(AlexMI_chn_names):
        raise ValueError('AlexMI broadband channel order differs from LaBraM channel names')
    selected = np.flatnonzero(np.isin(labels, ('feet', 'right_hand')))
    selected_trials = trials.iloc[selected].reset_index(drop=True)
    selected_labels = labels[selected]
    subjects = selected_trials['subject'].to_numpy(dtype=np.int64) - 1
    if set(subjects.tolist()) != set(range(8)):
        raise ValueError('AlexMI broadband subject IDs are incomplete')
    if set(selected_trials['session'].astype(str)) != {'0'}:
        raise ValueError('AlexMI broadband session identity is unexpected')
    y = np.where(selected_labels == 'feet', 0, 1).astype(np.int64)
    sample_uids = np.empty((len(selected), 2), dtype=np.int64)
    for subject in range(8):
        indices = np.flatnonzero(subjects == subject)
        counts = np.bincount(y[indices], minlength=2)
        if counts.tolist() != [20, 20]:
            raise ValueError(f'AlexMI subject {subject + 1} has unexpected class coverage')
        sample_uids[indices, 0] = subject
        sample_uids[indices, 1] = np.arange(len(indices))
    return SimpleNamespace(x=np.asarray(x[selected]), y=y, subjects=subjects,
                           sample_uid=sample_uids,
                           trial_uid=selected_trials['trial_uid'].astype(str).to_numpy(),
                           source_manifest_sha256=_sha256(manifest_path),
                           identity={'source_variant': 'AlexMI_wide_fre_band',
                                     'class_mapping': {'feet': 0, 'right_hand': 1}},
                           source_path=SOURCE_ROOT)
