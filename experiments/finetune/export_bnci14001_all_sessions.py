#!/usr/bin/env python
"""Export a broadband BNCI2014001 NPY source with both native sessions.

The source retains MOABB's native 250 Hz, 22 EEG channels and inclusive
1001-point epochs. Model input cropping, resampling, CAR and EA are separate.
Existing Git LFS datasets are read through verified local entities without
changing their working-tree files. Run with the isolated MOABB 1.2 overlay:

  PYTHONPATH=/tmp/loso_alignment_deps_moabb \
    /home/lixinli/anaconda3/envs/cbramod/bin/python \
    experiments/finetune/export_bnci14001_all_sessions.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import re
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (BNCI14001_SOURCE_ROOT, DATA_CACHE_ROOT,
                                 require_external_output, resolve_local_file)
OUTPUT = BNCI14001_SOURCE_ROOT
V2_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2'
LEGACY = Path('/data1/llx/BNCI2014001')
CLASS_MAP = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
SESSION_MAP = {'0train': 'session_T', '1test': 'session_E'}
CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
            'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entity(path: Path) -> Path:
    """Resolve existing datasets through the external artifact/LFS store."""
    return resolve_local_file(path)


def write_json(path: Path, value: dict) -> None:
    path = require_external_output(path)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    partial.replace(path)


def run_key(value: object) -> str:
    return re.sub(r'^run[_-]?', '', str(value))


def hydrate_raw(raw_root: Path) -> tuple[dict, list[dict]]:
    """Reuse locally downloaded MAT files in a new writable MOABB raw root."""
    relative = Path('MNE-bnci-data/database/data-sets/001-2014')
    old_manifest = json.loads(entity(V2_ROOT / 'rebuilt/BNCI2014001-4/manifest.json').read_text())
    expected = {Path(item['path']).name: item for item in old_manifest['raw_files']}
    identities = {}
    records = []
    for subject in range(1, 10):
        for role, session in [('T', '0train'), ('E', '1test')]:
            name = f'A{subject:02d}{role}.mat'
            old_path = V2_ROOT / 'mne_data' / relative / name
            resolved = entity(old_path)
            digest = sha256(resolved)
            if digest != expected[name]['sha256']:
                raise RuntimeError(f'Original raw identity changed: {name}')
            target = raw_root / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) != digest:
                    raise RuntimeError(f'Refusing to overwrite different raw file: {target}')
            else:
                partial = target.with_name(target.name + '.partial')
                shutil.copyfile(resolved, partial)
                partial.replace(target)
            identities[(subject, session)] = {'path': str(target), 'sha256': digest}
            records.append({'subject': subject, 'session': session, 'path': str(target),
                            'sha256': digest, 'bytes': target.stat().st_size,
                            'download_url': f'http://bnci-horizon-2020.eu/database/data-sets/001-2014/{name}',
                            'reused_from': str(old_path)})
            print(f'[raw-ready] subject={subject} session={session} sha256={digest}', flush=True)
    return identities, records


def legacy_mapping(trials: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    old_meta = pd.read_csv(entity(LEGACY / 'meta.csv'))
    old_labels = np.load(entity(LEGACY / 'labels.npy'), allow_pickle=True).astype(str)
    old_x = np.load(entity(LEGACY / 'X.npy'), mmap_mode='r')
    if len(old_meta) != 5184 or len(old_labels) != 5184 or old_x.shape != (5184, 22, 1001):
        raise RuntimeError('Unexpected legacy full-session shape or trial count')
    old_meta['class_name'] = old_labels
    old_meta['legacy_raw_row'] = np.arange(len(old_meta), dtype=np.int64)
    if 'Unnamed: 0' in old_meta and not np.array_equal(old_meta['Unnamed: 0'], old_meta['legacy_raw_row']):
        raise RuntimeError('Legacy metadata row index differs from its array order')
    old_meta['event_ordinal'] = old_meta.groupby(['subject', 'session', 'run'], sort=False).cumcount()
    old_meta['session'] = old_meta['session'].map({v: k for k, v in SESSION_MAP.items()})
    old_meta['run'] = old_meta['run'].map(run_key)
    keys = ['subject', 'session', 'run', 'event_ordinal']
    if old_meta[keys].isna().any().any() or old_meta.duplicated(keys).any() or trials.duplicated(keys).any():
        raise RuntimeError('Session mapping is incomplete or duplicate trial keys exist')
    merged = trials[['source_row', 'trial_uid', 'class_name'] + keys].merge(
        old_meta[['legacy_raw_row', 'class_name'] + keys], on=keys,
        how='outer', suffixes=('_new', '_legacy'), validate='one_to_one', indicator=True)
    all_present = bool(merged['_merge'].eq('both').all())
    labels_match = bool(merged['class_name_new'].eq(merged['class_name_legacy']).all())
    if not all_present or not labels_match or len(merged) != 5184:
        raise RuntimeError(f'Legacy trial identity check failed: all_present={all_present}, labels_match={labels_match}')
    merged = merged.sort_values('source_row').reset_index(drop=True)
    mapping = merged[['legacy_raw_row', 'source_row', 'trial_uid'] + keys].copy()
    mapping = mapping.rename(columns={'source_row': 'rebuilt_row', 'session': 'native_session', 'run': 'run_key'})
    mapping['class_name'] = merged['class_name_new']
    mapping['legacy_session'] = mapping['native_session'].map(SESSION_MAP)
    mapping['paired_verified'] = True
    return mapping, {'one_to_one': True, 'mapped_trials': 5184, 'all_labels_match': True,
                     'mismatch_count': 0, 'legacy_shape': list(old_x.shape),
                     'legacy_files': {name: {'path': str(LEGACY / name),
                                            'sha256': sha256(entity(LEGACY / name))}
                                      for name in ['X.npy', 'labels.npy', 'meta.csv']}}


def verify_v2(x: np.ndarray, trials: pd.DataFrame) -> dict:
    folder = V2_ROOT / 'rebuilt/BNCI2014001-4'
    prior_x_path = entity(folder / 'X.npy')
    prior_x = np.load(prior_x_path, mmap_mode='r')
    prior_trials = pd.read_csv(entity(folder / 'trials.csv'))
    selected = trials['session'].eq('0train').to_numpy()
    current_trials = trials.loc[selected].reset_index(drop=True)
    if current_trials['trial_uid'].tolist() != prior_trials['trial_uid'].tolist():
        raise RuntimeError('New train trial UID order differs from v2 reference source')
    if not np.array_equal(current_trials['label_id'], prior_trials['label_id']):
        raise RuntimeError('New train label order differs from v2 reference source')
    new_rows = np.flatnonzero(selected)
    exact = True
    close = True
    max_abs = 0.0
    squared_error = 0.0
    elements = 0
    for start in range(0, len(new_rows), 96):
        current = x[new_rows[start:start + 96], :, :1000]
        previous = np.asarray(prior_x[start:start + len(current)])
        delta = current - previous
        exact = exact and bool(np.array_equal(current, previous))
        close = close and bool(np.allclose(current, previous, rtol=1e-10, atol=1e-10))
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        squared_error += float(np.sum(delta * delta))
        elements += delta.size
    if not close:
        raise RuntimeError(f'New train waveform differs from v2 reference source: max_abs={max_abs}')
    return {'selected_session': '0train', 'selected_trials': len(new_rows),
            'comparison_points_per_trial': 1000, 'trial_uid_order_identical': True,
            'class_ids_identical': True, 'arrays_exactly_equal': exact,
            'arrays_allclose': close, 'allclose_rtol': 1e-10, 'allclose_atol': 1e-10,
            'max_absolute_difference': max_abs,
            'rmse': float(np.sqrt(squared_error / elements)),
            'reference_X_sha256': sha256(prior_x_path)}


def export(output: Path) -> dict:
    output = require_external_output(output)
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / 'raw/mne_data'
    raw_root.mkdir(parents=True, exist_ok=True)
    os.environ['MNE_DATA'] = str(raw_root)
    os.environ['MNE_DATASETS_BNCI_PATH'] = str(raw_root)

    import mne
    import moabb
    import scipy
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    if moabb.__version__ != '1.2.0':
        raise RuntimeError(f'Export requires pinned MOABB 1.2.0; got {moabb.__version__}')
    if mne.__version__ != '1.11.0':
        raise RuntimeError(f'Export requires pinned MNE 1.11.0; got {mne.__version__}')
    moabb.set_log_level('ERROR')
    mne.set_log_level('ERROR')
    identities, raw_records = hydrate_raw(raw_root)
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=0.1, fmax=75.0,
                            tmin=0.0, tmax=None, baseline=None, resample=None)
    subjects = list(dataset.subject_list)
    if subjects != list(range(1, 10)):
        raise RuntimeError(f'Unexpected subject set: {subjects}')
    print('[extract] MOABB raw filter=0.1–75 Hz all 9 subjects, both sessions', flush=True)
    x, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    if x.shape != (5184, 22, 1001) or not np.isfinite(x).all():
        raise RuntimeError(f'Unexpected source shape or nonfinite input: {x.shape}')
    y = np.asarray([CLASS_MAP[label] for label in labels], dtype=np.int64)
    trials = meta.reset_index(drop=True).copy()
    trials['run'] = trials['run'].astype(str)
    trials['session'] = trials['session'].astype(str)
    trials['event_ordinal'] = trials.groupby(['subject', 'session', 'run'], sort=False).cumcount()
    trials['source_row'] = np.arange(len(trials), dtype=np.int64)
    trials['class_name'] = labels
    trials['label_id'] = y
    trials['native_session'] = trials['session']
    trials['legacy_session'] = trials['session'].map(SESSION_MAP)
    trials['native_fs_hz'] = 250
    trials['channels'] = 22
    raw_ids = [identities[(int(row.subject), str(row.session))] for row in trials.itertuples(index=False)]
    trials['raw_file'] = [item['path'] for item in raw_ids]
    trials['raw_file_sha256'] = [item['sha256'] for item in raw_ids]
    trials['trial_uid'] = [
        f'BNCI2014001-4|{row.raw_file_sha256}|sub-{int(row.subject):02d}'
        f'|session-{row.session}|run-{run_key(row.run)}'
        f'|event-{int(row.event_ordinal):04d}|class-{row.class_name}'
        for row in trials.itertuples(index=False)]
    if trials['trial_uid'].duplicated().any():
        raise RuntimeError('Duplicate physical trial UIDs')
    counts = trials.groupby(['subject', 'session']).size()
    if len(counts) != 18 or not counts.eq(288).all() or set(trials['session']) != set(SESSION_MAP):
        raise RuntimeError(f'Unexpected subject/session trial counts: {counts.to_dict()}')
    # Inspect native Raw metadata without refiltering or constructing a second full Epochs array.
    native_runs = dataset._get_single_subject_data(1)
    native_info = next(iter(native_runs['0train'].values())).info
    eeg_names = [native_info['ch_names'][idx] for idx in mne.pick_types(native_info, eeg=True)]
    if eeg_names != CHANNELS or float(native_info['sfreq']) != 250.0:
        raise RuntimeError(f'Unexpected native EEG channels/fs: {eeg_names}/{native_info["sfreq"]}')
    mapping, mapping_report = legacy_mapping(trials)
    v2_report = verify_v2(x, trials)
    print(f'[verified] legacy all5184 paired; v2 train exact={v2_report["arrays_exactly_equal"]}', flush=True)
    tmp = output / 'X.npy.partial'
    with tmp.open('wb') as handle:
        np.save(handle, x, allow_pickle=False)
    tmp.replace(output / 'X.npy')
    np.save(output / 'labels.npy', labels, allow_pickle=False)
    np.save(output / 'y.npy', y, allow_pickle=False)
    trials.to_csv(output / 'trials.csv', index=False)
    trials.to_csv(output / 'meta.csv', index=False)
    mapping.to_csv(output / 'legacy_row_mapping.csv', index=False)
    write_json(output / 'verification.json', {'legacy_mapping': mapping_report,
                                             'existing_broadband_v2_train': v2_report})
    manifest = {
        'protocol_version': 'loso_source_v3', 'source_dataset': 'BNCI2014001',
        'source_kind': 'MOABB_all_sessions_broadband_NPY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'moabb_version': moabb.__version__, 'mne_version': mne.__version__,
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'source_shape': list(x.shape), 'source_dtype': str(x.dtype),
        'effective_native_fs_hz': 250, 'sampling_rate_hz': 250,
        'channels': CHANNELS, 'channel_count': len(CHANNELS),
        'source_unit': 'microvolt', 'moabb_unit_factor': float(dataset.unit_factor),
        'unit_conversion': 'Raw Volts multiplied by dataset.unit_factor in get_data(return_epochs=False)',
        'class_map': CLASS_MAP, 'retained_sessions': list(SESSION_MAP),
        'legacy_session_map': SESSION_MAP,
        'trial_counts_by_session': {str(k): int(v) for k, v in trials.groupby('session').size().items()},
        'trial_counts_by_subject': {str(k): int(v) for k, v in trials.groupby('subject').size().items()},
        'moabb_fmin_hz': 0.1, 'moabb_fmax_hz': 75.0,
        'moabb_tmin_s': 0.0, 'moabb_tmax_s': None, 'moabb_baseline': None,
        'paradigm_resample_hz': None, 'dataset_event_interval_s': list(dataset.interval),
        'preprocessing_order': ['continuous_Raw_0.1_to_75_Hz_MNE_filter',
                                'event_epoch_native_250_Hz_inclusive_endpoint',
                                'EEG_channel_selection', 'MOABB_unit_factor_to_microvolt'],
        'source_filter': {'method': 'MNE Raw.filter defaults via MOABB 1.2.0',
                          'continuous_before_epoching': True, 'fmin_hz': 0.1, 'fmax_hz': 75.0},
        'model_side_processing_applied': {'crop': False, 'resample': False,
                                          'CAR': False, 'EA': False, 'notch': False},
        'trial_uid_scheme': 'existing BNCI2014001-4 physical UID prefix preserved across both sessions',
        'raw_files': raw_records, 'legacy_pairing_report': mapping_report,
        'reference_v2_train_verification': v2_report,
        'export_script_sha256': sha256(Path(__file__)),
        'build_elapsed_seconds': time.time() - started,
        'files': {name: {'bytes': (output / name).stat().st_size,
                          'sha256': sha256(output / name)}
                  for name in ['X.npy', 'labels.npy', 'y.npy', 'trials.csv', 'meta.csv',
                               'legacy_row_mapping.csv', 'verification.json']},
    }
    write_json(output / 'manifest.json', manifest)
    print(f'[complete] {output} shape={x.shape} seconds={time.time() - started:.1f}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    export(arguments.output.resolve())
