#!/usr/bin/env python
"""Prepare model inputs for the broadband 004/5001 source refresh."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import require_external_output, resolve_local_file

DATA_ROOT = Path('/data1/llx')
PROJECT_ROOT = Path('/data1/llx/BigSmallcollab')
SPEC_PATH = ROOT / 'configs/reproductions/loso_source_refresh_004_5001_v1.yaml'
INPUT_ROOT = PROJECT_ROOT / 'cache/reproductions/loso_source_refresh_004_5001_v1/model_inputs'

DATASETS = {
    'BNCI2014004': {
        'source_variant': 'broadband_0_120hz', 'session': 'session_3',
        'legacy_meta': 'meta004.csv', 'fs_hz': 250, 'n_subjects': 9,
        'channels': 3, 'per_subject_trials': [160, 120, 160, 160, 160, 160, 160, 160, 160],
        'class_map': {'left_hand': 0, 'right_hand': 1}, 'canonical_samples': 1000,
    },
    'BNCI2015001': {
        'source_variant': 'broadband_0p1_75hz', 'session': 'session_A',
        'legacy_meta': 'meta.csv', 'fs_hz': 512, 'n_subjects': 12,
        'channels': 13, 'per_subject_trials': [200] * 12,
        'class_map': {'feet': 0, 'right_hand': 1}, 'canonical_samples': 1000,
    },
}
MODELS = ('mirepnet', 'cbramod', 'ifnet', 'eegnet', 'adfcnn')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_window(x: np.ndarray, fs_hz: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if fs_hz == 250:
        x = x[..., :1000]
    elif fs_hz == 512:
        x = scipy_resample(x, int(round(x.shape[-1] * 250 / fs_hz)), axis=-1)
        x = x[..., :1000]
    else:
        raise ValueError(f'unsupported native sampling rate: {fs_hz}')
    if x.shape[-1] != 1000:
        raise RuntimeError(f'expected a 4 s/250 Hz window, got {x.shape}')
    return x


def load_selected(dataset: str):
    cfg = DATASETS[dataset]
    source = DATA_ROOT / dataset / cfg['source_variant']
    source_manifest_path = resolve_local_file(source / 'manifest.json')
    source_manifest = json.loads(source_manifest_path.read_text())
    for name, record in source_manifest['files'].items():
        path = resolve_local_file(source / name)
        if path.stat().st_size != int(record['bytes']) or sha256(path) != record['sha256']:
            raise RuntimeError(f'{dataset}: source file fails manifest verification: {path}')
    x = np.load(resolve_local_file(source / 'X.npy'), mmap_mode='r')
    trials = pd.read_csv(resolve_local_file(source / 'trials.csv'))
    mapping = pd.read_csv(resolve_local_file(source / 'legacy_row_mapping.csv'))
    legacy_meta_path = resolve_local_file(DATA_ROOT / dataset / cfg['legacy_meta'])
    legacy_meta = pd.read_csv(legacy_meta_path)
    if len(x) != len(trials) or len(mapping) != len(trials):
        raise RuntimeError(f'{dataset}: source array and row-map counts differ')
    selected = mapping.loc[
        mapping.legacy_raw_row.isin(legacy_meta.index[legacy_meta.session.eq(cfg['session'])])
    ].copy()
    selected = selected.sort_values('legacy_raw_row').reset_index(drop=True)
    if len(selected) != sum(cfg['per_subject_trials']):
        raise RuntimeError(f'{dataset}: selected session trial count mismatch: {len(selected)}')
    source_rows = selected.source_row.to_numpy(dtype=np.int64)
    selected_trials = trials.iloc[source_rows].copy().reset_index(drop=True)
    expected_legacy = legacy_meta.iloc[selected.legacy_raw_row.to_numpy(dtype=np.int64)]
    if not np.array_equal(selected.subject.to_numpy(), expected_legacy.subject.to_numpy(dtype=int)):
        raise RuntimeError(f'{dataset}: selected source subject IDs do not match legacy metadata')
    legacy_labels = np.load(resolve_local_file(DATA_ROOT / dataset / 'labels.npy'),
                            allow_pickle=True).astype(str)
    if not np.array_equal(selected.class_name.astype(str).to_numpy(),
                          legacy_labels[selected.legacy_raw_row.to_numpy(dtype=np.int64)]):
        raise RuntimeError(f'{dataset}: selected source class names do not match legacy labels')
    labels = np.load(resolve_local_file(source / 'labels.npy'), mmap_mode='r').astype(str)
    selected_labels = labels[source_rows]
    if not np.array_equal(selected_labels, selected.class_name.astype(str).to_numpy()):
        raise RuntimeError(f'{dataset}: selected source labels disagree with the row map')
    y = np.asarray([cfg['class_map'][str(label)] for label in selected_labels], dtype=np.int64)
    if not np.array_equal(y, selected.label_id.to_numpy(dtype=np.int64)):
        raise RuntimeError(f'{dataset}: selected label IDs disagree with the task encoding')
    subjects = selected.subject.to_numpy(dtype=np.int64) - 1
    expected_subjects = np.arange(cfg['n_subjects'])
    if not np.array_equal(np.bincount(subjects, minlength=cfg['n_subjects']),
                          np.asarray(cfg['per_subject_trials'])):
        raise RuntimeError(f'{dataset}: per-subject trial counts do not match the LOSO manifest')
    x_canonical = canonical_window(x[source_rows], cfg['fs_hz']).astype(np.float32)
    selected_trials['source_row'] = source_rows
    selected_trials['legacy_raw_row'] = selected.legacy_raw_row.to_numpy(dtype=np.int64)
    selected_trials['subject_zero_based'] = subjects
    selected_trials['label_id'] = y
    selected_trials['class_name'] = selected_labels
    return x_canonical, y, subjects, selected_trials, source_manifest, source


def prepare_model(dataset: str, model: str, overwrite: bool = False) -> None:
    from data.preproc import EA, bandpass, notch, pad_missing_channels_diff
    from data.channels import use_channels_names

    cfg = DATASETS[dataset]
    x, y, subjects, trials, source_manifest, source = load_selected(dataset)
    if model == 'mirepnet':
        from data.channels import (
            BNCI2014004_chn_names, BNCI2015001_chn_names,
        )
        source_channels = (BNCI2014004_chn_names if dataset == 'BNCI2014004'
                           else BNCI2015001_chn_names)
        processed = np.empty((len(x), 45, x.shape[-1]), dtype=np.float32)
        filtered = bandpass(x.astype(np.float64), 250, 8.0, 30.0)
        for subject in np.unique(subjects):
            mask = subjects == subject
            aligned = EA(filtered[mask]).astype(np.float32)
            processed[mask] = pad_missing_channels_diff(
                aligned, use_channels_names, source_channels).astype(np.float32)
        pipeline = ['first_1000_native_or_resampled_samples', '8_30_hz_order4_zero_phase',
                    'euclidean_alignment_per_subject', 'inverse_distance_pad_to_45_channels']
        resolved = {'epochs': 10, 'lr': 0.001, 'batch_size': 8, 'weight_decay': 0.000001,
                    'optimizer': 'adam', 'lr_schedule': 'cosine_per_epoch',
                    'test_ea': 'full_unlabeled_test_subject_pool'}
    elif model == 'cbramod':
        from mne.filter import resample as mne_resample
        processed = np.empty((len(x), x.shape[1], 4, 200), dtype=np.float32)
        for start in range(0, len(x), 96):
            stop = min(start + 96, len(x))
            batch = mne_resample(x[start:stop].astype(np.float64), down=250 / 200, axis=-1)
            batch = bandpass(batch, 200, 0.3, 75.0)
            batch = notch(batch, 200, 60.0)
            processed[start:stop] = batch[..., :800].reshape(stop-start, x.shape[1], 4, 200)
        pipeline = ['canonical_4s_250hz', 'mne_resample_200hz', '0p3_75_hz_order4_zero_phase',
                    '60_hz_notch_q30', 'no_EA', 'no_CAR', 'reshape_native_channels_4x200']
        base = {'optimizer': 'adamw', 'lr': 0.0001, 'weight_decay': 0.05,
                'batch_size': 64, 'epochs': 50, 'dropout': 0.1,
                'label_smoothing': 0.1, 'warmup_epochs': 0,
                'lr_schedule': 'cosine_per_epoch'}
        resolved = base | {'duration_seconds': 4, 'target_fs_hz': 200}
    elif model == 'ifnet':
        processed = x
        pipeline = ['canonical_4s_250hz_wideband_source',
                    'IFNet_internal_filterbank_4_16_and_16_40_hz']
        resolved = {'epochs': 100, 'lr': 0.001, 'batch_size': 16,
                    'weight_decay': 0.01, 'optimizer': 'adamw',
                    'lr_schedule': 'cosine_per_epoch', 'use_filter_bank': True}
    else:
        processed = bandpass(x.astype(np.float64), 250, 8.0, 32.0).astype(np.float32)
        pipeline = ['canonical_4s_250hz', '8_32_hz_order4_zero_phase']
        resolved = {'epochs': 100, 'lr': 0.001, 'batch_size': 32,
                    'weight_decay': 0.0001, 'optimizer': 'adamw',
                    'lr_schedule': 'cosine_per_epoch'}

    output = require_external_output(INPUT_ROOT / dataset / model)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and not overwrite:
        old = json.loads(manifest_path.read_text())
        if (old.get('source_manifest_sha256') == sha256(source / 'manifest.json')
                and old.get('model') == model and old.get('dataset') == dataset
                and old.get('resolved_config') == resolved):
            if all((output / name).is_file()
                   and sha256(output / name) == old['files'][name]['sha256']
                   for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv')):
                print(f'[skip-valid] {dataset}/{model} {output}', flush=True)
                return
        raise RuntimeError(f'refusing to overwrite a different model-input cache: {output}')
    if manifest_path.exists() and overwrite:
        raise RuntimeError('--overwrite is intentionally not supported; use a new cache version')

    np.save(output / 'X.npy', processed.astype(np.float32), allow_pickle=False)
    np.save(output / 'y.npy', y.astype(np.int64), allow_pickle=False)
    np.save(output / 'subjects.npy', subjects.astype(np.int64), allow_pickle=False)
    trials.to_csv(output / 'trials.csv', index=False)
    source_manifest_sha = sha256(source / 'manifest.json')
    manifest = {
        'protocol': 'loso_source_refresh_004_5001_v1',
        'dataset': dataset, 'model': model,
        'source_variant': cfg['source_variant'],
        'source_manifest': str(source / 'manifest.json'),
        'source_manifest_sha256': source_manifest_sha,
        'source_X_sha256': source_manifest['files']['X.npy']['sha256'],
        'source_filter_hz': source_manifest['source_filter_hz'],
        'selected_session': cfg['session'],
        'selected_trials': len(y),
        'subject_trial_counts': cfg['per_subject_trials'],
        'class_mapping': cfg['class_map'],
        'source_fs_hz': cfg['fs_hz'], 'canonical_fs_hz': 250,
        'canonical_window_samples': 1000,
        'input_shape': list(processed.shape), 'input_dtype': 'float32',
        'input_unit': 'microvolts as returned by MOABB, model transforms applied as listed',
        'preprocessing': pipeline, 'resolved_config': resolved,
        'ea_regime': 'transductive_loso_for_mirepnet' if model == 'mirepnet' else 'disabled',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'prepare_script_sha256': sha256(Path(__file__)),
        'files': {},
    }
    for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'):
        manifest['files'][name] = {'bytes': (output / name).stat().st_size,
                                   'sha256': sha256(output / name)}
    tmp = manifest_path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, manifest_path)
    print(f'[prepared] {dataset}/{model} n={len(y)} shape={processed.shape} '
          f'cache={output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS),
                        default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            prepare_model(dataset, model)


if __name__ == '__main__':
    main()
