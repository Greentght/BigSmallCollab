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
import config
from experiments.storage import require_external_output, resolve_local_file

DATA_ROOT = Path('/data1/llx')
DATASETS = {}
for _dataset, _meta in {'BNCI2014004': 'meta004.csv', 'BNCI2015001': 'meta.csv'}.items():
    _facts = config.load_dataset_config(_dataset)
    _current = config.load_loso_dataset_config(_dataset)
    DATASETS[_dataset] = dict(
        source_directory=_current['source_directory'],
        source_variant=Path(_current['source_directory']).name,
        session=_current['selected_session'],
        fs_hz=int(_current['native_sampling_rate_hz']),
        per_subject_trials=_current['per_subject_trials'],
        class_map=_current['class_mapping'],
        legacy_meta=_meta, n_subjects=_facts['num_subjects'],
        channels=_facts['channels'], canonical_samples=1000,
    )
MODELS = ('mirepnet', 'cbramod', 'ifnet', 'eegnet', 'adfcnn')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def current_input_policy(dataset: str, model: str, model_cfg: dict | None = None) -> tuple[dict, list[str]]:
    """Validate the fixed preprocessing supported by the current baseline.

    Training hyperparameters do not identify an input cache. The previously
    prepared manifests already record this fixed pipeline, so describing its
    defaults explicitly in YAML does not require rewriting any artifact.
    """
    dataset_cfg = config.load_dataset_config(dataset)
    source_cfg = config.load_loso_dataset_config(dataset)
    model_cfg = model_cfg or config.load_model_config(model, dataset, 'loso')
    common = {'duration_seconds': 4.0, 'sample_rate': 250, 'samples': 1000,
              'skip_preprocess': model in ('mirepnet', 'cbramod')}
    pipelines = {
        'mirepnet': ['first_1000_native_or_resampled_samples', '8_30_hz_order4_zero_phase',
                    'euclidean_alignment_per_subject', 'inverse_distance_pad_to_45_channels'],
        'cbramod': ['canonical_4s_250hz', 'mne_resample_200hz', '0p3_75_hz_order4_zero_phase',
                   '60_hz_notch_q30', 'no_EA', 'no_CAR', 'reshape_native_channels_4x200'],
        'ifnet': ['canonical_4s_250hz_wideband_source',
                  'IFNet_internal_filterbank_4_16_and_16_40_hz'],
        'eegnet': ['canonical_4s_250hz', '8_32_hz_order4_zero_phase'],
        'adfcnn': ['canonical_4s_250hz', '8_32_hz_order4_zero_phase'],
    }
    if model == 'mirepnet':
        common.update(in_channels=45, l_freq=8.0, h_freq=30.0, apply_EA=True,
                      filter_order=4, filter_phase='zero_phase', notch_q=30,
                      ea_scope='per_subject',
                      test_ea_policy='all_unlabeled_held_out_subject_trials',
                      channel_mapping='inverse_distance_to_45_channels')
    elif model == 'cbramod':
        common.update(target_fs=200, l_freq=0.3, h_freq=75.0, notch_freq=60.0,
                      filter_order=4, filter_phase='zero_phase', notch_q=30,
                      norm_method=None, apply_EA=False, scale=1.0)
    elif model == 'ifnet':
        common.update(use_filter_bank=True, filter_bank_hz=[[4.0, 16.0], [16.0, 40.0]],
                      filter_order=5, filter_bank_order=5)
    elif model in ('eegnet', 'adfcnn'):
        common.update(l_freq=8.0, h_freq=32.0, filter_order=4, apply_EA=False,
                      filter_phase='zero_phase', preprocessing_stage='cached_input')
    else:
        raise ValueError(f'unsupported current LOSO model: {model}')
    for key, expected in common.items():
        actual = model_cfg.get(key, expected)
        if actual != expected:
            raise ValueError(f'{dataset}/{model}: current baseline supports {key}={expected!r}, '
                             f'got {actual!r}; register another input recipe before changing preprocessing')
    native_fs = 250 if dataset == 'BNCI2014004' else 512
    if (source_cfg['native_sampling_rate_hz'] != native_fs
            or source_cfg.get('duration_seconds', 4.0) != 4.0
            or dataset_cfg['sample_rate'] != 250):
        raise ValueError(f'{dataset}: current baseline requires native_fs={native_fs}, '
                         'canonical_fs=250, and a 4-second input')
    return source_cfg, pipelines[model]


def validate_input_identity(dataset: str, model: str, manifest: dict,
                            model_cfg: dict | None = None) -> None:
    """Read-only check shared by cache preparation and training consumers."""
    source_cfg, pipeline = current_input_policy(dataset, model, model_cfg)
    source_path = resolve_local_file(Path(source_cfg['source_directory']) / 'manifest.json')
    recorded_path = resolve_local_file(Path(manifest.get('source_manifest', '')))
    if recorded_path.resolve() != source_path.resolve():
        raise RuntimeError(f'{dataset}/{model}: cached source path differs from configs/datasets')
    source = json.loads(source_path.read_text())
    counts = source_cfg['per_subject_trials']
    channels = 45 if model == 'mirepnet' else config.load_dataset_config(dataset)['channels']
    shape = [sum(counts), channels, 4, 200] if model == 'cbramod' else [sum(counts), channels, 1000]
    expected = {
        'protocol': source_cfg['current_baseline_id'],
        'dataset': dataset, 'model': model,
        'source_variant': Path(source_cfg['source_directory']).name,
        'source_manifest_sha256': sha256(source_path),
        'source_X_sha256': source['files']['X.npy']['sha256'],
        'source_filter_hz': source['source_filter_hz'],
        'selected_session': source_cfg['selected_session'],
        'selected_trials': sum(counts), 'subject_trial_counts': counts,
        'class_mapping': source_cfg['class_mapping'],
        'source_fs_hz': source_cfg['native_sampling_rate_hz'],
        'canonical_fs_hz': 250, 'canonical_window_samples': 1000,
        'input_shape': shape, 'input_dtype': 'float32', 'preprocessing': pipeline,
        'ea_regime': 'transductive_loso_for_mirepnet' if model == 'mirepnet' else 'disabled',
    }
    if source_cfg.get('source_bandpass_hz', source['source_filter_hz']) != source['source_filter_hz']:
        raise RuntimeError(f'{dataset}: configured source frequency range differs from the source manifest')
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f'{dataset}/{model}: cached {key} differs from the active input policy')


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
    source = Path(cfg['source_directory'])
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
    cfg = DATASETS[dataset]
    model_cfg = config.load_model_config(model, dataset, 'loso')
    source_cfg, pipeline = current_input_policy(dataset, model, model_cfg)
    output = require_external_output(Path(source_cfg['input_root']) / dataset / model)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if overwrite:
            raise RuntimeError('--overwrite is intentionally not supported; use a new cache version')
        old = json.loads(manifest_path.read_text())
        validate_input_identity(dataset, model, old, model_cfg)
        if all((output / name).is_file()
               and sha256(output / name) == old['files'][name]['sha256']
               for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv')):
            print(f'[skip-valid] {dataset}/{model} {output}', flush=True)
            return
        raise RuntimeError(f'refusing to overwrite a different model-input cache: {output}')

    from data.preproc import EA, bandpass, notch, pad_missing_channels_diff
    from data.channels import use_channels_names

    x, y, subjects, trials, source_manifest, source = load_selected(dataset)
    if model == 'mirepnet':
        from data.channels import (
            BNCI2014004_chn_names, BNCI2015001_chn_names,
        )
        source_channels = (BNCI2014004_chn_names if dataset == 'BNCI2014004'
                           else BNCI2015001_chn_names)
        processed = np.empty((len(x), 45, x.shape[-1]), dtype=np.float32)
        filtered = bandpass(x.astype(np.float64), 250,
                            float(model_cfg['l_freq']), float(model_cfg['h_freq']))
        for subject in np.unique(subjects):
            mask = subjects == subject
            aligned = EA(filtered[mask]).astype(np.float32)
            processed[mask] = pad_missing_channels_diff(
                aligned, use_channels_names, source_channels).astype(np.float32)
        resolved = {key: model_cfg[key] for key in
                    ('epochs', 'lr', 'batch_size', 'weight_decay', 'optimizer')}
        resolved.update(lr_schedule='cosine_per_epoch',
                        test_ea='full_unlabeled_test_subject_pool')
    elif model == 'cbramod':
        from mne.filter import resample as mne_resample
        processed = np.empty((len(x), x.shape[1], 4, 200), dtype=np.float32)
        for start in range(0, len(x), 96):
            stop = min(start + 96, len(x))
            target_fs = int(model_cfg['target_fs'])
            batch = mne_resample(x[start:stop].astype(np.float64), down=250 / target_fs, axis=-1)
            batch = bandpass(batch, target_fs, float(model_cfg['l_freq']), float(model_cfg['h_freq']))
            batch = notch(batch, target_fs, float(model_cfg['notch_freq']))
            processed[start:stop] = batch[..., :800].reshape(stop-start, x.shape[1], 4, 200)
        base = {key: model_cfg[key] for key in
                ('optimizer', 'lr', 'weight_decay', 'batch_size', 'epochs',
                 'dropout', 'label_smoothing', 'warmup_epochs')}
        resolved = base | {'lr_schedule': 'cosine_per_epoch',
                           'duration_seconds': model_cfg['duration_seconds'],
                           'target_fs_hz': model_cfg['target_fs']}
    elif model == 'ifnet':
        processed = x
        resolved = {key: model_cfg[key] for key in
                    ('epochs', 'lr', 'batch_size', 'weight_decay', 'optimizer', 'use_filter_bank')}
        resolved['lr_schedule'] = 'cosine_per_epoch'
    else:
        processed = bandpass(x.astype(np.float64), 250,
                             float(model_cfg['l_freq']), float(model_cfg['h_freq'])).astype(np.float32)
        resolved = {key: model_cfg[key] for key in
                    ('epochs', 'lr', 'batch_size', 'weight_decay', 'optimizer')}
        resolved['lr_schedule'] = 'cosine_per_epoch'

    output.mkdir(parents=True, exist_ok=True)

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
