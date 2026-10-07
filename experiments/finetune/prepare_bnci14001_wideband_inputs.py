#!/usr/bin/env python
"""Prepare both MIRepNet/CBraMod tasks from the all-session broadband NPY.

The source retains both sessions. These LOSO inputs deliberately select only
0train/session_T and preserve physical trial UIDs across binary/four-class tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.finetune.export_bnci14001_all_sessions import entity, sha256, write_json
from experiments.finetune import prepare_loso_alignment_inputs as reference

SOURCE = ROOT / 'data_cache/loso_source_v3/BNCI2014001'
PROFILE = 'wideband_npy_v3'
VARIANT = 'all_sessions_source_train_session'
DATASETS = ('BNCI2014001', 'BNCI2014001-4')
MODELS = ('cbramod', 'mirepnet')


def validate_source() -> tuple[np.ndarray, pd.DataFrame, dict]:
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if (manifest['effective_native_fs_hz'] != 250
            or manifest['moabb_fmin_hz'] != 0.1 or manifest['moabb_fmax_hz'] != 75.0):
        raise RuntimeError('Unexpected source sampling rate or source filter')
    for name, record in manifest['files'].items():
        path = SOURCE / name
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Source file changed: {path}')
    x = np.load(SOURCE / 'X.npy', mmap_mode='r')
    trials = pd.read_csv(SOURCE / 'trials.csv')
    y = np.load(SOURCE / 'y.npy')
    labels = np.load(SOURCE / 'labels.npy', allow_pickle=False).astype(str)
    if x.shape != (5184, 22, 1001) or len(trials) != 5184:
        raise RuntimeError('Expected the complete 5184-trial, 1001-point source')
    if trials['session'].value_counts().to_dict() != {'0train': 2592, '1test': 2592}:
        raise RuntimeError('Source must preserve both full sessions')
    if (trials['trial_uid'].duplicated().any()
            or not np.array_equal(trials['label_id'].to_numpy(), y)
            or not np.array_equal(trials['class_name'].astype(str).to_numpy(), labels)):
        raise RuntimeError('Source trial table and arrays disagree')
    return x, trials, manifest


def recipe(model: str, spec: dict) -> dict:
    if model == 'cbramod':
        cfg = reference.get_profile('reference_aligned', model, 'BNCI2014001-4', spec)
        cfg['source_cast'] = 'float64_broadband_all_session_npy'
        return cfg
    return {'target_fs': 250, 'duration': 4, 'l_freq': 8.0, 'h_freq': 30.0,
            'filter_order': 4, 'filter_phase': 'zero_phase_filtfilt',
            'source_cast': 'float64_broadband_all_session_npy',
            'ea': 'separate_covariance_for_each_subject_in_selected_task',
            'ea_test_regime': 'transductive_all_unlabeled_target_trials',
            'channel_mapping': 'inverse_distance_to_45_channel_template'}


def compare_reference_cbramod(output: Path) -> dict:
    prior = reference.INPUT_ROOT / 'reference_aligned/BNCI2014001-4/cbramod/rebuilt_source'
    current_trials = pd.read_csv(output / 'trials.csv')
    prior_trials = pd.read_csv(entity(prior / 'trials.csv'))
    same_uids = current_trials['trial_uid'].tolist() == prior_trials['trial_uid'].tolist()
    same_labels = np.array_equal(current_trials['label_id'], prior_trials['label_id'])
    current = np.load(output / 'X.npy', mmap_mode='r')
    previous = np.load(entity(prior / 'X.npy'), mmap_mode='r')
    if current.shape != previous.shape:
        raise RuntimeError('Reference model input shape differs')
    exact, max_abs = True, 0.0
    for start in range(0, len(current), 64):
        a, b = current[start:start + 64], previous[start:start + 64]
        exact = exact and bool(np.array_equal(a, b))
        max_abs = max(max_abs, float(np.max(np.abs(a.astype(np.float64) - b))))
    if not (same_uids and same_labels and exact):
        raise RuntimeError(f'Wideband input differs from previous aligned source: max_abs={max_abs}')
    return {'same_trial_uid_order': same_uids, 'same_labels': same_labels,
            'all_input_elements_identical': exact, 'max_absolute_difference': max_abs,
            'previous_input_sha256': sha256(entity(prior / 'X.npy'))}


def prepare(model: str, dataset: str, x_source, source_trials: pd.DataFrame,
            source_manifest: dict, spec: dict) -> dict:
    selected = source_trials['session'].eq('0train')
    if dataset == 'BNCI2014001':
        selected &= source_trials['class_name'].isin(['left_hand', 'right_hand'])
    rows = np.flatnonzero(selected.to_numpy())
    trials = source_trials.iloc[rows].reset_index(drop=True)
    per_subject = 144 if dataset == 'BNCI2014001' else 288
    if len(rows) != 9 * per_subject or trials.groupby('subject').size().to_dict() != {
            subject: per_subject for subject in range(1, 10)}:
        raise RuntimeError(f'{dataset}: wrong selected subject/trial count')
    y = trials['label_id'].to_numpy(dtype=np.int64)
    nclasses = 2 if dataset == 'BNCI2014001' else 4
    if not np.array_equal(np.unique(y), np.arange(nclasses)):
        raise RuntimeError('Unexpected class IDs')
    subjects = trials['subject'].to_numpy(dtype=np.int64) - 1
    cfg = recipe(model, spec)
    output = reference.INPUT_ROOT / PROFILE / dataset / model / VARIANT
    source_sha = sha256(SOURCE / 'manifest.json')
    existing = output / 'manifest.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved['source_manifest_sha256'] != source_sha or saved['resolved_profile'] != cfg:
            raise RuntimeError(f'Refusing to overwrite a different input cache: {output}')
        for name, record in saved['files'].items():
            p = output / name
            if p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']:
                raise RuntimeError(f'Existing input cache is incomplete or changed: {p}')
        print(f'[reuse-input] {dataset}/{model}: immutable cache verified', flush=True)
        return saved
    output.mkdir(parents=True, exist_ok=True)
    input_shape = (22, 4, 200) if model == 'cbramod' else (45, 1000)
    partial = output / 'X.npy.partial'
    target = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                     shape=(len(rows),) + input_shape)
    if model == 'cbramod':
        for start in range(0, len(rows), 96):
            stop = min(start + 96, len(rows))
            # Reference source crops the inclusive endpoint before resampling.
            chunk = np.asarray(x_source[rows[start:stop], :, :1000], dtype=np.float64)
            result, _ = reference.transform_chunk(chunk, 250, cfg, model)
            if not np.isfinite(result).all():
                raise RuntimeError('Non-finite CBraMod input')
            target[start:stop] = result
            print(f'[prepare-input] {dataset}/{model}: {stop}/{len(rows)}', flush=True)
        processing = ['select_0train', 'select_task_classes', 'first_1000_native_samples',
                      'MNE_resample_250_to_200', 'bandpass_0.3_75', 'notch_60_Q30',
                      'CAR', 'patchify_22_4_200', 'float32']
    else:
        from data.preproc import bandpass
        from models.mirepnet.adapter import MIRepNetAdapter
        adapter = MIRepNetAdapter(device='cpu', dataset_name=dataset, skip_preprocess=True)
        for subject in range(9):
            locations = np.flatnonzero(subjects == subject)
            chunk = np.asarray(x_source[rows[locations], :, :1000], dtype=np.float64)
            filtered = bandpass(chunk, 250, 8.0, 30.0)
            result = adapter.ea_pad_per_subject(filtered, np.full(len(locations), subject))
            if not np.isfinite(result).all() or result.shape[1:] != input_shape:
                raise RuntimeError('Invalid MIRepNet EA/channel interpolation output')
            target[locations] = result
            print(f'[prepare-input] {dataset}/{model}: subject {subject + 1}/9', flush=True)
        processing = ['select_0train', 'select_task_classes', 'first_1000_native_samples',
                      'epoch_Butterworth_8_30_order4_filtfilt', 'per_subject_EA',
                      'inverse_distance_interpolation_45_channels', 'float32']
    target.flush()
    del target
    partial.replace(output / 'X.npy')
    np.save(output / 'y.npy', y, allow_pickle=False)
    np.save(output / 'subjects.npy', subjects, allow_pickle=False)
    pd.DataFrame({'trial_uid': trials['trial_uid'], 'subject_zero_based': subjects,
                  'label_id': y}).to_csv(output / 'trials.csv', index=False)
    manifest = {'profile': PROFILE, 'dataset': dataset, 'model': model, 'variant': VARIANT,
                'source_manifest': str(SOURCE / 'manifest.json'),
                'source_manifest_sha256': source_sha, 'source_protocol': 'loso_source_v3',
                'source_all_session_trial_count': 5184,
                'source_manifest_or_hashes': source_manifest,
                'selected_session': '0train', 'legacy_selected_session': 'session_T',
                'selected_source_rows': rows.tolist(), 'trial_count': len(rows),
                'source_fs_hz': 250, 'input_shape': list(input_shape), 'input_dtype': 'float32',
                'class_mapping': {k: v for k, v in source_manifest['class_map'].items()
                                  if v < nclasses}, 'resolved_profile': cfg,
                'preprocessing_order': processing, 'files': {}}
    for name in ['X.npy', 'y.npy', 'subjects.npy', 'trials.csv']:
        p = output / name
        manifest['files'][name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    if model == 'cbramod' and dataset == 'BNCI2014001-4':
        manifest['previous_broadband_input_comparison'] = compare_reference_cbramod(output)
    write_json(existing, manifest)
    print(f'[prepared] {dataset}/{model} shape={[len(rows), *input_shape]}', flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    x, trials, source = validate_source()
    spec = yaml.safe_load(reference.SPEC_PATH.read_text())
    for model in args.models:
        for dataset in args.datasets:
            prepare(model, dataset, x, trials, source, spec)


if __name__ == '__main__':
    main()
