#!/usr/bin/env python
"""Prepare immutable model-input arrays for the LOSO alignment profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import signal
import yaml


ROOT = Path(__file__).resolve().parents[2]
OLD_DATA = Path('/data1/llx')
SPEC_PATH = ROOT / 'configs/reproductions/loso_config_alignment_v2.yaml'
SOURCE_ROOT = ROOT / 'data_cache/eegfm_alignment_v2/rebuilt'
INPUT_ROOT = ROOT / 'data_cache/eegfm_alignment_v2/model_inputs'
MOABB_OVERLAY = Path('/tmp/loso_alignment_deps_moabb')
if MOABB_OVERLAY.is_dir():
    sys.path.insert(0, str(MOABB_OVERLAY))

LABELS_REFERENCE = {
    'BNCI2014001-4': {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3},
    'BNCI2014004': {'left_hand': 0, 'right_hand': 1},
    'BNCI2015001': {'right_hand': 0, 'feet': 1},
}
LABELS_LEGACY = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
DATASETS = ('BNCI2014001-4', 'BNCI2014004', 'BNCI2015001')
MODELS = ('cbramod', 'eegnet')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_existing_input(folder: Path, profile: str, dataset: str, model: str,
                        variant: str, cfg: dict, source_manifest: dict) -> bool:
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        return False
    try:
        saved = json.loads(manifest_path.read_text())
        if (saved.get('profile') != profile or saved.get('dataset') != dataset
                or saved.get('model') != model or saved.get('variant') != variant
                or saved.get('resolved_profile') != cfg
                or saved.get('source_manifest_or_hashes') != source_manifest):
            return False
        for filename, record in saved.get('files', {}).items():
            path = folder / filename
            if not path.is_file() or path.stat().st_size != record['bytes']:
                return False
            if sha256(path) != record['sha256']:
                return False
        return all((folder / name).is_file() for name in
                   ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'))
    except Exception:
        return False


def source_info(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    folder = SOURCE_ROOT / dataset
    x = np.load(folder / 'X.npy', mmap_mode='r')
    y = np.load(folder / 'y.npy', mmap_mode='r')
    trials = pd.read_csv(folder / 'trials.csv')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if len(x) != len(y) or len(y) != len(trials):
        raise RuntimeError(f'{dataset}: rebuilt source files disagree on trial count')
    order = np.lexsort((trials.event_ordinal.to_numpy(),
                        trials.run.astype(str).to_numpy(),
                        trials.subject.to_numpy()))
    return x, np.asarray(y), order, trials.trial_uid.astype(str).tolist(), manifest


def bridge_info(dataset: str, variant: str):
    if dataset != 'BNCI2014001-4':
        raise ValueError('source bridge currently only covers BNCI2014001-4')
    new_x, new_y, _, new_uids, new_manifest = source_info(dataset)
    folder = SOURCE_ROOT / dataset
    mapping = pd.read_csv(folder / 'legacy_row_mapping.csv')
    if (len(mapping) != 2592 or not mapping.paired_verified.all()
            or mapping.legacy_raw_row.duplicated().any()
            or mapping.rebuilt_row.duplicated().any()):
        raise RuntimeError('001-4 legacy trial mapping is not fully one-to-one verified')
    mapping = mapping.sort_values('rebuilt_row').reset_index(drop=True)
    rebuilt_idx = mapping.rebuilt_row.to_numpy(dtype=np.int64)
    subjects = mapping.subject.to_numpy(dtype=np.int64) - 1
    uids = mapping.trial_uid.astype(str).tolist()
    class_names = mapping.class_name.astype(str).tolist()
    y_legacy = np.asarray([LABELS_LEGACY[name] for name in class_names], dtype=np.int64)
    if not np.array_equal(new_y[rebuilt_idx], np.asarray([
            LABELS_REFERENCE[dataset][name] for name in class_names], dtype=np.int64)):
        raise RuntimeError('001-4 rebuilt labels do not agree with the verified mapping')
    if variant == 'rebuilt_source':
        return new_x, rebuilt_idx, y_legacy, subjects, uids, new_manifest

    old_meta = pd.read_csv(OLD_DATA / 'BNCI2014001' / 'meta.csv')
    old_labels = np.load(OLD_DATA / 'BNCI2014001' / 'labels.npy', allow_pickle=True).astype(str)
    old_x = np.load(OLD_DATA / 'BNCI2014001' / 'X.npy', mmap_mode='r')
    legacy_rows = mapping.legacy_raw_row.to_numpy(dtype=np.int64)
    if (not np.array_equal(old_meta.session.to_numpy()[legacy_rows],
                           np.asarray(['session_T'] * len(mapping)))
            or not np.array_equal(old_labels[legacy_rows], np.asarray(class_names))):
        raise RuntimeError('legacy source rows fail selected-session/class verification')
    return old_x, legacy_rows, y_legacy, subjects, uids, {
        'source_dataset': 'BNCI2014001', 'legacy_meta_sha256': sha256(OLD_DATA/'BNCI2014001/meta.csv'),
        'legacy_labels_sha256': sha256(OLD_DATA/'BNCI2014001/labels.npy'),
        'legacy_x_sha256': sha256(OLD_DATA/'BNCI2014001/X.npy'),
        'mapping_sha256': sha256(folder/'legacy_row_mapping.csv'),
        'rebuilt_source_manifest_sha256': sha256(folder/'manifest.json'),
    }


def get_profile(profile: str, model: str, dataset: str, spec: dict) -> dict:
    if profile == 'source_bridge':
        if model == 'cbramod':
            source = spec['source_bridge']['legacy_profiles']['cbramod']
            return {
                'optimizer': source['optimizer'], 'lr': source['lr'],
                'weight_decay': source['weight_decay'], 'batch_size': source['batch_size'],
                'epochs': source['epochs'], 'dropout': source['head_dropout'],
                'label_smoothing': source['label_smoothing'], 'target_fs': source['target_fs'],
                'l_freq': source['bandpass_hz'][0], 'h_freq': source['bandpass_hz'][1],
                'notch_freq': source['notch_hz'], 'norm_method': source['normalization'],
                'apply_EA': False, 'duration': source['duration_seconds'],
            'schedule': 'step_table', 'schedule_update': source['lr_schedule_update'],
            'warmup_epochs': 5, 'min_lr': 1e-6, 'scale': 1.0,
            'source_cast': 'float32_legacy_adapter',
            'crop_before_resample': True,
            }
        source = spec['source_bridge']['legacy_profiles']['eegnet']
        return {
            'optimizer': source['optimizer'], 'lr': source['lr'],
            'weight_decay': source['weight_decay'], 'batch_size': source['batch_size'],
            'epochs': source['epochs'], 'dropout': source['dropout'],
            'label_smoothing': source['label_smoothing'], 'target_fs': source['target_fs'],
            'l_freq': None, 'h_freq': None, 'notch_freq': None, 'norm_method': None,
            'apply_EA': False, 'duration': source['duration_seconds'],
            'schedule': 'epoch_cosine', 'schedule_update': 'epoch_end',
            'warmup_epochs': 0, 'min_lr': 0.0, 'scale': 1.0,
            'source_cast': 'float32_legacy_loader',
            'crop_before_resample': True,
        }

    cfg = spec['reference_aligned']['models'][model]
    if model == 'eegnet':
        schedule = spec['reference_aligned']['lr_schedule']
        return {
            'optimizer': cfg['optimizer'], 'lr': cfg['lr'],
            'weight_decay': cfg['weight_decay'], 'batch_size': cfg['batch_size'],
            'epochs': cfg['epochs'], 'dropout': cfg['dropout'],
            'label_smoothing': spec['reference_aligned']['label_smoothing'],
            'target_fs': cfg['target_fs'],
            'l_freq': cfg['bandpass_hz'][0], 'h_freq': cfg['bandpass_hz'][1],
            'notch_freq': cfg['notch_hz'], 'norm_method': cfg['normalization'],
            'apply_EA': spec['reference_aligned']['apply_ea'], 'duration': cfg['duration_seconds'],
            'schedule': 'step_table', 'schedule_update': 'epoch_end_global_step_table',
            'warmup_epochs': schedule['warmup_epochs'], 'min_lr': schedule['min_lr'], 'scale': 1.0,
            'source_cast': 'float64_moabb',
            'crop_before_resample': False,
            'class_weights': cfg['class_weights'],
        }
    per_dataset = cfg['per_dataset'][dataset]
    return {
        'optimizer': cfg['optimizer'], 'lr': cfg['lr'],
        'weight_decay': per_dataset['weight_decay'],
        'batch_size': per_dataset['batch_size'], 'epochs': cfg['epochs'],
        'dropout': cfg['head_dropout'], 'label_smoothing': cfg['label_smoothing'],
        'target_fs': cfg['target_fs'], 'l_freq': cfg['bandpass_hz'][0],
        'h_freq': cfg['bandpass_hz'][1], 'notch_freq': cfg['notch_hz'],
        'norm_method': per_dataset['normalization'],
        'apply_EA': spec['reference_aligned']['apply_ea'],
        'duration': per_dataset['duration_seconds'], 'schedule': 'step_table',
        'schedule_update': 'epoch_end_global_step_table',
        'warmup_epochs': spec['reference_aligned']['lr_schedule']['warmup_epochs'],
        'min_lr': spec['reference_aligned']['lr_schedule']['min_lr'], 'scale': 1.0,
        'source_cast': 'float64_moabb',
        'crop_before_resample': False,
        'class_weights': False,
    }


def transform_chunk(x: np.ndarray, src_fs: int, cfg: dict, model: str) -> tuple[np.ndarray, int]:
    from mne.filter import resample as mne_resample
    x = np.asarray(x, dtype=np.float64)
    if cfg.get('crop_before_resample'):
        source_target_length = int(round(float(cfg['duration']) * src_fs))
        if x.shape[-1] >= source_target_length:
            x = x[..., :source_target_length].copy()
    target_fs = int(cfg['target_fs'])
    if src_fs != target_fs:
        x = mne_resample(x, down=(src_fs / target_fs), axis=-1)
    after_resample = int(x.shape[-1])
    target_length = int(round(float(cfg['duration']) * target_fs))
    if after_resample >= target_length:
        x = x[..., :target_length].copy()
    else:
        out = np.empty(x.shape[:-1] + (target_length,), dtype=np.float64)
        out[..., :after_resample] = x
        remaining = target_length - after_resample
        dst = after_resample
        while remaining > 0:
            ncopy = min(remaining, after_resample)
            out[..., dst:dst+ncopy] = x[..., :ncopy]
            dst += ncopy
            remaining -= ncopy
        x = out

    lo, hi = cfg['l_freq'], cfg['h_freq']
    if lo is not None or hi is not None:
        nyq = target_fs / 2.0
        if lo is None:
            b, a = signal.butter(4, float(hi) / nyq, btype='low')
        elif hi is None:
            b, a = signal.butter(4, float(lo) / nyq, btype='high')
        else:
            b, a = signal.butter(4, [float(lo) / nyq, float(hi) / nyq], btype='band')
        x = signal.filtfilt(b, a, x, axis=-1)
    notch = cfg['notch_freq']
    if notch is not None:
        b, a = signal.iirnotch(float(notch) / (target_fs / 2.0), 30)
        x = signal.filtfilt(b, a, x, axis=-1)
    if str(cfg['norm_method']).lower() == 'car':
        x = x - np.mean(x, axis=1, keepdims=True)
    x = (x / float(cfg['scale'])).astype(np.float32)
    if model == 'cbramod':
        patch_size = 200
        if x.shape[-1] % patch_size:
            raise RuntimeError(f'CBraMod input duration is not divisible by 200: {x.shape}')
        x = x.reshape(len(x), x.shape[1], x.shape[-1] // patch_size, patch_size)
    return x, after_resample


def prepare(profile: str, model: str, dataset: str, variant: str, spec: dict,
            chunk_size: int = 96) -> None:
    if profile == 'source_bridge':
        source_x, source_rows, y, subjects, uids, src_manifest = bridge_info(dataset, variant)
        src_fs = 250
    else:
        source_x, y, order, all_uids, src_manifest = source_info(dataset)
        trials = pd.read_csv(SOURCE_ROOT / dataset / 'trials.csv')
        source_rows = order
        y = y[order]
        subjects = trials.subject.to_numpy(dtype=np.int64)[order] - 1
        uids = [all_uids[int(i)] for i in order]
        src_fs = int(json.loads((SOURCE_ROOT / dataset / 'manifest.json').read_text())[
            'effective_native_fs_hz'])
    cfg = get_profile(profile, model, dataset, spec)
    if len(source_rows) != len(y) or len(y) != len(subjects) or len(y) != len(uids):
        raise RuntimeError(f'{profile}/{dataset}/{model}/{variant}: sample metadata mismatch')
    if profile == 'reference_aligned':
        expected_class = set(range(len(LABELS_REFERENCE[dataset])))
        if set(np.unique(y).tolist()) != expected_class:
            raise RuntimeError(f'{dataset}: rebuilt labels do not cover expected classes')

    output = INPUT_ROOT / profile / dataset / model / variant
    if valid_existing_input(output, profile, dataset, model, variant, cfg, src_manifest):
        print(f'[skip-valid] {profile} {dataset} {model} {variant}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    sample, _ = transform_chunk(source_x[source_rows[:1]], src_fs, cfg, model)
    out_path = output / 'X.npy'
    partial = output / 'X.npy.partial'
    x_out = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float32, shape=(len(y),) + sample.shape[1:])
    after_resample_values = set()
    for start in range(0, len(y), chunk_size):
        stop = min(start + chunk_size, len(y))
        input_chunk = source_x[source_rows[start:stop]]
        if cfg['source_cast'].startswith('float32'):
            input_chunk = np.asarray(input_chunk, dtype=np.float32)
        data, after_resample = transform_chunk(input_chunk, src_fs, cfg, model)
        if data.shape[1:] != sample.shape[1:]:
            raise RuntimeError(f'inconsistent input shape for chunk {start}: {data.shape}')
        x_out[start:stop] = data
        after_resample_values.add(after_resample)
        if start == 0 or stop == len(y) or stop % 768 == 0:
            print(f'[prepare] {profile} {dataset} {model} {variant} '
                  f'{stop}/{len(y)} input_shape={sample.shape[1:]}', flush=True)
    x_out.flush()
    del x_out
    os.replace(partial, out_path)
    np.save(output / 'y.npy', np.asarray(y, dtype=np.int64), allow_pickle=False)
    np.save(output / 'subjects.npy', np.asarray(subjects, dtype=np.int64), allow_pickle=False)
    pd.DataFrame({'trial_uid': uids, 'subject_zero_based': subjects,
                  'label_id': y}).to_csv(output / 'trials.csv', index=False)
    manifest = {
        'profile': profile, 'dataset': dataset, 'model': model, 'variant': variant,
        'source_fs_hz': src_fs, 'source_manifest_or_hashes': src_manifest,
        'resolved_profile': cfg, 'after_resample_sample_lengths': sorted(after_resample_values),
        'preprocessing_order': (
            ['crop_at_native_fs', 'resample_to_model_fs', 'filter', 'notch', 'normalization', 'float32']
            if cfg.get('crop_before_resample') else
            ['resample_to_model_fs', 'trim_or_repeat_pad', 'filter', 'notch', 'normalization', 'float32']
        ),
        'duration_target_samples': int(round(cfg['duration'] * cfg['target_fs'])),
        'input_shape': list(sample.shape[1:]), 'input_dtype': 'float32',
        'class_mapping': LABELS_LEGACY if profile == 'source_bridge' else LABELS_REFERENCE[dataset],
        'trial_count': len(y), 'subjects': sorted(np.unique(subjects).tolist()),
        'files': {},
    }
    for filename in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'):
        p = output / filename
        manifest['files'][filename] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'[prepared] {profile} {dataset} {model} {variant} '
          f'count={len(y)} shape={tuple(sample.shape[1:])}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--include-bridge', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=96)
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    for dataset in args.datasets:
        for model in MODELS:
            prepare('reference_aligned', model, dataset, 'rebuilt_source', spec,
                    chunk_size=args.chunk_size)
    if args.include_bridge:
        for model in MODELS:
            for variant in ('legacy_cache', 'rebuilt_source'):
                prepare('source_bridge', model, 'BNCI2014001-4', variant, spec,
                        chunk_size=args.chunk_size)


if __name__ == '__main__':
    main()
