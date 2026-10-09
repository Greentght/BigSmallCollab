#!/usr/bin/env python
"""Shared input transforms; the former alignment CLI has been retired."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
OLD_DATA = Path('/data1/llx')
SPEC_PATH = ROOT / 'configs/protocols/loso_001.yaml'
SOURCE_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/rebuilt'
INPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/model_inputs'
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


def _require_config_values(context: str, cfg: dict, expected: dict) -> None:
    changed = {key: {'configured': cfg.get(key, value), 'supported': value}
               for key, value in expected.items() if cfg.get(key, value) != value}
    if changed:
        raise ValueError(f'{context}: unsupported current 001 baseline settings: {changed}')


def validate_001_dataset(dataset: str, source_root: Path, input_root: Path,
                         result_root: Path, spec: dict | None = None) -> dict:
    """Reject source/protocol changes that the fixed current 001 pipeline ignores."""
    import config
    if dataset not in ('BNCI2014001', 'BNCI2014001-4'):
        raise ValueError(f'{dataset}: current 001 protocol supports 001 and 001-4 only')
    data_cfg = config.load_dataset_config(dataset)
    classes = ({'left_hand': 0, 'right_hand': 1} if dataset == 'BNCI2014001' else
               {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3})
    _require_config_values(dataset, data_cfg, {
        'num_subjects': 9, 'channels': 22, 'sample_rate': 250,
        'num_classes': len(classes),
    })
    loso = data_cfg['loso']
    _require_config_values(f'{dataset}.loso', loso, {
        'source_directory': str(source_root), 'input_root': str(input_root),
        'result_root': str(result_root), 'native_sampling_rate_hz': 250,
        'selected_session': '0train', 'duration_seconds': 4.0,
        'source_bandpass_hz': [0.1, 75.0], 'current_baseline_id': 'wideband_npy_v3',
        'class_mapping': classes, 'seeds': [0, 1, 2],
        'per_subject_trials': [144 if len(classes) == 2 else 288] * 9,
    })
    if spec is not None:
        _require_config_values('current 001 protocol source', spec['source'], {
            'directory': str(source_root), 'native_fs_hz': 250,
            'bandpass_hz': [0.1, 75], 'array_shape': [5184, 22, 1001],
        })
        _require_config_values('current 001 protocol selection', spec['selection'], {
            'loso_session': '0train',
            'source_window': 'first_1000_native_samples_before_model_resampling',
        })
        _require_config_values('current 001 protocol storage', spec['storage'], {
            'inputs': str(input_root), 'results': str(result_root),
        })
        _require_config_values('current 001 protocol seeds', spec['protocol'], {
            'teacher_seeds': [0, 1, 2], 'student_seeds': [0, 1, 2],
        })
    return loso


def validate_001_model(model: str, dataset: str, cfg: dict) -> None:
    """Check fixed preprocessing/scheduler operations before cache reuse or training."""
    common = {'duration_seconds': 4.0, 'sample_rate': 250, 'samples': 1000}
    if model == 'mirepnet':
        fixed = dict(common, target_fs=250, in_channels=45, skip_preprocess=True,
                     apply_EA=True, ea_scope='per_subject',
                     test_ea_policy='all_unlabeled_held_out_subject_trials',
                     channel_mapping='inverse_distance_to_45_channels',
                     l_freq=8.0, h_freq=30.0, filter_order=4,
                     lr_schedule='epoch_cosine', schedule_update='epoch_end',
                     warmup_epochs=0, min_lr=0.0, class_weights=False,
                     label_smoothing=0.0, dropout=0.5)
    elif model == 'cbramod':
        fixed = dict(common, target_fs=200, in_channels=22, skip_preprocess=True,
                     feature_head='flatten', apply_EA=False, norm_method='car',
                     l_freq=0.3, h_freq=75.0, notch_freq=60.0, scale=1.0,
                     filter_order=4, notch_q=30,
                     lr_schedule='reference_step_table', schedule='step_table',
                     schedule_update='epoch_end_global_step_table')
    elif model in ('ifnet', 'eegnet', 'adfcnn'):
        fixed = dict(common, target_fs=250, in_channels=22, skip_preprocess=False,
                     lr_schedule='epoch_cosine', schedule_update='epoch_end',
                     warmup_epochs=0, min_lr=0.0, label_smoothing=0.0,
                     class_weights=False, apply_EA=False)
        if model == 'ifnet':
            fixed.update(use_filter_bank=True, filter_bank_hz=[[4.0, 16.0], [16.0, 40.0]],
                         filter_order=5, filter_bank_order=5)
        else:
            fixed.update(l_freq=8.0, h_freq=32.0, filter_order=4,
                         preprocessing_stage='cached_input')
    else:
        raise ValueError(f'{model}: unsupported current 001 baseline model')
    _require_config_values(f'{dataset}/{model}.loso', cfg, fixed)
    if cfg.get('filter_phase', 'zero_phase') not in ('zero_phase', 'zero_phase_filtfilt'):
        raise ValueError(f'{dataset}/{model}: only zero-phase filtering is supported')


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
        saved = json.loads(resolve_local_file(manifest_path).read_text())
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
    x = np.load(resolve_local_file(folder / 'X.npy'), mmap_mode='r')
    y = np.load(resolve_local_file(folder / 'y.npy'), mmap_mode='r')
    trials = pd.read_csv(resolve_local_file(folder / 'trials.csv'))
    manifest = json.loads(resolve_local_file(folder / 'manifest.json').read_text())
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
    mapping = pd.read_csv(resolve_local_file(folder / 'legacy_row_mapping.csv'))
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


def get_profile(profile: str, model: str, dataset: str, spec: dict | None = None) -> dict:
    if profile != 'wideband_npy_v3':
        raise ValueError(
            f'{profile} is retired. Prepare current 001/001-4 inputs with '
            'experiments/finetune/prepare_bnci14001_wideband_inputs.py; '
            'prepare current 004/5001 inputs with '
            'experiments/finetune/prepare_loso_source_refresh_004_5001.py.')
    if model != 'cbramod' or dataset not in ('BNCI2014001', 'BNCI2014001-4'):
        raise ValueError('wideband_npy_v3 shared profile supports CBraMod 001/001-4 only')
    import config
    cfg = config.load_model_config(model, dataset, 'loso')
    validate_001_model(model, dataset, cfg)
    # Keep the existing immutable input manifest's numeric representation.
    # Whole-valued duration/cutoffs were exported as integers in this cache.
    def native_number(value):
        number = float(value)
        return int(number) if number.is_integer() else number

    return {
        'optimizer': cfg['optimizer'], 'lr': cfg['lr'],
        'weight_decay': cfg['weight_decay'],
        'batch_size': cfg['batch_size'], 'epochs': cfg['epochs'],
        'dropout': cfg['dropout'], 'label_smoothing': cfg['label_smoothing'],
        'target_fs': int(cfg['target_fs']), 'l_freq': float(cfg['l_freq']),
        'h_freq': native_number(cfg['h_freq']), 'notch_freq': native_number(cfg['notch_freq']),
        'norm_method': cfg['norm_method'], 'apply_EA': cfg['apply_EA'],
        'duration': native_number(cfg['duration_seconds']), 'schedule': 'step_table',
        'schedule_update': cfg['schedule_update'],
        'warmup_epochs': cfg['warmup_epochs'],
        'min_lr': cfg['min_lr'], 'scale': float(cfg['scale']),
        'source_cast': 'float64_broadband_all_session_npy',
        'crop_before_resample': False,
        'class_weights': cfg['class_weights'],
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
        trials = pd.read_csv(resolve_local_file(SOURCE_ROOT / dataset / 'trials.csv'))
        source_rows = order
        y = y[order]
        subjects = trials.subject.to_numpy(dtype=np.int64)[order] - 1
        uids = [all_uids[int(i)] for i in order]
        src_fs = int(json.loads(resolve_local_file(SOURCE_ROOT / dataset / 'manifest.json').read_text())[
            'effective_native_fs_hz'])
    cfg = get_profile(profile, model, dataset, spec)
    if len(source_rows) != len(y) or len(y) != len(subjects) or len(y) != len(uids):
        raise RuntimeError(f'{profile}/{dataset}/{model}/{variant}: sample metadata mismatch')
    if profile == 'reference_aligned':
        expected_class = set(range(len(LABELS_REFERENCE[dataset])))
        if set(np.unique(y).tolist()) != expected_class:
            raise RuntimeError(f'{dataset}: rebuilt labels do not cover expected classes')

    output = require_external_output(INPUT_ROOT / profile / dataset / model / variant)
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
    raise SystemExit(
        'The alignment/source_bridge input CLI is retired. Use '
        'experiments/finetune/prepare_bnci14001_wideband_inputs.py for 001/001-4 '
        'or experiments/finetune/prepare_loso_source_refresh_004_5001.py '
        'for 004/5001. Shared transform_chunk remains available for imports.')


if __name__ == '__main__':
    main()
