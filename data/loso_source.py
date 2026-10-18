"""Canonical source, model-input, prediction, and teacher-target readers for LOSO.

The ordinary within-subject path remains in :mod:`data.eeg_dataset`. This module
is deliberately scoped to current LOSO runs and only reads the external
``artifact-loso`` layout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample

import config
from experiments.storage import LOSO_CACHE_ROOT, resolve_local_file


ARTIFACT_LOSO_ROOT = LOSO_CACHE_ROOT
_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_DATA_CACHE: dict[tuple[str, str | None, str], 'LOSOData'] = {}


@dataclass(frozen=True)
class LOSOData:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    trial_uid: np.ndarray
    sample_uid: np.ndarray
    class_mapping: dict[str, int]
    source_path: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest: dict[str, Any]
    input_manifest_path: Path | None
    input_manifest_sha256: str | None
    input_manifest: dict[str, Any] | None
    identity: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    path = Path(resolve_local_file(path))
    stat = path.stat()
    cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[cache_key] = value
    return value


def _json(path: Path) -> dict[str, Any]:
    with Path(resolve_local_file(path)).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'expected a JSON object in {path}')
    return value


def _verify_files(root: Path, manifest: dict[str, Any], required: tuple[str, ...], context: str) -> None:
    records = manifest.get('files')
    if not isinstance(records, dict):
        raise ValueError(f'{context}: manifest is missing a files mapping')
    for name in required:
        record = records.get(name)
        if not isinstance(record, dict) or 'bytes' not in record or 'sha256' not in record:
            raise ValueError(f'{context}: manifest is missing size/hash for {name}')
        path = Path(resolve_local_file(root / name))
        if not path.is_file():
            raise FileNotFoundError(f'{context}: required file is missing: {path}')
        if path.stat().st_size != int(record['bytes']):
            raise ValueError(f'{context}: byte size mismatch for {path}')
        if sha256_file(path) != str(record['sha256']):
            raise ValueError(f'{context}: SHA-256 mismatch for {path}')


def _canonical_session(dataset: str, configured: str | None) -> str:
    aliases = {
        'BNCI2014001': {'session_T': '0train', '0train': '0train'},
        'BNCI2014001-4': {'session_T': '0train', '0train': '0train'},
        'BNCI2014004': {'session_3': '3test', '3test': '3test'},
        'BNCI2015001': {'session_A': '0A', '0A': '0A'},
    }
    if configured is None:
        configured = {
            'BNCI2014001': '0train', 'BNCI2014001-4': '0train',
            'BNCI2014004': '3test', 'BNCI2015001': '0A',
        }.get(dataset)
    if configured is None:
        raise ValueError(f'{dataset}: no LOSO session is configured')
    return aliases.get(dataset, {}).get(str(configured), str(configured))


def _task_mapping(dataset: str, loso_cfg: dict[str, Any]) -> dict[str, int]:
    mapping = loso_cfg.get('class_mapping')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f'{dataset}: LOSO class_mapping is missing')
    return {str(name): int(label) for name, label in mapping.items()}


def _load_source_index(dataset: str, loso_cfg: dict[str, Any], source_dir: Path):
    manifest_path = Path(resolve_local_file(source_dir / 'manifest.json'))
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{dataset}: wide LOSO source manifest is missing: {manifest_path}')
    manifest = _json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if str(manifest.get('source_dataset', manifest.get('dataset', dataset))) not in {
            dataset, 'BNCI2014001' if dataset == 'BNCI2014001-4' else dataset}:
        raise ValueError(f'{dataset}: source manifest names another dataset')
    _verify_files(source_dir, manifest, ('X.npy', 'labels.npy', 'trials.csv'),
                  f'{dataset} wide source')

    labels_path = Path(resolve_local_file(source_dir / 'labels.npy'))
    labels = np.asarray(np.load(labels_path, allow_pickle=False)).astype(str)
    trials = pd.read_csv(resolve_local_file(source_dir / 'trials.csv'))
    required_columns = {'subject', 'session', 'class_name', 'trial_uid'}
    if not required_columns.issubset(trials.columns):
        raise ValueError(f'{dataset}: source trials.csv lacks {sorted(required_columns - set(trials.columns))}')
    if len(labels) != len(trials):
        raise ValueError(f'{dataset}: labels.npy and trials.csv row counts differ')
    if not np.array_equal(labels, trials['class_name'].astype(str).to_numpy()):
        raise ValueError(f'{dataset}: labels.npy and trials.csv class names differ')
    if trials['trial_uid'].astype(str).duplicated().any():
        raise ValueError(f'{dataset}: source trial_uid values are not unique')

    source_class_map = manifest.get('class_map')
    if not isinstance(source_class_map, dict):
        source_class_map = manifest.get('class_mapping')
    if not isinstance(source_class_map, dict):
        raise ValueError(f'{dataset}: source manifest has no source class mapping')
    task_map = _task_mapping(dataset, loso_cfg)
    for name, label in task_map.items():
        if name not in source_class_map or int(source_class_map[name]) != label:
            raise ValueError(f'{dataset}: configured class mapping for {name!r} disagrees with source manifest')
    if 'label_id' in trials:
        source_ids = np.asarray([int(source_class_map[name]) for name in labels], dtype=np.int64)
        if not np.array_equal(source_ids, trials['label_id'].to_numpy(dtype=np.int64)):
            raise ValueError(f'{dataset}: trials.csv label_id disagrees with source class mapping')

    session = _canonical_session(dataset, loso_cfg.get('selected_session'))
    keep = trials['session'].astype(str).eq(session).to_numpy()
    keep &= trials['class_name'].astype(str).isin(task_map).to_numpy()
    selected = trials.loc[keep].copy()
    source_rows = np.flatnonzero(keep)
    if selected.empty:
        raise ValueError(f'{dataset}: no trials match selected LOSO session {session!r} and task class map')
    selected['label_id'] = np.asarray(
        [task_map[str(name)] for name in selected['class_name']], dtype=np.int64)

    # The old metadata row mapping records the historical within-task ordering.
    # Use it when present so cache arrays and saved teacher targets keep their
    # established identity even if the wide source was exported in another order.
    mapping_path = source_dir / 'legacy_row_mapping.csv'
    if mapping_path.is_file():
        row_map = pd.read_csv(resolve_local_file(mapping_path))
        row_column = next((name for name in ('source_row', 'rebuilt_row')
                           if name in row_map.columns), None)
        if row_column is None or 'legacy_raw_row' not in row_map.columns:
            raise ValueError(f'{dataset}: legacy_row_mapping.csv has an unsupported schema')
        row_map = row_map[[row_column, 'legacy_raw_row']].copy()
        if row_map[row_column].duplicated().any():
            raise ValueError(f'{dataset}: legacy source-row map is not one-to-one')
        legacy_rows = row_map.set_index(row_column)['legacy_raw_row'].reindex(source_rows)
        if legacy_rows.isna().any():
            raise ValueError(f'{dataset}: selected source rows are missing from the legacy row map')
        order = np.argsort(legacy_rows.to_numpy(dtype=np.int64), kind='stable')
        selected = selected.iloc[order].reset_index(drop=True)
        source_rows = source_rows[order]
    elif 'source_row' in selected.columns:
        order = np.argsort(selected['source_row'].to_numpy(dtype=np.int64), kind='stable')
        selected = selected.iloc[order].reset_index(drop=True)
        source_rows = source_rows[order]

    subject_column = ('subject_zero_based' if 'subject_zero_based' in selected.columns
                      else 'subject')
    subject_values = selected[subject_column].to_numpy(dtype=np.int64)
    # Source metadata stores one-based subject numbers unless it explicitly
    # exposes a zero-based subject column.
    n_subjects = int(config.load_dataset_config(dataset)['num_subjects'])
    if subject_column == 'subject_zero_based' or int(subject_values.min()) == 0:
        subjects = subject_values
    else:
        subjects = subject_values - 1
    if np.any(subjects < 0) or np.any(subjects >= n_subjects):
        raise ValueError(f'{dataset}: source subject IDs are outside [0, {n_subjects})')
    if selected['trial_uid'].astype(str).duplicated().any():
        raise ValueError(f'{dataset}: selected trial_uid values are not unique')
    labels_y = selected['label_id'].to_numpy(dtype=np.int64)
    trial_uids = selected['trial_uid'].astype(str).to_numpy()
    # Stable local index within the complete selected-task subject block.
    counts: dict[int, int] = {}
    sample_uids = np.empty((len(subjects), 2), dtype=np.int64)
    for row, subject in enumerate(subjects.tolist()):
        local = counts.get(int(subject), 0)
        sample_uids[row] = (int(subject), local)
        counts[int(subject)] = local + 1
    return {
        'manifest_path': manifest_path,
        'manifest_sha256': manifest_sha,
        'manifest': manifest,
        'source_dir': source_dir,
        'source_rows': source_rows,
        'trials': selected,
        'subjects': subjects.astype(np.int64),
        'y': labels_y,
        'trial_uid': trial_uids,
        'sample_uid': sample_uids,
        'class_mapping': task_map,
        'session': session,
    }


def _canonical_window(x: np.ndarray, fs_hz: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if fs_hz == 250:
        x = x[..., :1000]
    elif fs_hz == 512:
        x = scipy_resample(x, int(round(x.shape[-1] * 250 / fs_hz)), axis=-1)
        x = x[..., :1000]
    else:
        raise ValueError(f'unsupported native sampling rate: {fs_hz}')
    if x.shape[-1] != 1000:
        raise ValueError(f'expected a 4 s/250 Hz window, got {x.shape}')
    return x.astype(np.float32)


def _legacy_alexmi(model: str | None = None) -> LOSOData:
    import config as project_config
    from types import SimpleNamespace
    from data.eeg_dataset import EEGDataset

    dcfg = project_config.load_dataset_config('AlexMI')
    xs, ys, subject_ids, uids, local_uids = [], [], [], [], []
    for subject in range(int(dcfg['num_subjects'])):
        block = EEGDataset(args=SimpleNamespace(dataset_name='AlexMI', sub=[subject]))
        x = np.asarray(block.X, dtype=np.float32)
        y = np.asarray(block.y, dtype=np.int64)
        xs.append(x)
        ys.append(y)
        subject_ids.append(np.full(len(y), subject, dtype=np.int64))
        local = np.column_stack((np.full(len(y), subject, dtype=np.int64),
                                 np.arange(len(y), dtype=np.int64)))
        local_uids.append(local)
        uids.extend(f'AlexMI|subject-{subject:02d}|trial-{index:04d}'
                    for index in range(len(y)))
    all_y = np.concatenate(ys)
    policy_identity = _input_policy_identity('AlexMI', model)
    policy_sha = hashlib.sha256(policy_identity.encode()).hexdigest()
    data_obj = LOSOData(
        x=np.concatenate(xs), y=all_y, subjects=np.concatenate(subject_ids),
        trial_uid=np.asarray(uids, dtype=str),
        sample_uid=np.concatenate(local_uids),
        class_mapping={'feet': 0, 'right_hand': 1},
        source_path=Path('EEGDataset:AlexMI'), source_manifest_path=Path(''),
        source_manifest_sha256='', source_manifest={},
        input_manifest_path=None, input_manifest_sha256=None, input_manifest=None,
        identity={'dataset': 'AlexMI', 'model': model,
                  'source': 'legacy EEGDataset',
                  'input_policy_sha256': policy_sha},
    )
    from data import loso_alexmi

    source_identity = loso_alexmi.source_identity(data_obj)
    identity = dict(data_obj.identity)
    # AlexMI has no synthetic source manifest. Its identity is derived from
    # the preserved source files and the fixed legacy loader recipe.
    identity.update({
        'source_path': source_identity['source_root'],
        'source_manifest_sha256': source_identity['source_identity_sha256'],
        'source_files_sha256': source_identity['source_files_sha256'],
        'source_identity_sha256': source_identity['source_identity_sha256'],
        'source_files': source_identity['source_files'],
        'loader_recipe': source_identity['loader_recipe'],
        'source_variant': 'legacy_alexmi_verified',
    })
    return replace(
        data_obj,
        source_path=Path(source_identity['source_root']),
        source_manifest_path=Path(''),
        source_manifest_sha256=source_identity['source_identity_sha256'],
        source_manifest=source_identity['source_files'],
        identity=identity,
    )


def _input_policy_identity(dataset: str, model: str | None) -> str:
    """Fingerprint only input-shaping/filtering policy, not training knobs."""
    if dataset == 'AlexMI':
        # The preserved AlexMI loader and its prepared targets share one fixed
        # historical preprocessing recipe, independent of the model config.
        return 'legacy_alexmi'
    if model is None:
        return 'canonical_wide_source'
    if dataset in {'BNCI2014004', 'BNCI2015001'}:
        from experiments.finetune.prepare_loso_source_refresh_004_5001 import current_input_policy
        _, pipeline = current_input_policy(dataset, model)
        signature = {'pipeline': pipeline, 'model': model,
                     'native_sampling_rate_hz': config.load_loso_dataset_config(dataset)[
                         'native_sampling_rate_hz']}
    elif dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        cfg = config.load_model_config(model, dataset, 'loso')
        keys = ('sample_rate', 'target_fs', 'samples', 'duration_seconds', 'in_channels',
                'skip_preprocess', 'l_freq', 'h_freq', 'filter_order', 'filter_phase',
                'notch_freq', 'notch_q', 'norm_method', 'apply_EA', 'ea_scope',
                'channel_mapping', 'use_filter_bank', 'filter_bank_hz',
                'filter_bank_order', 'preprocessing_stage')
        signature = {'model': model, **{key: cfg.get(key) for key in keys if key in cfg}}
    else:
        return f'legacy_{dataset}'
    return json.dumps(signature, sort_keys=True, separators=(',', ':'))


def _validate_cached_input_policy(dataset: str, model: str,
                                  manifest: dict[str, Any], x_shape: tuple[int, ...]) -> None:
    """Reject caches whose declared fixed preprocessing differs from this run."""
    if dataset in {'BNCI2014004', 'BNCI2015001'}:
        from experiments.finetune.prepare_loso_source_refresh_004_5001 import validate_input_identity
        validate_input_identity(dataset, model, manifest,
                                config.load_model_config(model, dataset, 'loso'))
        return
    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        from experiments.finetune.prepare_loso_alignment_inputs import validate_001_model
        cfg = config.load_model_config(model, dataset, 'loso')
        validate_001_model(model, dataset, cfg)
        if manifest.get('selected_session') != '0train':
            raise ValueError(f'{dataset}/{model}: cached selected_session is not 0train')
        if manifest.get('class_mapping') != _task_mapping(
                dataset, config.load_loso_dataset_config(dataset)):
            raise ValueError(f'{dataset}/{model}: cached class mapping differs from current LOSO task')
        expected_channels = 45 if model == 'mirepnet' else 22
        expected_shape = ((expected_channels, 4, 200) if model == 'cbramod'
                          else (expected_channels, 1000))
        if (tuple(x_shape[1:]) != expected_shape
                or tuple(manifest.get('input_shape', ())) != expected_shape
                or int(manifest.get('trial_count', -1)) != int(x_shape[0])):
            raise ValueError(f'{dataset}/{model}: cached input shape differs from fixed 001 recipe')
        pipeline = manifest.get('preprocessing_order')
        expected_prefix = {
            'cbramod': ['select_0train', 'select_task_classes', 'first_1000_native_samples',
                        'MNE_resample_250_to_200', 'bandpass_0.3_75', 'notch_60_Q30',
                        'CAR', 'patchify_22_4_200', 'float32'],
            'mirepnet': ['select_0train', 'select_task_classes',
                         'first_1000_native_samples',
                         'epoch_Butterworth_8_30_order4_filtfilt', 'per_subject_EA',
                         'inverse_distance_interpolation_45_channels', 'float32']
        }
        if model in expected_prefix and pipeline != expected_prefix[model]:
            raise ValueError(f'{dataset}/{model}: cached preprocessing differs from fixed 001 recipe')


def load_dataset(dataset: str, model: str | None = None,
                 cache_root: str | Path | None = None) -> LOSOData:
    """Load one source-aligned LOSO dataset, preferring a verified model cache.

    Returns arrays in the selected source trial order plus both physical string
    trial UIDs and stable ``(subject_zero_based, within_task_index)`` sample UIDs.
    ``cache_root`` is the root containing ``model_inputs/`` and
    ``teacher_targets/``; by default it is the canonical ``artifact-loso`` tree.
    """
    dataset = str(dataset)
    model = None if model is None else str(model).lower()
    root = Path(cache_root) if cache_root is not None else ARTIFACT_LOSO_ROOT
    if dataset == 'AlexMI':
        return _legacy_alexmi(model)
    loso_cfg = config.load_loso_dataset_config(dataset)
    source_dir = Path(loso_cfg['source_directory'])
    policy_identity = _input_policy_identity(dataset, model)
    policy_sha = hashlib.sha256(policy_identity.encode()).hexdigest()
    cache_key = (dataset, model, f'{root.resolve(strict=False)}|{policy_identity}')
    if cache_key in _DATA_CACHE:
        cached = _DATA_CACHE[cache_key]
        source_path_now = Path(resolve_local_file(source_dir / 'manifest.json'))
        source_ok = (cached.source_manifest_path == source_path_now
                     and cached.source_manifest_sha256 == sha256_file(source_path_now))
        if source_ok:
            _verify_files(cached.source_path, cached.source_manifest,
                          ('X.npy', 'labels.npy', 'trials.csv'),
                          f'{dataset} wide source')
        input_ok = (cached.input_manifest_path is None
                    or cached.input_manifest_sha256 == sha256_file(cached.input_manifest_path))
        if input_ok and cached.input_manifest_path is not None:
            _verify_files(cached.input_manifest_path.parent, cached.input_manifest,
                          ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'),
                          f'{dataset}/{model} model input')
            _validate_cached_input_policy(dataset, str(model), cached.input_manifest,
                                          tuple(cached.x.shape))
        if source_ok and input_ok:
            return cached
        _DATA_CACHE.pop(cache_key, None)

    source_index = _load_source_index(dataset, loso_cfg, source_dir)
    task_map = source_index['class_mapping']
    model_dir = root / 'model_inputs' / dataset / model if model else None
    input_manifest_path = model_dir / 'manifest.json' if model_dir else None
    input_manifest = None
    input_manifest_sha = None
    if model_dir is not None and model_dir.exists() and not input_manifest_path.is_file():
        raise FileNotFoundError(f'{dataset}/{model}: incomplete canonical model-input cache at {model_dir}')
    if model_dir is not None and input_manifest_path.is_file():
        input_manifest = _json(input_manifest_path)
        input_manifest_sha = sha256_file(input_manifest_path)
        context = f'{dataset}/{model} model input'
        if input_manifest.get('dataset') != dataset or input_manifest.get('model') != model:
            raise ValueError(f'{context}: manifest dataset/model identity mismatch')
        recorded_source_sha = input_manifest.get('source_manifest_sha256')
        if recorded_source_sha != source_index['manifest_sha256']:
            raise ValueError(f'{context}: source_manifest_sha256 does not match the current wide source')
        recorded_mapping = {str(k): int(v) for k, v in
                            input_manifest.get('class_mapping', {}).items()}
        if recorded_mapping != task_map:
            raise ValueError(f'{context}: class_mapping does not match the current LOSO task')
        _verify_files(model_dir, input_manifest,
                      ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'), context)
        x = np.load(resolve_local_file(model_dir / 'X.npy'), mmap_mode='r', allow_pickle=False)
        _validate_cached_input_policy(dataset, str(model), input_manifest, tuple(x.shape))
        y = np.asarray(np.load(resolve_local_file(model_dir / 'y.npy'), allow_pickle=False),
                       dtype=np.int64)
        subjects = np.asarray(np.load(resolve_local_file(model_dir / 'subjects.npy'),
                                      allow_pickle=False), dtype=np.int64)
        cached_trials = pd.read_csv(resolve_local_file(model_dir / 'trials.csv'))
        required = {'trial_uid', 'label_id'}
        if not required.issubset(cached_trials.columns):
            raise ValueError(f'{context}: trials.csv lacks {sorted(required - set(cached_trials.columns))}')
        trial_uid = cached_trials['trial_uid'].astype(str).to_numpy()
        if len(set(trial_uid.tolist())) != len(trial_uid):
            raise ValueError(f'{context}: duplicate trial_uid values')
        if len(x) != len(y) or len(y) != len(subjects) or len(y) != len(trial_uid):
            raise ValueError(f'{context}: array/trials row counts differ')
        source_by_uid = {uid: i for i, uid in enumerate(source_index['trial_uid'].tolist())}
        if set(trial_uid.tolist()) != set(source_by_uid):
            raise ValueError(f'{context}: model-input trial_uid set differs from the current wide source')
        expected_y = np.asarray([source_index['y'][source_by_uid[uid]] for uid in trial_uid],
                                dtype=np.int64)
        expected_subjects = np.asarray(
            [source_index['subjects'][source_by_uid[uid]] for uid in trial_uid], dtype=np.int64)
        cached_y = cached_trials['label_id'].to_numpy(dtype=np.int64)
        if not np.array_equal(y, cached_y) or not np.array_equal(y, expected_y):
            raise ValueError(f'{context}: y, trials.csv labels, and current source labels differ')
        if not np.array_equal(subjects, expected_subjects):
            raise ValueError(f'{context}: subjects.npy does not match trial_uid subject identities')
        sample_by_uid = {uid: source_index['sample_uid'][i]
                         for i, uid in enumerate(source_index['trial_uid'].tolist())}
        sample_uid = np.asarray([sample_by_uid[uid] for uid in trial_uid], dtype=np.int64)
        input_path = model_dir
        input_id = {'model': model, 'path': str(model_dir),
                    'manifest_sha256': input_manifest_sha,
                    'source_manifest_sha256': source_index['manifest_sha256']}
    else:
        # No fallback to a legacy model-input path. Only the configured wide
        # source is read when a canonical model cache has not been prepared.
        _verify_files(source_dir, source_index['manifest'],
                      ('X.npy', 'labels.npy', 'trials.csv'), f'{dataset} wide source')
        source_x = np.load(resolve_local_file(source_dir / 'X.npy'), mmap_mode='r',
                           allow_pickle=False)
        fs_hz = int(loso_cfg['native_sampling_rate_hz'])
        pieces = []
        rows = source_index['source_rows']
        for start in range(0, len(rows), 128):
            stop = min(start + 128, len(rows))
            pieces.append(_canonical_window(source_x[rows[start:stop]], fs_hz))
        x = np.concatenate(pieces, axis=0)
        if model in {'eegnet', 'adfcnn'}:
            model_cfg = config.load_model_config(model, dataset, 'loso')
            from data.preproc import bandpass
            x = bandpass(x.astype(np.float64), 250,
                         float(model_cfg.get('l_freq', 8.0)),
                         float(model_cfg.get('h_freq', 32.0))).astype(np.float32)
        y = source_index['y']
        subjects = source_index['subjects']
        trial_uid = source_index['trial_uid']
        sample_uid = source_index['sample_uid']
        input_path = None
        input_id = {'model': model, 'path': None, 'manifest_sha256': None,
                    'source_manifest_sha256': source_index['manifest_sha256'],
                    'fallback': 'wide_source',
                    'fallback_preprocessing': (
                        (f"{float(config.load_model_config(model, dataset, 'loso').get('l_freq', 8.0)):g}_"
                         f"{float(config.load_model_config(model, dataset, 'loso').get('h_freq', 32.0)):g}_hz_once")
                        if model in {'eegnet', 'adfcnn'}
                        else 'ifnet_adapter_filterbank' if model == 'ifnet'
                        else 'canonical_wide_source')}
        input_manifest_path = None
        input_manifest = None
        input_manifest_sha = None

    data_obj = LOSOData(
        x=x, y=np.asarray(y, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
        trial_uid=np.asarray(trial_uid, dtype=str),
        sample_uid=np.asarray(sample_uid, dtype=np.int64),
        class_mapping=task_map, source_path=source_index['source_dir'],
        source_manifest_path=source_index['manifest_path'],
        source_manifest_sha256=source_index['manifest_sha256'],
        source_manifest=source_index['manifest'],
        input_manifest_path=input_manifest_path,
        input_manifest_sha256=input_manifest_sha,
        input_manifest=input_manifest,
        identity={'dataset': dataset, 'model': model,
                  'source_path': str(source_index['source_dir']),
                  'source_manifest_sha256': source_index['manifest_sha256'],
                  'source_variant': source_index['manifest'].get('source_variant'),
                  'input_policy_sha256': policy_sha,
                  'input': input_id},
    )
    _DATA_CACHE[cache_key] = data_obj
    return data_obj


def load_teacher_targets(dataset: str, teacher: str, seed: int, fold: int,
                         student_data: LOSOData, cache_root: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly align one canonical teacher training-target file."""
    if dataset == 'AlexMI':
        from data import loso_alexmi

        teacher_data = load_dataset(dataset, teacher, cache_root=cache_root)
        source_sha = getattr(student_data, 'source_manifest_sha256', None)
        if not source_sha or source_sha != getattr(teacher_data, 'source_manifest_sha256', None):
            raise ValueError('AlexMI student and teacher source identities differ')
        student_mapping = {str(k): int(v) for k, v in student_data.class_mapping.items()}
        teacher_mapping = {str(k): int(v) for k, v in teacher_data.class_mapping.items()}
        if student_mapping != teacher_mapping:
            raise ValueError('AlexMI student and teacher class mappings differ')

        def uid_labels_and_samples(data_obj: LOSOData) -> dict[str, tuple[int, tuple[int, int]]]:
            uids = np.asarray(data_obj.trial_uid).astype(str).reshape(-1)
            labels = np.asarray(data_obj.y, dtype=np.int64).reshape(-1)
            sample_uids = np.asarray(data_obj.sample_uid, dtype=np.int64)
            if (len(uids) != len(labels) or sample_uids.shape != (len(uids), 2)
                    or len(set(uids.tolist())) != len(uids)):
                raise ValueError('AlexMI student/teacher source view has invalid or duplicate UIDs')
            return {
                uid: (int(label), tuple(map(int, sample_uid.tolist())))
                for uid, label, sample_uid in zip(uids, labels, sample_uids)
            }

        student_uid_map = uid_labels_and_samples(student_data)
        teacher_uid_map = uid_labels_and_samples(teacher_data)
        if student_uid_map != teacher_uid_map:
            raise ValueError('AlexMI student and teacher trial_uid→label/sample_uid mappings differ')

        targets = loso_alexmi.load_teacher_targets(teacher, seed, fold, student_data)
        # The target arrays are reordered against the student's canonical view;
        # their input-policy provenance belongs to the selected teacher model.
        targets['_input_manifest_sha256'] = teacher_data.input_manifest_sha256
        targets['_input_policy_sha256'] = teacher_data.identity.get('input_policy_sha256')
        targets['_source_manifest_sha256'] = student_data.source_manifest_sha256
        return targets
    root = Path(cache_root) if cache_root is not None else ARTIFACT_LOSO_ROOT
    target_dir = root / 'teacher_targets' / dataset / teacher / f'seed_{int(seed)}' / f'subject_{int(fold) + 1:02d}'
    target_path = target_dir / 'targets.npz'
    manifest_path = target_dir / 'manifest.json'
    if not target_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f'missing canonical teacher targets for dataset={dataset} teacher={teacher} '
            f'seed={seed} fold={fold}: expected {target_path} and {manifest_path}')
    manifest = _json(manifest_path)
    input_data = load_dataset(dataset, teacher, cache_root=root)
    expected = {
        'dataset': dataset,
        'teacher': teacher,
        'seed': int(seed),
        'fold_zero_based': int(fold),
        'test_subject': int(fold) + 1,
        'source_manifest_sha256': student_data.source_manifest_sha256,
        'input_manifest_sha256': input_data.input_manifest_sha256,
        'class_mapping': student_data.class_mapping,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == 'class_mapping' and isinstance(actual, dict):
            actual = {str(k): int(v) for k, v in actual.items()}
        if actual != value:
            raise ValueError(f'{target_path}: manifest {key}={actual!r}, expected {value!r}')
    baseline_manifest_path = _baseline_teacher_manifest(dataset, teacher, seed, fold)
    baseline_manifest = _json(baseline_manifest_path)
    teacher_manifest_sha = manifest.get('teacher_manifest_sha256')
    if not teacher_manifest_sha or sha256_file(baseline_manifest_path) != str(teacher_manifest_sha):
        raise ValueError(f'{target_path}: teacher_manifest_sha256 does not match current same-fold baseline')
    _validate_baseline_manifest(baseline_manifest, baseline_manifest_path,
                                input_data, dataset, teacher, seed, fold,
                                check_prediction_hash=False)
    checkpoint_sha = manifest.get('teacher_checkpoint_sha256')
    baseline_checkpoint_sha = baseline_manifest.get('final_model_sha256')
    if not checkpoint_sha or checkpoint_sha != baseline_checkpoint_sha:
        raise ValueError(f'{target_path}: teacher checkpoint SHA differs from the current baseline')
    target_config = manifest.get('model_config')
    current_config = validated_teacher_model_config(
        baseline_manifest, baseline_manifest_path, input_data,
        dataset, teacher, seed, fold)
    if target_config is not None and not _equivalent_current_model_config(
            teacher, dataset, target_config, current_config,
            source_channels=input_data.x.shape[1]):
        raise ValueError(f'{manifest_path}: teacher target model config differs from current LOSO config')
    if manifest.get('input_X_sha256') != input_data.input_manifest.get('files', {}).get('X.npy', {}).get('sha256'):
        raise ValueError(f'{target_path}: input_X_sha256 differs from current teacher input cache')
    target_size = target_path.stat().st_size
    recorded_size = manifest.get('cache_bytes')
    if recorded_size is not None and int(recorded_size) != target_size:
        raise ValueError(f'{target_path}: target byte size differs from manifest')
    target_sha = manifest.get('cache_file_sha256')
    if not target_sha or sha256_file(target_path) != str(target_sha):
        raise ValueError(f'{target_path}: target SHA-256 differs from manifest')

    with np.load(resolve_local_file(target_path), allow_pickle=True) as archive:
        required = {'logits', 'feats', 'y', 'trial_uid'}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f'{target_path}: missing target arrays {sorted(missing)}')
        logits = np.asarray(archive['logits'])
        feats = np.asarray(archive['feats'])
        y = np.asarray(archive['y'], dtype=np.int64)
        raw_target_uid = np.asarray(archive['trial_uid'])
        target_uid = raw_target_uid.astype(str)
    if len(set(target_uid.tolist())) != len(target_uid):
        raise ValueError(f'{target_path}: trial_uid values are not unique')
    saved_uids = manifest.get('train_uids')
    if saved_uids is None or len(set(map(str, saved_uids))) != len(saved_uids):
        raise ValueError(f'{manifest_path}: missing or duplicate train_uids')
    if set(map(str, saved_uids)) != set(target_uid.tolist()):
        raise ValueError(f'{manifest_path}: train_uids and targets.npz trial_uid set differ')
    if len(logits) != len(target_uid) or len(feats) != len(target_uid) or len(y) != len(target_uid):
        raise ValueError(f'{target_path}: target array row counts differ')
    if not np.isfinite(logits).all() or not np.isfinite(feats).all():
        raise ValueError(f'{target_path}: target logits/features contain NaN or Inf')

    source_by_uid = {uid: i for i, uid in enumerate(student_data.trial_uid.tolist())}
    expected_train_uids = {uid for uid, subject in zip(student_data.trial_uid.tolist(),
                                                       student_data.subjects.tolist())
                           if int(subject) != int(fold)}
    if set(target_uid.tolist()) != expected_train_uids:
        raise ValueError(f'{target_path}: targets do not contain exactly all current training trial UIDs')
    if set(target_uid.tolist()) != {uid for uid, subject in
                                    zip(input_data.trial_uid.tolist(), input_data.subjects.tolist())
                                    if int(subject) != int(fold)}:
        raise ValueError(f'{target_path}: teacher-input train UIDs differ from target UIDs')
    expected_y = np.asarray([student_data.y[source_by_uid[uid]] for uid in target_uid], dtype=np.int64)
    if not np.array_equal(y, expected_y):
        raise ValueError(f'{target_path}: target y differs from current wide-source labels by trial_uid')
    if 'train_labels_sha256' in manifest:
        arr = np.asarray(y, dtype=np.int64)
        if _array_hash(arr) != manifest['train_labels_sha256']:
            raise ValueError(f'{target_path}: y hash differs from target manifest')
    if 'train_uid_sha256' in manifest:
        uid_array = np.asarray(target_uid, dtype=raw_target_uid.dtype)
        if _array_hash(uid_array) != manifest['train_uid_sha256']:
            raise ValueError(f'{target_path}: ordered trial_uid hash differs from target manifest')
    if 'train_subjects' in manifest:
        train_subjects = {int(value) for value in manifest['train_subjects']}
        expected_subjects = set(range(1, int(config.load_dataset_config(dataset)['num_subjects']) + 1))
        expected_subjects.remove(int(fold) + 1)
        if train_subjects not in (expected_subjects,
                                  {value - 1 for value in expected_subjects}):
            raise ValueError(f'{target_path}: train_subjects do not match the held-out fold')
    sample_by_uid = {uid: student_data.sample_uid[source_by_uid[uid]]
                     for uid in student_data.trial_uid.tolist()}
    target_sample_uid = np.asarray([sample_by_uid[uid] for uid in target_uid], dtype=np.int64)
    return {
        'logits': logits, 'feats': feats, 'y': y,
        'sample_uid': target_sample_uid, 'trial_uid': target_uid,
        'split_policy': 'loso', '_manifest': manifest,
        '_manifest_path': str(manifest_path), '_artifact_path': str(target_path),
        '_manifest_sha256': sha256_file(manifest_path),
        '_artifact_sha256': str(target_sha),
        '_input_manifest_sha256': input_data.input_manifest_sha256,
        '_input_policy_sha256': input_data.identity.get('input_policy_sha256'),
        '_source_manifest_sha256': student_data.source_manifest_sha256,
    }


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def validated_teacher_model_config(manifest: dict[str, Any], manifest_path: Path,
                                   input_data: LOSOData, dataset: str,
                                   teacher: str, seed: int, fold: int) -> dict[str, Any]:
    """Resolve a completed teacher recipe without weakening identity checks.

    The 001 wideband runner saved its effective training recipe separately from
    the hashed input recipe. Reconstruct that runner's recipe from the current
    formal configuration, and require an exact match before using its weights.
    """
    _validate_baseline_manifest(manifest, manifest_path, input_data,
                                dataset, teacher, seed, fold,
                                check_prediction_hash=False)
    current = config.load_model_config(teacher, dataset, 'loso')
    recorded = manifest.get('model_config')
    if isinstance(recorded, dict):
        if not _equivalent_current_model_config(
                teacher, dataset, recorded, current,
                source_channels=input_data.x.shape[1]):
            raise ValueError(f'{manifest_path}: teacher model config differs from current LOSO config')
        return current
    if (dataset not in {'BNCI2014001', 'BNCI2014001-4'}
            or teacher not in {'cbramod', 'mirepnet'}
            or manifest.get('profile') != 'wideband_npy_v3'
            or not isinstance(manifest.get('resolved_training_config'), dict)):
        raise ValueError(f'{manifest_path}: unrecognized teacher model configuration schema')
    if list(manifest.get('input_shape', ())) != list(input_data.x.shape[1:]):
        raise ValueError(f'{manifest_path}: teacher input shape differs from current cache')
    input_manifest = input_data.input_manifest or {}
    if (manifest.get('input_manifest_sha256') != input_data.input_manifest_sha256
            or manifest.get('input_files') != input_manifest.get('files')):
        raise ValueError(f'{manifest_path}: teacher input files differ from current cache')
    _validate_cached_input_policy(dataset, teacher, input_manifest,
                                  tuple(input_data.x.shape))
    if teacher == 'mirepnet' and (
            int(current.get('emb_size', 256)) != 256
            or int(current.get('depth', 6)) != 6):
        raise ValueError(f'{manifest_path}: current MIRepNet architecture differs from the fixed 001 teacher')
    # Importing the recipe builder does not construct or train a model.
    import yaml
    from experiments.finetune import run_loso_config_alignment as reference

    spec = yaml.safe_load(Path(reference.SPEC).read_text())
    expected = reference.profile_config(reference.WIDEBAND_PROFILE,
                                         teacher, dataset, spec)
    if manifest['resolved_training_config'] != expected:
        raise ValueError(f'{manifest_path}: teacher resolved training recipe differs from current LOSO config')
    return current


def _equivalent_current_model_config(model: str, dataset: str,
                                    recorded: dict[str, Any] | None,
                                    current: dict[str, Any],
                                    source_channels: int | None = None) -> bool:
    if not isinstance(recorded, dict):
        return False
    saved = dict(recorded)
    active = dict(current)
    if saved.get('dataset_name') == dataset:
        saved.pop('dataset_name')
    if active.get('dataset_name') == dataset:
        active.pop('dataset_name')
    if source_channels is not None and model == 'cbramod':
        saved.setdefault('in_channels', int(source_channels))
        active.setdefault('in_channels', int(source_channels))
    normalize = config.normalize_loso_model_config
    return normalize(model, saved) == normalize(model, active)


def _baseline_teacher_manifest(dataset: str, teacher: str, seed: int,
                               fold: int) -> Path:
    result_root = Path(config.load_loso_dataset_config(dataset)['result_root'])
    subject = f'subject_{int(fold) + 1:02d}'
    seed_part = f'seed_{int(seed)}'
    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        candidate = result_root / dataset / teacher / seed_part / subject / 'result.json'
        if candidate.is_file():
            return candidate
    preferred = result_root / dataset / teacher / 'source_refreshed_loso_v1' / seed_part / subject / 'manifest.json'
    if preferred.is_file():
        return preferred
    model_root = result_root / dataset / teacher
    if model_root.is_dir():
        candidates = sorted(model_root.rglob('manifest.json'))
        for path in candidates:
            parts = path.parts
            if subject in parts and seed_part in parts:
                return path
            # Some 001 baseline records use subject/seed nesting in the
            # opposite order while keeping a per-fold manifest.
            if (f'S{int(fold) + 1}' in path.as_posix()
                    and (seed_part in parts or f'seed{int(seed)}' in parts)):
                return path
    raise FileNotFoundError(
        f'missing current same-fold teacher baseline manifest for '
        f'{dataset}/{teacher} seed={seed} fold={fold} under {model_root}')


def _prediction_candidates(root: Path, dataset: str, model: str, seed: int,
                           fold: int, split: str) -> list[Path]:
    base = root / 'predictions' / dataset / model / f'seed_{int(seed)}' / f'subject_{int(fold) + 1:02d}'
    return [base / f'{split}.npz', base / 'predictions.npz']


def load_cached_prediction(dataset: str, model: str, seed: int, fold: int,
                           split: str, data_obj: LOSOData,
                           cache_root: str | Path | None = None) -> dict[str, Any]:
    """Read a future standard prediction cache under artifact-loso/predictions."""
    if dataset == 'AlexMI':
        from data import loso_alexmi

        return loso_alexmi.load_baseline_prediction(
            model, seed, fold, split, data_obj)
    root = Path(cache_root) if cache_root is not None else ARTIFACT_LOSO_ROOT
    candidates = _prediction_candidates(root, dataset, model, seed, fold, split)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f'missing current-model source-{split} prediction for {dataset}/{model} '
            f'seed={seed} fold={fold}; checked {candidates[0]}')
    manifest_path = path.parent / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{path}: prediction manifest is missing: {manifest_path}')
    manifest = _json(manifest_path)
    identity = {'dataset': dataset, 'model': model, 'seed': int(seed),
                'test_subject': int(fold) + 1,
                'source_manifest_sha256': data_obj.source_manifest_sha256,
                'class_mapping': data_obj.class_mapping}
    for key, value in identity.items():
        actual = manifest.get(key)
        if key == 'class_mapping' and isinstance(actual, dict):
            actual = {str(k): int(v) for k, v in actual.items()}
        if actual != value:
            raise ValueError(f'{path}: prediction manifest {key}={actual!r}, expected {value!r}')
    if manifest.get('input_manifest_sha256') != data_obj.input_manifest_sha256:
        raise ValueError(f'{path}: input_manifest_sha256 differs from current model input')
    if manifest.get('input_policy_sha256') != data_obj.identity.get('input_policy_sha256'):
        raise ValueError(f'{path}: input_policy_sha256 differs from current preprocessing policy')
    file_records = manifest.get('files', {})
    record = file_records.get(path.name) if isinstance(file_records, dict) else None
    if record is None and manifest.get('result_sha256'):
        record = {'bytes': manifest.get('result_bytes'), 'sha256': manifest['result_sha256']}
    if not isinstance(record, dict) or 'sha256' not in record or 'bytes' not in record:
        raise ValueError(f'{manifest_path}: prediction file size/hash is missing')
    if path.stat().st_size != int(record['bytes']) or sha256_file(path) != str(record['sha256']):
        raise ValueError(f'{path}: prediction file size/hash mismatch')
    with np.load(resolve_local_file(path), allow_pickle=True) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if 'split' in arrays and str(arrays['split'].item()) != split:
        raise ValueError(f'{path}: prediction file split metadata differs from {split!r}')
    return _align_prediction_arrays(arrays, data_obj, fold, split, path) | {
        '_path': str(path), '_manifest_path': str(manifest_path),
        '_manifest_sha256': sha256_file(manifest_path), '_manifest': manifest,
    }


def _align_prediction_arrays(arrays: dict[str, np.ndarray], data_obj: LOSOData,
                              fold: int, split: str, path: Path) -> dict[str, Any]:
    uid_key = next((name for name in ('trial_uid', 'trial_uids', 'sample_uid')
                    if name in arrays), None)
    if uid_key is None:
        raise ValueError(f'{path}: prediction is missing physical trial_uid values')
    raw_uid = np.asarray(arrays[uid_key])
    if raw_uid.ndim == 2:
        raise ValueError(f'{path}: {uid_key} must contain physical string trial UIDs, got {raw_uid.shape}')
    uid = raw_uid.astype(str).reshape(-1)
    if len(set(uid.tolist())) != len(uid):
        raise ValueError(f'{path}: prediction trial_uid values are not unique')
    source_by_uid = {value: i for i, value in enumerate(data_obj.trial_uid.tolist())}
    expected_mask = data_obj.subjects == int(fold)
    if split == 'train':
        expected_mask = ~expected_mask
    elif split != 'test':
        raise ValueError(f'unsupported prediction split {split!r}')
    expected_uids = set(data_obj.trial_uid[expected_mask].tolist())
    if set(uid.tolist()) != expected_uids:
        raise ValueError(f'{path}: prediction UIDs do not exactly match current {split} split')
    if 'y' not in arrays:
        labels_key = 'labels' if 'labels' in arrays else None
        if labels_key is None:
            raise ValueError(f'{path}: prediction is missing labels')
        y = arrays[labels_key]
    else:
        y = arrays['y']
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    expected_y = np.asarray([data_obj.y[source_by_uid[value]] for value in uid], dtype=np.int64)
    if len(y) != len(uid) or not np.array_equal(y, expected_y):
        raise ValueError(f'{path}: prediction labels disagree with current source by trial_uid')
    expected_rows = np.flatnonzero(expected_mask)
    local_order = np.lexsort((data_obj.sample_uid[expected_rows, 1],
                              data_obj.sample_uid[expected_rows, 0]))
    expected_rows = expected_rows[local_order]
    canonical_uids = data_obj.trial_uid[expected_rows].astype(str)
    row_by_uid = {value: index for index, value in enumerate(uid.tolist())}
    order = np.asarray([row_by_uid[value] for value in canonical_uids.tolist()], dtype=np.int64)
    canonical_sample_uid = np.asarray(data_obj.sample_uid[expected_rows], dtype=np.int64)
    if 'local_sample_uid' in arrays:
        local = np.asarray(arrays['local_sample_uid'], dtype=np.int64)
        if local.shape != (len(uid), 2):
            raise ValueError(f'{path}: local_sample_uid must have shape {(len(uid), 2)}')
        expected_local = np.asarray([data_obj.sample_uid[source_by_uid[value]] for value in uid],
                                    dtype=np.int64)
        if not np.array_equal(local, expected_local):
            raise ValueError(f'{path}: local_sample_uid disagrees with physical trial_uid')
    result = {'y': y[order], 'trial_uid': canonical_uids,
              'sample_uid': canonical_sample_uid}
    for output_name, source_names in {
        'logits': ('logits',), 'feats': ('feats', 'features'),
        'probs': ('probs', 'probabilities'), 'pred': ('pred', 'predictions'),
    }.items():
        source_name = next((name for name in source_names if name in arrays), None)
        if source_name is not None:
            values = np.asarray(arrays[source_name])
            if values.ndim == 0 or len(values) != len(uid):
                raise ValueError(f'{path}: {source_name} is not aligned to trial_uid rows')
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise ValueError(f'{path}: {source_name} contains NaN or Inf')
            result[output_name] = values[order]
    return result


def load_baseline_prediction(dataset: str, model: str, seed: int, fold: int,
                             split: str, data_obj: LOSOData) -> dict[str, Any]:
    """Read a current baseline's held-out prediction file without creating copies."""
    if dataset == 'AlexMI':
        from data import loso_alexmi

        return loso_alexmi.load_baseline_prediction(
            model, seed, fold, split, data_obj)
    loso_cfg = config.load_loso_dataset_config(dataset)
    result_root = Path(loso_cfg['result_root'])
    candidates = []
    if dataset in {'BNCI2014004', 'BNCI2015001'}:
        candidates.append(result_root / dataset / model / 'source_refreshed_loso_v1' /
                          f'seed_{int(seed)}' / f'subject_{int(fold) + 1:02d}' / 'result.npz')
    elif dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        candidates.extend([
            result_root / dataset / model / 'scratch_supervised_loso_wideband_v1' /
            f'subject_{int(fold) + 1:02d}' / f'seed_{int(seed)}' / 'result.npz',
            result_root / dataset / model / f'seed_{int(seed)}' /
            f'subject_{int(fold) + 1:02d}' / 'test_predictions.npz',
        ])
    else:
        raise ValueError(f'{dataset}: current baseline result path is not registered')
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f'missing current baseline {split} prediction for {dataset}/{model} '
            f'seed={seed} fold={fold}; checked {candidates}')
    manifest_candidates = (path.parent / 'manifest.json', path.parent / 'result_manifest.json',
                           path.parent / 'result.json')
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f'{path}: current baseline manifest is missing')
    manifest = _json(manifest_path)
    _validate_baseline_manifest(manifest, path, data_obj, dataset, model, seed, fold)
    with np.load(resolve_local_file(path), allow_pickle=True) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if split == 'train':
        # Baseline result files only contain held-out predictions. Training
        # rows must come from the explicit standard cache under artifact-loso.
        return load_cached_prediction(dataset, model, seed, fold, 'train', data_obj)
    return _align_prediction_arrays(arrays, data_obj, fold, split, path) | {
        '_path': str(path), '_manifest_path': str(manifest_path),
        '_manifest_sha256': sha256_file(manifest_path),
    }


def _validate_baseline_manifest(manifest: dict[str, Any], result_path: Path,
                                 data_obj: LOSOData, dataset: str, model: str,
                                 seed: int, fold: int,
                                 check_prediction_hash: bool = True) -> None:
    for key, value in {'dataset': dataset, 'model': model}.items():
        if manifest.get(key) != value:
            raise ValueError(f'{result_path}: baseline manifest {key}={manifest.get(key)!r}, expected {value!r}')
    recorded_seed = manifest.get('seed', manifest.get('fold_seed'))
    if recorded_seed is None or int(recorded_seed) != int(seed):
        raise ValueError(f'{result_path}: baseline seed={recorded_seed!r}, expected {seed}')
    held_out = manifest.get('test_subject', manifest.get('held_out_subject'))
    if held_out is None or int(held_out) != int(fold) + 1:
        raise ValueError(f'{result_path}: held_out_subject differs from requested fold')

    source_files = manifest.get('source_files', {})
    source = source_files.get('source', {}) if isinstance(source_files, dict) else {}
    source_entry = source.get('manifest.json', {}) if isinstance(source, dict) else {}
    source_hash = manifest.get('source_manifest_sha256')
    if source_hash is None and isinstance(source_entry, dict):
        source_hash = source_entry.get('sha256')
    if source_hash is None and dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        # Known 001 teacher result.json schema: input_manifest_sha256 and
        # input_files identify the migrated teacher input; its manifest has
        # already been verified against the current wide source by load_dataset.
        input_sha = manifest.get('input_manifest_sha256')
        current_input_sha = data_obj.input_manifest_sha256
        if not input_sha or input_sha != current_input_sha:
            raise ValueError(f'{result_path}: 001 teacher input manifest differs from current cache')
        input_files = manifest.get('input_files', {})
        input_x = input_files.get('X.npy', {}) if isinstance(input_files, dict) else {}
        current_x = (data_obj.input_manifest or {}).get('files', {}).get('X.npy', {})
        if not isinstance(input_x, dict) or input_x.get('sha256') != current_x.get('sha256'):
            raise ValueError(f'{result_path}: 001 teacher input X differs from current cache')
        source_hash = data_obj.source_manifest_sha256
    if source_hash != data_obj.source_manifest_sha256:
        raise ValueError(f'{result_path}: baseline source manifest SHA differs from current wide source')

    mapping = manifest.get('class_mapping', manifest.get('label_values'))
    if isinstance(mapping, dict):
        mapping = {str(key): int(value) for key, value in mapping.items()}
        if mapping != data_obj.class_mapping:
            raise ValueError(f'{result_path}: baseline class mapping differs from current LOSO task')
    elif dataset not in {'BNCI2014001', 'BNCI2014001-4'}:
        raise ValueError(f'{result_path}: baseline manifest lacks class_mapping')
    elif 'final_model_sha256' in manifest:
        # The current teacher result.json predates an explicit class mapping;
        # its exact label vector is still required to match source UIDs below.
        if not manifest.get('input_manifest_sha256') or not manifest.get('test_predictions_sha256'):
            raise ValueError(f'{result_path}: unrecognized 001 teacher result.json schema')
    elif int(manifest.get('num_classes', -1)) != len(data_obj.class_mapping):
        raise ValueError(f'{result_path}: 001 student num_classes differs from current task')

    if dataset in {'BNCI2014001', 'BNCI2014001-4'} and 'final_model_sha256' not in manifest:
        task_manifests = source_files.get('task_manifests', {}) if isinstance(source_files, dict) else {}
        task_ref = task_manifests.get(dataset) if isinstance(task_manifests, dict) else None
        if not isinstance(task_ref, dict):
            raise ValueError(f'{result_path}: 001 student manifest lacks source_files.task_manifests')
        cache_root = ARTIFACT_LOSO_ROOT / 'model_inputs' / dataset
        cbr_path = cache_root / 'cbramod' / 'manifest.json'
        mi_path = cache_root / 'mirepnet' / 'manifest.json'
        if not cbr_path.is_file() or not mi_path.is_file():
            raise FileNotFoundError(f'{result_path}: canonical 001 task input manifests are missing')
        cbr_sha, mi_sha = sha256_file(cbr_path), sha256_file(mi_path)
        trials_record = _json(cbr_path).get('files', {}).get('trials.csv', {})
        if (task_ref.get('cbr_manifest_sha256') != cbr_sha
                or task_ref.get('mi_manifest_sha256') != mi_sha
                or task_ref.get('sha256') != trials_record.get('sha256')
                or int(task_ref.get('bytes', -1)) != int(trials_record.get('bytes', -2))):
            raise ValueError(f'{result_path}: 001 task manifest identity differs from current canonical inputs')
        local_uids = manifest.get('local_test_sample_uids')
        expected_local = data_obj.sample_uid[data_obj.subjects == int(fold)].tolist()
        if local_uids is not None and np.asarray(local_uids, dtype=np.int64).tolist() != expected_local:
            raise ValueError(f'{result_path}: manifest test local UIDs differ from current source')

    if manifest.get('status') not in (None, 'complete'):
        raise ValueError(f'{result_path}: baseline status is not complete')
    prediction_sha = manifest.get('test_predictions_sha256')
    if check_prediction_hash and prediction_sha and sha256_file(result_path) != str(prediction_sha):
        raise ValueError(f'{result_path}: test prediction SHA differs from manifest')
    expected_sha = manifest.get('result_sha256')
    expected_size = manifest.get('result_bytes')
    if check_prediction_hash and expected_sha and sha256_file(result_path) != str(expected_sha):
        raise ValueError(f'{result_path}: result SHA-256 differs from manifest')
    if check_prediction_hash and expected_size is not None and result_path.stat().st_size != int(expected_size):
        raise ValueError(f'{result_path}: result size differs from manifest')
