"""Read the preserved, manually prepared AlexMI LOSO artifacts.

AlexMI predates the canonical ``artifact-loso`` tree. This module reads only
its original source files, same-fold result manifests, and existing teacher
target caches. It does not create or relocate artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.storage import resolve_local_file


ALEXMI_SOURCE_ROOT = Path('/data1/llx/AlexMI')
ALEXMI_ARTIFACT_ROOT = Path('/data1/llx/BigSmallcollab/cache/artifacts/AlexMI')
ALEXMI_TEACHER_TARGET_ROOT = Path(
    '/data1/llx/BigSmallcollab/cache/teacher_targets/'
    'loso_five_settings_kd_feature_warmup10_v1/AlexMI'
)
ALEXMI_RESULT_ROOT = Path(
    '/data1/llx/BigSmallcollab/results/reproductions/'
    'loso_five_settings_canonical4s_v1/AlexMI'
)
_PROJECT_RESULT_ROOT = Path(
    '/home/lixinli/BigSmallCollab/results/reproductions/'
    'loso_five_settings_canonical4s_v1/AlexMI'
)
_DATASET = 'AlexMI'
_MODELS = ('adfcnn', 'cbramod', 'eegnet', 'ifnet', 'mirepnet')
_TEACHERS = ('cbramod', 'mirepnet')
_SEEDS = (666, 667, 668)
_LABELS = {'feet': 0, 'right_hand': 1}
_SOURCE_FILES = ('X.npy', 'labels.npy', 'meta.csv')
_PROTOCOL = 'loso_five_settings_canonical4s_v1'
_TEACHER_PROTOCOL = 'loso_five_settings_kd_feature_warmup10_v1'
_LOADER_RECIPE = {
    'source_trial_selection': {
        'session': '0',
        'run': '0',
        'window': 'mne_resample_up125_down256_take750_then_repeat_first250',
    },
    'window': {
        'canonical_fs_hz': 250,
        'canonical_samples': 1000,
        'window_label': '3s_plus_1s_repeat',
    },
    'label_values': _LABELS,
}
_INPUT_POLICY_SHA256 = hashlib.sha256(b'legacy_alexmi').hexdigest()
_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_SOURCE_ARRAY_CACHE: tuple[tuple[int, int, int, int], dict[str, Any]] | None = None


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    resolved = Path(resolve_local_file(path))
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(resolve_local_file(path))
    if not resolved.is_file():
        raise FileNotFoundError(f'AlexMI LOSO manifest is missing: {resolved}')
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{resolved}: expected a JSON object')
    return value


def _source_payload(manifest: dict[str, Any], context: str) -> dict[str, Any]:
    payload = manifest.get('source_files')
    if not isinstance(payload, dict) or payload.get('dataset_source') != _DATASET:
        raise ValueError(f'{context}: source_files does not identify AlexMI')
    records = payload.get('files')
    if not isinstance(records, dict) or set(records) != set(_SOURCE_FILES):
        raise ValueError(f'{context}: source_files must cover exactly {list(_SOURCE_FILES)}')
    return payload


def _verify_source_files(payload: dict[str, Any], context: str) -> str:
    """Verify the original AlexMI X/labels/meta files and return their record hash."""
    records = payload.get('files')
    if not isinstance(records, dict) or set(records) != set(_SOURCE_FILES):
        raise ValueError(f'{context}: source_files must cover exactly {list(_SOURCE_FILES)}')
    if payload.get('dataset_source') != _DATASET:
        raise ValueError(f'{context}: source_files dataset_source is not AlexMI')
    for name in _SOURCE_FILES:
        record = records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f'{context}: source_files record for {name} is missing')
        expected = ALEXMI_SOURCE_ROOT / name
        recorded = record.get('path')
        if recorded != str(expected):
            raise ValueError(
                f'{context}: {name} must reference the preserved source {expected}, got {recorded!r}')
        path = Path(resolve_local_file(expected))
        if not path.is_file():
            raise FileNotFoundError(f'{context}: AlexMI source file is missing: {path}')
        if path.stat().st_size != int(record.get('bytes', -1)):
            raise ValueError(f'{context}: byte size differs for {path}')
        recorded_hash = str(record.get('sha256', ''))
        if len(recorded_hash) != 64 or _sha256_file(path) != recorded_hash:
            raise ValueError(f'{context}: SHA-256 differs for {path}')
    for key in ('spec_sha256', 'trial_manifest_sha256'):
        value = str(payload.get(key, ''))
        if len(value) != 64:
            raise ValueError(f'{context}: source_files is missing historical {key}')
    return _json_hash(payload)


def _current_source_files() -> tuple[dict[str, Any], str]:
    """Fingerprint the three physical AlexMI source files without a run artifact."""
    records: dict[str, dict[str, Any]] = {}
    for name in _SOURCE_FILES:
        path = Path(resolve_local_file(ALEXMI_SOURCE_ROOT / name))
        if not path.is_file():
            raise FileNotFoundError(f'AlexMI source file is missing: {path}')
        records[name] = {
            'path': str(ALEXMI_SOURCE_ROOT / name),
            'bytes': int(path.stat().st_size),
            'sha256': _sha256_file(path),
        }
    payload = {'dataset_source': _DATASET, 'files': records}
    return payload, _json_hash(payload)


def _source_arrays() -> dict[str, Any]:
    """Read only labels/metadata plus the memory-mapped source shape."""
    global _SOURCE_ARRAY_CACHE
    labels_path = Path(resolve_local_file(ALEXMI_SOURCE_ROOT / 'labels.npy'))
    meta_path = Path(resolve_local_file(ALEXMI_SOURCE_ROOT / 'meta.csv'))
    x_path = Path(resolve_local_file(ALEXMI_SOURCE_ROOT / 'X.npy'))
    signature = (int(labels_path.stat().st_mtime_ns), int(meta_path.stat().st_mtime_ns),
                 int(x_path.stat().st_mtime_ns), int(x_path.stat().st_size))
    if _SOURCE_ARRAY_CACHE is not None and _SOURCE_ARRAY_CACHE[0] == signature:
        return _SOURCE_ARRAY_CACHE[1]

    labels = np.asarray(np.load(labels_path, allow_pickle=False)).astype(str).reshape(-1)
    with meta_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(labels) != 480 or len(rows) != len(labels):
        raise ValueError(f'AlexMI expected 480 source rows, got labels={len(labels)}, meta={len(rows)}')
    raw_x = np.load(x_path, mmap_mode='r', allow_pickle=False)
    if raw_x.ndim != 3 or raw_x.shape[0] != len(labels):
        raise ValueError(f'AlexMI X.npy has an unexpected shape: {raw_x.shape}')

    raw_uid_to_label: dict[tuple[int, int], int] = {}
    raw_uid_to_row: dict[tuple[int, int], int] = {}
    source_rows_by_subject: dict[int, list[int]] = {subject: [] for subject in range(8)}
    seen_per_subject = {subject: 0 for subject in range(8)}
    for source_row, row in enumerate(rows):
        raw_index = row.get('', row.get('Unnamed: 0', str(source_row)))
        if int(raw_index) != source_row:
            raise ValueError(f'AlexMI meta.csv row index differs at source row {source_row}')
        subject_one_based = int(row['subject'])
        subject = subject_one_based - 1
        if subject not in seen_per_subject:
            raise ValueError(f'AlexMI meta.csv subject is outside 1..8: {subject_one_based}')
        if str(row.get('session', '0')) != '0' or str(row.get('run', '0')) != '0':
            raise ValueError(f'AlexMI meta.csv has an unexpected session/run at row {source_row}')
        seen_per_subject[subject] += 1
        label = str(labels[source_row])
        if label in _LABELS:
            uid = (subject, source_row)
            raw_uid_to_label[uid] = _LABELS[label]
            raw_uid_to_row[uid] = source_row
            source_rows_by_subject[subject].append(source_row)
    if seen_per_subject != {subject: 60 for subject in range(8)}:
        raise ValueError(f'AlexMI source metadata has unexpected per-subject counts: {seen_per_subject}')
    if len(raw_uid_to_label) != 320:
        raise ValueError(f'AlexMI source task expected 320 right-hand/feet trials, got {len(raw_uid_to_label)}')

    result = {
        'labels': labels,
        'meta_rows': rows,
        'raw_uid_to_label': raw_uid_to_label,
        'raw_uid_to_row': raw_uid_to_row,
        'source_rows_by_subject': source_rows_by_subject,
        'source_shape': tuple(int(value) for value in raw_x.shape),
    }
    _SOURCE_ARRAY_CACHE = (signature, result)
    return result


def _data_uid_map(data_obj: Any, source: dict[str, Any]) -> dict[tuple[int, int], int]:
    if data_obj is None:
        raise ValueError('AlexMI LOSO artifact loading requires the current data_obj')
    identity = getattr(data_obj, 'identity', {}) or {}
    if isinstance(identity, dict) and identity.get('dataset') not in (None, _DATASET):
        raise ValueError(f'data_obj identifies another dataset: {identity.get("dataset")!r}')
    if isinstance(identity, dict):
        policy_sha = identity.get('input_policy_sha256')
        if policy_sha not in (None, _INPUT_POLICY_SHA256):
            raise ValueError('data_obj AlexMI input policy differs from the preserved legacy loader')
    class_mapping = getattr(data_obj, 'class_mapping', None)
    if class_mapping is not None and {str(k): int(v) for k, v in class_mapping.items()} != _LABELS:
        raise ValueError(f'data_obj AlexMI class mapping differs from {_LABELS}')

    trial_uid = np.asarray(getattr(data_obj, 'trial_uid', [])).astype(str).reshape(-1)
    subjects = np.asarray(getattr(data_obj, 'subjects', []), dtype=np.int64).reshape(-1)
    y = np.asarray(getattr(data_obj, 'y', []), dtype=np.int64).reshape(-1)
    sample_uid = np.asarray(getattr(data_obj, 'sample_uid', []), dtype=np.int64)
    n = len(trial_uid)
    if (n != 320 or len(subjects) != n or len(y) != n
            or sample_uid.shape != (n, 2) or len(set(trial_uid.tolist())) != n):
        raise ValueError('data_obj does not contain the expected unique 320-trial AlexMI LOSO view')
    if hasattr(data_obj, 'x') and len(data_obj.x) != n:
        raise ValueError('data_obj.x row count differs from its AlexMI UIDs')

    raw_uid_to_data_row: dict[tuple[int, int], int] = {}
    for subject in range(8):
        raw_rows = source['source_rows_by_subject'][subject]
        rows = np.flatnonzero(subjects == subject)
        if len(rows) != len(raw_rows):
            raise ValueError(f'data_obj subject {subject} trial count differs from AlexMI source')
        local_to_row: dict[int, int] = {}
        for data_row in rows.tolist():
            sample_subject, local = map(int, sample_uid[data_row].tolist())
            if sample_subject != subject or local in local_to_row:
                raise ValueError(f'data_obj sample_uid is invalid or duplicated for subject {subject}')
            local_to_row[local] = data_row
        if set(local_to_row) != set(range(len(raw_rows))):
            raise ValueError(f'data_obj sample_uid local indices differ for subject {subject}')
        for local, source_row in enumerate(raw_rows):
            data_row = local_to_row[local]
            uid = (subject, source_row)
            if int(y[data_row]) != int(source['raw_uid_to_label'][uid]):
                raise ValueError(f'data_obj label differs from AlexMI source for sample_uid={uid}')
            raw_uid_to_data_row[uid] = data_row
    if len(raw_uid_to_data_row) != n:
        raise ValueError('data_obj UIDs do not map one-to-one to the selected AlexMI source trials')
    return raw_uid_to_data_row


def _validate_data(manifest: dict[str, Any], data_obj: Any, context: str):
    source_files = _source_payload(manifest, context)
    source_files_sha = _verify_source_files(source_files, context)
    current_source_files, current_source_files_sha = _current_source_files()
    if source_files.get('files') != current_source_files.get('files'):
        raise ValueError(f'{context}: manifest source file records differ from current AlexMI files')
    source = _source_arrays()
    uid_to_row = _data_uid_map(data_obj, source)
    labels = {uid: int(value) for uid, value in source['raw_uid_to_label'].items()}
    recipe = _LOADER_RECIPE
    source_identity_sha = _json_hash({
        'dataset': _DATASET,
        'source_files_sha256': current_source_files_sha,
        'loader_recipe': recipe,
    })
    return (source_files, source_files_sha, current_source_files_sha,
            source_identity_sha, source, uid_to_row, labels, recipe)


def source_identity(data_obj: Any | None = None) -> dict[str, Any]:
    """Fingerprint the preserved source files and fixed AlexMI loader recipe.

    This is independent of trained models, result manifests, and teacher caches.
    """
    source_files, source_files_sha = _current_source_files()
    source = _source_arrays()
    if data_obj is not None:
        _data_uid_map(data_obj, source)
    identity_sha = _json_hash({
        'dataset': _DATASET,
        'source_files_sha256': source_files_sha,
        'loader_recipe': _LOADER_RECIPE,
    })
    return {
        'dataset': _DATASET,
        'source_root': str(ALEXMI_SOURCE_ROOT),
        'source_files': source_files,
        'source_files_sha256': source_files_sha,
        'source_manifest_sha256': identity_sha,
        'source_identity_sha256': identity_sha,
        'loader_recipe': _LOADER_RECIPE,
        'input_policy_sha256': _INPUT_POLICY_SHA256,
        'source_shape': source['source_shape'],
        'selected_trial_count': len(source['raw_uid_to_label']),
    }


def _validate_manifest_identity(manifest: dict[str, Any], path: Path,
                                model: str, seed: int, fold: int) -> None:
    expected = {
        'dataset': _DATASET,
        'model': model,
        'seed': int(seed),
        'test_subject': int(fold) + 1,
        'protocol': _PROTOCOL,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'{path}: manifest {key}={manifest.get(key)!r}, expected {value!r}')
    if manifest.get('dataset_source') != _DATASET:
        raise ValueError(f'{path}: dataset_source is not AlexMI')
    if manifest.get('label_values') != _LABELS or int(manifest.get('num_classes', -1)) != 2:
        raise ValueError(f'{path}: AlexMI label mapping/class count differs from {_LABELS}')
    if manifest.get('source_trial_selection') != _LOADER_RECIPE['source_trial_selection']:
        raise ValueError(f'{path}: unsupported AlexMI source trial selection')
    if manifest.get('window') != _LOADER_RECIPE['window']:
        raise ValueError(f'{path}: unsupported AlexMI canonical window')


def _uids(value: Any, path: Path, key: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path}: {key} is not an integer Nx2 UID list') from error
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f'{path}: {key} must have shape (N, 2), got {result.shape}')
    if len({tuple(row) for row in result.tolist()}) != len(result):
        raise ValueError(f'{path}: {key} contains duplicate UIDs')
    return result


def _validate_split_manifest(manifest: dict[str, Any], path: Path, model: str,
                             seed: int, fold: int, source: dict[str, Any]):
    _validate_manifest_identity(manifest, path, model, seed, fold)
    source_uids = set(source['raw_uid_to_label'])
    train = _uids(manifest.get('train_sample_uids'), path, 'train_sample_uids')
    test = _uids(manifest.get('test_sample_uids'), path, 'test_sample_uids')
    train_set = {tuple(row) for row in train.tolist()}
    test_set = {tuple(row) for row in test.tolist()}
    expected_test = {uid for uid in source_uids if uid[0] == int(fold)}
    expected_train = source_uids - expected_test
    if train_set != expected_train or test_set != expected_test or train_set & test_set:
        raise ValueError(f'{path}: train/test UIDs do not match the source and held-out fold')
    if int(manifest.get('n_train', -1)) != len(train) or int(manifest.get('n_test', -1)) != len(test):
        raise ValueError(f'{path}: train/test counts disagree with their UID lists')
    if set(map(int, manifest.get('train_subjects', []))) != set(range(8)) - {int(fold)}:
        raise ValueError(f'{path}: train_subjects differ from held-out subject {fold}')
    return train, test


def _find_baseline_manifest(model: str, seed: int, fold: int) -> tuple[Path, dict[str, Any]]:
    if model not in _MODELS:
        raise ValueError(f'unsupported AlexMI baseline model {model!r}; expected one of {_MODELS}')
    if int(seed) not in _SEEDS or not 0 <= int(fold) < 8:
        raise ValueError(f'unsupported AlexMI LOSO seed/fold: seed={seed}, fold={fold}')
    model_root = ALEXMI_RESULT_ROOT / model
    if not model_root.is_dir():
        raise FileNotFoundError(f'AlexMI baseline result directory is missing: {model_root}')
    candidates = []
    for candidate in model_root.rglob('manifest.json'):
        try:
            manifest = _read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (manifest.get('dataset') == _DATASET
                and manifest.get('model') == model
                and manifest.get('seed') == int(seed)
                and manifest.get('test_subject') == int(fold) + 1):
            candidates.append((candidate, manifest))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'expected one same-fold AlexMI result manifest for {model}/seed_{seed}/'
            f'subject_{int(fold) + 1:02d}, found {len(candidates)} under {model_root}')
    path, manifest = candidates[0]
    _validate_manifest_identity(manifest, path, model, int(seed), int(fold))
    return path, manifest


def _source_uid_rows(raw_uids: np.ndarray, uid_to_data_row: dict[tuple[int, int], int],
                     path: Path) -> np.ndarray:
    rows = []
    for uid in raw_uids.tolist():
        key = tuple(map(int, uid))
        if key not in uid_to_data_row:
            raise ValueError(f'{path}: artifact UID {key} is not in current AlexMI task source')
        rows.append(uid_to_data_row[key])
    return np.asarray(rows, dtype=np.int64)


def _ordered_result(arrays: dict[str, np.ndarray], raw_uids: np.ndarray,
                    expected_uids: np.ndarray, data_obj: Any,
                    uid_to_data_row: dict[tuple[int, int], int], labels: dict[tuple[int, int], int],
                    path: Path) -> dict[str, Any]:
    if set(map(tuple, raw_uids.tolist())) != set(map(tuple, expected_uids.tolist())):
        raise ValueError(f'{path}: prediction UIDs do not exactly match the recorded split')
    data_rows = _source_uid_rows(raw_uids, uid_to_data_row, path)
    y = np.asarray(arrays.get('y'), dtype=np.int64).reshape(-1)
    expected_y = np.asarray([labels[tuple(uid)] for uid in raw_uids.tolist()], dtype=np.int64)
    if len(y) != len(raw_uids) or not np.array_equal(y, expected_y):
        raise ValueError(f'{path}: prediction labels differ from current AlexMI source by UID')
    for key in ('logits', 'feats'):
        value = arrays.get(key)
        if value is None:
            raise ValueError(f'{path}: prediction is missing {key}')
        value = np.asarray(value)
        if value.ndim == 0 or len(value) != len(raw_uids):
            raise ValueError(f'{path}: {key} row count differs from its sample UIDs')
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f'{path}: {key} contains non-numeric or non-finite values')

    sample_uid_all = np.asarray(data_obj.sample_uid, dtype=np.int64)
    trial_uid_all = np.asarray(data_obj.trial_uid).astype(str)
    order = np.lexsort((sample_uid_all[data_rows, 1], sample_uid_all[data_rows, 0]))
    ordered_rows = data_rows[order]
    result = {
        'logits': np.asarray(arrays['logits'])[order],
        'feats': np.asarray(arrays['feats'])[order],
        'y': y[order],
        'sample_uid': sample_uid_all[ordered_rows],
        'trial_uid': trial_uid_all[ordered_rows],
        'split_policy': 'loso',
    }
    if len(set(result['trial_uid'].tolist())) != len(result['trial_uid']):
        raise ValueError(f'{path}: mapped data_obj trial_uid values are not unique')
    return result


def _recorded_result_hash(manifest: dict[str, Any], path: Path) -> tuple[int | None, str | None]:
    files = manifest.get('files')
    record = files.get(path.name) if isinstance(files, dict) else None
    if not isinstance(record, dict):
        record = None
    size = None
    digest = None
    if record is not None:
        if record.get('bytes') is not None:
            size = int(record['bytes'])
        if record.get('sha256') is not None:
            digest = str(record['sha256'])
    if manifest.get('result_bytes') is not None:
        size = int(manifest['result_bytes'])
    for key in ('result_sha256', 'result_npz_sha256', 'artifact_sha256'):
        if manifest.get(key) is not None:
            digest = str(manifest[key])
            break
    if size is not None and path.stat().st_size != size:
        raise ValueError(f'{path}: result NPZ byte size differs from its manifest')
    if digest is not None and _sha256_file(path) != digest:
        raise ValueError(f'{path}: result NPZ SHA-256 differs from its manifest')
    return size, digest


def load_teacher_targets(teacher: str, seed: int, fold: int, data_obj: Any) -> dict[str, Any]:
    """Load one existing AlexMI teacher-training target cache and align it."""
    teacher = str(teacher)
    seed = int(seed)
    fold = int(fold)
    if teacher not in _TEACHERS:
        raise ValueError(f'AlexMI has no verified historical teacher targets for {teacher!r}')
    if seed not in _SEEDS or not 0 <= fold < 8:
        raise ValueError(f'unsupported AlexMI LOSO seed/fold: seed={seed}, fold={fold}')

    manifest_path = (ALEXMI_TEACHER_TARGET_ROOT / teacher / f'subject_{fold + 1:02d}'
                     / f'seed_{seed}' / 'cache_manifest.json')
    artifact_path = manifest_path.parent / 'cache.npz'
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(
            f'missing verified AlexMI teacher cache for {teacher}/seed_{seed}/'
            f'subject_{fold + 1:02d}: {manifest_path} and {artifact_path}')
    manifest = _read_json(manifest_path)

    expected = {
        'dataset': _DATASET,
        'teacher': teacher,
        'protocol': _TEACHER_PROTOCOL,
        'seed': seed,
        'test_subject': fold + 1,
        'fold_zero_based': fold,
        'target_data_policy': 'source_trials_only_no_held_out_teacher_inference',
        'class_mapping': _LABELS,
        'num_classes': 2,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == 'class_mapping' and isinstance(actual, dict):
            actual = {str(name): int(label) for name, label in actual.items()}
        if actual != value:
            raise ValueError(f'{manifest_path}: {key}={actual!r}, expected {value!r}')

    teacher_manifest_record = manifest.get('teacher_manifest')
    if not isinstance(teacher_manifest_record, dict):
        raise ValueError(f'{manifest_path}: original teacher_manifest path/hash record is missing')
    recorded_baseline_path = teacher_manifest_record.get('path')
    recorded_baseline_sha = teacher_manifest_record.get('sha256')
    if not recorded_baseline_path or not recorded_baseline_sha:
        raise ValueError(f'{manifest_path}: original teacher_manifest path/hash is incomplete')
    recorded_path = Path(str(recorded_baseline_path))
    allowed_roots = (ALEXMI_RESULT_ROOT, _PROJECT_RESULT_ROOT)
    if not any(_is_relative_to(recorded_path, root) for root in allowed_roots):
        raise ValueError(f'{manifest_path}: teacher_manifest path is outside the historical AlexMI result tree')
    baseline_manifest_path = Path(resolve_local_file(recorded_path))
    expected_parts = {f'subject_{fold + 1:02d}', f'seed_{seed}'}
    if (baseline_manifest_path.name != 'manifest.json'
            or not expected_parts.issubset(set(baseline_manifest_path.parts))
            or not baseline_manifest_path.is_file()):
        raise ValueError(f'{manifest_path}: teacher_manifest does not name this same-fold baseline')
    baseline_manifest_sha = _sha256_file(baseline_manifest_path)
    if (str(recorded_baseline_sha) != baseline_manifest_sha
            or str(manifest.get('teacher_manifest_sha256', '')) != baseline_manifest_sha):
        raise ValueError(f'{manifest_path}: teacher_manifest SHA differs from the same-fold baseline')
    baseline_manifest = _read_json(baseline_manifest_path)
    _validate_manifest_identity(baseline_manifest, baseline_manifest_path, teacher, seed, fold)
    if manifest.get('teacher') != baseline_manifest.get('model'):
        raise ValueError(f'{manifest_path}: teacher model differs from its baseline manifest')

    (source_files, source_files_sha, raw_source_files_sha, source_identity_sha,
     source, uid_to_data_row, labels, recipe) = (
        _validate_data(baseline_manifest, data_obj, str(baseline_manifest_path)))
    target_source_files = _source_payload(manifest, str(manifest_path))
    target_source_sha = _verify_source_files(target_source_files, str(manifest_path))
    if target_source_files != source_files or target_source_sha != source_files_sha:
        raise ValueError(f'{manifest_path}: teacher cache source_files differ from same-fold baseline')
    if str(manifest.get('source_files_sha256', '')) != source_files_sha:
        raise ValueError(f'{manifest_path}: source_files_sha256 differs from current AlexMI sources')
    if manifest.get('source_snapshot_sha256') != source_files.get('spec_sha256'):
        # Older snapshots record a separate source snapshot hash; keep that
        # historical value, but require it to be present and well formed.
        snapshot_sha = str(manifest.get('source_snapshot_sha256', ''))
        if len(snapshot_sha) != 64:
            raise ValueError(f'{manifest_path}: source_snapshot_sha256 is missing or invalid')
    expected_train_subjects = set(range(8)) - {fold}
    if set(map(int, manifest.get('train_subjects_zero_based', []))) != expected_train_subjects:
        raise ValueError(f'{manifest_path}: teacher target train_subjects_zero_based differs from fold')

    train_uids, test_uids = _validate_split_manifest(
        baseline_manifest, baseline_manifest_path, teacher, seed, fold, source)
    target_data_hash = str(manifest.get('teacher_checkpoint_sha256', ''))
    checkpoint = manifest.get('teacher_checkpoint')
    if not isinstance(checkpoint, dict) or not checkpoint.get('path') or not checkpoint.get('sha256'):
        raise ValueError(f'{manifest_path}: same-fold teacher checkpoint provenance is missing')
    recorded_checkpoint = Path(str(checkpoint['path']))
    if not any(_is_relative_to(recorded_checkpoint, root) for root in allowed_roots):
        raise ValueError(f'{manifest_path}: teacher checkpoint is outside the historical AlexMI result tree')
    checkpoint_path = Path(resolve_local_file(recorded_checkpoint))
    if checkpoint_path != baseline_manifest_path.parent / 'model.pt':
        raise ValueError(f'{manifest_path}: teacher checkpoint is not from the same fold and seed')
    checkpoint_sha = str(checkpoint['sha256'])
    if len(checkpoint_sha) != 64 or checkpoint_sha != target_data_hash:
        raise ValueError(f'{manifest_path}: recorded same-fold teacher checkpoint SHA differs')
    checkpoint_exists = checkpoint_path.is_file()
    if checkpoint_exists and _sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError(f'{manifest_path}: present same-fold teacher checkpoint SHA differs')

    target_sha = str(manifest.get('cache_file_sha256', ''))
    if len(target_sha) != 64 or _sha256_file(artifact_path) != target_sha:
        raise ValueError(f'{artifact_path}: cache_file_sha256 differs from target manifest')
    if manifest.get('cache_bytes') is not None and artifact_path.stat().st_size != int(manifest['cache_bytes']):
        raise ValueError(f'{artifact_path}: cache byte size differs from target manifest')

    with np.load(Path(resolve_local_file(artifact_path)), allow_pickle=False) as archive:
        required = {'logits', 'feats', 'y', 'sample_uid'}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f'{artifact_path}: missing arrays {sorted(missing)}')
        arrays = {name: np.asarray(archive[name]) for name in required}
    raw_uids = _uids(arrays['sample_uid'], artifact_path, 'sample_uid')
    target_y = np.asarray(arrays['y'], dtype=np.int64).reshape(-1)
    target_train_set = {tuple(row) for row in raw_uids.tolist()}
    expected_train_set = {tuple(row) for row in train_uids.tolist()}
    expected_test_set = {tuple(row) for row in test_uids.tolist()}
    if target_train_set != expected_train_set or target_train_set & expected_test_set:
        raise ValueError(f'{artifact_path}: teacher targets include missing, held-out, or test UIDs')
    if not np.array_equal(raw_uids, train_uids):
        raise ValueError(f'{artifact_path}: teacher target UID order differs from its baseline train manifest')
    if len(target_y) != len(raw_uids) or not np.array_equal(
            target_y, np.asarray([labels[tuple(uid)] for uid in raw_uids.tolist()], dtype=np.int64)):
        raise ValueError(f'{artifact_path}: teacher target labels differ from AlexMI source by UID')
    if _array_hash(raw_uids) != str(manifest.get('sample_uid_sha256', '')):
        raise ValueError(f'{artifact_path}: ordered sample_uid hash differs from target manifest')
    if _array_hash(target_y) != str(manifest.get('labels_sha256', '')):
        raise ValueError(f'{artifact_path}: ordered labels hash differs from target manifest')
    if int(manifest.get('n_train', -1)) != len(target_y):
        raise ValueError(f'{manifest_path}: n_train differs from target array count')
    if int(manifest.get('feature_dim', -1)) != int(np.asarray(arrays['feats']).shape[-1]):
        raise ValueError(f'{manifest_path}: feature_dim differs from target features')

    result = _ordered_result(arrays, raw_uids, train_uids, data_obj,
                             uid_to_data_row, labels, artifact_path)
    result.update({
        '_manifest': manifest,
        '_manifest_path': str(manifest_path),
        '_manifest_sha256': _sha256_file(manifest_path),
        '_artifact_path': str(artifact_path),
        '_artifact_sha256': target_sha,
        '_path': str(artifact_path),
        '_input_manifest_sha256': getattr(data_obj, 'input_manifest_sha256', None),
        '_input_policy_sha256': (getattr(data_obj, 'identity', {}) or {}).get(
            'input_policy_sha256', _INPUT_POLICY_SHA256) or _INPUT_POLICY_SHA256,
        '_source_manifest_sha256': source_identity_sha,
        '_source_files_sha256': source_files_sha,
        '_raw_source_files_sha256': raw_source_files_sha,
        '_source_identity_sha256': source_identity_sha,
        '_source_snapshot_sha256': manifest.get('source_snapshot_sha256'),
        '_teacher_manifest': baseline_manifest,
        '_teacher_manifest_path': recorded_baseline_path,
        '_teacher_manifest_resolved_path': str(baseline_manifest_path),
        '_teacher_manifest_sha256': baseline_manifest_sha,
        '_teacher_checkpoint_path': str(checkpoint_path),
        '_teacher_checkpoint_sha256': checkpoint_sha,
        '_teacher_checkpoint_available': checkpoint_exists,
        '_recipe': recipe,
    })
    return result


def load_baseline_prediction(model: str, seed: int, fold: int, split: str,
                             data_obj: Any) -> dict[str, Any]:
    """Load only a UID-bearing, same-fold AlexMI baseline test prediction.

    Historical ``*_loso`` training caches omit sample UIDs and a cache
    manifest, so the train split is deliberately reported as unavailable.
    """
    model = str(model)
    seed = int(seed)
    fold = int(fold)
    if model not in _MODELS:
        raise ValueError(f'unsupported AlexMI baseline model {model!r}; expected one of {_MODELS}')
    if seed not in _SEEDS or not 0 <= fold < 8:
        raise ValueError(f'unsupported AlexMI LOSO seed/fold: seed={seed}, fold={fold}')
    if split != 'test':
        legacy_path = ALEXMI_ARTIFACT_ROOT / f'{model}_loso' / f'{fold}_{seed}_{split}.npz'
        if split not in {'train', 'test'}:
            raise ValueError(f'unsupported AlexMI prediction split {split!r}')
        raise FileNotFoundError(
            f'AlexMI baseline {split} predictions are unavailable for strict UID alignment: '
            f'{legacy_path} has no sample_uid array or cache manifest')

    manifest_path, manifest = _find_baseline_manifest(model, seed, fold)
    (source_files, source_files_sha, raw_source_files_sha, source_identity_sha,
     source, uid_to_data_row, labels, recipe) = (
        _validate_data(manifest, data_obj, str(manifest_path)))
    expected_source_files = _source_payload(manifest, str(manifest_path))
    if expected_source_files != source_files:
        raise ValueError(f'{manifest_path}: source_files changed during AlexMI identity validation')
    train_uids, test_uids = _validate_split_manifest(
        manifest, manifest_path, model, seed, fold, source)
    result_path = manifest_path.parent / 'result.npz'
    if not result_path.is_file():
        raise FileNotFoundError(f'AlexMI same-fold baseline prediction is missing: {result_path}')
    _recorded_result_hash(manifest, result_path)
    with np.load(Path(resolve_local_file(result_path)), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if 'sample_uid' not in arrays:
        raise ValueError(f'{result_path}: baseline result is missing physical sample_uid values')
    raw_uids = _uids(arrays['sample_uid'], result_path, 'sample_uid')
    result = _ordered_result(arrays, raw_uids, test_uids, data_obj,
                             uid_to_data_row, labels, result_path)
    if 'local_sample_uid' in arrays:
        local_uid = np.asarray(arrays['local_sample_uid'], dtype=np.int64)
        data_rows = _source_uid_rows(raw_uids, uid_to_data_row, result_path)
        expected_local = np.asarray(data_obj.sample_uid, dtype=np.int64)[data_rows]
        if local_uid.shape != (len(raw_uids), 2) or not np.array_equal(local_uid, expected_local):
            raise ValueError(f'{result_path}: local_sample_uid differs from mapped AlexMI trial UIDs')
    result_sha = _sha256_file(result_path)
    result.update({
        '_manifest': manifest,
        '_manifest_path': str(manifest_path),
        '_manifest_sha256': _sha256_file(manifest_path),
        '_artifact_path': str(result_path),
        '_artifact_sha256': result_sha,
        '_path': str(result_path),
        '_input_manifest_sha256': getattr(data_obj, 'input_manifest_sha256', None),
        '_input_policy_sha256': (getattr(data_obj, 'identity', {}) or {}).get(
            'input_policy_sha256', _INPUT_POLICY_SHA256) or _INPUT_POLICY_SHA256,
        '_source_manifest_sha256': source_identity_sha,
        '_source_files_sha256': source_files_sha,
        '_raw_source_files_sha256': raw_source_files_sha,
        '_source_identity_sha256': source_identity_sha,
        '_source_snapshot_sha256': None,
        '_recipe': recipe,
    })
    return result


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
