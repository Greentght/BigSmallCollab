"""Few-shot structure-aware KD runner with calibration-only teacher targets.

The runner consumes the current model-agnostic source reader, creates a fixed
stratified 30/70 split for each subject/seed, caches model-ready inputs and
calibration-only teacher outputs outside the checkout, and records one durable
progress cell after each completed student fit.  It intentionally trains only
kd_proto, kd_relation and ea_kd; historical CE/KD files are read only by the
reporter.
"""
from __future__ import annotations

# Set backend limits before importing NumPy, SciPy or PyTorch in this worker.
import os
_THREADS = max(1, int(os.environ.get('BIGSMALLCOLLAB_CPU_THREADS', '1')))
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = str(_THREADS)

import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

import config
from collab import distill
from collab.seed import set_seed
from data import loso_source, split as split_utils
from experiments.distill.progress import ProgressStore
from experiments.storage import (DATA_CACHE_ROOT, REPORTS_ROOT,
                                 require_external_output, require_report_output,
                                 resolve_local_file)
from models import get_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/experiments/fewshot/structure_kd.yaml'
METHODS = ('kd_proto', 'kd_relation', 'ea_kd')
DATASETS = ('BNCI2014004', 'BNCI2014001', 'BNCI2014001-4',
            'BNCI2015001', 'AlexMI')
TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
CACHE_ROOT = DATA_CACHE_ROOT / 'artifact-fewshot'
SOURCE_CODE_FILES = (
    'config.py',
    'experiments/distill/run_fewshot_structure_kd.py',
    'collab/distill.py',
    'data/loso_source.py',
    'data/split.py',
    'data/preproc.py',
    'data/channels.py',
    'models/base.py',
    'models/mirepnet/adapter.py',
    'models/mirepnet/mlm.py',
    'models/cbramod/adapter.py',
    'models/cbramod/cbramod.py',
    'models/cbramod/criss_cross_transformer.py',
    'models/ifnet/adapter.py',
    'models/ifnet/ifnet.py',
    'models/eegnet/adapter.py',
    'models/eegnet/residual_eegnet.py',
    'models/adfcnn/adapter.py',
    'models/adfcnn/adfcnn.py',
    'configs/experiments/distill_kd.yaml',
    'configs/models/mirepnet.yaml',
    'configs/models/cbramod.yaml',
    'configs/models/ifnet.yaml',
    'configs/models/eegnet.yaml',
    'configs/models/adfcnn.yaml',
)
METRIC_FIELDS = (
    'dataset', 'subject', 'subject_index', 'session', 'protocol', 'seed',
    'teacher', 'student', 'method', 'status', 'accuracy_pct',
    'balanced_accuracy_pct', 'kappa', 'n_calibration', 'n_test', 'num_classes',
    'epochs', 'lr', 'batch_size', 'weight_decay', 'optimizer', 'scheduler',
    'warmup_epochs', 'lambda_kd', 'temperature', 'lambda_proto',
    'lambda_relation', 'lambda_intra', 'lambda_inter', 'entropy_temperature',
    'teacher_accuracy_pct', 'teacher_balanced_accuracy_pct', 'teacher_kappa',
    'teacher_cache_action',
    'teacher_checkpoint_sha256', 'teacher_target_manifest_sha256',
    'model_input_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
    'split_uid_sha256', 'initial_state_sha256', 'batch_order_sha256',
    'final_total_loss', 'final_ce_loss', 'final_train_accuracy_pct',
    'gradient_batches_checked', 'gradient_checks_passed',
    'relation_history_json', 'ea_kd_history_json', 'runtime_seconds',
    'predictions_path', 'prediction_count',
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode('utf-8'))


def _sha256_file(path: str | Path) -> str:
    path = Path(resolve_local_file(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp',
                                dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _set_runtime_threads() -> None:
    torch.set_num_threads(_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this to be set only before the first parallel work.
        pass


def _device(gpu: int | None) -> torch.device:
    if gpu is None:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'GPU {gpu} was assigned but this worker cannot access CUDA; '
            'refusing to silently train on CPU')
    torch.cuda.set_device(int(gpu))
    return torch.device(f'cuda:{int(gpu)}')


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'expected YAML mapping in {path}')
    return value


def _code_hashes() -> dict[str, str]:
    return {name: _sha256_file(PROJECT_ROOT / name) for name in SOURCE_CODE_FILES}


def resolve_experiment(config_path: str | Path, *, datasets=None, teachers=None,
                       students=None, seeds=None, methods=None, subjects=None,
                       run_id: str | None = None, smoke: bool = False,
                       smoke_epochs: int = 1) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)
    ds = list(datasets or raw['datasets'])
    tch = list(teachers or raw['teachers'])
    stu = list(students or raw['students'])
    seed_list = [int(x) for x in (seeds or raw['seeds'])]
    meth = list(methods or raw['methods'])
    if not ds or any(x not in DATASETS for x in ds):
        raise ValueError(f'datasets must be a non-empty subset of {DATASETS}')
    if not tch or any(x not in TEACHERS for x in tch):
        raise ValueError(f'teachers must be a non-empty subset of {TEACHERS}')
    if not stu or any(x not in STUDENTS for x in stu):
        raise ValueError(f'students must be a non-empty subset of {STUDENTS}')
    if not meth or any(x not in METHODS for x in meth):
        raise ValueError(f'methods must be a non-empty subset of {METHODS}')
    if not seed_list or len(seed_list) != len(set(seed_list)) or any(x < 0 for x in seed_list):
        raise ValueError('seeds must be unique non-negative integers')

    dataset_cfgs: dict[str, Any] = {}
    teacher_cfgs: dict[str, Any] = {}
    student_cfgs: dict[str, Any] = {}
    for dataset in ds:
        dcfg = config.load_dataset_config(dataset)
        source_cfg = (config.load_loso_dataset_config(dataset)
                      if dataset != 'AlexMI' else {})
        if subjects is not None:
            keys = [int(x) for x in subjects]
        else:
            keys = list(range(int(dcfg['num_subjects'])))
        if not keys or len(keys) != len(set(keys)) or any(
                key < 0 or key >= int(dcfg['num_subjects']) for key in keys):
            raise ValueError(f'{dataset}: subjects must be unique valid zero-based IDs')
        dataset_cfgs[dataset] = {
            'num_subjects': int(dcfg['num_subjects']),
            'num_classes': int(dcfg['num_classes']),
            'sample_rate': int(dcfg.get('sample_rate', 250)),
            'session': raw['split']['session'][dataset],
            'reader_session': source_cfg.get('selected_session', 'current AlexMI loader'),
            'source_directory': source_cfg.get('source_directory', 'current AlexMI loader'),
            'source_identity': source_cfg.get('source_variant', 'legacy_alexmi_verified'),
            'keys': keys,
        }
        for teacher in tch:
            tcfg = config.load_model_config(teacher, dataset, 'fewshot')
            tcfg['dataset_name'] = dataset
            if teacher == 'cbramod':
                tcfg.setdefault('feature_head', 'flatten')
                tcfg.setdefault('optimizer', 'adamw')
                tcfg['classifier_head'] = 'flatten -> Dropout -> Linear'
                tcfg['lr_schedule_resolved'] = (
                    'cosine step table; applied at epoch end using global step')
            else:
                tcfg['optimizer'] = str(tcfg.get('optimizer', 'adam')).lower()
                tcfg['lr_schedule_resolved'] = 'CosineAnnealingLR stepped once per epoch'
                tcfg['classifier_head'] = 'MIRepNet task classifier from current adapter'
            teacher_cfgs[f'{dataset}/{teacher}'] = tcfg
        for student in stu:
            scfg = config.load_model_config(student, dataset, 'fewshot')
            scfg['optimizer'] = str(scfg.get('optimizer', 'adamw')).lower()
            scfg['lr_schedule'] = str(scfg.get('lr_schedule', 'cosine_epoch_end'))
            scfg['warmup_epochs'] = int(scfg.get('warmup_epochs', 0))
            scfg['num_workers'] = 0
            student_cfgs[f'{dataset}/{student}'] = scfg

    # The fully resolved ordinary-KD pair parameters are frozen in the dedicated
    # experiment YAML, which mirrors configs/experiments/distill_kd.yaml without
    # expanding any grid. Where per-run few-shot metadata exists, the report
    # checks it cellwise against the exact split and resolved student settings.
    resolved_kd = copy.deepcopy(raw['kd_parameters'])
    expected_pairs = {(teacher, student) for teacher in tch for student in stu}
    missing = [(teacher, student) for teacher, student in expected_pairs
               if student not in resolved_kd.get(teacher, {})]
    if missing:
        raise ValueError(f'config lacks lambda/T values for pairs: {missing}')

    run_id = run_id or datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    output_dir = require_report_output(
        Path(REPORTS_ROOT) / 'fewshot' / 'structure_kd' / run_id /
        'config_resolved.json').parent
    cache_root = require_external_output(raw.get('cache_root', CACHE_ROOT))
    expected_subject_units = sum(len(dataset_cfgs[d]['keys']) for d in ds)
    resolved = {
        'schema_version': 1,
        'name': 'fewshot_structure_kd',
        'run_id': run_id,
        'protocol': 'fewshot',
        'source_config': str(path),
        'source_config_sha256': _sha256_file(path),
        'config': raw,
        'scope': {
            'datasets': ds, 'teachers': tch, 'students': stu,
            'seeds': seed_list, 'methods': meth,
            'subjects_by_dataset': {d: dataset_cfgs[d]['keys'] for d in ds},
            'subject_task_units': expected_subject_units,
            'new_student_fits': expected_subject_units * len(seed_list) * len(tch)
                                  * len(stu) * len(meth),
            'new_fits_per_method': expected_subject_units * len(seed_list)
                                   * len(tch) * len(stu),
            'teacher_finetune_upper_bound': expected_subject_units * len(seed_list)
                                           * len(tch),
            'split': raw['split'],
            'strict_historical_pairing': 'cellwise validation against saved UID/config fields',
        },
        'dataset_configs': dataset_cfgs,
        'teacher_configs': teacher_cfgs,
        'student_configs': student_cfgs,
        'kd_parameters': resolved_kd,
        'method_parameters': raw['method_parameters'],
        'historical_references': {
            'pattern': raw['historical_reference_pattern'],
            'full_per_run_candidates': [
                str(PROJECT_ROOT / 'results/distill/distill_mi_lam05/results_per_run.csv'),
                str(PROJECT_ROOT / 'results/distill/distill_mi/results_per_run.csv')],
            'policy': 'display saved CE/KD records; permit paired tests only when '
                      'protocol, task, full subject/seed coverage, UID split, '
                      'student settings and KD lambda/temperature match cellwise',
        },
        'parameter_resolution': {
            'student_training': 'configs/models/{student}.yaml resolved for each task/fewshot',
            'ordinary_kd': 'configs/experiments/distill_kd.yaml pair params copied to dedicated YAML; grid is not expanded',
            'mirepnet_ifnet_crosscheck': 'results/distill/distill_mi_lam05/results_per_run.csv records 100 epochs, lr=0.001, batch=16, wd=0.01, AdamW/CosineAnnealingLR, lambda=0.5, T=2; these agree with the frozen values',
            'other_pairs': 'compact historical result CSVs omit actual training parameters; frozen values come from the current resolved model YAML and pair params in distill_kd.yaml',
        },
        'storage': {'cache_root': str(cache_root), 'output_dir': str(output_dir)},
        'runtime': {**raw['runtime'], 'cpu_threads': _THREADS,
                    'pytorch_intraop_threads': 1,
                    'pytorch_interop_threads': 1},
        'code_sha256': _code_hashes(),
        'smoke': {'enabled': bool(smoke), 'epochs_override': int(smoke_epochs)
                  if smoke else None},
    }
    if smoke:
        if smoke_epochs < 1:
            raise ValueError('smoke epochs must be positive')
        # A smoke has an isolated run ID and only verifies the real calibration
        # pipeline; it can never be mistaken for a formal matrix result.
        resolved['smoke']['scope_override'] = {
            'dataset': ds[0], 'teacher': tch[0], 'student': stu[0],
            'subject_index': dataset_cfgs[ds[0]]['keys'][0],
            'seed': seed_list[0], 'methods': meth,
            'teacher_epochs': int(smoke_epochs),
            'student_epochs': int(smoke_epochs),
        }
    return resolved


def write_resolved_config(resolved: dict[str, Any]) -> Path:
    out = Path(resolved['storage']['output_dir']) / 'config_resolved.json'
    out = require_report_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f'.{out.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(resolved, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def _fit_ea(calibration: np.ndarray) -> np.ndarray:
    if calibration.ndim != 3 or len(calibration) < 2:
        raise ValueError('EA needs at least two calibration trials shaped (N,C,T)')
    covariance = np.mean(np.stack([np.cov(trial) for trial in calibration]), axis=0)
    whitening = fractional_matrix_power(covariance, -0.5)
    if np.iscomplexobj(whitening):
        max_imag = float(np.max(np.abs(whitening.imag)))
        if max_imag > 1e-6:
            raise ValueError(f'calibration EA matrix is complex (max imag={max_imag})')
        whitening = whitening.real
    return np.asarray(whitening, dtype=np.float64)


def _apply_ea(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum('ij,njt->nit', matrix, np.asarray(x, dtype=np.float64),
                     optimize=True).astype(np.float32)


def _prepare_mirepnet_input(dataset: str, raw_cal: np.ndarray,
                            raw_test: np.ndarray, adapter) -> tuple[np.ndarray, np.ndarray, str]:
    # MIRepNet's legacy adapter refits EA for every supplied array. Fit once on
    # calibration and explicitly apply that transform to the held-out trials.
    ea = _fit_ea(raw_cal)
    cal_ea, test_ea = _apply_ea(raw_cal, ea), _apply_ea(raw_test, ea)
    channels = adapter._src_channels[dataset]
    cal = adapter._pad(cal_ea, adapter._template, channels).astype(np.float32)
    test = adapter._pad(test_ea, adapter._template, channels).astype(np.float32)
    return cal, test, _sha256_array(ea)


def _prepare_cbramod_input(raw: np.ndarray, cfg: dict[str, Any], adapter,
                           ea_matrix: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)[:, :, :1000]
    x = adapter._resample_to_target(x, adapter._target_fs())
    x = adapter._apply_frequency_filters(x, adapter._target_fs())
    if bool(cfg.get('apply_EA', False)):
        if ea_matrix is None:
            raise ValueError('CBraMod EA must be fitted on calibration before transform')
        x = _apply_ea(x, ea_matrix)
    x = adapter._normalize(x)
    x = (x / float(cfg['scale'])).astype(np.float32)
    if x.shape[-1] != 800:
        raise ValueError(f'CBraMod needs 4 x 200 samples after resampling, got {x.shape}')
    return x.reshape(len(x), x.shape[1], 4, 200)


def _split_subject(source, dataset: str, key: int, seed: int,
                   calibration_fraction: float, test_fraction: float):
    mask = np.asarray(source.subjects, dtype=np.int64) == int(key)
    indices = np.flatnonzero(mask)
    y_subject = np.asarray(source.y[mask], dtype=np.int64)
    if len(indices) < 2 or len(np.unique(y_subject)) != int(
            config.load_dataset_config(dataset)['num_classes']):
        raise ValueError(f'{dataset} subject={key + 1}: incomplete class/trial coverage')
    if not np.isclose(calibration_fraction + test_fraction, 1.0):
        raise ValueError('calibration and test fractions must sum to one')
    if not np.isclose(test_fraction, 0.7):
        raise ValueError('this frozen experiment requires a 30/70 calibration/test split')
    local_train, local_test = split_utils.split_indices_with_val_ratio(
        np.arange(len(indices)), y_subject, test_fraction, int(seed))
    global_train = indices[np.asarray(local_train, dtype=np.int64)]
    global_test = indices[np.asarray(local_test, dtype=np.int64)]
    return {
        'x_cal_raw': np.asarray(source.x[global_train], dtype=np.float32),
        'y_cal': np.asarray(source.y[global_train], dtype=np.int64),
        'uid_cal': np.asarray(source.sample_uid[global_train], dtype=np.int64),
        'trial_uid_cal': np.asarray(source.trial_uid[global_train]).astype(str),
        'x_test_raw': np.asarray(source.x[global_test], dtype=np.float32),
        'y_test': np.asarray(source.y[global_test], dtype=np.int64),
        'uid_test': np.asarray(source.sample_uid[global_test], dtype=np.int64),
        'trial_uid_test': np.asarray(source.trial_uid[global_test]).astype(str),
    }


def _cache_path(root: Path, kind: str, dataset: str, model: str,
                key: int, seed: int, identity: dict[str, Any]) -> Path:
    identity_hash = _sha256_json(identity)
    return (root / kind / dataset / model / f'subject_{key + 1:02d}' /
            f'seed_{seed}' / identity_hash[:20])


def _read_cache(directory: Path, identity: dict[str, Any], stem: str):
    manifest_path = directory / f'{stem}.manifest.json'
    array_path = directory / f'{stem}.npz'
    if not manifest_path.is_file() or not array_path.is_file():
        return None, 'cache files missing'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('identity') != identity:
            return None, 'manifest identity differs'
        if _sha256_file(array_path) != manifest.get('arrays_sha256'):
            return None, 'array SHA-256 differs from manifest'
        manifest['manifest_path'] = str(manifest_path)
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        return (arrays, manifest), ''
    except Exception as exc:  # noqa: BLE001 - corrupt derived caches are rebuildable
        return None, f'{type(exc).__name__}: {exc}'


def _archive_invalid(directory: Path, stem: str) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for suffix in ('.npz', '.manifest.json'):
        path = directory / f'{stem}{suffix}'
        if path.exists():
            path.rename(path.with_name(f'{stem}.invalid_{stamp}{suffix}'))


def _write_cache(directory: Path, stem: str, identity: dict[str, Any],
                 arrays: dict[str, Any], extra_manifest: dict[str, Any]):
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f'{stem}.npz'
    _atomic_npz(array_path, **arrays)
    manifest = {
        'schema_version': 1, 'kind': stem, 'identity': identity,
        'identity_sha256': _sha256_json(identity),
        'arrays_sha256': _sha256_file(array_path),
        'arrays': {name: {'shape': list(np.asarray(value).shape),
                          'dtype': str(np.asarray(value).dtype)}
                   for name, value in arrays.items()},
        **extra_manifest,
    }
    _atomic_json(directory / f'{stem}.manifest.json', manifest)
    manifest['manifest_path'] = str(directory / f'{stem}.manifest.json')
    return manifest


def _prepare_model_inputs(resolved: dict[str, Any], dataset: str, teacher: str,
                          student: str, key: int, seed: int, source,
                          *, epochs_override: int | None = None,
                          teacher_epochs_override: int | None = None,
                          logger=print):
    ds_cfg = resolved['dataset_configs'][dataset]
    raw_unit = _split_subject(
        source, dataset, key, seed,
        resolved['config']['split']['calibration_fraction'],
        resolved['config']['split']['test_fraction'])
    split_hash = _sha256_json({
        'train_uid': _sha256_array(raw_unit['uid_cal']),
        'test_uid': _sha256_array(raw_unit['uid_test']),
    })
    source_hash = (source.source_manifest_sha256 or
                   source.identity.get('source_identity_sha256', ''))
    cache_root = Path(resolved['storage']['cache_root'])
    code_sha = resolved['code_sha256']
    input_bundles = {}
    input_manifests = {}
    for model, model_cfg_map in ((teacher, resolved['teacher_configs']),
                                 (student, resolved['student_configs'])):
        key_name = f'{dataset}/{model}'
        model_cfg = copy.deepcopy(model_cfg_map[key_name])
        if epochs_override is not None:
            model_cfg['epochs'] = int(teacher_epochs_override if model == teacher
                                      else epochs_override)
        model_cfg['dataset_name'] = dataset
        if model == 'mirepnet':
            model_cfg['in_channels'] = 45
            model_cfg['samples'] = int(raw_unit['x_cal_raw'].shape[-1])
            adapter = get_adapter(model, device='cpu', **model_cfg)
            x_cal, x_test, ea_hash = _prepare_mirepnet_input(
                dataset, raw_unit['x_cal_raw'], raw_unit['x_test_raw'], adapter)
            preprocess_id = {
                'policy': 'fit_EA_on_calibration_then_apply_to_calibration_and_test',
                'channel_mapping': 'current MIRepNet inverse-distance 45-channel map',
                'ea_matrix_sha256': ea_hash,
            }
        elif model == 'cbramod':
            model_cfg['in_channels'] = int(raw_unit['x_cal_raw'].shape[1])
            model_cfg['samples'] = 1000
            adapter = get_adapter(model, device='cpu', **model_cfg)
            ea_matrix = None
            if bool(model_cfg.get('apply_EA', False)):
                # Estimate the reference only from filtered calibration data.
                pre = adapter._resample_to_target(
                    raw_unit['x_cal_raw'][:, :, :1000], adapter._target_fs())
                pre = adapter._apply_frequency_filters(pre, adapter._target_fs())
                ea_matrix = _fit_ea(pre)
            x_cal = _prepare_cbramod_input(raw_unit['x_cal_raw'], model_cfg,
                                           adapter, ea_matrix)
            x_test = _prepare_cbramod_input(raw_unit['x_test_raw'], model_cfg,
                                            adapter, ea_matrix)
            preprocess_id = {
                'policy': 'current CBraMod fewshot resample/filter/CAR/scale pipeline',
                'apply_EA': bool(model_cfg.get('apply_EA', False)),
                'ea_matrix_sha256': _sha256_array(ea_matrix) if ea_matrix is not None else None,
            }
        else:
            model_cfg['in_channels'] = int(raw_unit['x_cal_raw'].shape[1])
            model_cfg['samples'] = int(raw_unit['x_cal_raw'].shape[-1])
            adapter = get_adapter(model, device='cpu', **model_cfg)
            x_cal = adapter.preprocess(raw_unit['x_cal_raw']).cpu().numpy()
            x_test = adapter.preprocess(raw_unit['x_test_raw']).cpu().numpy()
            preprocess_id = {
                'policy': ('current IFNet fixed 4-16/16-40 Hz filterbank'
                           if model == 'ifnet' and model_cfg.get('use_filter_bank')
                           else 'current fewshot student adapter preprocessing'),
                'use_filter_bank': bool(model_cfg.get('use_filter_bank', False)),
                'filter_bank_hz': model_cfg.get('filter_bank_hz', [[4.0, 16.0], [16.0, 40.0]])
                                  if model == 'ifnet' and model_cfg.get('use_filter_bank') else None,
            }
        x_cal = np.asarray(x_cal, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
        model_cfg['input_shape_calibration'] = list(x_cal.shape[1:])
        identity = {
            'dataset': dataset, 'subject_index_zero_based': int(key), 'seed': int(seed),
            'protocol': 'fewshot', 'split_policy': 'fewshot_stratified_random',
            'calibration_fraction': 0.30, 'test_fraction': 0.70,
            'train_uid_sha256': _sha256_array(raw_unit['uid_cal']),
            'test_uid_sha256': _sha256_array(raw_unit['uid_test']),
            'split_uid_sha256': split_hash,
            'source_identity_sha256': source_hash,
            'source_path': str(source.source_path),
            'source_variant': source.identity.get('source_variant',
                                                  'wide_fre_band' if dataset != 'AlexMI'
                                                  else 'legacy_alexmi_verified'),
            'model': model, 'model_config': model_cfg,
            'preprocessing': preprocess_id,
            'code_sha256': code_sha,
        }
        directory = _cache_path(cache_root, 'model_inputs', dataset, model,
                                key, seed, identity)
        stem = 'model_inputs'
        with _cache_lock(directory / '.lock'):
            cached, reason = _read_cache(directory, identity, stem)
            if cached is None:
                logger(f'[cache] prepare model_inputs {dataset}/{model} '
                       f'S{key + 1:02d}/seed={seed}; reason={reason or "not found"}')
                if (directory / f'{stem}.manifest.json').exists() or (directory / f'{stem}.npz').exists():
                    _archive_invalid(directory, stem)
                arrays = {
                    'X_cal': x_cal, 'y_cal': raw_unit['y_cal'],
                    'sample_uid_cal': raw_unit['uid_cal'],
                    'trial_uid_cal': raw_unit['trial_uid_cal'],
                    'X_test': x_test, 'y_test': raw_unit['y_test'],
                    'sample_uid_test': raw_unit['uid_test'],
                    'trial_uid_test': raw_unit['trial_uid_test'],
                }
                manifest = _write_cache(directory, stem, identity, arrays, {
                    'dataset': dataset, 'model': model,
                    'subject_index_zero_based': int(key), 'seed': int(seed),
                    'session': ds_cfg['session'], 'num_classes': ds_cfg['num_classes'],
                    'source_manifest_sha256': source_hash,
                    'input_policy': preprocess_id,
                })
                cached = (arrays, manifest)
            else:
                logger(f'[cache] reuse model_inputs {dataset}/{model} '
                       f'S{key + 1:02d}/seed={seed} sha256={cached[1]["arrays_sha256"]}')
        input_bundles[model] = cached[0]
        input_manifests[model] = cached[1]
    return raw_unit, input_bundles, input_manifests


def _teacher_cache_identity(resolved: dict[str, Any], dataset: str, teacher: str,
                            key: int, seed: int, input_manifest: dict[str, Any],
                            pretrain_path: Path, *, teacher_epochs_override=None):
    teacher_cfg = copy.deepcopy(resolved['teacher_configs'][f'{dataset}/{teacher}'])
    if teacher_epochs_override is not None:
        teacher_cfg['epochs'] = int(teacher_epochs_override)
    return {
        'dataset': dataset, 'teacher': teacher,
        'subject_index_zero_based': int(key), 'seed': int(seed),
        'protocol': 'fewshot', 'split_policy': 'fewshot_stratified_random',
        'calibration_fraction': 0.30, 'test_fraction': 0.70,
        'train_uid_sha256': input_manifest['identity']['train_uid_sha256'],
        'test_uid_sha256': input_manifest['identity']['test_uid_sha256'],
        'source_identity_sha256': input_manifest['identity']['source_identity_sha256'],
        'model_input_manifest_sha256': _sha256_file(
            Path(input_manifest['manifest_path'])),
        'teacher_config': teacher_cfg,
        'pretrained_weight_path': str(pretrain_path),
        'pretrained_weight_sha256': _sha256_file(pretrain_path),
        'code_sha256': resolved['code_sha256'],
    }


def _ensure_teacher_targets(resolved: dict[str, Any], dataset: str, teacher: str,
                            key: int, seed: int, inputs: dict[str, np.ndarray],
                            input_manifest: dict[str, Any], num_classes: int,
                            device: torch.device, *, teacher_epochs_override=None,
                            logger=print):
    configured_weight = config.weight_path(teacher)
    pretrain_path = Path(resolve_local_file(configured_weight)).resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(f'{teacher}: official pretrained weight missing: {pretrain_path}')
    identity = _teacher_cache_identity(
        resolved, dataset, teacher, key, seed, input_manifest, pretrain_path,
        teacher_epochs_override=teacher_epochs_override)
    root = Path(resolved['storage']['cache_root'])
    directory = _cache_path(root, 'teacher_targets', dataset, teacher,
                            key, seed, identity)
    target_stem, eval_stem = 'teacher_targets', 'teacher_eval_predictions'
    with _cache_lock(directory / '.lock'):
        target_cache, reason = _read_cache(directory, identity, target_stem)
        eval_cache, eval_reason = _read_cache(directory, identity, eval_stem)
        if target_cache is not None and eval_cache is not None:
            manifest = target_cache[1]
            logger(f'[cache] reuse teacher_targets {dataset}/{teacher} '
                   f'S{key + 1:02d}/seed={seed} '
                   f'sha256={manifest["arrays_sha256"]}')
            manifest['cache_action'] = 'reused'
            return target_cache[0], eval_cache[0], manifest
        if target_cache is not None or eval_cache is not None:
            reason = f'target={reason or "valid"}; eval={eval_reason or "valid"}'
        logger(f'[teacher-cache] fit {dataset}/{teacher} S{key + 1:02d}/seed={seed}; '
               f'reason={reason or eval_reason or "not found"}; '
               f'epochs={identity["teacher_config"].get("epochs")}')
        for stem in (target_stem, eval_stem):
            if ((directory / f'{stem}.manifest.json').exists() or
                    (directory / f'{stem}.npz').exists()):
                _archive_invalid(directory, stem)

        cfg = copy.deepcopy(identity['teacher_config'])
        cfg.update(dataset_name=dataset, skip_preprocess=True,
                   in_channels=(45 if teacher == 'mirepnet'
                                else int(inputs[teacher]['X_cal'].shape[1])),
                   samples=1000)
        teacher_adapter = get_adapter(teacher, device=str(device), **cfg)
        set_seed(int(seed))
        model = teacher_adapter.build(int(num_classes))
        model = teacher_adapter.finetune(
            model, inputs[teacher]['X_cal'], inputs[teacher]['y_cal'],
            int(num_classes))
        model.eval()
        train_features, train_logits = teacher_adapter.infer(
            model, inputs[teacher]['X_cal'])
        # Test inference is evaluation-only. Only class predictions are kept;
        # no test logits/features enter the teacher-target cache.
        _, test_logits = teacher_adapter.infer(model, inputs[teacher]['X_test'])
        test_pred = np.asarray(test_logits).argmax(axis=1).astype(np.int64)
        y_test = np.asarray(inputs[teacher]['y_test'], dtype=np.int64)
        teacher_eval = {
            'uid': np.asarray(inputs[teacher]['sample_uid_test'], dtype=np.int64),
            'y_true': y_test,
            'y_pred': test_pred,
        }
        teacher_state_hash = _sha256_state_dict(model.state_dict())
        del model, teacher_adapter
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        target_arrays = {
            'features': np.asarray(train_features, dtype=np.float32),
            'logits': np.asarray(train_logits, dtype=np.float32),
            'y': np.asarray(inputs[teacher]['y_cal'], dtype=np.int64),
            'sample_uid': np.asarray(inputs[teacher]['sample_uid_cal'], dtype=np.int64),
        }
        if (not np.array_equal(target_arrays['y'], inputs[teacher]['y_cal']) or
                not np.array_equal(target_arrays['sample_uid'], inputs[teacher]['sample_uid_cal'])):
            raise ValueError('teacher targets lost calibration UID/label alignment')
        target_manifest = _write_cache(directory, target_stem, identity,
                                       target_arrays, {
            'dataset': dataset, 'teacher': teacher,
            'subject_index_zero_based': int(key), 'seed': int(seed),
            'target_scope': 'calibration_only',
            'num_classes': int(num_classes),
            'teacher_checkpoint_sha256': teacher_state_hash,
            'pretrained_weight_path': str(pretrain_path),
            'pretrained_weight_sha256': identity['pretrained_weight_sha256'],
            'test_prediction_cache': eval_stem,
            'teacher_test_accuracy_pct': float(accuracy_score(y_test, test_pred) * 100.0),
            'teacher_test_balanced_accuracy_pct': float(
                balanced_accuracy_score(y_test, test_pred) * 100.0),
            'teacher_test_kappa': float(cohen_kappa_score(y_test, test_pred)),
        })
        eval_manifest = _write_cache(directory, eval_stem, identity,
                                     teacher_eval, {
            'dataset': dataset, 'teacher': teacher,
            'subject_index_zero_based': int(key), 'seed': int(seed),
            'scope': 'final_evaluation_predictions_only',
            'teacher_checkpoint_sha256': teacher_state_hash,
        })
        target_manifest['manifest_path'] = str(directory / f'{target_stem}.manifest.json')
        target_manifest['eval_manifest_sha256'] = _sha256_file(
            directory / f'{eval_stem}.manifest.json')
        _atomic_json(directory / f'{target_stem}.manifest.json', target_manifest)
        target_arrays, _ = _read_cache(directory, identity, target_stem)
        eval_arrays, _ = _read_cache(directory, identity, eval_stem)
        if target_arrays is None or eval_arrays is None:
            raise RuntimeError('new teacher cache failed its post-write validation')
        logger(f'[teacher-cache] ready {dataset}/{teacher} S{key + 1:02d}/seed={seed} '
               f'checkpoint_sha256={teacher_state_hash}')
        target_manifest['cache_action'] = 'fit'
        return target_arrays[0], eval_arrays[0], target_manifest


class _PreparedStudentAdapter:
    """Use identity preprocessing over the validated, already prepared cache."""
    def __init__(self, base_adapter):
        self.base = base_adapter
        self.device = base_adapter.device
        self.cfg = base_adapter.cfg

    @staticmethod
    def preprocess(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def build(self, num_classes):
        return self.base.build(num_classes)

    def forward(self, model, x):
        return self.base.forward(model, x)

    @torch.no_grad()
    def infer(self, model, x):
        value = self.preprocess(x).to(self.device)
        batch_size = int(self.cfg.get('batch_size', 32))
        model.eval()
        features, logits = [], []
        for start in range(0, len(value), batch_size):
            feature, logit = self.forward(model, value[start:start + batch_size])
            features.append(feature.detach().cpu().numpy())
            logits.append(logit.detach().cpu().numpy())
        return np.concatenate(features), np.concatenate(logits)


def _initial_student_state(adapter: _PreparedStudentAdapter, num_classes: int,
                           seed: int) -> tuple[dict[str, torch.Tensor], str]:
    set_seed(int(seed))
    model = adapter.build(int(num_classes))
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    digest = _sha256_state_dict(state)
    del model
    return state, digest


def _projection_seed(dataset: str, teacher: str, student: str,
                     key: int, seed: int) -> int:
    raw = f'{dataset}|{teacher}|{student}|{key}|{seed}|kd_proto'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')


def _method_kwargs(method: str, pair_params: dict[str, Any],
                   method_params: dict[str, Any], seed: int,
                   dataset: str, teacher: str, student: str,
                   key: int) -> dict[str, Any]:
    common = {
        'lam_kd': float(pair_params['lam_kd']),
        'temperature': float(pair_params['temperature']),
        'teacher_correct_only': False,
        'seed': int(seed),
        'return_training_details': True,
        'check_gradients': True,
    }
    if method == 'kd_proto':
        return {**common, 'feat_proto': True,
                'lam_feat': float(method_params['lambda_proto']),
                'projection_seed': _projection_seed(dataset, teacher, student,
                                                    key, seed)}
    if method == 'kd_relation':
        return {**common, 'relational': {
            'mode': 'intra_inter',
            'lam_intra': float(method_params['lambda_intra']),
            'lam_inter': float(method_params['lambda_inter']),
        }}
    if method == 'ea_kd':
        return {**common, 'ea_kd': True,
                'ea_temp': float(method_params['entropy_temperature'])}
    raise ValueError(f'unsupported method: {method}')


def _batch_order_sha256(details: dict[str, Any]) -> str:
    return _sha256_bytes('\n'.join(details.get('batch_order_hashes', [])).encode())


def _check_uid_alignment(raw: dict[str, Any], inputs: dict[str, np.ndarray],
                         teacher_targets: dict[str, np.ndarray]) -> None:
    for name, side in (('calibration', 'cal'), ('test', 'test')):
        if not np.array_equal(raw[f'uid_{side}'], inputs[f'sample_uid_{side}']):
            raise ValueError(f'{name} model inputs differ from the frozen split UID order')
    if not np.array_equal(raw['uid_cal'], teacher_targets['sample_uid']):
        raise ValueError('teacher target UID set/order differs from calibration split')
    if not np.array_equal(raw['y_cal'], teacher_targets['y']):
        raise ValueError('teacher target labels differ from calibration split')


def _write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS,
                                    extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, '') for key in METRIC_FIELDS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('dataset', 'subject', 'subject_index', 'seed', 'teacher', 'student',
              'method', 'trial_uid', 'uid_subject_index', 'uid_trial_index',
              'y_true', 'y_pred', 'correct')
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                uids = json.loads(row.get('test_uid_json') or '[]')
                trial_uids = json.loads(row.get('test_trial_uid_json') or '[]')
                labels = json.loads(row.get('test_y_json') or '[]')
                preds = json.loads(row.get('test_pred_json') or '[]')
                if not (len(uids) == len(trial_uids) == len(labels) == len(preds)):
                    raise ValueError('prediction arrays do not align to test UIDs')
                for uid, trial_uid, label, pred in zip(uids, trial_uids, labels, preds):
                    writer.writerow({
                        'dataset': row['dataset'], 'subject': row['subject'],
                        'subject_index': row['subject_index'], 'seed': row['seed'],
                        'teacher': row['teacher'], 'student': row['student'],
                        'method': row['method'], 'trial_uid': trial_uid,
                        'uid_subject_index': uid[0], 'uid_trial_index': uid[1],
                        'y_true': label, 'y_pred': pred,
                        'correct': int(int(label) == int(pred)),
                    })
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stop_requested(path: str | None) -> bool:
    return bool(path and Path(path).exists())


def _train_cell(resolved: dict[str, Any], dataset: str, teacher: str,
                student: str, key: int, seed: int, method: str,
                raw: dict[str, Any], inputs: dict[str, np.ndarray],
                input_manifests: dict[str, dict[str, Any]],
                teacher_targets: dict[str, np.ndarray],
                teacher_eval: dict[str, np.ndarray],
                teacher_manifest: dict[str, Any], device: torch.device,
                *, student_epochs_override: int | None = None,
                logger=print) -> dict[str, Any]:
    _check_uid_alignment(raw, inputs[student], teacher_targets)
    dcfg = resolved['dataset_configs'][dataset]
    scfg = copy.deepcopy(resolved['student_configs'][f'{dataset}/{student}'])
    if student_epochs_override is not None:
        scfg['epochs'] = int(student_epochs_override)
    scfg.update(in_channels=int(raw['x_cal_raw'].shape[1]),
                samples=int(raw['x_cal_raw'].shape[-1]), dataset_name=dataset)
    base_adapter = get_adapter(student, device=str(device), **scfg)
    adapter = _PreparedStudentAdapter(base_adapter)
    initial_state, initial_hash = _initial_student_state(
        adapter, int(dcfg['num_classes']), int(seed))
    pair = resolved['kd_parameters'][teacher][student]
    method_params = resolved['method_parameters'][method]
    kwargs = _method_kwargs(method, pair, method_params, seed,
                            dataset, teacher, student, key)
    started = time.monotonic()
    y_pred, details = distill.distill_student(
        adapter, int(dcfg['num_classes']), inputs[student]['X_cal'],
        inputs[student]['y_cal'], teacher_targets['features'],
        teacher_targets['logits'], inputs[student]['X_test'],
        epochs=int(scfg['epochs']), lr=float(scfg.get('lr', 0.001)),
        weight_decay=float(scfg.get('weight_decay', 0.0001)),
        batch_size=int(scfg.get('batch_size', 32)),
        initial_state_dict=initial_state, sample_uid=inputs[student]['sample_uid_cal'],
        **kwargs)
    runtime = time.monotonic() - started
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(inputs[student]['y_test'], dtype=np.int64)
    if len(y_pred) != len(y_true):
        raise ValueError('student prediction count differs from test labels')
    history = details['training_history']
    if not history:
        raise RuntimeError('distillation returned no training history')
    final = history[-1]
    eval_arrays = teacher_eval
    if not np.array_equal(eval_arrays['uid'], inputs[student]['sample_uid_test']):
        raise ValueError('teacher final-evaluation UID order differs from student test split')
    ea_history = details.get('ea_kd_history', [])
    flat_relation = details.get('relational_history', [])
    teacher_eval_y = np.asarray(eval_arrays['y_true'], dtype=np.int64)
    teacher_eval_pred = np.asarray(eval_arrays['y_pred'], dtype=np.int64)
    test_uid = np.asarray(inputs[student]['sample_uid_test'], dtype=np.int64)
    raw_source = raw['trial_uid_test']
    row = {
        'dataset': dataset, 'subject': int(key) + 1,
        'subject_index': int(key), 'session': dcfg['session'],
        'protocol': 'fewshot', 'seed': int(seed), 'teacher': teacher,
        'student': student, 'method': method, 'status': 'complete',
        'accuracy_pct': float(accuracy_score(y_true, y_pred) * 100.0),
        'balanced_accuracy_pct': float(balanced_accuracy_score(y_true, y_pred) * 100.0),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'n_calibration': int(len(inputs[student]['y_cal'])), 'n_test': int(len(y_true)),
        'num_classes': int(dcfg['num_classes']), 'epochs': int(scfg['epochs']),
        'lr': float(scfg.get('lr', 0.001)),
        'batch_size': int(scfg.get('batch_size', 32)),
        'weight_decay': float(scfg.get('weight_decay', 0.0001)),
        'optimizer': str(scfg.get('optimizer', 'adamw')),
        'scheduler': 'CosineAnnealingLR stepped once at epoch end',
        'warmup_epochs': int(scfg.get('warmup_epochs', 0)),
        'lambda_kd': float(pair['lam_kd']),
        'temperature': float(pair['temperature']),
        'lambda_proto': float(method_params['lambda_proto']) if method == 'kd_proto' else '',
        'lambda_relation': float(method_params['lambda_relation']) if method == 'kd_relation' else '',
        'lambda_intra': float(method_params['lambda_intra']) if method == 'kd_relation' else '',
        'lambda_inter': float(method_params['lambda_inter']) if method == 'kd_relation' else '',
        'entropy_temperature': (float(method_params['entropy_temperature'])
                                if method == 'ea_kd' else ''),
        'teacher_accuracy_pct': float(accuracy_score(teacher_eval_y, teacher_eval_pred) * 100.0),
        'teacher_balanced_accuracy_pct': float(
            balanced_accuracy_score(teacher_eval_y, teacher_eval_pred) * 100.0),
        'teacher_kappa': float(cohen_kappa_score(teacher_eval_y, teacher_eval_pred)),
        'teacher_cache_action': str(teacher_manifest.get('cache_action', 'unknown')),
        'teacher_checkpoint_sha256': teacher_manifest.get('teacher_checkpoint_sha256', ''),
        'teacher_target_manifest_sha256': _sha256_file(
            Path(teacher_manifest['manifest_path'])),
        'model_input_manifest_sha256': _sha256_file(
            Path(input_manifests[student]['manifest_path'])),
        'train_uid_sha256': _sha256_array(inputs[student]['sample_uid_cal']),
        'test_uid_sha256': _sha256_array(test_uid),
        'split_uid_sha256': _sha256_json({
            'train': _sha256_array(inputs[student]['sample_uid_cal']),
            'test': _sha256_array(test_uid)}),
        'initial_state_sha256': initial_hash,
        'batch_order_sha256': _batch_order_sha256(details),
        'final_total_loss': float(final['total_loss']),
        'final_ce_loss': float(final['ce_loss']),
        'final_train_accuracy_pct': float(final['train_accuracy']),
        'gradient_batches_checked': int(details.get('gradient_batches_checked', 0)),
        'gradient_checks_passed': bool(details.get('gradient_checks_passed', False)),
        'relation_history_json': _canonical_json(flat_relation),
        'ea_kd_history_json': _canonical_json(ea_history),
        'runtime_seconds': float(runtime),
        'test_uid_json': _canonical_json(test_uid.tolist()),
        'test_trial_uid_json': _canonical_json(raw_source.tolist()),
        'test_y_json': _canonical_json(y_true.tolist()),
        'test_pred_json': _canonical_json(y_pred.tolist()),
    }
    if not (np.isfinite(row['accuracy_pct']) and np.isfinite(row['balanced_accuracy_pct'])
            and np.isfinite(row['kappa']) and np.isfinite(row['final_total_loss'])):
        raise FloatingPointError('non-finite metric/loss in completed student cell')
    return row


def _run_task(resolved: dict[str, Any], task: dict[str, Any], device: torch.device,
              progress_path: str | Path, output_csv: str | Path,
              stop_file: str | None = None, *, smoke_epochs: int | None = None,
              teacher_epochs_override: int | None = None,
              student_epochs_override: int | None = None,
              logger=print) -> int:
    dataset, teacher, student = task['dataset'], task['teacher'], task['student']
    seeds, keys, methods = task['seeds'], task['keys'], task['methods']
    spec = {'resolved_config': resolved, 'task': task,
            'code_sha256': resolved['code_sha256']}
    total = len(seeds) * len(keys) * len(methods)
    results_path = require_report_output(output_csv)
    prediction_path = results_path.with_name('predictions.csv')
    with ProgressStore(progress_path, spec) as progress:
        progress.set_status('running', total_cells=total)
        rows = progress.all_rows()
        for seed in seeds:
            for key in keys:
                if _stop_requested(stop_file):
                    progress.set_status('paused')
                    return 75
                # The raw model-agnostic reader is the only BNCI input route.
                source = loso_source.load_dataset(dataset, model=None)
                raw, inputs, input_manifests = _prepare_model_inputs(
                    resolved, dataset, teacher, student, int(key), int(seed), source,
                    epochs_override=smoke_epochs,
                    teacher_epochs_override=teacher_epochs_override,
                    logger=logger)
                target_data, eval_data, teacher_manifest = _ensure_teacher_targets(
                    resolved, dataset, teacher, int(key), int(seed), inputs,
                    input_manifests[teacher], int(resolved['dataset_configs'][dataset]['num_classes']),
                    device, teacher_epochs_override=teacher_epochs_override,
                    logger=logger)
                for method in methods:
                    if _stop_requested(stop_file):
                        progress.set_status('paused')
                        return 75
                    cell = [dataset, teacher, student, int(seed), int(key), method]
                    identity = {
                        'dataset': dataset, 'teacher': teacher, 'student': student,
                        'seed': int(seed), 'subject_index_zero_based': int(key),
                        'method': method,
                        'student_config': resolved['student_configs'][f'{dataset}/{student}'],
                        'kd_parameters': resolved['kd_parameters'][teacher][student],
                        'method_parameters': resolved['method_parameters'][method],
                        'student_input_manifest_sha256': _sha256_file(
                            Path(input_manifests[student]['manifest_path'])),
                        'teacher_input_manifest_sha256': _sha256_file(
                            Path(input_manifests[teacher]['manifest_path'])),
                        'teacher_target_manifest_sha256': _sha256_file(
                            Path(teacher_manifest['manifest_path'])),
                        'split_uid_sha256': _sha256_json({
                            'train': _sha256_array(inputs[student]['sample_uid_cal']),
                            'test': _sha256_array(inputs[student]['sample_uid_test'])}),
                        'smoke_epochs': int(smoke_epochs) if smoke_epochs else None,
                    }
                    try:
                        completed = progress.get(cell, identity)
                    except ValueError as exc:
                        raise RuntimeError(f'resume identity mismatch for {cell}: {exc}') from exc
                    if completed is not None:
                        logger(f'[resume] reuse {dataset} {teacher}->{student} '
                               f'S{key + 1:02d}/seed={seed}/{method}')
                        continue
                    logger(f'[train] start {dataset} {teacher}->{student} '
                           f'S{key + 1:02d}/seed={seed}/{method} device={device}')
                    row = _train_cell(
                        resolved, dataset, teacher, student, int(key), int(seed),
                        method, raw, inputs, input_manifests,
                        target_data, eval_data, teacher_manifest, device,
                        student_epochs_override=student_epochs_override,
                        logger=logger)
                    row['predictions_path'] = str(prediction_path)
                    row['prediction_count'] = int(row['n_test'])
                    progress.put(cell, identity, row)
                    rows = progress.all_rows()
                    _write_metric_csv(results_path, rows)
                    _write_predictions(prediction_path, rows)
                    logger(f'[train] done {dataset} {teacher}->{student} '
                           f'S{key + 1:02d}/seed={seed}/{method} '
                           f'Accuracy={row["accuracy_pct"]:.2f}% '
                           f'BA={row["balanced_accuracy_pct"]:.2f}% '
                           f'{row["runtime_seconds"]:.1f}s')
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
        progress.set_status('completed')
    logger(f'[done] {results_path} cells={len(rows)}/{total}')
    return 0


def _task_snapshot(resolved: dict[str, Any], dataset: str, teacher: str,
                   student: str, *, methods=None, seeds=None, keys=None) -> dict[str, Any]:
    ds_cfg = resolved['dataset_configs'][dataset]
    task = {
        'dataset': dataset, 'teacher': teacher, 'student': student,
        'methods': list(methods or resolved['scope']['methods']),
        'seeds': list(seeds or resolved['scope']['seeds']),
        'keys': list(keys or ds_cfg['keys']),
    }
    return {
        'schema_version': 1,
        'kind': 'fewshot_structure_kd',
        'run_id': resolved['run_id'],
        'output_dir': resolved['storage']['output_dir'],
        'config_path': resolved['source_config'],
        'config_sha256': resolved['source_config_sha256'],
        'resolved_config': resolved,
        'task': task,
        'args': {
            'protocol': 'fewshot', 'dataset': dataset,
            'datasets': [dataset], 'teacher': teacher,
            'teachers': [teacher], 'student': student,
            'students': [student], 'seeds': task['seeds'],
            'keys': task['keys'], 'methods': task['methods'],
            'epochs': resolved['student_configs'][f'{dataset}/{student}']['epochs'],
            'batch_size': resolved['student_configs'][f'{dataset}/{student}']['batch_size'],
            'lr': resolved['student_configs'][f'{dataset}/{student}']['lr'],
            'weight_decay': resolved['student_configs'][f'{dataset}/{student}']['weight_decay'],
        },
    }


def enqueue_cli(args) -> int:
    from experiments.distill.queue import enqueue_snapshot
    resolved = resolve_experiment(
        args.config, datasets=args.datasets, teachers=args.teachers,
        students=args.students, seeds=args.seeds, methods=args.methods,
        subjects=args.subjects)
    write_resolved_config(resolved)
    print('[resolved] ' + json.dumps(resolved['scope'], ensure_ascii=False), flush=True)
    output_dir = Path(resolved['storage']['output_dir'])
    for dataset in resolved['scope']['datasets']:
        for teacher in resolved['scope']['teachers']:
            for student in resolved['scope']['students']:
                snapshot = _task_snapshot(resolved, dataset, teacher, student)
                pair_dir = output_dir / dataset / f'{teacher}_to_{student}'
                snapshot['output_csv'] = str(pair_dir / 'per_subject_seed.csv')
                snapshot['progress_path'] = str(pair_dir / 'progress.json')
                enqueue_snapshot(
                    snapshot, root=args.root,
                    argv=sys.argv[1:],
                    label=f'fewshot structure KD {dataset} {teacher}->{student}')
    print(f'[queue] registered output_dir={output_dir}', flush=True)
    return 0


def run_snapshot(snapshot_path: str | Path, *, progress_path: str | Path,
                 gpu: int | None, stop_file: str | None = None,
                 logger=print) -> int:
    path = Path(snapshot_path).resolve()
    wrapper = json.loads(path.read_text(encoding='utf-8'))
    snapshot = wrapper.get('snapshot', wrapper)
    if snapshot.get('schema_version') != 1 or snapshot.get('kind') != 'fewshot_structure_kd':
        raise ValueError('unsupported few-shot structure KD queue snapshot')
    if snapshot.get('config_sha256') != _sha256_file(snapshot['config_path']):
        raise ValueError('few-shot experiment YAML changed after queue submission')
    expected_hashes = snapshot['resolved_config']['code_sha256']
    for name, expected in expected_hashes.items():
        if _sha256_file(PROJECT_ROOT / name) != expected:
            raise ValueError(f'training code changed since enqueue: {name}')
    _set_runtime_threads()
    device = _device(gpu)
    task = snapshot['task']
    output_csv = require_report_output(snapshot['output_csv'])
    return _run_task(snapshot['resolved_config'], task, device,
                     progress_path, output_csv, stop_file,
                     logger=logger)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--teachers', nargs='+')
    parser.add_argument('--students', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--subjects', nargs='+', type=int,
                        help='zero-based subject IDs; default all subjects')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--snapshot', help=argparse.SUPPRESS)
    parser.add_argument('--progress', help=argparse.SUPPRESS)
    parser.add_argument('--stop-file', help=argparse.SUPPRESS)
    parser.add_argument('--output-csv', help=argparse.SUPPRESS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, default=1)
    parser.add_argument('--resolved-only', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _set_runtime_threads()
    if args.snapshot:
        try:
            return run_snapshot(args.snapshot, progress_path=args.progress,
                                gpu=args.gpu, stop_file=args.stop_file)
        except BaseException:
            traceback.print_exc()
            raise
    if args.smoke:
        run_id = 'smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        smoke_datasets = [(args.datasets or ['BNCI2014004'])[0]]
        smoke_teachers = list(args.teachers or ['mirepnet'])
        smoke_students = list(args.students or ['ifnet'])
        smoke_seeds = [int((args.seeds or [666])[0])]
        smoke_subjects = [int((args.subjects or [0])[0])]
        resolved = resolve_experiment(
            args.config, datasets=smoke_datasets,
            teachers=smoke_teachers, students=smoke_students,
            seeds=smoke_seeds, methods=args.methods or list(METHODS),
            subjects=smoke_subjects, run_id=run_id, smoke=True,
            smoke_epochs=args.smoke_epochs)
        write_resolved_config(resolved)
        ds = resolved['scope']['datasets'][0]
        key, seed = smoke_subjects[0], smoke_seeds[0]
        tasks = []
        for teacher in resolved['scope']['teachers']:
            for student in resolved['scope']['students']:
                snapshot = _task_snapshot(
                    resolved, ds, teacher, student,
                    methods=resolved['scope']['methods'], seeds=[seed], keys=[key])
                pair_dir = (Path(resolved['storage']['output_dir']) / ds /
                            f'{teacher}_to_{student}' / 'smoke')
                tasks.append((snapshot['task'], pair_dir / 'progress.json',
                              pair_dir / 'smoke_per_subject_seed.csv'))
        for task, progress_path, output_csv in tasks:
            result = _run_task(
                resolved, task, _device(args.gpu), progress_path, output_csv,
                student_epochs_override=args.smoke_epochs,
                teacher_epochs_override=args.smoke_epochs,
                smoke_epochs=args.smoke_epochs)
            if result != 0:
                return result
        print('[smoke-resume] rerunning selected teacher/student pairs to verify completed cells are reused',
              flush=True)
        for task, progress_path, output_csv in tasks:
            resume_result = _run_task(
                resolved, task, _device(args.gpu), progress_path, output_csv,
                student_epochs_override=args.smoke_epochs,
                teacher_epochs_override=args.smoke_epochs,
                smoke_epochs=args.smoke_epochs)
            if resume_result != 0:
                raise RuntimeError(f'smoke recovery check returned {resume_result}')
        print('[smoke-resume] passed: all completed method cells reused', flush=True)
        from experiments.distill.summarize_fewshot_structure_kd import summarize
        print('[smoke-summary] ' + json.dumps(
            summarize(resolved['storage']['output_dir']), ensure_ascii=False), flush=True)
        return 0
    if args.resolved_only:
        resolved = resolve_experiment(
            args.config, datasets=args.datasets, teachers=args.teachers,
            students=args.students, seeds=args.seeds, methods=args.methods,
            subjects=args.subjects)
        print(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False))
        return 0
    parser = argparse.ArgumentParser()
    parser.error('formal runs must be submitted through: '
                 'python -B -m experiments.distill.queue enqueue-fewshot-structure')


if __name__ == '__main__':
    raise SystemExit(main())
