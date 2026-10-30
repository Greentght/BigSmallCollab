"""Run STUDENT_AWARE_DIFFUSION_PRIOR_V1 on the audited 004 7/1/1 LOSO folds."""
from __future__ import annotations

# Set CPU limits before importing numerical backends (also enforced by queue.py).
import os
for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS',
                    'TORCH_NUM_THREADS'):
    os.environ[_thread_env] = '1'

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import random
import sys
import time
import traceback
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import pdist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, log_loss
from torch.nn import functional as F

from collab.student_feature_prior import (
    ConditionalNoisePredictor,
    apply_pca_whitener,
    apply_ridge_projection,
    fit_pca_whitener,
    fit_ridge_projection,
    gaussian_epsilon,
    gaussian_prior_fit,
    linear_ddpm_schedule,
    sds_direction,
    sds_surrogate,
)
from experiments.storage import (
    BNCI14004_SOURCE_ROOT,
    DATA_CACHE_ROOT,
    REPORTS_ROOT,
    RESULTS_ROOT,
    WEIGHTS_ROOT,
    require_external_output,
    require_report_output,
    resolve_local_file,
)
from models import get_adapter

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/experiments/loso/student_aware_diffusion_prior_v1.yaml'
REFERENCE_PROTOCOL = 'sample_utility_adaptive_weight_wideband_v1'
METHODS = {
    'A': 'STUDENT_CE',
    'B': 'KD_RAW',
    'C': 'KD_TEACHER_CONTINUE',
    'D': 'KD_AID_EEG',
    'E': 'KD_AID_KEEP',
    'F': 'KD_GAUSSIAN_PRIOR',
    'G': 'KD_DIFFUSION_PRIOR',
}
TEACHER_METHODS = ('C', 'D', 'E', 'F', 'G')
STUDENT_METHODS = ('B', 'C', 'D', 'E', 'F', 'G')
BOOTSTRAP_DRAWS = 10000


def _validate_v1_config(config: dict) -> None:
    """Fail closed if the committed, preregistered V1 constants are changed."""
    expected = {
        ('experiment_id',): 'STUDENT_AWARE_DIFFUSION_PRIOR_V1',
        ('dataset',): 'BNCI2014004', ('selected_session',): 'session_3',
        ('source_variant',): 'wide_fre_band', ('seed',): 666,
        ('student', 'architecture'): 'ifnet', ('student', 'epochs'): 100,
        ('student', 'kd_weight'): 0.5, ('student', 'temperature'): 2.0,
        ('teacher_adaptation', 'epochs'): 10, ('teacher_adaptation', 'batch_size'): 8,
        ('teacher_adaptation', 'optimizer'): 'adam',
        ('teacher_adaptation', 'lr'): 0.0001,
        ('teacher_adaptation', 'betas'): [0.9, 0.999],
        ('teacher_adaptation', 'weight_decay'): 0.000001,
        ('teacher_adaptation', 'scheduler'): 'none',
        ('teacher_adaptation', 'gradient_clip_norm'): 1.0,
        ('teacher_adaptation', 'keep_weight'): 1.0,
        ('teacher_adaptation', 'aid_weight'): 0.5,
        ('teacher_adaptation', 'aid_temperature'): 2.0,
        ('teacher_adaptation', 'keep_temperature'): 2.0,
        ('feature_space', 'pca_max_dimensions'): 32,
        ('feature_space', 'numerical_rank_relative_threshold'): 0.00001,
        ('feature_space', 'whitening_eigenvalue_floor_relative'): 0.0001,
        ('feature_space', 'teacher_standard_deviation_floor'): 0.000001,
        ('feature_space', 'ridge_rho'): 0.01,
        ('gaussian_prior', 'covariance'): 'diagonal',
        ('gaussian_prior', 'variance_floor'): 0.01,
        ('diffusion_prior', 'steps'): 1000,
        ('diffusion_prior', 'beta_start'): 0.0001,
        ('diffusion_prior', 'beta_end'): 0.02,
        ('diffusion_prior', 'training_updates'): 5000,
        ('diffusion_prior', 'batch_size_cap'): 128,
        ('diffusion_prior', 'optimizer'): 'adamw',
        ('diffusion_prior', 'lr'): 0.0003,
        ('diffusion_prior', 'betas'): [0.9, 0.999],
        ('diffusion_prior', 'weight_decay'): 0.0001,
        ('diffusion_prior', 'gradient_clip_norm'): 1.0,
        ('diffusion_prior', 'hidden_dimensions'): 128,
        ('diffusion_prior', 'time_embedding_dimensions'): 32,
        ('diffusion_prior', 'class_embedding_dimensions'): 8,
        ('diffusion_prior', 'diagnostic_interval'): 500,
        ('diffusion_prior', 'diagnostic_noise_bins'): 4,
        ('feature_prior_adaptation', 'lambda'): 0.1,
        ('feature_prior_adaptation', 'timestep_min'): 100,
        ('feature_prior_adaptation', 'timestep_max'): 500,
        ('storage', 'external_run_id'): 'student_aware_diffusion_prior_loso_v1_seed666',
        ('report', 'primary_metric'): 'balanced_accuracy',
        ('report', 'bootstrap_draws'): 10000,
        ('report', 'bootstrap_seed'): 20261029,
        ('report', 'tie_threshold_pp'): 0.000001,
    }
    for route, value in expected.items():
        actual = config
        for part in route:
            actual = actual.get(part) if isinstance(actual, dict) else None
        if actual != value:
            raise ValueError(f'V1 fixed setting {".".join(route)} must be {value!r}, got {actual!r}')
    if config.get('reference_run', {}).get('protocol_id') != REFERENCE_PROTOCOL:
        raise ValueError('V1 must reuse only the audited wideband seven-source reference run')


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _uid_hash(uids: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(uids, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=True) + '\n')
    os.replace(tmp, path)


def _atomic_torch(path: Path, value) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, require_external_output(tmp))
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp.npz')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object at {path}')
    return value


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(Path(resolve_local_file(path)).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected YAML object at {path}')
    return value


def _verify_manifest_files(root: Path, manifest: dict, names: tuple[str, ...], label: str,
                           *, hash_content: bool = True):
    records = manifest.get('files')
    if not isinstance(records, dict):
        raise ValueError(f'{label}: manifest has no files map')
    for name in names:
        rec = records.get(name)
        if not isinstance(rec, dict):
            raise ValueError(f'{label}: manifest has no record for {name}')
        path = Path(resolve_local_file(root / name))
        if not path.is_file():
            raise FileNotFoundError(f'{label}: missing {path}')
        if path.stat().st_size != int(rec['bytes']):
            raise RuntimeError(f'{label}: size mismatch for {path}')
        if hash_content and _sha256(path) != rec['sha256']:
            raise RuntimeError(f'{label}: hash mismatch for {path}')


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _capture_rng(device: torch.device, generator: torch.Generator | None = None) -> dict:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        'local_generator': generator.get_state() if generator is not None else None,
    }


def _restore_rng(state: dict, device: torch.device,
                 generator: torch.Generator | None = None) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda' and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)
    if generator is not None and state.get('local_generator') is not None:
        generator.set_state(state['local_generator'].cpu())


def _device(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('V1 formal training requires CUDA')
    device = torch.device('cuda', int(gpu))
    torch.cuda.set_device(device)
    return device


def _run_roots(run_id: str, seed: int, fold: int) -> dict[str, Path]:
    split_name = f'target_{fold + 1:02d}_feedback_{(fold + 1) % 9 + 1:02d}_seed_{seed}'
    split_root = DATA_CACHE_ROOT / REFERENCE_PROTOCOL / 'BNCI2014004' / f'seed_{seed}'
    matches = sorted(split_root.glob(split_name + '_*'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one audited split for fold {fold + 1}; found {matches}')
    split_dir = matches[0]
    run_name = split_dir.name
    return {
        'split_dir': split_dir,
        'base_results': RESULTS_ROOT / 'distill' / REFERENCE_PROTOCOL / f'seed_{seed}'
                        / 'folds' / run_name,
        'base_teacher': WEIGHTS_ROOT / REFERENCE_PROTOCOL / f'seed_{seed}' / run_name,
        'input_root': DATA_CACHE_ROOT / 'artifact-loso' / 'model_inputs' / 'BNCI2014004',
        'run_result': RESULTS_ROOT / 'distill' / run_id / f'seed_{seed}' / 'folds' / run_name,
        'run_cache': DATA_CACHE_ROOT / run_id / f'seed_{seed}' / 'folds' / run_name,
        'run_weights': WEIGHTS_ROOT / run_id / f'seed_{seed}' / 'folds' / run_name,
        'run_seed_weights': WEIGHTS_ROOT / run_id / f'seed_{seed}',
    }


def _load_resolved_reference(config: dict) -> tuple[dict, dict]:
    run_id = str(config['reference_run']['protocol_id'])
    seed = int(config['reference_run']['seed'])
    path = RESULTS_ROOT / 'distill' / run_id / f'seed_{seed}' / 'resolved.yaml'
    resolved = _read_yaml(path)
    expected = str(config['reference_run']['resolved_config_sha256'])
    if resolved.get('resolved_config_sha256') != expected:
        raise RuntimeError('historical resolved.yaml hash differs from V1 config')
    if (resolved.get('dataset') != 'BNCI2014004'
            or resolved.get('source_variant') != 'wide_fre_band'
            or resolved.get('selected_session') != '3test'
            or int(resolved.get('seed', -1)) != seed):
        raise RuntimeError('historical resolved.yaml is not the requested wideband seed-666 run')
    student = dict(resolved['student_config'])
    optimizer_defaults = dict(resolved['optimizer_defaults'])
    teacher = dict(resolved['teacher_config'])
    if (student.get('optimizer') != 'adamw' or int(student.get('epochs', 0)) != 100
            or int(student.get('batch_size', 0)) != 16
            or float(student.get('lr', -1)) != 0.001
            or float(student.get('weight_decay', -1)) != 0.01
            or student.get('lr_schedule') != 'epoch_cosine'
            or student.get('use_filter_bank') is not True):
        raise RuntimeError('historical IFNet CE resolved config differs from the V1 baseline')
    if (teacher.get('optimizer') != 'adam' or int(teacher.get('epochs', 0)) != 10
            or int(teacher.get('batch_size', 0)) != 8
            or float(teacher.get('lr', -1)) != 0.001
            or float(teacher.get('weight_decay', -1)) != 1e-6
            or teacher.get('skip_preprocess') is not True):
        raise RuntimeError('historical MIRepNet source-teacher config does not match T0 provenance')
    initial = resolved.get('pretrained_checkpoint', {})
    pretrain_path = Path(str(initial.get('path', '')))
    if not pretrain_path.is_file() or _sha256(pretrain_path) != initial.get('sha256'):
        raise RuntimeError('canonical MIRepNet pretrained checkpoint is missing or changed')
    return resolved, {'student_config': student, 'teacher_config': teacher,
                      'optimizer_defaults': optimizer_defaults,
                      'resolved_config_path': str(path),
                      'resolved_config_sha256': expected,
                      'resolved_yaml_sha256': _sha256(path),
                      'pretrained_checkpoint': str(pretrain_path),
                      'pretrained_checkpoint_sha256': str(initial['sha256'])}


def _load_fold_identity(config: dict, fold: int, roots: dict,
                        resolved_sha: str, *, include_test_labels: bool = False) -> dict:
    path = roots['split_dir'] / 'split_manifest.json'
    manifest = _read_json(path)
    if (manifest.get('dataset') != 'BNCI2014004'
            or manifest.get('source_variant') != 'wide_fre_band'
            or manifest.get('selected_session') != '3test'
            or int(manifest.get('seed', -1)) != int(config['seed'])
            or int(manifest.get('target_subject_id', -1)) != fold + 1
            or int(manifest.get('feedback_subject_id', -1)) != (fold + 1) % 9 + 1
            or manifest.get('resolved_config_sha256') != resolved_sha):
        raise RuntimeError(f'fold {fold + 1}: 7/1/1 split manifest identity mismatch')
    if manifest.get('class_mapping') != {'left_hand': 0, 'right_hand': 1}:
        raise RuntimeError(f'fold {fold + 1}: class encoding differs from V1')
    split = manifest.get('split_uids', {})
    labels = manifest.get('split_labels', {})
    if set(split) != {'train', 'feedback', 'test'} or set(labels) != set(split):
        raise RuntimeError(f'fold {fold + 1}: split manifest lacks a partition')
    arrays = {name: np.asarray(split[name], dtype=np.int64) for name in split}
    label_arrays = {name: np.asarray(labels[name], dtype=np.int64)
                    for name in ('train', 'feedback')}
    if include_test_labels:
        label_arrays['test'] = np.asarray(labels['test'], dtype=np.int64)
    if any(a.ndim != 2 or a.shape[1] != 2 for a in arrays.values()):
        raise RuntimeError(f'fold {fold + 1}: split UID arrays must be [N,2]')
    if any(len(arrays[n]) != len(labels[n]) for n in arrays):
        raise RuntimeError(f'fold {fold + 1}: split labels and UIDs have different lengths')
    uid_sets = {name: {tuple(row) for row in values.tolist()}
                for name, values in arrays.items()}
    if any(uid_sets[a] & uid_sets[b] for a, b in
           (('train', 'feedback'), ('train', 'test'), ('feedback', 'test'))):
        raise RuntimeError(f'fold {fold + 1}: train/feedback/test UID overlap')
    if len(arrays['train']) not in (1080, 1120):
        raise RuntimeError(f'fold {fold + 1}: unexpected D_train count {len(arrays["train"])}')
    if len(arrays['feedback']) != (120 if (fold + 1) % 9 == 1 else 160):
        raise RuntimeError(f'fold {fold + 1}: unexpected feedback subject count')
    if len(arrays['test']) != (120 if fold + 1 == 2 else 160):
        raise RuntimeError(f'fold {fold + 1}: unexpected target subject count')
    for name in label_arrays:
        if not np.isin(label_arrays[name], [0, 1]).all():
            raise RuntimeError(f'fold {fold + 1}: invalid labels in {name}')
    train_subjects = sorted({int(uid[0]) + 1 for uid in arrays['train']})
    feedback_subjects = sorted({int(uid[0]) + 1 for uid in arrays['feedback']})
    target_subjects = sorted({int(uid[0]) + 1 for uid in arrays['test']})
    expected_train = sorted(set(range(1, 10)) - {fold + 1, (fold + 1) % 9 + 1})
    if (train_subjects != expected_train or feedback_subjects != [(fold + 1) % 9 + 1]
            or target_subjects != [fold + 1]):
        raise RuntimeError(f'fold {fold + 1}: expected 7 train / 1 feedback / 1 target subjects')
    source_hashes = manifest.get('source_hashes', {})
    source_manifest_sha = source_hashes.get('manifest.json', {}).get('sha256')
    source_trials_sha = source_hashes.get('trials.csv', {}).get('sha256')
    source_manifest_path = Path(resolve_local_file(BNCI14004_SOURCE_ROOT / 'manifest.json'))
    if not source_manifest_path.is_file() or _sha256(source_manifest_path) != source_manifest_sha:
        raise RuntimeError(f'fold {fold + 1}: split does not match active wide_fre_band manifest')
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get('source_variant') not in ('wide_fre_band', 'broadband_0_120hz'):
        raise RuntimeError('the active source manifest is not the audited wideband source')
    if source_manifest.get('files', {}).get('trials.csv', {}).get('sha256') != source_trials_sha:
        raise RuntimeError(f'fold {fold + 1}: source trial manifest differs from archived split')
    _verify_manifest_files(BNCI14004_SOURCE_ROOT, source_manifest,
                           ('X.npy', 'labels.npy', 'trials.csv'), '004 wide_fre_band source',
                           hash_content=False)
    for model in ('mirepnet', 'ifnet'):
        model_root = roots['input_root'] / model
        input_manifest_path = model_root / 'manifest.json'
        input_manifest = _read_json(input_manifest_path)
        if (input_manifest.get('dataset') != 'BNCI2014004'
                or input_manifest.get('model') != model
                or input_manifest.get('source_manifest_sha256') != source_manifest_sha):
            raise RuntimeError(f'{model}: model input is not tied to the active wideband source')
        _verify_manifest_files(model_root, input_manifest,
                               ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'),
                               f'004/{model} model input', hash_content=False)
    return {'path': str(path), 'sha256': _sha256(path), 'manifest': manifest,
            'uids': arrays, 'labels': label_arrays,
            'uid_sha256': {name: _uid_hash(values) for name, values in arrays.items()},
            'source_manifest_sha256': source_manifest_sha,
            'source_trial_manifest_sha256': source_trials_sha}


def _input_rows(root: Path, required_uids: dict[str, np.ndarray],
                expected_labels: dict[str, np.ndarray], model: str) -> tuple[dict, dict]:
    manifest = _read_json(root / 'manifest.json')
    trials = pd.read_csv(resolve_local_file(root / 'trials.csv'),
                         usecols=['subject_zero_based', 'legacy_raw_row', 'trial_uid'])
    keys = list(zip(trials['subject_zero_based'].astype(int),
                    trials['legacy_raw_row'].astype(int)))
    if len(set(keys)) != len(keys):
        raise RuntimeError(f'{model}: duplicate (subject,legacy_raw_row) identifiers')
    index = {key: row for row, key in enumerate(keys)}
    arrays = {}
    for part, uids in required_uids.items():
        try:
            rows = np.asarray([index[tuple(map(int, uid))] for uid in uids], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f'{model}: missing archived UID {exc.args[0]}') from exc
        arrays[part] = rows
    y = np.load(resolve_local_file(root / 'y.npy'), mmap_mode='r', allow_pickle=False)
    subjects = np.load(resolve_local_file(root / 'subjects.npy'), mmap_mode='r',
                       allow_pickle=False)
    for part, rows in arrays.items():
        if part in expected_labels and not np.array_equal(
                np.asarray(y[rows], dtype=np.int64), expected_labels[part]):
            raise RuntimeError(f'{model}: {part} labels differ from split manifest')
        expected_subject = required_uids[part][:, 0]
        if not np.array_equal(np.asarray(subjects[rows], dtype=np.int64), expected_subject):
            raise RuntimeError(f'{model}: {part} subject IDs differ from split manifest')
    return arrays, {'manifest': manifest, 'trials_sha256': _sha256(root / 'trials.csv')}


def _preprocess_student(adapter, raw: np.ndarray, batch: int = 128) -> np.ndarray:
    chunks = []
    for start in range(0, len(raw), batch):
        chunk = adapter.preprocess(np.asarray(raw[start:start + batch], dtype=np.float32))
        chunks.append(chunk.detach().cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        return np.empty((0, 6, int(raw.shape[-1])), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _load_fold_data(identity: dict, roots: dict, student_cfg: dict,
                    device: torch.device) -> dict:
    split = identity['uids']
    labels = identity['labels']
    student_root = roots['input_root'] / 'ifnet'
    teacher_root = roots['input_root'] / 'mirepnet'
    train_feedback = {part: split[part] for part in ('train', 'feedback')}
    train_feedback_labels = {part: labels[part] for part in ('train', 'feedback')}
    student_rows, student_identity = _input_rows(
        student_root, split, train_feedback_labels, 'ifnet')
    teacher_rows, teacher_identity = _input_rows(
        teacher_root, split, train_feedback_labels, 'mirepnet')
    student_raw = np.load(resolve_local_file(student_root / 'X.npy'), mmap_mode='r',
                          allow_pickle=False)
    teacher_inputs = np.load(resolve_local_file(teacher_root / 'X.npy'), mmap_mode='r',
                             allow_pickle=False)
    if tuple(teacher_inputs.shape[1:]) != (45, 1000):
        raise RuntimeError(f'mirepnet cached input shape is {teacher_inputs.shape}')
    if tuple(student_raw.shape[1:]) != (3, 1000):
        raise RuntimeError(f'ifnet cached canonical input shape is {student_raw.shape}')
    student_adapter = get_adapter('ifnet', device='cpu', **student_cfg)
    processed = {}
    for part in ('train', 'feedback'):
        rows = student_rows[part]
        processed[part] = _preprocess_student(
            student_adapter, np.asarray(student_raw[rows], dtype=np.float32))
    preprocess_diffs = {}
    # The split-bound cache is the exact IFNet tensor used by S0. Validate it
    # against the current model-input source first, then reuse those exact
    # float32 values so the paired initialization replay is bit-for-bit.
    pilot_cache = roots['split_dir'] / 'student_inputs.npz'
    if pilot_cache.is_file():
        with np.load(resolve_local_file(pilot_cache), allow_pickle=False) as saved:
            for part in ('train', 'feedback'):
                for suffix, expected in (('uids', split[part]), ('y', labels[part])):
                    if not np.array_equal(saved[f'{part}_{suffix}'], expected):
                        raise RuntimeError(f'prepared {part} {suffix} differs from the split')
                cached_x = np.asarray(saved[f'{part}_x'], dtype=np.float32)
                if cached_x.shape != processed[part].shape:
                    raise RuntimeError(f'prepared {part} IFNet input shape differs from source')
                max_abs_diff = float(np.max(np.abs(cached_x - processed[part])))
                preprocess_diffs[part] = max_abs_diff
                if not np.allclose(cached_x, processed[part], rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f'prepared {part} IFNet input differs from verified source')
                processed[part] = cached_x.copy()
    teacher_values = {
        part: np.asarray(teacher_inputs[teacher_rows[part]], dtype=np.float32)
        for part in ('train', 'feedback')}
    return {
        'student_x': processed,
        'teacher_x': teacher_values,
        'y': {part: labels[part] for part in ('train', 'feedback')},
        'uids': {part: split[part] for part in ('train', 'feedback')},
        'student_rows': student_rows,
        'teacher_rows': teacher_rows,
        'target_uids': split['test'],
        'target_rows': None,
        'student_input_manifest': student_identity['manifest'],
        'teacher_input_manifest': teacher_identity['manifest'],
        'student_trials_sha256': student_identity['trials_sha256'],
        'teacher_trials_sha256': teacher_identity['trials_sha256'],
        'student_raw_mmap': student_raw,
        'teacher_input_mmap': teacher_inputs,
        'student_input_root': student_root,
        'teacher_input_root': teacher_root,
        'student_adapter': student_adapter,
        'student_preprocess_max_abs_diff': preprocess_diffs,
    }


def _target_rows(inputs: dict, target_uids: np.ndarray) -> np.ndarray:
    """Resolve target row positions without reading their labels or signals."""
    root = inputs['student_input_root']
    trials = pd.read_csv(resolve_local_file(root / 'trials.csv'),
                         usecols=['subject_zero_based', 'legacy_raw_row'])
    keys = list(zip(trials['subject_zero_based'].astype(int),
                    trials['legacy_raw_row'].astype(int)))
    index = {key: row for row, key in enumerate(keys)}
    return np.asarray([index[tuple(map(int, uid))] for uid in target_uids], dtype=np.int64)


def _load_reference_checkpoints(config: dict, resolved: dict, cfg: dict,
                                identity: dict, roots: dict,
                                device: torch.device) -> dict:
    seed = int(config['seed'])
    split = identity['manifest']
    expected_train_uid = identity['uid_sha256']['train']
    teacher_root = roots['base_teacher']
    teacher_checkpoint = teacher_root / 'teacher_final.pt'
    teacher_provenance_path = teacher_root / 'teacher_provenance.json'
    baseline_root = roots['base_results'] / 'BASE_CE'
    baseline_checkpoint = baseline_root / 'checkpoint_final.pt'
    baseline_manifest_path = baseline_root / 'manifest.json'
    warmup_checkpoint = roots['base_teacher'] / 'warmup_epoch10.pt'
    for path in (teacher_checkpoint, teacher_provenance_path, baseline_checkpoint,
                 baseline_manifest_path, warmup_checkpoint):
        if not Path(resolve_local_file(path)).is_file():
            raise FileNotFoundError(f'required reference artifact is missing: {path}')
    teacher_provenance = _read_json(teacher_provenance_path)
    if (teacher_provenance.get('model') != 'mirepnet'
            or int(teacher_provenance.get('seed', -1)) != seed
            or int(teacher_provenance.get('target_subject_id', -1)) != int(split['target_subject_id'])
            or int(teacher_provenance.get('feedback_subject_id', -1)) != int(split['feedback_subject_id'])
            or list(teacher_provenance.get('train_subject_ids', [])) != split['train_subject_ids']
            or teacher_provenance.get('train_uid_sha256') != expected_train_uid
            or int(teacher_provenance.get('epochs', -1)) != 10):
        raise RuntimeError(f'fold {int(split["target_subject_id"]):02d}: T0 provenance mismatch')
    source_teacher_cfg = teacher_provenance.get('config', {})
    for key in ('dataset_name', 'depth', 'emb_size', 'size', 'epochs', 'batch_size',
                'lr', 'weight_decay', 'optimizer', 'skip_preprocess'):
        if source_teacher_cfg.get(key) != resolved['teacher_config'].get(key):
            raise RuntimeError(f'T0 provenance {key} differs from its historical resolved config')
    pretrained_hash = teacher_provenance.get('initial_checkpoint_sha256')
    if pretrained_hash != cfg['pretrained_checkpoint_sha256']:
        raise RuntimeError('T0 was initialized from a different MIRepNet pretrained checkpoint')
    teacher_checkpoint_sha = _sha256(teacher_checkpoint)

    base_manifest = _read_json(baseline_manifest_path)
    if (base_manifest.get('status') != 'complete'
            or base_manifest.get('condition') != 'BASE_CE'
            or int(base_manifest.get('final_epoch', -1)) != 100
            or int(base_manifest.get('student_seed', -1)) != seed
            or int(base_manifest.get('target_subject_id', -1)) != int(split['target_subject_id'])
            or int(base_manifest.get('feedback_subject_id', -1)) != int(split['feedback_subject_id'])
            or base_manifest.get('train_uid_sha256') != expected_train_uid
            or base_manifest.get('resolved_config_sha256') != config['reference_run']['resolved_config_sha256']):
        raise RuntimeError(f'fold {int(split["target_subject_id"]):02d}: S0 BASE_CE provenance mismatch')
    history_path = baseline_root / 'history.json'
    history = json.loads(Path(resolve_local_file(history_path)).read_text())
    if (len(history) != 90 or int(history[0].get('epoch', -1)) != 11
            or int(history[-1].get('epoch', -1)) != 100
            or any(abs(float(row.get('effective_lam_kd', 0.0))) > 1e-12
                   for row in history)):
        raise RuntimeError('S0 BASE_CE history is incomplete or contains active KD')
    warmup = torch.load(resolve_local_file(warmup_checkpoint), map_location='cpu',
                        weights_only=False)
    if (int(warmup.get('epoch', -1)) != 10
            or int(warmup.get('seed', -1)) != seed
            or warmup.get('train_uid_sha256') != expected_train_uid
            or warmup.get('resolved_config_sha256') != config['reference_run']['resolved_config_sha256']
            or warmup.get('warmup_state_sha256') != base_manifest.get('warmup_state_sha256')):
        raise RuntimeError('S0 pure-CE warmup checkpoint does not match BASE_CE provenance')
    baseline_checkpoint_sha = _sha256(baseline_checkpoint)

    # The schedule hash proves that the historical 100-epoch CE trajectory and
    # the new paired students visit exactly the same D_train UID batches.
    from experiments.distill import sample_utility_protocol as pilot_protocol
    from experiments.distill import run_sample_utility_adaptive_rl_loso as pilot_runner
    train_uids = identity['uids']['train']
    student_schedule = pilot_protocol.epoch_schedule(
        len(train_uids), int(resolved['student_config']['batch_size']), 100, seed)
    student_schedule_sha = pilot_runner._schedule_hash(student_schedule, train_uids)
    if student_schedule_sha != base_manifest.get('schedule_sha256'):
        raise RuntimeError('new paired-student data order differs from the audited S0 schedule')
    for part, saved in (('train', identity['uids']['train']),):
        if len(saved) != len(train_uids):
            raise RuntimeError(f'{part}: invalid paired UID schedule input')

    teacher_adapter = get_adapter('mirepnet', device=device,
                                  **resolved['teacher_config'])
    teacher_model = teacher_adapter.build(2)
    teacher_payload = torch.load(resolve_local_file(teacher_checkpoint), map_location='cpu',
                                 weights_only=False)
    if 'model' not in teacher_payload:
        raise RuntimeError('T0 checkpoint has no model state')
    teacher_model.load_state_dict(teacher_payload['model'], strict=True)
    teacher_model.eval()
    if any(parameter.grad is not None for parameter in teacher_model.parameters()):
        raise RuntimeError('T0 unexpectedly has gradients before adaptation')

    student_adapter = get_adapter('ifnet', device=device, **resolved['student_config'])
    student_model = student_adapter.build(2)
    student_payload = torch.load(resolve_local_file(baseline_checkpoint), map_location='cpu',
                                 weights_only=False)
    if 'model' not in student_payload:
        raise RuntimeError('S0 checkpoint has no model state')
    student_model.load_state_dict(student_payload['model'], strict=True)
    student_model.eval()
    student_model_sha = _state_hash(student_model.state_dict())
    if student_model_sha != base_manifest.get('final_model_sha256'):
        raise RuntimeError('S0 state hash differs from its completion manifest')
    return {
        'teacher_adapter': teacher_adapter,
        'teacher_model': teacher_model,
        'student_adapter': student_adapter,
        'student_model': student_model,
        'teacher_checkpoint': str(teacher_checkpoint),
        'teacher_checkpoint_sha256': teacher_checkpoint_sha,
        'teacher_provenance': teacher_provenance,
        'baseline_checkpoint': str(baseline_checkpoint),
        'baseline_checkpoint_sha256': baseline_checkpoint_sha,
        'baseline_manifest': base_manifest,
        'baseline_model_sha256': student_model_sha,
        'warmup_checkpoint': str(warmup_checkpoint),
        'warmup_payload': warmup,
        'student_schedule': student_schedule,
        'student_schedule_sha256': student_schedule_sha,
        'student_cfg': resolved['student_config'],
        'optimizer_defaults': resolved['optimizer_defaults'],
    }


def _batch_slices(n: int, batch_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + batch_size, n), dtype=np.int64)
            for start in range(0, n, int(batch_size))]


def _epoch_schedule(n: int, batch_size: int, epochs: int, seed: int) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    result = []
    for _ in range(int(epochs)):
        order = rng.permutation(int(n))
        result.append([order[start:start + int(batch_size)]
                       for start in range(0, int(n), int(batch_size))])
    return result


def _schedule_sha(schedule: list[list[np.ndarray]], uids: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(uids, dtype=np.int64)).tobytes())
    for epoch in schedule:
        digest.update(np.asarray([len(epoch)], dtype=np.int64).tobytes())
        for batch in epoch:
            digest.update(np.ascontiguousarray(batch, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _infer(adapter, model, values: np.ndarray, device: torch.device,
           batch_size: int = 64, return_features: bool = True) -> tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    before_hash = _state_hash(before)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    model.eval()
    features, logits = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(values), int(batch_size)):
                xb = torch.as_tensor(np.asarray(values[start:start + batch_size],
                                                dtype=np.float32), device=device)
                if adapter.name == 'ifnet':
                    result = model(xb, return_features=True)
                else:
                    result = adapter.forward(model, xb)
                feat, logit = result[0], result[1]
                if not return_features:
                    feat = torch.zeros((len(xb), 1), dtype=logit.dtype, device=logit.device)
                features.append(feat.detach().float().cpu().numpy())
                logits.append(logit.detach().float().cpu().numpy())
        if not logits:
            return np.empty((0, 1), np.float32), np.empty((0, 2), np.float32)
    finally:
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if _state_hash(model.state_dict()) != before_hash:
            model.load_state_dict(before, strict=True)
            if _state_hash(model.state_dict()) != before_hash:
                raise RuntimeError('inference mutated a frozen reference model')
    feat = np.concatenate(features, axis=0).astype(np.float32)
    logit = np.concatenate(logits, axis=0).astype(np.float32)
    if not np.isfinite(feat).all() or not np.isfinite(logit).all():
        raise FloatingPointError('model inference returned NaN or Inf')
    return feat, logit


def _reference_cache(identity: dict, roots: dict, models: dict,
                     data: dict, run_fingerprint: str,
                     device: torch.device) -> dict:
    path = roots['run_cache'] / 'reference_train_feedback.npz'
    manifest_path = roots['run_cache'] / 'reference_train_feedback.json'
    if path.is_file() and manifest_path.is_file():
        saved_manifest = _read_json(manifest_path)
        if (saved_manifest.get('run_fingerprint') != run_fingerprint
                or saved_manifest.get('file_sha256') != _sha256(path)):
            raise RuntimeError('existing reference feature cache belongs to another identity')
        with np.load(resolve_local_file(path), allow_pickle=False) as source:
            result = {key: np.asarray(source[key]) for key in source.files}
        for part in ('train', 'feedback'):
            if (not np.array_equal(result[f'{part}_uids'], identity['uids'][part])
                    or not np.array_equal(result[f'{part}_y'], identity['labels'][part])):
                raise RuntimeError(f'{part} reference cache UID/label mismatch')
        return result
    result = {}
    for part in ('train', 'feedback'):
        st_feat, st_logits = _infer(
            models['student_adapter'], models['student_model'],
            data['student_x'][part], device, batch_size=64)
        te_feat, te_logits = _infer(
            models['teacher_adapter'], models['teacher_model'],
            data['teacher_x'][part], device, batch_size=32)
        if st_feat.ndim != 2 or te_feat.ndim != 2:
            raise RuntimeError('feature taps must return one vector per trial')
        result.update({f'{part}_student_features': st_feat,
                       f'{part}_student_logits': st_logits,
                       f'{part}_teacher_features': te_feat,
                       f'{part}_teacher_logits': te_logits,
                       f'{part}_uids': identity['uids'][part],
                       f'{part}_y': identity['labels'][part]})
    if result['train_student_features'].shape[0] != len(identity['uids']['train']):
        raise RuntimeError('S0 feature cache lost trial alignment')
    _atomic_npz(path, **result)
    _atomic_json(manifest_path, {
        'run_fingerprint': run_fingerprint,
        'file_sha256': _sha256(path),
        'feature_taps': {
            'student': 'IFNet.forward(return_features=True).feat; exact input to IFNet.fc',
            'teacher': 'MIRepNet native pooled feature returned before classifier',
        },
        'student_feature_shape': list(result['train_student_features'].shape),
        'teacher_feature_shape': list(result['train_teacher_features'].shape),
        'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'teacher_checkpoint_sha256': models['teacher_checkpoint_sha256'],
        'student_checkpoint_sha256': models['baseline_checkpoint_sha256'],
    })
    return result


def _pairwise_sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    distances = (np.sum(a * a, axis=1)[:, None]
                 + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T))
    return np.maximum(distances, 0.0)


def _rbf_mmd(left: np.ndarray, right: np.ndarray, bandwidth_squared: float) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    gamma = 1.0 / (2.0 * max(float(bandwidth_squared), 1e-12))
    kxx = np.exp(-gamma * _pairwise_sqdist(x, x)).mean()
    kyy = np.exp(-gamma * _pairwise_sqdist(y, y)).mean()
    kxy = np.exp(-gamma * _pairwise_sqdist(x, y)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def _effective_rank(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _distribution_summary(reference: np.ndarray, projected: np.ndarray,
                          bandwidth_squared: float) -> dict:
    target = np.asarray(reference, dtype=np.float64)
    value = np.asarray(projected, dtype=np.float64)
    mse = float(np.mean((target - value) ** 2))
    variance = float(np.mean((target - target.mean(axis=0)) ** 2))
    r2 = float(1.0 - mse / variance) if variance > 0 else float('nan')
    covariance_trace = float(np.var(value, axis=0, ddof=1).sum()) if len(value) > 1 else 0.0
    return {
        'mse': mse,
        'r2': r2,
        'mean_distance_l2': float(np.linalg.norm(target.mean(axis=0) - value.mean(axis=0))),
        'covariance_trace': covariance_trace,
        'effective_rank': _effective_rank(value),
        'mmd_rbf': _rbf_mmd(target, value, bandwidth_squared),
    }


def _fit_feature_prior(reference: dict, identity: dict, roots: dict,
                       config: dict, run_fingerprint: str) -> dict:
    path = roots['run_weights'] / 'fixed_feature_prior.npz'
    manifest_path = roots['run_weights'] / 'fixed_feature_prior.json'
    if path.is_file() or manifest_path.is_file():
        if not (path.is_file() and manifest_path.is_file()):
            raise RuntimeError('incomplete fixed prior artifact; use a new run identity')
        old = _read_json(manifest_path)
        if old.get('run_fingerprint') != run_fingerprint or old.get('file_sha256') != _sha256(path):
            raise RuntimeError('fixed prior artifact belongs to another run identity')
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            pca = {key: np.asarray(saved[f'pca_{key}']) for key in
                   ('mean', 'components', 'eigenvalues', 'scales')}
            pca.update({key: old['pca'][key] for key in
                        ('effective_rank', 'dimension', 'total_variance',
                         'retained_variance', 'retained_variance_fraction',
                         'maximum_eigenvalue')})
            projection = {key: np.asarray(saved[f'projection_{key}']) for key in
                          ('teacher_mean', 'teacher_std', 'weight', 'bias', 'singular_values')}
            projection.update({key: old['projection'][key] for key in
                               ('rho', 'train_mse', 'train_r2', 'effective_rank')})
            gaussian = {'means': np.asarray(saved['gaussian_means']),
                        'variances': np.asarray(saved['gaussian_variances']),
                        'counts': np.asarray(saved['gaussian_counts']),
                        'variance_floor': float(old['gaussian']['variance_floor'])}
            whitening_train = np.asarray(saved['train_whitened'])
            whitening_feedback = np.asarray(saved['feedback_whitened'])
            projection_train = np.asarray(saved['train_projected_t0'])
            projection_feedback = np.asarray(saved['feedback_projected_t0'])
        return {'pca': pca, 'projection': projection, 'gaussian': gaussian,
                'train_student_features': whitening_train,
                'feedback_student_features': whitening_feedback,
                'train_projected_t0': projection_train,
                'feedback_projected_t0': projection_feedback,
                'diagnostics': old['diagnostics'], 'path': str(path),
                'manifest': old}

    feature_cfg = config['feature_space']
    pca = fit_pca_whitener(
        reference['train_student_features'],
        max_dimensions=int(feature_cfg['pca_max_dimensions']),
        rank_relative_threshold=float(feature_cfg['numerical_rank_relative_threshold']),
        eigenvalue_floor_relative=float(feature_cfg['whitening_eigenvalue_floor_relative']))
    student_train = apply_pca_whitener(reference['train_student_features'], pca)
    student_feedback = apply_pca_whitener(reference['feedback_student_features'], pca)
    projection = fit_ridge_projection(
        reference['train_teacher_features'], student_train,
        rho=float(feature_cfg['ridge_rho']),
        standard_deviation_floor=float(feature_cfg['teacher_standard_deviation_floor']))
    projected_train = apply_ridge_projection(reference['train_teacher_features'], projection)
    projected_feedback = apply_ridge_projection(reference['feedback_teacher_features'], projection)
    gaussian = gaussian_prior_fit(
        student_train, identity['labels']['train'], classes=2,
        variance_floor=float(config['gaussian_prior']['variance_floor']))
    pair_distances_squared = pdist(student_train.astype(np.float64), metric='sqeuclidean')
    median_squared = float(np.median(pair_distances_squared))
    if not math.isfinite(median_squared) or median_squared <= 0:
        raise RuntimeError('D_train feature-pair median distance is not positive')
    diagnostics = {
        'train_projection': _distribution_summary(student_train, projected_train, median_squared),
        'feedback_projection': _distribution_summary(student_feedback, projected_feedback,
                                                      median_squared),
        'bandwidth_squared_median_train': median_squared,
        'train_student_effective_rank': _effective_rank(student_train),
        'feedback_student_effective_rank': _effective_rank(student_feedback),
        'initial_train_mmd': _rbf_mmd(student_train, projected_train, median_squared),
        'initial_feedback_mmd': _rbf_mmd(student_feedback, projected_feedback, median_squared),
    }
    if pca['effective_rank'] <= 0 or pca['dimension'] <= 0:
        raise RuntimeError('PCA feature space has zero effective rank')
    arrays = {
        'pca_mean': pca['mean'], 'pca_components': pca['components'],
        'pca_eigenvalues': pca['eigenvalues'], 'pca_scales': pca['scales'],
        'projection_teacher_mean': projection['teacher_mean'],
        'projection_teacher_std': projection['teacher_std'],
        'projection_weight': projection['weight'], 'projection_bias': projection['bias'],
        'projection_singular_values': projection['singular_values'],
        'gaussian_means': gaussian['means'], 'gaussian_variances': gaussian['variances'],
        'gaussian_counts': gaussian['counts'],
        'train_whitened': student_train, 'feedback_whitened': student_feedback,
        'train_projected_t0': projected_train,
        'feedback_projected_t0': projected_feedback,
        'train_uids': identity['uids']['train'], 'train_y': identity['labels']['train'],
        'feedback_uids': identity['uids']['feedback'],
        'feedback_y': identity['labels']['feedback'],
    }
    _atomic_npz(path, **arrays)
    manifest = {
        'run_fingerprint': run_fingerprint,
        'file_sha256': _sha256(path),
        'fit_partition': 'D_train only',
        'pca': {key: (value if isinstance(value, (float, int)) else None)
                for key, value in pca.items() if key not in
                ('mean', 'components', 'eigenvalues', 'scales')},
        'projection': {'rho': projection['rho'], 'train_mse': projection['train_mse'],
                       'train_r2': projection['train_r2'],
                       'effective_rank': projection['effective_rank']},
        'gaussian': {'variance_floor': gaussian['variance_floor'],
                     'class_counts': gaussian['counts'].tolist()},
        'diagnostics': diagnostics,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'student_feature_shape': list(reference['train_student_features'].shape),
        'teacher_feature_shape': list(reference['train_teacher_features'].shape),
    }
    _atomic_json(manifest_path, manifest)
    return {'pca': pca, 'projection': projection, 'gaussian': gaussian,
            'train_student_features': student_train,
            'feedback_student_features': student_feedback,
            'train_projected_t0': projected_train,
            'feedback_projected_t0': projected_feedback,
            'diagnostics': diagnostics, 'path': str(path), 'manifest': manifest}


def _metric_values(y: np.ndarray, logits: np.ndarray) -> dict:
    labels = np.asarray(y, dtype=np.int64)
    scores = np.asarray(logits, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    log_probs = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    pred = probs.argmax(axis=1)
    counts = np.bincount(pred, minlength=2).astype(int)
    return {
        'n': int(len(labels)),
        'accuracy': float(accuracy_score(labels, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
        'kappa': float(cohen_kappa_score(labels, pred)),
        'cross_entropy': float(-log_probs[np.arange(len(labels)), labels].mean()),
        'prediction_counts': counts.tolist(),
        'collapse': bool(np.count_nonzero(counts) < 2),
    }


def _kl_teacher_to_student(teacher_logits: torch.Tensor,
                           student_logits: torch.Tensor,
                           temperature: float) -> torch.Tensor:
    t = float(temperature)
    return F.kl_div(F.log_softmax(student_logits / t, dim=1),
                    F.softmax(teacher_logits.detach() / t, dim=1),
                    reduction='batchmean') * (t * t)


def _aid_student_to_teacher(teacher_logits: torch.Tensor,
                            frozen_student_logits: torch.Tensor,
                            temperature: float) -> torch.Tensor:
    t = float(temperature)
    p_teacher = F.softmax(teacher_logits / t, dim=1)
    log_p_teacher = F.log_softmax(teacher_logits / t, dim=1)
    log_p_student = F.log_softmax(frozen_student_logits.detach() / t, dim=1)
    return (p_teacher * (log_p_teacher - log_p_student)).sum(dim=1).mean() * (t * t)


def _audit_loss_math(device: torch.device) -> dict:
    torch.manual_seed(123)
    t_logits = torch.randn(5, 2, device=device, requires_grad=True)
    s_logits = torch.randn(5, 2, device=device)
    temp = 2.0
    explicit = _aid_student_to_teacher(t_logits, s_logits, temp)
    alternate = (F.softmax(t_logits / temp, dim=1)
                 * (F.log_softmax(t_logits / temp, dim=1)
                    - F.log_softmax(s_logits / temp, dim=1))).sum(1).mean() * temp ** 2
    explicit_grad = torch.autograd.grad(explicit, t_logits, retain_graph=True)[0]
    alternate_grad = torch.autograd.grad(alternate, t_logits)[0]
    if not torch.allclose(explicit, alternate, atol=1e-7, rtol=1e-7):
        raise RuntimeError('AID KL does not match the explicit KL(p_T || p_S0) formula')
    if not torch.allclose(explicit_grad, alternate_grad, atol=1e-7, rtol=1e-7):
        raise RuntimeError('AID KL gradient differs from explicit formula')
    u = torch.randn(4, 7, device=device, requires_grad=True)
    g = torch.randn_like(u)
    surrogate = sds_surrogate(u, g)
    actual = torch.autograd.grad(surrogate, u)[0]
    expected = g / u.numel()
    if not torch.allclose(actual, expected, atol=1e-7, rtol=1e-7):
        raise RuntimeError('SDS surrogate gradient is not g/(B*d)')
    # The analytic Gaussian epsilon must equal sigma times the score residual.
    mean = torch.randn(2, 7, device=device)
    variance = torch.rand(2, 7, device=device) + 0.1
    alpha = torch.linspace(0.99, 0.3, 1000, device=device)
    sigma = torch.sqrt(1 - alpha.square())
    t = torch.tensor([170, 450, 210, 300], dtype=torch.long, device=device)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)
    eps = torch.randn(4, 7, device=device)
    clean = torch.randn(4, 7, device=device)
    noisy = alpha[t - 1, None] * clean + sigma[t - 1, None] * eps
    pred = gaussian_epsilon(noisy, t, y, mean, variance, alpha, sigma)
    a = alpha[t - 1, None]
    s = sigma[t - 1, None]
    analytic = s * (noisy - a * mean[y]) / (a.square() * variance[y] + s.square())
    if not torch.allclose(pred, analytic, atol=1e-7, rtol=1e-7):
        raise RuntimeError('analytic Gaussian epsilon score has the wrong direction or scale')
    return {'aid_loss_abs_error': float((explicit - alternate).abs().cpu()),
            'aid_gradient_max_abs_error': float((explicit_grad - alternate_grad).abs().max().cpu()),
            'sds_gradient_max_abs_error': float((actual - expected).abs().max().cpu()),
            'gaussian_score_max_abs_error': float((pred - analytic).abs().max().cpu())}


def _diagnostic_noise(train_x: np.ndarray, train_y: np.ndarray,
                      feedback_x: np.ndarray, feedback_y: np.ndarray,
                      config: dict, roots: dict, fingerprint: str,
                      seed: int) -> dict:
    path = roots['run_cache'] / 'diffusion_diagnostic_noise.npz'
    manifest_path = roots['run_cache'] / 'diffusion_diagnostic_noise.json'
    if path.is_file() and manifest_path.is_file():
        old = _read_json(manifest_path)
        if old.get('run_fingerprint') != fingerprint or old.get('file_sha256') != _sha256(path):
            raise RuntimeError('fixed diffusion diagnostic noise belongs to another run')
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            return {key: np.asarray(saved[key]) for key in saved.files}
    bins = int(config['diffusion_prior']['diagnostic_noise_bins'])
    steps = int(config['diffusion_prior']['steps'])
    rng = np.random.default_rng(int(seed) + 88021)
    arrays = {}
    for part, x, y in (('train', train_x, train_y), ('feedback', feedback_x, feedback_y)):
        count = min(len(x), 512)
        rows = np.sort(rng.choice(len(x), size=count, replace=False))
        arrays[f'{part}_x'] = np.asarray(x[rows], dtype=np.float32)
        arrays[f'{part}_y'] = np.asarray(y[rows], dtype=np.int64)
        arrays[f'{part}_rows'] = np.asarray(rows, dtype=np.int64)
        for bin_id in range(bins):
            lower = int(math.floor(bin_id * steps / bins)) + 1
            upper = int(math.floor((bin_id + 1) * steps / bins))
            arrays[f'{part}_t_{bin_id}'] = rng.integers(
                lower, upper + 1, size=count, dtype=np.int64)
            arrays[f'{part}_eps_{bin_id}'] = rng.standard_normal(
                (count, x.shape[1]), dtype=np.float32)
    _atomic_npz(path, **arrays)
    _atomic_json(manifest_path, {'run_fingerprint': fingerprint,
                                 'file_sha256': _sha256(path),
                                 'noise_bins': bins,
                                 'train_count': len(arrays['train_y']),
                                 'feedback_count': len(arrays['feedback_y']),
                                 'seed': int(seed) + 88021})
    return arrays


def _denoiser_diagnostics(model: torch.nn.Module, data: dict,
                          gaussian: dict, alpha: torch.Tensor,
                          sigma: torch.Tensor, device: torch.device,
                          bins: int) -> dict:
    model.eval()
    output = {}
    gaussian_means = torch.as_tensor(gaussian['means'], dtype=torch.float32, device=device)
    gaussian_vars = torch.as_tensor(gaussian['variances'], dtype=torch.float32, device=device)
    with torch.no_grad():
        for part in ('train', 'feedback'):
            x_all = torch.as_tensor(data[f'{part}_x'], dtype=torch.float32, device=device)
            y_all = torch.as_tensor(data[f'{part}_y'], dtype=torch.long, device=device)
            for bin_id in range(int(bins)):
                t_all = torch.as_tensor(data[f'{part}_t_{bin_id}'], dtype=torch.long,
                                        device=device)
                eps_all = torch.as_tensor(data[f'{part}_eps_{bin_id}'], dtype=torch.float32,
                                          device=device)
                preds, gaussian_preds = [], []
                for start in range(0, len(x_all), 128):
                    stop = min(start + 128, len(x_all))
                    t = t_all[start:stop]
                    y = y_all[start:stop]
                    eps = eps_all[start:stop]
                    noised = alpha[t - 1, None] * x_all[start:stop] + sigma[t - 1, None] * eps
                    preds.append(model(noised, t, y))
                    gaussian_preds.append(gaussian_epsilon(
                        noised, t, y, gaussian_means, gaussian_vars, alpha, sigma))
                eps_hat = torch.cat(preds, dim=0)
                eps_gaussian = torch.cat(gaussian_preds, dim=0)
                output[f'{part}_bin{bin_id}_diffusion_mse'] = float(
                    (eps_hat - eps_all).square().mean().cpu())
                output[f'{part}_bin{bin_id}_zero_predictor_mse'] = float(
                    eps_all.square().mean().cpu())
                output[f'{part}_bin{bin_id}_gaussian_mse'] = float(
                    (eps_gaussian - eps_all).square().mean().cpu())
    return output


def _progress(path: str | None, *, run_id: str, fold: int,
              stage: str, completed: int = 0, total: int = 0,
              status: str = 'running', message: str = '') -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps({
        'run_id': run_id, 'fold': int(fold) + 1,
        'stage': stage, 'completed': int(completed), 'total': int(total),
        'status': status, 'message': message, 'updated_at_unix': time.time(),
    }, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, target)


def _stopped(stop_file: str | None) -> bool:
    return bool(stop_file and Path(stop_file).is_file())


def _fit_diffusion_prior(identity: dict, roots: dict, config: dict,
                         feature_artifacts: dict, reference: dict,
                         fingerprint: str, seed: int, device: torch.device,
                         stop_file: str | None, progress_path: str | None,
                         run_id: str, fold: int) -> tuple[ConditionalNoisePredictor | None, dict]:
    weights_path = roots['run_weights'] / 'prior_diffusion.pt'
    manifest_path = roots['run_weights'] / 'prior_diffusion.json'
    if weights_path.is_file() or manifest_path.is_file():
        if not (weights_path.is_file() and manifest_path.is_file()):
            raise RuntimeError('incomplete Diffusion prior output; use a new run identity')
        manifest = _read_json(manifest_path)
        if manifest.get('run_fingerprint') != fingerprint:
            raise RuntimeError('Diffusion prior fingerprint mismatch')
        payload = torch.load(resolve_local_file(weights_path), map_location='cpu',
                             weights_only=False)
        if payload.get('run_fingerprint') != fingerprint or payload.get('step') != 5000:
            raise RuntimeError('Diffusion prior checkpoint is not the prescribed step-5000 model')
        model = ConditionalNoisePredictor(feature_artifacts['pca']['dimension']).to(device)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        return model, manifest

    cfg = config['diffusion_prior']
    steps = int(cfg['steps'])
    updates = int(cfg['training_updates'])
    train_x = feature_artifacts['train_student_features']
    train_y = identity['labels']['train']
    feedback_x = feature_artifacts['feedback_student_features']
    feedback_y = identity['labels']['feedback']
    diagnostic = _diagnostic_noise(train_x, train_y, feedback_x, feedback_y,
                                   config, roots, fingerprint, seed)
    model = ConditionalNoisePredictor(
        feature_artifacts['pca']['dimension'], classes=2,
        hidden_dimensions=int(cfg['hidden_dimensions']),
        time_dimensions=int(cfg['time_embedding_dimensions']),
        class_dimensions=int(cfg['class_embedding_dimensions'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg['lr']),
        betas=tuple(float(v) for v in cfg['betas']),
        weight_decay=float(cfg['weight_decay']))
    _, alpha, sigma = linear_ddpm_schedule(
        steps, float(cfg['beta_start']), float(cfg['beta_end']), device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 330001 + int(fold) * 1009)
    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.long, device=device)
    batch_size = min(int(cfg['batch_size_cap']), len(train_x_tensor))
    resume_path = roots['run_result'] / 'resume_diffusion.pt'
    start_step, history = 0, []
    if resume_path.is_file():
        saved = torch.load(resolve_local_file(resume_path), map_location='cpu',
                           weights_only=False)
        if saved.get('run_fingerprint') != fingerprint:
            raise RuntimeError('Diffusion resume fingerprint mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        _restore_rng(saved['rng'], device, generator)
        start_step = int(saved['step'])
        history = list(saved.get('history', []))
        if start_step != len(history) * int(cfg['diagnostic_interval']):
            if start_step != updates or start_step != len(history) * int(cfg['diagnostic_interval']):
                raise RuntimeError('Diffusion resume update/history cursor mismatch')
    model.train()
    interval = int(cfg['diagnostic_interval'])
    for step in range(start_step + 1, updates + 1):
        indices = torch.randint(0, len(train_x_tensor), (batch_size,),
                                generator=generator, device=device)
        clean = train_x_tensor[indices]
        labels = train_y_tensor[indices]
        t = torch.randint(1, steps + 1, (batch_size,),
                          generator=generator, device=device)
        eps = torch.randn(clean.shape, dtype=torch.float32,
                          generator=generator, device=device)
        noised = alpha[t - 1, None] * clean + sigma[t - 1, None] * eps
        predicted = model(noised, t, labels)
        loss = (predicted - eps).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Diffusion loss is non-finite at update {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg['gradient_clip_norm']))
        optimizer.step()
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'Diffusion gradient norm is non-finite at update {step}')
        if step % interval == 0 or step == updates:
            diagnostics = _denoiser_diagnostics(
                model, diagnostic, feature_artifacts['gaussian'], alpha, sigma,
                device, int(cfg['diagnostic_noise_bins']))
            row = {'step': step, 'train_batch_mse': float(loss.detach().cpu()),
                   'gradient_norm_pre_clip': float(grad_norm.detach().cpu()), **diagnostics}
            history.append(row)
            _atomic_csv(roots['run_result'] / 'diffusion_history.csv', history)
            _atomic_json(roots['run_result'] / 'diffusion_history.json', history)
            _progress(progress_path, run_id=run_id, fold=fold, stage='diffusion_prior',
                      completed=step, total=updates, message=f'update {step}/{updates}')
            payload = {
                'run_fingerprint': fingerprint, 'step': step,
                'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'rng': _capture_rng(device, generator),
                'history': history,
            }
            _atomic_torch(resume_path, payload)
            if _stopped(stop_file):
                return None, {'status': 'paused', 'completed_step': step}
    model.eval()
    state_hash = _state_hash(model.state_dict())
    payload = {'run_fingerprint': fingerprint, 'step': updates,
               'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
               'model_sha256': state_hash, 'feature_dimension': feature_artifacts['pca']['dimension'],
               'history': history}
    _atomic_torch(weights_path, payload)
    manifest = {
        'status': 'complete', 'run_fingerprint': fingerprint,
        'step': updates, 'model_sha256': state_hash,
        'checkpoint': str(weights_path), 'checkpoint_sha256': _sha256(weights_path),
        'fit_partition': 'D_train only', 'train_uid_sha256': identity['uid_sha256']['train'],
        'architecture': {'hidden_dimensions': cfg['hidden_dimensions'],
                         'time_embedding_dimensions': cfg['time_embedding_dimensions'],
                         'class_embedding_dimensions': cfg['class_embedding_dimensions']},
        'noise_schedule': {'steps': steps, 'beta_start': cfg['beta_start'],
                           'beta_end': cfg['beta_end']},
        'optimizer': {'name': 'AdamW', 'lr': cfg['lr'], 'betas': cfg['betas'],
                      'weight_decay': cfg['weight_decay'],
                      'gradient_clip_norm': cfg['gradient_clip_norm']},
        'batch_size': batch_size,
        'updates': updates,
        'diagnostic_noise_sha256': _sha256(roots['run_cache'] / 'diffusion_diagnostic_noise.npz'),
        'history_path': str(roots['run_result'] / 'diffusion_history.csv'),
    }
    _atomic_json(manifest_path, manifest)
    if resume_path.is_file():
        resume_path.unlink()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    return model, manifest


def _softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    probs = np.exp(values)
    return probs / probs.sum(axis=1, keepdims=True)


def _kl_np(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-12))
                                   - np.log(np.maximum(q, 1e-12))), axis=1)))


def _teacher_diagnostics(method: str, model: torch.nn.Module, adapter,
                         identity: dict, roots: dict, reference: dict,
                         data: dict, feature_artifacts: dict,
                         device: torch.device, epoch: int) -> dict:
    feature_feedback, logits_feedback = _infer(
        adapter, model, data['teacher_x']['feedback'], device, batch_size=32)
    y_feedback = identity['labels']['feedback']
    t0_feedback = reference['feedback_teacher_logits']
    s0_feedback = reference['feedback_student_logits']
    p0 = _softmax_np(t0_feedback, 2.0)
    p_now = _softmax_np(logits_feedback, 2.0)
    p_s0 = _softmax_np(s0_feedback, 2.0)
    projected_feedback = apply_ridge_projection(
        feature_feedback, feature_artifacts['projection'])
    student_feedback = feature_artifacts['feedback_student_features']
    bandwidth = feature_artifacts['diagnostics']['bandwidth_squared_median_train']
    result = {
        'method': method, 'epoch': int(epoch),
        'feedback_accuracy': _metric_values(y_feedback, logits_feedback)['accuracy'],
        'feedback_balanced_accuracy': _metric_values(y_feedback, logits_feedback)['balanced_accuracy'],
        'feedback_cross_entropy': _metric_values(y_feedback, logits_feedback)['cross_entropy'],
        'feedback_teacher_t0_prediction_agreement': float(
            np.mean(logits_feedback.argmax(axis=1) == t0_feedback.argmax(axis=1))),
        'feedback_teacher_s0_prediction_agreement': float(
            np.mean(logits_feedback.argmax(axis=1) == s0_feedback.argmax(axis=1))),
        'feedback_kl_t0_to_adapted_tau2': _kl_np(p0, p_now),
        'feedback_kl_adapted_to_s0_tau2': _kl_np(p_now, p_s0),
        'feedback_hard_flip_rate_from_t0': float(
            np.mean(logits_feedback.argmax(axis=1) != t0_feedback.argmax(axis=1))),
        'feedback_feature_mmd_to_s0': _rbf_mmd(
            student_feedback, projected_feedback, bandwidth),
        'feedback_feature_effective_rank': _effective_rank(projected_feedback),
        'feedback_feature_covariance_trace': float(
            np.var(projected_feedback, axis=0, ddof=1).sum()),
    }
    if epoch in (0, 10):
        feature_train, _ = _infer(
            adapter, model, data['teacher_x']['train'], device, batch_size=32)
        projected_train = apply_ridge_projection(
            feature_train, feature_artifacts['projection'])
        student_train = feature_artifacts['train_student_features']
        result.update({
            'train_feature_mmd_to_s0': _rbf_mmd(student_train, projected_train, bandwidth),
            'train_feature_effective_rank': _effective_rank(projected_train),
            'train_feature_covariance_trace': float(
                np.var(projected_train, axis=0, ddof=1).sum()),
        })
    return result


def _adapt_teacher(method: str, identity: dict, roots: dict, config: dict,
                   models: dict, reference: dict, data: dict,
                   feature_artifacts: dict,
                   diffusion_prior: ConditionalNoisePredictor | None,
                   run_fingerprint: str, seed: int, device: torch.device,
                   stop_file: str | None, progress_path: str | None,
                   run_id: str, fold: int, max_epochs: int | None = None,
                   lambda_override: float | None = None) -> tuple[torch.nn.Module | None, dict]:
    if method not in TEACHER_METHODS:
        raise ValueError(f'unknown Teacher adaptation method {method}')
    final_path = roots['run_result'] / f'teacher_{method}.pt'
    manifest_path = roots['run_result'] / f'teacher_{method}.json'
    target_path = roots['run_cache'] / f'teacher_{method}_train_targets.npz'
    method_fingerprint = _json_hash({
        'run_fingerprint': run_fingerprint, 'method': method,
        'teacher_checkpoint_sha256': models['teacher_checkpoint_sha256'],
        'teacher_schedule_sha256': _schedule_sha(
            _epoch_schedule(len(identity['uids']['train']), 8, 10, seed + 21001 + fold),
            identity['uids']['train']),
        'prior_lambda': (float(lambda_override) if lambda_override is not None
                         else float(config['feature_prior_adaptation']['lambda'])),
    })
    if final_path.is_file() or manifest_path.is_file():
        if not (final_path.is_file() and manifest_path.is_file()):
            raise RuntimeError(f'{method}: incomplete Teacher result; use a new run identity')
        manifest = _read_json(manifest_path)
        if (manifest.get('run_fingerprint') != method_fingerprint
                or manifest.get('checkpoint_sha256') != _sha256(final_path)):
            raise RuntimeError(f'{method}: Teacher checkpoint provenance mismatch')
        payload = torch.load(resolve_local_file(final_path), map_location='cpu',
                             weights_only=False)
        model = copy.deepcopy(models['teacher_model'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        return model, manifest

    epochs = int(max_epochs if max_epochs is not None else
                 config['teacher_adaptation']['epochs'])
    teacher_cfg = config['teacher_adaptation']
    train_schedule = _epoch_schedule(
        len(identity['uids']['train']), int(teacher_cfg['batch_size']),
        int(config['teacher_adaptation']['epochs']), seed + 21001 + fold)
    # One immutable UID batch order is shared by C--G.
    if _schedule_sha(train_schedule, identity['uids']['train']) != _schedule_sha(
            _epoch_schedule(len(identity['uids']['train']), int(teacher_cfg['batch_size']),
                            int(config['teacher_adaptation']['epochs']), seed + 21001 + fold),
            identity['uids']['train']):
        raise RuntimeError('Teacher batch schedule is not deterministic')
    t0_hash_before = _state_hash(models['teacher_model'].state_dict())
    model = copy.deepcopy(models['teacher_model'])
    model.eval()  # Dropout disabled; BN buffers fixed, autograd remains enabled.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(teacher_cfg['lr']),
        betas=tuple(float(v) for v in teacher_cfg['betas']),
        weight_decay=float(teacher_cfg['weight_decay']))
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(int(seed) + 99001 + int(fold) * 1009)
    resume_path = roots['run_result'] / f'resume_teacher_{method}.pt'
    start_epoch, history = 0, []
    if resume_path.is_file():
        saved = torch.load(resolve_local_file(resume_path), map_location='cpu',
                           weights_only=False)
        if saved.get('run_fingerprint') != method_fingerprint:
            raise RuntimeError(f'{method}: Teacher resume fingerprint mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        _restore_rng(saved['rng'], device, noise_generator)
        start_epoch = int(saved['epoch'])
        history = list(saved.get('history', []))
        if len(history) != start_epoch:
            raise RuntimeError(f'{method}: Teacher resume cursor/history mismatch')

    reference_train = {
        't0_logits': torch.as_tensor(reference['train_teacher_logits'],
                                     dtype=torch.float32, device=device),
        's0_logits': torch.as_tensor(reference['train_student_logits'],
                                     dtype=torch.float32, device=device),
    }
    train_x = data['teacher_x']['train']
    student_x = data['student_x']['train']
    train_y = identity['labels']['train']
    pca = feature_artifacts['pca']
    projection = feature_artifacts['projection']
    t_mean = torch.as_tensor(projection['teacher_mean'], dtype=torch.float32, device=device)
    t_std = torch.as_tensor(projection['teacher_std'], dtype=torch.float32, device=device)
    p_weight = torch.as_tensor(projection['weight'], dtype=torch.float32, device=device)
    p_bias = torch.as_tensor(projection['bias'], dtype=torch.float32, device=device)
    for fixed in (t_mean, t_std, p_weight, p_bias):
        fixed.requires_grad_(False)
    alpha: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    gaussian_means: torch.Tensor | None = None
    gaussian_vars: torch.Tensor | None = None
    if method in ('F', 'G'):
        _, alpha, sigma = linear_ddpm_schedule(
            int(config['diffusion_prior']['steps']),
            float(config['diffusion_prior']['beta_start']),
            float(config['diffusion_prior']['beta_end']), device)
        if method == 'G':
            gaussian_means = torch.as_tensor(feature_artifacts['gaussian']['means'],
                                             dtype=torch.float32, device=device)
            gaussian_vars = torch.as_tensor(feature_artifacts['gaussian']['variances'],
                                            dtype=torch.float32, device=device)
        elif diffusion_prior is None:
            raise RuntimeError('G requires a completed frozen Diffusion prior')
        else:
            diffusion_prior.eval()
            for parameter in diffusion_prior.parameters():
                parameter.requires_grad_(False)

    diagnostic_rows = list(saved.get('diagnostics', [])) if resume_path.is_file() else []
    if start_epoch == 0 and max_epochs is None:
        diagnostic_rows.append(_teacher_diagnostics(
            method, model, models['teacher_adapter'], identity, roots, reference,
            data, feature_artifacts, device, 0))
    warmup = float(config['feature_prior_adaptation']['lambda']
                   if lambda_override is None else lambda_override)
    for epoch_index in range(start_epoch, epochs):
        epoch_no = epoch_index + 1
        totals = {'ce': 0.0, 'keep': 0.0, 'aid': 0.0, 'prior': 0.0,
                  'samples': 0, 'grad_norm': 0.0}
        gradient_diagnostics = {}
        for batch_index, indices in enumerate(train_schedule[epoch_index]):
            batch_ids = torch.as_tensor(indices, dtype=torch.long, device=device)
            xb_t = torch.as_tensor(train_x[indices], dtype=torch.float32, device=device)
            xb_s = torch.as_tensor(student_x[indices], dtype=torch.float32, device=device)
            yb = torch.as_tensor(train_y[indices], dtype=torch.long, device=device)
            ref_t0 = reference_train['t0_logits'][batch_ids]
            ref_s0 = reference_train['s0_logits'][batch_ids]
            feature, logits = models['teacher_adapter'].forward(model, xb_t)
            ce = F.cross_entropy(logits, yb)
            keep = _kl_teacher_to_student(
                ref_t0, logits, float(teacher_cfg['keep_temperature']))
            aid = _aid_student_to_teacher(
                logits, ref_s0, float(teacher_cfg['aid_temperature']))
            prior = logits.sum() * 0.0
            if method in ('F', 'G') and warmup > 0:
                projected = F.linear((feature - t_mean) / t_std, p_weight, p_bias)
                min_t = int(config['feature_prior_adaptation']['timestep_min'])
                max_t = int(config['feature_prior_adaptation']['timestep_max'])
                t = torch.randint(min_t, max_t + 1, (len(yb),),
                                  generator=noise_generator, device=device)
                eps = torch.randn(projected.shape, dtype=torch.float32,
                                  generator=noise_generator, device=device)
                a = alpha[t - 1, None]
                s = sigma[t - 1, None]
                noisy = a * projected.detach() + s * eps
                with torch.no_grad():
                    if method == 'G':
                        predicted_eps = gaussian_epsilon(
                            noisy, t, yb, gaussian_means, gaussian_vars, alpha, sigma)
                    else:
                        predicted_eps = diffusion_prior(noisy, t, yb)
                    direction = sds_direction(predicted_eps, eps, t, alpha, sigma)
                prior = sds_surrogate(projected, direction)
            loss_keep = float(teacher_cfg['keep_weight']) * keep if method in ('C', 'E', 'F', 'G') else keep * 0.0
            loss_aid = float(teacher_cfg['aid_weight']) * aid if method in ('D', 'E') else aid * 0.0
            loss_prior = warmup * prior if method in ('F', 'G') else prior * 0.0
            loss = ce + loss_keep + loss_aid + loss_prior
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{method} Teacher loss is non-finite at epoch {epoch_no}')
            optimizer.zero_grad(set_to_none=True)
            if batch_index == 0 and epoch_index == 0 and method in ('F', 'G') and warmup > 0:
                parameters = [p for p in model.parameters() if p.requires_grad]
                ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True,
                                               allow_unused=True)
                prior_grads = torch.autograd.grad(loss_prior, parameters, retain_graph=True,
                                                  allow_unused=True)
                ce_norm = torch.linalg.vector_norm(torch.stack([
                    grad.detach().norm() for grad in ce_grads if grad is not None]))
                prior_norm = torch.linalg.vector_norm(torch.stack([
                    grad.detach().norm() for grad in prior_grads if grad is not None]))
                gradient_diagnostics = {
                    'first_batch_ce_grad_norm': float(ce_norm.cpu()),
                    'first_batch_weighted_prior_grad_norm': float(prior_norm.cpu()),
                    'first_batch_prior_to_ce_grad_ratio': float(
                        (prior_norm / ce_norm.clamp_min(1e-12)).cpu()),
                }
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(teacher_cfg['gradient_clip_norm']))
            optimizer.step()
            count = len(indices)
            totals['samples'] += count
            totals['ce'] += float(ce.detach().cpu()) * count
            totals['keep'] += float(keep.detach().cpu()) * count
            totals['aid'] += float(aid.detach().cpu()) * count
            totals['prior'] += float(prior.detach().cpu()) * count
            totals['grad_norm'] += float(grad_norm.detach().cpu())
        row = {
            'epoch': epoch_no,
            'ce': totals['ce'] / totals['samples'],
            'keep_kl_tau2': totals['keep'] / totals['samples'],
            'aid_kl_tau2': totals['aid'] / totals['samples'],
            'prior_surrogate': totals['prior'] / totals['samples'],
            'weighted_keep': float(teacher_cfg['keep_weight']) * totals['keep'] / totals['samples']
            if method in ('C', 'E', 'F', 'G') else 0.0,
            'weighted_aid': float(teacher_cfg['aid_weight']) * totals['aid'] / totals['samples']
            if method in ('D', 'E') else 0.0,
            'weighted_prior': warmup * totals['prior'] / totals['samples']
            if method in ('F', 'G') else 0.0,
            'gradient_norm_pre_clip_mean_over_batches': totals['grad_norm'] / len(train_schedule[epoch_index]),
            'schedule_sha256': _schedule_sha(train_schedule, identity['uids']['train']),
            **gradient_diagnostics,
        }
        history.append(row)
        if epoch_no in (1, 5, 10) and max_epochs is None:
            diagnostic_rows.append(_teacher_diagnostics(
                method, model, models['teacher_adapter'], identity, roots, reference,
                data, feature_artifacts, device, epoch_no))
        _atomic_csv(roots['run_result'] / f'teacher_{method}_history.csv', history)
        if diagnostic_rows:
            _atomic_csv(roots['run_result'] / f'teacher_{method}_source_diagnostics.csv',
                        diagnostic_rows)
        _progress(progress_path, run_id=run_id, fold=fold, stage=f'teacher_{method}',
                  completed=epoch_no, total=epochs, message=f'epoch {epoch_no}/{epochs}')
        state = {
            'run_fingerprint': method_fingerprint, 'epoch': epoch_no,
            'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'rng': _capture_rng(device, noise_generator), 'history': history,
            'diagnostics': diagnostic_rows,
        }
        _atomic_torch(resume_path, state)
        if _state_hash(models['teacher_model'].state_dict()) != t0_hash_before:
            raise RuntimeError('T0 changed while adapting its independent copy')
        if _stopped(stop_file):
            return None, {'status': 'paused', 'completed_epoch': epoch_no}

    model.eval()
    train_features, train_logits = _infer(
        models['teacher_adapter'], model, data['teacher_x']['train'], device, batch_size=32)
    checkpoint_sha = _state_hash(model.state_dict())
    payload = {'run_fingerprint': method_fingerprint,
               'method': METHODS[method], 'method_code': method,
               'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
               'model_sha256': checkpoint_sha, 'final_epoch': epochs,
               'initial_teacher_checkpoint_sha256': models['teacher_checkpoint_sha256']}
    _atomic_torch(final_path, payload)
    _atomic_npz(target_path, uids=identity['uids']['train'],
                y=identity['labels']['train'], logits=train_logits,
                features=train_features)
    manifest = {
        'status': 'complete', 'run_fingerprint': method_fingerprint,
        'method': METHODS[method], 'method_code': method,
        'target_subject_id': int(identity['manifest']['target_subject_id']),
        'feedback_subject_id': int(identity['manifest']['feedback_subject_id']),
        'train_uid_sha256': identity['uid_sha256']['train'],
        'teacher_checkpoint': str(final_path), 'checkpoint_sha256': _sha256(final_path),
        'model_sha256': checkpoint_sha, 'initial_T0_checkpoint_sha256': models['teacher_checkpoint_sha256'],
        'epochs': epochs, 'optimizer': {'name': 'Adam', 'lr': teacher_cfg['lr'],
                                        'betas': teacher_cfg['betas'],
                                        'weight_decay': teacher_cfg['weight_decay'],
                                        'scheduler': 'none'},
        'forward_mode': 'eval() with autograd; dropout off; BN running statistics frozen',
        'loss': {'true_label_ce': 1.0,
                 'teacher_keep_weight': float(teacher_cfg['keep_weight'])
                 if method in ('C', 'E', 'F', 'G') else 0.0,
                 'aid_weight': float(teacher_cfg['aid_weight'])
                 if method in ('D', 'E') else 0.0,
                 'feature_prior_lambda': warmup if method in ('F', 'G') else 0.0},
        'teacher_schedule_sha256': _schedule_sha(train_schedule, identity['uids']['train']),
        'train_targets_cache': str(target_path),
        'train_targets_sha256': _sha256(target_path),
        'source_diagnostics': str(roots['run_result'] / f'teacher_{method}_source_diagnostics.csv'),
        'history': str(roots['run_result'] / f'teacher_{method}_history.csv'),
    }
    _atomic_json(manifest_path, manifest)
    if resume_path.is_file():
        resume_path.unlink()
    if _state_hash(models['teacher_model'].state_dict()) != t0_hash_before:
        raise RuntimeError('T0 changed after Teacher adaptation')
    for parameter in (models['teacher_model'].parameters()):
        if parameter.grad is not None:
            parameter.grad = None
    return model, manifest


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _ensure_student_initial_state(roots: dict, resolved: dict,
                                  cfg: dict, seed: int) -> dict:
    """Create/read one canonical, untrained IFNet I_s for all folds and arms."""
    path = roots['run_seed_weights'] / f'ifnet_initial_seed_{int(seed)}.pt'
    lock_path = roots['run_seed_weights'] / f'.ifnet_initial_seed_{int(seed)}.lock'
    path = require_external_output(path)
    lock_path = require_external_output(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        expected = {
            'seed': int(seed),
            'resolved_config_sha256': resolved['resolved_config_sha256'],
            'student_config_sha256': _json_hash(resolved['student_config']),
        }
        if path.is_file():
            payload = torch.load(resolve_local_file(path), map_location='cpu',
                                 weights_only=False)
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise RuntimeError(f'canonical IFNet initial state {key} mismatch')
            if _state_hash(payload['model']) != payload.get('model_sha256'):
                raise RuntimeError('canonical IFNet initial state hash mismatch')
        else:
            cpu = torch.device('cpu')
            _seed_everything(seed, cpu)
            adapter = get_adapter('ifnet', device=cpu, **resolved['student_config'])
            model = adapter.build(2)
            state = {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()}
            payload = {**expected, 'model': state, 'model_sha256': _state_hash(state),
                       'architecture': 'IFNet', 'initialization': 'seeded adapter.build(2)',
                       'created_by': Path(__file__).name}
            _atomic_torch(path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {'path': str(path), 'file_sha256': _sha256(path),
            'model_sha256': payload['model_sha256'], 'payload': payload}


def _new_paired_student(models: dict, initial: dict, seed: int,
                        device: torch.device):
    """Instantiate a fresh IFNet from I_s and reproduce its initial RNG stream."""
    _seed_everything(seed, device)
    adapter = models['student_adapter']
    model = adapter.build(2)
    freshly_initialized_hash = _state_hash(model.state_dict())
    rng_after_build = _capture_rng(device)
    if freshly_initialized_hash != initial['model_sha256']:
        raise RuntimeError('fresh IFNet initialization does not reproduce canonical I_s')
    model.load_state_dict(initial['payload']['model'], strict=True)
    if _state_hash(model.state_dict()) != initial['model_sha256']:
        raise RuntimeError('loaded IFNet initial state differs from canonical I_s')
    _restore_rng(rng_after_build, device)
    model.train()
    opt_cfg = models['optimizer_defaults']['student']
    student_cfg = models['student_cfg']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(student_cfg['lr']),
        weight_decay=float(student_cfg['weight_decay']),
        betas=tuple(float(v) for v in opt_cfg['betas']),
        eps=float(opt_cfg['eps']), amsgrad=bool(opt_cfg['amsgrad']),
        foreach=bool(opt_cfg['foreach']), fused=bool(opt_cfg['fused']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(student_cfg['epochs']))
    return model, optimizer, scheduler, rng_after_build


def _verify_s0_initialization(identity: dict, roots: dict, resolved: dict,
                              models: dict, data: dict, initial: dict,
                              seed: int, device: torch.device,
                              run_fingerprint: str) -> dict:
    """Replay the first ten CE epochs and verify historical S0 came from I_s."""
    audit_path = roots['run_result'] / 'student_initialization_audit.json'
    if audit_path.is_file():
        old = _read_json(audit_path)
        if (old.get('run_fingerprint') != run_fingerprint
                or old.get('initial_model_sha256') != initial['model_sha256']
                or old.get('replayed_warmup_sha256')
                != models['warmup_payload']['warmup_state_sha256']):
            raise RuntimeError('existing S0 initialization audit does not match this run')
        return old

    model, optimizer, scheduler, _ = _new_paired_student(models, initial, seed, device)
    train_x = torch.as_tensor(data['student_x']['train'], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(identity['labels']['train'], dtype=torch.long, device=device)
    schedule = models['student_schedule']
    for epoch in range(10):
        model.train()
        for ids in schedule[epoch]:
            index = torch.as_tensor(ids, dtype=torch.long, device=device)
            xb, yb = train_x[index], train_y[index]
            optimizer.zero_grad(set_to_none=True)
            _, logits = models['student_adapter'].forward(model, xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    replay_hash = _state_hash(model.state_dict())
    expected_hash = models['warmup_payload']['warmup_state_sha256']
    if replay_hash != expected_hash:
        raise RuntimeError(
            'historical S0 does not match a ten-epoch CE replay from this I_s; '
            'cannot use it as the V1 paired reference')
    record = {
        'status': 'verified', 'run_fingerprint': run_fingerprint,
        'initial_state_path': initial['path'],
        'initial_state_file_sha256': initial['file_sha256'],
        'initial_model_sha256': initial['model_sha256'],
        'replayed_warmup_sha256': replay_hash,
        'historical_warmup_sha256': expected_hash,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'schedule_sha256': models['student_schedule_sha256'],
        'warmup_epochs_replayed': 10,
        'matched': True,
    }
    _atomic_json(audit_path, record)
    return record


def _student_feedback_record(method: str, epoch: int, model, models: dict,
                             identity: dict, data: dict, device: torch.device) -> dict:
    _, logits = _infer(models['student_adapter'], model, data['student_x']['feedback'],
                       device, batch_size=64)
    metrics = _metric_values(identity['labels']['feedback'], logits)
    return {'method_code': method, 'method': METHODS[method], 'epoch': int(epoch),
            'feedback_accuracy': metrics['accuracy'],
            'feedback_balanced_accuracy': metrics['balanced_accuracy'],
            'feedback_cross_entropy': metrics['cross_entropy'],
            'feedback_prediction_counts': json.dumps(metrics['prediction_counts']),
            'feedback_collapse': metrics['collapse']}


def _teacher_train_targets(method: str, roots: dict, identity: dict,
                           reference: dict, teacher_manifest: dict | None) -> tuple[np.ndarray, dict]:
    if method == 'B':
        logits = np.asarray(reference['train_teacher_logits'], dtype=np.float32)
        manifest = {'method': 'T0', 'checkpoint_sha256': teacher_manifest.get(
            'teacher_checkpoint_sha256') if teacher_manifest else None,
                    'target_source': 'reference_train_feedback.npz'}
        return logits, manifest
    if teacher_manifest is None or teacher_manifest.get('status') != 'complete':
        raise RuntimeError(f'{method}: adapted Teacher is not complete')
    path = Path(teacher_manifest['train_targets_cache'])
    if _sha256(path) != teacher_manifest['train_targets_sha256']:
        raise RuntimeError(f'{method}: adapted Teacher train-target cache hash mismatch')
    with np.load(resolve_local_file(path), allow_pickle=False) as saved:
        uids = np.asarray(saved['uids'], dtype=np.int64)
        labels = np.asarray(saved['y'], dtype=np.int64)
        logits = np.asarray(saved['logits'], dtype=np.float32)
    if (not np.array_equal(uids, identity['uids']['train'])
            or not np.array_equal(labels, identity['labels']['train'])
            or logits.shape != (len(uids), 2) or not np.isfinite(logits).all()):
        raise RuntimeError(f'{method}: adapted Teacher logits are not aligned to D_train')
    return logits, teacher_manifest


def _fit_student(method: str, identity: dict, roots: dict, config: dict,
                 models: dict, reference: dict, data: dict, initial: dict,
                 teacher_manifest: dict | None, run_fingerprint: str,
                 seed: int, device: torch.device, stop_file: str | None,
                 progress_path: str | None, run_id: str, fold: int,
                 max_epochs: int | None = None) -> dict:
    if method not in STUDENT_METHODS:
        raise ValueError(f'unknown Student method {method}')
    final_path = roots['run_weights'] / f'student_{method}.pt'
    manifest_path = roots['run_result'] / f'student_{method}.json'
    target_logits, target_manifest = _teacher_train_targets(
        method, roots, identity, reference, teacher_manifest)
    method_fingerprint = _json_hash({
        'run_fingerprint': run_fingerprint, 'method': method,
        'initial_model_sha256': initial['model_sha256'],
        'teacher_targets_sha256': (teacher_manifest or {}).get('train_targets_sha256'),
        'teacher_checkpoint_sha256': (teacher_manifest or {}).get(
            'checkpoint_sha256', models['teacher_checkpoint_sha256']),
        'student_schedule_sha256': models['student_schedule_sha256'],
        'kd_weight': float(config['student']['kd_weight']),
        'temperature': float(config['student']['temperature']),
    })
    if final_path.is_file() or manifest_path.is_file():
        if not (final_path.is_file() and manifest_path.is_file()):
            raise RuntimeError(f'{method}: incomplete final Student; use resume or a fresh run ID')
        manifest = _read_json(manifest_path)
        if (manifest.get('run_fingerprint') != method_fingerprint
                or manifest.get('checkpoint_sha256') != _sha256(final_path)):
            raise RuntimeError(f'{method}: final Student provenance mismatch')
        return manifest

    epochs = int(max_epochs if max_epochs is not None else config['student']['epochs'])
    if epochs < 1 or epochs > int(config['student']['epochs']):
        raise ValueError(f'invalid Student epoch count {epochs}')
    model, optimizer, scheduler, _ = _new_paired_student(models, initial, seed, device)
    resume_path = roots['run_result'] / f'resume_student_{method}.pt'
    start_epoch, history, diagnostics = 0, [], []
    if resume_path.is_file():
        saved = torch.load(resolve_local_file(resume_path), map_location='cpu',
                           weights_only=False)
        if saved.get('run_fingerprint') != method_fingerprint:
            raise RuntimeError(f'{method}: Student resume fingerprint mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        _restore_rng(saved['rng'], device)
        start_epoch = int(saved['epoch'])
        history = list(saved.get('history', []))
        diagnostics = list(saved.get('diagnostics', []))
        if len(history) != start_epoch:
            raise RuntimeError(f'{method}: Student resume cursor/history mismatch')

    train_x = data['student_x']['train']
    train_y = identity['labels']['train']
    teacher_logits_device = torch.as_tensor(target_logits, dtype=torch.float32, device=device)
    schedule = models['student_schedule']
    student_cfg = config['student']
    kd_weight, temperature = float(student_cfg['kd_weight']), float(student_cfg['temperature'])
    for epoch_index in range(start_epoch, epochs):
        model.train()
        seen = 0
        total_ce = total_kd = total_loss = total_grad_norm = 0.0
        for ids in schedule[epoch_index]:
            xb = torch.as_tensor(train_x[ids], dtype=torch.float32, device=device)
            yb = torch.as_tensor(train_y[ids], dtype=torch.long, device=device)
            batch_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
            teacher_batch = teacher_logits_device[batch_ids]
            optimizer.zero_grad(set_to_none=True)
            _, student_logits = models['student_adapter'].forward(model, xb)
            ce = F.cross_entropy(student_logits, yb)
            kd = _kl_teacher_to_student(teacher_batch, student_logits, temperature)
            loss = ce + kd_weight * kd
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{method} Student loss is non-finite')
            loss.backward()
            grad_norm = torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().norm() for parameter in model.parameters()
                if parameter.grad is not None]))
            optimizer.step()
            count = len(ids)
            seen += count
            total_ce += float(ce.detach().cpu()) * count
            total_kd += float(kd.detach().cpu()) * count
            total_loss += float(loss.detach().cpu()) * count
            total_grad_norm += float(grad_norm.detach().cpu())
        lr_used = float(optimizer.param_groups[0]['lr'])
        scheduler.step()
        epoch_no = epoch_index + 1
        row = {
            'method_code': method, 'method': METHODS[method], 'epoch': epoch_no,
            'train_ce': total_ce / seen, 'train_kd_tau2': total_kd / seen,
            'train_total': total_loss / seen,
            'gradient_norm_mean_over_batches': total_grad_norm / len(schedule[epoch_index]),
            'lr_used': lr_used, 'lr_next': float(scheduler.get_last_lr()[0]),
            'n_train': seen,
            'schedule_sha256': _schedule_sha(schedule, identity['uids']['train']),
        }
        history.append(row)
        if epoch_no in (1, 10, 50, 100) and max_epochs is None:
            diagnostics.append(_student_feedback_record(
                method, epoch_no, model, models, identity, data, device))
        _atomic_csv(roots['run_result'] / f'student_{method}_history.csv', history)
        if diagnostics:
            _atomic_csv(roots['run_result'] / f'student_{method}_feedback_diagnostics.csv',
                        diagnostics)
        _progress(progress_path, run_id=run_id, fold=fold, stage=f'student_{method}',
                  completed=epoch_no, total=epochs, message=f'epoch {epoch_no}/{epochs}')
        _atomic_torch(resume_path, {
            'run_fingerprint': method_fingerprint, 'epoch': epoch_no,
            'model': _cpu_tree(model.state_dict()),
            'optimizer': _cpu_tree(optimizer.state_dict()),
            'scheduler': scheduler.state_dict(), 'rng': _capture_rng(device),
            'history': history, 'diagnostics': diagnostics,
        })
        if _stopped(stop_file):
            return {'status': 'paused', 'completed_epoch': epoch_no}

    model.eval()
    state_hash = _state_hash(model.state_dict())
    _atomic_torch(final_path, {
        'run_fingerprint': method_fingerprint, 'method': METHODS[method],
        'method_code': method, 'final_epoch': epochs,
        'initial_model_sha256': initial['model_sha256'],
        'model_sha256': state_hash, 'model': _cpu_tree(model.state_dict()),
    })
    manifest = {
        'status': 'complete', 'run_fingerprint': method_fingerprint,
        'method': METHODS[method], 'method_code': method,
        'target_subject_id': int(identity['manifest']['target_subject_id']),
        'feedback_subject_id': int(identity['manifest']['feedback_subject_id']),
        'train_uid_sha256': identity['uid_sha256']['train'],
        'final_epoch': epochs, 'initial_model_sha256': initial['model_sha256'],
        'student_checkpoint': str(final_path), 'checkpoint_sha256': _sha256(final_path),
        'model_sha256': state_hash,
        'student_config': models['student_cfg'],
        'optimizer': {'name': 'AdamW',
                      'lr': models['student_cfg']['lr'],
                      'weight_decay': models['student_cfg']['weight_decay'],
                      'defaults': models['optimizer_defaults']['student'],
                      'scheduler': 'CosineAnnealingLR, T_max=100'},
        'loss': {'ce_weight': 1.0, 'kd_weight': kd_weight,
                 'temperature': temperature, 'kd_start_epoch': 1},
        'teacher_target': target_manifest,
        'student_schedule_sha256': models['student_schedule_sha256'],
        'history': str(roots['run_result'] / f'student_{method}_history.csv'),
        'feedback_diagnostics': str(
            roots['run_result'] / f'student_{method}_feedback_diagnostics.csv'),
    }
    _atomic_json(manifest_path, manifest)
    resume_path.unlink(missing_ok=True)
    return manifest


def _fold_fingerprint(config: dict, config_path: Path, resolved: dict,
                      cfg: dict, identity: dict, roots: dict, run_id: str) -> str:
    teacher_path = roots['base_teacher'] / 'teacher_final.pt'
    student_path = roots['base_results'] / 'BASE_CE' / 'checkpoint_final.pt'
    return _json_hash({
        'experiment_id': config['experiment_id'], 'run_id': run_id,
        'config_sha256': _sha256(config_path),
        'resolved_config_sha256': cfg['resolved_config_sha256'],
        'resolved_yaml_sha256': cfg['resolved_yaml_sha256'],
        'implementation_sha256': _sha256(__file__),
        'prior_helper_sha256': _sha256(PROJECT_ROOT / 'collab/student_feature_prior.py'),
        'split_manifest_sha256': identity['sha256'],
        'source_manifest_sha256': identity['source_manifest_sha256'],
        'source_trials_sha256': identity['source_trial_manifest_sha256'],
        'teacher_input_manifest_sha256': _sha256(roots['input_root'] / 'mirepnet/manifest.json'),
        'student_input_manifest_sha256': _sha256(roots['input_root'] / 'ifnet/manifest.json'),
        'prepared_student_inputs_sha256': _sha256(roots['split_dir'] / 'student_inputs.npz'),
        'teacher_checkpoint_sha256': _sha256(teacher_path),
        'student_baseline_checkpoint_sha256': _sha256(student_path),
        'teacher_provenance_sha256': _sha256(roots['base_teacher'] / 'teacher_provenance.json'),
        'baseline_manifest_sha256': _sha256(roots['base_results'] / 'BASE_CE/manifest.json'),
        'seed': int(config['seed']), 'target_subject_id': int(identity['manifest']['target_subject_id']),
    })


def _save_resolved_record(roots: dict, config: dict, config_path: Path,
                          resolved: dict, cfg: dict, fingerprint: str) -> None:
    path = roots['run_result'] / 'resolved_run.json'
    if path.is_file():
        old = _read_json(path)
        if old.get('run_fingerprint_inputs') != fingerprint:
            raise RuntimeError('run-level resolved provenance changed after jobs were queued')
        return
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    record = {
        'experiment_id': config['experiment_id'], 'run_id': config['storage']['external_run_id'],
        'config_path': str(config_path), 'config_sha256': _sha256(config_path),
        'resolved_reference': resolved, 'reference_provenance': cfg,
        'implementation_sha256': _sha256(__file__),
        'feature_prior_helper_sha256': _sha256(PROJECT_ROOT / 'collab/student_feature_prior.py'),
        'git_commit_at_start': commit, 'run_fingerprint_inputs': fingerprint,
        'target_evaluation_policy': 'not read during fold training; separate evaluator after all 9 folds complete',
        'cpu_threads_per_process': 1, 'dataloader_workers': 0,
    }
    _atomic_json(path, record)


def train_fold(config_path: Path, run_id: str, fold: int, gpu: int = 0,
               progress_path: str | None = None, stop_file: str | None = None) -> int:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    _validate_v1_config(config)
    if not 0 <= int(fold) < 9:
        raise ValueError('fold must be a zero-based target subject index in 0..8')
    seed = int(config['seed'])
    device = _device(gpu)
    roots = _run_roots(run_id, seed, int(fold))
    for key in ('run_result', 'run_cache', 'run_weights', 'run_seed_weights'):
        require_external_output(roots[key]).mkdir(parents=True, exist_ok=True)
    resolved, cfg = _load_resolved_reference(config)
    identity = _load_fold_identity(
        config, int(fold), roots, cfg['resolved_config_sha256'], include_test_labels=False)
    fingerprint = _fold_fingerprint(config, config_path, resolved, cfg, identity, roots, run_id)
    complete_path = roots['run_result'] / 'fold_complete.json'
    if complete_path.is_file():
        done = _read_json(complete_path)
        if done.get('run_fingerprint') != fingerprint:
            raise RuntimeError('existing completed fold belongs to another experiment identity')
        _progress(progress_path, run_id=run_id, fold=fold, stage='fold_complete',
                  completed=1, total=1, status='completed', message='validated existing fold')
        return 0

    data = _load_fold_data(identity, roots, resolved['student_config'], device)
    models = _load_reference_checkpoints(config, resolved, cfg, identity, roots, device)
    reference = _reference_cache(identity, roots, models, data, fingerprint, device)
    initial = _ensure_student_initial_state(roots, resolved, cfg, seed)
    init_audit = _verify_s0_initialization(
        identity, roots, resolved, models, data, initial, seed, device, fingerprint)
    _save_resolved_record(roots, config, config_path, resolved, cfg, fingerprint)
    feature_artifacts = _fit_feature_prior(reference, identity, roots, config, fingerprint)
    _atomic_json(roots['run_result'] / 'feature_prior_diagnostics.json',
                 feature_artifacts['manifest'])
    diffusion, diffusion_manifest = _fit_diffusion_prior(
        identity, roots, config, feature_artifacts, reference, fingerprint,
        seed, device, stop_file, progress_path, run_id, fold)
    if diffusion is None:
        _progress(progress_path, run_id=run_id, fold=fold,
                  stage='diffusion_prior', completed=0, total=5000,
                  status='paused', message='paused at an update checkpoint')
        return 75

    teacher_manifests = {}
    teacher_A = {'status': 'reference', 'method': 'T0',
                 'teacher_checkpoint': models['teacher_checkpoint'],
                 'checkpoint_sha256': models['teacher_checkpoint_sha256'],
                 'teacher_checkpoint_sha256': models['teacher_checkpoint_sha256']}
    # Every adapted Teacher starts from a deep copy of the same frozen T0.
    for method in TEACHER_METHODS:
        model, manifest = _adapt_teacher(
            method, identity, roots, config, models, reference, data,
            feature_artifacts, diffusion, fingerprint, seed, device,
            stop_file, progress_path, run_id, fold)
        if model is None:
            _progress(progress_path, run_id=run_id, fold=fold,
                      stage=f'teacher_{method}', completed=0, total=10,
                      status='paused', message='paused at an epoch checkpoint')
            return 75
        # The frozen model is represented by its saved checkpoint and D_train
        # target cache; release the GPU copy before adapting the next arm.
        del model
        teacher_manifests[method] = manifest

    student_manifests = {'A': {
        'status': 'reused_verified_baseline', 'method': METHODS['A'],
        'student_checkpoint': models['baseline_checkpoint'],
        'checkpoint_sha256': models['baseline_checkpoint_sha256'],
        'model_sha256': models['baseline_model_sha256'], 'final_epoch': 100,
        'initial_model_sha256': initial['model_sha256'],
        'initialization_audit': init_audit,
        'baseline_manifest': str(roots['base_results'] / 'BASE_CE/manifest.json'),
    }}
    # Train all final Students fresh from I_s. B uses T0; C--G use their
    # separately adapted, now frozen Teacher logits on precisely D_train.
    for method in STUDENT_METHODS:
        teacher_manifest = teacher_A if method == 'B' else teacher_manifests[method]
        manifest = _fit_student(
            method, identity, roots, config, models, reference, data, initial,
            teacher_manifest, fingerprint, seed, device, stop_file,
            progress_path, run_id, fold)
        if manifest.get('status') == 'paused':
            _progress(progress_path, run_id=run_id, fold=fold,
                      stage=f'student_{method}', completed=manifest['completed_epoch'],
                      total=100, status='paused', message='paused at an epoch checkpoint')
            return 75
        student_manifests[method] = manifest

    completion = {
        'status': 'complete', 'experiment_id': config['experiment_id'],
        'run_id': run_id, 'seed': seed, 'fold': int(fold),
        'target_subject_id': int(identity['manifest']['target_subject_id']),
        'feedback_subject_id': int(identity['manifest']['feedback_subject_id']),
        'run_fingerprint': fingerprint,
        'split_manifest_sha256': identity['sha256'],
        'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'student_preprocess_max_abs_diff_vs_split_cache':
            data['student_preprocess_max_abs_diff'],
        'initialization_audit': init_audit,
        'feature_prior_manifest': feature_artifacts['manifest'],
        'diffusion_manifest': diffusion_manifest,
        'teacher_manifests': teacher_manifests,
        'student_manifests': student_manifests,
        'target_test_read': False,
    }
    _atomic_json(complete_path, completion)
    _progress(progress_path, run_id=run_id, fold=fold, stage='fold_complete',
              completed=1, total=1, status='completed', message='all train-only conditions fixed')
    print(f'[fold-complete] {run_id} seed={seed} target={fold + 1:02d}; '
          'no test labels/signals were accessed', flush=True)
    return 0


def _verify_completed_run(config_path: Path, run_id: str) -> tuple[dict, dict, dict]:
    """Validate all train-only artifacts for all folds before target access."""
    config = _read_yaml(config_path)
    _validate_v1_config(config)
    resolved, cfg = _load_resolved_reference(config)
    fold_records = {}
    for fold in range(9):
        roots = _run_roots(run_id, int(config['seed']), fold)
        identity = _load_fold_identity(
            config, fold, roots, cfg['resolved_config_sha256'], include_test_labels=False)
        fingerprint = _fold_fingerprint(config, config_path, resolved, cfg,
                                        identity, roots, run_id)
        complete_path = roots['run_result'] / 'fold_complete.json'
        if not complete_path.is_file():
            raise RuntimeError(f'target evaluation blocked: fold {fold + 1} is not complete')
        completion = _read_json(complete_path)
        if (completion.get('status') != 'complete'
                or completion.get('run_fingerprint') != fingerprint
                or completion.get('target_test_read') is not False
                or completion.get('seed') != int(config['seed'])):
            raise RuntimeError(f'target evaluation blocked: fold {fold + 1} completion identity mismatch')
        teachers = completion.get('teacher_manifests', {})
        students = completion.get('student_manifests', {})
        if set(teachers) != set(TEACHER_METHODS) or set(students) != set('ABCDEFG'):
            raise RuntimeError(f'target evaluation blocked: fold {fold + 1} lacks a condition')
        for method in TEACHER_METHODS:
            record = teachers[method]
            manifest_path = roots['run_result'] / f'teacher_{method}.json'
            checkpoint_path = roots['run_result'] / f'teacher_{method}.pt'
            target_path = Path(record.get('train_targets_cache', ''))
            if (record.get('status') != 'complete' or int(record.get('epochs', -1)) != 10
                    or not manifest_path.is_file() or not checkpoint_path.is_file()
                    or not target_path.is_file()
                    or _sha256(target_path) != record.get('train_targets_sha256')):
                raise RuntimeError(f'target evaluation blocked: Teacher {method} incomplete in fold {fold + 1}')
            saved_teacher_manifest = _read_json(manifest_path)
            if (saved_teacher_manifest.get('run_fingerprint') != record.get('run_fingerprint')
                    or saved_teacher_manifest.get('checkpoint_sha256') != _sha256(checkpoint_path)):
                raise RuntimeError(f'Teacher {method} checkpoint hash mismatch in fold {fold + 1}')
        for method in STUDENT_METHODS:
            record = students[method]
            if method == 'A':
                checkpoint = roots['base_results'] / 'BASE_CE/checkpoint_final.pt'
                if (record.get('status') != 'reused_verified_baseline'
                        or _sha256(checkpoint) != record.get('checkpoint_sha256')):
                    raise RuntimeError(f'target evaluation blocked: baseline A changed in fold {fold + 1}')
            else:
                manifest_path = roots['run_result'] / f'student_{method}.json'
                checkpoint = roots['run_weights'] / f'student_{method}.pt'
                if (record.get('status') != 'complete' or int(record.get('final_epoch', -1)) != 100
                        or not manifest_path.is_file() or not checkpoint.is_file()):
                    raise RuntimeError(f'target evaluation blocked: Student {method} incomplete in fold {fold + 1}')
                saved_student_manifest = _read_json(manifest_path)
                if (saved_student_manifest.get('run_fingerprint') != record.get('run_fingerprint')
                        or saved_student_manifest.get('checkpoint_sha256') != _sha256(checkpoint)):
                    raise RuntimeError(f'Student {method} checkpoint hash mismatch in fold {fold + 1}')
        diffusion_path = roots['run_weights'] / 'prior_diffusion.pt'
        diffusion_manifest_path = roots['run_weights'] / 'prior_diffusion.json'
        if not (diffusion_path.is_file() and diffusion_manifest_path.is_file()):
            raise RuntimeError(f'target evaluation blocked: diffusion prior missing in fold {fold + 1}')
        diffusion_manifest = _read_json(diffusion_manifest_path)
        if (diffusion_manifest.get('status') != 'complete'
                or int(diffusion_manifest.get('step', -1)) != 5000
                or diffusion_manifest.get('checkpoint_sha256') != _sha256(diffusion_path)):
            raise RuntimeError(f'diffusion prior checkpoint mismatch in fold {fold + 1}')
        fold_records[fold] = {
            'roots': roots, 'identity_train_only': identity,
            'completion': completion, 'fingerprint': fingerprint,
            'complete_sha256': _sha256(complete_path),
        }
    # All 9 folds and every condition are now fixed. Full hashes may now scan
    # the pooled model-input arrays, including held-out target rows.
    first_roots = fold_records[0]['roots']
    source_manifest = _read_json(BNCI14004_SOURCE_ROOT / 'manifest.json')
    _verify_manifest_files(BNCI14004_SOURCE_ROOT, source_manifest,
                           ('X.npy', 'labels.npy', 'trials.csv'),
                           '004 wide_fre_band source')
    for model_name in ('mirepnet', 'ifnet'):
        model_root = first_roots['input_root'] / model_name
        model_manifest = _read_json(model_root / 'manifest.json')
        _verify_manifest_files(model_root, model_manifest,
                               ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'),
                               f'004/{model_name} model input')
    return config, {'resolved': resolved, 'reference_cfg': cfg}, fold_records


def _load_eval_student(adapter, checkpoint_path: Path, device: torch.device):
    payload = torch.load(resolve_local_file(checkpoint_path), map_location='cpu',
                         weights_only=False)
    model = adapter.build(2)
    model.load_state_dict(payload['model'], strict=True)
    if _state_hash(model.state_dict()) != payload.get('model_sha256'):
        raise RuntimeError(f'Student model state hash mismatch: {checkpoint_path}')
    model.eval()
    return model, payload


def _load_eval_teacher(adapter, base_teacher, checkpoint_path: Path,
                       device: torch.device):
    payload = torch.load(resolve_local_file(checkpoint_path), map_location='cpu',
                         weights_only=False)
    model = copy.deepcopy(base_teacher)
    model.load_state_dict(payload['model'], strict=True)
    if _state_hash(model.state_dict()) != payload.get('model_sha256'):
        raise RuntimeError(f'Teacher model state hash mismatch: {checkpoint_path}')
    model.eval()
    return model, payload


def _evaluate_fold(config: dict, resolved: dict, cfg: dict, run_id: str,
                   fold: int, roots: dict, device: torch.device,
                   stop_file: str | None, progress_path: str | None) -> list[dict]:
    identity = _load_fold_identity(
        config, fold, roots, cfg['resolved_config_sha256'], include_test_labels=True)
    models = _load_reference_checkpoints(config, resolved, cfg, identity, roots, device)
    student_root, teacher_root = roots['input_root'] / 'ifnet', roots['input_root'] / 'mirepnet'
    student_rows, _ = _input_rows(student_root, {'test': identity['uids']['test']},
                                  {'test': identity['labels']['test']}, 'ifnet')
    teacher_rows, _ = _input_rows(teacher_root, {'test': identity['uids']['test']},
                                  {'test': identity['labels']['test']}, 'mirepnet')
    student_raw = np.load(resolve_local_file(student_root / 'X.npy'), mmap_mode='r',
                          allow_pickle=False)
    teacher_all = np.load(resolve_local_file(teacher_root / 'X.npy'), mmap_mode='r',
                          allow_pickle=False)
    student_x = _preprocess_student(
        models['student_adapter'], np.asarray(student_raw[student_rows['test']], dtype=np.float32))
    teacher_x = np.asarray(teacher_all[teacher_rows['test']], dtype=np.float32)
    y = identity['labels']['test']
    uids = identity['uids']['test']
    prediction_arrays = {'uids': uids, 'y': y}
    model_hashes = {}
    rows = []
    eval_result = roots['run_result'] / 'target_evaluation'
    require_external_output(eval_result).mkdir(parents=True, exist_ok=True)

    for method in STUDENT_METHODS:
        if method == 'A':
            student_model = models['student_model']
            student_hash = models['baseline_model_sha256']
        else:
            path = roots['run_weights'] / f'student_{method}.pt'
            student_model, payload = _load_eval_student(models['student_adapter'], path, device)
            student_hash = payload['model_sha256']
        logits = _infer(models['student_adapter'], student_model, student_x,
                        device, batch_size=64, return_features=False)[1]
        prediction_arrays[f'student_{method}_logits'] = logits
        model_hashes[f'student_{method}'] = student_hash
        metrics = _metric_values(y, logits)
        rows.append({'fold': fold, 'target_subject_id': fold + 1,
                     'seed': int(config['seed']), 'condition': METHODS[method],
                     'model_role': 'student', 'model_code': method, **metrics})
        del logits
        if method != 'A':
            del student_model

    teacher_names = [('T0', 'TEACHER_T0')] + [
        (method, f'TEACHER_{method}_{METHODS[method]}') for method in TEACHER_METHODS]
    for code, name in teacher_names:
        if code == 'T0':
            teacher_model = models['teacher_model']
            teacher_hash = models['teacher_checkpoint_sha256']
        else:
            path = roots['run_result'] / f'teacher_{code}.pt'
            teacher_model, payload = _load_eval_teacher(
                models['teacher_adapter'], models['teacher_model'], path, device)
            teacher_hash = payload['model_sha256']
        logits = _infer(models['teacher_adapter'], teacher_model, teacher_x,
                        device, batch_size=32, return_features=False)[1]
        prediction_arrays[f'teacher_{code}_logits'] = logits
        model_hashes[f'teacher_{code}'] = teacher_hash
        metrics = _metric_values(y, logits)
        rows.append({'fold': fold, 'target_subject_id': fold + 1,
                     'seed': int(config['seed']), 'condition': name,
                     'model_role': 'teacher', 'model_code': code, **metrics})
        del logits
        if code != 'T0':
            del teacher_model

    prediction_path = eval_result / 'test_predictions.npz'
    _atomic_npz(prediction_path, **prediction_arrays)
    metrics_path = eval_result / 'test_metrics.csv'
    _atomic_csv(metrics_path, rows)
    fold_manifest_path = eval_result / 'evaluation_manifest.json'
    _atomic_json(fold_manifest_path, {
        'status': 'complete', 'run_id': run_id, 'seed': int(config['seed']),
        'fold': fold, 'target_subject_id': fold + 1,
        'run_fingerprint': _read_json(roots['run_result'] / 'fold_complete.json')['run_fingerprint'],
        'test_uid_sha256': identity['uid_sha256']['test'],
        'prediction_file': str(prediction_path),
        'prediction_file_sha256': _sha256(prediction_path),
        'metrics_file': str(metrics_path), 'metrics_file_sha256': _sha256(metrics_path),
        'model_sha256': model_hashes,
        'target_evaluation_count': len(y),
    })
    print(f'[target-eval] target={fold + 1:02d} rows={len(y)} predictions=' 
          f'{prediction_path}', flush=True)
    del student_raw, teacher_all, models
    if _stopped(stop_file):
        _progress(progress_path, run_id=run_id, fold=fold,
                  stage='target_evaluation', completed=1, total=9,
                  status='paused', message='pause requested at fold boundary')
        raise InterruptedError('evaluation paused at a fold boundary')
    return rows


PAIRED_COMPARISONS = (
    ('G', 'F', 'primary_G_minus_F'),
    ('G', 'B', 'G_minus_B'),
    ('G', 'E', 'G_minus_E'),
    ('G', 'C', 'G_minus_C'),
    ('D', 'B', 'D_minus_B'),
    ('E', 'D', 'E_minus_D'),
    ('F', 'C', 'F_minus_C'),
)


def _report_csv(path: Path, rows: list[dict]) -> None:
    target = require_report_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report_json(path: Path, value: dict) -> None:
    target = require_report_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=True) + '\n', encoding='utf-8')


def _write_final_reports(run_id: str, config: dict, subject_rows: list[dict],
                         bootstrap_draws: int = BOOTSTRAP_DRAWS) -> dict:
    report_dir = REPORTS_ROOT / 'distill' / run_id
    require_report_output(report_dir / 'summary.csv').parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    methods = [(METHODS[code], code, 'student') for code in 'ABCDEFG']
    methods += [('TEACHER_T0', 'T0', 'teacher')]
    methods += [(f'TEACHER_{code}_{METHODS[code]}', code, 'teacher')
                for code in TEACHER_METHODS]
    for condition, code, role in methods:
        group = [row for row in subject_rows
                 if row['condition'] == condition and row['model_role'] == role]
        if len(group) != 9:
            raise RuntimeError(f'{condition}: expected nine completed subject metrics, got {len(group)}')
        record = {'condition': condition, 'model_role': role, 'seed': int(config['seed']),
                  'n_seeds': 1, 'n_subjects': 9}
        for key, output_name in (('balanced_accuracy', 'balanced_accuracy'),
                                 ('accuracy', 'accuracy'), ('kappa', 'kappa'),
                                 ('cross_entropy', 'cross_entropy')):
            values = np.asarray([row[key] for row in group], dtype=np.float64)
            scale = 100.0 if key in ('balanced_accuracy', 'accuracy') else 1.0
            mean = float(values.mean() * scale)
            std = float(values.std(ddof=1) * scale)
            record[f'{output_name}_mean' + ('_percent' if scale == 100 else '')] = mean
            record[f'{output_name}_std' + ('_percent' if scale == 100 else '')] = std
            if scale == 100:
                record[f'{output_name}_mean_pm_std_percent'] = f'{mean:.2f} ± {std:.2f}'
        summary_rows.append(record)

    student_ba = {
        code: np.asarray([row['balanced_accuracy'] for row in subject_rows
                          if row['model_role'] == 'student' and row['model_code'] == code],
                         dtype=np.float64) * 100.0
        for code in 'ABCDEFG'}
    comparison_rows = []
    rng = np.random.default_rng(int(config['report']['bootstrap_seed']))
    for better, baseline, label in PAIRED_COMPARISONS:
        differences = student_ba[better] - student_ba[baseline]
        draws = rng.integers(0, len(differences),
                             size=(int(bootstrap_draws), len(differences)))
        means = differences[draws].mean(axis=1)
        comparison_rows.append({
            'comparison': label, 'method': METHODS[better], 'baseline': METHODS[baseline],
            'seed': int(config['seed']), 'n_seeds': 1, 'n_subjects': len(differences),
            'mean_balanced_accuracy_difference_pp': float(differences.mean()),
            'std_difference_pp': float(differences.std(ddof=1)),
            'paired_bootstrap_95ci_low_pp': float(np.percentile(means, 2.5)),
            'paired_bootstrap_95ci_high_pp': float(np.percentile(means, 97.5)),
            'bootstrap_draws': int(bootstrap_draws),
            'bootstrap_seed': int(config['report']['bootstrap_seed']),
            'wins': int(np.sum(differences > 1e-6)),
            'ties': int(np.sum(np.abs(differences) <= 1e-6)),
            'losses': int(np.sum(differences < -1e-6)),
            'per_subject_differences_pp': json.dumps(differences.tolist()),
        })
    _report_csv(report_dir / 'summary.csv', summary_rows)
    _report_csv(report_dir / 'paired_comparisons.csv', comparison_rows)
    _report_csv(report_dir / 'subject_metrics.csv', subject_rows)
    report = {
        'experiment_id': config['experiment_id'], 'run_id': run_id,
        'dataset': 'BNCI2014004', 'session': 'session_3', 'seed': int(config['seed']),
        'n_seeds': 1, 'subjects': list(range(1, 10)),
        'primary_metric': 'Balanced Accuracy (%)',
        'aggregation': 'mean and sample standard deviation over nine target-subject folds; seed 666 only',
        'bootstrap_unit': 'target subject', 'bootstrap_draws': int(bootstrap_draws),
        'summary_csv': str(report_dir / 'summary.csv'),
        'subject_metrics_csv': str(report_dir / 'subject_metrics.csv'),
        'paired_comparisons_csv': str(report_dir / 'paired_comparisons.csv'),
        'conclusions': [],
    }
    primary = comparison_rows[0]
    report['conclusions'] = [
        (f"Primary G−F: {primary['mean_balanced_accuracy_difference_pp']:+.2f} pp "
         f"(paired bootstrap 95% CI {primary['paired_bootstrap_95ci_low_pp']:+.2f} to "
         f"{primary['paired_bootstrap_95ci_high_pp']:+.2f}; "
         f"W/T/L={primary['wins']}/{primary['ties']}/{primary['losses']}, 9 subjects, seed 666)."),
        'Exploratory single-seed pilot; LOSO training folds overlap, so the subject bootstrap does not establish cross-dataset or cross-seed stability.',
    ]
    _report_json(report_dir / 'summary.json', report)
    md = [
        '# STUDENT_AWARE_DIFFUSION_PRIOR_V1', '',
        'Dataset: BNCI2014004 session_3, binary left/right hand. Seed: 666 (one seed).',
        'Aggregation: equal-weight mean and sample SD across the nine target-subject folds.',
        'Primary metric: per-subject Balanced Accuracy, shown as mean ± SD in percent.',
        'All model outputs and paired comparisons are from the final IFNet or named teacher; no ensembles.',
        '', '## Student and teacher summary', '',
        '| Condition | Role | Balanced Accuracy (%) mean ± SD | Accuracy (%) mean ± SD | Kappa mean | CE mean |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        md.append(f"| {row['condition']} | {row['model_role']} | "
                  f"{row.get('balanced_accuracy_mean_pm_std_percent','')} | "
                  f"{row.get('accuracy_mean_pm_std_percent','')} | "
                  f"{row['kappa_mean']:.4f} | {row['cross_entropy_mean']:.4f} |")
    md.extend(['', '## Paired comparisons', '',
               '| Comparison | Mean Δ BA (pp) | 95% paired subject bootstrap CI | W/T/L |',
               '|---|---:|---:|---:|'])
    for row in comparison_rows:
        md.append(f"| {row['comparison']} | {row['mean_balanced_accuracy_difference_pp']:+.2f} | "
                  f"[{row['paired_bootstrap_95ci_low_pp']:+.2f}, "
                  f"{row['paired_bootstrap_95ci_high_pp']:+.2f}] | "
                  f"{row['wins']}/{row['ties']}/{row['losses']} |")
    md.extend(['', '## Conclusions', ''] + [f'{index}. {line}' for index, line
                                           in enumerate(report['conclusions'], 1)])
    md.extend(['', 'The confidence interval is descriptive for this nine-subject, single-seed pilot.'])
    md_path = require_report_output(report_dir / 'conclusions.md')
    md_path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    report['conclusions_md'] = str(md_path)
    _report_json(report_dir / 'summary.json', report)
    return report


def evaluate_run(config_path: Path, run_id: str, gpu: int = 0,
                 progress_path: str | None = None,
                 stop_file: str | None = None) -> int:
    config, cfg_pack, folds = _verify_completed_run(Path(config_path).resolve(), run_id)
    device = _device(gpu)
    resolved, cfg = cfg_pack['resolved'], cfg_pack['reference_cfg']
    all_rows = []
    for fold in range(9):
        rows = _evaluate_fold(config, resolved, cfg, run_id, fold,
                              folds[fold]['roots'], device, stop_file, progress_path)
        all_rows.extend(rows)
        _progress(progress_path, run_id=run_id, fold=fold,
                  stage='target_evaluation', completed=fold + 1, total=9,
                  message=f'target {fold + 1}/9 evaluated')
    report = _write_final_reports(
        run_id, config, all_rows,
        bootstrap_draws=int(config['report']['bootstrap_draws']))
    _progress(progress_path, run_id=run_id, fold=8,
              stage='evaluation_complete', completed=9, total=9,
              status='completed', message='all target folds evaluated after training completion')
    print(f"[evaluation-complete] summary={report['summary_csv']}", flush=True)
    return 0


def _student_update(model, optimizer, adapter, x: np.ndarray, y: np.ndarray,
                    teacher_logits: torch.Tensor, indices: np.ndarray,
                    device: torch.device, kd_weight: float = 0.5,
                    temperature: float = 2.0) -> tuple[float, float]:
    xb = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
    yb = torch.as_tensor(y[indices], dtype=torch.long, device=device)
    batch_ids = torch.as_tensor(indices, dtype=torch.long, device=device)
    tb = teacher_logits[batch_ids]
    optimizer.zero_grad(set_to_none=True)
    _, logits = adapter.forward(model, xb)
    ce = F.cross_entropy(logits, yb)
    kd = _kl_teacher_to_student(tb, logits, temperature)
    loss = ce + kd_weight * kd
    loss.backward()
    optimizer.step()
    return float(ce.detach().cpu()), float(kd.detach().cpu())


def _smoke_student_recovery(models: dict, initial: dict, data: dict,
                            identity: dict, reference: dict, seed: int,
                            device: torch.device) -> dict:
    teacher_logits = torch.as_tensor(reference['train_teacher_logits'],
                                     dtype=torch.float32, device=device)
    first, second = models['student_schedule'][0][:2]
    # Uninterrupted two-update trajectory.
    full_model, full_opt, _, _ = _new_paired_student(models, initial, seed, device)
    _student_update(full_model, full_opt, models['student_adapter'],
                    data['student_x']['train'], identity['labels']['train'],
                    teacher_logits, first, device)
    _student_update(full_model, full_opt, models['student_adapter'],
                    data['student_x']['train'], identity['labels']['train'],
                    teacher_logits, second, device)
    expected = _state_hash(full_model.state_dict())
    del full_model, full_opt

    resumed_model, resumed_opt, resumed_scheduler, _ = _new_paired_student(
        models, initial, seed, device)
    _student_update(resumed_model, resumed_opt, models['student_adapter'],
                    data['student_x']['train'], identity['labels']['train'],
                    teacher_logits, first, device)
    checkpoint = {
        'model': _cpu_tree(resumed_model.state_dict()),
        'optimizer': _cpu_tree(resumed_opt.state_dict()),
        'scheduler': resumed_scheduler.state_dict(), 'rng': _capture_rng(device),
    }
    resumed_model2, resumed_opt2, resumed_scheduler2, _ = _new_paired_student(
        models, initial, seed, device)
    resumed_model2.load_state_dict(checkpoint['model'], strict=True)
    resumed_opt2.load_state_dict(checkpoint['optimizer'])
    resumed_scheduler2.load_state_dict(checkpoint['scheduler'])
    _restore_rng(checkpoint['rng'], device)
    _student_update(resumed_model2, resumed_opt2, models['student_adapter'],
                    data['student_x']['train'], identity['labels']['train'],
                    teacher_logits, second, device)
    actual = _state_hash(resumed_model2.state_dict())
    if expected != actual:
        raise RuntimeError('Student interrupted/resumed KD trajectory differs from uninterrupted run')
    del resumed_model, resumed_opt, resumed_model2, resumed_opt2
    return {'two_kd_updates': True, 'resume_state_hash_matches': True,
            'final_model_sha256': expected,
            'train_uids_used': int(len(first) + len(second))}


def _teacher_smoke_steps(method: str, models: dict, identity: dict,
                         data: dict, reference: dict, features: dict,
                         prior: ConditionalNoisePredictor | None,
                         config: dict, seed: int, fold: int,
                         device: torch.device, *, lambda_override: float | None = None):
    base = models['teacher_model']
    initial_hash = _state_hash(base.state_dict())
    model = copy.deepcopy(base).to(device)
    model.eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config['teacher_adaptation']['lr']),
        betas=tuple(config['teacher_adaptation']['betas']),
        weight_decay=float(config['teacher_adaptation']['weight_decay']))
    schedule = _epoch_schedule(len(identity['uids']['train']), 8, 10,
                               seed + 21001 + fold)[0][:2]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 99001 + int(fold) * 1009)
    _, alpha, sigma = linear_ddpm_schedule(
        int(config['diffusion_prior']['steps']),
        float(config['diffusion_prior']['beta_start']),
        float(config['diffusion_prior']['beta_end']), device)
    projection = features['projection']
    t_mean = torch.as_tensor(projection['teacher_mean'], dtype=torch.float32, device=device)
    t_std = torch.as_tensor(projection['teacher_std'], dtype=torch.float32, device=device)
    p_weight = torch.as_tensor(projection['weight'], dtype=torch.float32, device=device)
    p_bias = torch.as_tensor(projection['bias'], dtype=torch.float32, device=device)
    for value in (t_mean, t_std, p_weight, p_bias):
        value.requires_grad_(False)
    if prior is not None:
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
    train_y = identity['labels']['train']
    t0_logits = torch.as_tensor(reference['train_teacher_logits'], dtype=torch.float32,
                                device=device)
    s0_logits = torch.as_tensor(reference['train_student_logits'], dtype=torch.float32,
                                device=device)
    prior_grad_finite_nonzero = False
    for batch_index, ids in enumerate(schedule):
        batch_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        xb = torch.as_tensor(data['teacher_x']['train'][ids], dtype=torch.float32, device=device)
        yb = torch.as_tensor(train_y[ids], dtype=torch.long, device=device)
        feature, logits = models['teacher_adapter'].forward(model, xb)
        ce = F.cross_entropy(logits, yb)
        keep = _kl_teacher_to_student(t0_logits[batch_ids], logits, 2.0)
        aid = _aid_student_to_teacher(logits, s0_logits[batch_ids], 2.0)
        weight = (float(config['feature_prior_adaptation']['lambda'])
                  if lambda_override is None else float(lambda_override))
        prior_loss = logits.sum() * 0.0
        if method in ('F', 'G') and weight > 0:
            projected = F.linear((feature - t_mean) / t_std, p_weight, p_bias)
            low = int(config['feature_prior_adaptation']['timestep_min'])
            high = int(config['feature_prior_adaptation']['timestep_max'])
            t = torch.randint(low, high + 1, (len(ids),), generator=generator, device=device)
            eps = torch.randn(projected.shape, generator=generator, device=device)
            a, s = alpha[t - 1, None], sigma[t - 1, None]
            noisy = a * projected.detach() + s * eps
            with torch.no_grad():
                if method == 'F':
                    means = torch.as_tensor(features['gaussian']['means'],
                                             dtype=torch.float32, device=device)
                    variances = torch.as_tensor(features['gaussian']['variances'],
                                                dtype=torch.float32, device=device)
                    eps_hat = gaussian_epsilon(noisy, t, yb, means, variances, alpha, sigma)
                else:
                    if prior is None:
                        raise RuntimeError('smoke G update requires the trained smoke Diffusion prior')
                    eps_hat = prior(noisy, t, yb)
                direction = sds_direction(eps_hat, eps, t, alpha, sigma)
            prior_loss = sds_surrogate(projected, direction)
            if batch_index == 0 and method == 'G':
                gradients = torch.autograd.grad(prior_loss, tuple(model.parameters()),
                                                retain_graph=True, allow_unused=True)
                norms = [gradient.detach().norm() for gradient in gradients
                         if gradient is not None]
                prior_grad_finite_nonzero = bool(
                    norms and torch.isfinite(torch.stack(norms)).all()
                    and torch.stack(norms).norm() > 0)
        keep_weight = float(config['teacher_adaptation']['keep_weight'])
        aid_weight = float(config['teacher_adaptation']['aid_weight'])
        loss = ce
        if method in ('C', 'E', 'F', 'G'):
            loss = loss + keep_weight * keep
        if method in ('D', 'E'):
            loss = loss + aid_weight * aid
        if method in ('F', 'G'):
            loss = loss + weight * prior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if _state_hash(base.state_dict()) != initial_hash:
        raise RuntimeError('Teacher smoke mutated the frozen T0 reference')
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise RuntimeError('frozen T0 received gradients during Teacher smoke')
    if prior is not None and any(parameter.grad is not None for parameter in prior.parameters()):
        raise RuntimeError('frozen Diffusion prior received gradients during Teacher smoke')
    return model, {'two_teacher_updates': True,
                   'prior_gradient_finite_nonzero': prior_grad_finite_nonzero}


def run_smoke(config_path: Path, fold: int = 0, gpu: int = 0) -> dict:
    """Run the plan's train-only, temporary single-fold wiring/recovery smoke."""
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    _validate_v1_config(config)
    if not 0 <= int(fold) < 9:
        raise ValueError('smoke fold must be in 0..8')
    seed = int(config['seed'])
    device = _device(gpu)
    formal_roots = _run_roots(config['storage']['external_run_id'], seed, int(fold))
    resolved, cfg = _load_resolved_reference(config)
    identity = _load_fold_identity(config, fold, formal_roots,
                                   cfg['resolved_config_sha256'], include_test_labels=False)
    with tempfile.TemporaryDirectory(prefix='student_aware_diffusion_smoke_', dir='/tmp') as name:
        temp_root = Path(name)
        roots = dict(formal_roots)
        roots.update({
            'run_result': temp_root / 'results' / 'fold',
            'run_cache': temp_root / 'cache' / 'fold',
            'run_weights': temp_root / 'weights' / 'fold',
            'run_seed_weights': temp_root / 'weights',
        })
        for key in ('run_result', 'run_cache', 'run_weights', 'run_seed_weights'):
            roots[key].mkdir(parents=True, exist_ok=True)
        data = _load_fold_data(identity, roots, resolved['student_config'], device)
        models = _load_reference_checkpoints(config, resolved, cfg, identity, roots, device)
        t0_hash = _state_hash(models['teacher_model'].state_dict())
        s0_hash = _state_hash(models['student_model'].state_dict())
        fingerprint = _fold_fingerprint(config, config_path, resolved, cfg,
                                        identity, roots, f'{config["storage"]["external_run_id"]}_smoke')
        reference = _reference_cache(identity, roots, models, data, fingerprint, device)
        initial = _ensure_student_initial_state(roots, resolved, cfg, seed)
        init_audit = _verify_s0_initialization(
            identity, roots, resolved, models, data, initial, seed, device, fingerprint)
        feature_artifacts = _fit_feature_prior(reference, identity, roots, config, fingerprint)
        math_audit = _audit_loss_math(device)

        # Train four tiny diffusion updates; interrupt at step 2 and resume at
        # the prescribed checkpoint boundary. The temporary weights are deleted
        # with this /tmp directory after all assertions pass.
        smoke_config = copy.deepcopy(config)
        smoke_config['diffusion_prior']['training_updates'] = 4
        smoke_config['diffusion_prior']['diagnostic_interval'] = 2
        smoke_fingerprint = _json_hash({'fold': fingerprint, 'smoke_updates': 4})
        pause_file = temp_root / 'stop-at-two'
        pause_file.touch()
        prior, partial = _fit_diffusion_prior(
            identity, roots, smoke_config, feature_artifacts, reference,
            smoke_fingerprint, seed, device, str(pause_file), None,
            'smoke', fold)
        if prior is not None or partial.get('status') != 'paused' or partial.get('completed_step') != 2:
            raise RuntimeError('Diffusion smoke did not pause at update 2')
        pause_file.unlink()
        prior, prior_manifest = _fit_diffusion_prior(
            identity, roots, smoke_config, feature_artifacts, reference,
            smoke_fingerprint, seed, device, None, None, 'smoke', fold)
        if prior is None or prior_manifest.get('step') != 4:
            raise RuntimeError('Diffusion smoke did not resume through update 4')

        c_model, c_diag = _teacher_smoke_steps(
            'C', models, identity, data, reference, feature_artifacts,
            None, config, seed, fold, device)
        f_zero_model, f_zero_diag = _teacher_smoke_steps(
            'F', models, identity, data, reference, feature_artifacts,
            prior, config, seed, fold, device, lambda_override=0.0)
        if _state_hash(c_model.state_dict()) != _state_hash(f_zero_model.state_dict()):
            raise RuntimeError('lambda=0 F Teacher does not exactly reduce to C on the smoke trajectory')
        g_model, g_diag = _teacher_smoke_steps(
            'G', models, identity, data, reference, feature_artifacts,
            prior, config, seed, fold, device)
        if not g_diag['prior_gradient_finite_nonzero']:
            raise RuntimeError('frozen Diffusion score did not send a finite non-zero gradient to Teacher')
        student_resume = _smoke_student_recovery(
            models, initial, data, identity, reference, seed, device)
        if _state_hash(models['teacher_model'].state_dict()) != t0_hash:
            raise RuntimeError('smoke changed the frozen T0 checkpoint')
        if _state_hash(models['student_model'].state_dict()) != s0_hash:
            raise RuntimeError('smoke changed the frozen S0 checkpoint')
        for parameter in models['student_model'].parameters():
            if parameter.grad is not None:
                raise RuntimeError('frozen S0 received gradients in the smoke')
        output = {
            'status': 'passed', 'fold': fold, 'target_subject_id': fold + 1,
            'target_labels_or_signals_used': False,
            'train_count': len(identity['uids']['train']),
            'feedback_count': len(identity['uids']['feedback']),
            'teacher_feature_shape': list(reference['train_teacher_features'].shape),
            'student_feature_shape': list(reference['train_student_features'].shape),
            'student_preprocess_max_abs_diff_vs_split_cache':
                data['student_preprocess_max_abs_diff'],
            'pca_dimension': feature_artifacts['pca']['dimension'],
            'pca_effective_rank': feature_artifacts['pca']['effective_rank'],
            'projection_train_mse': feature_artifacts['projection']['train_mse'],
            'projection_train_r2': feature_artifacts['projection']['train_r2'],
            'projection_feedback_mse': feature_artifacts['diagnostics']['feedback_projection']['mse'],
            'math_audit': math_audit,
            'diffusion_resume_updates': [2, 4],
            'teacher_smoke': {'C': c_diag, 'F_lambda0': f_zero_diag,
                              'G': g_diag, 'lambda0_C_equals_F': True},
            'student_smoke': student_resume,
            'S0_initialization_replay': init_audit,
            'frozen_T0_and_S0_unchanged': True,
            'temporary_artifacts_removed_after_exit': True,
        }
        del prior, c_model, f_zero_model, g_model, models, data
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return output


def enqueue_run(config_path: Path, queue_root: Path | None = None,
                folds: list[int] | None = None, start_watcher: bool = True) -> dict:
    from experiments.distill.queue import enqueue_snapshot
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    _validate_v1_config(config)
    run_id = str(config['storage']['external_run_id'])
    selected_folds = [int(value) for value in (folds if folds is not None else range(9))]
    if (not selected_folds or len(set(selected_folds)) != len(selected_folds)
            or any(value < 0 or value > 8 for value in selected_folds)):
        raise ValueError('folds must be a unique non-empty subset of zero-based 0..8')
    config_sha = _sha256(config_path)
    queue_kwargs = {'root': queue_root} if queue_root is not None else {}
    from experiments.distill.queue import DEFAULT_ROOT
    actual_queue_root = Path(queue_root or DEFAULT_ROOT).resolve()
    job_ids = []
    for fold in selected_folds:
        snapshot = {
            'kind': 'student_aware_diffusion_prior_v1', 'stage': 'train_fold',
            'config': str(config_path), 'config_sha256': config_sha,
            'run_id': run_id, 'fold': fold,
            'args': {'dataset': 'BNCI2014004', 'teacher': 'mirepnet',
                     'student': 'ifnet', 'protocol': 'loso_7_1_1',
                     'keys': [fold], 'seeds': [int(config['seed'])],
                     'methods': ['student_aware_diffusion_prior_v1'],
                     'conditions': list(METHODS.values()), 'epochs': 100,
                     'source_variant': 'wide_fre_band', 'selected_session': 'session_3'},
        }
        job_ids.append(enqueue_snapshot(
            snapshot, label=f"student-aware diffusion prior seed={config['seed']} "
                            f"target={fold + 1:02d}", **queue_kwargs))
    dispatch_dir = require_report_output(REPORTS_ROOT / 'distill' / run_id / 'dispatch')
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    dispatch = {'status': 'submitted', 'run_id': run_id, 'config': str(config_path),
                'config_sha256': config_sha, 'folds': selected_folds,
                'training_job_ids': job_ids, 'queue_root': str(actual_queue_root),
                'target_evaluation': 'submit only after all training jobs complete'}
    if start_watcher:
        watcher_log = dispatch_dir / 'evaluation_watcher.log'
        watcher_cmd = [sys.executable, '-B', '-m',
                       'experiments.distill.run_student_aware_diffusion_loso',
                       '--watch-evaluate', '--config', str(config_path), '--run-id', run_id,
                       '--expected-config-sha256', config_sha,
                       '--job-ids', *job_ids, '--queue-root', str(actual_queue_root),
                       '--watch-state', str(dispatch_dir / 'watcher_state.json')]
        environment = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS',
                     'TORCH_NUM_THREADS'):
            environment[name] = '1'
        with watcher_log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(
                watcher_cmd, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=environment, close_fds=True)
        dispatch.update({'watcher_pid': process.pid, 'watcher_log': str(watcher_log),
                         'watcher_state': str(dispatch_dir / 'watcher_state.json')})
    _report_json(dispatch_dir / 'dispatch.json', dispatch)
    print(json.dumps(dispatch, ensure_ascii=False, indent=2), flush=True)
    return dispatch


def _watch_train_jobs(config_path: Path, config_sha: str, run_id: str,
                      job_ids: list[str], queue_root: Path,
                      state_path: Path | None = None, poll_seconds: float = 15.0) -> int:
    from experiments.distill.queue import (QueueStore, enqueue_snapshot,
                                           validated_root)
    config_path = Path(config_path).resolve()
    if _sha256(config_path) != config_sha:
        raise RuntimeError('watcher config changed after training jobs were submitted')
    store = QueueStore(validated_root(queue_root))
    while True:
        states = {job_id: store.load(job_id) for job_id in job_ids}
        failed = {job_id: state.get('status') for job_id, state in states.items()
                  if state.get('status') in {'failed', 'cancelled'}}
        if failed:
            result = {'status': 'training_failed', 'run_id': run_id,
                      'failed_training_jobs': failed,
                      'message': 'target evaluator was not submitted'}
            if state_path:
                _report_json(state_path, result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 1
        if all(state.get('status') == 'completed' for state in states.values()):
            snapshot = {
                'kind': 'student_aware_diffusion_prior_v1', 'stage': 'evaluate',
                'config': str(config_path), 'config_sha256': config_sha,
                'run_id': run_id,
                'args': {'dataset': 'BNCI2014004', 'teacher': 'mirepnet',
                         'student': 'ifnet', 'protocol': 'loso_7_1_1',
                         'seeds': [666], 'methods': ['student_aware_diffusion_prior_v1'],
                         'conditions': list(METHODS.values()),
                         'stage': 'target_evaluation', 'source_variant': 'wide_fre_band'},
            }
            evaluation_job_id = enqueue_snapshot(
                snapshot, root=queue_root,
                label=f'student-aware diffusion prior target evaluation {run_id}')
            result = {'status': 'evaluation_submitted', 'run_id': run_id,
                      'training_job_ids': job_ids,
                      'evaluation_job_id': evaluation_job_id,
                      'message': 'all nine training fold jobs completed before target evaluation submission'}
            if state_path:
                _report_json(state_path, result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        time.sleep(float(poll_seconds))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fold', type=int, help='zero-based LOSO target subject index 0..8')
    mode.add_argument('--evaluate', action='store_true', help='evaluate only after all 9 fold manifests pass')
    mode.add_argument('--smoke', action='store_true', help='run the temporary single-fold wiring/resume smoke')
    mode.add_argument('--enqueue', action='store_true', help='submit all nine folds and start the evaluation watcher')
    mode.add_argument('--watch-evaluate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--expected-config-sha256', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--progress', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--stop-file', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--queue-root', type=Path, default=None)
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='folds to submit; --enqueue defaults to all 0..8')
    parser.add_argument('--job-ids', nargs='+', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--watch-state', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no-watcher', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.expected_config_sha256 and _sha256(args.config) != args.expected_config_sha256:
        parser.error('config file hash differs from the queued immutable snapshot')
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = _read_yaml(args.config)
        _validate_v1_config(config)
        run_id = str(args.run_id or config['storage']['external_run_id'])
        if args.fold is not None:
            return train_fold(args.config, run_id, args.fold, args.gpu,
                              args.progress, args.stop_file)
        if args.evaluate:
            return evaluate_run(args.config, run_id, args.gpu,
                                args.progress, args.stop_file)
        if args.smoke:
            smoke_fold = 0 if args.folds is None else args.folds[0]
            run_smoke(args.config, smoke_fold, args.gpu)
            _progress(args.progress, run_id='smoke', fold=smoke_fold,
                      stage='smoke_complete', completed=1, total=1,
                      status='completed', message='train-only recovery smoke passed')
            return 0
        if args.enqueue:
            enqueue_run(args.config, args.queue_root, args.folds,
                        start_watcher=not args.no_watcher)
            return 0
        if args.watch_evaluate:
            if not args.job_ids:
                raise ValueError('--watch-evaluate requires the submitted training --job-ids')
            if not args.expected_config_sha256:
                raise ValueError('--watch-evaluate requires the immutable config hash')
            from experiments.distill.queue import DEFAULT_ROOT
            queue_root = Path(args.queue_root or DEFAULT_ROOT)
            return _watch_train_jobs(args.config, args.expected_config_sha256,
                                     run_id, args.job_ids, queue_root,
                                     args.watch_state)
        raise AssertionError('unreachable command mode')
    except InterruptedError as exc:
        print(f'[paused] {exc}', flush=True)
        return 75
    except Exception as exc:
        traceback.print_exc()
        print(f'[student-aware-v1] error: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())




