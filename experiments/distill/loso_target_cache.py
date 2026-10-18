#!/usr/bin/env python
"""Create one verified LOSO teacher-target cache from a completed baseline.

The caller remains in the student environment.  Teacher model imports and
forward passes run in a subprocess using the teacher's configured conda env.
Only the source-training rows are passed through the completed teacher.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from data import loso_source
from experiments.storage import (
    LOSO_CACHE_ROOT,
    require_external_output,
    require_loso_cache_output,
)

SUPPORTED_DATASETS = (
    'BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI',
)
SUPPORTED_TEACHERS = ('cbramod', 'mirepnet')
_TEACHER_CACHE_FILES = ('targets.npz', 'manifest.json')


def _target_paths(dataset: str, teacher: str, seed: int, fold: int,
                  cache_root: str | Path | None) -> tuple[Path, Path, Path, Path]:
    """Return target directory, NPZ, manifest and per-fold lock file."""
    if dataset == 'AlexMI':
        # AlexMI retains its historical cache layout and strict reader.
        from data.loso_alexmi import ALEXMI_TEACHER_TARGET_ROOT

        cell = (ALEXMI_TEACHER_TARGET_ROOT / teacher / f'subject_{fold + 1:02d}'
                / f'seed_{seed}')
        cell = require_external_output(cell)
        return cell, cell / 'cache.npz', cell / 'cache_manifest.json', cell / 'cache.lock'

    root = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT
    cell = root / 'teacher_targets' / dataset / teacher / f'seed_{seed}' / f'subject_{fold + 1:02d}'
    cell = require_loso_cache_output(cell)
    return cell, cell / 'targets.npz', cell / 'manifest.json', cell / '.lock'


def _cache_state(cell: Path, artifact: Path, manifest: Path,
                 lock_path: Path) -> str:
    artifact_exists, manifest_exists = artifact.is_file(), manifest.is_file()
    if artifact_exists and manifest_exists:
        return 'complete'
    if artifact_exists or manifest_exists:
        raise RuntimeError(
            f'incomplete teacher-target cache at {cell}; refusing to overwrite '
            f'{artifact if artifact_exists else manifest}')
    if cell.exists():
        unexpected = sorted(path.name for path in cell.iterdir()
                            if path != lock_path and path.name != lock_path.name)
        if unexpected:
            raise RuntimeError(
                f'unrecognized or partial files in teacher-target cell {cell}: {unexpected}; '
                'refusing to overwrite them')
    return 'missing'


@contextmanager
def _fold_lock(cell: Path, lock_path: Path):
    cell.mkdir(parents=True, exist_ok=True)
    lock_path = require_external_output(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _preserve_parent_rng_state():
    """Keep cache validation and subprocess orchestration RNG-neutral."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = sys.modules.get('torch')
    torch_cpu_state = None
    torch_cuda_device = None
    torch_cuda_state = None
    if torch_module is not None:
        torch_cpu_state = torch_module.random.get_rng_state().clone()
        cuda = getattr(torch_module, 'cuda', None)
        if cuda is not None and cuda.is_initialized():
            torch_cuda_device = int(cuda.current_device())
            torch_cuda_state = cuda.get_rng_state(torch_cuda_device).clone()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_cpu_state is not None:
            torch_module.random.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch_module.cuda.set_rng_state(torch_cuda_state, torch_cuda_device)


def _strict_load(dataset: str, teacher: str, seed: int, fold: int,
                 student_data: Any, cache_root: str | Path | None) -> dict[str, Any]:
    return loso_source.load_teacher_targets(
        dataset, teacher, seed, fold, student_data, cache_root=cache_root)


def _teacher_python(teacher: str, dataset: str) -> Path:
    raw = config.load_model_yaml(teacher)
    env_name = str(raw.get('env', '')).strip()
    if not env_name:
        raise ValueError(f'{teacher}: configs/models/{teacher}.yaml does not declare env')
    if Path(sys.prefix).name == env_name:
        return Path(sys.executable)

    candidates: list[Path] = []
    prefix = Path(sys.prefix)
    if prefix.parent.name in {'envs', 'env'}:
        candidates.append(prefix.parent / env_name / 'bin' / 'python')
    if prefix.name in {'base', 'anaconda3', 'miniconda3', 'mambaforge', 'miniforge3'}:
        candidates.append(prefix / 'envs' / env_name / 'bin' / 'python')
    for env_root in os.environ.get('CONDA_ENVS_PATH', '').split(os.pathsep):
        if env_root:
            candidates.append(Path(env_root) / env_name / 'bin' / 'python')
    mamba_root = os.environ.get('MAMBA_ROOT_PREFIX')
    if mamba_root:
        candidates.append(Path(mamba_root) / 'envs' / env_name / 'bin' / 'python')
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    search = ', '.join(map(str, candidates)) or '(no sibling env candidates)'
    raise FileNotFoundError(
        f'{teacher}/{dataset} requires conda env {env_name!r}; current env is '
        f'{sys.prefix!r}, and no usable interpreter was found at {search}')


def _device_name(gpu: int | str | None) -> str:
    if gpu is None:
        return 'cpu'
    if isinstance(gpu, bool):
        raise TypeError('gpu must be None, a CUDA index, or cpu/cuda:N')
    if isinstance(gpu, int):
        if gpu < 0:
            raise ValueError('CUDA index must be non-negative')
        return f'cuda:{gpu}'
    value = str(gpu).lower()
    if value == 'cpu' or re.fullmatch(r'cuda:\d+', value):
        return value
    raise ValueError(f'unsupported device {gpu!r}; expected cpu or cuda:N')


def ensure_teacher_targets(dataset: str, teacher: str, seed: int, fold: int,
                           student_data: Any, cache_root: str | Path | None = None,
                           gpu: int | str | None = None) -> dict[str, Any]:
    """Return strictly validated targets, creating them only when both files are absent.

    The first strict load is intentionally performed before any export work.
    Existing invalid or partial artifacts are left untouched and reported.
    """
    dataset, teacher = str(dataset), str(teacher).lower()
    seed, fold = int(seed), int(fold)
    if dataset not in SUPPORTED_DATASETS or teacher not in SUPPORTED_TEACHERS:
        raise ValueError(f'unsupported LOSO teacher request: {dataset}/{teacher}')
    n_subjects = int(config.load_dataset_config(dataset)['num_subjects'])
    if seed < 0 or not 0 <= fold < n_subjects:
        raise ValueError(f'invalid seed/fold: seed={seed}, fold={fold}')
    cell, artifact, manifest, lock_path = _target_paths(dataset, teacher, seed, fold, cache_root)
    root_arg = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT

    with _preserve_parent_rng_state():
        try:
            bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
            print(f'[teacher-cache] reuse {dataset}/{teacher} seed={seed} '
                  f'fold={fold + 1:02d}', flush=True)
            return bundle
        except (FileNotFoundError, ValueError) as error:
            state = _cache_state(cell, artifact, manifest, lock_path)
            if state == 'complete':
                # Existing targets can be validated only against their exact
                # teacher input identity. If the whole input directory is
                # absent, prepare that canonical cache in the teacher env and
                # then retry strict loading. Existing partial/invalid inputs
                # and target identity errors remain untouched and fail below.
                if (dataset != 'AlexMI'
                        and not (root_arg / 'model_inputs' / dataset / teacher).exists()):
                    _run_teacher_input_prepare(teacher, dataset, root_arg)
                    bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
                    print(f'[teacher-cache] ready {dataset}/{teacher} seed={seed} '
                          f'fold={fold + 1:02d} (existing targets)', flush=True)
                    return bundle
                raise

        with _fold_lock(cell, lock_path):
            try:
                bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
                print(f'[teacher-cache] reuse-after-lock {dataset}/{teacher} seed={seed} '
                      f'fold={fold + 1:02d}', flush=True)
                return bundle
            except (FileNotFoundError, ValueError):
                if _cache_state(cell, artifact, manifest, lock_path) != 'missing':
                    raise

            device = _device_name(gpu)
            print(f'[teacher-cache] generate {dataset}/{teacher} seed={seed} '
                  f'fold={fold + 1:02d} device={device}', flush=True)
            python = _teacher_python(teacher, dataset)
            command = [
                str(python), '-B', '-m', 'experiments.distill.loso_target_cache',
                '--export-one', dataset, teacher, str(seed), str(fold),
                '--cache-root', str(root_arg), '--device', device, '--lock-held',
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
            print(f'[teacher-cache] ready {dataset}/{teacher} seed={seed} '
                  f'fold={fold + 1:02d}', flush=True)
            return bundle


def _run_teacher_input_prepare(teacher: str, dataset: str, root: Path) -> None:
    python = _teacher_python(teacher, dataset)
    command = [
        str(python), '-B', '-m', 'experiments.distill.loso_target_cache',
        '--prepare-input', dataset, teacher,
        '--cache-root', str(root),
    ]
    print(f'[teacher-cache] prepare-input {dataset}/{teacher}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_targets(cell: Path, artifact: Path, manifest_path: Path,
                          lock_path: Path, arrays: dict[str, np.ndarray],
                          metadata: dict[str, Any]) -> dict[str, Any]:
    """Write a fresh NPZ/manifest pair with per-file atomic replacement."""
    if _cache_state(cell, artifact, manifest_path, lock_path) != 'missing':
        raise RuntimeError(f'teacher-target cache already exists at {cell}; refusing to overwrite')
    temps: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=cell, prefix='targets.', suffix='.npz.tmp',
                                         delete=False) as stream:
            npz_temp = Path(stream.name)
            temps.append(npz_temp)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        saved_manifest = dict(metadata)
        saved_manifest['cache_bytes'] = npz_temp.stat().st_size
        saved_manifest['cache_file_sha256'] = _sha256_file(npz_temp)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cell,
                                         prefix='manifest.', suffix='.json.tmp',
                                         delete=False) as stream:
            manifest_temp = Path(stream.name)
            temps.append(manifest_temp)
            json.dump(saved_manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Neither final path existed under the fold lock. Each rename is atomic;
        # a crash between them leaves a detectable partial pair, never a silent
        # replacement of an older target cache.
        if artifact.exists() or manifest_path.exists():
            raise RuntimeError(f'teacher-target cache appeared during export at {cell}')
        os.replace(npz_temp, artifact)
        os.replace(manifest_temp, manifest_path)
        dir_fd = os.open(cell, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return saved_manifest
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def _ensure_teacher_input(dataset: str, teacher: str, root: Path) -> None:
    """Prepare a teacher cache only when its entire model-input directory is absent."""
    model_dir = root / 'model_inputs' / dataset / teacher
    if model_dir.exists():
        if not (model_dir / 'manifest.json').is_file():
            raise FileNotFoundError(
                f'{dataset}/{teacher}: existing model-input directory is incomplete: {model_dir}; '
                'refusing to overwrite or fall back to unprepared source data')
        return
    if root.resolve(strict=False) != LOSO_CACHE_ROOT.resolve(strict=False):
        raise FileNotFoundError(
            f'{dataset}/{teacher}: model-input cache is absent under {model_dir}; '
            'automatic preparation writes only to the canonical artifact-loso tree')

    # Multiple seeds/folds may arrive together. Serialize creation of this one
    # shared teacher input cache and recheck after acquiring the preparation lock.
    input_parent = require_loso_cache_output(model_dir.parent)
    input_parent.mkdir(parents=True, exist_ok=True)
    input_lock = require_loso_cache_output(input_parent / f'.{teacher}.prepare.lock')
    with input_lock.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            if model_dir.exists():
                if not (model_dir / 'manifest.json').is_file():
                    raise FileNotFoundError(
                        f'{dataset}/{teacher}: existing model-input directory is incomplete: {model_dir}')
                return
            print(f'[teacher-cache] prepare-input {dataset}/{teacher}', flush=True)
            if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
                from experiments.finetune import prepare_bnci14001_wideband_inputs as prepare_001

                x_source, trials, source_manifest = prepare_001.validate_source()
                prepare_001.prepare(teacher, dataset, x_source, trials, source_manifest)
            elif dataset in {'BNCI2014004', 'BNCI2015001'}:
                from experiments.finetune import prepare_loso_source_refresh_004_5001 as prepare_refresh

                prepare_refresh.prepare_model(dataset, teacher)
            else:
                raise ValueError(f'{dataset}: no canonical teacher-input preparation is registered')
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load_teacher_inputs(dataset: str, teacher: str, root: Path):
    _ensure_teacher_input(dataset, teacher, root)
    data = loso_source.load_dataset(dataset, teacher, cache_root=root)
    if data.input_manifest is None or data.input_manifest_sha256 is None:
        raise RuntimeError(f'{dataset}/{teacher}: verified teacher model-input cache was not loaded')
    return data


def _load_checkpoint(dataset: str, teacher: str, seed: int, fold: int,
                     teacher_data: Any):
    baseline_path = loso_source._baseline_teacher_manifest(dataset, teacher, seed, fold)
    baseline = loso_source._json(baseline_path)
    model_config = loso_source.validated_teacher_model_config(
        baseline, baseline_path, teacher_data, dataset, teacher, seed, fold)
    checkpoint = baseline_path.parent / 'final_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'{dataset}/{teacher} seed={seed} fold={fold}: completed baseline checkpoint is missing: {checkpoint}')
    expected_sha = baseline.get('final_model_sha256')
    if not expected_sha:
        raise ValueError(f'{baseline_path}: final_model_sha256 is missing')
    actual_sha = _sha256_file(checkpoint)
    if actual_sha != str(expected_sha):
        raise ValueError(f'{checkpoint}: SHA-256 differs from current baseline manifest')
    return baseline_path, baseline, checkpoint, actual_sha, model_config


def _split_output(output: Any, batch_size: int):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError('teacher forward must return (features, logits)')
    features, logits = output[0], output[-1]
    features = features.reshape(batch_size, -1)
    logits = logits.reshape(batch_size, -1)
    return features, logits


def _build_teacher(dataset: str, teacher: str, model_config: dict[str, Any],
                   input_shape: tuple[int, ...], n_classes: int, device: Any):
    if not bool(model_config.get('skip_preprocess', False)):
        raise ValueError(
            f'{dataset}/{teacher}: baseline must consume its verified canonical input cache '
            'with skip_preprocess=true')
    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        from experiments.finetune import run_loso_config_alignment as reference

        model = reference.make_model(
            teacher, dataset, n_classes, input_shape,
            float(model_config.get('dropout', 0.5)), reference.WIDEBAND_PROFILE)
        return model.to(device), None

    from models import get_adapter

    adapter = get_adapter(teacher, device=device, **model_config)
    return adapter.build(n_classes), adapter


def _export_standard(dataset: str, teacher: str, seed: int, fold: int,
                     root: Path, device_name: str, cell: Path, artifact: Path,
                     manifest_path: Path, lock_path: Path) -> dict[str, Any]:
    import torch

    student_data = loso_source.load_dataset(dataset, cache_root=root)
    teacher_data = _load_teacher_inputs(dataset, teacher, root)
    if (teacher_data.source_manifest_sha256 != student_data.source_manifest_sha256
            or teacher_data.class_mapping != student_data.class_mapping):
        raise ValueError(f'{dataset}: teacher inputs and student source identity differ')
    if not np.array_equal(teacher_data.subjects, student_data.subjects):
        raise ValueError(f'{dataset}: teacher input subject rows differ from source rows')

    student_train = np.flatnonzero(student_data.subjects != fold)
    teacher_uid_rows = {str(uid): index for index, uid in enumerate(teacher_data.trial_uid.tolist())}
    train_uids = np.asarray(student_data.trial_uid[student_train]).astype(str)
    if len(teacher_uid_rows) != len(teacher_data.trial_uid):
        raise ValueError(f'{dataset}/{teacher}: duplicate teacher input trial UIDs')
    if set(train_uids.tolist()) != {
            str(uid) for uid, subject in zip(teacher_data.trial_uid.tolist(),
                                             teacher_data.subjects.tolist())
            if int(subject) != fold}:
        raise ValueError(f'{dataset}/{teacher}: teacher source-train UIDs differ from student source')
    teacher_train = np.asarray([teacher_uid_rows[uid] for uid in train_uids], dtype=np.int64)
    train_y = np.asarray(student_data.y[student_train], dtype=np.int64)
    if not np.array_equal(teacher_data.y[teacher_train], train_y):
        raise ValueError(f'{dataset}/{teacher}: teacher source-train labels differ by trial UID')

    (baseline_path, baseline, checkpoint, checkpoint_sha,
     model_config) = _load_checkpoint(dataset, teacher, seed, fold, teacher_data)
    teacher_manifest_sha = loso_source.sha256_file(baseline_path)
    if device_name.startswith('cuda:'):
        if not torch.cuda.is_available():
            raise RuntimeError(f'{device_name} requested but CUDA is unavailable in {sys.prefix}')
        torch.cuda.set_device(torch.device(device_name))
    device = torch.device(device_name)
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    n_classes = len(student_data.class_mapping)
    build_config = dict(model_config)
    build_config.update(dataset_name=dataset, samples=1000, skip_preprocess=True)
    if teacher == 'cbramod':
        build_config['in_channels'] = int(teacher_data.x.shape[1])
    elif 'in_channels' not in build_config:
        build_config['in_channels'] = 45
    model, adapter = _build_teacher(
        dataset, teacher, build_config, tuple(teacher_data.x.shape[1:]), n_classes, device)
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f'{checkpoint}: expected a raw state_dict checkpoint')
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    model.requires_grad_(False)
    batch_size = int(model_config['batch_size'])
    logits_parts, feature_parts = [], []
    n_batches = (len(teacher_train) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for start in range(0, len(teacher_train), batch_size):
            rows = teacher_train[start:start + batch_size]
            x_batch = torch.as_tensor(
                np.asarray(teacher_data.x[rows], dtype=np.float32),
                dtype=torch.float32, device=device)
            output = adapter.forward(model, x_batch) if adapter is not None else model(x_batch)
            features, logits = _split_output(output, len(rows))
            if logits.shape[1] != n_classes:
                raise ValueError(f'{dataset}/{teacher}: expected {n_classes} logits, got {tuple(logits.shape)}')
            feature_parts.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            batch_number = start // batch_size + 1
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == n_batches:
                print(f'[teacher-cache] forward {dataset}/{teacher} '
                      f'{batch_number}/{n_batches} batches', flush=True)
    features = np.concatenate(feature_parts, axis=0)
    logits = np.concatenate(logits_parts, axis=0)
    if (not np.isfinite(features).all() or not np.isfinite(logits).all()
            or len(features) != len(train_y) or len(logits) != len(train_y)):
        raise ValueError(f'{dataset}/{teacher}: exported targets have invalid values or row counts')
    arrays = {
        'logits': logits,
        'feats': features,
        'y': train_y,
        'trial_uid': train_uids,
    }
    input_manifest = teacher_data.input_manifest or {}
    input_x_sha = input_manifest.get('files', {}).get('X.npy', {}).get('sha256')
    if not input_x_sha:
        raise ValueError(f'{dataset}/{teacher}: verified input manifest lacks X.npy SHA')
    metadata = {
        'dataset': dataset,
        'teacher': teacher,
        'seed': int(seed),
        'fold_zero_based': int(fold),
        'test_subject': int(fold) + 1,
        'status': 'complete',
        'target_data_policy': 'source_train_trials_only_no_held_out_teacher_inference',
        'source_manifest_sha256': student_data.source_manifest_sha256,
        'input_manifest_sha256': teacher_data.input_manifest_sha256,
        'input_X_sha256': input_x_sha,
        'teacher_manifest_path': str(baseline_path),
        'teacher_manifest_sha256': teacher_manifest_sha,
        'teacher_checkpoint_path': str(checkpoint),
        'teacher_checkpoint_sha256': checkpoint_sha,
        'model_config': model_config,
        'class_mapping': student_data.class_mapping,
        'num_classes': n_classes,
        'feature_dim': int(features.shape[1]),
        'logit_dim': int(logits.shape[1]),
        'train_uids': train_uids.tolist(),
        'train_uid_sha256': _array_hash(train_uids),
        'train_labels_sha256': _array_hash(train_y),
        'train_subjects': sorted(np.unique(student_data.subjects[student_train]).astype(int).tolist()),
        'train_count': int(len(train_y)),
        'target_order': 'student_source_order_excluding_held_out_subject',
    }
    _atomic_write_targets(cell, artifact, manifest_path, lock_path, arrays, metadata)
    del model, adapter
    return loso_source.load_teacher_targets(
        dataset, teacher, seed, fold, student_data, cache_root=root)


def _export_alexmi(teacher: str, seed: int, fold: int, device_name: str,
                   cell: Path, artifact: Path, manifest_path: Path,
                   lock_path: Path) -> dict[str, Any]:
    """Reuse the preserved historical exporter after its strict target check."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from data import loso_alexmi
    from experiments.distill import loso_teacher_cache as historical

    # The public ensure path owns this cell's matching historical cache.lock.
    # Make the legacy exporter use that already-held lock rather than deadlock.
    device = None if device_name == 'cpu' else int(device_name.split(':', 1)[1])
    old_lock = historical._lock
    old_cache_root = historical.CACHE_ROOT
    historical._lock = lambda _cell, exclusive: nullcontext()
    historical.CACHE_ROOT = loso_alexmi.ALEXMI_TEACHER_TARGET_ROOT
    try:
        historical.export(SimpleNamespace(
            teacher=teacher, datasets=['AlexMI'], seeds=[seed], folds=[fold], gpu=device))
    finally:
        historical._lock = old_lock
        historical.CACHE_ROOT = old_cache_root
    student_data = loso_source.load_dataset('AlexMI', cache_root=None)
    return loso_source.load_teacher_targets('AlexMI', teacher, seed, fold, student_data)


def _export_one(dataset: str, teacher: str, seed: int, fold: int,
                cache_root: str | Path | None, device_name: str,
                lock_held: bool) -> dict[str, Any]:
    dataset, teacher = str(dataset), str(teacher).lower()
    seed, fold = int(seed), int(fold)
    if dataset not in SUPPORTED_DATASETS or teacher not in SUPPORTED_TEACHERS:
        raise ValueError(f'unsupported LOSO teacher request: {dataset}/{teacher}')
    root = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT
    cell, artifact, manifest_path, lock_path = _target_paths(
        dataset, teacher, seed, fold, root)
    if lock_held:
        if _cache_state(cell, artifact, manifest_path, lock_path) == 'complete':
            return _load_existing_targets(dataset, teacher, seed, fold, root)
        return _export_dispatch(dataset, teacher, seed, fold, root, device_name,
                                cell, artifact, manifest_path, lock_path)
    with _fold_lock(cell, lock_path):
        if _cache_state(cell, artifact, manifest_path, lock_path) == 'complete':
            return _load_existing_targets(dataset, teacher, seed, fold, root)
        return _export_dispatch(dataset, teacher, seed, fold, root, device_name,
                                cell, artifact, manifest_path, lock_path)


def _load_existing_targets(dataset: str, teacher: str, seed: int, fold: int,
                           root: Path) -> dict[str, Any]:
    """Strictly reuse a complete pair, preparing only a wholly absent input cache."""
    student_data = loso_source.load_dataset(dataset, cache_root=root)
    if (dataset != 'AlexMI'
            and not (root / 'model_inputs' / dataset / teacher).exists()):
        _ensure_teacher_input(dataset, teacher, root)
        student_data = loso_source.load_dataset(dataset, cache_root=root)
    return loso_source.load_teacher_targets(
        dataset, teacher, seed, fold, student_data, cache_root=root)


def _prepare_input_one(dataset: str, teacher: str,
                       cache_root: str | Path | None) -> None:
    if dataset not in SUPPORTED_DATASETS or teacher not in SUPPORTED_TEACHERS:
        raise ValueError(f'unsupported LOSO teacher input request: {dataset}/{teacher}')
    if dataset == 'AlexMI':
        return
    root = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT
    _ensure_teacher_input(dataset, teacher, root)
    data = loso_source.load_dataset(dataset, teacher, cache_root=root)
    if data.input_manifest is None or data.input_manifest_sha256 is None:
        raise RuntimeError(f'{dataset}/{teacher}: preparation did not create a verified input cache')


def _export_dispatch(dataset: str, teacher: str, seed: int, fold: int,
                     root: Path, device_name: str, cell: Path, artifact: Path,
                     manifest_path: Path, lock_path: Path) -> dict[str, Any]:
    if dataset == 'AlexMI':
        return _export_alexmi(teacher, seed, fold, device_name,
                              cell, artifact, manifest_path, lock_path)
    return _export_standard(dataset, teacher, seed, fold, root, device_name,
                            cell, artifact, manifest_path, lock_path)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-one', nargs=4, metavar=('DATASET', 'TEACHER', 'SEED', 'FOLD'),
                        help='export exactly one completed baseline fold')
    parser.add_argument('--prepare-input', nargs=2, metavar=('DATASET', 'TEACHER'),
                        help='prepare and verify one missing teacher model-input cache')
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:N')
    parser.add_argument('--lock-held', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.export_one is None) == (args.prepare_input is None):
        parser.error('provide exactly one of --export-one or --prepare-input')
    if args.prepare_input is not None:
        if args.lock_held:
            parser.error('--lock-held applies only to --export-one')
        dataset, teacher = args.prepare_input
        _prepare_input_one(dataset, teacher, args.cache_root)
        return 0
    dataset, teacher, seed, fold = args.export_one
    seed, fold = int(seed), int(fold)
    _device_name(args.device)
    _export_one(dataset, teacher, seed, fold, args.cache_root,
                args.device.lower(), args.lock_held)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
