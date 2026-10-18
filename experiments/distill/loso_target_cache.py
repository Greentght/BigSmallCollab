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
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

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
    artifact_exists = artifact.exists() or artifact.is_symlink()
    manifest_exists = manifest.exists() or manifest.is_symlink()
    if artifact.is_file() and manifest.is_file():
        return 'complete'
    if artifact_exists or manifest_exists:
        return 'partial'
    # Crashes can leave temporary siblings, which are safe to replace under
    # the fold lock. Keep unrecognized files untouched and out of cache state.
    return 'missing'


def _cache_state_reason(cell: Path, artifact: Path, manifest: Path) -> str:
    artifact_exists = artifact.exists() or artifact.is_symlink()
    manifest_exists = manifest.exists() or manifest.is_symlink()
    if artifact.is_file() and manifest.is_file():
        return 'both target files exist but strict validation failed'
    if artifact_exists:
        return f'manifest is missing: {manifest}'
    if manifest_exists:
        return f'target artifact is missing: {artifact}'
    return f'target pair is missing under {cell}'


def _recoverable_cache_error(error: Exception) -> bool:
    if isinstance(error, (PermissionError, MemoryError)):
        return False
    return isinstance(error, (FileNotFoundError, ValueError, RuntimeError, KeyError,
                              EOFError, OSError, TypeError, AttributeError,
                              zipfile.BadZipFile))


def _log_target_cache(action: str, dataset: str, teacher: str,
                      seed: int, fold: int, reason: str | None = None) -> None:
    message = (f'[teacher-cache] {action} dataset={dataset} teacher={teacher} '
               f'seed={seed} fold={fold + 1:02d}')
    if reason:
        message += f' reason={reason}'
    print(message, flush=True)


def _clear_dataset_memory_cache(dataset: str, model: str | None = None) -> None:
    """Drop loader memmaps so rebuilt files are reopened from their new inode."""
    cache = getattr(loso_source, '_DATA_CACHE', None)
    if not isinstance(cache, dict):
        return
    for key in list(cache):
        if key[0] == dataset and (model is None or key[1] == model):
            cache.pop(key, None)


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


def _run_streaming_subprocess(command: list[str], cwd: Path) -> None:
    """Stream child output through the parent logger and raise on failure."""
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    if process.stdout is None:
        raise RuntimeError('subprocess stdout pipe was not created')
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        process.stdout.close()
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


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
    """Return strict targets, rebuilding missing, partial, or mismatched caches.

    The first strict load happens before export work; failures are rechecked
    under the per-fold lock before any cache or teacher preparation begins.
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
            _clear_dataset_memory_cache(dataset, teacher)
            bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
            _log_target_cache('reuse', dataset, teacher, seed, fold)
            return bundle
        except Exception as error:
            if not _recoverable_cache_error(error):
                raise
            state = _cache_state(cell, artifact, manifest, lock_path)
            action = 'missing' if state == 'missing' else 'mismatch'
            reason = (f'{_cache_state_reason(cell, artifact, manifest)}; strict load '
                      f'{type(error).__name__}: {error}')
            _log_target_cache(action, dataset, teacher, seed, fold, reason)

        with _fold_lock(cell, lock_path):
            try:
                _clear_dataset_memory_cache(dataset, teacher)
                bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
                _log_target_cache('reuse-after-lock', dataset, teacher, seed, fold)
                return bundle
            except Exception as error:
                if not _recoverable_cache_error(error):
                    raise
                state = _cache_state(cell, artifact, manifest, lock_path)
                reason = (f'{_cache_state_reason(cell, artifact, manifest)}; strict load '
                          f'{type(error).__name__}: {error}')
                _log_target_cache('rebuild', dataset, teacher, seed, fold, reason)

            # Validate or repair the teacher's shared model-input cache before
            # exporting this fold. The helper runs in the teacher environment,
            # where its preprocessing dependencies are available.
            if dataset != 'AlexMI':
                _run_teacher_input_prepare(teacher, dataset, root_arg)
                _clear_dataset_memory_cache(dataset, teacher)
                try:
                    bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
                    _log_target_cache('reuse-after-input-rebuild', dataset, teacher, seed, fold)
                    return bundle
                except Exception as error:
                    if not _recoverable_cache_error(error):
                        raise
                    _log_target_cache(
                        'mismatch', dataset, teacher, seed, fold,
                        f'targets remain invalid after teacher-input validation: '
                        f'{type(error).__name__}: {error}')

            device = _device_name(gpu)
            _log_target_cache('rebuild-start', dataset, teacher, seed, fold,
                              f'export device={device}')
            python = _teacher_python(teacher, dataset)
            command = [
                str(python), '-B', '-m', 'experiments.distill.loso_target_cache',
                '--export-one', dataset, teacher, str(seed), str(fold),
                '--cache-root', str(root_arg), '--device', device, '--lock-held',
                '--rebuild-invalid',
            ]
            _run_streaming_subprocess(command, cwd=ROOT)
            _clear_dataset_memory_cache(dataset, teacher)
            bundle = _strict_load(dataset, teacher, seed, fold, student_data, cache_root)
            _log_target_cache('ready', dataset, teacher, seed, fold)
            return bundle


def _run_teacher_input_prepare(teacher: str, dataset: str, root: Path) -> None:
    python = _teacher_python(teacher, dataset)
    command = [
        str(python), '-B', '-m', 'experiments.distill.loso_target_cache',
        '--prepare-input', dataset, teacher,
        '--cache-root', str(root),
    ]
    print(f'[teacher-cache] prepare-input dataset={dataset} teacher={teacher}', flush=True)
    _run_streaming_subprocess(command, cwd=ROOT)
    _clear_dataset_memory_cache(dataset, teacher)


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
                          metadata: dict[str, Any], *, replace_existing: bool = False,
                          retain_backup: bool = False):
    """Write a target pair with rollback if either atomic replacement fails.

    The caller holds the fold lock. Old files are moved aside only after both
    new files have been fully written and fsynced. When ``retain_backup`` is
    true, the caller owns finalizing the pair after strict reload validation.
    """
    state = _cache_state(cell, artifact, manifest_path, lock_path)
    if state != 'missing' and not replace_existing:
        raise RuntimeError(f'teacher-target cache already exists at {cell}')
    temps: list[Path] = []
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
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
        for destination in (artifact, manifest_path):
            if destination.exists() or destination.is_symlink():
                backup = cell / f'.{destination.name}.backup-{os.getpid()}-{time.time_ns()}'
                os.replace(destination, backup)
                backups[destination] = backup
        os.replace(npz_temp, artifact)
        installed.append(artifact)
        os.replace(manifest_temp, manifest_path)
        installed.append(manifest_path)
        dir_fd = os.open(cell, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if retain_backup:
            return saved_manifest, backups
        _discard_target_backups(backups)
        return saved_manifest
    except BaseException:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def _discard_target_backups(backups: dict[Path, Path]) -> None:
    for backup in backups.values():
        _remove_path(backup)


def _restore_target_backups(backups: dict[Path, Path], *destinations: Path) -> None:
    for destination in destinations:
        _remove_path(destination)
    for destination, backup in backups.items():
        if backup.exists():
            os.replace(backup, destination)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _input_cache_dir(dataset: str, model: str, root: Path) -> Path:
    return root / 'model_inputs' / dataset / model


def _has_registered_input_recipe(dataset: str, model: str) -> bool:
    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        return model in {'cbramod', 'mirepnet'}
    if dataset in {'BNCI2014004', 'BNCI2015001'}:
        return model in {'mirepnet', 'cbramod', 'ifnet', 'eegnet', 'adfcnn'}
    return False


def _canonical_input_root(root: Path) -> bool:
    return root.resolve(strict=False) == LOSO_CACHE_ROOT.resolve(strict=False)


@contextmanager
def _input_lock(dataset: str, model: str, root: Path):
    parent = require_loso_cache_output(root / 'model_inputs' / dataset)
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = require_loso_cache_output(parent / f'.{model}.prepare.lock')
    with lock_path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_input_manifest(model_dir: Path) -> tuple[bytes | None, dict[str, Any] | None]:
    path = model_dir / 'manifest.json'
    if not path.is_file():
        return None, None
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload, None
    return payload, value if isinstance(value, dict) else None


def _existing_input_identity_matches(dataset: str, model: str,
                                     manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    try:
        if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
            from experiments.finetune import prepare_bnci14001_wideband_inputs as prepare_001

            source_sha = _sha256_file(prepare_001.SOURCE / 'manifest.json')
            if not (manifest.get('dataset') == dataset
                    and manifest.get('model') == model
                    and manifest.get('source_manifest_sha256') == source_sha
                    and manifest.get('resolved_profile') == prepare_001.recipe(model, dataset)):
                return False
            expected_shape = tuple(manifest.get('input_shape', ()))
            trial_count = int(manifest.get('trial_count', -1))
            loso_source._validate_cached_input_policy(
                dataset, model, manifest, (trial_count, *expected_shape))
            return trial_count > 0
        if dataset in {'BNCI2014004', 'BNCI2015001'}:
            from experiments.finetune import prepare_loso_source_refresh_004_5001 as prepare_refresh

            prepare_refresh.validate_input_identity(
                dataset, model, manifest, config.load_model_config(model, dataset, 'loso'))
            return True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False
    return False


def _restore_manifest_bytes(manifest_path: Path, payload: bytes) -> None:
    temp = manifest_path.with_name(f'.manifest.restore-{os.getpid()}-{time.time_ns()}.tmp')
    try:
        with temp.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, manifest_path)
    finally:
        temp.unlink(missing_ok=True)


def _preserve_input_identity_if_reproduced(model_dir: Path, old_bytes: bytes | None,
                                           old_manifest: dict[str, Any] | None,
                                           identity_matches: bool) -> bool:
    if not old_bytes or not identity_matches or not isinstance(old_manifest, dict):
        return False
    old_files = old_manifest.get('files')
    if not isinstance(old_files, dict):
        return False
    required = ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv')
    for name in required:
        record = old_files.get(name)
        path = model_dir / name
        try:
            matches = (isinstance(record, dict) and path.is_file()
                       and int(record.get('bytes', -1)) == path.stat().st_size
                       and str(record.get('sha256', '')) == _sha256_file(path))
        except (OSError, TypeError, ValueError):
            matches = False
        if not matches:
            return False
    _restore_manifest_bytes(model_dir / 'manifest.json', old_bytes)
    return True


def _prepare_registered_input(dataset: str, model: str, root: Path) -> None:
    if not _canonical_input_root(root):
        raise FileNotFoundError(
            f'{dataset}/{model}: automatic model-input preparation is restricted to '
            f'the canonical artifact-loso tree ({LOSO_CACHE_ROOT}), requested {root}')
    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        from experiments.finetune import prepare_bnci14001_wideband_inputs as prepare_001

        x_source, trials, source_manifest = prepare_001.validate_source()
        prepare_001.prepare(model, dataset, x_source, trials, source_manifest)
        return
    if dataset in {'BNCI2014004', 'BNCI2015001'}:
        from experiments.finetune import prepare_loso_source_refresh_004_5001 as prepare_refresh

        prepare_refresh.prepare_model(dataset, model)
        return
    raise ValueError(f'{dataset}/{model}: no canonical model-input preparation is registered')


def _archive_input_dir(model_dir: Path) -> Path | None:
    if not (model_dir.exists() or model_dir.is_symlink()):
        return None
    archived = model_dir.with_name(
        f'.{model_dir.name}.input-backup-{os.getpid()}-{time.time_ns()}')
    os.replace(model_dir, archived)
    return archived


def _load_or_rebuild_input_locked(dataset: str, model: str, root: Path,
                                  auto_prepare: bool, initial_error: Exception | None = None):
    model_dir = _input_cache_dir(dataset, model, root)
    with _input_lock(dataset, model, root):
        _clear_dataset_memory_cache(dataset, model)
        try:
            data = loso_source.load_dataset(dataset, model, cache_root=root)
            if not (auto_prepare and _has_registered_input_recipe(dataset, model)
                    and not model_dir.exists()):
                return data
        except Exception as retry_error:
            if not _recoverable_cache_error(retry_error):
                raise
            initial_error = retry_error

        if not auto_prepare:
            if initial_error is not None:
                raise initial_error
            return loso_source.load_dataset(dataset, model, cache_root=root)

        registered = _has_registered_input_recipe(dataset, model)
        if (dataset in {'BNCI2014001', 'BNCI2014001-4'}
                and model in {'ifnet', 'eegnet', 'adfcnn'} and model_dir.exists()):
            archived = _archive_input_dir(model_dir)
            _clear_dataset_memory_cache(dataset, model)
            try:
                data = loso_source.load_dataset(dataset, model, cache_root=root)
            except Exception:
                if archived is not None and archived.exists():
                    os.replace(archived, model_dir)
                _clear_dataset_memory_cache(dataset, model)
                raise
            if archived is not None:
                _remove_path(archived)
            print(f'[model-input-cache] archived invalid optional 001 cache '
                  f'dataset={dataset} model={model}; using verified wide-source preprocessing',
                  flush=True)
            return data

        if not registered:
            if initial_error is not None:
                raise initial_error
            return loso_source.load_dataset(dataset, model, cache_root=root)

        old_bytes, old_manifest = _read_input_manifest(model_dir)
        old_identity_matches = _existing_input_identity_matches(
            dataset, model, old_manifest)
        archived = _archive_input_dir(model_dir)
        _clear_dataset_memory_cache(dataset, model)
        try:
            _prepare_registered_input(dataset, model, root)
            generated_manifest_bytes = (model_dir / 'manifest.json').read_bytes()
            preserved = _preserve_input_identity_if_reproduced(
                model_dir, old_bytes, old_manifest, old_identity_matches)
            _clear_dataset_memory_cache(dataset, model)
            try:
                data = loso_source.load_dataset(dataset, model, cache_root=root)
            except Exception as preserved_error:
                if not preserved or not _recoverable_cache_error(preserved_error):
                    raise
                _restore_manifest_bytes(
                    model_dir / 'manifest.json', generated_manifest_bytes)
                _clear_dataset_memory_cache(dataset, model)
                data = loso_source.load_dataset(dataset, model, cache_root=root)
                preserved = False
        except Exception:
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            if archived is not None and archived.exists():
                os.replace(archived, model_dir)
            _clear_dataset_memory_cache(dataset, model)
            raise
        if archived is not None:
            _remove_path(archived)
        detail = 'preserved prior input manifest identity' if preserved else 'wrote current recipe identity'
        print(f'[model-input-cache] rebuilt dataset={dataset} model={model} '
              f'reason={initial_error or "missing required model-input cache"}; {detail}',
              flush=True)
        return data


def _load_or_prepare_loso_dataset_impl(dataset: str, model: str,
                                      cache_root: str | Path | None = None,
                                      *, auto_prepare: bool = True):
    """Strictly load LOSO inputs, rebuilding registered broken caches if enabled.

    Dataset source files are validated independently before any model-input
    cache is archived. AlexMI retains its legacy source and layout unchanged.
    """
    dataset, model = str(dataset), str(model).lower()
    root = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT
    if dataset == 'AlexMI':
        _clear_dataset_memory_cache(dataset, model)
        return loso_source.load_dataset(dataset, model, cache_root=cache_root)

    model_dir = _input_cache_dir(dataset, model, root)
    _clear_dataset_memory_cache(dataset, model)
    try:
        data = loso_source.load_dataset(dataset, model, cache_root=root)
    except Exception as input_error:
        if not _recoverable_cache_error(input_error):
            raise
        # This model-free load verifies raw configured sources before the
        # helper treats any failure as a rebuildable model-input cache issue.
        _clear_dataset_memory_cache(dataset, None)
        try:
            loso_source.load_dataset(dataset, None, cache_root=root)
        except Exception as source_error:
            raise source_error from input_error
        if not auto_prepare:
            raise input_error
        return _load_or_rebuild_input_locked(
            dataset, model, root, auto_prepare=True, initial_error=input_error)

    if auto_prepare and _has_registered_input_recipe(dataset, model) and not model_dir.exists():
        return _load_or_rebuild_input_locked(dataset, model, root, auto_prepare=True)
    return data


def load_or_prepare_loso_dataset(dataset: str, model: str,
                                 cache_root: str | Path | None = None,
                                 *, auto_prepare: bool = True):
    """Load or rebuild LOSO inputs without changing the caller's RNG state."""
    with _preserve_parent_rng_state():
        return _load_or_prepare_loso_dataset_impl(
            dataset, model, cache_root=cache_root, auto_prepare=auto_prepare)


def _ensure_teacher_input(dataset: str, teacher: str, root: Path):
    return load_or_prepare_loso_dataset(dataset, teacher, cache_root=root, auto_prepare=True)


def _load_teacher_inputs(dataset: str, teacher: str, root: Path):
    data = _ensure_teacher_input(dataset, teacher, root)
    if _has_registered_input_recipe(dataset, teacher):
        if data.input_manifest is None or data.input_manifest_sha256 is None:
            raise RuntimeError(f'{dataset}/{teacher}: verified teacher model-input cache was not loaded')
    return data


def _load_checkpoint(dataset: str, teacher: str, seed: int, fold: int,
                     teacher_data: Any, device_name: str = 'cpu'):
    from experiments.distill.loso_teacher_prepare import ensure_teacher_checkpoint

    ensure_teacher_checkpoint(dataset, teacher, seed, fold, teacher_data, device_name)
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
                     manifest_path: Path, lock_path: Path,
                     replace_existing: bool = False) -> dict[str, Any]:
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
     model_config) = _load_checkpoint(dataset, teacher, seed, fold,
                                      teacher_data, device_name)
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
    _, backups = _atomic_write_targets(
        cell, artifact, manifest_path, lock_path, arrays, metadata,
        replace_existing=replace_existing, retain_backup=True)
    del model, adapter
    try:
        bundle = loso_source.load_teacher_targets(
            dataset, teacher, seed, fold, student_data, cache_root=root)
    except BaseException:
        _restore_target_backups(backups, artifact, manifest_path)
        raise
    _discard_target_backups(backups)
    return bundle


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
    teacher_data = loso_source.load_dataset('AlexMI', teacher)
    from experiments.distill.loso_teacher_prepare import ensure_teacher_checkpoint

    ensure_teacher_checkpoint('AlexMI', teacher, seed, fold, teacher_data, device_name)
    old_lock = historical._lock
    old_cache_root = historical.CACHE_ROOT
    target_backups: dict[Path, Path] = {}
    for path in (artifact, manifest_path):
        if path.exists():
            backup = cell / f'.{path.name}.backup-{os.getpid()}-{time.time_ns()}'
            os.replace(path, backup)
            target_backups[path] = backup
    historical._lock = lambda _cell, exclusive: nullcontext()
    historical.CACHE_ROOT = loso_alexmi.ALEXMI_TEACHER_TARGET_ROOT
    torch_module = sys.modules.get('torch')
    original_manual_seed_all = None
    try:
        if torch_module is not None:
            if device_name.startswith('cuda:'):
                torch_module.cuda.set_device(torch_module.device(device_name))
            original_manual_seed_all = torch_module.cuda.manual_seed_all
            torch_module.cuda.manual_seed_all = lambda _seed: None
        historical.export(SimpleNamespace(
            teacher=teacher, datasets=['AlexMI'], seeds=[seed], folds=[fold], gpu=device))
        student_data = loso_source.load_dataset('AlexMI', cache_root=None)
        bundle = loso_source.load_teacher_targets('AlexMI', teacher, seed, fold, student_data)
    except BaseException:
        _restore_target_backups(target_backups, artifact, manifest_path)
        raise
    finally:
        historical._lock = old_lock
        historical.CACHE_ROOT = old_cache_root
        if torch_module is not None and original_manual_seed_all is not None:
            torch_module.cuda.manual_seed_all = original_manual_seed_all
    _discard_target_backups(target_backups)
    return bundle


def _export_one(dataset: str, teacher: str, seed: int, fold: int,
                cache_root: str | Path | None, device_name: str,
                lock_held: bool, rebuild_invalid: bool = False) -> dict[str, Any]:
    dataset, teacher = str(dataset), str(teacher).lower()
    seed, fold = int(seed), int(fold)
    if dataset not in SUPPORTED_DATASETS or teacher not in SUPPORTED_TEACHERS:
        raise ValueError(f'unsupported LOSO teacher request: {dataset}/{teacher}')
    root = Path(cache_root) if cache_root is not None else LOSO_CACHE_ROOT
    cell, artifact, manifest_path, lock_path = _target_paths(
        dataset, teacher, seed, fold, root)
    if lock_held:
        return _export_or_reuse_one(
            dataset, teacher, seed, fold, root, device_name,
            cell, artifact, manifest_path, lock_path, rebuild_invalid)
    with _fold_lock(cell, lock_path):
        return _export_or_reuse_one(
            dataset, teacher, seed, fold, root, device_name,
            cell, artifact, manifest_path, lock_path, rebuild_invalid)


def _export_or_reuse_one(dataset: str, teacher: str, seed: int, fold: int,
                         root: Path, device_name: str, cell: Path,
                         artifact: Path, manifest_path: Path, lock_path: Path,
                         rebuild_invalid: bool) -> dict[str, Any]:
    state = _cache_state(cell, artifact, manifest_path, lock_path)
    if state == 'complete' or state == 'partial':
        try:
            bundle = _load_existing_targets(dataset, teacher, seed, fold, root)
            _log_target_cache('reuse-in-exporter', dataset, teacher, seed, fold)
            return bundle
        except Exception as error:
            if not _recoverable_cache_error(error):
                raise
            reason = f'{type(error).__name__}: {error}'
            _log_target_cache('mismatch', dataset, teacher, seed, fold, reason)
            # Input/source validation is separate from target validity. These
            # reads make source failures visible before replacing old targets.
            if dataset != 'AlexMI':
                _clear_dataset_memory_cache(dataset, None)
                loso_source.load_dataset(dataset, None, cache_root=root)
                _load_teacher_inputs(dataset, teacher, root)
            else:
                loso_source.load_dataset(dataset, teacher, cache_root=root)
    else:
        _log_target_cache('missing', dataset, teacher, seed, fold,
                          _cache_state_reason(cell, artifact, manifest_path))
    return _export_dispatch(
        dataset, teacher, seed, fold, root, device_name,
        cell, artifact, manifest_path, lock_path,
        replace_existing=(state != 'missing'))


def _load_existing_targets(dataset: str, teacher: str, seed: int, fold: int,
                           root: Path) -> dict[str, Any]:
    """Strictly load a target pair after validating the current source inputs."""
    student_data = loso_source.load_dataset(dataset, cache_root=root)
    if dataset != 'AlexMI':
        _load_teacher_inputs(dataset, teacher, root)
        _clear_dataset_memory_cache(dataset, teacher)
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
                     manifest_path: Path, lock_path: Path,
                     replace_existing: bool = False) -> dict[str, Any]:
    if dataset == 'AlexMI':
        return _export_alexmi(teacher, seed, fold, device_name,
                              cell, artifact, manifest_path, lock_path)
    return _export_standard(dataset, teacher, seed, fold, root, device_name,
                            cell, artifact, manifest_path, lock_path,
                            replace_existing=replace_existing)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-one', nargs=4, metavar=('DATASET', 'TEACHER', 'SEED', 'FOLD'),
                        help='export exactly one completed baseline fold')
    parser.add_argument('--prepare-input', nargs=2, metavar=('DATASET', 'TEACHER'),
                        help='strictly validate or rebuild one teacher model-input cache')
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:N')
    parser.add_argument('--lock-held', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--rebuild-invalid', action='store_true', help=argparse.SUPPRESS)
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
                args.device.lower(), args.lock_held, args.rebuild_invalid)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
