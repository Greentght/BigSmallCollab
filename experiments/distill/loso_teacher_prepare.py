"""Prepare a missing same-fold LOSO teacher with its current formal recipe.

Teacher-target caches are disposable acceleration artifacts. This module is
called only after the canonical same-fold teacher checkpoint is found missing
or fails strict identity validation. It delegates training to the existing
dataset-specific LOSO workers and preserves any superseded result artifacts
under the external migrations tree before rebuilding their canonical paths.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from experiments.storage import (
    RESULTS_ROOT, external_path, require_external_output, resolve_local_file,
)


MIGRATIONS_ROOT = Path('/data1/llx/BigSmallcollab/migrations/loso_teacher_rebuild')
_SUPPORTED_DATASETS = {
    'BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI',
}
_SUPPORTED_TEACHERS = {'cbramod', 'mirepnet'}
_ALEXMI_RUNNER_DEFAULT_DATASETS = {
    'BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001', 'AlexMI',
}
_ALEXMI_RUNNER_DEFAULT_SEEDS = {'666', '667', '668'}


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _device(value: str | int | torch.device) -> torch.device:
    if isinstance(value, bool):
        raise TypeError('device_name must be cpu, cuda:N, or a CUDA index')
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, int):
        if value < 0:
            raise ValueError('CUDA index must be non-negative')
        device = torch.device(f'cuda:{value}')
    else:
        raw = str(value).lower()
        if raw == 'cpu':
            device = torch.device('cpu')
        elif re.fullmatch(r'cuda:\d+', raw):
            device = torch.device(raw)
        else:
            raise ValueError(f'unsupported device {value!r}; expected cpu or cuda:N')
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError(f'unsupported training device: {device}')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'{device} was requested but CUDA is unavailable')
        if device.index is None:
            raise ValueError('CUDA device must include its explicit index, for example cuda:1')
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f'{device} is outside the visible CUDA range 0..{torch.cuda.device_count()-1}')
        # Select exactly the requested card. Never substitute another GPU.
        torch.cuda.set_device(device)
    return device


@contextmanager
def _selected_device_seed_scope(device: torch.device) -> Iterator[None]:
    """Keep legacy torch.manual_seed calls from seeding every visible GPU."""
    old_all = torch.cuda.manual_seed_all
    old_single = torch.cuda.manual_seed

    def selected_manual_seed(seed: int) -> None:
        if device.type == 'cuda':
            old_single(seed)

    torch.cuda.manual_seed_all = lambda _seed: None
    torch.cuda.manual_seed = selected_manual_seed
    try:
        yield
    finally:
        torch.cuda.manual_seed_all = old_all
        torch.cuda.manual_seed = old_single


@contextmanager
def _cpu_fold_rng_compat(module: Any) -> Iterator[None]:
    """Make the 004/5001 fold API's RNG checkpoint CPU-safe without changing its recipe."""
    old_capture = module.rng_state
    old_restore = module.restore_rng

    def capture(device: torch.device, loader_generator: torch.Generator) -> dict:
        return {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': None,
            'loader_generator': loader_generator.get_state(),
        }

    def restore(state: dict, device: torch.device,
                loader_generator: torch.Generator) -> None:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'].cpu())
        loader_generator.set_state(state['loader_generator'].cpu())

    module.rng_state = capture
    module.restore_rng = restore
    try:
        yield
    finally:
        module.rng_state = old_capture
        module.restore_rng = old_restore


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(resolve_local_file(path)).read_text())


def load_verified_alexmi_spec(protocol: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the retired AlexMI recipe only after restoring its frozen bytes.

    Storage migration rewrote three output path prefixes in the archived YAML,
    which makes the legacy runner's whole-file snapshot check fail. Reversing
    only that known path relocation is accepted when the reconstructed bytes
    exactly match the frozen SHA recorded by the source snapshot. No recipe or
    training configuration is inferred here.
    """
    if protocol is None:
        from experiments.finetune import run_loso_five_datasets as protocol

    snapshot_path = Path(resolve_local_file(protocol.SNAPSHOT_PATH))
    snapshot = json.loads(snapshot_path.read_text())
    expected_sha = str(snapshot.get('spec_sha256', ''))
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise ValueError(f'{snapshot_path}: frozen spec SHA-256 is missing or invalid')

    spec_path = Path(resolve_local_file(protocol.SPEC_PATH))
    payload = spec_path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        migrated_prefix = (str(RESULTS_ROOT / 'reproductions').encode() + b'/')
        original_prefix = b'results/reproductions/'
        if migrated_prefix not in payload:
            raise RuntimeError(
                f'{spec_path}: retired AlexMI recipe differs from its frozen source snapshot')
        restored = payload.replace(migrated_prefix, original_prefix)
        restored_sha = hashlib.sha256(restored).hexdigest()
        if restored_sha != expected_sha:
            raise RuntimeError(
                f'{spec_path}: path-relocated AlexMI recipe cannot be restored to its '
                f'frozen source snapshot (expected {expected_sha}, got {restored_sha})')
        payload = restored

    spec = yaml.safe_load(payload)
    if not isinstance(spec, dict) or 'main_recipes' not in spec or 'datasets' not in spec:
        raise ValueError(f'{spec_path}: frozen AlexMI recipe has an invalid structure')
    return spec, snapshot


def _option_values(argv: list[str], options: set[str]) -> set[str]:
    values: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        matched = next((option for option in options
                        if token == option or token.startswith(option + '=')), None)
        if matched is None:
            index += 1
            continue
        if token.startswith(matched + '='):
            values.add(token.split('=', 1)[1])
            index += 1
            continue
        index += 1
        while index < len(argv) and not argv[index].startswith('-'):
            values.add(argv[index])
            index += 1
    return values


def _assert_no_active_worker(script_name: str, dataset: str,
                             teacher: str, seed: int) -> bool:
    """Wait for a baseline worker to exit before touching its result paths.

    The 004/5001 and AlexMI baseline entrypoints have no shared fold lock, so
    process inspection is required before archival. The caller revalidates
    the checkpoint after this returns in case the worker completed it while we
    waited. Return whether any matching worker was seen.
    """
    dataset_aliases = {dataset}
    if dataset == 'BNCI2014001':
        dataset_aliases.add('001')
    elif dataset == 'BNCI2014001-4':
        dataset_aliases.add('001-4')
    elif dataset == 'BNCI2014004':
        dataset_aliases.add('004')
    elif dataset == 'BNCI2015001':
        dataset_aliases.add('5001')
    current_pid = os.getpid()
    waited = False
    while True:
        try:
            proc = subprocess.run(
                ['ps', '-eo', 'pid=,args='], check=True, capture_output=True,
                text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f'cannot inspect running trainer processes safely: {error}') from error
        active_pid = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_text, _, command = line.partition(' ')
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            script_stem = Path(script_name).stem
            if (pid == current_pid
                    or (script_name not in command and script_stem not in command)):
                continue
            try:
                args = __import__('shlex').split(command)
            except ValueError:
                args = command.split()
            models = _option_values(args, {'--model'})
            seeds = _option_values(args, {'--seed', '--seeds'})
            datasets = _option_values(args, {'--dataset', '--datasets'})
            if script_stem == 'run_loso_five_datasets':
                if not seeds:
                    seeds = set(_ALEXMI_RUNNER_DEFAULT_SEEDS)
                if not datasets:
                    datasets = set(_ALEXMI_RUNNER_DEFAULT_DATASETS)
            if (teacher in models and str(seed) in seeds
                    and dataset_aliases.intersection(datasets)):
                active_pid = pid
                break
        if active_pid is None:
            if waited:
                print(f'[teacher-prepare] active {script_name} worker exited for '
                      f'{dataset}/{teacher}/seed_{seed}; rechecking its checkpoint',
                      flush=True)
            return waited
        waited = True
        print(f'[teacher-prepare] waiting for active {script_name} worker '
              f'pid={active_pid} ({dataset}/{teacher}/seed_{seed}); '
              'its output will be revalidated before any rebuild', flush=True)
        time.sleep(5)


def _archive(path: Path, dataset: str, teacher: str, seed: int,
             reason: str, fold: int | None = None) -> Path | None:
    """Move old artifacts intact to the approved external migration tree."""
    path = external_path(path)
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'.{time.time_ns() % 1_000_000_000:09d}Z'
    fold_name = f'fold_{fold + 1:02d}' if fold is not None else path.name
    base = MIGRATIONS_ROOT / dataset / teacher / f'seed_{seed}'
    if fold is not None:
        base = base / fold_name
    destination = require_external_output(base / f'{path.name}.{stamp}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    record = {
        'source_path': str(path), 'archive_path': str(destination),
        'dataset': dataset, 'teacher': teacher, 'seed': int(seed),
        'fold_zero_based': fold, 'reason': reason,
        'archived_utc': datetime.now(timezone.utc).isoformat(),
    }
    record_path = require_external_output(
        destination.parent / f'{destination.name}.archive_record.json')
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(f'[teacher-prepare] archived {path} -> {destination}; reason={reason}', flush=True)
    return destination


def _validate_standard_checkpoint(dataset: str, teacher: str, seed: int,
                                  fold: int, teacher_data: Any):
    from data import loso_source

    manifest_path = Path(resolve_local_file(
        loso_source._baseline_teacher_manifest(dataset, teacher, seed, fold)))
    manifest = loso_source._json(manifest_path)
    resolved_config = loso_source.validated_teacher_model_config(
        manifest, manifest_path, teacher_data, dataset, teacher, seed, fold)
    checkpoint = manifest_path.parent / 'final_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: same-fold final teacher checkpoint is missing')
    expected_sha = manifest.get('final_model_sha256')
    if not expected_sha:
        raise ValueError(f'{manifest_path}: final_model_sha256 is missing')
    actual_sha = _sha256(checkpoint)
    if actual_sha != str(expected_sha):
        raise ValueError(f'{checkpoint}: SHA-256 differs from same-fold baseline manifest')

    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        from experiments.finetune import run_loso_config_alignment as alignment

        spec = yaml.safe_load(Path(resolve_local_file(alignment.SPEC)).read_text())
        cfg = alignment.profile_config(
            alignment.WIDEBAND_PROFILE, teacher, dataset, spec)
        x, y, _subjects, _trials, _input_manifest = alignment.load_fold_data(
            alignment.WIDEBAND_PROFILE, dataset, teacher, alignment.WIDEBAND_VARIANT)
        identity = alignment.wideband_stream_identity(
            alignment.WIDEBAND_PROFILE, dataset, teacher, seed,
            alignment.WIDEBAND_VARIANT, cfg, list(x.shape[1:]),
            int(len(np.unique(y))))
        completion = alignment.completed_fold(
            manifest_path.parent, {**identity, 'subject': fold + 1}, int(cfg['epochs']))
        if completion is None:
            raise ValueError(
                f'{manifest_path.parent}: completed teacher lacks its validated completion identity')
        if not isinstance(resolved_config, dict):
            raise ValueError(f'{manifest_path}: current resolved training config is invalid')
    return manifest_path, manifest, checkpoint, actual_sha


def _valid_alignment_resume(alignment: Any, output: Path,
                            identity: dict[str, Any], epochs: int) -> bool:
    # A broken completion record must be archived and regenerated. A resume
    # checkpoint cannot substitute for the required completed-fold identity.
    if (output / 'completed_state.pt').exists():
        return False
    result_file = output / 'result.json'
    if result_file.is_file():
        try:
            if alignment.read_json(result_file).get('status') == 'complete':
                return False
        except Exception:
            return False
    checkpoint = output / 'training_state.pt'
    if not checkpoint.is_file():
        return False
    try:
        saved = torch.load(resolve_local_file(checkpoint), map_location='cpu', weights_only=False)
        alignment.validate_wideband_checkpoint(saved, identity, checkpoint)
        next_epoch = int(saved.get('next_epoch', -1))
        if saved.get('completed') is True or not 0 <= next_epoch < epochs:
            return False
        if not isinstance(saved.get('rng_state'), dict):
            return False
        return all(key in saved for key in ('model_state', 'optimizer_state', 'next_epoch'))
    except Exception:
        return False


def _prepare_alignment(dataset: str, teacher: str, seed: int, fold: int,
                       device: torch.device) -> None:
    from experiments.finetune import run_loso_config_alignment as alignment

    spec = yaml.safe_load(Path(resolve_local_file(alignment.SPEC)).read_text())
    cfg = alignment.profile_config(alignment.WIDEBAND_PROFILE, teacher, dataset, spec)
    x, y, _subjects, _trials, _manifest = alignment.load_fold_data(
        alignment.WIDEBAND_PROFILE, dataset, teacher, alignment.WIDEBAND_VARIANT)
    identity = alignment.wideband_stream_identity(
        alignment.WIDEBAND_PROFILE, dataset, teacher, seed,
        alignment.WIDEBAND_VARIANT, cfg, list(x.shape[1:]), int(len(np.unique(y))))
    epochs = int(cfg['epochs'])
    n_subjects = int(spec['datasets'][dataset]['subjects'])
    root = alignment.RESULT_ROOT / alignment.WIDEBAND_PROFILE / dataset / teacher / f'seed_{seed}'
    affected_from: int | None = None
    reason = ''
    reusable_prefix = 0
    for subject in range(n_subjects):
        output = alignment.result_path(root, subject)
        fold_identity = {**identity, 'subject': subject + 1}
        try:
            completion = alignment.completed_fold(output, fold_identity, epochs)
        except Exception as error:
            if _valid_alignment_resume(alignment, output, fold_identity, epochs):
                affected_from = subject + 1
                reason = (f'S{subject + 1:02d} is a valid resume checkpoint; '
                          'later fold outputs are archived to preserve the ordered RNG stream')
            else:
                affected_from = subject
                reason = (f'S{subject + 1:02d} completion identity/result is invalid or missing: '
                          f'{type(error).__name__}: {error}')
            break
        if completion is not None:
            reusable_prefix += 1
            continue
        if _valid_alignment_resume(alignment, output, fold_identity, epochs):
            affected_from = subject + 1
            reason = (f'S{subject + 1:02d} is an identity-matched resume checkpoint; '
                      'later fold outputs are archived to preserve the ordered RNG stream')
        else:
            affected_from = subject
            reason = f'S{subject + 1:02d} has no validated completion or resumable training state'
        break

    valid_resume_at_request = (
        affected_from == fold + 1
        and (reason.startswith(f'S{fold + 1:02d} is a valid resume checkpoint')
             or reason.startswith(f'S{fold + 1:02d} is an identity-matched resume checkpoint'))
    )
    if affected_from is None or (affected_from > fold and not valid_resume_at_request):
        affected_from = fold
        reason = (f'requested S{fold + 1:02d} baseline failed strict teacher validation '
                  'although the ordered fold records appear complete')
    if affected_from <= fold:
        archived = 0
        for subject in range(affected_from, n_subjects):
            output = alignment.result_path(root, subject)
            if output.exists():
                if _archive(output, dataset, teacher, seed, reason, subject):
                    archived += 1
        summary = root / 'seed_summary.json'
        if summary.exists():
            _archive(summary, dataset, teacher, seed,
                     f'stale seed summary before ordered stream rebuild: {reason}')
        print(f'[teacher-prepare] 001 ordered stream {dataset}/{teacher}/seed_{seed}: '
              f'reusable completed prefix={min(reusable_prefix, affected_from)}; '
              f'archive directories={archived}; requested through S{fold+1:02d}; '
              f'epochs/fold={epochs}; output={root}', flush=True)
    else:
        print(f'[teacher-prepare] 001 ordered stream {dataset}/{teacher}/seed_{seed}: '
              f'reusable completed prefix={reusable_prefix}; valid resume at '
              f'S{affected_from:02d}; archive later folds before continuing; '
              f'epochs/fold={epochs}; output={root}', flush=True)
        for subject in range(affected_from, n_subjects):
            output = alignment.result_path(root, subject)
            if output.exists():
                _archive(output, dataset, teacher, seed, reason, subject)
        summary = root / 'seed_summary.json'
        if summary.exists():
            _archive(summary, dataset, teacher, seed,
                     f'stale seed summary before ordered stream continuation: {reason}')

    args = SimpleNamespace(
        profile=alignment.WIDEBAND_PROFILE, dataset=dataset, model=teacher,
        seed=seed, epochs_override=None, preflight_steps=None,
    )
    started = time.monotonic()
    alignment.run_worker(args, spec, device, stop_after_subject=fold)
    print(f'[teacher-prepare] 001 ordered stream request finished in '
          f'{time.monotonic() - started:.1f}s; output={root}', flush=True)


def _output_lock_path(output: Path, fold: int) -> Path:
    return output.parent / f'.subject_{fold + 1:02d}.teacher_prepare.lock'


def _prepare_refresh(dataset: str, teacher: str, seed: int, fold: int,
                     device: torch.device, teacher_data: Any) -> None:
    from experiments.finetune import run_loso_source_refresh_004_5001 as refresh

    x, y, subjects, trials, input_manifest = refresh.load_input(dataset, teacher)
    paths = refresh.input_paths(dataset, teacher)
    file_hashes = {
        name: {'path': str(path), 'sha256': _sha256(path), 'bytes': path.stat().st_size}
        for name, path in paths.items() if name != 'manifest.json'
    }
    manifest_path = paths['manifest.json']
    file_hashes['manifest.json'] = {
        'path': str(manifest_path), 'sha256': _sha256(manifest_path),
        'bytes': manifest_path.stat().st_size,
    }
    cfg = refresh.model_config(teacher, dataset, tuple(x.shape[1:]))
    output = refresh.fold_output(dataset, teacher, seed, fold)
    expected = refresh.make_fold_manifest(
        dataset, teacher, seed, fold, cfg, input_manifest, file_hashes,
        None, None, int(np.sum(subjects != fold)), int(np.sum(subjects == fold)), device)
    weight_path = None
    weight_hash = None
    if teacher in {'mirepnet', 'cbramod'}:
        weight_path = Path(refresh.config.weight_path(teacher)).resolve()
        weight_hash = _sha256(weight_path)
    expected['pretrained_checkpoint'] = str(weight_path) if weight_path else None
    expected['pretrained_sha256'] = weight_hash

    with _file_lock(_output_lock_path(output, fold)):
        try:
            _validate_standard_checkpoint(dataset, teacher, seed, fold,
                                          teacher_data)
            print(f'[teacher-prepare] reuse verified {dataset}/{teacher}/seed_{seed}/'
                  f'S{fold+1:02d}', flush=True)
            return
        except Exception as error:
            reason = f'{type(error).__name__}: {error}'
        waited = _assert_no_active_worker('run_loso_source_refresh_004_5001.py',
                                          dataset, teacher, seed)
        if waited:
            try:
                _validate_standard_checkpoint(
                    dataset, teacher, seed, fold, teacher_data)
                print(f'[teacher-prepare] reuse verified {dataset}/{teacher}/seed_{seed}/'
                      f'S{fold+1:02d} after active worker exit', flush=True)
                return
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'

        manifest_path = output / 'manifest.json'
        resume_valid = False
        if manifest_path.is_file() and (output / 'training_state.pt').is_file():
            try:
                old_manifest = _read_json(manifest_path)
                saved = torch.load(resolve_local_file(output / 'training_state.pt'),
                                   map_location='cpu', weights_only=False)
                saved_identity = dict(saved.get('identity', {}))
                expected_identity = dict(expected)
                saved_cfg = saved_identity.pop('model_config', {})
                expected_cfg = expected_identity.pop('model_config')
                resume_valid = (
                    old_manifest.get('status') == 'running'
                    and saved_identity == expected_identity
                    and refresh.config.equivalent_loso_model_config(
                        teacher, saved_cfg, expected_cfg)
                    and all(key in saved for key in (
                        'model_state', 'optimizer_state', 'scheduler_state',
                        'rng_state', 'next_epoch'))
                    and 0 <= int(saved['next_epoch']) < int(cfg['epochs'])
                )
            except Exception:
                resume_valid = False

        if output.exists() and not resume_valid:
            _archive(output, dataset, teacher, seed,
                     f'non-reusable 004/5001 result before same-fold training: {reason}', fold)
        elif resume_valid:
            print(f'[teacher-prepare] resume identity-matched {dataset}/{teacher}/'
                  f'seed_{seed}/S{fold+1:02d}; epochs={cfg["epochs"]}; output={output}', flush=True)
        else:
            print(f'[teacher-prepare] train {dataset}/{teacher}/seed_{seed}/'
                  f'S{fold+1:02d}; epochs={cfg["epochs"]}; output={output}; '
                  f'previous artifact={reason}', flush=True)

        started = time.monotonic()
        with _selected_device_seed_scope(device):
            if device.type == 'cpu':
                with _cpu_fold_rng_compat(refresh):
                    refresh.train_fold(dataset, teacher, seed, fold, x, y, subjects,
                                       trials, input_manifest, file_hashes, cfg, device)
            else:
                refresh.train_fold(dataset, teacher, seed, fold, x, y, subjects,
                                   trials, input_manifest, file_hashes, cfg, device)
        _validate_standard_checkpoint(
            dataset, teacher, seed, fold, teacher_data)
        print(f'[teacher-prepare] 004/5001 fold finished in '
              f'{time.monotonic() - started:.1f}s; output={output}', flush=True)


def _alexmi_cell_paths(teacher: str, seed: int, fold: int,
                       spec: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    from experiments.finetune import run_loso_five_datasets as protocol

    recipe = spec['main_recipes'][teacher]['name']
    output_dir = protocol.RESULTS_ROOT / 'AlexMI' / teacher / recipe
    cell = output_dir / f'subject_{fold + 1:02d}' / f'seed_{seed}'
    return (cell, cell / 'result.npz', cell / 'model.pt',
            cell / 'manifest.json', cell / 'train_history.csv')


def _validate_alexmi_checkpoint(teacher: str, seed: int, fold: int,
                                teacher_data: Any, spec: dict[str, Any],
                                snapshot: dict[str, Any]):
    from data import loso_alexmi
    from experiments.finetune import run_loso_five_datasets as protocol

    manifest_path, manifest = loso_alexmi._find_baseline_manifest(teacher, seed, fold)
    manifest_path = Path(resolve_local_file(manifest_path))
    if manifest.get('status') not in (None, 'complete'):
        raise ValueError(f'{manifest_path}: AlexMI teacher result is not complete')
    expected_cfg = protocol._config(
        teacher, 'AlexMI', 'main', spec['datasets']['AlexMI'], spec)
    if manifest.get('model_config') != expected_cfg:
        raise ValueError(f'{manifest_path}: AlexMI teacher model config changed')
    if manifest.get('resolved_config_sha256') != protocol._json_hash(expected_cfg):
        raise ValueError(f'{manifest_path}: AlexMI resolved config identity changed')
    expected_recipe = spec['main_recipes'][teacher]['name']
    if manifest.get('recipe') != expected_recipe:
        raise ValueError(f'{manifest_path}: AlexMI recipe differs from the current fixed main recipe')
    # This strict reader verifies source files, fold UIDs, labels, and result SHA.
    loso_alexmi.load_baseline_prediction(teacher, seed, fold, 'test', teacher_data)
    checkpoint = manifest_path.parent / 'model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}: AlexMI same-fold teacher checkpoint is missing')
    checkpoint_sha = _sha256(checkpoint)

    cache_manifest = external_path(
        loso_alexmi.ALEXMI_TEACHER_TARGET_ROOT / teacher / f'subject_{fold + 1:02d}'
        / f'seed_{seed}' / 'cache_manifest.json')
    if cache_manifest.is_file():
        try:
            cache_meta = _read_json(cache_manifest)
            record = (cache_meta.get('teacher_checkpoint', {})
                      if isinstance(cache_meta, dict) else {})
            if not isinstance(record, dict):
                raise ValueError(f'{cache_manifest}: teacher checkpoint reference is malformed')
            recorded_path = Path(resolve_local_file(record.get('path', ''))).resolve()
            canonical_checkpoint = Path(resolve_local_file(checkpoint)).resolve()
            if (recorded_path != canonical_checkpoint
                    or str(record.get('sha256', '')) != checkpoint_sha):
                raise ValueError(f'{cache_manifest}: cached teacher checkpoint identity differs')
        except Exception as error:
            # An invalid cache manifest does not prove the teacher is bad. The
            # model is loaded strictly against the current official adapter below.
            print(f'[teacher-prepare] AlexMI cache checkpoint reference unavailable: {error}',
                  flush=True)

    cfg = dict(expected_cfg)
    adapter = protocol._make_adapter(teacher, cfg, 'cpu', protocol.PROTOCOL,
                                     'AlexMI', 'main')
    model = adapter.build(len(spec['datasets']['AlexMI']['classes']))
    state = torch.load(resolve_local_file(checkpoint), map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return manifest_path, manifest, checkpoint, checkpoint_sha


def _prepare_alexmi(teacher: str, seed: int, fold: int,
                    device: torch.device, teacher_data: Any) -> None:
    from data import loso_alexmi
    from experiments.finetune import run_loso_five_datasets as protocol

    spec, snapshot = load_verified_alexmi_spec(protocol)
    cell, result_path, checkpoint_path, manifest_path, history_path = _alexmi_cell_paths(
        teacher, seed, fold, spec)
    model_root = external_path(loso_alexmi.ALEXMI_RESULT_ROOT / teacher)
    # The historical runner also updates one seed-level split manifest and
    # summary, so serialize automatic single-fold requests within a seed.
    lock_path = model_root / f'.seed_{seed}.teacher_prepare.lock'
    with _file_lock(lock_path):
        try:
            _validate_alexmi_checkpoint(teacher, seed, fold, teacher_data, spec, snapshot)
            print(f'[teacher-prepare] reuse verified AlexMI/{teacher}/seed_{seed}/'
                  f'S{fold+1:02d}', flush=True)
            return
        except Exception as error:
            reason = f'{type(error).__name__}: {error}'
        waited = _assert_no_active_worker('run_loso_five_datasets.py', 'AlexMI', teacher, seed)
        if waited:
            try:
                _validate_alexmi_checkpoint(teacher, seed, fold, teacher_data, spec, snapshot)
                print(f'[teacher-prepare] reuse verified AlexMI/{teacher}/seed_{seed}/'
                      f'S{fold+1:02d} after active worker exit', flush=True)
                return
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
        if cell.exists():
            _archive(cell, 'AlexMI', teacher, seed,
                     f'non-reusable AlexMI result before same-fold training: {reason}', fold)
        print(f'[teacher-prepare] train AlexMI/{teacher}/seed_{seed}/S{fold+1:02d}; '
              f'epochs={protocol._config(teacher, "AlexMI", "main", spec["datasets"]["AlexMI"], spec)["epochs"]}; '
              f'output={cell}; previous artifact={reason}', flush=True)
        args = SimpleNamespace(
            model=teacher, datasets=['AlexMI'], seeds=[seed], folds=[fold],
            recipe='main', gpu=(device.index if device.type == 'cuda' else None),
            force=False, no_reuse_004=False,
        )
        started = time.monotonic()
        device_name = str(device)
        with _selected_device_seed_scope(device):
            protocol._run(args, spec, snapshot, device_name)
        _validate_alexmi_checkpoint(teacher, seed, fold, teacher_data, spec, snapshot)
        print(f'[teacher-prepare] AlexMI fold finished in '
              f'{time.monotonic() - started:.1f}s; output={cell}', flush=True)


def ensure_teacher_checkpoint(dataset: str, teacher: str, seed: int, fold: int,
                              teacher_data: Any, device_name: str) -> None:
    """Ensure a strictly identified same-fold teacher checkpoint exists.

    ``seed`` and ``fold`` use the student/cache request exactly: seeds are
    non-negative integers and folds are zero-based. A valid checkpoint is a
    no-op. Missing, corrupt, or incompatible completed artifacts are preserved
    under external ``migrations/`` before the existing baseline recipe writes
    its canonical result path.
    """
    dataset, teacher = str(dataset), str(teacher).lower()
    seed, fold = int(seed), int(fold)
    if dataset not in _SUPPORTED_DATASETS or teacher not in _SUPPORTED_TEACHERS:
        raise ValueError(f'unsupported LOSO teacher request: {dataset}/{teacher}')
    if seed < 0:
        raise ValueError(f'seed must be non-negative, got {seed}')
    if dataset == 'AlexMI':
        n_subjects = 8
    elif dataset == 'BNCI2015001':
        n_subjects = 12
    else:
        n_subjects = 9
    if not 0 <= fold < n_subjects:
        raise ValueError(f'{dataset}: fold must be zero-based in [0,{n_subjects - 1}]')
    device = _device(device_name)
    print(f'[teacher-prepare] request dataset={dataset} teacher={teacher} '
          f'seed={seed} fold={fold} device={device}', flush=True)

    if dataset in {'BNCI2014001', 'BNCI2014001-4'}:
        from experiments.finetune import run_loso_config_alignment as alignment

        root = alignment.RESULT_ROOT / alignment.WIDEBAND_PROFILE / dataset / teacher / f'seed_{seed}'
        lock = root / 'worker.lock'
        with _file_lock(lock):
            try:
                _validate_standard_checkpoint(dataset, teacher, seed, fold, teacher_data)
                print(f'[teacher-prepare] reuse verified {dataset}/{teacher}/seed_{seed}/'
                      f'S{fold+1:02d}', flush=True)
                return
            except Exception as error:
                print(f'[teacher-prepare] rebuild required for {dataset}/{teacher}/'
                      f'seed_{seed}/S{fold+1:02d}: {type(error).__name__}: {error}',
                      flush=True)
            _prepare_alignment(dataset, teacher, seed, fold, device)
    elif dataset in {'BNCI2014004', 'BNCI2015001'}:
        _prepare_refresh(dataset, teacher, seed, fold, device, teacher_data)
    else:
        _prepare_alexmi(teacher, seed, fold, device, teacher_data)

    # The public contract is a side-effecting ensure operation; retain the
    # exporter's existing strict loader as the final identity gate.
    if dataset != 'AlexMI':
        _validate_standard_checkpoint(dataset, teacher, seed, fold, teacher_data)
