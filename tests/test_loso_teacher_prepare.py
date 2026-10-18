"""Focused, CPU-only checks for transparent LOSO teacher preparation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest
import torch

from data import loso_alexmi
from experiments.distill import loso_teacher_prepare as prepare
from experiments.finetune import run_loso_config_alignment as alignment


@contextmanager
def _unlocked(_path):
    yield


def _stub_eeg_dataset_import(monkeypatch):
    """Avoid importing the optional MNE stack for these mocked recipe tests."""
    stub = types.ModuleType('data.eeg_dataset')
    stub.EEGDataset = object
    stub._DATA_ROOT = Path('/unused')
    monkeypatch.setitem(sys.modules, 'data.eeg_dataset', stub)


def test_wideband_through_fold_restores_terminal_rng_after_completed_prefix(
        monkeypatch, tmp_path, capsys):
    """A request through S08 restores S01..S07 and never resets S08 alone."""
    device = torch.device('cpu')
    seed = 666
    prefix_count = 7
    cfg = {'epochs': 20, 'dropout': 0.1}
    x = np.zeros((18, 2), dtype=np.float32)
    y = np.asarray([0, 1] * 9, dtype=np.int64)
    subjects = np.repeat(np.arange(9), 2)
    trials = object()
    manifest = {'fixture': True}
    loads = []

    def fake_load(*args):
        loads.append(args)
        return x, y, subjects, trials, manifest

    def fake_model(*_args, **_kwargs):
        # Use a tiny real torch module so construction consumes the same CPU
        # RNG stream as the formal worker's throwaway model build.
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(alignment, 'profile_config', lambda *_args: cfg)
    monkeypatch.setattr(alignment, 'load_fold_data', fake_load)
    monkeypatch.setattr(alignment, 'make_model', fake_model)
    monkeypatch.setattr(alignment, 'wideband_stream_identity',
                        lambda *_args: {'fixture_stream_identity': True})
    monkeypatch.setattr(alignment, 'RESULT_ROOT', tmp_path / 'results')

    # Generate the seven terminal RNG snapshots the mocked completed folds
    # would have left in their compact completed_state.pt records.
    alignment.set_seed(seed, device)
    throwaway = fake_model()
    throwaway.to(device)
    del throwaway
    completed = {}
    for subject in range(prefix_count):
        random.random()
        np.random.random()
        torch.rand(5)
        completed[subject] = (
            {'status': 'complete', 'held_out_subject': subject + 1,
             'metrics': {'accuracy': 0.75, 'balanced_accuracy': 0.7,
                         'kappa': 0.5, 'macro_f1': 0.72}},
            {'rng_state': alignment.capture_rng(device)},
        )
    expected_draws = (random.random(), np.random.random(), torch.rand(5).tolist())

    scanned = []

    def fake_completed(output, identity, epochs):
        subject = identity['subject'] - 1
        scanned.append(subject)
        assert epochs == cfg['epochs']
        return completed.get(subject)

    monkeypatch.setattr(alignment, 'completed_fold', fake_completed)
    observed = {}

    def fake_train_one_fold(profile, dataset, model, got_seed, subject, *args, **kwargs):
        observed['fold'] = subject
        observed['seed'] = got_seed
        observed['draws'] = (random.random(), np.random.random(), torch.rand(5).tolist())
        assert kwargs['stream_resume_rng'] is True
        return {
            'status': 'complete', 'held_out_subject': subject + 1,
            'wall_seconds': 1.0,
            'metrics': {'accuracy': 0.8, 'balanced_accuracy': 0.75,
                        'kappa': 0.6, 'macro_f1': 0.77},
        }

    monkeypatch.setattr(alignment, 'train_one_fold', fake_train_one_fold)
    args = SimpleNamespace(
        profile=alignment.WIDEBAND_PROFILE, dataset='BNCI2014001',
        model='cbramod', seed=seed, epochs_override=None, preflight_steps=None,
    )
    spec = {'datasets': {'BNCI2014001': {'subjects': 9}}}
    (tmp_path / 'results' / alignment.WIDEBAND_PROFILE / 'BNCI2014001' /
     'cbramod' / f'seed_{seed}').mkdir(parents=True)

    alignment.run_worker(args, spec, device, stop_after_subject=7)

    assert observed['fold'] == 7
    assert observed['seed'] == seed
    assert observed['draws'] == expected_draws
    assert scanned == list(range(8))
    summary = json.loads((tmp_path / 'results' / alignment.WIDEBAND_PROFILE /
                          'BNCI2014001' / 'cbramod' / f'seed_{seed}' /
                          'seed_summary.json').read_text())
    assert summary['folds_complete'] == 8
    assert summary['folds_expected'] == 9
    assert summary['status'] == 'partial'
    output = capsys.readouterr().out
    assert 'requested through S08' in output
    assert 'epochs/fold=20' in output
    assert 'epochs to train=20' in output


def test_valid_001_teacher_reuses_current_checkpoint_without_preparing(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, '_device', lambda _value: torch.device('cpu'))
    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(alignment, 'RESULT_ROOT', tmp_path / 'results')
    verified = []
    monkeypatch.setattr(
        prepare, '_validate_standard_checkpoint',
        lambda *args: verified.append(args) or (Path('result.json'), {},
                                                  Path('final_model.pt'), 'sha'))
    monkeypatch.setattr(
        prepare, '_prepare_alignment',
        lambda *_args: pytest.fail('a valid same-fold teacher must be a no-op'))

    prepare.ensure_teacher_checkpoint(
        'BNCI2014001', 'cbramod', 666, 0, object(), 'cpu')

    assert len(verified) == 1
    assert verified[0][:4] == ('BNCI2014001', 'cbramod', 666, 0)


def test_missing_001_seed_prepares_requested_fold_without_touching_seed_zero(
        monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, '_device', lambda _value: torch.device('cpu'))
    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(alignment, 'RESULT_ROOT', tmp_path / 'results')
    old_seed_marker = (alignment.RESULT_ROOT / alignment.WIDEBAND_PROFILE /
                       'BNCI2014001' / 'cbramod' / 'seed_0' / 'keep.txt')
    old_seed_marker.parent.mkdir(parents=True)
    old_seed_marker.write_text('preserve old seed')
    validations = []

    def validate(*args):
        validations.append(args)
        if len(validations) == 1:
            raise FileNotFoundError('requested seed/fold has no baseline checkpoint')
        return Path('result.json'), {}, Path('final_model.pt'), 'new-sha'

    monkeypatch.setattr(prepare, '_validate_standard_checkpoint', validate)
    prepared = []
    monkeypatch.setattr(
        prepare, '_prepare_alignment',
        lambda *args: prepared.append(args))

    prepare.ensure_teacher_checkpoint(
        'BNCI2014001', 'cbramod', 666, 7, object(), 'cpu')

    assert len(validations) == 2
    assert prepared == [('BNCI2014001', 'cbramod', 666, 7, torch.device('cpu'))]
    assert old_seed_marker.read_text() == 'preserve old seed'


def test_001_invalid_completion_archives_suffix_and_keeps_shared_worker_lock(
        monkeypatch, tmp_path):
    monkeypatch.setattr(prepare, 'external_path', lambda path: Path(path))
    monkeypatch.setattr(prepare, 'require_external_output', lambda path: Path(path))
    monkeypatch.setattr(prepare, 'MIGRATIONS_ROOT', tmp_path / 'migrations')
    monkeypatch.setattr(alignment, 'RESULT_ROOT', tmp_path / 'results')
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text('datasets:\n  BNCI2014001:\n    subjects: 9\n')
    monkeypatch.setattr(alignment, 'SPEC', spec_path)
    monkeypatch.setattr(alignment, 'profile_config',
                        lambda *_args: {'epochs': 20, 'dropout': 0.1})
    monkeypatch.setattr(
        alignment, 'load_fold_data',
        lambda *_args: (np.zeros((18, 2), dtype=np.float32),
                        np.asarray([0, 1] * 9), np.repeat(np.arange(9), 2),
                        object(), {'fixture': True}))
    monkeypatch.setattr(alignment, 'wideband_stream_identity',
                        lambda *_args: {'fixture': True})
    root = (alignment.RESULT_ROOT / alignment.WIDEBAND_PROFILE / 'BNCI2014001'
            / 'cbramod' / 'seed_666')
    root.mkdir(parents=True)
    worker_lock = root / 'worker.lock'
    worker_lock.write_text('same lock inode')
    lock_inode = worker_lock.stat().st_ino
    for subject in range(9):
        output = alignment.result_path(root, subject)
        output.mkdir()
        for name in ('completed_state.pt', 'result.json', 'final_model.pt'):
            (output / name).write_text(f'old S{subject + 1:02d} {name}')
    summary = root / 'seed_summary.json'
    summary.write_text('old full-seed summary')
    old_seed_marker = (alignment.RESULT_ROOT / alignment.WIDEBAND_PROFILE /
                       'BNCI2014001' / 'cbramod' / 'seed_0' / 'keep.txt')
    old_seed_marker.parent.mkdir(parents=True)
    old_seed_marker.write_text('preserve old seed')

    def completion(output, identity, epochs):
        if identity['subject'] == 1:
            return ({'status': 'complete', 'metrics': {}},
                    {'rng_state': {'python': None}})
        if identity['subject'] == 2:
            raise RuntimeError('completed result lacks its completion identity')
        return None

    monkeypatch.setattr(alignment, 'completed_fold', completion)
    monkeypatch.setattr(prepare, '_valid_alignment_resume', lambda *_args: False)
    calls = []
    monkeypatch.setattr(
        alignment, 'run_worker',
        lambda args, got_spec, device, stop_after_subject=None:
        calls.append((args, got_spec, device, stop_after_subject)))

    prepare._prepare_alignment('BNCI2014001', 'cbramod', 666, 3,
                               torch.device('cpu'))

    assert worker_lock.stat().st_ino == lock_inode
    assert worker_lock.read_text() == 'same lock inode'
    assert alignment.result_path(root, 0).is_dir()
    assert not any(alignment.result_path(root, subject).exists()
                   for subject in range(1, 9))
    archived_folds = list((tmp_path / 'migrations').rglob('completed_state.pt'))
    assert len(archived_folds) == 8
    assert all(path.with_name('result.json').is_file()
               and path.with_name('final_model.pt').is_file() for path in archived_folds)
    assert not summary.exists()
    assert list((tmp_path / 'migrations').rglob('seed_summary.json.*'))
    assert old_seed_marker.read_text() == 'preserve old seed'
    assert calls and calls[0][0].seed == 666 and calls[0][3] == 3


def test_refresh_waits_for_running_worker_then_reuses_completed_result(monkeypatch, tmp_path):
    _stub_eeg_dataset_import(monkeypatch)
    from experiments.finetune import run_loso_source_refresh_004_5001 as refresh

    files = {}
    for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv', 'manifest.json'):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1])
    subjects = np.asarray([0, 0, 1, 1])
    trial_table = SimpleNamespace()
    input_manifest = {'fixture': True}
    output = tmp_path / 'results' / 'fold'
    calls = []

    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(refresh, 'load_input',
                        lambda *_args: (x, y, subjects, trial_table, input_manifest))
    monkeypatch.setattr(refresh, 'input_paths', lambda *_args: files)
    monkeypatch.setattr(refresh, 'model_config', lambda *_args: {'epochs': 20})
    monkeypatch.setattr(refresh, 'fold_output', lambda *_args: output)
    monkeypatch.setattr(refresh, 'make_fold_manifest', lambda *_args: {'identity': True})
    monkeypatch.setattr(refresh.config, 'weight_path', lambda _teacher: tmp_path / 'weight.pt')
    (tmp_path / 'weight.pt').write_bytes(b'fixture-weight')
    teacher_data = object()

    def validate(*_args):
        calls.append('validate')
        if len(calls) == 1:
            raise ValueError('baseline is currently incomplete')
        return Path('manifest.json'), {}, Path('final_model.pt'), 'sha'

    monkeypatch.setattr(prepare, '_validate_standard_checkpoint', validate)
    monkeypatch.setattr(prepare, '_assert_no_active_worker',
                        lambda *_args: calls.append('waited') or True)
    monkeypatch.setattr(prepare, '_archive',
                        lambda *_args: pytest.fail('completed output must be reused'))
    monkeypatch.setattr(refresh, 'train_fold',
                        lambda *_args, **_kwargs: pytest.fail('must not retrain'))

    prepare._prepare_refresh('BNCI2014004', 'cbramod', 7, 2,
                             torch.device('cpu'), teacher_data)

    assert calls == ['validate', 'waited', 'validate']


def test_missing_refresh_seed_calls_existing_fold_api_with_same_key_and_device(
        monkeypatch, tmp_path):
    _stub_eeg_dataset_import(monkeypatch)
    from experiments.finetune import run_loso_source_refresh_004_5001 as refresh

    paths = {}
    for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv', 'manifest.json'):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    (tmp_path / 'weight.pt').write_bytes(b'fixture-weight')
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1])
    subjects = np.asarray([0, 0, 1, 1])
    input_manifest = {'fixture': True}
    output = tmp_path / 'results' / 'fold'
    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(refresh, 'load_input',
                        lambda *_args: (x, y, subjects, object(), input_manifest))
    monkeypatch.setattr(refresh, 'input_paths', lambda *_args: paths)
    monkeypatch.setattr(refresh, 'model_config', lambda *_args: {'epochs': 20})
    monkeypatch.setattr(refresh, 'fold_output', lambda *_args: output)
    monkeypatch.setattr(refresh, 'make_fold_manifest', lambda *_args: {'identity': True})
    monkeypatch.setattr(refresh.config, 'weight_path', lambda _teacher: tmp_path / 'weight.pt')
    teacher_data = object()
    validations = []

    def validate(*_args):
        validations.append(True)
        if len(validations) == 1:
            raise FileNotFoundError('requested refresh seed/fold is missing')
        return Path('manifest.json'), {}, Path('final_model.pt'), 'sha'

    monkeypatch.setattr(prepare, '_validate_standard_checkpoint', validate)
    monkeypatch.setattr(prepare, '_assert_no_active_worker', lambda *_args: False)
    train_calls = []
    monkeypatch.setattr(
        refresh, 'train_fold',
        lambda dataset, teacher, seed, fold, *args:
        train_calls.append((dataset, teacher, seed, fold, args[-1])))

    prepare._prepare_refresh('BNCI2014004', 'cbramod', 777, 4,
                             torch.device('cpu'), teacher_data)

    assert len(validations) == 2
    assert train_calls == [('BNCI2014004', 'cbramod', 777, 4, torch.device('cpu'))]


def test_process_check_waits_and_returns_after_worker_exits(monkeypatch):
    responses = iter([
        SimpleNamespace(stdout=(
            '20 python experiments/finetune/run_loso_source_refresh_004_5001.py '
            '--dataset BNCI2014004 --model cbramod --seed 7\n')),
        SimpleNamespace(stdout=''),
    ])
    sleeps = []
    monkeypatch.setattr(prepare.os, 'getpid', lambda: 10)
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(prepare.time, 'sleep', sleeps.append)

    waited = prepare._assert_no_active_worker(
        'run_loso_source_refresh_004_5001.py', 'BNCI2014004', 'cbramod', 7)

    assert waited is True
    assert sleeps == [5]


def test_alexmi_process_check_applies_legacy_default_dataset_and_seeds(monkeypatch):
    responses = iter([
        SimpleNamespace(stdout=(
            '20 python -m experiments.finetune.run_loso_five_datasets '
            '--model cbramod --gpu 1\n')),
        SimpleNamespace(stdout=''),
    ])
    sleeps = []
    monkeypatch.setattr(prepare.os, 'getpid', lambda: 10)
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(prepare.time, 'sleep', sleeps.append)

    waited = prepare._assert_no_active_worker(
        'run_loso_five_datasets.py', 'AlexMI', 'cbramod', 666)

    assert waited is True
    assert sleeps == [5]


def test_verified_alexmi_spec_restores_only_path_relocation(monkeypatch, tmp_path):
    original = (
        b'main_recipes: {}\n'
        b'datasets: {}\n'
        b'outputs:\n'
        b'  first: results/reproductions/one\n'
        b'  second: results/reproductions/two\n'
        b'  third: results/reproductions/three\n'
    )
    migrated_prefix = str(prepare.RESULTS_ROOT / 'reproductions').encode() + b'/'
    migrated = original.replace(b'results/reproductions/', migrated_prefix)
    assert migrated != original
    assert migrated.count(migrated_prefix) == 3
    spec_path = tmp_path / 'archived.yaml'
    spec_path.write_bytes(migrated)
    snapshot_path = tmp_path / 'snapshot.json'
    snapshot_path.write_text(json.dumps({
        'spec_sha256': hashlib.sha256(original).hexdigest(),
    }))
    paths = {'fixture-spec': spec_path, 'fixture-snapshot': snapshot_path}
    monkeypatch.setattr(prepare, 'resolve_local_file', lambda path: paths[str(path)])
    protocol = SimpleNamespace(SPEC_PATH='fixture-spec', SNAPSHOT_PATH='fixture-snapshot')

    spec, snapshot = prepare.load_verified_alexmi_spec(protocol)

    assert spec['outputs'] == {
        'first': 'results/reproductions/one',
        'second': 'results/reproductions/two',
        'third': 'results/reproductions/three',
    }
    assert snapshot['spec_sha256'] == hashlib.sha256(original).hexdigest()

    spec_path.write_bytes(migrated.replace(b'three', b'changed'))
    with pytest.raises(RuntimeError, match='cannot be restored'):
        prepare.load_verified_alexmi_spec(protocol)


def test_alexmi_legacy_manifest_without_status_or_cache_manifest_is_reused(
        monkeypatch, tmp_path, capsys):
    _stub_eeg_dataset_import(monkeypatch)
    from experiments.finetune import run_loso_five_datasets as protocol

    teacher, seed, fold = 'cbramod', 7, 0
    model_config = {'epochs': 20, 'input_channels': 2}
    spec = {
        'main_recipes': {teacher: {'name': 'canonical4s_v1'}},
        'datasets': {'AlexMI': {'classes': ['feet', 'right_hand']}},
    }
    result_root = tmp_path / 'alexmi_results'
    cell = (result_root / teacher / spec['main_recipes'][teacher]['name'] /
            'subject_01' / f'seed_{seed}')
    cell.mkdir(parents=True)
    checkpoint = cell / 'model.pt'
    adapter_model = torch.nn.Linear(2, 2)
    torch.save(adapter_model.state_dict(), checkpoint)
    manifest = {
        'dataset': 'AlexMI', 'model': teacher, 'seed': seed,
        'test_subject': fold + 1,
        'model_config': model_config,
        'resolved_config_sha256': protocol._json_hash(model_config),
        'recipe': spec['main_recipes'][teacher]['name'],
        # Historical canonical manifests have no status key.
    }
    manifest.update({
        'protocol': loso_alexmi._PROTOCOL,
        'dataset_source': 'AlexMI',
        'label_values': loso_alexmi._LABELS,
        'num_classes': 2,
        'source_trial_selection': loso_alexmi._LOADER_RECIPE['source_trial_selection'],
        'window': loso_alexmi._LOADER_RECIPE['window'],
    })
    manifest_path = cell / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(prepare, 'load_verified_alexmi_spec',
                        lambda _protocol: (spec, {}))
    monkeypatch.setattr(loso_alexmi, 'ALEXMI_RESULT_ROOT', result_root)
    monkeypatch.setattr(loso_alexmi, 'external_path', lambda path: Path(path))
    monkeypatch.setattr(protocol, '_config', lambda *_args: model_config)
    monkeypatch.setattr(
        protocol, '_make_adapter',
        lambda *_args: SimpleNamespace(build=lambda _classes: torch.nn.Linear(2, 2)))
    monkeypatch.setattr(loso_alexmi, 'load_baseline_prediction',
                        lambda *_args: {'logits': np.zeros((2, 2))})
    monkeypatch.setattr(
        loso_alexmi, 'ALEXMI_TEACHER_TARGET_ROOT', tmp_path / 'missing_teacher_cache')
    monkeypatch.setattr(prepare, 'resolve_local_file', lambda path: Path(path))
    monkeypatch.setattr(prepare, '_device', lambda _value: torch.device('cpu'))
    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(prepare, '_alexmi_cell_paths',
                        lambda *_args: (tmp_path / 'cell', *(tmp_path / f'{name}' for name in
                                                             ('result.npz', 'cell.pt', 'cell.json', 'history.csv'))))
    monkeypatch.setattr(
        protocol, '_run',
        lambda *_args: pytest.fail('valid legacy AlexMI checkpoint must be reused'))

    prepare.ensure_teacher_checkpoint(
        'AlexMI', teacher, seed, fold, object(), 'cpu')

    cache_manifest = (tmp_path / 'missing_teacher_cache' / teacher / 'subject_01' /
                      f'seed_{seed}' / 'cache_manifest.json')
    assert not cache_manifest.exists()

    # A historical project-root path resolves to the same canonical external
    # checkpoint; its presence in an old cache manifest must stay a valid hit.
    old_alias = ('/home/lixinli/BigSmallCollab/results/reproductions/'
                 'loso_five_settings_canonical4s_v1/AlexMI/cbramod/'
                 'canonical4s_v1/subject_01/seed_7/model.pt')
    cache_manifest.parent.mkdir(parents=True)
    cache_manifest.write_text(json.dumps({
        'teacher_checkpoint': {
            'path': old_alias,
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    }))
    monkeypatch.setattr(
        prepare, 'resolve_local_file',
        lambda path: checkpoint if str(path) == old_alias else Path(path))
    prepare.ensure_teacher_checkpoint(
        'AlexMI', teacher, seed, fold, object(), 'cpu')
    assert 'cached teacher checkpoint identity differs' not in capsys.readouterr().out

    cache_manifest.write_text(json.dumps({'teacher_checkpoint': None}))
    prepare.ensure_teacher_checkpoint(
        'AlexMI', teacher, seed, fold, object(), 'cpu')
    assert 'teacher checkpoint reference is malformed' in capsys.readouterr().out


def test_alexmi_reader_keeps_small_model_fallback_and_accepts_any_nonnegative_seed(
        monkeypatch, tmp_path):
    monkeypatch.setattr(loso_alexmi, 'ALEXMI_RESULT_ROOT', tmp_path / 'results')
    monkeypatch.setattr(loso_alexmi, 'external_path', lambda path: Path(path))
    manifest_path = (tmp_path / 'results' / 'adfcnn' / 'scratch_supervised_loso_v1'
                    / 'subject_01' / 'seed_7' / 'manifest.json')
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        'dataset': 'AlexMI', 'model': 'adfcnn', 'seed': 7,
        'test_subject': 1, 'protocol': loso_alexmi._PROTOCOL,
        'dataset_source': 'AlexMI', 'label_values': loso_alexmi._LABELS,
        'num_classes': 2,
        'source_trial_selection': loso_alexmi._LOADER_RECIPE['source_trial_selection'],
        'window': loso_alexmi._LOADER_RECIPE['window'],
    }
    manifest_path.write_text(json.dumps(manifest))

    found_path, found_manifest = loso_alexmi._find_baseline_manifest('adfcnn', 7, 0)

    assert found_path == manifest_path
    assert found_manifest == manifest


def test_missing_alexmi_seed_calls_legacy_runner_for_one_requested_fold(
        monkeypatch, tmp_path):
    _stub_eeg_dataset_import(monkeypatch)
    from experiments.finetune import run_loso_five_datasets as protocol

    teacher, seed, fold = 'mirepnet', 777, 5
    spec = {
        'main_recipes': {teacher: {'name': 'canonical4s_v1'}},
        'datasets': {'AlexMI': {'classes': ['feet', 'right_hand']}},
    }
    monkeypatch.setattr(prepare, 'load_verified_alexmi_spec',
                        lambda _protocol: (spec, {'fixture': True}))
    monkeypatch.setattr(protocol, '_config', lambda *_args: {'epochs': 10})
    cell = tmp_path / 'canonical' / 'alexmi' / 'cell'
    outputs = (cell, cell / 'result.npz', cell / 'model.pt',
               cell / 'manifest.json', cell / 'train_history.csv')
    monkeypatch.setattr(prepare, '_alexmi_cell_paths', lambda *_args: outputs)
    monkeypatch.setattr(prepare, '_file_lock', _unlocked)
    monkeypatch.setattr(prepare, '_device', lambda _value: torch.device('cpu'))
    monkeypatch.setattr(prepare, 'external_path', lambda path: Path(path))
    monkeypatch.setattr(prepare, '_assert_no_active_worker', lambda *_args: False)
    validations = []

    def validate(*_args):
        validations.append(True)
        if len(validations) == 1:
            raise FileNotFoundError('requested AlexMI seed/fold is missing')
        return Path('manifest.json'), {}, Path('model.pt'), 'sha'

    monkeypatch.setattr(prepare, '_validate_alexmi_checkpoint', validate)
    run_calls = []
    monkeypatch.setattr(
        protocol, '_run',
        lambda args, got_spec, snapshot, device:
        run_calls.append((args, got_spec, snapshot, device)))

    prepare.ensure_teacher_checkpoint(
        'AlexMI', teacher, seed, fold, object(), 'cpu')

    assert len(validations) == 2
    args, got_spec, snapshot, device = run_calls[0]
    assert args.model == teacher
    assert args.datasets == ['AlexMI']
    assert args.seeds == [seed]
    assert args.folds == [fold]
    assert args.recipe == 'main'
    assert args.force is False and args.no_reuse_004 is False
    assert got_spec is spec and snapshot == {'fixture': True}
    assert device == 'cpu'


def test_negative_seed_is_rejected_before_any_training(monkeypatch):
    monkeypatch.setattr(prepare, '_device', lambda _value: torch.device('cpu'))
    with pytest.raises(ValueError, match='seed must be non-negative'):
        prepare.ensure_teacher_checkpoint(
            'BNCI2014004', 'mirepnet', -1, 0, object(), 'cpu')
