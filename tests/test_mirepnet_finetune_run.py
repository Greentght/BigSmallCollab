"""CPU-only regressions for fresh fine-tuning and explicit experiment resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import config
from experiments.finetune import run_loso_config_alignment as runner
from experiments.finetune import run_mirepnet_loso as standalone


def test_new_runs_never_reopen_an_existing_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(standalone, 'RUNS_ROOT', tmp_path)
    first = standalone.claim_run_directory('BNCI2014001', None, False)
    second = standalone.claim_run_directory('BNCI2014001', None, False)
    assert first != second
    named = standalone.claim_run_directory('BNCI2014001', 'trial', False)
    (named / 'run_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='run already exists'):
        standalone.claim_run_directory('BNCI2014001', 'trial', False)
    assert standalone.claim_run_directory('BNCI2014001', 'trial', True) == named


def test_resume_uses_frozen_yaml_and_rejects_parameter_or_input_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(standalone, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'CONFIG_DIR', str(tmp_path / 'configs'))
    monkeypatch.delenv('BIGSMALL_RUN_SNAPSHOT', raising=False)
    # Restore the environment automatically after prepare_plan sets its snapshot.
    monkeypatch.setenv('BIGSMALL_RUN_SNAPSHOT', '')
    model_path = tmp_path / 'configs/models/mirepnet.yaml'
    dataset_path = tmp_path / 'configs/datasets/BNCI2014001.yaml'
    model_path.parent.mkdir(parents=True)
    dataset_path.parent.mkdir(parents=True)
    raw = {'env': 'mirepnet', 'finetune': {'BNCI2014001': {'loso': {
        'epochs': 10, 'lr': 0.001, 'batch_size': 8, 'weight_decay': 1e-6, 'optimizer': 'adam'}}}}
    model_path.write_text(yaml.safe_dump(raw))
    dataset_path.write_text('num_subjects: 9\n')
    spec = tmp_path / 'protocol.yaml'
    spec.write_text('protocol: fixture\n')
    fake = SimpleNamespace(
        SPEC=spec, WIDEBAND_PROFILE='wideband_npy_v3',
        sha256_file=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        profile_config=lambda *_: config.load_model_config('mirepnet', 'BNCI2014001', 'loso'))
    identity = {'input_manifest_sha256': 'input-a', 'pretrained_checkpoint_sha256': 'weight-a'}
    monkeypatch.setattr(standalone, 'validate_inputs', lambda *_: deepcopy(identity))
    root = tmp_path / 'trial'
    root.mkdir()
    args = standalone.parse_args(['--gpu', '9', '--lr', '0.0001'])
    plan, _ = standalone.prepare_plan(args, root, fake)
    assert plan['seeds'] == [666, 667, 668]
    assert plan['resolved_training_config']['lr'] == 0.0001
    assert plan['resolved_training_config']['epochs'] == 10
    raw['finetune']['BNCI2014001']['loso']['epochs'] = 999
    model_path.write_text(yaml.safe_dump(raw))
    args = standalone.parse_args(['--gpu', '9', '--resume', '--run-name', 'trial'])
    resumed, _ = standalone.prepare_plan(args, root, fake)
    assert resumed == plan
    changed = standalone.parse_args(['--gpu', '9', '--resume', '--run-name', 'trial', '--epochs', '20'])
    with pytest.raises(ValueError, match='cannot change epochs'):
        standalone.prepare_plan(changed, root, fake)
    identity['input_manifest_sha256'] = 'input-b'
    with pytest.raises(ValueError, match='EEG input or pretrained checkpoint changed'):
        standalone.prepare_plan(args, root, fake)


@pytest.mark.parametrize('separate_run', [False, True])
def test_worker_trains_all_subjects_in_selected_output_without_mutating_baseline(
        monkeypatch, tmp_path, separate_run):
    canonical = tmp_path / 'baseline'
    canonical.mkdir()
    marker = canonical / 'preserve.txt'
    marker.write_text('old baseline')
    monkeypatch.setattr(runner, 'RESULT_ROOT', canonical)
    original = {'epochs': 10, 'lr': 0.001, 'batch_size': 8, 'weight_decay': 1e-6,
                'optimizer': 'adam', 'dropout': 0.5}
    monkeypatch.setattr(runner, 'profile_config', lambda *_: dict(original))
    x, y = np.zeros((18, 2), np.float32), np.asarray([0, 1] * 9)
    subjects = np.repeat(np.arange(9), 2)
    monkeypatch.setattr(runner, 'load_fold_data', lambda *_: (x, y, subjects, object(), {}))
    monkeypatch.setattr(runner, 'make_model', lambda *_: torch.nn.Linear(2, 2))
    monkeypatch.setattr(runner, 'wideband_stream_identity', lambda *_: {})
    scanned = []
    monkeypatch.setattr(runner, 'completed_fold', lambda path, *_: scanned.append(path) or None)
    seen = []

    def train(*args, **kwargs):
        seen.append((args[4], args[11], args[13]))
        return {'status': 'complete', 'held_out_subject': args[4] + 1, 'wall_seconds': 0,
                'metrics': {'accuracy': 0.75, 'balanced_accuracy': 0.75, 'kappa': 0.5, 'macro_f1': 0.75}}

    monkeypatch.setattr(runner, 'train_one_fold', train)
    worker = SimpleNamespace(profile=runner.WIDEBAND_PROFILE, dataset='BNCI2014001',
                             model='mirepnet', seed=666, epochs_override=None, preflight_steps=None)
    output = (tmp_path / 'fresh/seed_666' if separate_run else
              canonical / runner.WIDEBAND_PROFILE / 'BNCI2014001/mirepnet/seed_666')
    output.mkdir(parents=True)
    overrides = {'epochs': 20, 'lr': 0.0001} if separate_run else None
    runner.run_worker(worker, {'datasets': {'BNCI2014001': {'subjects': 9}}}, torch.device('cpu'),
                      seed_output=output if separate_run else None, training_overrides=overrides)
    assert [subject for subject, _, _ in seen] == list(range(9))
    assert scanned == [output / 'subject_01']
    assert all(path.parent == output for _, _, path in seen)
    assert all(cfg['epochs'] == (20 if separate_run else 10) for _, cfg, _ in seen)
    assert original['epochs'] == 10
    assert marker.read_text() == 'old baseline'
    summary = json.loads((output / 'seed_summary.json').read_text())
    assert summary['status'] == 'complete' and summary['folds_complete'] == 9
