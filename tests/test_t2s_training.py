"""CPU trajectory checks for resumable T2S stages, with no EEG/artifact inputs."""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from collab import t2s_training as training
from collab.mcc_diffkd import EpsilonPredictor


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(6, 4)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        features = self.dropout(torch.tanh(self.bn(self.stem(x))))
        return features, self.head(features)


class TinyAdapter:
    device = torch.device('cpu')

    @staticmethod
    def forward(model, x):
        return model(x)


@pytest.fixture(autouse=True)
def external_tmp(monkeypatch):
    # The production guard stays intact. Test outputs are disposable tmp_path
    # artifacts, so replace only the destination check in this utility module.
    monkeypatch.setattr(training, 'require_external_output', lambda path: Path(path))
    torch.set_num_threads(1)


@pytest.fixture
def inputs():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((12, 6)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 6)
    target = rng.standard_normal((12, 4)).astype(np.float32)
    target = (target - target.mean(1, keepdims=True)) / np.sqrt(target.var(1, keepdims=True) + 1e-5)
    torch.manual_seed(3)
    model = TinyStudent()
    return x, y, target, copy.deepcopy(model.state_dict())


def _student_cfg():
    return {'epochs': 3, 'batch_size': 4, 'optimizer': 'adamw',
            'lr': .001, 'weight_decay': .01}


def _paths(tmp_path, name):
    return {'results_dir': tmp_path / name / 'results',
            'weights_dir': tmp_path / name / 'weights'}


def test_student_epoch_resume_matches_uninterrupted(tmp_path, inputs):
    x, y, target, initial = inputs
    cfg = _student_cfg()
    first = training.train_student(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='same', seed=42, initial_state=initial, teacher_targets=target,
        schedule_seed=1234, **_paths(tmp_path, 'full'))
    stop = tmp_path / 'stop'
    stop.touch()
    paused = training.train_student(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='same', seed=42, initial_state=initial, teacher_targets=target,
        schedule_seed=1234, stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert paused == {'status': 'paused', 'epoch': 1}
    stop.unlink()
    resumed = training.train_student(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='same', seed=42, initial_state=initial, teacher_targets=target,
        schedule_seed=1234, stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert first['model_state_sha256'] == resumed['model_state_sha256']
    assert Path(first['history']).read_text() == Path(resumed['history']).read_text()


def test_teacher_mcc_epoch_resume_matches_uninterrupted(tmp_path, inputs):
    x, y, _, initial = inputs
    cfg = {'epochs': 3, 'batch_size': 4, 'lr': 1e-4,
           'weight_decay': 1e-6, 'mcc': {'temperature': 2., 'weight': 1.}}
    _, full = training.adapt_teacher(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='teacher', seed=91, initial_state=initial, mcc_weight=1.,
        schedule_seed=92, **_paths(tmp_path, 'full'))
    stop = tmp_path / 'stop'
    stop.touch()
    _, paused = training.adapt_teacher(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='teacher', seed=91, initial_state=initial, mcc_weight=1.,
        schedule_seed=92, stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert paused['epoch'] == 1
    stop.unlink()
    _, resumed = training.adapt_teacher(TinyAdapter(), TinyStudent(), x, y, cfg,
        fingerprint='teacher', seed=91, initial_state=initial, mcc_weight=1.,
        schedule_seed=92, stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert full['model_state_sha256'] == resumed['model_state_sha256']
    assert Path(full['history']).read_text() == Path(resumed['history']).read_text()


def test_prior_resume_and_constructor_preserve_dropout_rng(tmp_path, inputs, monkeypatch):
    _, _, target, _ = inputs
    cfg = {'training_updates': 4, 'batch_size_cap': 4,
           'hidden_dimensions': 8, 'time_embedding_dimensions': 32,
           'diagnostic_interval': 2}
    torch.manual_seed(811)
    np.random.seed(812)
    random.seed(813)
    before = training.capture_rng('cpu')
    _, full = training.fit_prior(target, cfg, fingerprint='prior', seed=71,
        device='cpu', **_paths(tmp_path, 'full'))
    after = training.capture_rng('cpu')
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])
    assert before['python'] == after['python']
    assert before['numpy'][0] == after['numpy'][0]
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    stop = tmp_path / 'stop'
    original_save = training.atomic_torch

    def save_and_stop(path, value):
        result = original_save(path, value)
        if Path(path).name == 'resume.pt' and value.get('update') == 2:
            stop.touch()
        return result

    monkeypatch.setattr(training, 'atomic_torch', save_and_stop)
    _, paused = training.fit_prior(target, cfg, fingerprint='prior', seed=71,
        device='cpu', stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert paused == {'status': 'paused', 'update': 2}
    stop.unlink()
    monkeypatch.setattr(training, 'atomic_torch', original_save)
    prior, resumed = training.fit_prior(target, cfg, fingerprint='prior', seed=71,
        device='cpu', stop_file=stop, **_paths(tmp_path, 'resumed'))
    assert full['model_state_sha256'] == resumed['model_state_sha256']
    assert not prior.training
    assert all(not p.requires_grad and p.grad is None for p in prior.parameters())


def test_diffkd_reaches_student_without_changing_prior(tmp_path, inputs):
    x, y, target, initial = inputs
    torch.manual_seed(17)
    prior = EpsilonPredictor(4, 8, 32)
    before = training.state_hash(prior)
    done = training.train_student(TinyAdapter(), TinyStudent(), x, y, _student_cfg(),
        fingerprint='diff', seed=42, initial_state=initial, teacher_targets=target,
        prior=prior, epochs=1, max_batches=2, **_paths(tmp_path, 'diff'))
    assert done['uses_diffusion']
    assert training.state_hash(prior) == before
    assert all(not p.requires_grad and p.grad is None for p in prior.parameters())
    import csv
    with Path(done['history']).open() as stream:
        first = next(csv.DictReader(stream))
    assert float(first['feature_native_gradient_norm']) > 0
    assert float(first['feature_backbone_gradient_norm']) > 0


def test_zero_feature_bypasses_prior_and_matches_ce(tmp_path, inputs):
    x, y, _, initial = inputs

    class PoisonPrior(nn.Module):
        def eval(self):
            raise AssertionError('zero feature weight must not call the prior')

    plain = training.train_student(TinyAdapter(), TinyStudent(), x, y, _student_cfg(),
        fingerprint='zero', seed=42, initial_state=initial, feature_weight=0.,
        **_paths(tmp_path, 'plain'))
    bypassed = training.train_student(TinyAdapter(), TinyStudent(), x, y, _student_cfg(),
        fingerprint='zero', seed=42, initial_state=initial, feature_weight=0.,
        prior=PoisonPrior(), **_paths(tmp_path, 'bypassed'))
    assert plain['model_state_sha256'] == bypassed['model_state_sha256']
    assert Path(plain['history']).read_text() == Path(bypassed['history']).read_text()


def test_complete_without_weights_never_retrains(tmp_path, inputs):
    x, y, _, initial = inputs
    paths = _paths(tmp_path, 'complete')
    done = training.train_student(TinyAdapter(), TinyStudent(), x, y, _student_cfg(),
        fingerprint='done', seed=42, initial_state=initial, feature_weight=0.,
        epochs=1, **paths)
    Path(done['checkpoint']).unlink()

    class PoisonAdapter(TinyAdapter):
        @staticmethod
        def forward(model, x):
            raise AssertionError('a completed result must never retrain')

    saved = training.read_completion(paths['results_dir'], 'done')
    returned = training.train_student(PoisonAdapter(), TinyStudent(), x, y, _student_cfg(),
        fingerprint='done', seed=42, initial_state=initial, feature_weight=0.,
        epochs=1, **paths)
    assert saved == done == returned
