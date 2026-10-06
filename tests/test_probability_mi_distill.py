"""Unit tests for the class-joint probability-MI distillation pilot."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import collab.distill as distill


def test_probability_mi_matches_reference_formula_and_uses_c_by_c_joint():
    teacher_prob = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
         [0.7, 0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    student_prob = torch.tensor(
        [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
         [0.6, 0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    eps = 1e-8
    joint = teacher_prob.detach().T @ student_prob / teacher_prob.shape[0]
    assert joint.shape == (3, 3)
    joint = joint / joint.sum()
    tm = joint.sum(dim=1, keepdim=True)
    sm = joint.sum(dim=0, keepdim=True)
    expected = -(joint * torch.log((joint + eps) / (tm * sm + eps))).sum()

    actual = distill.probability_mi_loss(teacher_prob, student_prob, eps=eps)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    actual.backward()
    assert teacher_prob.grad is None
    assert student_prob.grad is not None
    assert torch.isfinite(student_prob.grad).all()


def test_probability_mi_rejects_nonmatching_or_invalid_probability_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        distill.probability_mi_loss(torch.ones(2, 2), torch.ones(2, 3))
    with pytest.raises(ValueError, match="B>=1 and C>=2"):
        distill.probability_mi_loss(torch.ones(2, 1), torch.ones(2, 1))


class _ToyModel(nn.Module):
    """A parameter-only model with no Linear module (for projection tests)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        self.bias = nn.Parameter(torch.zeros(2))
        self.train_ids = []
        self.train_batch_sizes = []

    def forward(self, x):
        if self.training:
            self.train_ids.extend(x[:, 0].detach().cpu().tolist())
            self.train_batch_sizes.append(int(x.shape[0]))
        return x, x @ self.weight + self.bias


class _ToyAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = {"batch_size": 3}
        self.models = []

    def build(self, _num_classes):
        model = _ToyModel()
        self.models.append(model)
        return model

    def preprocess(self, values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32)

    def forward(self, model, x):
        return model(x)

    def infer(self, model, values):
        x = self.preprocess(values)
        model.eval()
        feats, logits = [], []
        with torch.no_grad():
            for start in range(0, len(x), self.cfg["batch_size"]):
                feat, logit = model(x[start:start + self.cfg["batch_size"]])
                feats.append(feat.cpu().numpy())
                logits.append(logit.cpu().numpy())
        return np.concatenate(feats), np.concatenate(logits)


def _toy_data():
    # The first column is a stable trial id used to inspect DataLoader coverage.
    X = np.asarray([[0., 1.], [1., -1.], [2., 1.], [3., -1.], [4., 1.]])
    y = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
    teacher_logits = np.asarray([[3., -1.], [-1., 3.], [2., -2.],
                                 [-2., 2.], [3., -1.]], dtype=np.float32)
    return X, y, teacher_logits


def _initial_state():
    model = _ToyModel()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_ce_mi_is_logits_only_no_projection_no_kl_and_monitors_full_train():
    X, y, teacher_logits = _toy_data()
    adapter = _ToyAdapter()
    calls = []
    original_linear = distill.nn.Linear

    def spy_linear(*args, **kwargs):
        calls.append((args, kwargs))
        return original_linear(*args, **kwargs)

    def forbidden_kl(*_args, **_kwargs):
        raise AssertionError("CE+MI must not call KL")

    # No feature object is supplied at all; any access would fail the path.
    old_kl = distill.F.kl_div
    distill.nn.Linear = spy_linear
    distill.F.kl_div = forbidden_kl
    try:
        preds, mi_history = distill.distill_student(
            adapter, 2, X, y, None, teacher_logits, X[:2],
            lam_kd=0.0, lam_feat=0.0, lam_mmd=0.0,
            teacher_correct_only=False, probability_mi=True, lam_mi=0.1,
            epochs=2, lr=0.01, weight_decay=0.0, batch_size=3, seed=123,
            return_mi_history=True, initial_state_dict=_initial_state())
    finally:
        distill.nn.Linear = original_linear
        distill.F.kl_div = old_kl

    assert preds.shape == (2,)
    assert len(mi_history) == 2
    assert np.all(np.isfinite(mi_history))
    assert calls == []
    model = adapter.models[0]
    assert len(model.train_ids) == 2 * len(X)
    for start in range(0, len(model.train_ids), len(X)):
        assert sorted(model.train_ids[start:start + len(X)]) == list(range(len(X)))
    assert model.train_batch_sizes == [3, 2, 3, 2]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_base_and_vanilla_kd_do_not_create_projection_but_mmd_does(monkeypatch):
    X, y, teacher_logits = _toy_data()
    original_linear = distill.nn.Linear
    calls = []

    def spy_linear(*args, **kwargs):
        calls.append((args, kwargs))
        return original_linear(*args, **kwargs)

    class _ForbiddenFeatures:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("non-feature distillation path read teacher feats")

    monkeypatch.setattr(distill.nn, "Linear", spy_linear)
    for method_kwargs, feat in (
        ({"lam_kd": 0.0, "lam_feat": 0.0,
          "teacher_correct_only": False}, _ForbiddenFeatures()),
        ({"lam_kd": 0.1, "lam_feat": 0.0,
          "teacher_correct_only": False}, _ForbiddenFeatures()),
        ({"lam_kd": 0.0, "lam_feat": 0.0, "lam_mmd": 0.1,
          "teacher_correct_only": False},
         np.asarray(X, dtype=np.float32)),
    ):
        adapter = _ToyAdapter()
        distill.distill_student(
            adapter, 2, X, y, feat, teacher_logits, X[:1],
            epochs=1, lr=0.01, weight_decay=0.0, batch_size=3, seed=7,
            initial_state_dict=_initial_state(), **method_kwargs)
    assert len(calls) == 1


def test_probability_mi_rejects_masks_and_loso_sampler():
    X, y, teacher_logits = _toy_data()
    with pytest.raises(ValueError, match="teacher-correct"):
        distill.distill_student(
            _ToyAdapter(), 2, X, y, None, teacher_logits, X[:1],
            lam_kd=0.0, lam_feat=0.0, teacher_correct_only=True,
            probability_mi=True, lam_mi=0.1, epochs=1)
    with pytest.raises(ValueError, match="balanced or alternate"):
        distill.distill_student(
            _ToyAdapter(), 2, X, y, None, teacher_logits, X[:1],
            lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False,
            probability_mi=True, lam_mi=0.1, balanced_batch=True, epochs=1)


def test_initial_state_dict_is_loaded_strictly():
    X, y, teacher_logits = _toy_data()
    state = _initial_state()
    first = _ToyAdapter()
    second = _ToyAdapter()
    distill.distill_student(
        first, 2, X, y, None, teacher_logits, X[:1],
        lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False,
        epochs=1, lr=0.0, weight_decay=0.0, batch_size=3, seed=1,
        initial_state_dict=state)
    distill.distill_student(
        second, 2, X, y, None, teacher_logits, X[:1],
        lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False,
        epochs=1, lr=0.0, weight_decay=0.0, batch_size=3, seed=999,
        initial_state_dict=state)
    for left, right in zip(first.models[0].state_dict().values(),
                           second.models[0].state_dict().values()):
        # lr=0 leaves both at the shared initial state despite different RNG seeds.
        assert torch.equal(left, right)


def test_runner_ce_mi_is_logits_only_and_has_fixed_bundle(monkeypatch):
    from experiments.distill import run_distill as runner

    assert runner._parse_methods(["mi"]) == ["Base", "KD_all", "CE_MI"]
    args = SimpleNamespace(
        temperature=2.0, lam_kd=0.5, lam_mi=0.1, lam_mmd=0.5,
        mmd_sigmas=[0.5, 1.0], mmd_normalize=True,
        mmd_class_conditional=False, epochs=1, lr=0.01,
        weight_decay=0.0, batch_size=2,
    )
    X, y, teacher_logits = _toy_data()
    X = X[:, :, None]  # runner runtime config expects (N, channels, samples)
    seen = {}

    class _Teacher(dict):
        def __getitem__(self, key):
            if key == "feats":
                raise AssertionError("CE_MI must not read teacher feats")
            return super().__getitem__(key)

    teacher = _Teacher(logits=teacher_logits)
    monkeypatch.setattr(
        runner.config, "load_model_config",
        lambda *_args, **_kwargs: {"epochs": 1, "lr": 0.01,
                                   "weight_decay": 0.0, "batch_size": 2})
    monkeypatch.setattr(runner, "get_adapter", lambda *_args, **_kwargs: object())

    def fake_distill(adapter, nc, Xtr, ytr, feat_t, log_t, Xte, **kwargs):
        seen.update(feat_t=feat_t, log_t=log_t, kwargs=kwargs)
        return np.zeros(len(Xte), dtype=np.int64), [0.2]

    monkeypatch.setattr(runner, "distill_student", fake_distill)
    monkeypatch.setattr(runner.metrics, "evaluate",
                        lambda _y, _p: {"acc": 50.0, "kappa": 0.0})
    row = runner._run_method(
        args, "toy", "fewshot", "mirepnet", "ifnet", "mirepnet",
        0, 666, "CE_MI", X, y, X[:2], y[:2], None, teacher, 2, "cpu",
        initial_state_dict={"shared": "state"})
    assert seen["feat_t"] is None
    assert np.array_equal(seen["log_t"], teacher_logits)
    assert seen["kwargs"]["lam_kd"] == 0.0
    assert seen["kwargs"]["lam_feat"] == 0.0
    assert seen["kwargs"]["lam_mmd"] == 0.0
    assert seen["kwargs"]["lam_mi"] == 0.1
    assert seen["kwargs"]["probability_mi"] is True
    assert row["method"] == "CE_MI"
    assert row["full_train_mi_last"] == 0.2
