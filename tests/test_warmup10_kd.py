"""Fixed YAML warmup KD and the CE-to-KD training boundary."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from collab import distill
from experiments import config_loader
from experiments.distill import config_runner_impl as runner
from experiments.distill import warmup_kd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/loso/distill_kd.yaml"


def test_yaml_warmup_covers_full_matrix_with_fixed_pair_parameters():
    loaded = config_loader.load_experiment(CONFIG)
    selected = config_loader.select_experiment_runs(
        loaded, variants=["warmup10_kd"], seeds=[666, 667, 668],
        allow_loso_seed_override=True)
    runs = selected["runs"]
    raw = yaml.safe_load(CONFIG.read_text())
    pairs = {(pair["teacher"], pair["student"]): pair for pair in raw["pairs"]}
    assert len(runs) == 30
    assert {(run["dataset"], run["teacher"], run["student"]) for run in runs} == {
        (dataset, teacher, student)
        for dataset, (teacher, student) in product(raw["datasets"], pairs)
    }
    assert sum("Base" in run["methods"] for run in runs) == 30
    assert len({run["run_id"] for run in runs}) == 30
    for run in runs:
        pair = pairs[run["teacher"], run["student"]]
        assert run["variant"] == "warmup10_kd"
        assert run["methods"] == ["Base", "KD_warmup10"]
        assert run["seeds"] == [666, 667, 668]
        assert run["training"]["epochs"] == 100
        for key, value in pair["components"]["kd"]["params"].items():
            assert run["params"][key] == value
        args = runner._namespace(run, run["artifact_root"])
        kwargs, weight_mode = warmup_kd._method_kwargs(args)
        assert kwargs["distill_warmup_epochs"] == 10
        assert kwargs["lam_kd"] == run["params"]["lam_kd"]
        assert kwargs["temperature"] == run["params"]["temperature"]
        assert kwargs["lam_feat"] == kwargs["lam_mmd"] == 0
        assert kwargs["teacher_correct_only"] is False
        assert weight_mode == "all"


@pytest.mark.parametrize("override", [{"epochs": 10}, {"lam_kd": 0.0}])
def test_yaml_rejects_warmup_recipe_without_any_active_kd(tmp_path, override):
    raw = yaml.safe_load(CONFIG.read_text())
    if "epochs" in override:
        raw["pairs"][0]["training"] = override
    else:
        raw["pairs"][0]["components"]["kd"]["params"].update(override)
    path = tmp_path / "invalid-warmup.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config_loader.ExperimentConfigError, match="warmup10_kd"):
        config_loader.load_experiment(path)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))


class _TinyAdapter:
    device = torch.device("cpu")

    def build(self, _num_classes):
        return _TinyModel()

    def preprocess(self, values):
        return torch.as_tensor(values, dtype=torch.float32).reshape(len(values), -1)

    def forward(self, model, values):
        return values, values @ model.weight

    def infer(self, model, values):
        model.eval()
        with torch.no_grad():
            features, logits = self.forward(model, self.preprocess(values))
        return features.numpy(), logits.numpy()


def _data():
    values = np.array([[2., 0.], [0., 2.], [1., 0.], [0., 1.]], np.float32)
    labels = np.array([0, 1, 0, 1])
    # Deliberately disagree with CE labels so activation changes the update.
    teacher = np.array([[-3., 3.], [3., -3.], [-3., 3.], [3., -3.]], np.float32)
    return values, labels, teacher


def test_first_ten_updates_match_ce_then_kd_starts_without_optimizer_reset(monkeypatch):
    values, labels, teacher = _data()
    optimizers = []
    original_adamw = distill.optim.AdamW

    class RecordingAdamW(original_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weights = []
            self.learning_rates = []
            optimizers.append(self)

        def step(self, *args, **kwargs):
            self.learning_rates.append(self.param_groups[0]["lr"])
            result = super().step(*args, **kwargs)
            self.weights.append(self.param_groups[0]["params"][0].detach().clone())
            return result

    def no_projection(*_args, **_kwargs):
        raise AssertionError("ordinary warmup KD must not create a feature projector")

    monkeypatch.setattr(distill.optim, "AdamW", RecordingAdamW)
    monkeypatch.setattr(distill.nn, "Linear", no_projection)
    common = dict(lam_feat=0.0, teacher_correct_only=False, epochs=12,
                  batch_size=4, lr=0.01, weight_decay=0.0, seed=666,
                  return_training_details=True)
    distill.distill_student(
        _TinyAdapter(), 2, values, labels, None, None, values,
        lam_kd=0.0, **common)
    original_kl = distill.F.kl_div
    kl_calls = []

    def record_kl(*args, **kwargs):
        kl_calls.append(len(optimizers[-1].weights) + 1)
        return original_kl(*args, **kwargs)

    monkeypatch.setattr(distill.F, "kl_div", record_kl)
    _, details = warmup_kd.distill_student(
        _TinyAdapter(), 2, values, labels, None, teacher, values,
        lam_kd=0.5, distill_warmup_epochs=10, **common)

    assert len(optimizers) == 2  # One for CE and one for the entire warmup+KD run.
    ce, warmup = optimizers
    assert len(ce.weights) == len(warmup.weights) == 12
    assert ce.learning_rates == warmup.learning_rates
    for left, right in zip(ce.weights[:10], warmup.weights[:10]):
        assert torch.equal(left, right)
    assert not torch.equal(ce.weights[10], warmup.weights[10])
    assert kl_calls == [11, 12]
    history = details["training_history"]
    assert [row["kd_active"] for row in history] == [False] * 10 + [True] * 2
    assert all(row["total_loss"] == row["ce_loss"] for row in history[:10])
    assert all(row["total_loss"] > row["ce_loss"] for row in history[10:])


@pytest.mark.parametrize("warmup", [-1, 1.5, True, 12])
def test_trainer_rejects_invalid_warmup(warmup):
    values, labels, teacher = _data()
    with pytest.raises(ValueError, match="distill_warmup_epochs"):
        warmup_kd.distill_student(
            _TinyAdapter(), 2, values, labels, None, teacher, values,
            lam_feat=0.0, teacher_correct_only=False, epochs=12,
            distill_warmup_epochs=warmup)


@pytest.mark.parametrize("balanced", [False, True])
def test_independent_kd_path_matches_existing_kd_sampling_and_updates(balanced):
    values, labels, teacher = _data()
    kwargs = dict(lam_kd=0.5, lam_feat=0.0, teacher_correct_only=False,
                  epochs=3, batch_size=4, seed=666, lr=0.01, weight_decay=0.0,
                  balanced_batch=balanced,
                  subject_ids=np.array([0, 0, 1, 1]) if balanced else None,
                  return_training_details=True)
    _, original = distill.distill_student(
        _TinyAdapter(), 2, values, labels, None, teacher, values, **kwargs)
    _, independent = warmup_kd.distill_student(
        _TinyAdapter(), 2, values, labels, None, teacher, values,
        distill_warmup_epochs=0, **kwargs)
    assert original["batch_order_hashes"] == independent["batch_order_hashes"]
    torch.testing.assert_close(original["model"].weight, independent["model"].weight)


def test_warmup_result_keeps_normal_metrics_provenance_and_history(monkeypatch):
    values, labels, teacher_logits = _data()
    values = values[:, :, None]
    loaded = config_loader.load_experiment(CONFIG)
    spec = next(run for run in loaded["runs"] if run["variant"] == "warmup10_kd")
    args = runner._namespace(spec, spec["artifact_root"])
    args.epochs, args.batch_size = 12, 4
    monkeypatch.setattr(warmup_kd.legacy, "get_adapter", lambda *_args, **_kw: _TinyAdapter())

    class Teacher(dict):
        def __getitem__(self, key):
            if key == "feats":
                raise AssertionError("warmup KD must not read teacher features")
            return super().__getitem__(key)

    teacher = Teacher(logits=teacher_logits, _manifest_sha256="target-manifest",
                      _input_policy_sha256="teacher-policy")
    uids = np.array([[0, index] for index in range(len(labels))])
    row, details = warmup_kd.run_method(
        args, "BNCI2014004", "loso", "mirepnet", "ifnet", "mirepnet_loso",
        0, 666, "KD_warmup10", values, labels, values, labels, None,
        teacher, 2, "cpu", uid_tr=uids, uid_te=uids, return_details=True)
    assert row["method"] == "KD_warmup10"
    assert row["distill_warmup_epochs"] == 10
    assert row["student_epochs"] == 12
    assert row["lam_mmd"] == 0
    assert row["failure_status"] == "complete"
    assert row["n_train"] == row["n_test"] == len(labels)
    assert row["acc"] == row["test_accuracy"]
    assert row["teacher_target_manifest_sha256"] == "target-manifest"
    assert row["teacher_input_policy_sha256"] == "teacher-policy"
    assert len(row["split_uid_hash"]) == len(row["batch_order_hash"]) == 64
    assert len(details["training_history"]) == 12
