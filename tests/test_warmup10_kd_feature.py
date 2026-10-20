"""Combined CE warmup, logits KD and native feature alignment."""
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


CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/loso/distill_kd.yaml"
VARIANT = "warmup10_kd_feature"
METHOD = "KD_warmup10_feature"


def test_combined_variant_covers_full_fixed_matrix_with_distinct_identity():
    loaded = config_loader.load_experiment(CONFIG)
    selected = config_loader.select_experiment_runs(
        loaded, variants=[VARIANT], seeds=[666, 667, 668], allow_loso_seed_override=True)
    raw = yaml.safe_load(CONFIG.read_text())
    pairs = {(pair["teacher"], pair["student"]): pair for pair in raw["pairs"]}
    runs = selected["runs"]
    assert len(runs) == 30
    assert {(run["dataset"], run["teacher"], run["student"]) for run in runs} == {
        (dataset, teacher, student)
        for dataset, (teacher, student) in product(raw["datasets"], pairs)
    }
    prior = {run["run_id"] for run in loaded["runs"]
             if run["variant"] in {"kd_feature", "warmup10_kd"}}
    assert prior.isdisjoint(run["run_id"] for run in runs)
    for run in runs:
        assert run["methods"] == ["Base", METHOD]
        assert run["seeds"] == [666, 667, 668]
        assert run["training"]["epochs"] == 100
        pair = pairs[run["teacher"], run["student"]]
        for component in ("kd", "feature"):
            for key, value in pair["components"][component]["params"].items():
                assert run["params"][key] == value
        kwargs, weight_mode = warmup_kd._method_kwargs(
            runner._namespace(run, run["artifact_root"]), METHOD)
        assert kwargs["distill_warmup_epochs"] == 10
        assert kwargs["lam_feat"] == 0.5
        assert kwargs["lam_kd"] == run["params"]["lam_kd"]
        assert kwargs["temperature"] == run["params"]["temperature"]
        assert kwargs["lam_mmd"] == 0
        assert weight_mode == "all"


@pytest.mark.parametrize("setting", ["no_kd_epochs", "zero_kd_weight"])
def test_combined_recipe_rejects_a_budget_or_loss_without_kd(tmp_path, setting):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["variants"] = {VARIANT: raw["variants"][VARIANT]}
    raw["default_variants"] = [VARIANT]
    for pair in raw["pairs"]:
        pair["components"] = {key: value for key, value in pair["components"].items()
                              if key in {"kd", "feature"}}
    if setting == "no_kd_epochs":
        raw["pairs"][0]["training"] = {"epochs": 10}
    else:
        raw["pairs"][0]["components"]["kd"]["params"]["lam_kd"] = 0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config_loader.ExperimentConfigError, match=VARIANT):
        config_loader.load_experiment(path)


class _Model(nn.Module):
    feat_dim = 2

    def __init__(self):
        super().__init__()
        self.embedding = nn.Parameter(torch.eye(2))
        self.classifier = nn.Parameter(torch.eye(2))


class _Adapter:
    device = torch.device("cpu")

    def build(self, _num_classes):
        return _Model()

    def preprocess(self, values):
        return torch.as_tensor(values, dtype=torch.float32).reshape(len(values), -1)

    def forward(self, model, values):
        features = values @ model.embedding
        return features, features @ model.classifier

    def infer(self, model, values):
        model.eval()
        with torch.no_grad():
            features, logits = self.forward(model, self.preprocess(values))
        return features.numpy(), logits.numpy()


def _data():
    values = np.array([[2., 0.], [0., 2.], [1., 1.], [1., -1.]], np.float32)
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[-2., 2.], [2., -2.], [-2., 2.], [2., -2.]], np.float32)
    features = np.array([[1., 2., 0.], [0., 1., 2.], [2., 0., 1.], [1., -1., 2.]], np.float32)
    return values, labels, logits, features


@pytest.mark.parametrize("balanced", [False, True])
def test_kd_and_projector_both_start_at_epoch_eleven_without_reset(monkeypatch, balanced):
    values, labels, teacher_logits, teacher_features = _data()
    optimizers = []
    original = warmup_kd.optim.AdamW

    class RecordingAdamW(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial = [p.detach().clone() for p in self.param_groups[0]["params"]]
            self.updates, self.learning_rates = [], []
            optimizers.append(self)

        def step(self, *args, **kwargs):
            self.learning_rates.append(self.param_groups[0]["lr"])
            result = super().step(*args, **kwargs)
            self.updates.append([p.detach().clone() for p in self.param_groups[0]["params"]])
            return result

    monkeypatch.setattr(warmup_kd.optim, "AdamW", RecordingAdamW)
    common = dict(lam_feat=0.0, teacher_correct_only=False, epochs=12,
                  batch_size=4, lr=0.01, weight_decay=0.01, seed=666,
                  balanced_batch=balanced,
                  subject_ids=np.array([0, 0, 1, 1]) if balanced else None,
                  return_training_details=True)
    _, ce_details = distill.distill_student(
        _Adapter(), 2, values, labels, None, None, values, lam_kd=0.0, **common)
    kl_calls, feature_calls = [], []
    original_kl, original_cosine = warmup_kd.F.kl_div, warmup_kd.F.cosine_similarity

    def kl(*args, **kwargs):
        kl_calls.append(len(optimizers[-1].updates) + 1)
        return original_kl(*args, **kwargs)

    def cosine(*args, **kwargs):
        feature_calls.append(len(optimizers[-1].updates) + 1)
        return original_cosine(*args, **kwargs)

    monkeypatch.setattr(warmup_kd.F, "kl_div", kl)
    monkeypatch.setattr(warmup_kd.F, "cosine_similarity", cosine)
    common["lam_feat"] = 0.5
    _, details = warmup_kd.distill_student(
        _Adapter(), 2, values, labels, teacher_features, teacher_logits, values,
        lam_kd=0.5, distill_warmup_epochs=10, **common)
    assert len(optimizers) == 2
    ce, combined = optimizers
    assert len(combined.updates) == 12
    assert ce.learning_rates == combined.learning_rates
    assert ce_details["batch_order_hashes"] == details["batch_order_hashes"]
    for epoch in range(10):
        for expected, actual in zip(ce.updates[epoch], combined.updates[epoch][:2]):
            assert torch.equal(expected, actual)
        for expected, actual in zip(combined.initial[2:], combined.updates[epoch][2:]):
            assert torch.equal(expected, actual)
    assert any(not torch.equal(a, b) for a, b in zip(combined.initial[2:], combined.updates[10][2:]))
    assert kl_calls == feature_calls == [11, 12]
    history = details["training_history"]
    assert all(row["kd_loss"] == row["feature_loss"] == 0 for row in history[:10])
    assert all(row["total_loss"] == row["ce_loss"] for row in history[:10])
    assert all(row["kd_loss"] > 0 and row["feature_loss"] > 0 for row in history[10:])


@pytest.mark.parametrize("invalid_projector", ["missing", "changed"])
def test_combined_result_preserves_projector_and_refuses_invalid_completion(tmp_path, monkeypatch,
                                                                         invalid_projector):
    values, labels, logits, features = _data()
    values = values[:, :, None]
    spec = next(run for run in config_loader.load_experiment(CONFIG)["runs"] if run["variant"] == VARIANT)
    args = runner._namespace(spec, spec["artifact_root"])
    args.epochs, args.batch_size = 12, 4
    monkeypatch.setattr(warmup_kd.legacy, "get_adapter", lambda *_a, **_kw: _Adapter())
    teacher = {"logits": logits, "feats": features,
               "_source_manifest_sha256": "source",
               "_student_input_policy_sha256": "student-policy",
               "_input_policy_sha256": "teacher-policy"}
    train_uids = np.array([[0, index] for index in range(len(labels))])
    test_uids = np.array([[1, index] for index in range(2)])
    row, details = warmup_kd.run_method(
        args, "BNCI2014004", "loso", "mirepnet", "ifnet", "mirepnet_loso",
        0, 666, METHOD, values, labels, values[:2], labels[:2], None,
        teacher, 2, "cpu", uid_tr=train_uids, uid_te=test_uids, return_details=True)
    assert row["method"] == METHOD
    assert "10-epoch CE warmup" in row["condition_label"]
    assert row["distill_warmup_epochs"] == 10
    assert row["lam_feat"] == 0.5
    assert row["feature_metric"] == "cosine"
    row.update(artifact_run_id="combined-test", config_name="combined-test")
    payload = {"complete": True,
               "run_key": runner._run_key(row["dataset"], 0, 666, METHOD, "combined-test"),
               "state_dict": details["model"].state_dict()}
    root = tmp_path / "artifacts"
    runner._save_run_artifacts(root, row, details, payload)
    checkpoint = runner._artifact_file_map(root, row)["checkpoint"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert warmup_kd.legacy._sha256_state_dict(saved["projector_state_dict"]) == row["projector_state_hash"]
    assert runner._row_complete(row, root)
    if invalid_projector == "missing":
        del saved["projector_state_dict"]
    else:
        saved["projector_state_dict"]["weight"][0, 0] += 0.1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="projector"):
        runner._row_complete(row, root)
