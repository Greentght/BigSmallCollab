"""Full-matrix native feature KD, cosine loss and retained projector checks."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from experiments import config_loader
from experiments.distill import config_runner_impl as runner
from experiments.distill import warmup_kd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/loso/distill_kd.yaml"


def test_feature_yaml_covers_all_datasets_pairs_and_seeds_with_fixed_params():
    selected = config_loader.select_experiment_runs(
        config_loader.load_experiment(CONFIG), variants=["kd_feature"],
        seeds=[666, 667, 668], allow_loso_seed_override=True)
    raw = yaml.safe_load(CONFIG.read_text())
    pairs = {(pair["teacher"], pair["student"]): pair for pair in raw["pairs"]}
    runs = selected["runs"]
    assert len(runs) == 30
    assert {(run["dataset"], run["teacher"], run["student"]) for run in runs} == {
        (dataset, teacher, student)
        for dataset, (teacher, student) in product(raw["datasets"], pairs)
    }
    for run in runs:
        pair = pairs[run["teacher"], run["student"]]
        assert run["methods"] == ["Base", "KD_feature"]
        assert run["seeds"] == [666, 667, 668]
        assert run["training"]["epochs"] == 100
        assert run["params"]["lam_feat"] == pair["components"]["feature"]["params"]["lam_feat"]
        for key, value in pair["components"]["kd"]["params"].items():
            assert run["params"][key] == value
        args = runner._namespace(run, run["artifact_root"])
        kwargs, weight_mode = warmup_kd._method_kwargs(args, "KD_feature")
        assert kwargs["distill_warmup_epochs"] == 0
        assert kwargs["lam_feat"] == 0.5
        assert kwargs["lam_mmd"] == 0
        assert weight_mode == "all"


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


def _train(*, lr=0.0, teacher_features=None):
    values, labels, logits, features = _data()
    return warmup_kd.distill_student(
        _Adapter(), 2, values, labels, features if teacher_features is None else teacher_features,
        logits, values, epochs=1, batch_size=4, lr=lr, weight_decay=0.0,
        lam_kd=0.25, lam_feat=0.5, temperature=2.0, seed=666,
        distill_warmup_epochs=0, return_training_details=True)


def test_feature_loss_matches_ce_kd_and_native_cosine_formula():
    values, labels, teacher_logits, teacher_features = _data()
    _, details = _train()
    projector = details["projector"]
    assert projector.weight.shape == (3, 2)
    features = torch.as_tensor(values)
    ce = F.cross_entropy(features, torch.as_tensor(labels))
    kd = 4 * F.kl_div(F.log_softmax(features / 2, dim=1),
                      F.softmax(torch.as_tensor(teacher_logits) / 2, dim=1),
                      reduction="batchmean")
    cosine = (1 - F.cosine_similarity(projector(features),
                                      torch.as_tensor(teacher_features), dim=1)).mean()
    history = details["training_history"][0]
    assert history["kd_active"] is True
    assert history["feature_loss"] == pytest.approx(cosine.item())
    assert history["total_loss"] == pytest.approx((ce + 0.25 * kd + 0.5 * cosine).item())


def test_projector_and_native_student_features_receive_training_gradients():
    _, initial = _train()
    _, trained = _train(lr=0.01)
    assert any(not torch.equal(initial["projector"].state_dict()[name], value)
               for name, value in trained["projector"].state_dict().items())
    assert not torch.equal(initial["model"].embedding, trained["model"].embedding)
    for parameter in list(trained["model"].parameters()) + list(trained["projector"].parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("features", [None, np.ones((3, 3)), np.ones(4), np.full((4, 3), np.nan)])
def test_feature_kd_rejects_missing_or_invalid_teacher_features(features):
    values, labels, teacher_logits, _ = _data()
    with pytest.raises(ValueError, match="teacher features"):
        warmup_kd.distill_student(
            _Adapter(), 2, values, labels, features, teacher_logits, values,
            lam_feat=0.5, epochs=1, distill_warmup_epochs=0)


def test_feature_checkpoint_retains_and_validates_trained_projector(tmp_path, monkeypatch):
    values, labels, logits, features = _data()
    values = values[:, :, None]
    spec = next(run for run in config_loader.load_experiment(CONFIG)["runs"]
                if run["variant"] == "kd_feature")
    args = runner._namespace(spec, spec["artifact_root"])
    args.epochs, args.batch_size = 1, 4
    monkeypatch.setattr(warmup_kd.legacy, "get_adapter", lambda *_a, **_kw: _Adapter())
    teacher = {"logits": logits, "feats": features,
               "_source_manifest_sha256": "source",
               "_student_input_policy_sha256": "student-policy",
               "_input_policy_sha256": "teacher-policy"}
    uids = np.array([[0, index] for index in range(len(labels))])
    row, details = warmup_kd.run_method(
        args, "BNCI2014004", "loso", "mirepnet", "ifnet", "mirepnet_loso",
        0, 666, "KD_feature", values, labels, values, labels, None,
        teacher, 2, "cpu", uid_tr=uids, uid_te=uids, return_details=True)
    assert row["distill_warmup_epochs"] == 0
    assert row["lam_feat"] == 0.5
    assert row["feature_metric"] == "cosine"
    assert row["feature_transform"] == "native_penultimate"
    row.update(artifact_run_id="feature-test", config_name="feature-test")
    payload = {"complete": True,
               "run_key": runner._run_key(row["dataset"], 0, 666, "KD_feature", "feature-test"),
               "state_dict": details["model"].state_dict()}
    root = tmp_path / "artifacts"
    runner._save_run_artifacts(root, row, details, payload)
    checkpoint = runner._artifact_file_map(root, row)["checkpoint"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert warmup_kd.legacy._sha256_state_dict(saved["projector_state_dict"]) == row["projector_state_hash"]
    assert runner._row_complete(row, root)
    del saved["projector_state_dict"]
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="trained projector"):
        runner._row_complete(row, root)
