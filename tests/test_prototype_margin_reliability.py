"""Unit tests for the offline prototype-margin diagnostic.

All tests use synthetic arrays; no project artifact (and especially no test
artifact) is opened here.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "qc"))
import prototype_margin_reliability as pmr  # noqa: E402


def _artifact(order=(0, 1, 2, 3), labels=(0, 0, 1, 1),
              logits=None, feats=None, policy="fewshot_stratified_random"):
    uid = np.asarray([[0, i] for i in order], dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits is None:
        logits = np.column_stack([labels == 0, labels == 1]).astype(float)
    if feats is None:
        base = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        feats = base[np.asarray(order)]
    return {"logits": np.asarray(logits), "feats": np.asarray(feats),
            "y": labels, "sample_uid": uid, "split_policy": policy}


def test_uid_reorder_and_label_validation():
    ref = _artifact()
    # Other artifact is in the reverse UID order; labels/features follow it.
    other = _artifact(order=(3, 2, 1, 0), labels=(1, 1, 0, 0),
                     feats=np.asarray([[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    aligned = pmr.align_by_uid(ref, other)
    assert np.array_equal(aligned["sample_uid"], ref["sample_uid"])
    assert np.array_equal(aligned["y"], ref["y"])


def test_uid_duplicate_set_and_label_mismatch_fail():
    ref = _artifact()
    duplicate = dict(_artifact())
    duplicate["sample_uid"] = duplicate["sample_uid"].copy()
    duplicate["sample_uid"][1] = duplicate["sample_uid"][0]
    with pytest.raises(ValueError, match="duplicates"):
        pmr.align_by_uid(ref, duplicate)

    mismatch = dict(_artifact())
    mismatch["sample_uid"] = mismatch["sample_uid"].copy()
    mismatch["sample_uid"][0] = [0, 99]
    with pytest.raises(ValueError, match="UID set mismatch"):
        pmr.align_by_uid(ref, mismatch)

    bad_label = dict(_artifact(order=(3, 2, 1, 0), labels=(0, 1, 0, 1),
                              feats=np.asarray([[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])))
    with pytest.raises(ValueError, match="label mismatch"):
        pmr.align_by_uid(ref, bad_label)


def test_l2_normalization_and_loo_excludes_self():
    feats = np.asarray([[3.0, 0.0], [0.0, 4.0], [-3.0, 0.0], [0.0, -4.0]])
    y = np.asarray([0, 0, 1, 1])
    uid = np.asarray([[0, i] for i in range(4)])
    result = pmr.prototype_metrics(feats, y, uid)
    assert np.allclose(result["z"], feats / np.linalg.norm(feats, axis=1, keepdims=True))
    # For class 0, LOO prototype for row 0 is exactly row 1's normalized vector.
    assert np.allclose(result["loo_proto_by_row"][0], [0.0, 1.0])
    assert np.allclose(result["loo_proto_by_row"][1], [1.0, 0.0])
    # The other-class full prototype uses both class-1 samples, not a LOO value.
    assert np.allclose(result["full_proto"][1], [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])
    assert np.isclose(result["sim_wrong"][0], -1.0 / np.sqrt(2))


def test_proto_pred_and_margin_are_loo_values():
    feats = np.asarray([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [-0.8, -0.6]])
    y = np.asarray([0, 0, 1, 1])
    result = pmr.prototype_metrics(feats, y)
    assert np.array_equal(result["proto_pred"], y)
    assert np.all(result["proto_margin"] > 0)
    # Recomputing row 0 manually confirms its own vector was not in the true prototype.
    z = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    true = z[1]
    wrong = np.mean(z[2:], axis=0)
    wrong /= np.linalg.norm(wrong)
    assert np.isclose(result["sim_true"][0], np.dot(z[0], true))
    assert np.isclose(result["sim_wrong"][0], np.dot(z[0], wrong))


def test_groups_and_router_tie_abstain():
    assert [pmr.correctness_group(a, b) for a, b in
            ((True, True), (True, False), (False, True), (False, False))] == ["A", "B", "C", "D"]
    row = {"fm_classifier_margin": 0.2, "sm_classifier_margin": 0.2,
           "fm_normalized_entropy": 0.4, "sm_normalized_entropy": 0.4,
           "fm_proto_margin": 0.1, "sm_proto_margin": 0.1,
           "fm_view_agree": True, "sm_view_agree": True}
    for name in ("classifier_margin", "entropy", "prototype", "view_agreement"):
        assert pmr.router_choice(name, row) is None


def test_empty_disagreement_router_is_na():
    rows = [{"group": "A", "fm_classifier_margin": 0.1, "sm_classifier_margin": 0.2,
             "fm_normalized_entropy": 0.3, "sm_normalized_entropy": 0.4,
             "fm_proto_margin": 0.2, "sm_proto_margin": 0.1,
             "fm_view_agree": True, "sm_view_agree": False}]
    stats = pmr._routing_stats(rows, "prototype")
    assert stats["routing_acc"] is None
    assert stats["coverage"] is None
    assert stats["na_reason"] == "no routing evaluation opportunity"


def test_each_seed_is_independent():
    a = _artifact()
    b = _artifact(feats=np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]))
    out_a = pmr.analyze_seed("D", 0, "session_A", 666, a, a)
    out_b = pmr.analyze_seed("D", 0, "session_A", 667, b, b)
    assert out_a["seed"] == 666 and out_b["seed"] == 667
    assert out_a["feature_dims"] == out_b["feature_dims"]
    assert out_a["sample_rows"] is not out_b["sample_rows"]


def test_rejects_test_paths_and_requires_train_fields(tmp_path):
    with pytest.raises(ValueError, match=r"only \*_train.npz"):
        pmr.load_train_artifact(tmp_path / "0_666_test.npz")
    with pytest.raises(ValueError, match=r"only \*_train.npz"):
        pmr.validate_train_path(tmp_path / "0_666_val.npz")
    bad = tmp_path / "0_666_train.npz"
    np.savez(bad, logits=np.zeros((2, 2)), feats=np.ones((2, 2)), y=np.zeros(2))
    with pytest.raises(ValueError, match="missing required fields"):
        pmr.load_train_artifact(bad)
    split_test = tmp_path / "0_667_train.npz"
    np.savez(split_test, logits=np.zeros((2, 2)), feats=np.ones((2, 2)),
             y=np.zeros(2), sample_uid=np.asarray([[0, 0], [0, 1]]), split_policy="test")
    with pytest.raises(ValueError, match="split=test"):
        pmr.load_train_artifact(split_test)


def test_nonempty_output_refused_and_force_scope(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "old.txt").write_text("keep")
    with pytest.raises(ValueError, match="non-empty"):
        pmr._ensure_output_dir(out)
    with pytest.raises(ValueError, match="only inside"):
        pmr._ensure_output_dir(out, force=True)


def test_deterministic_csv_serialization(tmp_path):
    rows = [{"seed": 666, "value": 0.1}, {"seed": 667, "value": None}]
    fields = ["seed", "value"]
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    pmr.write_csv(first, rows, fields)
    pmr.write_csv(second, rows, fields)
    assert first.read_bytes() == second.read_bytes()
    with first.open(newline="") as handle:
        assert list(csv.DictReader(handle))[1]["value"] == "NA"


def test_stability_stats_include_pairwise_and_loo():
    result = pmr.prototype_metrics(np.asarray([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]]),
                                   np.asarray([0, 0, 1, 1]))
    for label, row in result["stability"].items():
        assert row["n_samples"] == 2
        assert row["within_class"]["mean"] == pytest.approx(1.0)
        assert row["sample_to_loo"]["mean"] == pytest.approx(1.0)
        assert row["loo_full"]["mean"] == pytest.approx(1.0)
