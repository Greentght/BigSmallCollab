"""Synthetic tests for the independent epochwise diagnostic."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "qc"))
import epochwise_prototype_reliability as epr  # noqa: E402


def _snapshot(logit_pred):
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    uid = np.asarray([[0, i] for i in range(4)], dtype=np.int64)
    logits = np.asarray([[2., 0.] if p == 0 else [0., 2.] for p in logit_pred])
    feats = np.asarray([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
    return {"sample_uid": uid, "y": y, "logits": logits,
            "probability": epr.stable_softmax(logits),
            "prediction": np.asarray(logit_pred), "feats": feats,
            "split_policy": "fewshot_stratified_random"}


def test_fixed_schedule_covers_shared_and_model_specific_final():
    schedule = epr.observation_schedule(10, 100)
    assert [row["epoch"] for row in schedule] == ["1", "2", "3", "5", "10", "final"]
    assert schedule[-1]["fm_epoch"] == 10
    assert schedule[-1]["sm_epoch"] == 100
    assert schedule[-1]["same_epoch"] is False
    assert epr.snapshot_epochs(7) == [1, 2, 3, 5, 7]


def test_epoch_pair_has_complete_abcd_and_prototype_fields():
    # FM correct/SM wrong at row 1; FM wrong/SM correct at row 2.
    fm = _snapshot([0, 0, 0, 1])
    sm = _snapshot([0, 1, 1, 1])
    schedule = {"epoch": "1", "epoch_label": "epoch_1", "fm_epoch": 1,
                "sm_epoch": 1, "same_epoch": True}
    result = epr.analyze_epoch_pair("D", 0, 666, schedule, fm, sm)
    rows = result["sample_rows"]
    assert [r["group"] for r in rows] == ["A", "B", "C", "A"]
    summary = result["group_summary"]
    assert (summary["n_A"], summary["n_B"], summary["n_C"], summary["n_D"]) == (2, 1, 1, 0)
    assert "fm_proto_margin" in rows[0] and "sm_proto_margin" in rows[0]
    assert summary["n_disagreement"] == 2


def test_router_tie_abstains_and_small_disagreement_is_marked():
    rows = [
        {"group": "B", "fm_classifier_margin": .2, "sm_classifier_margin": .2,
         "fm_normalized_entropy": .5, "sm_normalized_entropy": .5,
         "fm_proto_margin": .1, "sm_proto_margin": .1,
         "fm_view_agree": True, "sm_view_agree": True},
        {"group": "C", "fm_classifier_margin": .2, "sm_classifier_margin": .2,
         "fm_normalized_entropy": .5, "sm_normalized_entropy": .5,
         "fm_proto_margin": .1, "sm_proto_margin": .1,
         "fm_view_agree": False, "sm_view_agree": False},
    ]
    stats = epr._stats_choice(rows, "prototype")
    assert stats["n_covered"] == 0
    assert stats["accuracy"] is None
    assert stats["na_reason"] == "no covered samples"

    one = epr._stats_choice(rows[:1], "prototype")
    assert one["accuracy"] is None
    assert one["na_reason"] == "insufficient disagreement samples"


def test_uid_alignment_and_test_snapshot_rejection(tmp_path):
    ref = _snapshot([0, 0, 0, 1])
    other = _snapshot([0, 1, 1, 1])
    other = dict(other)
    order = [3, 2, 1, 0]
    for key in ("sample_uid", "y", "logits", "probability", "prediction", "feats"):
        other[key] = np.asarray(other[key])[order]
    aligned = epr._align_snapshot(ref, other)
    assert np.array_equal(aligned["sample_uid"], ref["sample_uid"])
    assert np.array_equal(aligned["y"], ref["y"])

    forbidden = tmp_path / "epoch_001_test.npz"
    np.savez(forbidden, sample_uid=ref["sample_uid"], y=ref["y"],
             logits=ref["logits"], probability=ref["probability"],
             prediction=ref["prediction"], feats=ref["feats"])
    with pytest.raises(ValueError, match="test snapshot"):
        epr.load_snapshot(forbidden)


def test_uid_alignment_rejects_duplicate_set_and_label_mismatch():
    reference = _snapshot([0, 0, 0, 1])

    duplicate = dict(reference)
    duplicate["sample_uid"] = np.array(reference["sample_uid"], copy=True)
    duplicate["sample_uid"][1] = duplicate["sample_uid"][0]
    with pytest.raises(ValueError, match="duplicates"):
        epr._align_snapshot(reference, duplicate)

    different_set = dict(reference)
    different_set["sample_uid"] = np.array(reference["sample_uid"], copy=True)
    different_set["sample_uid"][0] = [0, 99]
    with pytest.raises(ValueError, match="UID sets do not match"):
        epr._align_snapshot(reference, different_set)

    different_label = dict(reference)
    different_label["y"] = np.array(reference["y"], copy=True)
    different_label["y"][0] = 1
    with pytest.raises(ValueError, match="labels differ"):
        epr._align_snapshot(reference, different_label)


def test_snapshot_probability_and_prediction_must_match_logits(tmp_path):
    snap = _snapshot([0, 0, 1, 1])
    path = tmp_path / "epoch_001.npz"
    np.savez(path, **snap)
    loaded = epr.load_snapshot(path)
    assert np.array_equal(loaded["prediction"], np.argmax(loaded["probability"], axis=1))

    bad_probability = dict(snap)
    bad_probability["probability"] = np.array(snap["probability"], copy=True)
    bad_probability["probability"][0, 0] = 0.01
    bad_path = tmp_path / "epoch_002.npz"
    np.savez(bad_path, **bad_probability)
    with pytest.raises(ValueError, match="probability is inconsistent"):
        epr.load_snapshot(bad_path)


def test_l2_normalization_loo_and_margin_are_explicit():
    # Each class has two non-unit vectors.  For row 0, the true prototype
    # must be the normalized row-1 vector, not the full class mean including
    # row 0.  The wrong-class prototype uses both class-1 rows.
    features = np.asarray([[2., 0.], [0., 3.], [0., 4.], [3., 0.]])
    labels = np.asarray([0, 0, 1, 1])
    uid = np.asarray([[0, i] for i in range(4)])
    metrics = epr.prototype_metrics(features, labels, uid)
    assert np.allclose(metrics["z"], np.asarray([[1., 0.], [0., 1.], [0., 1.], [1., 0.]]))
    assert np.isclose(metrics["sim_true"][0], 0.0)
    assert np.isclose(metrics["sim_wrong"][0], 1 / np.sqrt(2))
    assert np.isclose(metrics["proto_margin"][0], -1 / np.sqrt(2))
    # Row 0 is predicted as the wrong class by the LOO prototype.  If its
    # own feature leaked into the true prototype, the result would differ.
    assert metrics["proto_pred"][0] == 1
    assert np.isclose(metrics["loo_full_cos"][0], 1 / np.sqrt(2))


def test_empty_group_and_no_disagreement_return_na():
    rows = [{"group": "A", "fm_proto_margin": .2, "sm_proto_margin": .1,
             "fm_classifier_margin": .2, "sm_classifier_margin": .1,
             "fm_normalized_entropy": .2, "sm_normalized_entropy": .1,
             "fm_view_agree": True, "sm_view_agree": True}]
    assert epr._group_accuracy(rows, "C", "SM")["accuracy"] is None
    stats = epr._stats_choice(rows, "prototype")
    assert stats["accuracy"] is None
    assert stats["na_reason"] == "no disagreement samples"


def test_nonempty_output_refusal_and_force_scope(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal-root"
    monkeypatch.setattr(epr, "FORMAL_OUTPUT_ROOT", formal_root)
    target = formal_root / "diagnostic"
    target.mkdir(parents=True)
    (target / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="non-empty"):
        epr._ensure_out_dir(target, force=False)
    epr._ensure_out_dir(target, force=True)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="only under the formal"):
        epr._ensure_out_dir(outside, force=True)


def test_repeated_epoch_pair_is_byte_stable_in_rows_and_summary():
    fm = _snapshot([0, 0, 0, 1])
    sm = _snapshot([0, 1, 1, 1])
    schedule = {"epoch": "2", "epoch_label": "epoch_2", "fm_epoch": 2,
                "sm_epoch": 2, "same_epoch": True}
    first = epr.analyze_epoch_pair("D", 0, 666, schedule, fm, sm)
    second = epr.analyze_epoch_pair("D", 0, 666, schedule, fm, sm)
    assert first["sample_rows"] == second["sample_rows"]
    assert first["group_summary"] == second["group_summary"]


def test_sample_trajectory_counts_requested_patterns():
    schedule = [{"epoch": str(i), "epoch_label": str(i)} for i in (1, 2, 3, 5)]
    rows = []
    groups = ["D", "C", "C", "A"]
    for epoch, group in zip(("1", "2", "3", "5"), groups):
        rows.append({"seed": 666, "sample_uid": "(0, 7)", "epoch": epoch,
                     "group": group, "fm_correct": group in ("A", "B"),
                     "sm_correct": group in ("A", "C"), "fm_proto_margin": .2,
                     "sm_proto_margin": .1, "delta_proto": .1})
    trajectories, counts = epr.build_trajectories(rows, schedule)
    assert trajectories[0]["group_trajectory"] == "D|C|C|A"
    assert counts["D_to_C_to_A"] == 1
    assert counts["C_to_A"] == 1
