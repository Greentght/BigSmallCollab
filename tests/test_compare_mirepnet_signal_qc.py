import inspect
import numpy as np
import pytest

from test.qc.compare_mirepnet_signal_qc import (
    MIREPNET_THRESHOLD,
    SIGNAL_THRESHOLD,
    QCScoreError,
    align_by_uid,
    compute_mirepnet_knn_scores,
    compute_signal_features,
    compute_signal_scores,
    mirepnet_robust_z,
    _random_masks_for_composition,
)


def test_mirepnet_knn_excludes_self_and_isolated_point_is_highest():
    x = np.asarray([
        [1.0, 0.0], [1.0, 0.01], [0.99, 0.02], [1.0, -0.01],
        [0.98, 0.01], [1.0, 0.02], [-1.0, 0.0],
    ])
    score = compute_mirepnet_knn_scores(x, k=3)
    assert score[-1] == pytest.approx(np.max(score))
    assert score[-1] > 0.1


def test_mirepnet_zero_mad_is_explicit_failure_and_threshold_is_fixed():
    with pytest.raises(QCScoreError, match="MAD is zero"):
        mirepnet_robust_z(np.ones(8))
    score = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.7])
    z, median, mad = mirepnet_robust_z(score)
    assert median == pytest.approx(0.45)
    assert mad == pytest.approx(0.2)
    assert np.array_equal(z > MIREPNET_THRESHOLD, z > 3.5)


def test_signal_spike_flatline_and_nonfinite_are_flagged():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 4, 40)).astype(np.float32)
    x[0, 0, 10] = 1000.0
    x[1, 1, :] = 0.0
    x[2, 2, 3] = np.nan
    scored = compute_signal_scores(x)
    assert scored["flag"][0]
    assert scored["flag"][1]
    assert scored["flag"][2]
    assert "nonfinite_fraction" in scored["trigger_features"][2]


def test_signal_uses_robust_relative_scores_not_absolute_voltage_threshold():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 3, 30)).astype(np.float32)
    scaled = x * 1e6
    normal = compute_signal_scores(x)
    scaled_scores = compute_signal_scores(scaled)
    assert np.array_equal(normal["flag"], scaled_scores["flag"])
    assert np.allclose(normal["score"], scaled_scores["score"], atol=1e-7)


def test_signal_qc_input_does_not_accept_labels_test_or_known_bad():
    assert "label" not in inspect.signature(compute_signal_scores).parameters
    assert "test" not in inspect.signature(compute_signal_scores).parameters
    assert "known_bad" not in inspect.signature(compute_signal_scores).parameters
    assert "label" not in inspect.signature(compute_mirepnet_knn_scores).parameters


def test_uid_alignment_detects_set_mismatch():
    ref = np.asarray([[0, 0], [0, 1], [0, 2]])
    other = np.asarray([[0, 2], [0, 0], [0, 9]])
    with pytest.raises(ValueError, match="UID set mismatch"):
        align_by_uid(ref, other, {"x": np.zeros((3, 1))}, "test")


def test_same_input_repeats_same_scores():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 3, 30)).astype(np.float32)
    a = compute_signal_scores(x)
    b = compute_signal_scores(x.copy())
    np.testing.assert_array_equal(a["flag"], b["flag"])
    np.testing.assert_array_equal(a["score"], b["score"])


def test_random_masks_are_unique_and_class_matched():
    uids = np.asarray([[0, i] for i in range(60)], dtype=np.int64)
    labels = np.asarray([0, 1] * 30, dtype=np.int64)
    target = [(0, 1), (0, 2), (0, 5)]
    masks = _random_masks_for_composition(uids, labels, target, seed=666, n_masks=20)
    assert len(masks) == 20
    assert len({tuple(mask) for mask in masks}) == 20
    for mask in masks:
        selected = set(mask)
        assert sum(labels[i] == 0 for i, uid in enumerate(uids) if tuple(uid) in selected) == 1
        assert sum(labels[i] == 1 for i, uid in enumerate(uids) if tuple(uid) in selected) == 2
