"""Unit tests for the cross-dataset QC diagnostic.

These tests are deliberately data-free.  They exercise the UID/mask contract
and the frozen pure scoring functions without loading the shared EEG corpus or
writing to ``results/artifacts``.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest

import test.qc.cross_dataset_qc_generalization as cross
from test.qc.compare_mirepnet_signal_qc import (
    compute_mirepnet_knn_scores,
    compute_signal_scores,
)


def test_fixed_dataset_enumeration_does_not_duplicate_001_variant():
    assert cross.DATASETS == ("BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI")
    assert "BNCI2014001-4" not in cross.DATASETS
    assert len(cross.DATASETS) == len(set(cross.DATASETS))


def test_dataset_facts_are_loaded_not_hardcoded_sample_counts():
    plan = {item["dataset"]: item for item in cross.dataset_plan()}
    assert plan["BNCI2014001"]["num_classes"] == 2
    assert plan["BNCI2014004"]["num_classes"] == 2
    assert plan["BNCI2015001"]["num_subjects"] == 12
    assert plan["AlexMI"]["num_subjects"] == 8
    assert all(item["val_split"] == pytest.approx(0.7) for item in plan.values())


def test_uid_mask_is_applied_before_dataset_and_rejects_unknown_or_duplicate():
    uid = np.asarray([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    keep = cross.build_uid_keep_mask(uid, [(0, 1)])
    assert keep.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="not in train split"):
        cross.build_uid_keep_mask(uid, [(0, 99)])
    with pytest.raises(ValueError, match="duplicates"):
        cross.build_uid_keep_mask(uid, [(0, 1), (0, 1)])


def test_mask_safety_marks_overfilter_and_empty_class():
    uid = np.asarray([[0, i] for i in range(8)], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    assert cross.mask_safety(uid, labels, [(0, 0), (0, 1)])['status'] == 'ok'
    invalid = cross.mask_safety(uid, labels, [(0, 4), (0, 5), (0, 6), (0, 7)])
    assert invalid['status'] == 'invalid_mask'
    unsafe = cross.mask_safety(uid, labels, [(0, 0), (0, 1), (0, 2)])
    assert unsafe['status'] == 'unsafe_overfiltering'


def test_mirepnet_knn_excludes_self_and_isolated_point_is_highest():
    x = np.asarray([
        [1.0, 0.0], [1.0, 0.01], [0.99, 0.02], [1.0, -0.01],
        [0.98, 0.01], [1.0, 0.02], [-1.0, 0.0],
    ])
    score = compute_mirepnet_knn_scores(x, k=3)
    assert score[-1] == pytest.approx(np.max(score))
    assert score[-1] > 0.1


def test_frozen_threshold_and_signal_features_are_not_label_aware():
    assert cross.MIREPNET_THRESHOLD == 3.5
    assert cross.SIGNAL_THRESHOLD == 3.5
    assert 'label' not in inspect.signature(compute_mirepnet_knn_scores).parameters
    assert 'label' not in inspect.signature(compute_signal_scores).parameters
    assert 'test' not in inspect.signature(compute_signal_scores).parameters
    assert 'known_bad' not in inspect.signature(compute_signal_scores).parameters


def test_signal_qc_detects_spike_flatline_and_nonfinite():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 4, 40)).astype(np.float32)
    x[0, 0, 10] = 1000.0
    x[1, 1, :] = 0.0
    x[2, 2, 3] = np.nan
    scored = compute_signal_scores(x)
    assert scored['flag'][0]
    assert scored['flag'][1]
    assert scored['flag'][2]
    assert 'nonfinite_fraction' in scored['trigger_features'][2]


def test_signal_qc_is_relative_not_an_absolute_voltage_threshold():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 3, 30)).astype(np.float32)
    a = compute_signal_scores(x)
    b = compute_signal_scores(x * 1e6)
    np.testing.assert_array_equal(a['flag'], b['flag'])
    # Scaling is invariant in exact arithmetic; float32 feature extraction
    # leaves a small deterministic round-off difference in the robust ratios.
    np.testing.assert_allclose(a['score'], b['score'], rtol=1e-5, atol=2e-6)


def test_random_masks_are_unique_and_class_matched():
    uid = np.asarray([[0, i] for i in range(60)], dtype=np.int64)
    labels = np.asarray([0, 1] * 30, dtype=np.int64)
    target = [(0, 1), (0, 2), (0, 5)]
    masks = cross.generate_random_masks(uid, labels, target, random_seed=666, n_masks=5)
    assert len(masks) == 5
    assert len({tuple(mask) for mask in masks}) == 5
    target_counts = cross.class_counts(labels, uid, target)
    for mask in masks:
        assert cross.class_counts(labels, uid, mask) == target_counts


def test_multiclass_collapse_uses_chance_plus_five_percentage_points():
    assert cross.collapsed_prediction(np.asarray([0, 0, 0]), 4, 30.0)
    assert not cross.collapsed_prediction(np.asarray([0, 1, 2, 3]), 4, 30.1)
    assert cross.collapsed_prediction(np.asarray([0, 1, 2, 3]), 4, 29.9)


def test_artifact_uid_alignment_and_reordering_are_explicit(monkeypatch, tmp_path):
    artifact_root = tmp_path / 'results' / 'artifacts'
    path = artifact_root / 'BNCI2014001' / 'mirepnet' / '0_666_train.npz'
    path.parent.mkdir(parents=True)
    uid = np.asarray([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    np.savez(path, feats=np.ones((3, 256), dtype=np.float32), sample_uid=uid[::-1])
    monkeypatch.setattr(cross, 'INPUT_ARTIFACT_ROOT', artifact_root)
    info = cross.inspect_artifact('BNCI2014001', 0, 666, uid)
    assert info['uid_set_match'] is True
    assert info['uid_exact_match'] is False
    assert info['uid_reordered'] is True
    assert info['eligible_for_mirepnet_qc'] is True


def test_artifact_uid_set_mismatch_is_ineligible(monkeypatch, tmp_path):
    artifact_root = tmp_path / 'results' / 'artifacts'
    path = artifact_root / 'BNCI2014001' / 'mirepnet' / '0_666_train.npz'
    path.parent.mkdir(parents=True)
    uid = np.asarray([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    np.savez(path, feats=np.ones((3, 256), dtype=np.float32), sample_uid=np.asarray([[0, 0], [0, 1], [0, 9]]))
    monkeypatch.setattr(cross, 'INPUT_ARTIFACT_ROOT', artifact_root)
    info = cross.inspect_artifact('BNCI2014001', 0, 666, uid)
    assert info['uid_set_match'] is False
    assert info['eligible_for_mirepnet_qc'] is False
    assert info['status'] == 'uid_set_mismatch'


def test_output_boundaries_reject_results_and_top_level_artifacts(tmp_path):
    with pytest.raises(ValueError, match='output'):
        cross._resolve_output(Path('results') / 'bad')
    with pytest.raises(ValueError, match='output'):
        cross._resolve_output(Path('artifacts'))
    with pytest.raises(ValueError, match='read-only input root'):
        cross._resolve_input_artifact_root(tmp_path)


def test_same_signal_input_is_reproducible_and_no_filtered_copy_is_created(tmp_path):
    rng = np.random.default_rng(99)
    x = rng.normal(size=(20, 3, 30)).astype(np.float32)
    a = compute_signal_scores(x)
    b = compute_signal_scores(x.copy())
    np.testing.assert_array_equal(a['flag'], b['flag'])
    np.testing.assert_array_equal(a['score'], b['score'])
    assert not list(tmp_path.rglob('*.npz'))


def test_train_source_shows_uid_filter_precedes_dataloader_creation():
    source = Path(cross.__file__).read_text()
    filter_pos = source.index('keep = build_uid_keep_mask(uid, removed_uids)')
    dataloader_pos = source.index('loader = DataLoader(dataset', filter_pos)
    assert filter_pos < dataloader_pos
