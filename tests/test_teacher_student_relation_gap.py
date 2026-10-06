import numpy as np
import pytest

from test.qc.teacher_student_relation_gap import (
    align_by_uid,
    choose_random_mask,
    compute_sample_rows,
    prediction_group,
    relation_matrix,
)


def _uid(n=6):
    return np.asarray([[0, i] for i in range(n)], dtype=np.int64)


def test_uid_alignment_reorders_and_rejects_set_mismatch():
    ref = _uid(3)
    other = ref[[2, 0, 1]]
    _, aligned = align_by_uid(ref, other, {"x": np.asarray([[2], [0], [1]])})
    np.testing.assert_array_equal(aligned["x"].ravel(), [0, 1, 2])
    with pytest.raises(ValueError, match="UID set mismatch"):
        align_by_uid(ref, np.asarray([[0, 0], [0, 1], [0, 9]]))


def test_relation_matrix_size_symmetry_and_diagonal():
    rng = np.random.default_rng(1)
    _, relation = relation_matrix(rng.normal(size=(60, 7)))
    assert relation.shape == (60, 60)
    np.testing.assert_allclose(relation, relation.T, atol=1e-7)
    np.testing.assert_allclose(np.diag(relation), 1.0, atol=1e-6)


def test_prediction_groups():
    assert prediction_group(True, True) == "T_correct_S_correct"
    assert prediction_group(True, False) == "T_correct_S_wrong"
    assert prediction_group(False, True) == "T_wrong_S_correct"
    assert prediction_group(False, False) == "T_wrong_S_wrong"


def test_seed_rows_are_ranked_reproducibly_and_groups_partition_trials():
    rng = np.random.default_rng(4)
    n = 60
    uid = _uid(n)
    y = np.asarray([0, 1] * 30, dtype=np.int64)
    teacher_feat = rng.normal(size=(n, 8)).astype(np.float32)
    student_feat = rng.normal(size=(n, 5)).astype(np.float32)
    teacher_logits = rng.normal(size=(n, 2)).astype(np.float32)
    student_logits = rng.normal(size=(n, 2)).astype(np.float32)
    teacher = {"sample_uid": uid, "y": y, "feats": teacher_feat,
               "logits": teacher_logits}
    student = {"sample_uid": uid[::-1], "y": y[::-1],
               "feats": student_feat[::-1], "logits": student_logits[::-1]}
    rows_a, _ = compute_sample_rows(666, teacher, student, "D", 0, "train", (0, 1))
    rows_b, _ = compute_sample_rows(666, teacher, student, "D", 0, "train", (0, 1))
    assert sorted(r["rank_in_seed"] for r in rows_a) == list(range(1, 61))
    assert sum(r["is_known_bad"] for r in rows_a) == 1
    assert sorted(r["prediction_group"] for r in rows_a) == sorted(
        r["prediction_group"] for r in rows_b)
    assert [r["rank_in_seed"] for r in rows_a] == [r["rank_in_seed"] for r in rows_b]


def test_random_mask_is_unique_class_matched_and_excludes_top3():
    uid = [tuple(x) for x in _uid(60)]
    labels = np.asarray([0, 1] * 30)
    top3 = set(uid[:3])
    target = [labels[0], labels[1], labels[2]]
    chosen = choose_random_mask(uid, labels, top3, target, np.random.default_rng(9))
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    assert not set(chosen) & top3
    assert sorted(np.bincount([labels[uid.index(k)] for k in chosen])) == [1, 2]
