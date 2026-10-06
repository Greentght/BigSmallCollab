import numpy as np
import pytest

from test.qc.minimal_causal_diagnostic import (
    align_by_uid,
    assert_condition_uids,
    filter_by_uid,
)


def _uids(n=60):
    return np.asarray([[0, i] for i in range(n)], dtype=np.int64)


def test_filter_is_uid_based_and_happens_before_dataset_creation():
    uids = _uids()
    mask = filter_by_uid(uids, [(0, 1)])
    assert mask.dtype == bool
    assert int(mask.sum()) == 59
    assert not mask[1]
    assert np.array_equal(uids[mask], np.delete(uids, 1, axis=0))
    assert_condition_uids(uids, uids[mask], [(0, 1)])


def test_uid_alignment_reorders_and_rejects_mismatch():
    reference = _uids(3)
    candidate = reference[[2, 0, 1]]
    aligned = align_by_uid(
        reference, candidate, {"value": np.asarray([[2], [0], [1]])})
    np.testing.assert_array_equal(aligned["value"].ravel(), [0, 1, 2])
    with pytest.raises(ValueError, match="UID set mismatch"):
        align_by_uid(reference, np.asarray([[0, 0], [0, 1], [0, 9]]),
                     {"value": np.zeros((3, 1))})


def test_filter_does_not_remove_test_uid_or_duplicate_samples():
    train = _uids(60)
    test = np.asarray([[0, 100 + i] for i in range(140)], dtype=np.int64)
    clean = train[filter_by_uid(train, [(0, 1)])]
    assert len({tuple(x) for x in clean}) == 59
    assert not set(map(tuple, clean)) & set(map(tuple, test))
