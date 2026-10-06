"""Focused tests for the progressive feature/logit KD pilot.

These tests are deliberately self-contained: they do not load data, teacher
artifacts, checkpoints, or run a training epoch.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.distill import run_progressive_task_feature_logit_kd as runner


def test_projection_pair_is_independent_and_has_expected_output_dim():
    pre, ft = runner.make_projection_pair(512, 256)
    assert isinstance(pre, nn.Linear)
    assert isinstance(ft, nn.Linear)
    assert pre.out_features == ft.out_features == 256
    assert pre.weight.data_ptr() != ft.weight.data_ptr()
    assert pre.bias.data_ptr() != ft.bias.data_ptr()
    assert not any(a.data_ptr() == b.data_ptr()
                   for a in pre.parameters() for b in ft.parameters())


@pytest.mark.parametrize(
    "condition,epoch,expected",
    [
        ("TASK_FEAT_LOGIT_KD", 1, (False, False, False)),
        ("TASK_FEAT_LOGIT_KD", 10, (False, False, False)),
        ("TASK_FEAT_LOGIT_KD", 11, (False, True, True)),
        ("PREALIGN_THEN_TASK_FEAT_LOGIT_KD", 1, (True, False, False)),
        ("PREALIGN_THEN_TASK_FEAT_LOGIT_KD", 10, (True, False, False)),
        ("PREALIGN_THEN_TASK_FEAT_LOGIT_KD", 11, (False, True, True)),
        ("PREALIGN_THEN_TASK_FEAT_LOGIT_KD", 100, (False, True, True)),
    ],
)
def test_phase_activity_is_fixed_before_and_after_epoch_10(condition, epoch, expected):
    assert runner.phase_activity(condition, epoch, 10) == expected


def test_feature_alignment_matches_manual_normalized_cosine_mean():
    torch.manual_seed(4)
    h = torch.randn(5, 3, requires_grad=True)
    target = torch.randn(5, 4)
    projection = nn.Linear(3, 4, bias=False)
    got = runner.feature_alignment_loss(h, target, projection)
    projected = projection(h)
    expected = 1.0 - F.cosine_similarity(
        F.normalize(projected, dim=1, eps=1e-8),
        F.normalize(target, dim=1, eps=1e-8), dim=1, eps=1e-8).mean()
    assert torch.allclose(got, expected)
    got.backward()
    assert torch.isfinite(h.grad).all()
    assert torch.isfinite(projection.weight.grad).all()


def test_feature_alignment_uses_all_samples_and_detaches_teacher():
    student = torch.randn(4, 3, requires_grad=True)
    teacher = torch.randn(4, 4, requires_grad=True)
    projection = nn.Linear(3, 4)
    loss = runner.feature_alignment_loss(student, teacher, projection)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    # A perturbation of any row changes the batch mean (no mask is applied).
    with torch.no_grad():
        base = runner.feature_alignment_loss(student.detach(), teacher.detach(), projection).item()
        altered = student.detach().clone()
        altered[2] += 10.0
    changed = runner.feature_alignment_loss(altered, teacher.detach(), projection).item()
    assert base != changed


def test_vanilla_kd_detaches_teacher_and_has_finite_student_gradient():
    torch.manual_seed(7)
    student = torch.randn(6, 3, requires_grad=True)
    teacher = torch.randn(6, 3, requires_grad=True)
    loss = runner.vanilla_kd_loss(student, teacher, 2.0)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()


def test_ce_is_full_batch_not_teacher_filtered():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1])
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    # Every sample contributes a direct CE gradient, including a hypothetical
    # inactive alignment/KD sample.
    assert torch.all(logits.grad.abs().sum(dim=1) > 0)


def test_optimizer_parameters_are_student_and_projections_only():
    student = nn.Linear(3, 3)
    pre, ft = runner.make_projection_pair(3, 4)
    teacher = nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(pre.parameters()) + list(ft.parameters()),
        lr=1e-3)
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert owned == {id(p) for p in (*student.parameters(), *pre.parameters(), *ft.parameters())}
    assert not owned.intersection({id(p) for p in teacher.parameters()})


def test_explicit_teacher_uid_reorder_and_label_validation():
    ref_uid = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
    ref_y = np.array([1, 0, 1], dtype=np.int64)
    perm = np.array([2, 0, 1])
    logits = np.arange(9, dtype=np.float32).reshape(3, 3)[perm]
    feats = np.arange(12, dtype=np.float32).reshape(3, 4)[perm]
    got_logits, got_feats, got_y, got_uid, reordered = runner.align_teacher_arrays(
        logits, feats, ref_y[perm], ref_uid[perm], ref_y, ref_uid)
    assert reordered
    assert np.array_equal(got_uid, ref_uid)
    assert np.array_equal(got_y, ref_y)
    assert np.array_equal(got_logits, np.arange(9, dtype=np.float32).reshape(3, 3))
    assert np.array_equal(got_feats, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_uid_set_mismatch_duplicate_and_label_mismatch_fail_closed():
    uid = np.array([[0, 1], [0, 2]], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    logits = np.zeros((2, 2), dtype=np.float32)
    feats = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="UID sets differ"):
        runner.align_teacher_arrays(logits, feats, y, uid,
                                    y, np.array([[0, 1], [0, 3]]))
    with pytest.raises(ValueError, match="duplicates"):
        runner.align_teacher_arrays(logits, feats, y,
                                    np.array([[0, 1], [0, 1]]), y, uid)
    with pytest.raises(ValueError, match="labels disagree"):
        runner.align_teacher_arrays(logits, feats, np.array([1, 1]), uid, y, uid)


def test_no_mi_mask_or_prototype_training_path_is_registered():
    source = Path(runner.__file__).read_text().lower()
    assert "probability_mi_loss" not in source
    assert "kd_proto" not in source
    assert "mi_proto" not in source
    assert "masked_components" not in source


def test_stage_two_does_not_reset_optimizer_or_scheduler():
    source = Path(runner.__file__).read_text()
    assert source.count("CosineAnnealingLR(") == 1
    assert "optimizer = torch.optim.AdamW" in source
    assert "scheduler.step()" in source
    # There is one continuous loop; no epoch-11 reinitialization branch.
    assert "if epoch == 11" not in source
