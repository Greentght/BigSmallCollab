import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / 'experiments' / 'distill' / 'run_task_feature_logit_kd_six_pairs.py'
spec = importlib.util.spec_from_file_location('six_pair_runner', RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


CONFIG = ROOT / 'configs' / 'experiments' / 'task_feature_logit_kd_six_pairs_seed666.yaml'


def test_config_is_exact_six_pair_scope():
    cfg = runner.validate_config(CONFIG)
    assert tuple(cfg['datasets']) == runner.DATASETS
    assert tuple(cfg['teachers']) == runner.TEACHERS
    assert tuple(cfg['students']) == runner.STUDENTS
    assert tuple(cfg['conditions']) == runner.CONDITIONS
    assert runner.TEACHER_CONDITIONS == ('DELAYED_LOGIT_KD', 'TASK_FEAT_LOGIT_KD')
    assert sum(runner.SUBJECT_COUNTS.values()) == 38
    assert cfg['protocol'] == 'fewshot'
    assert float(cfg['val_split']) == 0.7


def _write_artifact(path, uids, labels, logits=None, feats=None):
    uids = np.asarray(uids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits is None:
        logits = np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2)
    if feats is None:
        feats = np.arange(len(labels) * 3, dtype=np.float32).reshape(len(labels), 3)
    np.savez(path, logits=np.asarray(logits, dtype=np.float32),
             feats=np.asarray(feats, dtype=np.float32), y=labels,
             sample_uid=uids, split_policy=np.asarray('fewshot_stratified_random'))


def test_teacher_uid_alignment_explicitly_reorders(tmp_path):
    uid_ref = np.array([[0, 10], [0, 11], [0, 12]], dtype=np.int64)
    y_ref = np.array([1, 0, 1], dtype=np.int64)
    order = [2, 0, 1]
    path = tmp_path / 'teacher_train.npz'
    _write_artifact(path, uid_ref[order], y_ref[order])
    out = runner.align_teacher_artifact(path, y_ref, uid_ref)
    assert np.array_equal(out['sample_uid'], uid_ref)
    assert np.array_equal(out['y'], y_ref)
    assert out['reordered'] is True
    assert out['alignment_hash'] == runner._hash_bytes(
        uid_ref.astype('<i8').tobytes(), y_ref.astype('<i8').tobytes())


def test_teacher_uid_duplicate_set_and_label_errors(tmp_path):
    uid_ref = np.array([[0, 1], [0, 2]], dtype=np.int64)
    y_ref = np.array([0, 1], dtype=np.int64)
    duplicate = tmp_path / 'duplicate_train.npz'
    _write_artifact(duplicate, [[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError, match='duplicate'):
        runner.align_teacher_artifact(duplicate, y_ref, uid_ref)
    missing = tmp_path / 'missing_train.npz'
    _write_artifact(missing, [[0, 1], [0, 3]], [0, 1])
    with pytest.raises(ValueError, match='UID sets differ'):
        runner.align_teacher_artifact(missing, y_ref, uid_ref)
    labels = tmp_path / 'labels_train.npz'
    _write_artifact(labels, uid_ref, [1, 0])
    with pytest.raises(ValueError, match='labels disagree'):
        runner.align_teacher_artifact(labels, y_ref, uid_ref)


def test_test_artifact_is_rejected_before_read(tmp_path):
    path = tmp_path / '0_666_test.npz'
    with pytest.raises(ValueError, match='test'):
        runner.align_teacher_artifact(path, np.array([0]), np.array([[0, 0]]))


def test_feature_alignment_is_normalized_cosine_and_teacher_detached():
    student = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 1.0], [1.0, 0.0]], requires_grad=True)
    projection = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    loss = runner.feature_alignment_loss(student, teacher, projection)
    expected = 1.0 - F.cosine_similarity(
        F.normalize(student, dim=1, eps=runner.FEATURE_EPS),
        F.normalize(teacher, dim=1, eps=runner.FEATURE_EPS), dim=1,
        eps=runner.FEATURE_EPS).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert torch.isfinite(projection.weight.grad).all()


def test_vanilla_kd_detaches_teacher_and_has_finite_student_grad():
    student = torch.tensor([[0.5, -0.1], [0.2, 0.8]], requires_grad=True)
    teacher = torch.tensor([[1.2, -0.4], [-0.2, 0.7]], requires_grad=True)
    loss = runner.vanilla_kd_loss(student, teacher)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert loss.ndim == 0


def test_two_projections_are_independent_and_have_teacher_dimension():
    p_pre = nn.Linear(5, 7)
    p_ft = nn.Linear(5, 11)
    assert p_pre is not p_ft
    assert p_pre.weight.data_ptr() != p_ft.weight.data_ptr()
    assert p_pre(torch.zeros(3, 5)).shape == (3, 7)
    assert p_ft(torch.zeros(3, 5)).shape == (3, 11)


def test_schedule_is_deterministic_and_each_epoch_covers_every_sample_once():
    a = runner.make_schedule(7, 3, 4, seed=666)
    b = runner.make_schedule(7, 3, 4, seed=666)
    assert [[x.tolist() for x in e] for e in a] == [[x.tolist() for x in e] for e in b]
    for epoch in a:
        flat = np.concatenate(epoch)
        assert np.array_equal(np.sort(flat), np.arange(7))
    ha, whole_a = runner.schedule_hashes(a, np.column_stack([np.zeros(7, dtype=np.int64), np.arange(7)]))
    hb, whole_b = runner.schedule_hashes(b, np.column_stack([np.zeros(7, dtype=np.int64), np.arange(7)]))
    assert ha == hb
    assert whole_a == whole_b


def test_scope_source_excludes_forbidden_extra_conditions_and_pretrained_teacher():
    source = RUNNER_PATH.read_text()
    assert 'KD_AGREE' not in source
    assert 'MI_PROTO' not in source
    assert 'T_pre' not in source
    assert 'WARMUP_EPOCHS' in source
    assert 'CosineAnnealingLR' in source
