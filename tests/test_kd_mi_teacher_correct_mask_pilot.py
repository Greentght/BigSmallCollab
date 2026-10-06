import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.distill import run_kd_mi_teacher_correct_mask_pilot as pilot


CFG = {
    "lam_kd": 0.5,
    "temperature_kd": 2.0,
    "lam_mi": 0.1,
    "mi_eps": 1e-8,
}


def _logits(seed=0, n=4, c=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g, dtype=torch.float32)


def test_keep_all_matches_unmasked_ce_kd_mi_and_gradients():
    labels = torch.tensor([0, 1, 2, 1])
    teacher = _logits(1)
    keep = torch.ones(4, dtype=torch.bool)
    a = _logits(2).requires_grad_()
    b = a.detach().clone().requires_grad_()
    masked = pilot.batch_loss_components(a, labels, teacher, keep,
                                         "KD_MI_TCORRECT_MASK", **CFG)
    all_loss = pilot.batch_loss_components(b, labels, teacher, keep,
                                           "KD_MI_ALL", **CFG)
    assert torch.allclose(masked["ce_loss"], all_loss["ce_loss"])
    assert torch.allclose(masked["kd_loss"], all_loss["kd_loss"])
    assert torch.allclose(masked["mi_loss"], all_loss["mi_loss"])
    assert torch.allclose(masked["total"], all_loss["total"])
    masked["total"].backward()
    all_loss["total"].backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def test_masked_wrong_logits_have_zero_direct_gradient():
    labels = torch.tensor([0, 1, 2, 1])
    teacher = _logits(3)
    keep = torch.tensor([True, False, True, False])
    logits = _logits(4).requires_grad_()
    result = pilot.batch_loss_components(logits, labels, teacher, keep,
                                         "KD_MI_TCORRECT_MASK", **CFG)
    result["total"].backward()
    assert torch.equal(logits.grad[~keep], torch.zeros_like(logits.grad[~keep]))
    assert torch.isfinite(logits.grad[keep]).all()


def test_ce_and_kd_are_means_over_keep_only():
    labels = torch.tensor([0, 1, 0])
    teacher = _logits(5, n=3, c=2)
    keep = torch.tensor([True, False, True])
    logits = _logits(6, n=3, c=2).requires_grad_()
    result = pilot.batch_loss_components(logits, labels, teacher, keep,
                                         "KD_MI_TCORRECT_MASK", **CFG)
    selected = logits[keep]
    expected_ce = torch.nn.functional.cross_entropy(selected, labels[keep])
    teacher_p = torch.softmax(teacher[keep] / 2.0, dim=1)
    expected_kd = torch.nn.functional.kl_div(
        torch.log_softmax(selected / 2.0, dim=1), teacher_p,
        reduction="none").sum(1).mean()
    assert torch.allclose(result["ce_loss"], expected_ce)
    assert torch.allclose(result["kd_loss"], expected_kd)


def test_masked_mi_uses_selected_rows_and_class_joint(monkeypatch):
    seen = []
    original = pilot.probability_mi_loss

    def wrapped(teacher_prob, student_prob, eps=1e-8):
        seen.append((tuple(teacher_prob.shape), tuple(student_prob.shape)))
        return original(teacher_prob, student_prob, eps=eps)

    monkeypatch.setattr(pilot, "probability_mi_loss", wrapped)
    labels = torch.tensor([0, 1, 2, 0])
    teacher = _logits(7)
    keep = torch.tensor([True, False, True, False])
    logits = _logits(8).requires_grad_()
    result = pilot.batch_loss_components(logits, labels, teacher, keep,
                                         "KD_MI_TCORRECT_MASK", **CFG)
    result["total"].backward()
    assert seen == [((2, 3), (2, 3))]
    pt = torch.softmax(teacher[keep], dim=1)
    ps = torch.softmax(logits.detach()[keep], dim=1)
    assert (pt.T @ ps).shape == (3, 3)


def test_one_selected_sample_has_graph_connected_zero_mi():
    labels = torch.tensor([0, 1])
    teacher = _logits(9, n=2, c=2)
    keep = torch.tensor([True, False])
    logits = _logits(10, n=2, c=2).requires_grad_()
    result = pilot.batch_loss_components(logits, labels, teacher, keep,
                                         "KD_MI_TCORRECT_MASK", **CFG)
    assert result["mi_valid"] is False
    assert result["mi_loss"].requires_grad
    assert float(result["mi_loss"].detach()) == 0.0
    result["total"].backward()
    assert torch.isfinite(logits.grad[keep]).all()
    assert torch.equal(logits.grad[~keep], torch.zeros_like(logits.grad[~keep]))


def test_all_masked_batch_forwards_but_does_not_step():
    class Adapter:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0
            self.batch_sizes = []

        def forward(self, model, x):
            self.calls += 1
            self.batch_sizes.append(int(x.shape[0]))
            return None, model(x)

    model = nn.Linear(3, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapter = Adapter()
    logits, result, stepped = pilot.forward_loss_step(
        adapter, model, optimizer, torch.randn(5, 3),
        torch.tensor([0, 1, 0, 1, 0]), torch.randn(5, 2),
        torch.zeros(5, dtype=torch.bool), "CE_TCORRECT_MASK", CFG)
    assert logits.shape == (5, 2)
    assert result is None and stepped is False
    assert adapter.calls == 1 and adapter.batch_sizes == [5]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_teacher_logits_are_detached():
    labels = torch.tensor([0, 1, 0, 1])
    teacher = _logits(11, n=4, c=2).requires_grad_()
    logits = _logits(12, n=4, c=2).requires_grad_()
    result = pilot.batch_loss_components(
        logits, labels, teacher, torch.ones(4, dtype=torch.bool),
        "KD_MI_ALL", **CFG)
    result["total"].backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all()


def test_uid_alignment_rejects_set_order_and_label_mismatch():
    uid = np.asarray([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    payload = {
        "sample_uid": uid.copy(), "y": labels.copy(),
        "logits": np.zeros((3, 2), dtype=np.float32),
        "split_policy": "fewshot_stratified_random",
    }
    logits, aligned_y, aligned_uid = pilot._align_teacher(payload, uid, labels, "ok")
    assert logits.shape == (3, 2)
    assert np.array_equal(aligned_y, labels)
    assert np.array_equal(aligned_uid, uid)
    with pytest.raises(ValueError, match="UID sets differ"):
        pilot._align_teacher(payload, uid[:2], labels[:2], "set")
    with pytest.raises(ValueError, match="labels differ"):
        pilot._align_teacher(payload, uid, np.asarray([1, 1, 0]), "labels")
    reordered = dict(payload, sample_uid=uid[[1, 0, 2]], y=labels[[1, 0, 2]])
    _, aligned_y, aligned_uid = pilot._align_teacher(reordered, uid, labels, "reorder")
    assert np.array_equal(aligned_y, labels)
    assert np.array_equal(aligned_uid, uid)


def test_teacher_loader_does_not_return_features(tmp_path):
    path = tmp_path / "0_666_train.npz"
    np.savez(path, logits=np.zeros((2, 2), dtype=np.float32),
             feats=np.ones((2, 7), dtype=np.float32), y=np.asarray([0, 1]),
             sample_uid=np.asarray([[0, 0], [0, 1]], dtype=np.int64),
             split_policy=np.asarray("fewshot_stratified_random"))
    payload = pilot._load_teacher_logits_only(path)
    assert set(payload) == {"logits", "y", "sample_uid", "split_policy"}
    assert "feats" not in payload

