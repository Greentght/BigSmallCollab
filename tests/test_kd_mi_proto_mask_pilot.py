import numpy as np
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.distill import run_kd_mi_proto_mask_pilot as pilot


def _logits(seed=0, n=4, c=3):
    return torch.randn(n, c, generator=torch.Generator().manual_seed(seed))


def test_mask_truth_table():
    agreement, rescue, keep = pilot.mask_from_predictions(
        [0, 1, 2, 0], [0, 2, 2, 1], [False, True, False, True])
    assert agreement.tolist() == [True, False, True, False]
    assert rescue.tolist() == [False, True, False, True]
    assert keep.tolist() == [True, True, True, True]
    _, rescue, keep = pilot.mask_from_predictions(
        [0, 1], [1, 0], [False, False])
    assert rescue.tolist() == [False, False]
    assert keep.tolist() == [False, False]


def test_prototype_loo_and_teacher_reliability():
    feats = np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    labels = np.asarray([0, 0, 1, 1])
    uid = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    pm = pilot._proto.prototype_metrics(feats, labels, uid, epsilon=1e-12)
    assert np.allclose(pm['z'], feats)
    assert np.allclose(pm['loo_proto_by_row'][0], [1., 0.])
    assert np.allclose(pm['full_proto'][1], [0., 1.])
    pred, support = pilot._prototype_support(pm, np.asarray([0, 0, 1, 1]), labels)
    assert pred.tolist() == [0, 0, 1, 1]
    assert np.all(support > 1e-12)
    reliable = (pred == np.asarray([0, 0, 1, 1])) & (support > 1e-12)
    assert reliable.all()


def test_prototype_singleton_fails_closed():
    with pytest.raises(ValueError, match='fewer than two'):
        pilot._proto.prototype_metrics(np.eye(3), np.asarray([0, 0, 1]), epsilon=1e-12)


def test_keep_all_matches_unmasked_losses_and_gradients():
    labels = torch.tensor([0, 1, 2, 1])
    teacher = _logits(1)
    a = _logits(2).requires_grad_()
    b = a.detach().clone().requires_grad_()
    keep = torch.ones(4, dtype=torch.bool)
    kd = pilot.masked_components(a, labels, teacher, keep, 'KD_PROTO')
    mi = pilot.masked_components(a, labels, teacher, keep, 'MI_PROTO')
    both = pilot.masked_components(a, labels, teacher, keep, 'KD_MI_PROTO')
    manual_kd = pilot.masked_components(b, labels, teacher, keep, 'KD_MI_PROTO')
    assert torch.allclose(kd['ce_loss'], F_ce(a, labels))
    assert torch.allclose(kd['kd_loss'], manual_kd['kd_loss'])
    assert torch.allclose(mi['mi_loss'], manual_kd['mi_loss'])
    assert torch.allclose(both['total'], manual_kd['total'])
    both['total'].backward(); manual_kd['total'].backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def F_ce(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def test_ce_is_full_batch_and_dropped_rows_have_ce_gradient():
    labels = torch.tensor([0, 1, 0])
    teacher = _logits(3, n=3, c=2)
    logits = _logits(4, n=3, c=2).requires_grad_()
    keep = torch.tensor([True, False, True])
    result = pilot.masked_components(logits, labels, teacher, keep, 'KD_PROTO')
    assert torch.allclose(result['ce_loss'], F_ce(logits, labels))
    result['ce_loss'].backward()
    assert torch.isfinite(logits.grad[keep]).all()
    assert torch.isfinite(logits.grad[~keep]).all()
    assert not torch.equal(logits.grad[~keep], torch.zeros_like(logits.grad[~keep]))


def test_kd_mask_uses_keep_mean_and_dropped_gradient_zero():
    labels = torch.tensor([0, 1, 0])
    teacher = _logits(5, n=3, c=2)
    logits = _logits(6, n=3, c=2).requires_grad_()
    keep = torch.tensor([True, False, True])
    result = pilot.masked_components(logits, labels, teacher, keep, 'KD_PROTO')
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(logits[keep] / 2., 1), torch.softmax(teacher[keep] / 2., 1),
        reduction='none').sum(1).mean()
    assert torch.allclose(result['kd_loss'], expected)
    result['kd_loss'].backward()
    assert torch.equal(logits.grad[~keep], torch.zeros_like(logits.grad[~keep]))
    assert torch.isfinite(logits.grad[keep]).all()


def test_mi_uses_selected_rows_and_class_joint(monkeypatch):
    seen=[]; original=pilot.probability_mi_loss
    def wrapped(pt, ps, eps=1e-8):
        seen.append((tuple(pt.shape), tuple(ps.shape)))
        return original(pt, ps, eps=eps)
    monkeypatch.setattr(pilot, 'probability_mi_loss', wrapped)
    labels=torch.tensor([0, 1, 2, 0]); teacher=_logits(7); logits=_logits(8).requires_grad_()
    keep=torch.tensor([True, False, True, False])
    out=pilot.masked_components(logits, labels, teacher, keep, 'MI_PROTO')
    out['mi_loss'].backward()
    assert seen == [((2, 3), (2, 3))]
    assert torch.equal(logits.grad[~keep], torch.zeros_like(logits.grad[~keep]))


def test_zero_keep_connected_zero_still_ce_step():
    model=nn.Linear(3, 2); opt=optim.SGD(model.parameters(), lr=.1)
    x=torch.randn(4, 3); labels=torch.tensor([0, 1, 0, 1]); logits=model(x)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    out=pilot.masked_components(logits, labels, torch.randn(4, 2), torch.zeros(4, dtype=torch.bool), 'KD_MI_PROTO')
    assert out['kd_loss'].requires_grad and out['mi_loss'].requires_grad
    (out['total']).backward(); opt.step()
    assert any(not torch.equal(before[k], v) for k,v in model.state_dict().items())
    assert torch.isfinite(logits.grad if logits.grad is not None else torch.tensor(0.)).all()


def test_one_keep_mi_is_connected_zero_but_kd_is_live():
    labels=torch.tensor([0, 1]); teacher=_logits(9, n=2, c=2); logits=_logits(10, n=2, c=2).requires_grad_()
    out=pilot.masked_components(logits, labels, teacher, torch.tensor([True, False]), 'KD_MI_PROTO')
    assert out['selected_count']==1 and out['mi_valid'] is False
    assert out['mi_loss'].requires_grad and float(out['mi_loss'].detach())==0.
    out['total'].backward(); assert torch.isfinite(logits.grad).all()


def test_teacher_logits_are_detached():
    labels=torch.tensor([0, 1, 0, 1]); teacher=_logits(11, n=4, c=2).requires_grad_(); logits=_logits(12, n=4, c=2).requires_grad_()
    out=pilot.masked_components(logits, labels, teacher, torch.ones(4, dtype=torch.bool), 'KD_MI_PROTO')
    out['total'].backward(); assert teacher.grad is None; assert torch.isfinite(logits.grad).all()


def test_uid_alignment_rejects_mismatch_and_reorders():
    uid=np.asarray([[0,0],[0,1],[0,2]], dtype=np.int64); y=np.asarray([0,1,0])
    other={'sample_uid':uid[[1,0,2]],'y':y[[1,0,2]],'logits':np.zeros((3,2)), 'feats':np.ones((3,4))}
    aligned=pilot._proto.align_by_uid({'sample_uid':uid,'y':y}, other)
    assert np.array_equal(aligned['sample_uid'], uid) and np.array_equal(aligned['y'], y)
    with pytest.raises(ValueError): pilot._proto.align_by_uid({'sample_uid':uid[:2],'y':y[:2]}, other)
    bad=dict(other); bad['y']=np.asarray([0,0,0]);
    with pytest.raises(ValueError): pilot._proto.align_by_uid({'sample_uid':uid,'y':y}, bad)


def test_conditions_are_exact_and_no_extra_condition():
    assert pilot.CONDITIONS == ('KD_PROTO', 'MI_PROTO', 'KD_MI_PROTO')
    assert set(pilot.CONDITIONS).isdisjoint({'KD_AGREE','MI_AGREE','KD_MI_AGREE','KD_AGREE_PROTO'})


def test_full_batch_gate_is_post_forward_and_no_feature_loss():
    source=Path(pilot.__file__).read_text()
    assert source.index('adapter.forward(model,xb.to(device))') < source.index('comp=masked_components')
    assert 'projection' not in source.lower()


def test_test_artifact_path_is_rejected():
    with pytest.raises(ValueError):
        pilot._proto.validate_train_path('0_666_test.npz')
