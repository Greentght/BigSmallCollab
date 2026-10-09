"""Focused checks for the sample utility pilot's gradients and state isolation."""
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from collab.lookahead import (FunctionalIFNet, feedback_logits,
                              functional_adamw_step, functional_training_step,
                              named_param_groups, optimizer_state_by_name)
from collab.sample_utility import (SampleUtilityMLP, bernoulli_policy_loss,
                                  build_controller_input, student_loss)
from experiments.distill.sample_utility_splits import (align_replay_by_uid,
                                                       batchwise_shuffle, fold_masks)
from models.ifnet.ifnet import IFNet


def _model_hash(model):
    digest = 0
    for value in model.state_dict().values():
        tensor = value.detach().cpu().contiguous()
        digest = hash((digest, str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes()))
    return digest


def _optimizer_copy_hash(optimizer):
    pieces = []
    for parameter, state in optimizer.state.items():
        pieces.append((id(parameter), tuple(sorted(
            (name, value.detach().cpu().clone() if torch.is_tensor(value) else value)
            for name, value in state.items()))))
    return repr(pieces)


def test_fixed_next_subject_feedback_split_has_expected_counts():
    counts = [160, 120, 160, 160, 160, 160, 160, 160, 160]
    subject_ids = np.repeat(np.arange(9), counts)
    expected = [(1120, 120, 160), (1120, 160, 120)] + [(1080, 160, 160)] * 7
    for fold, (train_n, feedback_n, test_n) in enumerate(expected):
        feedback, masks = fold_masks(subject_ids, fold)
        assert feedback == (fold + 1) % 9
        assert (int(masks['train'].sum()), int(masks['feedback'].sum()),
                int(masks['test'].sum())) == (train_n, feedback_n, test_n)
        assert not (masks['train'] & masks['feedback']).any()
        assert not (masks['train'] & masks['test']).any()
        assert not (masks['feedback'] & masks['test']).any()
        assert (masks['train'] | masks['feedback'] | masks['test']).all()


def test_controller_inputs_detach_and_include_progress():
    teacher = torch.randn(3, 4, requires_grad=True)
    reference = torch.randn(3, 6, requires_grad=True)
    teacher_p = torch.softmax(torch.randn(3, 2), dim=1)
    current_p = torch.softmax(torch.randn(3, 2, requires_grad=True), dim=1)
    state = build_controller_input(teacher, reference, teacher_p, current_p, 0.25)
    assert state.shape == (3, 15)
    assert not state.requires_grad
    assert torch.allclose(state[:, -1], torch.full((3,), 0.25))
    controller = SampleUtilityMLP(15)
    value = controller(state)
    assert torch.all((value >= 0.05) & (value <= 0.95))
    with pytest.raises(ValueError, match='detached'):
        controller(state.requires_grad_())


def test_weighted_kd_matches_batchmean_and_has_meta_gradient():
    torch.manual_seed(12)
    student_logits = torch.randn(5, 2, requires_grad=True)
    teacher_logits = torch.randn(5, 2)
    labels = torch.tensor([0, 1, 1, 0, 1])
    weights = nn.Parameter(torch.full((5,), 0.5))
    total, parts = student_loss(student_logits, labels, teacher_logits, weights)
    old_kd = F.kl_div(F.log_softmax(student_logits / 2.0, dim=1),
                      F.softmax(teacher_logits / 2.0, dim=1),
                      reduction='batchmean') * 4.0
    assert torch.allclose(parts['kd'], old_kd * 0.5, atol=1e-7, rtol=1e-6)
    assert torch.allclose(parts['ce'], F.cross_entropy(student_logits, labels))
    grad = torch.autograd.grad(total, weights)[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() == len(weights)
    ce_only, _ = student_loss(student_logits, labels, teacher_logits,
                              torch.zeros(5))
    assert torch.allclose(ce_only, F.cross_entropy(student_logits, labels))


def test_policy_score_function_gradient_is_nonzero_and_reward_detached():
    logits = nn.Parameter(torch.tensor([-1.0, 0.8]))
    probabilities = torch.sigmoid(logits)
    actions = torch.tensor([1.0, 0.0])
    reward = torch.tensor(0.7, requires_grad=True)
    baseline = torch.tensor(0.2, requires_grad=True)
    loss = bernoulli_policy_loss(probabilities, actions, reward, baseline)
    grad = torch.autograd.grad(loss, (logits, reward, baseline), allow_unused=True)
    assert torch.isfinite(grad[0]).all()
    assert torch.count_nonzero(grad[0]).item() == 2
    assert grad[1] is None and grad[2] is None


def test_functional_adamw_matches_native_optimizer_with_existing_moments():
    torch.manual_seed(91)
    native = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    virtual = nn.Linear(4, 3, bias=False, dtype=torch.float64)
    virtual.load_state_dict(native.state_dict())
    opt_native = torch.optim.AdamW(native.parameters(), lr=0.003,
                                   weight_decay=0.02, foreach=False, fused=False)
    opt_virtual = torch.optim.AdamW(virtual.parameters(), lr=0.003,
                                    weight_decay=0.02, foreach=False, fused=False)
    for _ in range(2):
        fixed_grad = torch.randn_like(native.weight)
        native.weight.grad = fixed_grad.clone()
        virtual.weight.grad = fixed_grad.clone()
        opt_native.step()
        opt_virtual.step()
        opt_native.zero_grad(set_to_none=True)
        opt_virtual.zero_grad(set_to_none=True)

    fixed_grad = torch.randn_like(native.weight)
    native.weight.grad = fixed_grad.clone()
    initial_params = OrderedDict((name, value.detach().clone())
                                 for name, value in virtual.named_parameters())
    slots = optimizer_state_by_name(virtual, opt_virtual)
    groups = named_param_groups(virtual, opt_virtual)
    native_before = native.weight.detach().clone()
    opt_native.step()
    stepped, next_slots = functional_adamw_step(
        initial_params, {'weight': fixed_grad}, slots, groups)
    assert torch.allclose(stepped['weight'], native.weight, atol=1e-7, rtol=1e-6)
    for key in ('exp_avg', 'exp_avg_sq'):
        assert torch.allclose(next_slots['weight'][key], opt_native.state[native.weight][key],
                              atol=1e-7, rtol=1e-6)
    assert int(next_slots['weight']['step']) == int(opt_native.state[native.weight]['step'])
    assert not torch.equal(native_before, native.weight)


def test_ifnet_lookahead_reaches_controller_and_preserves_real_state():
    torch.manual_seed(17)
    model = IFNet(in_channels=3, samples=1000, num_classes=2,
                  use_filter_bank=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                  foreach=False, fused=False)
    view = FunctionalIFNet(model)
    before_model = _model_hash(model)
    before_optimizer = _optimizer_copy_hash(optimizer)
    x = torch.randn(2, 6, 1000)
    y = torch.tensor([0, 1])
    teacher_logits = torch.randn(2, 2)
    controller = SampleUtilityMLP(5)
    inputs = torch.randn(2, 5)
    weights = controller(inputs)
    before_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[], enabled=True):
        functional_state = view.snapshot(model, requires_grad=True)
        def objective(logits):
            return student_loss(logits, y, teacher_logits, weights)[0]
        stepped, _, _, _ = functional_training_step(
            view, model, optimizer, functional_state, x, objective,
            create_graph=True)
        feedback = F.cross_entropy(feedback_logits(view, stepped, x), y)
        gradients = torch.autograd.grad(feedback, tuple(controller.parameters()),
                                        allow_unused=False)
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert sum(float(grad.abs().sum()) for grad in gradients) > 0.0
    assert _model_hash(model) == before_model
    assert _optimizer_copy_hash(optimizer) == before_optimizer
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_batch_shuffle_preserves_each_batch_multiset():
    values = np.asarray([0.05, 0.25, 0.75, 0.75, 0.95, 0.5, 0.4])
    shuffled = batchwise_shuffle(values, [3, 3, 1], np.random.default_rng(37))
    offsets = [0, 3, 6, 7]
    for start, stop in zip(offsets, offsets[1:]):
        assert np.array_equal(np.sort(values[start:stop]), np.sort(shuffled[start:stop]))
    assert np.array_equal(np.sort(values), np.sort(shuffled))
    assert float(values.sum()) == float(shuffled.sum())


def test_replay_arrays_are_reordered_by_uid_and_reject_ambiguous_identity():
    expected_uids = np.asarray([[0, 10], [1, 21], [2, 32]], dtype=np.int64)
    replay_uids = expected_uids[[2, 0, 1]]
    replay_weights = np.asarray([0.8, 0.2, 0.5], dtype=np.float32)
    aligned = align_replay_by_uid(
        replay_uids, expected_uids, weights=replay_weights,
        actions=np.asarray([1, 0, 1], dtype=np.float32))
    assert np.array_equal(aligned['weights'], np.asarray([0.2, 0.5, 0.8], dtype=np.float32))
    assert np.array_equal(aligned['actions'], np.asarray([0, 1, 1], dtype=np.float32))
    with pytest.raises(ValueError, match='unique'):
        align_replay_by_uid(replay_uids[[0, 0, 2]], expected_uids,
                            weights=replay_weights)
    with pytest.raises(ValueError, match='UID set'):
        align_replay_by_uid(replay_uids, expected_uids + 100,
                            weights=replay_weights)
