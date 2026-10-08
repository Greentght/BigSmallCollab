"""State-isolated functional IFNet forwards and one-step AdamW lookahead.

This module is scoped to the sample utility pilot. The regular IFNet classifier
projects its weight with ``.data`` on every forward, including eval. Lookahead
uses an ordinary Linear calculation and carries the projection explicitly.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.autograd import Function


MAX_NORM = 0.5


class _SafeSqrt(Function):
    """Exact sqrt forward with the finite zero subgradient for second moments."""

    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.sqrt(value)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        positive = value > 0
        root = torch.sqrt(torch.where(positive, value, torch.ones_like(value)))
        derivative = torch.where(positive, 0.5 / root, torch.zeros_like(root))
        return grad_output * derivative


def project_classifier(weight: torch.Tensor) -> torch.Tensor:
    """Return the value IFNet's native LinearWithConstraint would install."""
    return torch.renorm(weight, p=2, dim=0, maxnorm=MAX_NORM)


def _project_train_start(params: Mapping[str, torch.Tensor]) -> OrderedDict:
    result = OrderedDict(params)
    # The native training forward mutates this value before autograd starts.
    # It is independent of controller parameters, so make the projected value
    # an ordinary leaf for the matching virtual training gradient.
    result['fc.weight'] = project_classifier(params['fc.weight']).detach().clone().requires_grad_(True)
    return result


@dataclass
class FunctionalState:
    params: OrderedDict
    buffers: OrderedDict


class FunctionalIFNet:
    """A pure view over IFNet's stem and constrained classifier.

    The copied stem supplies the module topology and train/eval flags. All
    parameter and buffer values come from the dictionaries passed to each call.
    BatchNorm writes therefore affect only the caller's copied buffer mapping.
    """

    def __init__(self, model: torch.nn.Module):
        self.template = copy.deepcopy(model)

    def snapshot(self, model: torch.nn.Module, *, requires_grad: bool = False) -> FunctionalState:
        params = OrderedDict()
        for name, value in model.named_parameters():
            cloned = value.detach().clone()
            cloned.requires_grad_(requires_grad and value.requires_grad)
            params[name] = cloned
        buffers = OrderedDict((name, value.detach().clone())
                              for name, value in model.named_buffers())
        return FunctionalState(params, buffers)

    def train_start(self, state: FunctionalState) -> FunctionalState:
        return FunctionalState(_project_train_start(state.params), state.buffers)

    def set_training(self, training: bool) -> None:
        self.template.stem.train(training)

    def forward(self, state: FunctionalState, x: torch.Tensor, *,
                training: bool, differentiable_projection: bool = False):
        self.set_training(training)
        stem_params = OrderedDict(
            (name[len('stem.'):], value)
            for name, value in state.params.items() if name.startswith('stem.'))
        stem_buffers = OrderedDict(
            (name[len('stem.'):], value)
            for name, value in state.buffers.items() if name.startswith('stem.'))
        features = functional_call(self.template.stem, (stem_params, stem_buffers), (x,))
        features = features.flatten(1)
        weight = state.params['fc.weight']
        if differentiable_projection:
            weight = project_classifier(weight)
        logits = F.linear(features, weight, state.params['fc.bias'])
        return features, logits


def optimizer_state_by_name(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Clone optimizer slots using stable model parameter names."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    result = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            name = names[parameter]
            slots = optimizer.state.get(parameter, {})
            result[name] = {
                key: (value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value))
                for key, value in slots.items()
            }
    return result


def _slot_tensor(slots, key, reference):
    value = slots.get(key)
    if value is None:
        return torch.zeros_like(reference)
    return value.to(device=reference.device, dtype=reference.dtype)


def functional_adamw_step(params: Mapping[str, torch.Tensor],
                          grads: Mapping[str, torch.Tensor | None],
                          slots: Mapping[str, Mapping[str, object]],
                          param_groups: list[dict]) -> tuple[OrderedDict, dict]:
    """Apply one differentiable AdamW step, retaining the graph through grads.

    This implements the non-capturable, non-fused AdamW path used by the pilot.
    Existing moments are detached state. Per-parameter ``step`` and AMSGrad
    state are retained in the returned copy; scheduler state is not advanced.
    """
    result = OrderedDict(params)
    next_slots = {name: dict(value) for name, value in slots.items()}
    owners = {}
    for group in param_groups:
        for name in group['_parameter_names']:
            owners[name] = group

    for name, parameter in params.items():
        grad = grads.get(name)
        if grad is None:
            continue
        group = owners[name]
        if group.get('maximize', False):
            grad = -grad
        if grad.is_sparse:
            raise RuntimeError('functional AdamW does not support sparse gradients')
        beta1, beta2 = group['betas']
        lr = float(group['lr'])
        eps = float(group['eps'])
        weight_decay = float(group['weight_decay'])
        old = slots.get(name, {})
        old_step = old.get('step', 0)
        step = int(old_step.item()) if torch.is_tensor(old_step) else int(old_step)
        step += 1
        exp_avg = _slot_tensor(old, 'exp_avg', parameter) * beta1 + grad * (1.0 - beta1)
        exp_avg_sq = _slot_tensor(old, 'exp_avg_sq', parameter) * beta2 + grad.square() * (1.0 - beta2)
        if group.get('amsgrad', False):
            max_old = _slot_tensor(old, 'max_exp_avg_sq', parameter)
            max_sq = torch.maximum(max_old, exp_avg_sq)
            denom_sq = max_sq
            next_slots.setdefault(name, {})['max_exp_avg_sq'] = max_sq.detach()
        else:
            denom_sq = exp_avg_sq
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denominator = _SafeSqrt.apply(denom_sq) / (bias_correction2 ** 0.5) + eps
        updated = parameter * (1.0 - lr * weight_decay)
        updated = updated - (lr / bias_correction1) * exp_avg / denominator
        result[name] = updated
        current = next_slots.setdefault(name, {})
        current.update({
            'step': torch.tensor(float(step), dtype=torch.float32),
            'exp_avg': exp_avg.detach(),
            'exp_avg_sq': exp_avg_sq.detach(),
        })
    return result, next_slots


def named_param_groups(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Add model parameter names to copied AdamW groups for functional steps."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        item = {key: copy.deepcopy(value) for key, value in group.items()
                if key != 'params'}
        item['_parameter_names'] = [names[parameter] for parameter in group['params']]
        groups.append(item)
    return groups


def functional_training_step(view: FunctionalIFNet, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             state: FunctionalState, x: torch.Tensor,
                             loss_fn, *, create_graph: bool):
    """Run an isolated train forward, gradient, and AdamW step."""
    train_state = view.train_start(state)
    _, logits = view.forward(train_state, x, training=True)
    loss = loss_fn(logits)
    names = list(train_state.params)
    values = [train_state.params[name] for name in names]
    gradients = torch.autograd.grad(loss, values, create_graph=create_graph,
                                    allow_unused=True)
    grad_map = dict(zip(names, gradients))
    stepped, slots = functional_adamw_step(
        train_state.params, grad_map, optimizer_state_by_name(model, optimizer),
        named_param_groups(model, optimizer))
    return FunctionalState(stepped, train_state.buffers), loss, logits, slots


def feedback_logits(view: FunctionalIFNet, state: FunctionalState, x: torch.Tensor):
    """Eval with the real max-norm value map and a live graph to updated params."""
    return view.forward(state, x, training=False,
                        differentiable_projection=True)[1]
