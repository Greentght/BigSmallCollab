"""A disabled HFD term must reproduce the existing DML-DS training step."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

import config
from collab.dcm import mutual_losses
from collab.dml_ds_heads import build_heads
from experiments.bidir.run_dml_ds_hfd_fewshot import make_models, state_digest
from models import get_adapter


def baseline_models(base, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    big_cfg = config.load_model_config('cbramod', 'BNCI2014004', 'fewshot')
    big_cfg.update(in_channels=3, samples=1000, skip_preprocess=False)
    small_cfg = config.load_model_config('eegnet', 'BNCI2014004', 'fewshot')
    small_cfg.update(in_channels=3, samples=1000)
    adapters = [get_adapter(name, device='cpu', dataset_name='BNCI2014004', **settings)
                for name, settings in (('cbramod', big_cfg), ('eegnet', small_cfg))]
    models = [adapter.build(2) for adapter in adapters]
    heads = [build_heads(model, name, 2).to('cpu')
             for model, name in zip(models, ('cbramod', 'eegnet'))]
    optimizers = []
    for model, head, training in zip(models, heads,
            (base['teacher_training']['cbramod'], base['student_training']['eegnet'])):
        optimizers.append(torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=float(training['lr']), weight_decay=float(training['weight_decay'])))
    return adapters, models, heads, optimizers


def step(base, adapters, models, heads, optimizers, inputs, labels):
    for opt in optimizers: opt.zero_grad(set_to_none=True)
    logits = [adapter.forward(model, x)[1] for adapter, model, x in zip(adapters, models, inputs)]
    outputs = [[z] + heads[j](len(labels)) for j, z in enumerate(logits)]
    loss = sum(mutual_losses(outputs, labels, 'dml_ds', base['temperature'],
                             base['alpha'], base['lambda']))
    loss.backward()
    for opt in optimizers:
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
        opt.step()
    return loss.detach()


def test_disabled_hfd_preserves_initial_rng_and_dml_ds_step():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    base = yaml.safe_load(Path('configs/experiments/fewshot_dml_ds_pairs.yaml').read_text())
    original = baseline_models(base, 666)
    original_rng = torch.get_rng_state().clone()
    adapters, models, heads, hfd, optimizers = make_models(base, 666, torch.device('cpu'))
    assert torch.equal(original_rng, torch.get_rng_state())
    assert [state_digest(x) for x in original[1]] == [state_digest(x) for x in models]
    assert [state_digest(x) for x in original[2]] == [state_digest(x) for x in heads]
    generator = torch.Generator().manual_seed(987)
    inputs = [torch.randn(2, 3, 4, 200, generator=generator),
              torch.randn(2, 3, 1000, generator=generator)]
    labels = torch.tensor([0, 1])
    torch.set_rng_state(original_rng)
    first_loss = step(base, *original, inputs, labels)
    torch.set_rng_state(original_rng)
    second_loss = step(base, adapters, models, heads, optimizers, inputs, labels)
    assert torch.equal(first_loss, second_loss)
    assert [state_digest(x) for x in original[1]] == [state_digest(x) for x in models]
    assert [state_digest(x) for x in original[2]] == [state_digest(x) for x in heads]
    for head in original[2] + heads: head.close()
    hfd.close()
