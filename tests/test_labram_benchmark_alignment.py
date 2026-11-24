"""Regression checks for benchmark-aligned LaBraM fine-tuning."""
import numpy as np
import pytest
import torch

import config
from experiments.finetune.run_labram_broadband_fewshot import build_optimizer, scheduler_for


@pytest.mark.parametrize('dataset', ['BNCI2014001', 'BNCI2014001-4', 'BNCI2014004',
                                    'BNCI2015001', 'AlexMI'])
@pytest.mark.parametrize('protocol', ['fewshot', 'loso'])
def test_benchmark_config_resolves(dataset, protocol):
    cfg = config.load_model_config('labram', dataset, protocol)
    assert (cfg['epochs'], cfg['lr'], cfg['batch_size'], cfg['weight_decay']) == (20, .001, 16, .01)
    assert (cfg['readout'], cfg['dropout_rate'], cfg['layer_decay']) == ('flatten', .5, 1.)
    assert cfg['label_smoothing'] == 0. and cfg['use_grad_clipping'] is False


def test_benchmark_optimizer_and_schedule():
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, .001, .01, 1.)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['weight_decay'] == .01
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in model.parameters()}
    steps = scheduler_for(20, 5, .001, 1e-6, 3)
    assert len(steps) == 60
    np.testing.assert_allclose(steps[:15], np.linspace(0., .001, 15))
    assert steps[15] == pytest.approx(.001)
    with pytest.raises(ValueError):
        build_optimizer(model, .001, .01, .9)
