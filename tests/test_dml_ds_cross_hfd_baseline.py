"""A disabled cross-operator term must leave the existing DML-DS step intact."""
import runpy
from pathlib import Path

import torch
import yaml

from experiments.bidir.run_dml_ds_cross_hfd_fewshot import make_models, state_digest


def test_disabled_cross_hfd_matches_baseline_step():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    old = runpy.run_path('tests/test_dml_ds_hfd_baseline.py')
    base = yaml.safe_load(Path('configs/experiments/fewshot_dml_ds_pairs.yaml').read_text())
    original = old['baseline_models'](base, 666)
    original_rng = torch.get_rng_state().clone()
    adapters, models, heads, hfd, optimizers = make_models(base, 666, torch.device('cpu'))
    try:
        assert torch.equal(original_rng, torch.get_rng_state())
        assert [state_digest(x) for x in original[1]] == [state_digest(x) for x in models]
        assert [state_digest(x) for x in original[2]] == [state_digest(x) for x in heads]
        generator = torch.Generator().manual_seed(987)
        inputs = [torch.randn(2, 3, 4, 200, generator=generator),
                  torch.randn(2, 3, 1000, generator=generator)]
        labels = torch.tensor([0, 1])
        torch.set_rng_state(original_rng)
        first = old['step'](base, *original, inputs, labels)
        torch.set_rng_state(original_rng)
        second = old['step'](base, adapters, models, heads, optimizers, inputs, labels)
        assert torch.equal(first, second)
        assert [state_digest(x) for x in original[1]] == [state_digest(x) for x in models]
        assert [state_digest(x) for x in original[2]] == [state_digest(x) for x in heads]
    finally:
        for head in original[2] + heads:
            head.close()
        hfd.close()
