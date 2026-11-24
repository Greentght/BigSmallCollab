"""Check DCM forward and gradient flow on source-only BNCI2014001-4 trials."""
import os

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json

import numpy as np
import torch

import config
from collab.dcm import DCMHeads, mutual_losses
from data.loso_source import load_dataset
from experiments.distill.mirepnet_t2s_data import ensure_uid_alignment
from experiments.finetune.run_neurogpt_loso import _adapter_for, _load_spec, DEFAULT_CONFIG
from models import get_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True, choices=('cbramod_ifnet', 'neurogpt_eegnet'))
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    dataset = 'BNCI2014001-4'
    big_name, small_name = args.pair.split('_')
    big_data = load_dataset(dataset, 'cbramod' if big_name == 'cbramod' else None)
    small_data = load_dataset(dataset, small_name)
    big_data, small_data = ensure_uid_alignment(big_data, small_data)
    rows = np.flatnonzero(np.asarray(small_data.subjects) != 0)[:4]
    if big_name == 'neurogpt':
        exp, spec, _ = _load_spec(DEFAULT_CONFIG, dataset)
        big_adapter, _ = _adapter_for(dataset, 22, device, exp, spec)
    else:
        big_cfg = config.load_model_config(big_name, dataset, 'loso')
        big_cfg.update(in_channels=22, skip_preprocess=big_data.input_manifest_path is not None)
        big_adapter = get_adapter(big_name, device=device, dataset_name=dataset, **big_cfg)
    small_cfg = config.load_model_config(small_name, dataset, 'loso')
    small_cfg.update(in_channels=22, samples=1000)
    small_adapter = get_adapter(small_name, device=device, dataset_name=dataset, **small_cfg)
    adapters = [big_adapter, small_adapter]
    views = [big_data, small_data]
    inputs = [adapter.preprocess(np.asarray(view.x[rows])).to(device)
              for adapter, view in zip(adapters, views)]
    labels = torch.as_tensor(small_data.y[rows], dtype=torch.long, device=device)
    for condition in ('dml_ds', 'dcm'):
        models = [adapter.build(4) for adapter in adapters]
        heads = [DCMHeads(model, name, 4).to(device)
                 for model, name in zip(models, (big_name, small_name))]
        optimizers = [torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-4)
                      for model, head in zip(models, heads)]
        for step in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            main_logits = [adapter.forward(model, x)[1] for adapter, model, x in zip(adapters, models, inputs)]
            outputs = [[logits] + head(len(labels)) for logits, head in zip(main_logits, heads)]
            losses = mutual_losses(outputs, labels, condition)
            sum(losses).backward()
            for model, head in zip(models, heads):
                for module in (model, head):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise RuntimeError('missing or non-finite gradients')
            for optimizer in optimizers:
                optimizer.step()
        print(json.dumps({'pair': args.pair, 'condition': condition, 'steps': 2,
                          'input_shapes': [list(x.shape) for x in inputs],
                          'output_shapes': [[list(z.shape) for z in side] for side in outputs],
                          'losses': [float(x.detach()) for x in losses],
                          'peak_bytes': torch.cuda.max_memory_allocated(device),
                          'source_manifest_sha256': small_data.source_manifest_sha256}), flush=True)
        for head in heads:
            head.close()


if __name__ == '__main__':
    main()
