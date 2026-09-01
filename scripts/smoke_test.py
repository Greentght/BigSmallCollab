"""Smoke test: canonical data split + adapter forward contract.

Run inside the `mirepnet` env (covers ifnet/eegnet/adfcnn/mirepnet):
    conda run -n mirepnet python scripts/smoke_test.py --models ifnet mirepnet

Validates, for one subject/seed:
  1. core.data.subject_split returns aligned (N,C,1000) splits.
  2. each adapter builds, preprocesses, and forwards -> (feat, logits) of the
     right shapes, and infer() runs end to end.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config
import data
from models import get_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='BNCI2014004')
    ap.add_argument('--subject', type=int, default=1)
    ap.add_argument('--seed', type=int, default=666)
    ap.add_argument('--models', nargs='+', default=['ifnet', 'mirepnet'])
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()

    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    info = config.load_dataset_config(a.dataset)
    nc = info['num_classes']

    X_tr, y_tr, X_te, y_te = data.subject_split(a.dataset, a.subject, seed=a.seed)
    print(f'[data] {a.dataset} S{a.subject} seed{a.seed}: '
          f'train {X_tr.shape} test {X_te.shape} classes={nc}')
    assert X_tr.shape[1] == info['channels'], 'native channel mismatch'

    for name in a.models:
        cfg = config.load_model_config(name)
        cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                   dataset_name=a.dataset, epochs=1, batch_size=8)
        ad = get_adapter(name, device=device, **cfg)
        model = ad.build(nc)
        xb = ad.preprocess(X_tr[:4]).to(device)
        feat, logits = ad.forward(model, xb)
        print(f'[{name}] input {tuple(xb.shape)} -> feat {tuple(feat.shape)} '
              f'logits {tuple(logits.shape)}')
        assert logits.shape[1] == nc, f'{name} logits class dim != {nc}'
        feats, logs = ad.infer(model, X_te[:8])
        print(f'[{name}] infer -> feats {feats.shape} logits {logs.shape}')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
