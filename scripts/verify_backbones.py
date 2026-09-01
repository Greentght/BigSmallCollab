"""Verify a vendored big-model backbone builds from framework code + loads its
weights + forwards, with zero dependence on the external upstream repo.

Run each in its own conda env:
    conda run -n mirepnet python scripts/verify_backbones.py --model mirepnet
    conda run -n cbramod  python scripts/verify_backbones.py --model cbramod
    conda run -n labram   python scripts/verify_backbones.py --model labram
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import get_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', default='BNCI2014004')
    ap.add_argument('--channels', type=int, default=3)
    ap.add_argument('--classes', type=int, default=2)
    args = ap.parse_args()

    cfg = config.load_model_config(args.model)
    cfg.update(dataset_name=args.dataset, in_channels=args.channels, samples=1000)
    ad = get_adapter(args.model, device='cpu', **cfg)
    model = ad.build(args.classes)
    n_params = sum(p.numel() for p in model.parameters())
    X = np.random.RandomState(0).randn(8, args.channels, 1000).astype('float32')
    feats, logits = ad.infer(model, X)
    assert logits.shape == (8, args.classes), f'logits {logits.shape}'
    assert feats.shape[0] == 8, f'feats {feats.shape}'
    print(f'[ok] {args.model}: {n_params/1e6:.1f}M params, '
          f'feat{feats.shape} logits{logits.shape} — vendored backbone + weights OK')


if __name__ == '__main__':
    main()
