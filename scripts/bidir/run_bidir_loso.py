"""LOSO sample-level asymmetric bidirectional distillation: MIRepNet <-> IFNet.

Per fold, jointly train B (MIRepNet) + S (IFNet) on 8 subjects, evaluate both on
the held-out subject. Conditions:
  Uni   : forward routed only (lam_sb=0)     -- B teaches S on (B right, S wrong)
  Bidir : + reverse routed (lam_sb=0.1)      -- S also nudges B on (S right, B wrong)
Compare S's held-out acc to LOSO Base (S alone). Records S and B accuracy per fold.

    conda run -n mirepnet python scripts/bidir/run_bidir_loso.py --dataset BNCI2014001-4 --gpu 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

from collab.bidirectional import bidirectional_distill
import config
import data
from eval import metrics
from models import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014001-4')
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--lam_bs', type=float, default=1.0)
    p.add_argument('--lam_sb', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr_big', type=float, default=1e-4)
    p.add_argument('--conds', default='Uni,Bidir')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_csv', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    bcfg = config.load_model_config('mirepnet')
    scfg = config.load_model_config('ifnet')
    n_sub = {'BNCI2014004': 9, 'BNCI2014001-4': 9}[a.dataset]
    folds = a.folds if a.folds is not None else list(range(n_sub))
    seeds = a.seeds or dcfg['seeds']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    out_csv = a.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'results', 'metrics', f'{a.dataset}_loso_bidir_mirepnet_ifnet.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    conds = {'Uni': 0.0, 'Bidir': a.lam_sb}
    conds = {k: v for k, v in conds.items() if k in a.conds.split(',')}

    rows = []
    for seed in seeds:
        for f in folds:
            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, f)
            for cond, lam_sb in conds.items():
                bacfg = dict(bcfg); bacfg.update(
                    in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                    dataset_name=a.dataset, skip_preprocess=True)
                sacfg = dict(scfg); sacfg.update(
                    in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                    dataset_name=a.dataset)
                big_ad = get_adapter('mirepnet', device=device, **bacfg)
                small_ad = get_adapter('ifnet', device=device, **sacfg)
                s_pred, b_pred = bidirectional_distill(
                    big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, f,
                    lam_bs=a.lam_bs, lam_sb=lam_sb, epochs=a.epochs,
                    lr_big=a.lr_big, lr_small=scfg.get('lr', 1e-3),
                    weight_decay=scfg.get('weight_decay', 0.01),
                    batch_size=scfg.get('batch_size', 16), seed=seed)
                ms = metrics.evaluate(y_te, s_pred)
                mb = metrics.evaluate(y_te, b_pred)
                rows.append(dict(dataset=a.dataset, fold=f, seed=seed,
                                 condition=cond, S_acc=ms['acc'], S_kappa=ms['kappa'],
                                 B_acc=mb['acc'], B_kappa=mb['kappa']))
                print(f"fold{f} seed{seed} {cond} | S_acc={ms['acc']} "
                      f"B_acc={mb['acc']}", flush=True)
    if not rows:
        print('nothing to do'); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['S_acc', 'B_acc']].mean().round(3))


if __name__ == '__main__':
    main()
