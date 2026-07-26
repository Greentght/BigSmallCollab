"""BD-EEG LOSO: MIRepNet<->IFNet difference-aware direction-asymmetric bidirectional
distillation. Per fold: shared CE warm-up -> fork into groups
(G0_CE/G1_FixKD/SymDML/StrictRouted/BD_EEG/BD_EEG_Swap). Records per-(group,fold,
seed) S_acc/B_acc (+kappa) and per-epoch routing diagnostics. Primary metric =
small model (S) held-out accuracy; secondary = big model (B).

    conda run -n mirepnet python scripts/run_bdeeg_loso.py --dataset BNCI2014001-4 --gpu 8
"""
import argparse
import os
import sys

# shared 128-core box: cap thread pools BEFORE importing torch/numpy so we don't
# oversubscribe cores (see memory cap-cpu-usage). Overridable via env.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))

from collab.bdeeg import bd_eeg_fold, GROUPS
import config
import data
from eval import metrics
from models import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014001-4')
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--groups', default=None, help='comma subset of GROUPS keys')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--total', type=int, default=100)
    p.add_argument('--lam_bs', type=float, default=1.0)
    p.add_argument('--lam_sb', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--rho', type=float, default=0.5)
    p.add_argument('--lr_big', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--tag', default='bdeeg')
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
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    groups = ({k: GROUPS[k] for k in a.groups.split(',')} if a.groups else GROUPS)

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'metrics')
    out_csv = f'{root}/{a.dataset}_loso_{a.tag}_mirepnet_ifnet.csv'
    diag_csv = f'{root}/{a.dataset}_loso_{a.tag}_diag.csv'

    rows, diags = [], []
    for seed in seeds:
        for f in folds:
            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, f)
            bacfg = dict(bcfg); bacfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                             dataset_name=a.dataset, skip_preprocess=True)
            sacfg = dict(scfg); sacfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                             dataset_name=a.dataset)
            big_ad = get_adapter('mirepnet', device=device, **bacfg)
            small_ad = get_adapter('ifnet', device=device, **sacfg)
            res, diag = bd_eeg_fold(
                big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, f,
                groups=groups, warmup=a.warmup, total=a.total,
                lam_bs=a.lam_bs, lam_sb=a.lam_sb, gamma=a.gamma, rho=a.rho,
                lr_big=a.lr_big, lr_small=scfg.get('lr', 1e-3),
                wd=scfg.get('weight_decay', 0.01),
                bs=scfg.get('batch_size', 16), seed=seed)
            for gp, (sp, bp) in res.items():
                ms = metrics.evaluate(y_te, sp); mb = metrics.evaluate(y_te, bp)
                rows.append(dict(dataset=a.dataset, fold=f, seed=seed, group=gp,
                                 S_acc=ms['acc'], S_kappa=ms['kappa'],
                                 B_acc=mb['acc'], B_kappa=mb['kappa']))
                print(f"fold{f} seed{seed} {gp} | S={ms['acc']} B={mb['acc']}", flush=True)
            for d in diag:
                d.update(dataset=a.dataset, fold=f, seed=seed); diags.append(d)
            pd.DataFrame(rows).to_csv(out_csv, index=False)      # incremental save
            pd.DataFrame(diags).to_csv(diag_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(pd.DataFrame(rows).groupby('group')[['S_acc', 'B_acc']].mean().round(3))


if __name__ == '__main__':
    main()
