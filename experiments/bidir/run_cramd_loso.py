"""CR-AMD LOSO: MIRepNet<->IFNet, groups G0/G1/G3/G5/G6. Per fold: shared warm-up
then fork. Records per-(group,fold,seed) S_acc/B_acc, plus per-epoch train-quadrant
diagnostics. Primary metric = small model (S) held-out accuracy.

    conda run -n mirepnet python experiments/bidir/run_cramd_loso.py --dataset BNCI2014001-4 --gpu 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch

from collab.mutual import cr_amd_fold, GROUPS
import config
import data
from eval import metrics
from models import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014001-4')
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--groups', default=None, help='comma subset of G0_CE,G1_FixKD,G3_SymDML,G5_Routed,G6_CRAMD')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--total', type=int, default=100)
    p.add_argument('--lam_bs', type=float, default=0.5)
    p.add_argument('--lam_sb', type=float, default=0.1)
    p.add_argument('--lr_big', type=float, default=1e-4)
    p.add_argument('--lam_feat', type=float, default=0.5)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--tag', default='')
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

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        '/data1/llx/BigSmallCollab_results', 'metrics')
    tag = a.tag or 'cramd'
    out_csv = f'{root}/{a.dataset}_loso_{tag}_mirepnet_ifnet.csv'
    diag_csv = f'{root}/{a.dataset}_loso_{tag}_diag.csv'

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
            res, diag = cr_amd_fold(
                big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, f,
                groups=groups, warmup=a.warmup, total=a.total,
                lam_bs=a.lam_bs, lam_sb=a.lam_sb, lam_feat=a.lam_feat, lr_big=a.lr_big,
                lr_small=scfg.get('lr', 1e-3), wd=scfg.get('weight_decay', 0.01),
                bs=scfg.get('batch_size', 16), seed=seed)
            for g, (sp, bp) in res.items():
                ms = metrics.evaluate(y_te, sp); mb = metrics.evaluate(y_te, bp)
                rows.append(dict(dataset=a.dataset, fold=f, seed=seed, group=g,
                                 S_acc=ms['acc'], S_kappa=ms['kappa'],
                                 B_acc=mb['acc'], B_kappa=mb['kappa']))
                print(f"fold{f} seed{seed} {g} | S={ms['acc']} B={mb['acc']}", flush=True)
            for d in diag:
                d.update(dataset=a.dataset, fold=f, seed=seed); diags.append(d)
            pd.DataFrame(rows).to_csv(out_csv, index=False)      # incremental save
            pd.DataFrame(diags).to_csv(diag_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(pd.DataFrame(rows).groupby('group')[['S_acc', 'B_acc']].mean().round(3))


if __name__ == '__main__':
    main()
