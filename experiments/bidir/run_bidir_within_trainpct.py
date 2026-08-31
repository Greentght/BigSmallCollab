"""Within-subject percentage-split bidirectional distillation.

This is the low-resource finetune protocol, not K-shot sampling:
  * load the selected downstream session through data.subject_split;
  * use val_split=0.7 by default, i.e. 30% train / 70% test;
  * train MIRepNet(B) and IFNet(S) end-to-end on the full 30% train split;
  * evaluate on the full 70% held-out split.

Methods:
  cramd : G0_CE/G1_FixKD/G3_SymDML/G5_Routed/G6_CRAMD/G7_DisagCE
  bdeeg : G0_CE/G1_FixKD/SymDML/StrictRouted/BD_EEG/BD_EEG_Swap

Example:
    conda run -n mirepnet python -u experiments/bidir/run_bidir_within_trainpct.py \
        --datasets BNCI2014004 BNCI2014001-4 --methods cramd bdeeg \
        --val_split 0.7 --gpu 0 --tag train30_v1
"""
import argparse
import os
import sys

# Shared server: cap thread pools before importing torch/numpy unless overridden.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))

from collab.mutual import cr_amd_fold, GROUPS as CRAMD_GROUPS
from collab.bdeeg import bd_eeg_fold, GROUPS as BDEEG_GROUPS
import config
import data
from eval import metrics
from models import get_adapter


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(ROOT, 'results', 'metrics')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['BNCI2014004'])
    p.add_argument('--methods', nargs='+', default=['cramd', 'bdeeg'],
                   choices=['cramd', 'bdeeg'])
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--val_split', type=float, default=0.7,
                   help='test fraction: 0.7 means 30% train / 70% test')
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--total', type=int, default=60)
    p.add_argument('--lam_bs', type=float, default=1.0)
    p.add_argument('--lam_sb_cramd', type=float, default=0.1)
    p.add_argument('--lam_sb_bdeeg', type=float, default=0.25)
    p.add_argument('--alpha_ce', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--rho', type=float, default=0.5)
    p.add_argument('--lr_big', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--tag', default='train30_v1')
    return p.parse_args()


def _groups_for(method):
    if method == 'cramd':
        keys = ['G0_CE', 'G1_FixKD', 'G3_SymDML', 'G5_Routed',
                'G6_CRAMD', 'G7_DisagCE']
        return {k: CRAMD_GROUPS[k] for k in keys}
    if method == 'bdeeg':
        keys = ['G0_CE', 'G1_FixKD', 'SymDML', 'StrictRouted',
                'BD_EEG', 'BD_EEG_Swap']
        return {k: BDEEG_GROUPS[k] for k in keys}
    raise ValueError(method)


def _run_method(method, big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, subj,
                a, scfg):
    if method == 'cramd':
        return cr_amd_fold(
            big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, subj,
            groups=_groups_for(method), warmup=a.warmup, total=a.total,
            lam_bs=a.lam_bs, lam_sb=a.lam_sb_cramd, alpha_ce=a.alpha_ce,
            lr_big=a.lr_big, lr_small=scfg.get('lr', 1e-3),
            wd=scfg.get('weight_decay', 0.01),
            bs=scfg.get('batch_size', 16), seed=a.current_seed)
    if method == 'bdeeg':
        return bd_eeg_fold(
            big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, subj,
            groups=_groups_for(method), warmup=a.warmup, total=a.total,
            lam_bs=a.lam_bs, lam_sb=a.lam_sb_bdeeg,
            gamma=a.gamma, rho=a.rho,
            lr_big=a.lr_big, lr_small=scfg.get('lr', 1e-3),
            wd=scfg.get('weight_decay', 0.01),
            bs=scfg.get('batch_size', 16), seed=a.current_seed)
    raise ValueError(method)


def main():
    a = parse_args()
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    os.makedirs(METRICS, exist_ok=True)
    out_csv = os.path.join(METRICS, f'bidir_within_{a.tag}_mirepnet_ifnet.csv')
    diag_csv = os.path.join(METRICS, f'bidir_within_{a.tag}_diag.csv')

    rows, diags = [], []
    for ds in a.datasets:
        dcfg = config.load_dataset_config(ds)
        bcfg = config.load_model_config('mirepnet')
        scfg = config.load_model_config('ifnet')
        nc = dcfg['num_classes']
        subjects = a.subjects if a.subjects is not None else list(range(dcfg['num_subjects']))
        seeds = a.seeds or dcfg['seeds']

        for seed in seeds:
            a.current_seed = seed
            for subj in subjects:
                X_tr, y_tr, X_te, y_te = data.subject_split(
                    ds, subj, val_split=a.val_split, seed=seed)
                subj_tr = np.full(len(y_tr), subj, dtype=np.int64)
                train_pct = round((1.0 - a.val_split) * 100)
                test_pct = round(a.val_split * 100)

                for method in a.methods:
                    bacfg = dict(bcfg)
                    bacfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                 dataset_name=ds, skip_preprocess=True)
                    sacfg = dict(scfg)
                    sacfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                 dataset_name=ds)
                    big_ad = get_adapter('mirepnet', device=device, **bacfg)
                    small_ad = get_adapter('ifnet', device=device, **sacfg)
                    res, diag = _run_method(method, big_ad, small_ad, nc,
                                            X_tr, y_tr, subj_tr, X_te, subj,
                                            a, scfg)
                    for group, (sp, bp) in res.items():
                        ms = metrics.evaluate(y_te, sp)
                        mb = metrics.evaluate(y_te, bp)
                        rows.append(dict(
                            dataset=ds, method=method, group=group,
                            split=f'{train_pct}tr_{test_pct}te',
                            val_split=a.val_split, train_pct=train_pct,
                            test_pct=test_pct, subject=subj, seed=seed,
                            n_train=len(y_tr), n_test=len(y_te),
                            S_acc=ms['acc'], S_kappa=ms['kappa'],
                            B_acc=mb['acc'], B_kappa=mb['kappa'],
                            best_acc=max(ms['acc'], mb['acc'])))
                        print(f"{ds} {method} {train_pct}/{test_pct} "
                              f"S{subj} seed{seed} {group} | "
                              f"S={ms['acc']} B={mb['acc']} "
                              f"best={max(ms['acc'], mb['acc'])}",
                              flush=True)
                    for d in diag:
                        d.update(dataset=ds, method=method, split=f'{train_pct}tr_{test_pct}te',
                                 val_split=a.val_split, subject=subj, seed=seed,
                                 n_train=len(y_tr), n_test=len(y_te))
                        diags.append(d)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    pd.DataFrame(diags).to_csv(diag_csv, index=False)

                    if device != 'cpu':
                        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pd.DataFrame(diags).to_csv(diag_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby(['dataset', 'method', 'group'])[['S_acc', 'B_acc', 'best_acc']]
          .mean().round(3))


if __name__ == '__main__':
    main()
