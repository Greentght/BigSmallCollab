"""Few-shot WITHIN-SUBJECT margin-routed asymmetric mutual distillation:
MIRepNet(B) <-> IFNet(S), on the 2-class BNCI2014004.

Motivation (user): 2-class MI has no rich non-target dark knowledge, so a
recommender-style rank-discrepancy loss does not transfer. What DOES survive is
"complementary-sample bidirectional collaboration": P(B wrong, S correct)~16%
means the small model can still correct the big one on some samples, so the
knowledge direction must be decided PER SAMPLE. Router = margin sign
M=(2y-1)(z1-z0); for 2 classes this equals correctness routing (see mutual.py).

Per (shots, subject, seed): sample --shots trials PER CLASS as the few-shot
calibration set (both models train on the SAME shots), shared CE warm-up, then
fork into groups, evaluate BOTH models on the full 30% test split. Primary metric
= small model (S) test accuracy; B_acc tracked to watch the reverse S->B nudge.

Groups (user's minimal validation set):
  G0_CE      training-length control
  G1_FixKD   traditional offline all-sample B->S (big frozen)
  G3_SymDML  symmetric all-sample mutual learning
  G5_Routed  routed B->S only (one-way)
  G6_CRAMD   routed B->S + weak routed S->B  (core bidirectional)
  G7_DisagCE up-weight S's CE on routed-B->S samples, NO KL
             (is the gain just hard-sample re-weighting, not teacher probs?)

    conda run -n mirepnet python experiments/bidir/run_bidir_fewshot.py \
        --dataset BNCI2014004 --shots 5 10 20 --lam_bs 1.0 --lam_sb 0.1 --gpu 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

from collab.mutual import cr_amd_fold, GROUPS
import config
import data
from eval import metrics
from models import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--shots', type=int, nargs='+', default=[5, 10, 20],
                   help='labeled trials PER CLASS for the few-shot calibration set')
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--groups', default=None,
                   help='comma subset of G0_CE,G1_FixKD,G3_SymDML,G5_Routed,'
                        'G6_CRAMD,G7_DisagCE')
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--total', type=int, default=60)
    p.add_argument('--lam_bs', type=float, default=1.0)
    p.add_argument('--lam_sb', type=float, default=0.1)
    p.add_argument('--alpha_ce', type=float, default=1.0,
                   help='G7 disagreement CE up-weight')
    p.add_argument('--lr_big', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--tag', default='')
    return p.parse_args()


def _fewshot_idx(y_tr, n_shot, nc, seed):
    """n_shot indices per class from the train pool (seeded); take all if fewer."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(nc):
        pool = np.where(y_tr == c)[0]
        idx += list(rng.choice(pool, min(n_shot, len(pool)), replace=False))
    return np.array(sorted(idx))


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    bcfg = config.load_model_config('mirepnet')
    scfg = config.load_model_config('ifnet')
    subjects = a.subjects if a.subjects is not None else list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']
    val_split = dcfg['val_split']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    groups = ({k: GROUPS[k] for k in a.groups.split(',')} if a.groups else GROUPS)

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'results', 'metrics')
    tag = a.tag or f'bidirfs_sb{a.lam_sb}'
    out_csv = f'{root}/{a.dataset}_fewshot_{tag}_mirepnet_ifnet.csv'
    diag_csv = f'{root}/{a.dataset}_fewshot_{tag}_diag.csv'

    rows, diags = [], []
    for n_shot in a.shots:
        for seed in seeds:
            for subj in subjects:
                X_tr, y_tr, X_te, y_te = data.subject_split(
                    a.dataset, subj, val_split=val_split, seed=seed)
                sel = _fewshot_idx(y_tr, n_shot, nc, seed)
                Xs, ys = X_tr[sel], y_tr[sel]
                subj_tr = np.full(len(sel), subj)     # single-subject few-shot pool

                bacfg = dict(bcfg); bacfg.update(
                    in_channels=Xs.shape[1], samples=Xs.shape[2],
                    dataset_name=a.dataset, skip_preprocess=True)
                sacfg = dict(scfg); sacfg.update(
                    in_channels=Xs.shape[1], samples=Xs.shape[2],
                    dataset_name=a.dataset)
                big_ad = get_adapter('mirepnet', device=device, **bacfg)
                small_ad = get_adapter('ifnet', device=device, **sacfg)
                res, diag = cr_amd_fold(
                    big_ad, small_ad, nc, Xs, ys, subj_tr, X_te, subj,
                    groups=groups, warmup=a.warmup, total=a.total,
                    lam_bs=a.lam_bs, lam_sb=a.lam_sb, alpha_ce=a.alpha_ce,
                    lr_big=a.lr_big, lr_small=scfg.get('lr', 1e-3),
                    wd=scfg.get('weight_decay', 0.01),
                    bs=scfg.get('batch_size', 16), seed=seed)
                for g, (sp, bp) in res.items():
                    ms = metrics.evaluate(y_te, sp); mb = metrics.evaluate(y_te, bp)
                    rows.append(dict(dataset=a.dataset, shots=n_shot, subject=subj,
                                     seed=seed, group=g, n_train=len(sel),
                                     S_acc=ms['acc'], S_kappa=ms['kappa'],
                                     B_acc=mb['acc'], B_kappa=mb['kappa']))
                    print(f"shots{n_shot} S{subj} seed{seed} {g} | "
                          f"S={ms['acc']} B={mb['acc']}", flush=True)
                for d in diag:
                    d.update(dataset=a.dataset, shots=n_shot, subject=subj, seed=seed)
                    diags.append(d)
                pd.DataFrame(rows).to_csv(out_csv, index=False)   # incremental save
                pd.DataFrame(diags).to_csv(diag_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(pd.DataFrame(rows).groupby(['shots', 'group'])[['S_acc', 'B_acc']]
          .mean().round(3))


if __name__ == '__main__':
    main()
