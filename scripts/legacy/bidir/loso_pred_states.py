"""LOSO diagnostic: per fold, train a Base student (deterministic, = run-1 Base)
and compare its predictions with the teacher's, to (1) count the four prediction
states — (B correct,S correct)/(B correct,S wrong)/(B wrong,S correct)/(B wrong,
S wrong) — on train and test, and (2) check teacher confidence calibration under
LOSO. The (B wrong, S correct) fraction is what would justify reverse (S->B)
distillation; teacher calibration informs CorrectMaskKD vs ConfidenceKD.

    conda run -n mirepnet python scripts/legacy/bidir/loso_pred_states.py --dataset BNCI2014004 --gpu 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
import torch

from collab.distill import distill_student
import config
import data
from collab import artifacts
from models import get_adapter

LOSO_NAME = 'mirepnet_loso'


def quad(bc, sc):
    return dict(B1S1=float((bc & sc).mean()), B1S0=float((bc & ~sc).mean()),
                B0S1=float((~bc & sc).mean()), B0S0=float((~bc & ~sc).mean()))


def softmax_conf(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='BNCI2014004')
    ap.add_argument('--student', default='ifnet')
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    scfg = config.load_model_config(a.student)
    n_sub = {'BNCI2014004': 9, 'BNCI2014001-4': 9}[a.dataset]
    seeds = a.seeds or dcfg['seeds']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')

    qrows, crows = [], []
    for seed in seeds:
        for f in range(n_sub):
            try:
                tr = artifacts.load(a.dataset, LOSO_NAME, f, seed, 'train')
                te = artifacts.load(a.dataset, LOSO_NAME, f, seed, 'test')
            except FileNotFoundError:
                print(f'[miss] fold{f} seed{seed}'); continue
            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, f)
            acfg = dict(scfg); acfg.update(in_channels=X_tr.shape[1],
                                           samples=X_tr.shape[2], dataset_name=a.dataset)
            student = get_adapter(a.student, device=device, **acfg)
            te_pred, tr_pred = distill_student(
                student, nc, X_tr, y_tr, tr['feats'], tr['logits'], X_te,
                lam_kd=0.0, lam_feat=0.0, teacher_correct_only=False,
                balanced_batch=True, seed=seed, subject_ids=subj_tr,
                return_train_preds=True, epochs=scfg.get('epochs', 100),
                lr=scfg.get('lr', 1e-3), weight_decay=scfg.get('weight_decay', 0.01),
                batch_size=scfg.get('batch_size', 16))
            # quadrants (train & test): B=teacher, S=base student
            for split, tp, sp, yy in [('train', tr['logits'].argmax(1), tr_pred, y_tr),
                                      ('test', te['logits'].argmax(1), te_pred, y_te)]:
                q = quad(tp == yy, sp == yy)
                qrows.append(dict(dataset=a.dataset, fold=f, seed=seed, split=split, **q))
            # teacher confidence calibration on TEST (held-out subject)
            conf = softmax_conf(te['logits']); corr = (te['logits'].argmax(1) == y_te)
            crows.append(dict(dataset=a.dataset, fold=f, seed=seed,
                              conf_correct=float(conf[corr].mean()) if corr.any() else np.nan,
                              conf_wrong=float(conf[~corr].mean()) if (~corr).any() else np.nan,
                              teacher_test_acc=float(corr.mean() * 100)))
            print(f"fold{f} seed{seed} done", flush=True)

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        'results', 'metrics')
    qdf = pd.DataFrame(qrows); qdf.to_csv(f'{root}/{a.dataset}_loso_predstates.csv', index=False)
    cdf = pd.DataFrame(crows); cdf.to_csv(f'{root}/{a.dataset}_loso_teachercalib.csv', index=False)
    print('\n=== prediction states (mean over folds/seeds) ===')
    print(qdf.groupby('split')[['B1S1', 'B1S0', 'B0S1', 'B0S0']].mean().round(3))
    print('\n=== teacher confidence (test) ===')
    print(cdf[['conf_correct', 'conf_wrong', 'teacher_test_acc']].mean().round(3))


if __name__ == '__main__':
    main()
