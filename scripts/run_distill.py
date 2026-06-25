"""Offline KD + feature-align: distill a cached (frozen) teacher into a student.

Run in the STUDENT's conda env (small models -> mirepnet). The teacher's
train-split artifact (feats + logits) must already exist — exported earlier via
finetune_export.py in the teacher's own env. The big teacher is never loaded here.

    conda run -n mirepnet python scripts/run_distill.py \
        --dataset BNCI2014004 --teacher cbramod --student ifnet \
        --lam_kd 0.5 --lam_feat 0.5

Trains two conditions per (subject, seed): the plain student (lam=0) baseline and
the distilled student, and writes acc/kappa to
results/metrics/<dataset>_distill_<teacher>_to_<student>.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from collab.distill import distill_student
from core import artifacts, config, data, metrics
from core.registry import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--teacher', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--lam_kd', type=float, default=0.5)
    p.add_argument('--lam_feat', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_csv', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    scfg = config.load_model_config(a.student)
    subjects = a.subjects or list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']
    val_split = dcfg['val_split']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    out_csv = a.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics',
        f'{a.dataset}_distill_{a.teacher}_to_{a.student}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    conditions = {
        f'{a.student}_base': (0.0, 0.0),
        f'{a.student}_KD<-{a.teacher}': (a.lam_kd, a.lam_feat),
    }

    rows = []
    for seed in seeds:
        for subj in subjects:
            try:
                tch = artifacts.load(a.dataset, a.teacher, subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss teacher] S{subj} seed{seed}: {e}'); continue

            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)
            assert np.array_equal(tch['y'], y_tr), (
                f'teacher train rows misaligned for S{subj} seed{seed}')

            for cond, (lk, lf) in conditions.items():
                acfg = dict(scfg)
                acfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                            dataset_name=a.dataset)
                student = get_adapter(a.student, device=device, **acfg)
                preds = distill_student(
                    student, nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te,
                    lam_kd=lk, lam_feat=lf, temperature=a.temperature,
                    epochs=scfg.get('epochs', 50), lr=scfg.get('lr', 1e-3),
                    weight_decay=scfg.get('weight_decay', 0.01),
                    batch_size=scfg.get('batch_size', 16))
                m = metrics.evaluate(y_te, preds)
                rows.append(dict(dataset=a.dataset, subject=subj, seed=seed,
                                 condition=cond, acc=m['acc'], kappa=m['kappa'],
                                 lam_kd=lk, lam_feat=lf))
                print(f"S{subj} seed{seed} {cond} | acc={m['acc']} "
                      f"kappa={m['kappa']}", flush=True)

    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa']].mean().round(3))


if __name__ == '__main__':
    main()
