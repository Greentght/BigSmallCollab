"""LOSO student distillation: per fold, train IFNet on 8 subjects, test on the
held-out subject. Conditions Base / VanillaKD / ProtoOnly / KDProto. Reads the
per-fold cached teacher (mirepnet_loso). Class-prototypes + KD targets come only
from training-subject teacher feats/logits (no leakage). Deterministic seeds,
class x subject balanced batches. Records per held-out subject.

    conda run -n mirepnet python scripts/run_loso_distill.py --dataset BNCI2014004 --gpu 8
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

LOSO_NAME = 'mirepnet_loso'


_per_class = metrics.per_class   # per-class breakdown (shared, core.metrics)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--student', default='ifnet')
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--lam_kd', type=float, default=0.5)
    p.add_argument('--lam_proto', type=float, default=0.5)
    p.add_argument('--lam_sim', type=float, default=0.5)
    p.add_argument('--lam_intra', type=float, default=0.5)
    p.add_argument('--lam_inter', type=float, default=0.5)
    p.add_argument('--conds', default=None,
                   help='comma list to restrict conditions, e.g. CorrectMaskKD,ConfidenceKD')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_csv', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    scfg = config.load_model_config(a.student)
    n_sub = {'BNCI2014004': 9, 'BNCI2014001-4': 9}[a.dataset]
    folds = a.folds if a.folds is not None else list(range(n_sub))
    seeds = a.seeds or dcfg['seeds']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    out_csv = a.out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics', f'{a.dataset}_loso_mirepnet_to_{a.student}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    lp = a.lam_proto
    # weight_src: None (uniform), 'mask' (teacher-correct 0/1), 'conf' (teacher
    # max-softmax confidence). All read the same cached teacher -> fully fair.
    conds = {
        'Base':          dict(lam_kd=0.0, lam_feat=0.0),
        'VanillaKD':     dict(lam_kd=a.lam_kd, lam_feat=0.0),
        'ProtoOnly':     dict(lam_kd=0.0, lam_feat=lp, feat_proto=True),
        'KDProto':       dict(lam_kd=a.lam_kd, lam_feat=lp, feat_proto=True),
        'CorrectMaskKD': dict(lam_kd=a.lam_kd, lam_feat=0.0, weight_src='mask'),
        'ConfidenceKD':  dict(lam_kd=a.lam_kd, lam_feat=0.0, weight_src='conf'),
        # EA-KD: w = 1/2 H_T (1 + H_S/logC), high teacher-entropy = high value.
        'EA_KD':         dict(lam_kd=a.lam_kd, lam_feat=0.0, ea_kd=True),
        # reliability x value: correctness mask gates the EA weight.
        'CorrectMaskEA': dict(lam_kd=a.lam_kd, lam_feat=0.0, ea_kd=True, weight_src='mask'),
        # relational (similarity-preserving) KD — batch B x B structure. class x
        # subject balanced sampling is already on (subject_ids passed in common).
        'SimFull':    dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='sim', lam_sim=a.lam_sim)),
        'IntraInter': dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=a.lam_intra, lam_inter=a.lam_inter)),
        'IntraOnly':  dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=a.lam_intra, lam_inter=0.0)),
        'InterOnly':  dict(lam_kd=0.0, lam_feat=0.0,
                           relational=dict(mode='intra_inter',
                                           lam_intra=0.0, lam_inter=a.lam_inter)),
    }
    if a.conds:
        keep = set(a.conds.split(',')); conds = {k: v for k, v in conds.items() if k in keep}

    rows = []
    for seed in seeds:
        for test_subj in folds:
            try:
                tch = artifacts.load(a.dataset, LOSO_NAME, test_subj, seed, 'train')
            except FileNotFoundError as e:
                print(f'[miss teacher] fold{test_subj} seed{seed}: {e}'); continue
            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, test_subj)
            assert np.array_equal(tch['y'], y_tr), 'teacher train rows misaligned'
            common = dict(temperature=2.0, teacher_correct_only=False,
                          balanced_batch=True, seed=seed, subject_ids=subj_tr,
                          epochs=scfg.get('epochs', 100), lr=scfg.get('lr', 1e-3),
                          weight_decay=scfg.get('weight_decay', 0.01),
                          batch_size=scfg.get('batch_size', 16))
            # teacher-derived per-sample weights (training samples, true labels
            # only -> no test leakage)
            t_logits = tch['logits']
            t_conf = np.exp(t_logits - t_logits.max(1, keepdims=True))
            t_conf = (t_conf / t_conf.sum(1, keepdims=True)).max(1).astype(np.float32)
            t_mask = (t_logits.argmax(1) == y_tr).astype(np.float32)

            for cond, kw0 in conds.items():
                kw = dict(kw0)
                wsrc = kw.pop('weight_src', None)
                if wsrc == 'mask':
                    kw['sample_weight'] = t_mask
                elif wsrc == 'conf':
                    kw['sample_weight'] = t_conf
                acfg = dict(scfg); acfg.update(
                    in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                    dataset_name=a.dataset)
                student = get_adapter(a.student, device=device, **acfg)
                preds = distill_student(
                    student, nc, X_tr, y_tr, tch['feats'], tch['logits'], X_te,
                    **kw, **common)
                m = metrics.evaluate(y_te, preds)
                row = dict(dataset=a.dataset, fold=test_subj, seed=seed,
                           condition=f'{a.student}_{cond}',
                           acc=m['acc'], kappa=m['kappa'])
                row.update(_per_class(y_te, preds, nc))
                rows.append(row)
                print(f"fold{test_subj} seed{seed} {a.student}_{cond} | "
                      f"acc={m['acc']} kappa={m['kappa']}", flush=True)
    if not rows:
        print('No teacher artifacts found.'); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')
    print(df.groupby('condition')[['acc', 'kappa']].mean().round(3))


if __name__ == '__main__':
    main()
