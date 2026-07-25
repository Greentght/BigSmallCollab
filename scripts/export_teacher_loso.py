"""LOSO teacher: per-fold finetune MIRepNet on the 8 training subjects, export
train (8-subject) + test (held-out subject) logits/feats. EA is applied PER
SUBJECT (each subject whitened by its own reference covariance), then 45ch pad;
the adapter runs with skip_preprocess (data pre-EA'd) so the mixed multi-subject
train set is never re-whitened with one covariance.

    conda run -n mirepnet python scripts/export_teacher_loso.py --dataset BNCI2014004 --gpu 8

Artifacts keyed as model='mirepnet_loso', subject=<held-out test subject> (=fold).
No leakage: MIRepNet pretraining excludes these downstream datasets (paper Table 1).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from collab.distill import _set_seed
from core import artifacts, config, data
from core.registry import get_adapter

TEACHER = 'mirepnet'
LOSO_NAME = 'mirepnet_loso'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--folds', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    mcfg = config.load_model_config(TEACHER)
    n_sub = {'BNCI2014004': 9, 'BNCI2014001-4': 9}[a.dataset]
    folds = a.folds if a.folds is not None else list(range(n_sub))
    seeds = a.seeds or dcfg['seeds']
    nc = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    for seed in seeds:
        for test_subj in folds:
            if (artifacts.exists(a.dataset, LOSO_NAME, test_subj, seed, 'test')
                    and artifacts.exists(a.dataset, LOSO_NAME, test_subj, seed, 'train')
                    and not a.overwrite):
                print(f'[skip] fold {test_subj} seed{seed}', flush=True)
                continue

            X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(a.dataset, test_subj)

            acfg = dict(mcfg)
            acfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                        dataset_name=a.dataset, skip_preprocess=True)
            ad = get_adapter(TEACHER, device=device, **acfg)
            ad.name = LOSO_NAME          # key artifacts separately from within-subject

            # per-subject EA + 45ch pad, then the adapter passes it through
            Xp_tr = ad.ea_pad_per_subject(X_tr, subj_tr)
            Xp_te = ad.ea_pad_per_subject(X_te, np.full(len(y_te), test_subj))

            _set_seed(seed)
            model = ad.build(nc)
            model = ad.finetune(model, Xp_tr, y_tr, nc)

            ad.export(model, Xp_te, y_te, a.dataset, test_subj, seed, 'test')
            ad.export(model, Xp_tr, y_tr, a.dataset, test_subj, seed, 'train')
            te_acc = None
            d = artifacts.load(a.dataset, LOSO_NAME, test_subj, seed, 'test')
            te_acc = (d['logits'].argmax(1) == d['y']).mean() * 100
            print(f'[ok] fold {test_subj} seed{seed} '
                  f'| n_tr={len(y_tr)} n_te={len(y_te)} test_acc={te_acc:.1f}',
                  flush=True)
            del model
            if device != 'cpu':
                torch.cuda.empty_cache()
    print('Done.')


if __name__ == '__main__':
    main()
