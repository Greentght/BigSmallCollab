"""Finetune a teacher and export standardized artifacts + MC-dropout uncertainty.

Like ``finetune_export.py`` but ALSO computes per-train-sample MC-dropout
uncertainty (predictive entropy + BALD) from the SAME finetuned teacher, so the
uncertainty aligns row-for-row with the cached train logits/feats that KD uses.
This regenerates the train/test artifacts (fresh teacher instance) and writes a
sidecar ``<subj>_<seed>_train_mc.npz`` {pred_entropy, bald, y}.

    conda run -n mirepnet python scripts/export_teacher_mc.py --model mirepnet --dataset BNCI2014004 --gpu 5
    conda run -n cbramod  python scripts/export_teacher_mc.py --model cbramod_native --dataset BNCI2014004 --gpu 5

Consumed by run_distill.py's --adaptive conditions (entropy-weighted KD/Combo).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core import artifacts, config, data
from core.registry import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--mc_passes', type=int, default=20)
    p.add_argument('--overwrite', action='store_true',
                   help='recompute even if train_mc sidecar exists')
    return p.parse_args()


def mc_path(dataset, model, subject, seed):
    return os.path.join(artifacts.ARTIFACT_ROOT, dataset, model,
                        f'{subject}_{seed}_train_mc.npz')


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    mcfg = config.load_model_config(a.model)
    subjects = a.subjects or list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']
    val_split = dcfg['val_split']
    num_classes = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    for seed in seeds:
        for subj in subjects:
            mcp = mc_path(a.dataset, a.model, subj, seed)
            if os.path.exists(mcp) and not a.overwrite:
                print(f'[skip] {a.model} {a.dataset} S{subj} seed{seed}', flush=True)
                continue

            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)

            acfg = dict(mcfg)
            acfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                        dataset_name=a.dataset)
            ad = get_adapter(a.model, device=device, **acfg)
            model = ad.build(num_classes)
            model = ad.finetune(model, X_tr, y_tr, num_classes)

            # refresh standard artifacts from THIS teacher instance
            ad.export(model, X_te, y_te, a.dataset, subj, seed, 'test')
            ad.export(model, X_tr, y_tr, a.dataset, subj, seed, 'train')
            # matching MC-dropout uncertainty on the train (=KD) split
            pe, bald = ad.mc_uncertainty(model, X_tr, K=a.mc_passes)
            os.makedirs(os.path.dirname(mcp), exist_ok=True)
            np.savez(mcp, pred_entropy=pe, bald=bald,
                     y=np.asarray(y_tr, dtype=np.int64))
            print(f'[ok] {a.model} {a.dataset} S{subj} seed{seed} '
                  f'| meanH={pe.mean():.3f} meanBALD={bald.mean():.3f} -> {mcp}',
                  flush=True)

            del model
            if device != 'cpu':
                torch.cuda.empty_cache()
    print('Done.')


if __name__ == '__main__':
    main()
