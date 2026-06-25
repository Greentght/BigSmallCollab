"""Finetune ONE model and export its standardized artifacts.

Run inside that model's conda env (see configs/models/<model>.yaml `env`):

    conda run -n mirepnet python scripts/finetune_export.py --model ifnet  --dataset BNCI2014004
    conda run -n cbramod  python scripts/finetune_export.py --model cbramod --dataset BNCI2014004 --gpu 1
    conda run -n labram   python scripts/finetune_export.py --model labram  --dataset BNCI2014004 --gpu 1

For every (subject, seed) it finetunes on the calibration split and writes BOTH:
  - split='test'  -> used by ensemble + as eval set for distillation
  - split='train' -> teacher feats/logits consumed by offline distillation
Restartable: a (subject, seed) whose test+train artifacts already exist is skipped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from core import artifacts, config, data
from core.registry import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--subjects', type=int, nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--val_split', type=float, default=None)
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None, help='override config')
    p.add_argument('--export_train', action='store_true', default=True,
                   help='also export the train split (teacher signal for KD)')
    p.add_argument('--no_export_train', dest='export_train', action='store_false')
    return p.parse_args()


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    mcfg = config.load_model_config(a.model)

    subjects = a.subjects or list(range(dcfg['num_subjects']))
    seeds = a.seeds or dcfg['seeds']
    val_split = a.val_split if a.val_split is not None else dcfg['val_split']
    num_classes = dcfg['num_classes']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')

    for seed in seeds:
        for subj in subjects:
            have_test = artifacts.exists(a.dataset, a.model, subj, seed, 'test')
            have_train = (not a.export_train or
                          artifacts.exists(a.dataset, a.model, subj, seed, 'train'))
            if have_test and have_train:
                print(f'[skip] {a.model} {a.dataset} S{subj} seed{seed}', flush=True)
                continue

            X_tr, y_tr, X_te, y_te = data.subject_split(
                a.dataset, subj, val_split=val_split, seed=seed)

            adapter_cfg = dict(mcfg)
            adapter_cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                               dataset_name=a.dataset)
            if a.epochs is not None:
                adapter_cfg['epochs'] = a.epochs

            ad = get_adapter(a.model, device=device, **adapter_cfg)
            model = ad.build(num_classes)
            model = ad.finetune(model, X_tr, y_tr, num_classes)

            p_te = ad.export(model, X_te, y_te, a.dataset, subj, seed, 'test')
            msg = f'[ok] {a.model} {a.dataset} S{subj} seed{seed} -> {p_te}'
            if a.export_train:
                ad.export(model, X_tr, y_tr, a.dataset, subj, seed, 'train')
            print(msg, flush=True)

            del model
            if device != 'cpu':
                torch.cuda.empty_cache()
    print('Done.')


if __name__ == '__main__':
    main()
