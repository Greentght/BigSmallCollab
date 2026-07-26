"""Unified per-sample prediction/feature exporter (within + LOSO).

Trains ONE model under a chosen protocol and writes standardized artifacts
(logits/feats/y) that the collab hub + Phase-D0 diagnostics consume. Run inside
that model's conda env (see configs/models/<model>.yaml `env`):

    conda run -n mirepnet python scripts/export_preds.py --model ifnet   --dataset BNCI2014004    --protocol loso   --gpu 2
    conda run -n mirepnet python scripts/export_preds.py --model eegnet  --dataset BNCI2014001-4  --protocol within --gpu 2
    conda run -n cbramod  python scripts/export_preds.py --model cbramod_native --dataset BNCI2014004 --protocol loso --gpu 2

Protocols
---------
within : per-subject calibration/test split (val_split = TEST fraction, default
         from the dataset config). Artifact model dir = ``<model>``; key = subject.
         Row-identical to the existing mirepnet / cbramod_native within cache.
loso   : leave-one-subject-out. Fold f = subject f held out for test, all others
         train. Artifact model dir = ``<model>_loso``; key = fold index. Rows are
         fully determined by the held-out subject (no split randomness), so they
         align with the cached ``mirepnet_loso`` artifacts row-for-row.

Restartable: a (key, seed) whose test (+train) artifacts already exist is skipped,
so re-launching a crashed worker only fills the gaps. ``--folds/--subjects`` and
``--seeds`` narrow the job set for multi-worker sharding.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
import data
from collab import artifacts
from models import get_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--protocol', choices=['within', 'loso'], default='loso')
    p.add_argument('--keys', type=int, nargs='+', default=None,
                   help='subjects (within) or folds (loso) to run; default all')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--val_split', type=float, default=None,
                   help='within only: TEST fraction (default from dataset cfg)')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None, help='override config')
    p.add_argument('--no_export_train', dest='export_train',
                   action='store_false', default=True)
    return p.parse_args()


def _fit_and_export(model_name, artifact_dir, dataset, key, seed, num_classes,
                    X_tr, y_tr, X_te, y_te, device, mcfg, epochs, export_train):
    """Train `model_name` on (X_tr,y_tr), export test (+train) under artifact_dir."""
    adapter_cfg = dict(mcfg)
    adapter_cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                       dataset_name=dataset)
    if epochs is not None:
        adapter_cfg['epochs'] = epochs

    torch.manual_seed(seed)
    np.random.seed(seed)
    ad = get_adapter(model_name, device=device, **adapter_cfg)
    model = ad.build(num_classes)
    model = ad.finetune(model, X_tr, y_tr, num_classes)

    for split, X, y in [('test', X_te, y_te)] + (
            [('train', X_tr, y_tr)] if export_train else []):
        feats, logits = ad.infer(model, X)
        artifacts.save(dataset, artifact_dir, key, seed, split,
                       logits=logits, feats=feats, y=y)
    acc = float((ad.infer(model, X_te)[1].argmax(1) == y_te).mean() * 100)
    del model
    if device != 'cpu':
        torch.cuda.empty_cache()
    return acc


def main():
    a = parse_args()
    dcfg = config.load_dataset_config(a.dataset)
    mcfg = config.load_model_config(a.model)
    seeds = a.seeds or dcfg['seeds']
    num_classes = dcfg['num_classes']
    n_sub = dcfg['num_subjects']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    artifact_dir = a.model if a.protocol == 'within' else f'{a.model}_loso'
    keys = a.keys if a.keys is not None else list(range(n_sub))
    val_split = a.val_split if a.val_split is not None else dcfg['val_split']

    print(f'[export] {a.model} {a.dataset} {a.protocol} -> dir={artifact_dir} '
          f'device={device} keys={keys} seeds={seeds}', flush=True)

    for seed in seeds:
        for key in keys:
            have_test = artifacts.exists(a.dataset, artifact_dir, key, seed, 'test')
            have_train = (not a.export_train or
                          artifacts.exists(a.dataset, artifact_dir, key, seed, 'train'))
            if have_test and have_train:
                print(f'[skip] {artifact_dir} k{key} seed{seed}', flush=True)
                continue
            if a.protocol == 'within':
                X_tr, y_tr, X_te, y_te = data.subject_split(
                    a.dataset, key, val_split=val_split, seed=seed)
            else:
                X_tr, y_tr, _subj_tr, X_te, y_te = data.loso_split(a.dataset, key)
            try:
                acc = _fit_and_export(
                    a.model, artifact_dir, a.dataset, key, seed, num_classes,
                    X_tr, y_tr, X_te, y_te, device, mcfg, a.epochs, a.export_train)
                print(f'[ok] {artifact_dir} k{key} seed{seed} acc={acc:.2f}',
                      flush=True)
            except Exception as e:  # noqa: BLE001 — worker keeps going, gap refilled on rerun
                print(f'[ERR] {artifact_dir} k{key} seed{seed}: {e}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
