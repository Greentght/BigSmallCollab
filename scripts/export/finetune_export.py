"""Compatibility wrapper for few-shot artifact export.

Preferred entry point:

    python scripts/export/export_preds.py --model <model> --dataset <dataset> --protocol fewshot

This file keeps the historical ``finetune_export.py`` command working while the
implementation lives in ``export_preds.py``.
"""
import argparse
import os
import sys
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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
    argv = [
        '--model', a.model,
        '--dataset', a.dataset,
        '--protocol', 'fewshot',
    ]
    if a.subjects is not None:
        argv += ['--keys'] + [str(s) for s in a.subjects]
    if a.seeds is not None:
        argv += ['--seeds'] + [str(s) for s in a.seeds]
    if a.val_split is not None:
        argv += ['--val_split', str(a.val_split)]
    if a.gpu is not None:
        argv += ['--gpu', str(a.gpu)]
    if a.epochs is not None:
        argv += ['--epochs', str(a.epochs)]
    if not a.export_train:
        argv += ['--no_export_train']

    export_preds = import_module('scripts.export.export_preds')
    export_preds.main(argv)


if __name__ == '__main__':
    main()
