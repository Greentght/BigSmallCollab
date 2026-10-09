"""Unified single-model finetune + artifact export (fewshot + LOSO).

Trains ONE model under a chosen protocol and writes standardized artifacts
(logits/feats/y) that the collab hub + Phase-D0 diagnostics consume. Run inside
that model's conda env (see configs/models/<model>.yaml `env`):

    conda run -n mirepnet python experiments/finetune/finetune.py --model ifnet   --dataset BNCI2014004    --protocol fewshot --gpu 2
    conda run -n mirepnet python experiments/finetune/finetune.py --model eegnet  --dataset BNCI2014001-4  --protocol fewshot --gpu 2
    conda run -n cbramod python experiments/finetune/run_loso_source_refresh_004_5001.py --model cbramod --dataset BNCI2014004 --seed 666 --gpu 2

Protocols
---------
fewshot: per-subject calibration/test split. ``--train_percentage`` is the
         TRAIN fraction; ``--val_split`` is the legacy TEST fraction. Artifact
         model dir = ``<model>``; key = subject. ``within`` is accepted as an
         alias for existing scripts/caches.
loso   : leave-one-subject-out. Fold f = subject f held out for test, all others
         train. Artifact model dir = ``<model>_loso``; key = fold index. Rows are
         fully determined by the held-out subject (no split randomness), so they
         align with the cached ``mirepnet_loso`` artifacts row-for-row.

The registered broadband LOSO tasks now use the dedicated current-baseline
entries in docs/loso_baseline.md. This generic entry rejects those tasks.

Finetune hyperparameters are resolved from
``configs/models/<model>.yaml`` under ``finetune.<dataset>.<protocol>``.

Restartable by default: a (key, seed) whose test (+train) artifacts already
exist is skipped, so re-launching a crashed worker only fills the gaps. Pass
``--force`` to retrain and overwrite existing artifacts. ``--keys`` and
``--seeds`` narrow the job set for multi-worker sharding.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import config
import data
from collab import artifacts
from data import split as split_utils
from models import get_adapter


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--protocol', choices=['fewshot', 'loso', 'within'], default='loso')
    p.add_argument('--keys', type=int, nargs='+', default=None,
                   help='subjects (fewshot) or folds (loso) to run; default all')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--val_split', type=float, default=None,
                   help='fewshot only: TEST fraction (default from dataset cfg)')
    p.add_argument('--train_percentage', type=float, default=None,
                   help='fewshot only: TRAIN fraction; overrides dataset val_split')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--no_export_train', dest='export_train',
                   action='store_false', default=True)
    p.add_argument('--force', action='store_true',
                   help='retrain and overwrite existing artifacts')
    p.add_argument('--out_csv', default=None,
                   help=('optional result CSV with columns '
                         'subject,seed,test_acc_pct'))
    p.add_argument('--log_file', default=None,
                   help='optional file that receives a copy of stdout/stderr')
    return p.parse_args(argv)


def _fit_and_export(model_name, artifact_dir, dataset, key, seed, num_classes,
                    X_tr, y_tr, X_te, y_te, device, mcfg, export_train,
                    uid_tr=None, uid_te=None, split_policy=None):
    """Train `model_name` on (X_tr,y_tr), export test (+train) under artifact_dir."""
    adapter_cfg = dict(mcfg)
    adapter_cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                       dataset_name=dataset)
    _set_seed(seed)
    ad = get_adapter(model_name, device=device, **adapter_cfg)
    model = ad.build(num_classes)
    model = ad.finetune(model, X_tr, y_tr, num_classes)

    export_items = [('test', X_te, y_te, uid_te)]
    if export_train:
        export_items.append(('train', X_tr, y_tr, uid_tr))
    for split, X, y, sample_uid in export_items:
        feats, logits = ad.infer(model, X)
        artifacts.save(dataset, artifact_dir, key, seed, split,
                       logits=logits, feats=feats, y=y,
                       sample_uid=sample_uid, split_policy=split_policy)
    acc = float((ad.infer(model, X_te)[1].argmax(1) == y_te).mean() * 100)
    del model
    if device != 'cpu':
        torch.cuda.empty_cache()
    return acc


def _result_row(key, seed, acc):
    return {
        'subject': int(key) + 1,
        'seed': int(seed),
        'test_acc_pct': round(float(acc), 2),
    }


def _artifact_test_acc_pct(dataset, artifact_dir, key, seed):
    d = artifacts.load(dataset, artifact_dir, key, seed, 'test')
    return float((d['logits'].argmax(1) == d['y']).mean() * 100)


def _result_stats(rows):
    accs = np.asarray([r['test_acc_pct'] for r in rows], dtype=np.float64)
    if len(accs):
        mean_acc = round(float(accs.mean()), 2)
        std_acc = round(float(accs.std(ddof=1)), 2) if len(accs) > 1 else 0.0
    else:
        mean_acc = ''
        std_acc = ''
    return len(rows), mean_acc, std_acc


def _write_result_csv(out_csv, rows):
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject', 'seed', 'test_acc_pct'])
        w.writeheader()
        w.writerows(rows)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _run(a):
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    dcfg = config.load_dataset_config(a.dataset)
    protocol = data.canonical_protocol(a.protocol)
    if protocol == 'loso' and 'loso' in dcfg:
        raise ValueError(
            '该数据集已采用当前宽带 LOSO baseline。请按 docs/loso_baseline.md '
            '使用专用 baseline 入口，避免通用旧入口加载旧 NPY 或忽略预处理缓存。')
    mcfg = config.load_model_config(a.model, a.dataset, protocol)
    seeds = a.seeds or dcfg['seeds']
    num_classes = dcfg['num_classes']
    n_sub = dcfg['num_subjects']
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    if a.val_split is not None and a.train_percentage is not None:
        raise ValueError('pass only one of --val_split or --train_percentage')
    artifact_dir = a.model if protocol == 'fewshot' else f'{a.model}_loso'
    keys = a.keys if a.keys is not None else list(range(n_sub))
    val_split = a.val_split if a.val_split is not None else dcfg['val_split']
    if a.train_percentage is not None:
        train_percentage = float(a.train_percentage)
        if train_percentage <= 0.0 or train_percentage >= 1.0:
            raise ValueError('--train_percentage must be in (0, 1)')
        val_split = 1.0 - train_percentage
    train_percentage = 1.0 - float(val_split)

    cfg_msg = ' '.join(
        f'{k}={mcfg[k]}' for k in (
            'epochs', 'lr', 'batch_size', 'weight_decay',
            'optimizer', 'optimizer_type', 'momentum',
            'dropout', 'scale', 'label_smoothing',
            'target_fs', 'l_freq', 'h_freq', 'notch_freq', 'apply_EA',
            'warmup_epochs', 'min_lr')
        if k in mcfg)
    print(f'[finetune] {a.model} {a.dataset} {protocol} -> dir={artifact_dir} '
          f'device={device} keys={keys} seeds={seeds}', flush=True)
    print(f'[config] {cfg_msg}', flush=True)
    rows = []
    for seed in seeds:
        for key in keys:
            have_test = artifacts.exists(a.dataset, artifact_dir, key, seed, 'test')
            have_train = (not a.export_train or
                          artifacts.exists(a.dataset, artifact_dir, key, seed, 'train'))
            if have_test and have_train and not a.force:
                print(f'[skip] {artifact_dir} k{key} seed{seed}', flush=True)
                if a.out_csv:
                    try:
                        acc = _artifact_test_acc_pct(a.dataset, artifact_dir, key, seed)
                        rows.append(_result_row(key, seed, acc))
                    except Exception as e:  # noqa: BLE001 — keep filling other rows
                        print(f'[ERR] {artifact_dir} k{key} seed{seed} cached artifact: {e}',
                              flush=True)
                continue
            if a.force and (have_test or have_train):
                print(f'[force] overwrite {artifact_dir} k{key} seed{seed}', flush=True)
            if protocol == 'fewshot':
                split_policy = split_utils.FEWSHOT_SPLIT_POLICY
                X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
                    a.dataset, key, val_split=val_split, seed=seed,
                    return_uid=True)
            else:
                split_policy = split_utils.LOSO_SPLIT_POLICY
                X_tr, y_tr, _subj_tr, X_te, y_te, uid_tr, uid_te = data.loso_split(
                    a.dataset, key, return_uid=True)
            try:
                acc = _fit_and_export(
                    a.model, artifact_dir, a.dataset, key, seed, num_classes,
                    X_tr, y_tr, X_te, y_te, device, mcfg, a.export_train,
                    uid_tr=uid_tr, uid_te=uid_te, split_policy=split_policy)
                rows.append(_result_row(key, seed, acc))
                print(f'[ok] {artifact_dir} k{key} seed{seed} acc={acc:.2f}',
                      flush=True)
            except Exception as e:  # noqa: BLE001 — worker keeps going, gap refilled on rerun
                print(f'[ERR] {artifact_dir} k{key} seed{seed}: {e}', flush=True)
    n_runs, mean_acc, std_acc = _result_stats(rows)
    if a.out_csv:
        _write_result_csv(a.out_csv, rows)
        print(f'Wrote {a.out_csv} ({len(rows)} run rows)', flush=True)
    print(f'[summary] n_runs={n_runs} mean_acc_pct={mean_acc} '
          f'std_acc_pct={std_acc}', flush=True)
    print('Done.', flush=True)


def main(argv=None):
    a = parse_args(argv)
    if not a.log_file:
        _run(a)
        return

    os.makedirs(os.path.dirname(os.path.abspath(a.log_file)), exist_ok=True)
    with open(a.log_file, 'w', buffering=1) as log_f:
        tee_out = _Tee(sys.stdout, log_f)
        tee_err = _Tee(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f'[log] writing stdout/stderr to {a.log_file}', flush=True)
            _run(a)


if __name__ == '__main__':
    main()
