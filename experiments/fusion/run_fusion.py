"""Run artifact-based big/small fusion experiments.

This script does not import or retrain the source models. It consumes
row-aligned train/test artifacts exported by experiments/finetune/finetune.py.
All fusion runs require sample_uid and split_policy metadata.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import config
import data
from collab import artifacts
from collab import fusion
from data import split as split_utils
from eval import metrics
from models import BIG_MODELS, SMALL_MODELS


METHODS = ('big_only', 'small_only', 'avg_prob', 'concat_mlp', 'gate_conf_acc')
METHOD_ALIASES = {
    'all': METHODS,
    'big': ('big_only',),
    'big_only': ('big_only',),
    'small': ('small_only',),
    'small_only': ('small_only',),
    'avg': ('avg_prob',),
    'avg_prob': ('avg_prob',),
    'concat': ('concat_mlp',),
    'concat_mlp': ('concat_mlp',),
    'gate': ('gate_conf_acc',),
    'gate_conf_acc': ('gate_conf_acc',),
}
FIELDNAMES = ['subject', 'seed', 'method', 'acc', 'kappa', 'n_test']


def parse_args(argv=None):
    p = argparse.ArgumentParser(epilog='Legacy direct matrix flags are deprecated; use --config configs/experiments/<name>.yaml')
    p.add_argument('--dataset', default='BNCI2014004')
    p.add_argument('--protocol', choices=['fewshot', 'within', 'loso'],
                   default='fewshot')
    p.add_argument('--big', default='cbramod', choices=BIG_MODELS)
    p.add_argument('--small', default='eegnet', choices=SMALL_MODELS)
    p.add_argument('--subjects', '--keys', dest='keys', type=int, nargs='+',
                   default=None,
                   help='zero-based subject/fold artifact keys; default all')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--methods', nargs='+', default=['all'],
                   help='all, big, small, avg, concat, gate')
    p.add_argument('--gpu', type=int, default=None,
                   help='GPU used only for concat MLP training')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--artifact_root', default=artifacts.ARTIFACT_ROOT)
    p.add_argument('--out_csv', default=None)
    p.add_argument('--fail_fast', action='store_true',
                   help='stop at the first missing or misaligned artifact')
    return p.parse_args(argv)


def _parse_methods(values):
    resolved = []
    for value in values:
        key = value.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(
                f'unknown method {value!r}; expected one of {sorted(METHOD_ALIASES)}')
        for method in METHOD_ALIASES[key]:
            if method not in resolved:
                resolved.append(method)
    return resolved


def _artifact_model(model, protocol):
    return model if protocol == 'fewshot' else f'{model}_loso'


def _expected_split_policy(protocol):
    if protocol == 'fewshot':
        return split_utils.FEWSHOT_SPLIT_POLICY
    if protocol == 'loso':
        return split_utils.LOSO_SPLIT_POLICY
    raise ValueError(f'unsupported protocol {protocol!r}')


def _load_pair(dataset, big_artifact, small_artifact, key, seed, split,
               artifact_root, expected_policy):
    pair, y = artifacts.load_aligned(
        dataset,
        [big_artifact, small_artifact],
        key,
        seed,
        split,
        root=artifact_root,
        require_uid=True,
        require_split_policy=True,
    )
    for model_name, item in pair.items():
        policy = item.get('split_policy')
        if policy != expected_policy:
            raise ValueError(
                f'{model_name} {split} split_policy={policy!r}, '
                f'expected {expected_policy!r}')
    return pair[big_artifact], pair[small_artifact], y


def _num_classes(*logit_arrays):
    cls = None
    for logits in logit_arrays:
        arr = np.asarray(logits)
        if arr.ndim != 2:
            raise ValueError(f'logits must be 2D, got {arr.shape}')
        if cls is None:
            cls = arr.shape[1]
        elif cls != arr.shape[1]:
            raise ValueError(
                f'class dimension mismatch: expected {cls}, got {arr.shape[1]}')
    return int(cls)


def _base_row(args, protocol, big_artifact, small_artifact, key, seed,
              y_train, y_test, num_classes, train_acc_big, train_acc_small):
    return {
        'dataset': args.dataset,
        'protocol': protocol,
        'big': args.big,
        'small': args.small,
        'big_artifact': big_artifact,
        'small_artifact': small_artifact,
        'subject': int(key) + 1,
        'key': int(key),
        'seed': int(seed),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'num_classes': int(num_classes),
        'train_acc_big_pct': round(float(train_acc_big) * 100.0, 2),
        'train_acc_small_pct': round(float(train_acc_small) * 100.0, 2),
    }


def _add_pred_row(rows, base, method, y_true, pred, extra=None):
    row = dict(base)
    row['method'] = method
    row.update(metrics.evaluate(y_true, pred))
    if extra:
        row.update(extra)
    rows.append(row)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})


def _print_summary(rows):
    if not rows:
        print('[summary] no rows produced', flush=True)
        return
    print('[summary]', flush=True)
    for method in METHODS:
        vals = [float(r['acc']) for r in rows if r['method'] == method]
        kappas = [float(r['kappa']) for r in rows if r['method'] == method]
        if not vals:
            continue
        acc = np.asarray(vals, dtype=np.float64)
        kap = np.asarray(kappas, dtype=np.float64)
        print(
            f'  {method}: n={len(vals)} acc={acc.mean():.2f} '
            f'+/- {acc.std(ddof=1) if len(vals) > 1 else 0.0:.2f} '
            f'kappa={kap.mean():.4f}',
            flush=True,
        )


def _run_cell(args, protocol, methods, device, big_artifact, small_artifact,
              expected_policy, key, seed):
    big_tr, small_tr, y_train = _load_pair(
        args.dataset, big_artifact, small_artifact, key, seed, 'train',
        args.artifact_root, expected_policy)
    big_te, small_te, y_test = _load_pair(
        args.dataset, big_artifact, small_artifact, key, seed, 'test',
        args.artifact_root, expected_policy)

    num_classes = _num_classes(
        big_tr['logits'], small_tr['logits'],
        big_te['logits'], small_te['logits'])
    train_acc_big = fusion.accuracy_fraction(big_tr['logits'], y_train)
    train_acc_small = fusion.accuracy_fraction(small_tr['logits'], y_train)
    base = _base_row(
        args, protocol, big_artifact, small_artifact, key, seed,
        y_train, y_test, num_classes, train_acc_big, train_acc_small)

    rows = []
    if 'big_only' in methods:
        _add_pred_row(rows, base, 'big_only', y_test,
                      fusion.preds_from_logits(big_te['logits']))
    if 'small_only' in methods:
        _add_pred_row(rows, base, 'small_only', y_test,
                      fusion.preds_from_logits(small_te['logits']))
    if 'avg_prob' in methods:
        probs = fusion.avg_prob_fusion(
            big_te['logits'], small_te['logits'],
            big_temperature=getattr(args, 'big_temperature', 1.0),
            small_temperature=getattr(args, 'small_temperature', 1.0),
            big_weight=getattr(args, 'big_weight', 0.5))
        _add_pred_row(rows, base, 'avg_prob', y_test,
                      fusion.preds_from_probs(probs))
    if 'concat_mlp' in methods:
        logits, info = fusion.train_concat_mlp(
            big_tr['feats'], small_tr['feats'], y_train,
            big_te['feats'], small_te['feats'], num_classes,
            seed=seed, device=device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            hidden=args.hidden, dropout=args.dropout)
        _add_pred_row(rows, base, 'concat_mlp', y_test,
                      fusion.preds_from_logits(logits), info)
    if 'gate_conf_acc' in methods:
        probs, info = fusion.gate_conf_acc(
            big_tr['logits'], small_tr['logits'], y_train,
            big_te['logits'], small_te['logits'],
            alpha=args.alpha, beta=args.beta,
            big_temperature=getattr(args, 'big_temperature', 1.0),
            small_temperature=getattr(args, 'small_temperature', 1.0))
        _add_pred_row(rows, base, 'gate_conf_acc', y_test,
                      fusion.preds_from_probs(probs), info)
    return rows


def main(argv=None):
    args = parse_args(argv)
    protocol = data.canonical_protocol(args.protocol)
    methods = _parse_methods(args.methods)
    dcfg = config.load_dataset_config(args.dataset)
    keys = args.keys if args.keys is not None else list(range(dcfg['num_subjects']))
    seeds = args.seeds if args.seeds is not None else dcfg['seeds']
    big_artifact = _artifact_model(args.big, protocol)
    small_artifact = _artifact_model(args.small, protocol)
    expected_policy = _expected_split_policy(protocol)
    device = (f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available()
              else 'cpu')
    out_csv = args.out_csv or os.path.join(
        '/data1/llx/BigSmallcollab/results',
        f'{args.dataset}_{protocol}_fusion_{args.big}_{args.small}.csv')

    print(
        f'[fusion] dataset={args.dataset} protocol={protocol} '
        f'big={args.big} small={args.small} methods={methods} '
        f'device={device}',
        flush=True,
    )
    print(
        f'[artifacts] big={big_artifact} small={small_artifact} '
        f'root={args.artifact_root} require_uid=True '
        f'require_split_policy=True expected_policy={expected_policy}',
        flush=True,
    )

    rows, errors = [], []
    for seed in seeds:
        for key in keys:
            try:
                cell_rows = _run_cell(
                    args, protocol, methods, device, big_artifact,
                    small_artifact, expected_policy, key, seed)
                rows.extend(cell_rows)
                print(
                    f'[ok] key={key} subject={int(key) + 1} seed={seed} '
                    f'rows={len(cell_rows)}',
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - keep filling other cells
                msg = f'key={key} subject={int(key) + 1} seed={seed}: {e}'
                errors.append(msg)
                print(f'[ERR] {msg}', flush=True)
                if args.fail_fast:
                    raise

    _write_csv(out_csv, rows)
    print(f'[write] {out_csv} rows={len(rows)}', flush=True)
    _print_summary(rows)
    if errors:
        print(f'[errors] {len(errors)} cells failed', flush=True)
    print('Done.', flush=True)
    return 1 if errors and not rows else 0


if __name__ == "__main__":
    if any(arg == "--config" or arg.startswith("--config=")
           for arg in sys.argv[1:]):
        from experiments.fusion.config_runner import main as config_main
        raise SystemExit(config_main())
    raise SystemExit(main())
