"""Export fine-tuned CBraMod train artifacts with an explicit feature tap.

This is an independent teacher-export utility for the six-pair pilot.  The
corrected pilot uses ``feature_head=original_mlp_200`` so the artifact feature
is the original task head's 200-D penultimate activation immediately before
``Linear(200, num_classes)``.  The earlier ``mean_pool_200`` output is retained
as a separate, explicitly non-original-head diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from collab import artifacts
from collab.seed import set_seed
from models import get_adapter
from experiments.storage import external_path, require_external_output, resolve_local_file

# Keep the canonical project spellings explicit; the export is intentionally
# limited to the four datasets in the six-pair pilot.
DEFAULT_DATASETS = ('BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
SUBJECT_COUNTS = {
    'BNCI2014001': 9,
    'BNCI2014001-4': 9,
    'BNCI2014004': 9,
    'BNCI2015001': 12,
    'AlexMI': 8,
}
SEED = 666


def sha256(path: Path) -> str:
    path = resolve_local_file(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', required=True,
                   help='directory containing dataset/cbramod train artifacts')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--feature-head', default='original_mlp_200',
                   choices=('original_mlp_200', 'mean_pool_200'))
    p.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS),
                   choices=tuple(SUBJECT_COUNTS))
    return p.parse_args(argv)


def run_one(dataset: str, subject: int, out_root: Path, device: torch.device,
            feature_head: str):
    dcfg = config.load_dataset_config(dataset)
    X_tr, y_tr, _X_te, _y_te, uid_tr, _uid_te = data.subject_split(
        dataset, subject, val_split=float(dcfg['val_split']), seed=SEED,
        return_uid=True)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    cfg = config.load_model_config('cbramod', dataset, 'fewshot')
    cfg.update({
        'dataset_name': dataset,
        'in_channels': int(X_tr.shape[1]),
        'samples': int(X_tr.shape[2]),
        'feature_head': feature_head,
    })
    set_seed(SEED)
    adapter = get_adapter('cbramod', device=device, **cfg)
    model = adapter.build(int(dcfg['num_classes']))
    model = adapter.finetune(model, X_tr, y_tr, int(dcfg['num_classes']))
    model.eval()
    with torch.no_grad():
        feats, logits = adapter.infer(model, X_tr)
    feats = np.asarray(feats, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[1] != 200:
        raise RuntimeError(f'{dataset} S{subject + 1}: corrected CBraMod feature shape {feats.shape}, expected (N,200)')
    if logits.shape[0] != len(y_tr) or not np.isfinite(feats).all() or not np.isfinite(logits).all():
        raise RuntimeError(f'{dataset} S{subject + 1}: invalid corrected teacher outputs')
    if len(set(map(tuple, uid_tr.tolist()))) != len(uid_tr):
        raise RuntimeError(f'{dataset} S{subject + 1}: duplicate train UID')
    target = out_root / dataset / 'cbramod'
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'{subject}_{SEED}_train.npz'
    artifacts.save(dataset, 'cbramod', subject, SEED, 'train', logits, feats,
                   y_tr, root=out_root, sample_uid=uid_tr,
                   split_policy='fewshot_stratified_random')
    del model, adapter
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'dataset': dataset, 'subject': subject + 1, 'subject_index': subject,
        'seed': SEED, 'train_count': len(y_tr), 'feature_dim': int(feats.shape[1]),
        'logit_dim': int(logits.shape[1]), 'feature_head': feature_head,
        'artifact_path': str(path.resolve()), 'artifact_sha256': sha256(path),
        'status': 'complete', 'failure_reason': '',
    }


def main(argv=None):
    args = parse_args(argv)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    manifest_path = out_root / 'teacher_export_manifest.csv'
    existing = {}
    if args.resume and manifest_path.is_file():
        with manifest_path.open() as f:
            existing = {(r['dataset'], int(r['subject_index'])): r for r in csv.DictReader(f)}
    rows = []
    datasets = tuple(args.datasets)
    if datasets not in (DEFAULT_DATASETS, ('BNCI2014001-4',)):
        raise ValueError('export scope must be the original four datasets or the isolated BNCI2014001-4 supplement')
    for dataset in datasets:
        for subject in range(SUBJECT_COUNTS[dataset]):
            key = (dataset, subject)
            old = existing.get(key)
            expected = out_root / dataset / 'cbramod' / f'{subject}_{SEED}_train.npz'
            if args.resume and old and old.get('status') == 'complete' and expected.is_file():
                rows.append(old)
                print(f'[skip] {dataset} S{subject + 1}', flush=True)
                continue
            started = time.time()
            try:
                row = run_one(dataset, subject, out_root, device,
                              args.feature_head)
                row['runtime_seconds'] = time.time() - started
                print(f'[done] {dataset} S{subject + 1} feature_dim={row["feature_dim"]}', flush=True)
            except Exception as exc:
                row = {'dataset': dataset, 'subject': subject + 1,
                       'subject_index': subject, 'seed': SEED,
                       'status': 'failed', 'failure_reason': repr(exc)}
                print(f'[failed] {dataset} S{subject + 1}: {exc}', flush=True)
                raise
            rows.append(row)
            fields = sorted({k for r in rows for k in r})
            tmp = manifest_path.with_suffix('.tmp')
            with tmp.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
            os.replace(tmp, manifest_path)
    expected = sum(SUBJECT_COUNTS[name] for name in datasets)
    print(f'[complete] {len(rows)}/{expected} corrected CBraMod teacher exports: {out_root}', flush=True)


if __name__ == '__main__':
    main()
