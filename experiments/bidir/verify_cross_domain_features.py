"""Calibration-only GPU smoke and exact interrupted/resumed training check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_cross_domain_features_fewshot import CONFIG, ROOT, fingerprint, run_folder
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.storage import require_report_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--expected-fingerprint')
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    frozen, config_sha = fingerprint(), sha256_file(args.config)
    if args.expected_fingerprint and args.expected_fingerprint != frozen:
        raise ValueError('Frozen preflight identity changed')
    prefix = frozen[:10]
    common = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_cross_domain_features_fewshot',
              '--config', str(args.config), '--dataset', args.dataset, '--method', args.method,
              '--subject', '0', '--seed', str(cfg['seeds'][0]), '--gpu', str(args.gpu),
              '--smoke', '--expected-fingerprint', frozen, '--expected-config-sha256', config_sha]
    for name, extra in (('continuous', []), ('resumed', ['--stop-after', '1']), ('resumed', [])):
        subprocess.run(common + ['--smoke-id', prefix + name] + extra, cwd=ROOT, check=True)
    folders = [run_folder(cfg, args.dataset, args.method, 0, cfg['seeds'][0], prefix + name)
               for name in ('continuous', 'resumed')]
    results = [json.loads((folder / 'evaluation.json').read_text()) for folder in folders]
    for key in ('identity', 'model_sha256', 'head_sha256', 'feature_sha256', 'big', 'small'):
        if results[0][key] != results[1][key]:
            raise RuntimeError(f'Resumed training differs at {key}')
    histories = [json.loads((folder / 'training.json').read_text())['history'] for folder in folders]
    for history in histories:
        for entry in history:
            entry.pop('seconds', None)
    if histories[0] != histories[1]:
        raise RuntimeError('Resumed losses or associations differ')
    with np.load(folders[0] / 'predictions.npz') as a, np.load(folders[1] / 'predictions.npz') as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise RuntimeError('Resumed calibration predictions differ')
    atomic_json(require_report_output(args.progress), {
        'status': 'completed', 'dataset': args.dataset, 'method': args.method,
        'implementation_sha256': frozen, 'config_sha256': config_sha,
        'calibration_only': True, 'gradient_isolation_checked': True, 'resume_exactly_matches': True,
        'results': [str(folder / 'evaluation.json') for folder in folders],
    })
    print(f'Preflight passed: {args.dataset} {args.method}', flush=True)


if __name__ == '__main__':
    main()
