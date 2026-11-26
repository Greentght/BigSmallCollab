"""GPU preflight for DML-DS+HFD gradient isolation and exact resume."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_hfd_fewshot import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--expected-implementation-sha256', required=True)
    parser.add_argument('--expected-verifier-sha256', required=True)
    parser.add_argument('--subject', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--progress', required=True, type=Path)
    parser.add_argument('--stop-file', required=True, type=Path)
    args = parser.parse_args()
    if sha256_file(Path(__file__)) != args.expected_verifier_sha256:
        raise ValueError('Queued preflight verifier changed')
    if sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError('Queued preflight configuration changed')
    cfg = yaml.safe_load(args.config.read_text())
    tag = 'gpu_' + args.expected_config_sha256[:8] + '_' + args.expected_implementation_sha256[:8]
    common = [sys.executable, '-B', '-m', 'experiments.bidir.run_dml_ds_hfd_fewshot',
              '--config', str(args.config), '--expected-config-sha256', args.expected_config_sha256,
              '--expected-implementation-sha256', args.expected_implementation_sha256,
              '--subject', str(args.subject), '--seed', str(args.seed), '--gpu', str(args.gpu),
              '--smoke', '--stop-file', str(args.stop_file)]
    for phase, smoke_id, extra in (('interrupt', tag + '_resume', ['--stop-after', '1']),
                                    ('resume', tag + '_resume', []),
                                    ('continuous', tag + '_continuous', [])):
        atomic_json(args.progress, {'status': 'running', 'phase': phase})
        print(f'GPU DML-DS+HFD preflight: {phase}', flush=True)
        completed = subprocess.run(common + ['--smoke-id', smoke_id] + extra, check=False)
        if completed.returncode == 75:
            atomic_json(args.progress, {'status': 'paused', 'phase': phase})
            return 75
        if completed.returncode:
            raise RuntimeError(f'{phase} exited {completed.returncode}')
    folders = [Path(cfg['output_root']) / 'smoke' / (tag + suffix) /
               f'subject_{args.subject}' / f'seed_{args.seed}'
               for suffix in ('_resume', '_continuous')]
    records = [json.loads((path / 'evaluation.json').read_text()) for path in folders]
    histories = [json.loads((path / 'training.json').read_text()) for path in folders]
    if any(record['evaluation_split'] != 'calibration' for record in records):
        raise ValueError('Preflight must evaluate calibration trials only')
    if any(len(hist['history']) != 2 or any(epoch['mean_losses'][1] <= 0
           for epoch in hist['history']) for hist in histories):
        raise ValueError('Preflight did not exercise HFD on both epochs')
    if records[0]['identity'] != records[1]['identity']:
        raise ValueError('Resumed and continuous identities differ')
    for key in ('big', 'small', 'model_sha256', 'head_sha256', 'hfd_sha256'):
        if records[0][key] != records[1][key]:
            raise ValueError(f'Resumed and continuous {key} differs')
    with np.load(folders[0] / 'predictions.npz', allow_pickle=False) as a, \
         np.load(folders[1] / 'predictions.npz', allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
            raise ValueError('Resumed and continuous predictions differ')
    atomic_json(args.progress, {
        'status': 'completed', 'pair': cfg['pair'], 'subject': args.subject, 'seed': args.seed,
        'calibration_only': True, 'hfd_gradient_isolation_checked': True,
        'resume_exactly_matches': True, 'config_sha256': args.expected_config_sha256,
        'implementation_sha256': args.expected_implementation_sha256,
        'verifier_sha256': args.expected_verifier_sha256,
        'results': [str(path / 'evaluation.json') for path in folders]})
    print('GPU DML-DS+HFD preflight passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
