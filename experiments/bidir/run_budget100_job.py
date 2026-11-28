"""Execute an original few-shot implementation with frozen 100-epoch config paths.

Only configuration/path constants are rebound for older entries without a
--config option. Their model, objective, preprocessing and update loops are
reused directly. All numerical code hashes and the original 20-epoch identity
are retained in a separate replay provenance record.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from experiments.bidir.budget100 import (
    ARTIFACT, CONFIGS, REPORT, epoch_count, runtime_fingerprint, sha, verify_completion, write_json,
)
from experiments.storage import require_external_output


def validate_result(job):
    folder = Path(job['output'])
    result = json.loads((folder / 'evaluation.json').read_text())
    if epoch_count(result) != 100 or result['identity'].get('smoke'):
        raise ValueError('Replay did not complete exactly 100 formal epochs')
    original = json.loads(Path(job['source_evaluation']).read_text())['identity']
    current = result['identity']
    for key in ('source_manifest_sha256', 'train_uid_sha256', 'test_uid_sha256',
                'initial_model_sha256', 'initial_head_sha256', 'initial_hashes', 'mirepnet_ea_sha256'):
        if key in original and key in current and original[key] != current[key]:
            raise ValueError(f'Replay changed the original identity at {key}')
    if current.get('subject') != job['subject'] or current.get('seed') != job['seed']:
        raise ValueError('Replay subject/seed differs')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        uids, truth = p['uids'].tolist(), p['y']
        if len(uids) != len(set(uids)) or len(uids) != len(truth):
            raise ValueError('Duplicate or missing trial predictions')
        for key in p.files:
            if key not in ('uids', 'y') and p[key].shape != truth.shape:
                raise ValueError('Standalone predictions have an invalid shape')
    with np.load(Path(job['source_evaluation']).parent / 'predictions.npz', allow_pickle=False) as old:
        if uids != old['uids'].tolist() or not np.array_equal(truth, old['y']):
            raise ValueError('Replay changed the original evaluation trials or labels')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--expected-fingerprint', required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job['epochs'] != 100 or sha(job['source_evaluation']) != job['source_sha256']:
        raise ValueError('Invalid replay source or epoch budget')
    fingerprint, files = runtime_fingerprint()
    if fingerprint != args.expected_fingerprint: raise ValueError('Frozen numerical implementation changed')
    out = require_external_output(job['output']); out.mkdir(parents=True, exist_ok=True)
    if verify_completion(job): return
    provenance = {'job': job, 'joint_epochs': 100, 'runtime_sha256': fingerprint,
                  'code_sha256': files, 'config_sha256': sha(job['config']),
                  'fresh_initialization': True, 'cosine_horizon': 100, 'gpu': args.gpu,
                  'cpu_threads': 1, 'num_workers': 0, 'temporary_checkpoint_interval': 5}
    record = out / 'budget100_provenance.json'
    if record.exists():
        old = json.loads(record.read_text())
        # A retry may receive a different GPU, but it may not change the recipe.
        if {k:v for k,v in old.items() if k != 'gpu'} != {k:v for k,v in provenance.items() if k != 'gpu'}:
            raise ValueError('Replay provenance changed')
    write_json(record, provenance)
    if not (out / 'evaluation.json').is_file():
        module = importlib.import_module('experiments.bidir.' + job['module'])
        if hasattr(module, 'CONFIG'): module.CONFIG = Path(job['config'])
        # Saving fewer temporary checkpoints changes no optimization step.
        # This bounds I/O when old entries save a full Adam state every epoch.
        for name in ('atomic_save', 'atomic_torch'):
            if hasattr(module, name):
                original_save = getattr(module, name)
                def periodic_save(path, value, save=original_save):
                    epoch = value.get('epoch') if isinstance(value, dict) else None
                    if epoch is None or epoch % 5 == 0 or epoch == 100: return save(path, value)
                setattr(module, name, periodic_save)
        if job['mode'] == 'position':
            module.BASE = CONFIGS / 'fewshot_dml_ds_pairs.yaml'
            module.OUTPUT = ARTIFACT / 'hfd_position_fewshot_v1'
        cfg = yaml.safe_load(Path(job['config']).read_text())
        if cfg.get('joint_epochs') != 100: raise ValueError('Missing explicit joint training budget')
        if 'teacher_training' in cfg:
            for value in cfg['teacher_training'].values():
                if isinstance(value, dict) and 'epochs' in value and value['epochs'] != 100:
                    raise ValueError('A model-specific epoch override escaped the 100-epoch budget')
        sys.argv = [job['module'], *job['argv'], '--gpu', str(args.gpu)]
        print(f"BUDGET100 START job={job['id']} source_epochs=20 joint_epochs=100 gpu={args.gpu}", flush=True)
        module.main()
    validate_result(job)
    write_json(out / 'budget100_completion.json', {
        'job_id': job['id'], 'epochs': 100, 'source_epochs': 20,
        'runtime_sha256': fingerprint, 'source_evaluation': job['source_evaluation'],
        'evaluation_sha256': sha(out / 'evaluation.json'),
        'predictions_sha256': sha(out / 'predictions.npz'), 'complete': True,
    })
    print('BUDGET100 COMPLETED', job['id'], flush=True)


if __name__ == '__main__': main()
