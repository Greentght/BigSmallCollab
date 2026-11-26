"""Run and summarize verified LaBraM→EEGNet broadband AFID cells."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.finetune.dispatch_labram_broadband_fewshot import gpu_uuids
from experiments.storage import PROJECT_ROOT, RESULTS_ROOT, REPORTS_ROOT, require_external_output, require_report_output

DATASETS = {'BNCI2014001': 9, 'BNCI2014001-4': 9, 'BNCI2014004': 9, 'BNCI2015001': 12, 'AlexMI': 8}
SEEDS = (666, 667, 668)
METHODS = ('none', 'afid_self_gate', 'afid')
FILES = ('experiments/distill/run_labram_eegnet_afid.py', 'experiments/distill/dispatch_labram_eegnet_afid.py',
         'collab/afid_labram.py', 'experiments/finetune/run_labram_broadband_fewshot.py',
         'experiments/finetune/labram_alexmi_broadband.py', 'experiments/distill/run_fewshot_structure_kd.py',
         'models/labram/adapter.py', 'models/labram/modeling_finetune.py', 'models/labram/montage.py',
         'models/eegnet/adapter.py', 'models/eegnet/residual_eegnet.py', 'models/base.py', 'models/__init__.py',
         'data/loso_source.py', 'data/preproc.py', 'data/split.py', 'data/channels.py', 'config.py',
         'configs/models/labram.yaml', 'configs/models/eegnet.yaml', 'collab/seed.py', 'experiments/storage.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def identity():
    return {'code': {name: sha(PROJECT_ROOT / name) for name in FILES}, 'seeds': list(SEEDS),
            'datasets': DATASETS, 'methods': list(METHODS), 'baseline': 'labram_benchmark_aligned_20261124_v1'}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def result_path(run_id, ds, subject, seed):
    return RESULTS_ROOT / 'fewshot/afid_labram_eegnet' / run_id / ds / f'subject{subject+1:02d}' / f'seed{seed}' / 'completed.json'


def validate(run_id, ds, subject, seed, campaign):
    path = result_path(run_id, ds, subject, seed)
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    ident = rec['identity']
    if (ident['dataset'] != ds or ident['subject'] != subject or ident['seed'] != seed or
        ident['smoke'] or ident['student_epochs'] != 100 or ident['protocol'] != 'within_subject_30_70' or
        not rec['teacher_prediction_match'] or set(rec['methods']) != set(METHODS)):
        raise ValueError(f'invalid result identity: {path}')
    for name in ('experiments/distill/run_labram_eegnet_afid.py', 'collab/afid_labram.py'):
        if ident['code_sha256'][name] != campaign['code'][name]:
            raise ValueError(f'code digest mismatch: {path}')
    train, test = rec['train_trial_uids'], rec['test_trial_uids']
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test) or len(test) != math.ceil(.7 * (len(train) + len(test))):
        raise ValueError(f'invalid trial split: {path}')
    for method, value in rec['methods'].items():
        expected = REPORTS_ROOT / 'fewshot/afid_labram_eegnet' / run_id / ds / f'subject{subject+1:02d}' / f'seed{seed}' / f'{method}_predictions.csv'
        if Path(value['path']) != expected or sha(expected) != value['sha256']:
            raise ValueError(f'invalid predictions: {expected}')
        with expected.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if [r['trial_uid'] for r in rows] != test:
            raise ValueError(f'wrong prediction trial order: {expected}')
        acc = sum(int(r['y_true']) == int(r['y_pred']) for r in rows) * 100. / len(rows)
        if abs(acc - value['accuracy_pct']) > 1e-8:
            raise ValueError(f'invalid accuracy: {expected}')
        if method != 'none' and value['gate_gradient_checks'] != [True] * 4:
            raise ValueError(f'missing gate gradients: {expected}')
    return rec


def summarize(run_id, campaign):
    rows = []
    for ds, subjects in DATASETS.items():
        records = {(subject, seed): validate(run_id, ds, subject, seed, campaign)
                   for seed in SEEDS for subject in range(subjects)}
        complete = all(records.values())
        for method in METHODS:
            seed_scores = {seed: (sum(records[(s, seed)]['methods'][method]['accuracy_pct'] for s in range(subjects)) / subjects
                                  if all(records[(s, seed)] for s in range(subjects)) else None) for seed in SEEDS}
            vals = [seed_scores[seed] for seed in SEEDS]
            rows.append({'dataset': ds, 'teacher': 'labram', 'student': 'eegnet', 'method': method,
                         'subjects': subjects, 'completed_cells': sum(x is not None for x in records.values()),
                         'expected_cells': subjects*3, 'complete': complete,
                         **{f'accuracy_pct_seed_{seed}': seed_scores[seed] if seed_scores[seed] is not None else '' for seed in SEEDS},
                         'accuracy_pct': sum(vals)/3 if complete else ''})
    output = require_report_output(REPORTS_ROOT / 'fewshot/afid_labram_eegnet' / run_id / 'summary.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[3,4])
    p.add_argument('--min-free-mb', type=int, default=4000)
    p.add_argument('--poll-seconds', type=float, default=10)
    args = p.parse_args()
    if Path(args.run_id).name != args.run_id or not args.run_id or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('invalid run ID or GPU list')
    root = require_external_output(RESULTS_ROOT / 'fewshot/afid_labram_eegnet' / args.run_id / 'controller')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'manager.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = identity()
    manifest = root / 'campaign.json'
    if manifest.exists() and json.loads(manifest.read_text()) != campaign:
        raise ValueError('campaign code or settings changed')
    if not manifest.exists():
        atomic_json(manifest, campaign)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    running = {}
    for ds, subjects in DATASETS.items():
        cells = [(s, seed) for seed in SEEDS for s in range(subjects)]
        while True:
            for gpu, job in list(running.items()):
                code = job['proc'].poll()
                if code is None:
                    continue
                job['log'].close()
                del running[gpu]
                if code != 0 or validate(args.run_id, ds, job['subject'], job['seed'], campaign) is None:
                    atomic_json(root / 'status.json', {'stage': ds, 'failed': True, 'cell': job['cell'], 'exit_code': code, 'log': str(job['path'])})
                    for other in running.values():
                        other['proc'].wait()
                        other['log'].close()
                    return 1
            active = {(j['subject'], j['seed']) for j in running.values()}
            done = {(s, seed) for s, seed in cells if validate(args.run_id, ds, s, seed, campaign)}
            missing = [x for x in cells if x not in done and x not in active]
            summarize(args.run_id, campaign)
            atomic_json(root / 'status.json', {'stage': ds, 'completed': len(done), 'total': len(cells),
                        'remaining': len(missing), 'failed': False,
                        'running': [{'gpu': gpu, 'cell': job['cell'], 'pid': job['proc'].pid} for gpu, job in running.items()]})
            if len(done) == len(cells) and not running:
                print(f'[completed] {ds}', flush=True)
                break
            available = gpu_uuids(args.gpus, args.min_free_mb)
            for gpu in args.gpus:
                if gpu in running or gpu not in available or not missing:
                    continue
                if identity() != campaign:
                    raise ValueError('campaign code changed during training')
                subject, seed = missing.pop(0)
                log_path = root / 'logs' / ds / f'subject{subject+1:02d}_seed{seed}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('a')
                worker_env = dict(env, CUDA_VISIBLE_DEVICES=available[gpu])
                cmd = [sys.executable, '-u', '-B', '-m', 'experiments.distill.run_labram_eegnet_afid',
                       '--dataset', ds, '--subject', str(subject), '--seed', str(seed), '--run-id', args.run_id, '--gpu', '0']
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=worker_env, stdout=log, stderr=subprocess.STDOUT)
                cell = f'{ds}/subject{subject+1:02d}/seed{seed}'
                running[gpu] = {'proc': proc, 'log': log, 'path': log_path, 'subject': subject, 'seed': seed, 'cell': cell}
                print(f'[start] {cell} GPU={gpu} pid={proc.pid}', flush=True)
            time.sleep(args.poll_seconds)
    output = summarize(args.run_id, campaign)
    atomic_json(root / 'status.json', {'stage': 'completed', 'failed': False, 'report': str(output)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
