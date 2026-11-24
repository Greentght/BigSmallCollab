"""Run the DCM matrix while allowing at most one queue worker per GPU."""
import csv
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.storage import require_external_output, require_report_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/loso/dcm.yaml'


def gpu_counts():
    gpu_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    app_output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    counts = {}
    for line in gpu_output.splitlines():
        index, uuid = [value.strip() for value in line.split(',')]
        counts[int(index)] = sum(row.split(',')[0].strip() == uuid for row in app_output.splitlines())
    return counts


def write_report(root, cfg):
    rows = []
    for pair, condition, seed, fold in itertools.product(cfg['pairs'], cfg['conditions'], cfg['seeds'], cfg['folds']):
        path = root / 'runs' / pair / condition / f'fold_{fold}' / f'seed_{seed}' / 'evaluation.json'
        result = json.loads(path.read_text())
        identity = result['identity']
        if (identity['pair'], identity['condition'], identity['seed'], identity['fold']) != (pair, condition, seed, fold):
            raise RuntimeError(f'Result identity mismatch: {path}')
        rows.append([pair, condition, seed, fold, result['big_accuracy'], result['small_accuracy'], str(path)])
    report = require_report_output(ROOT / 'results/dcm_0014_v1/fold_accuracy.csv')
    report.parent.mkdir(parents=True, exist_ok=True)
    tmp = report.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'condition', 'seed', 'fold', 'big_accuracy_percent', 'small_accuracy_percent', 'source'])
        writer.writerows(rows)
    os.replace(tmp, report)
    summary = require_report_output(report.parent / 'summary.csv')
    tmp = summary.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'condition', 'big_accuracy_percent', 'small_accuracy_percent',
                         'big_delta_vs_none_pp', 'small_delta_vs_none_pp',
                         'big_delta_vs_dml_kl_pp', 'small_delta_vs_dml_kl_pp',
                         'big_delta_vs_dml_ds_pp', 'small_delta_vs_dml_ds_pp'])
        for pair in cfg['pairs']:
            means = {}
            for condition in cfg['conditions']:
                selected = [r for r in rows if r[0] == pair and r[1] == condition]
                means[condition] = [sum(r[i] for r in selected) / len(selected) for i in (4, 5)]
            for condition in cfg['conditions']:
                scores = means[condition]
                deltas = [scores[i] - means[base][i] for base in ('none', 'dml_kl', 'dml_ds') for i in (0, 1)]
                writer.writerow([pair, condition] + [f'{v:.2f}' for v in scores + deltas])
    os.replace(tmp, summary)


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    root = require_external_output(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = list(itertools.product(cfg['folds'], cfg['seeds'], cfg['conditions'], cfg['pairs']))
        pending = []
        for fold, seed, condition, pair in jobs:
            folder = root / 'runs' / pair / condition / f'fold_{fold}' / f'seed_{seed}'
            pending.append((fold, seed, condition, pair, folder))
        workers = {}
        failure = None
        tracked = [CONFIG, ROOT / 'experiments/bidir/run_dcm_loso.py', ROOT / 'collab/dcm.py',
                   ROOT / 'data/loso_source.py', ROOT / 'data/preproc.py', ROOT / 'config.py',
                   ROOT / 'models/base.py', ROOT / 'models/cbramod/criss_cross_transformer.py',
                   ROOT / 'experiments/finetune/run_neurogpt_loso.py',
                   ROOT / 'experiments/distill/mirepnet_t2s_data.py',
                   ROOT / 'configs/datasets/BNCI2014001-4.yaml']
        for name in ('cbramod', 'ifnet', 'neurogpt', 'eegnet'):
            tracked.extend([ROOT / f'configs/models/{name}.yaml', ROOT / f'models/{name}/adapter.py'])
        tracked.extend([ROOT / 'models/cbramod/cbramod.py', ROOT / 'models/neurogpt/neurogpt.py',
                        ROOT / 'models/ifnet/ifnet.py', ROOT / 'models/eegnet/residual_eegnet.py'])
        def fingerprint():
            return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
        original = fingerprint()
        while pending or workers:
            for gpu, (process, job, log) in list(workers.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del workers[gpu]
                    if code:
                        failure = f'Worker failed with exit {code}: {job}; see {job[-1] / "worker.log"}'
                        print(failure, flush=True)
                    else:
                        print(f'completed {job[:4]}', flush=True)
            if fingerprint() != original:
                failure = 'Experiment source or configuration changed while queue was running'
            if failure:
                pending.clear()
            for gpu, count in (gpu_counts().items() if not failure else []):
                if not pending:
                    break
                if count >= 3 or gpu in workers:
                    continue
                job = pending.pop(0)
                fold, seed, condition, pair, folder = job
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / 'worker.log').open('a')
                command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_dcm_loso',
                           '--pair', pair, '--condition', condition, '--fold', str(fold),
                           '--seed', str(seed), '--gpu', str(gpu)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                workers[gpu] = (process, job, log)
                print(f'started {job[:4]} on GPU {gpu}, pid={process.pid}', flush=True)
            if pending or workers:
                time.sleep(30)
        if failure:
            raise RuntimeError(failure)
        write_report(root, cfg)
        print('All cells complete and reports written', flush=True)


if __name__ == '__main__':
    main()
