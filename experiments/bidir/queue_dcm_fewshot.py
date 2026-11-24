"""Dispatch the subject-wise DCM experiment with a bounded CPU budget."""
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

from experiments.bidir.queue_dcm_loso import gpu_processes, job_is_running
from experiments.storage import require_external_output, require_report_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dcm.yaml'


def active_workers():
    output = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    active = {}
    for line in output.splitlines():
        parts = line.split()
        if 'experiments.bidir.run_dcm_fewshot' in parts and '--gpu' in parts:
            active[int(parts[0])] = int(parts[parts.index('--gpu') + 1])
    return active


def write_reports(root, cfg):
    rows = []
    for pair, condition, seed, subject in itertools.product(cfg['pairs'], cfg['conditions'], cfg['seeds'], cfg['subjects']):
        path = root / 'runs' / pair / condition / f'subject_{subject}' / f'seed_{seed}' / 'evaluation.json'
        result = json.loads(path.read_text())
        identity = result['identity']
        if (identity['pair'], identity['condition'], identity['seed'], identity['subject']) != (pair, condition, seed, subject):
            raise RuntimeError(f'Result identity mismatch: {path}')
        rows.append([pair, condition, seed, subject, result['big_accuracy'], result['small_accuracy'], str(path)])
    folder = ROOT / 'results/dcm_fewshot_0014_v1'
    target = require_report_output(folder / 'subject_accuracy.csv')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'condition', 'seed', 'subject_zero_based', 'big_accuracy_percent', 'small_accuracy_percent', 'source'])
        writer.writerows(rows)
    os.replace(temporary, target)
    target = require_report_output(folder / 'summary.csv')
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'condition', 'big_accuracy_percent', 'small_accuracy_percent',
                         'big_delta_vs_none_pp', 'small_delta_vs_none_pp',
                         'big_delta_vs_dml_kl_pp', 'small_delta_vs_dml_kl_pp',
                         'big_delta_vs_dml_ds_pp', 'small_delta_vs_dml_ds_pp'])
        for pair in cfg['pairs']:
            means = {}
            for condition in cfg['conditions']:
                selected = [row for row in rows if row[0] == pair and row[1] == condition]
                means[condition] = [sum(row[index] for row in selected) / len(selected) for index in (4, 5)]
            for condition in cfg['conditions']:
                scores = means[condition]
                deltas = [scores[i] - means[base][i] for base in ('none', 'dml_kl', 'dml_ds') for i in (0, 1)]
                writer.writerow([pair, condition] + [f'{value:.2f}' for value in scores + deltas])
    os.replace(temporary, target)


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    root = require_external_output(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for subject, seed, condition, pair in itertools.product(cfg['subjects'], cfg['seeds'], cfg['conditions'], cfg['pairs']):
            folder = root / 'runs' / pair / condition / f'subject_{subject}' / f'seed_{seed}'
            pending.append((subject, seed, condition, pair, folder))
        pending.sort(key=lambda job: (job[-1] / 'evaluation.json').exists())
        files = [CONFIG, ROOT / 'experiments/bidir/run_dcm_fewshot.py', ROOT / 'collab/dcm.py',
                 ROOT / 'data/split.py', ROOT / 'data/loso_source.py', ROOT / 'data/preproc.py',
                 ROOT / 'config.py', ROOT / 'models/base.py', ROOT / 'models/cbramod/criss_cross_transformer.py',
                 ROOT / 'experiments/finetune/run_neurogpt_loso.py', ROOT / 'experiments/distill/run_fewshot_structure_kd.py']
        for model in ('cbramod', 'ifnet', 'neurogpt', 'eegnet'):
            files.extend([ROOT / f'configs/models/{model}.yaml', ROOT / f'models/{model}/adapter.py',
                          ROOT / ('models/eegnet/residual_eegnet.py' if model == 'eegnet' else f'models/{model}/{model}.py')])
        def fingerprint():
            return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        original = fingerprint()
        workers = {}
        failure = None
        while pending or workers:
            for pid, (gpu, process, job, log) in list(workers.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del workers[pid]
                    if code:
                        failure = f'Worker failed with exit {code}: {job}; see {job[-1] / "worker.log"}'
                        print(failure, flush=True)
                    else:
                        print(f'completed {job[:4]}', flush=True)
            if fingerprint() != original:
                failure = 'Experiment source or configuration changed while queue was running'
            if failure: pending.clear()
            devices = gpu_processes() if not failure else {}
            running = active_workers() if not failure else {}
            for gpu, registered in devices.items():
                reserved = sum(assigned == gpu and pid not in registered for pid, assigned in running.items())
                slots = max(0, 3 - len(registered) - reserved)
                while pending and slots and len(running) < 6:
                    available = next((i for i, job in enumerate(pending) if not job_is_running(job[-1])), None)
                    if available is None: break
                    job = pending.pop(available)
                    subject, seed, condition, pair, folder = job
                    folder.mkdir(parents=True, exist_ok=True)
                    log = (folder / 'worker.log').open('a')
                    command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_dcm_fewshot',
                               '--pair', pair, '--condition', condition, '--subject', str(subject),
                               '--seed', str(seed), '--gpu', str(gpu)]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    workers[process.pid] = (gpu, process, job, log)
                    running[process.pid] = gpu
                    slots -= 1
                    print(f'started {job[:4]} on GPU {gpu}, pid={process.pid}', flush=True)
            if pending or workers: time.sleep(30)
        if failure: raise RuntimeError(failure)
        write_reports(root, cfg)
        print('All cells complete and reports written', flush=True)


if __name__ == '__main__':
    main()
