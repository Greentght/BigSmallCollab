"""Dispatch independent few-shot baselines for the HFD position study."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

from data.loso_source import sha256_file
from experiments.bidir.dispatch_hfd_shared import suitable_gpus
from experiments.bidir.run_hfd_position_baselines import BASE, DATASETS, OUTPUT, PAIRS, ROOT
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.distill.queue import QueueStore, file_lock, process_identity, query_gpus, validated_root
from experiments.storage import require_external_output, require_report_output


def main():
    report = require_report_output(ROOT / 'results/hfd_position_fewshot_v1/baseline_queue.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    external = require_external_output(OUTPUT)
    log_dir = external / 'controller_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    shared = QueueStore(validated_root())
    runner = ROOT / 'experiments/bidir/run_hfd_position_baselines.py'
    fingerprint = {'runner': sha256_file(runner), 'base_config': sha256_file(BASE)}
    jobs = []
    for dataset, pair, subject, seed in product(DATASETS, PAIRS, range(9), (666, 667, 668)):
        usual_epochs = 100 if pair == 'neurogpt_ifnet' else 20
        jobs.append((dataset, pair, subject, seed, 'ds', usual_epochs))
        jobs.append((dataset, pair, subject, seed, 'ce', 20 if usual_epochs == 100 else 100))
        if pair == 'neurogpt_ifnet':
            jobs.append((dataset, pair, subject, seed, 'ce', 100))
    running, active, failures = {}, {}, {}
    attempts = Counter()

    def folder(job):
        dataset, pair, subject, seed, condition, epochs = job
        return external / 'runs' / dataset / pair / condition / str(epochs) / f'subject_{subject}' / f'seed_{seed}'

    def occupied(job):
        try:
            with file_lock(folder(job) / 'run.lock', blocking=False):
                return False
        except RuntimeError:
            return True

    def complete(job):
        path = folder(job) / 'evaluation.json'
        pred = folder(job) / 'predictions.npz'
        if not path.is_file() or not pred.is_file():
            return False
        result = json.loads(path.read_text())
        identity = result['identity']
        dataset, pair, subject, seed, condition, epochs = job
        return ((identity['dataset'], identity['pair'], identity['subject'], identity['seed'],
                 identity['condition'], identity['epochs']) == job and
                identity['runner_sha256'] == fingerprint['runner'] and
                result['predictions_sha256'] == sha256_file(pred))

    with file_lock(external / 'baseline_controller.lock', blocking=False) as lock:
        done = {job for job in jobs if complete(job)}
        while True:
            for job, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                details = active.pop(job)
                del running[job]
                if code == 0 and complete(job):
                    done.add(job)
                    print(f'Completed {job}', flush=True)
                elif attempts[job] >= 2:
                    failures[job] = {**details, 'exit_code': code}
                    print(f'Failed {job} exit={code}', flush=True)
                else:
                    print(f'Retrying {job} exit={code}', flush=True)
            if sha256_file(runner) != fingerprint['runner'] or sha256_file(BASE) != fingerprint['base_config']:
                raise RuntimeError('Baseline runner or configuration changed while running')
            # A separately started first unit may still hold its lock.
            for job in jobs:
                if job not in done and job not in running and (folder(job) / 'evaluation.json').is_file():
                    if complete(job):
                        done.add(job)
            pending = [job for job in jobs if job not in done and job not in running and job not in failures]
            if not pending and not running:
                break
            with shared.lock():
                devices, reservations = query_gpus(), shared.states()
                while len(running) < 4:
                    available = next((job for job in pending if not occupied(job)), None)
                    candidates = suitable_gpus(devices, reservations + list(active.values()), 3, 2048)
                    if available is None or not candidates:
                        break
                    pending.remove(available)
                    dataset, pair, subject, seed, condition, epochs = available
                    _, _, gpu, uuid = candidates[0]
                    path = folder(available)
                    path.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_hfd_position_baselines',
                               '--dataset', dataset, '--pair', pair, '--subject', str(subject), '--seed', str(seed),
                               '--condition', condition, '--epochs', str(epochs), '--gpu', str(gpu)]
                    with (path / 'worker.log').open('a') as stream:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                   pass_fds=(lock.fileno(),))
                    running[available] = process
                    active[available] = {'status': 'running', 'gpu': gpu, 'gpu_uuid': uuid,
                                         'worker': process_identity(process.pid), 'job': list(available)}
                    attempts[available] += 1
                    print(f'Started {available} on GPU {gpu} pid={process.pid}', flush=True)
            atomic_json(report, {'stage': 'training', 'updated_utc': datetime.now(timezone.utc).isoformat(),
                                 'fingerprint': fingerprint, 'completed': len(done), 'total': len(jobs),
                                 'active': list(active.values()), 'failures': list(failures.values())})
            time.sleep(5)
        atomic_json(report, {'stage': 'error' if failures else 'completed',
                             'updated_utc': datetime.now(timezone.utc).isoformat(), 'fingerprint': fingerprint,
                             'completed': len(done), 'total': len(jobs), 'active': [],
                             'failures': list(failures.values())})
        return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
