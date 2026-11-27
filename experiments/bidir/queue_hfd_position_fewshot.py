"""Run all requested HFD positions after each dataset/pair GPU preflight."""
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
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.bidir.run_hfd_position_baselines import DATASETS, OUTPUT, PAIRS, ROOT
from experiments.bidir.run_hfd_position_fewshot import fingerprint
from experiments.distill.queue import QueueStore, file_lock, process_identity, query_gpus, validated_root
from experiments.storage import require_external_output, require_report_output


def main():
    report = require_report_output(ROOT / 'results/hfd_position_fewshot_v1/position_queue.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    external = require_external_output(OUTPUT)
    logs = external / 'controller_logs'
    logs.mkdir(parents=True, exist_ok=True)
    shared = QueueStore(validated_root())
    run_hash = fingerprint()
    tag = 'gpu' + run_hash[:12]
    groups = list(product(DATASETS, PAIRS))
    preflights = [(d, p, 0, 666, 'C6', True) for d, p in groups]
    jobs = []
    for dataset, pair, subject, seed in product(DATASETS, PAIRS, range(9), (666, 667, 668)):
        conditions = ('C0', 'C2', 'C3', 'C4', 'C5', 'C6') if pair == 'neurogpt_ifnet' else ('C3', 'C4', 'C5', 'C6')
        jobs.extend((dataset, pair, subject, seed, condition, False) for condition in conditions)
    running, active, failures = {}, {}, {}
    attempts = Counter()

    def folder(job):
        dataset, pair, subject, seed, condition, smoke = job
        root = external / ('smoke' if smoke else 'runs')
        if smoke:
            root /= tag
        return root / dataset / pair / condition / f'subject_{subject}' / f'seed_{seed}'

    def complete(job):
        path, pred = folder(job) / 'evaluation.json', folder(job) / 'predictions.npz'
        if not path.is_file() or not pred.is_file():
            return False
        result = json.loads(path.read_text())
        ident = result['identity']
        d, p, s, z, c, smoke = job
        return ((ident['dataset'], ident['pair'], ident['subject'], ident['seed'],
                 ident['condition'], ident['smoke']) == job and ident['implementation_sha256'] == run_hash and
                result['evaluation_split'] == ('calibration' if smoke else 'test') and
                result['predictions_sha256'] == sha256_file(pred))

    def occupied(job):
        try:
            with file_lock(folder(job) / 'run.lock', blocking=False):
                return False
        except RuntimeError:
            return True

    with file_lock(external / 'position_controller.lock', blocking=False) as lock:
        done = {job for job in jobs if complete(job)}
        checked = {job for job in preflights if complete(job)}
        while True:
            for job, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                details = active.pop(job)
                del running[job]
                if code == 0 and complete(job):
                    (checked if job[-1] else done).add(job)
                    print(f'Completed {job}', flush=True)
                elif attempts[job] >= 2:
                    failures[job] = {**details, 'exit_code': code}
                    print(f'Failed {job} exit={code}', flush=True)
                else:
                    print(f'Retrying {job} exit={code}', flush=True)
            if fingerprint() != run_hash:
                raise RuntimeError('Frozen position implementation changed')
            blocked = {(j[0], j[1]) for j in failures if j[-1]}
            ready = {(j[0], j[1]) for j in checked}
            pre_pending = [j for j in preflights if j not in checked and j not in running and j not in failures]
            pending = [j for j in jobs if j not in done and j not in running and j not in failures and
                       (j[0], j[1]) not in blocked and (j[0], j[1]) in ready]
            remaining = [j for j in jobs if j not in done and j not in failures and (j[0], j[1]) not in blocked]
            if not pre_pending and not remaining and not running:
                break
            with shared.lock():
                devices, reservations = query_gpus(), shared.states()
                while len(running) < 3:
                    pool = pre_pending if pre_pending else pending
                    available = next((j for j in pool if not occupied(j)), None)
                    candidates = suitable_gpus(devices, reservations + list(active.values()), 3, 2048)
                    if available is None or not candidates:
                        break
                    pool.remove(available)
                    dataset, pair, subject, seed, condition, smoke = available
                    _, _, gpu, uuid = candidates[0]
                    path = folder(available)
                    path.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_hfd_position_fewshot',
                               '--dataset', dataset, '--pair', pair, '--subject', str(subject), '--seed', str(seed),
                               '--condition', condition, '--gpu', str(gpu), '--expected-fingerprint', run_hash]
                    if smoke:
                        command += ['--smoke', '--smoke-id', tag]
                    with (path / 'worker.log').open('a') as stream:
                        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                pass_fds=(lock.fileno(),))
                    running[available] = proc
                    active[available] = {'status': 'running', 'gpu': gpu, 'gpu_uuid': uuid,
                                         'worker': process_identity(proc.pid), 'job': list(available)}
                    attempts[available] += 1
                    print(f'Started {available} on GPU {gpu} pid={proc.pid}', flush=True)
            atomic_json(report, {'stage': 'training', 'updated_utc': datetime.now(timezone.utc).isoformat(),
                                 'implementation_sha256': run_hash, 'completed': len(done), 'total': len(jobs),
                                 'preflight_complete': len(checked), 'preflight_total': len(preflights),
                                 'active': list(active.values()), 'failures': list(failures.values()),
                                 'reused_C2': 'CBraMod–EEGNet and MIRepNet–ADFCNN from completed prior run'})
            time.sleep(5)
        atomic_json(report, {'stage': 'error' if failures else 'completed',
                             'updated_utc': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': run_hash,
                             'completed': len(done), 'total': len(jobs), 'preflight_complete': len(checked),
                             'preflight_total': len(preflights), 'active': [], 'failures': list(failures.values())})
        return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
