"""Persistent dependency-aware replay of every audited 20-epoch few-shot run."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.bidir.budget100 import ARTIFACT, REPORT, ROOT, runtime_fingerprint, sha, verify_completion, write_json
from experiments.bidir.dispatch_hfd_shared import suitable_gpus
from experiments.distill.queue import QueueStore, file_lock, process_identity, query_gpus, validated_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--poll-seconds', type=int, default=5)
    parser.add_argument('--skip-tuning', action='store_true', help='only for resuming an already completed primary replay')
    args = parser.parse_args()
    if args.workers < 1 or args.poll_seconds < 1: parser.error('Invalid scheduling budget')
    manifest_path = REPORT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest['jobs']
    by_id = {j['id']:j for j in jobs}
    by_out = {j['output']:j['id'] for j in jobs}
    frozen, source_files = runtime_fingerprint()
    identity = {'manifest_sha256': sha(manifest_path), 'runtime_sha256': frozen, 'joint_epochs': 100}
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    shared = QueueStore(validated_root())
    controller = REPORT / 'controller.json'
    with file_lock(ARTIFACT / 'controller.lock', blocking=False) as lock:
        if controller.exists() and json.loads(controller.read_text())['identity'] != identity:
            raise ValueError('Replay manifest or numerical code changed; do not silently restart')
        done = {j['id'] for j in jobs if verify_completion(j)}
        failed, active, processes = {}, {}, {}
        attempts = Counter()
        last_check, last_summary = time.monotonic(), time.monotonic()
        fatal = ''
        def update(stage):
            counts = Counter(by_id[k]['family'] for k in done)
            write_json(controller, {'stage': stage, 'identity': identity,
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'controller': process_identity(),
                'completed': len(done), 'total_units': len(jobs), 'all_joint_epochs': 100,
                'family_completed': dict(counts), 'family_expected': manifest['family_counts'],
                'workers_limit': args.workers, 'gpu_process_limit': 3, 'fixed_gpu_per_training': True,
                'active': list(active.values()), 'failures': list(failed.values()), 'error': fatal,
                'tuning_after_primary': not args.skip_tuning})
        def summarize():
            subprocess.run([sys.executable, '-B', '-m', 'experiments.bidir.summarize_budget100'], cwd=ROOT, check=True)
        def launch(job, gpu, uuid):
            out = Path(job['output']); out.mkdir(parents=True, exist_ok=True)
            spec = out / 'budget100_job.json'
            if spec.exists() and json.loads(spec.read_text()) != job: raise ValueError('Job spec differs')
            write_json(spec, job)
            command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_budget100_job',
                       '--job', str(spec), '--gpu', str(gpu), '--expected-fingerprint', frozen]
            with (out / 'worker.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),))
            key = job['id']; attempts[key] += 1; processes[key] = process
            active[key] = {'status': 'running', 'family': job['family'], 'dataset': job['dataset'],
                'pair': job['pair'], 'condition': job['condition'], 'subject': job['subject'], 'seed': job['seed'],
                'gpu': gpu, 'gpu_uuid': uuid, 'worker': process_identity(process.pid),
                'output': str(out), 'log': str(out / 'worker.log'), 'joint_epochs': 100}
            print(f"Start100 {job['family']} {job['dataset']} {job['pair']} {job['condition']} "
                  f"subject={job['subject']} seed={job['seed']} gpu={gpu} pid={process.pid}", flush=True)
        update('starting')
        while True:
            for key, process in list(processes.items()):
                code = process.poll()
                if code is None: continue
                details = active.pop(key); del processes[key]
                if code == 0 and verify_completion(by_id[key]):
                    done.add(key)
                elif attempts[key] >= 2:
                    failed[key] = {**details, 'exit_code': code, 'error': 'inspect worker.log'}
                else: print('Retry100', key, code, flush=True)
            if time.monotonic() - last_check >= 30:
                if runtime_fingerprint()[0] != frozen or sha(manifest_path) != identity['manifest_sha256']:
                    fatal = 'Frozen implementation or manifest changed'
                last_check = time.monotonic()
            draining = bool(fatal or (ARTIFACT / 'stop').exists())
            pending = [j for j in jobs if j['id'] not in done and j['id'] not in active and j['id'] not in failed]
            eligible = [j for j in pending if not j['dependency'] or by_out[j['dependency']] in done]
            if not processes and (draining or not pending): break
            if not processes and pending and not eligible:
                fatal = 'All remaining jobs depend on failed baselines'
                break
            if not draining and len(active) < args.workers:
                with shared.lock():
                    devices, reservations = query_gpus(), shared.states()
                    while eligible and len(active) < args.workers:
                        candidates = suitable_gpus(devices, reservations + list(active.values()), 3, 4096)
                        if not candidates: break
                        _, _, gpu, uuid = candidates[0]
                        launch(eligible.pop(0), gpu, uuid)
            update('draining' if draining else 'training' if active else 'waiting_for_gpu')
            if time.monotonic() - last_summary >= 300:
                summarize(); last_summary = time.monotonic()
            time.sleep(args.poll_seconds)
        summarize()
        complete = len(done) == len(jobs) and not failed and not fatal
        if complete and not args.skip_tuning:
            update('tuning_100epoch')
            code = subprocess.call([sys.executable, '-u', '-B', '-m',
                                    'experiments.bidir.budget100_tuning_queue'], cwd=ROOT)
            if code: fatal = f'100-epoch tuning exited {code}'
        update('completed' if complete and not fatal else 'error' if fatal or failed else 'paused')
        return 0 if complete and not fatal else 1


if __name__ == '__main__': raise SystemExit(main())
