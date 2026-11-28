"""Persistent fixed-GPU queue for the terminal cross-operator HFD pilot."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml
import torch

from data.loso_source import sha256_file
from experiments.bidir.dispatch_hfd_shared import suitable_gpus
from experiments.bidir.run_terminal_hfd_fewshot import CONFIG, ROOT, fingerprint, run_folder
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json
from experiments.distill.queue import QueueStore, file_lock, process_identity, query_gpus, validated_root
from experiments.storage import REPORTS_ROOT, require_external_output, require_report_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--poll-seconds', type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.poll_seconds < 1:
        parser.error('Workers and polling interval must be positive')
    if not torch.cuda.is_available():
        parser.error('Use the CUDA environment: /home/lixinli/anaconda3/envs/mirepnet/bin/python')
    cfg = yaml.safe_load(args.config.read_text())
    external = require_external_output(cfg['output_root'])
    external.mkdir(parents=True, exist_ok=True)
    report = require_report_output(REPORTS_ROOT / cfg['report_name'] / 'controller.json').parent
    report.mkdir(parents=True, exist_ok=True)
    verifier = ROOT / 'experiments/bidir/verify_terminal_hfd.py'
    frozen = {'implementation_sha256': fingerprint(), 'config_sha256': sha256_file(args.config),
              'verifier_sha256': sha256_file(verifier)}
    controller = report / 'controller.json'
    shared = QueueStore(validated_root())
    groups = [(d, m) for m in cfg['methods'] for d in cfg['datasets']]
    # Cover datasets and methods before cycling additional subjects and seeds.
    units = [(d, m, s, z) for s in range(max(cfg['datasets'].values())) for z in cfg['seeds']
             for d, m in groups if s < cfg['datasets'][d]]

    def path(key):
        if len(key) == 2:
            return report / 'preflight' / key[0] / (key[1] + '.json')
        return run_folder(cfg, *key) / 'progress.json'

    def verified(key):
        p = path(key)
        if len(key) == 2:
            if not p.is_file(): return False
            doc = json.loads(p.read_text())
            return (doc.get('status') == 'completed' and doc.get('dataset') == key[0] and
                    doc.get('method') == key[1] and doc.get('calibration_only') and
                    doc.get('gradient_isolation_checked') and doc.get('resume_exactly_matches') and
                    doc.get('native_terminal_operators_checked') and doc.get('native_rng_and_peer_buffers_preserved') and
                    all(doc.get(k) == frozen[k] for k in ('implementation_sha256', 'config_sha256')))
        p = p.parent / 'evaluation.json'
        if not p.is_file(): return False
        result = json.loads(p.read_text())
        i = result['identity']
        if ((i['dataset'], i['method'], i['subject'], i['seed']) != key or i['smoke'] or
                result['evaluation_split'] != 'test' or
                any(i[k] != frozen[k] for k in ('implementation_sha256', 'config_sha256')) or
                result['predictions_sha256'] != sha256_file(p.parent / 'predictions.npz') or
                not (p.parent / 'training.json').is_file()):
            raise ValueError(f'Invalid completed result: {p}')
        return True

    with file_lock(external / 'controller.lock', blocking=False) as lock:
        if controller.is_file() and json.loads(controller.read_text())['identity'] != frozen:
            raise RuntimeError('Existing controller belongs to a different implementation')
        checks = {g for g in groups if verified(g)}
        done = {u for u in units if verified(u)}
        active, processes, failed, attempts = {}, {}, {}, Counter()
        fatal = ''
        last_report = time.monotonic()

        def update(stage):
            atomic_json(controller, {'stage': stage, 'identity': frozen,
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'controller': process_identity(),
                'total_units': len(units), 'completed': len(done), 'reused_dml_ds_baselines': 54,
                'preflight_complete': len(checks), 'preflight_expected': len(groups),
                'workers_limit': args.workers, 'gpu_process_limit': 3, 'fixed_gpu_per_training': True,
                'active': list(active.values()), 'failures': list(failed.values()), 'error': fatal})

        def launch(key, gpu, uuid):
            check = len(key) == 2
            p = path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            log = (external / 'preflight_logs' / key[0] / (key[1] + '.log')) if check else p.parent / 'worker.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.' + (
                'verify_terminal_hfd' if check else 'run_terminal_hfd_fewshot'),
                '--config', str(args.config), '--dataset', key[0], '--method', key[1],
                '--gpu', str(gpu), '--progress', str(p), '--expected-fingerprint', frozen['implementation_sha256']]
            if not check:
                cmd += ['--subject', str(key[2]), '--seed', str(key[3]), '--stop-file', str(external / 'stop'),
                        '--expected-config-sha256', frozen['config_sha256']]
            with log.open('a') as stream:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        pass_fds=(lock.fileno(),))
            attempts[key] += 1
            processes[key] = proc
            active[key] = {'status': 'running', 'kind': 'preflight' if check else 'training',
                           'dataset': key[0], 'method': key[1], 'subject': 0 if check else key[2],
                           'seed': cfg['seeds'][0] if check else key[3], 'gpu': gpu, 'gpu_uuid': uuid,
                           'worker': process_identity(proc.pid), 'log': str(log), 'progress': str(p)}
            print(f'Start {key} GPU={gpu} pid={proc.pid}', flush=True)

        def summarize():
            subprocess.run([sys.executable, '-B', '-m', 'experiments.bidir.summarize_terminal_hfd',
                            '--config', str(args.config)], cwd=ROOT, check=True)

        update('starting')
        while True:
            for key, proc in list(processes.items()):
                code = proc.poll()
                if code is None: continue
                details = active.pop(key)
                del processes[key]
                if code == 0 and verified(key):
                    (checks if len(key) == 2 else done).add(key)
                elif code != 75 and attempts[key] >= 2:
                    failed[key] = {**details, 'status': 'failed', 'error': f'exit {code}'}
                else:
                    print(f'Retry {key}: exit {code}', flush=True)
            if (fingerprint() != frozen['implementation_sha256'] or
                    sha256_file(args.config) != frozen['config_sha256'] or
                    sha256_file(verifier) != frozen['verifier_sha256']):
                fatal = 'Frozen implementation changed'
            draining = bool((external / 'stop').exists() or fatal)
            blocked = {k[:2] for k in failed}
            pending_checks = [g for g in groups if g not in checks | blocked and g not in active]
            pending_runs = [u for u in units if u[:2] in checks and u[:2] not in blocked and
                            u not in done and u not in active]
            if not processes and (draining or not (pending_checks or pending_runs)):
                break
            if not draining and len(active) < args.workers and (pending_checks or pending_runs):
                with shared.lock():
                    devices, reservations = query_gpus(), shared.states()
                    while len(active) < args.workers and (pending_checks or pending_runs):
                        candidates = suitable_gpus(devices, reservations + list(active.values()), 3, 4096)
                        if not candidates: break
                        checking = sum(v['kind'] == 'preflight' for v in active.values())
                        pick_check = pending_checks and (not pending_runs or checking < 3)
                        key = (pending_checks if pick_check else pending_runs).pop(0)
                        _, _, gpu, uuid = candidates[0]
                        launch(key, gpu, uuid)
            update('draining' if draining else 'training' if any(
                v['kind'] == 'training' for v in active.values()) else 'preflight' if active else 'waiting_for_gpu')
            if done and time.monotonic() - last_report >= 300:
                summarize()
                last_report = time.monotonic()
            time.sleep(args.poll_seconds)
        summarize()
        stage = 'error' if fatal or failed else 'completed' if len(done) == len(units) else 'paused'
        update(stage)
        return 0 if stage == 'completed' else 1 if stage == 'error' else 75


if __name__ == '__main__':
    raise SystemExit(main())
