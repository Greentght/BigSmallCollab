"""Schedule fixed-GPU few-shot HFD runs after a matched per-group GPU preflight."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from data.loso_source import sha256_file
from experiments.bidir.dispatch_hfd_shared import suitable_gpus
from experiments.bidir.run_dml_ds_cross_hfd_all import CONFIG, ROOT, implementation_hashes
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, implementation_hashes as old_hashes
from experiments.bidir.summarize_dml_ds_cross_hfd_all import validated_pair
from experiments.distill.queue import QueueStore, file_lock, process_identity, query_gpus, validated_root
from experiments.storage import require_external_output, require_report_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--poll-seconds', type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.poll_seconds < 1:
        parser.error('worker count and polling interval must be positive')
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text())
    base = yaml.safe_load((ROOT / cfg['baseline_config']).read_text())
    external = require_external_output(cfg['output_root'])
    report = require_report_output(Path(cfg['report_root']) / 'controller.json').parent
    report.mkdir(parents=True, exist_ok=True)
    logs = external / 'controller_logs'
    logs.mkdir(parents=True, exist_ok=True)
    stop = external / 'stop'
    check_module = 'experiments.bidir.verify_dml_ds_cross_hfd_all'
    run_module = 'experiments.bidir.run_dml_ds_cross_hfd_all'
    verifier = ROOT / 'experiments/bidir/verify_dml_ds_cross_hfd_all.py'
    hashes, previous = implementation_hashes(), old_hashes()
    impl = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    frozen = dict(config_sha256=sha256_file(config), implementation_sha256=impl,
                  verifier_sha256=sha256_file(verifier))
    common = ['--config', str(config), '--expected-config-sha256', frozen['config_sha256'],
              '--expected-implementation-sha256', impl]
    pair_names = [f'{a}_{b}' for a, b in product(cfg['teachers'], cfg['students'])]
    datasets = list(cfg['datasets'])
    # Cover different montages and architectures in the first round of checks.
    groups = [(dataset, pair_names[(j + i) % len(pair_names)])
              for j in range(len(pair_names)) for i, dataset in enumerate(datasets)]
    old_group = ('BNCI2014004', 'cbramod_eegnet')
    new_groups = [g for g in groups if g != old_group]
    units = [(d, p, s, z) for d, p in new_groups for s, z in product(range(cfg['datasets'][d]), cfg['seeds'])]
    old_units = [(*old_group, s, z) for s, z in product(range(cfg['datasets'][old_group[0]]), cfg['seeds'])]
    controller = report / 'controller.json'
    shared = QueueStore(validated_root())

    def progress(key):
        dataset, pair = key[:2]
        if len(key) == 2:
            return require_report_output(report / 'preflight' / dataset / (pair + '.json'))
        return external / 'runs' / dataset / pair / f'subject_{key[2]}' / f'seed_{key[3]}' / 'progress.json'

    def verified(key):
        path = progress(key)
        if len(key) == 2:
            if not path.is_file():
                return False
            doc = json.loads(path.read_text())
            return (doc.get('status') == 'completed' and doc.get('dataset') == key[0] and
                    doc.get('pair') == key[1] and doc.get('subject') == 0 and
                    doc.get('seed') == cfg['seeds'][0] and doc.get('calibration_only') and
                    doc.get('hfd_gradient_isolation_checked') and doc.get('resume_exactly_matches') and
                    all(doc.get(k) == v for k, v in frozen.items()) and
                    len(doc.get('results', [])) == 2 and all(Path(p).is_file() for p in doc['results']))
        return validated_pair(cfg, base, config, *key, hashes, previous) is not None

    with file_lock(external / 'controller.lock', blocking=False) as lock:
        if controller.is_file() and json.loads(controller.read_text())['identity'] != frozen:
            raise RuntimeError('An existing controller has a different experiment identity')
        if not all(verified(key) for key in old_units):
            raise RuntimeError('The matched original 004 CBraMod/EEGNet group is incomplete')
        done = {key for key in units if (progress(key).parent / 'evaluation.json').is_file() and verified(key)}
        checks = {key for key in new_groups if verified(key)}
        active, failed, attempts = {}, {}, Counter()
        running = {}
        total = len(units) + len(old_units)
        summary = [sys.executable, '-B', '-m', 'experiments.bidir.summarize_dml_ds_cross_hfd_all',
                   '--config', str(config)]
        last_report_size = len(done)
        last_report = time.monotonic()
        fatal = ''

        def update(stage):
            atomic_json(controller, {
                'stage': stage, 'identity': frozen, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                'controller': process_identity(), 'workers_limit': args.workers, 'total_units': total,
                'reused': len(old_units), 'new_completed': len(done), 'completed': len(done) + len(old_units),
                'new_expected': len(units), 'preflight_complete': len(checks), 'preflight_expected': len(new_groups),
                'active': list(active.values()), 'failures': list(failed.values()), 'error': fatal,
            })

        def launch(key, gpu, uuid):
            is_check = len(key) == 2
            dataset, pair = key[:2]
            subject, seed = (0, cfg['seeds'][0]) if is_check else key[2:]
            path = progress(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = (logs / dataset / (pair + '.log')) if is_check else path.parent / 'worker.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', '-B', '-m', check_module if is_check else run_module, *common,
                       '--dataset', dataset, '--pair', pair, '--subject', str(subject), '--seed', str(seed),
                       '--gpu', str(gpu), '--progress', str(path), '--stop-file', str(stop)]
            if is_check:
                command += ['--expected-verifier-sha256', frozen['verifier_sha256']]
            with log.open('a') as stream:
                # A child keeps the controller lock if its parent exits unexpectedly.
                proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        pass_fds=(lock.fileno(),))
            attempts[key] += 1
            running[key] = proc
            active[key] = dict(kind='preflight' if is_check else 'training', dataset=dataset, pair=pair,
                               subject=subject, seed=seed, gpu=gpu, gpu_uuid=uuid, worker=process_identity(proc.pid),
                               status='running', log=str(log), progress=str(path))
            print(f'Start {key} on GPU {gpu} pid={proc.pid}', flush=True)

        update('starting')
        while True:
            for key, proc in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                details = active.pop(key)
                del running[key]
                error = f'exit {code}'
                if code == 0:
                    try:
                        if verified(key):
                            (checks if len(key) == 2 else done).add(key)
                            continue
                        error = 'worker returned without its completed artifacts'
                    except (OSError, ValueError, KeyError) as exc:
                        error = str(exc)
                if code != 75 and attempts[key] >= 2:
                    failed[key] = {**details, 'status': 'failed', 'error': error}
                print(f'Worker {key}: {error} (attempt {attempts[key]})', flush=True)

            if (not fatal and (implementation_hashes() != hashes or
                    sha256_file(config) != frozen['config_sha256'] or sha256_file(verifier) != frozen['verifier_sha256'])):
                fatal = 'Frozen implementation changed'
            waiting = bool(stop.exists() or fatal)
            blocked = {key[:2] for key in failed}
            pending_checks = [g for g in new_groups if g not in checks | blocked and g not in active]
            pending_runs = [k for k in units if k[:2] in checks and k[:2] not in blocked and
                            k not in done and k not in active]
            if not running and (waiting or not (pending_checks or pending_runs)):
                break
            if not waiting and (pending_checks or pending_runs) and len(active) < args.workers:
                try:
                    with shared.lock():
                        devices, reservations = query_gpus(), shared.states()
                        while len(active) < args.workers and (pending_checks or pending_runs):
                            candidates = suitable_gpus(devices, reservations + list(active.values()), 3, 2048)
                            if not candidates:
                                break
                            # As checks finish, let verified groups train alongside remaining checks.
                            checking = sum(item['kind'] == 'preflight' for item in active.values())
                            choose_check = pending_checks and (not pending_runs or checking < (args.workers + 1) // 2)
                            key = (pending_checks if choose_check else pending_runs).pop(0)
                            _, _, gpu, uuid = candidates[0]
                            launch(key, gpu, uuid)
                except RuntimeError as exc:
                    if 'cannot verify GPU availability' not in str(exc):
                        raise
                    print(str(exc), flush=True)
            update('draining' if waiting else 'training' if any(x['kind'] == 'training' for x in active.values())
                   else 'preflight' if active else 'waiting_for_gpu')
            if len(done) != last_report_size and time.monotonic() - last_report >= 300:
                subprocess.run(summary + ['--allow-partial'], cwd=ROOT, check=True)
                last_report_size, last_report = len(done), time.monotonic()
            time.sleep(args.poll_seconds)
        partial = len(done) != len(units) or bool(fatal)
        update('summarizing')
        try:
            subprocess.run(summary + (['--allow-partial'] if partial else []), cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            fatal = str(exc)
            update('error')
            raise
        stage = 'error' if fatal or failed else 'paused' if partial else 'completed'
        update(stage)
        return 1 if stage == 'error' else 75 if stage == 'paused' else 0


if __name__ == '__main__':
    raise SystemExit(main())
