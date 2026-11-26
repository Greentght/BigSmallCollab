"""Run the corrected HFD preflight and 27 paired few-shot runs serially."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import CONFIG, ROOT, atomic_json, implementation_hashes
from experiments.distill.queue import file_lock, query_gpus
from experiments.storage import require_external_output, require_report_output


def choose_gpu(index):
    """The requested fixed GPU must have fewer than three CUDA processes."""
    gpu = query_gpus().get(index)
    if not gpu:
        raise RuntimeError(f'GPU {index} is not present')
    if len(gpu['pids']) >= 3 or gpu['total_mb'] - gpu['used_mb'] < 2048:
        return None
    return {'index': index, 'uuid': gpu['uuid'], 'processes': len(gpu['pids']),
            'free_mb': gpu['total_mb'] - gpu['used_mb']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('poll interval must be at least five seconds')
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if (cfg['dataset'] != 'BNCI2014004' or cfg['pair'] != 'cbramod_eegnet' or
            cfg['condition'] != 'dml_ds_cross_hfd' or cfg['subjects'] != list(range(9)) or
            cfg['seeds'] != [666, 667, 668]):
        raise ValueError('Unexpected experiment identity')
    report = require_report_output(Path(cfg['report_root']) / 'controller.json').parent
    external = require_external_output(Path(cfg['output_root']))
    report.mkdir(parents=True, exist_ok=True)
    log_dir = external / 'controller_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    verifier = ROOT / 'experiments/bidir/verify_dml_ds_cross_hfd_preflight.py'
    config_hash = sha256_file(config_path)
    implementation = hashlib.sha256(json.dumps(implementation_hashes(), sort_keys=True).encode()).hexdigest()
    verifier_hash = sha256_file(verifier)
    identity = {'config_sha256': config_hash, 'implementation_sha256': implementation,
                'verifier_sha256': verifier_hash, 'gpu': args.gpu}
    controller_path = report / 'controller.json'
    progress_path = report / 'training_progress.json'

    def update(stage, **extra):
        value = {'stage': stage, 'identity': identity, **extra}
        atomic_json(controller_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    def wait_for_gpu(stage):
        while True:
            gpu = choose_gpu(args.gpu)
            if gpu:
                return gpu
            update('waiting_for_gpu', next_stage=stage)
            time.sleep(args.poll_seconds)

    def unchanged():
        digest = hashlib.sha256(json.dumps(implementation_hashes(), sort_keys=True).encode()).hexdigest()
        if sha256_file(config_path) != config_hash or digest != implementation or sha256_file(verifier) != verifier_hash:
            raise RuntimeError('Configuration or implementation changed after preflight')

    common_hash_args = ['--config', str(config_path), '--expected-config-sha256', config_hash,
                        '--expected-implementation-sha256', implementation]
    with file_lock(report / 'controller.lock', blocking=False):
        old = json.loads(controller_path.read_text()) if controller_path.exists() else None
        if old and old.get('identity') != identity:
            # A preflight correction may replace a failed configuration only
            # before any formal run exists. Preserve the failed controller.
            if old.get('stage') != 'error' or (external / 'runs').exists():
                raise ValueError('Existing controller has a different experiment identity')
            previous = old.get('identity', {}).get('implementation_sha256', 'unknown')[:12]
            atomic_json(report / f'failed_preflight_{previous}.json', old)
        try:
            preflight_progress = report / 'preflight_progress.json'
            def valid_preflight(check):
                return (check.get('status') == 'completed' and check.get('calibration_only') and
                        check.get('resume_exactly_matches') and check.get('hfd_gradient_isolation_checked') and
                        check.get('config_sha256') == config_hash and
                        check.get('implementation_sha256') == implementation and
                        check.get('verifier_sha256') == verifier_hash)

            check = json.loads(preflight_progress.read_text()) if preflight_progress.is_file() else {}
            if not valid_preflight(check):
                gpu = wait_for_gpu('preflight')
                update('preflight', gpu=gpu)
                command = [sys.executable, '-B', '-m',
                           'experiments.bidir.verify_dml_ds_cross_hfd_preflight', *common_hash_args,
                           '--expected-verifier-sha256', verifier_hash, '--subject', '0', '--seed', '666',
                           '--gpu', str(args.gpu), '--progress', str(preflight_progress),
                           '--stop-file', str(external / 'stop_preflight')]
                with (log_dir / 'preflight.log').open('a', encoding='utf8') as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'GPU preflight failed ({result.returncode}); see {log_dir / "preflight.log"}')
                check = json.loads(preflight_progress.read_text())
            if not valid_preflight(check):
                raise RuntimeError('GPU preflight did not verify the frozen experiment')
            unchanged()
            update('preflight_passed', gpu=args.gpu)
            if args.preflight_only:
                return 0
            for number, (subject, seed) in enumerate(product(cfg['subjects'], cfg['seeds']), 1):
                unchanged()
                gpu = wait_for_gpu('formal')
                out = external / 'runs' / f'subject_{subject}' / f'seed_{seed}' / 'evaluation.json'
                update('training', gpu=gpu, subject=subject, seed=seed, run=number, total=27,
                       existing_result=out.is_file())
                cmd = [sys.executable, '-B', '-m', 'experiments.bidir.run_dml_ds_cross_hfd_fewshot',
                       *common_hash_args, '--subject', str(subject), '--seed', str(seed),
                       '--gpu', str(args.gpu), '--progress', str(progress_path),
                       '--stop-file', str(external / 'stop')]
                with (log_dir / f'subject_{subject}_seed_{seed}.log').open('a', encoding='utf8') as stream:
                    result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode == 75:
                    update('paused', subject=subject, seed=seed, run=number, total=27)
                    return 75
                if result.returncode:
                    raise RuntimeError(f'Run {subject}/{seed} failed ({result.returncode}); see {stream.name}')
            unchanged()
            update('summarizing', total=27)
            subprocess.run([sys.executable, '-B', '-m',
                            'experiments.bidir.summarize_dml_ds_cross_hfd_fewshot',
                            '--config', str(config_path)], cwd=ROOT, check=True)
            update('completed', total=27, report=str(report))
            return 0
        except Exception as exc:
            update('error', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
