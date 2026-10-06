#!/usr/bin/env python
"""Wait for nonzero GPUs, then run the paired NPY-source CBraMod control."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/reproductions/loso_config_alignment_v2'
EXEC = RESULTS / 'execution_logs'
LOGS = EXEC / 'npy_source_control_0014'
WORKER = ROOT / 'experiments/finetune/run_loso_config_alignment.py'
CONDA = '/home/lixinli/anaconda3/bin/conda'
ALLOWED_GPUS = list(range(1, 10))
MAX_CONCURRENT = 3
MIN_FREE_MIB = 10000
MAX_UTIL = 40
POLL_SECONDS = 30
SEEDS = (0, 1, 2)
PROFILE = 'reference_aligned_npy'
DATASET = 'BNCI2014001-4'
MODEL = 'cbramod'


def log(message: str) -> None:
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(line, flush=True)
    EXEC.mkdir(parents=True, exist_ok=True)
    with (EXEC / 'npy_source_control_0014_dispatcher.log').open('a') as f:
        f.write(line + '\n')


def gpu_snapshot() -> list[dict]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 3:
            continue
        rows.append({'gpu': int(fields[0]), 'utilization_pct': int(fields[1]),
                     'free_memory_mib': int(fields[2])})
    return rows


def available_gpu(occupied: set[int]) -> int | None:
    candidates = [x for x in gpu_snapshot()
                  if x['gpu'] in ALLOWED_GPUS and x['gpu'] != 0
                  and x['gpu'] not in occupied
                  and x['utilization_pct'] <= MAX_UTIL
                  and x['free_memory_mib'] >= MIN_FREE_MIB]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x['utilization_pct'], -x['free_memory_mib'], x['gpu']))
    return candidates[0]['gpu']


def write_status(phase: str, pending: list[int], active: list[dict], **extra) -> None:
    EXEC.mkdir(parents=True, exist_ok=True)
    status = {
        'phase': phase, 'updated_at_epoch_s': time.time(),
        'dataset': DATASET, 'model': MODEL, 'profile': PROFILE,
        'seeds': list(SEEDS), 'pending_seeds': pending,
        'active_jobs': [
            {'seed': x['seed'], 'gpu': x['gpu'], 'pid': x['proc'].pid,
             'elapsed_seconds': time.time() - x['started']}
            for x in active
        ],
        'gpu_snapshot': gpu_snapshot(), **extra,
    }
    path = EXEC / 'npy_source_control_0014_dispatcher_status.json'
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def command(seed: int, gpu: int, preflight: bool = False) -> list[str]:
    args = [CONDA, 'run', '--no-capture-output', '-n', 'cbramod', 'python', '-u',
            str(WORKER), '--profile', PROFILE, '--dataset', DATASET,
            '--model', MODEL, '--seed', str(seed), '--gpu', str(gpu)]
    if preflight:
        args += ['--preflight-steps', '1']
    return args


def run_preflight() -> None:
    while True:
        gpu = available_gpu(set())
        if gpu is None:
            write_status('waiting_for_preflight_gpu', [], [])
            log('waiting for a nonzero GPU (utilization <=40%, free memory >=10000 MiB) for the one-batch preflight')
            time.sleep(POLL_SECONDS)
            continue
        path = LOGS / f'preflight_seed0_gpu{gpu}.log'
        LOGS.mkdir(parents=True, exist_ok=True)
        with path.open('ab', buffering=0) as out:
            proc = subprocess.Popen(command(0, gpu, True), cwd=ROOT,
                                    env={**os.environ, 'PYTHONPATH': str(ROOT)},
                                    stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    close_fds=True)
            log(f'started one-batch preflight pid={proc.pid} gpu={gpu}; log={path}')
            write_status('preflight_running', [], [{'seed': 0, 'gpu': gpu,
                                                    'proc': proc, 'started': time.time()}])
            code = proc.wait()
        if code == 0:
            log(f'preflight complete on gpu={gpu}')
            return
        raise RuntimeError(f'preflight failed with code={code}; inspect {path}')


def aggregate() -> dict:
    base = RESULTS / PROFILE / DATASET / MODEL
    reference = RESULTS / 'reference_aligned' / DATASET / MODEL
    per_seed = {}
    pairing = []
    for seed in SEEDS:
        npy_summary_path = base / f'seed_{seed}' / 'seed_summary.json'
        ref_summary_path = reference / f'seed_{seed}' / 'seed_summary.json'
        npy_summary = json.loads(npy_summary_path.read_text())
        ref_summary = json.loads(ref_summary_path.read_text())
        per_seed[str(seed)] = {
            'npy_accuracy': npy_summary['subject_equal_mean']['accuracy'],
            'moabb_accuracy': ref_summary['subject_equal_mean']['accuracy'],
            'accuracy_delta_npy_minus_moabb': (
                npy_summary['subject_equal_mean']['accuracy']
                - ref_summary['subject_equal_mean']['accuracy']),
            'npy_balanced_accuracy': npy_summary['subject_equal_mean']['balanced_accuracy'],
            'npy_kappa': npy_summary['subject_equal_mean']['kappa'],
            'npy_macro_f1': npy_summary['subject_equal_mean']['macro_f1'],
        }
        for subject in range(1, 10):
            npy_r = json.loads((base / f'seed_{seed}' / f'subject_{subject:02d}'
                                / 'result.json').read_text())
            ref_r = json.loads((reference / f'seed_{seed}' / f'subject_{subject:02d}'
                                / 'result.json').read_text())
            pairing.append({
                'seed': seed, 'held_out_subject': subject,
                'npy_accuracy': npy_r['metrics']['accuracy'],
                'moabb_accuracy': ref_r['metrics']['accuracy'],
                'accuracy_delta_npy_minus_moabb': (
                    npy_r['metrics']['accuracy'] - ref_r['metrics']['accuracy']),
                'initial_state_matches': (
                    npy_r['initial_state_sha256'] == ref_r['initial_state_sha256']),
                'first_epoch_batch_order_matches': (
                    npy_r['first_epoch_train_order_sha256']
                    == ref_r['first_epoch_train_order_sha256']),
                'same_trials_and_labels': (
                    npy_r['train_trials'] == ref_r['train_trials']
                    and npy_r['test_trials'] == ref_r['test_trials']
                    and npy_r['input_shape'] == ref_r['input_shape']),
            })
    scores = [per_seed[str(seed)]['npy_accuracy'] for seed in SEEDS]
    deltas = [per_seed[str(seed)]['accuracy_delta_npy_minus_moabb'] for seed in SEEDS]
    import statistics
    all_pair_flags = all(
        row['initial_state_matches'] and row['first_epoch_batch_order_matches']
        and row['same_trials_and_labels'] for row in pairing)
    return {
        'status': 'complete', 'dataset': DATASET, 'model': MODEL,
        'profile': PROFILE, 'folds_per_seed': 9, 'seeds': list(SEEDS),
        'npy_accuracy_seed_mean': statistics.mean(scores),
        'npy_accuracy_seed_sample_std': statistics.stdev(scores),
        'moabb_accuracy_seed_mean': statistics.mean(
            per_seed[str(seed)]['moabb_accuracy'] for seed in SEEDS),
        'mean_paired_delta_npy_minus_moabb': statistics.mean(deltas),
        'paired_delta_seed_sample_std': statistics.stdev(deltas),
        'all_pair_checks_match': all_pair_flags,
        'per_seed': per_seed, 'per_fold': pairing,
    }


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    # One-batch preflight before consuming the full 27-fold-seed budget.
    run_preflight()
    pending = deque(SEEDS)
    active: list[dict] = []
    completed: list[int] = []
    attempts = {seed: 0 for seed in SEEDS}
    while pending or active:
        for item in list(active):
            code = item['proc'].poll()
            if code is None:
                continue
            active.remove(item)
            item['log_handle'].close()
            if code == 0:
                completed.append(item['seed'])
                log(f"seed {item['seed']} complete on gpu={item['gpu']} elapsed_min="
                    f"{(time.time()-item['started'])/60:.1f}")
            elif attempts[item['seed']] < 1:
                attempts[item['seed']] += 1
                pending.appendleft(item['seed'])
                log(f"seed {item['seed']} failed code={code}; retrying once; log={item['log_path']}")
            else:
                write_status('failed', list(pending), active, completed_seeds=completed,
                             failed_seed=item['seed'], return_code=code,
                             failed_log=str(item['log_path']))
                raise RuntimeError(f"seed {item['seed']} failed twice; inspect {item['log_path']}")

        while pending and len(active) < MAX_CONCURRENT:
            gpu = available_gpu({item['gpu'] for item in active})
            if gpu is None:
                break
            seed = pending.popleft()
            log_path = LOGS / f'seed_{seed}_gpu{gpu}_try{attempts[seed] + 1}.log'
            log_file = log_path.open('ab', buffering=0)
            proc = subprocess.Popen(command(seed, gpu), cwd=ROOT,
                                    env={**os.environ, 'PYTHONPATH': str(ROOT)},
                                    stdin=subprocess.DEVNULL, stdout=log_file,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    close_fds=True)
            active.append({'seed': seed, 'gpu': gpu, 'proc': proc,
                           'started': time.time(), 'log_handle': log_file,
                           'log_path': log_path})
            log(f'started seed={seed} pid={proc.pid} gpu={gpu}; log={log_path}')

        write_status('running' if active else 'waiting_for_nonzero_gpu',
                     list(pending), active, completed_seeds=completed,
                     retry_count_by_seed=attempts)
        if pending or active:
            time.sleep(POLL_SECONDS)

    if sorted(completed) != list(SEEDS):
        raise RuntimeError(f'Unexpected completed seed list: {completed}')
    report = aggregate()
    report_path = RESULTS / PROFILE / DATASET / MODEL / 'paired_moabb_comparison.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    write_status('complete', [], [], completed_seeds=completed,
                 comparison=str(report_path), all_pair_checks_match=report['all_pair_checks_match'])
    log(f"all NPY-source runs complete; accuracy={report['npy_accuracy_seed_mean']:.4f} "
        f"paired_delta={report['mean_paired_delta_npy_minus_moabb']:+.4f}; "
        f"all_pair_checks_match={report['all_pair_checks_match']}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'fatal: {type(exc).__name__}: {exc}')
        raise
