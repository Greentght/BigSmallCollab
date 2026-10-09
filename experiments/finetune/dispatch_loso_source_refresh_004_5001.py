#!/usr/bin/env python
"""Queue five-model 004/5001 source-refresh LOSO on GPUs 1-9."""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import require_external_output, resolve_local_file

WORKER = ROOT / 'experiments/finetune/run_loso_source_refresh_004_5001.py'
PYTHON_BY_MODEL = {
    'cbramod': Path('/home/lixinli/anaconda3/envs/cbramod/bin/python'),
    'mirepnet': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
    'ifnet': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
    'eegnet': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
    'adfcnn': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
}
MODELS = ('mirepnet', 'cbramod', 'ifnet', 'eegnet', 'adfcnn')
DATASETS = ('BNCI2014004', 'BNCI2015001')
SEEDS = (666, 667, 668)
SUBJECTS = {'BNCI2014004': 9, 'BNCI2015001': 12}
GPU_INDICES = tuple(range(1, 10))
MAX_CONCURRENT = 2
MIN_FREE_MIB = 10_000
MAX_GPU_UTILIZATION = 40
POLL_SECONDS = 20
MAX_ATTEMPTS = 2
RESULT_ROOT = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_source_refresh_004_5001_v1')
LOG_ROOT = RESULT_ROOT / 'execution_logs'
STATUS_PATH = LOG_ROOT / 'dispatcher_status.json'


def gpu_snapshot():
    proc = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=False)
    if proc.returncode:
        return {}, proc.stderr.strip()
    result = {}
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) != 3:
            continue
        try:
            index, utilization, free_mib = map(int, parts)
        except ValueError:
            continue
        result[index] = {'utilization_pct': utilization,
                         'free_memory_mib': free_mib}
    return result, None


def eligible_gpus(active: set[int], snapshot: dict) -> list[int]:
    return [gpu for gpu in GPU_INDICES
            if gpu not in active and gpu in snapshot
            and snapshot[gpu]['utilization_pct'] <= MAX_GPU_UTILIZATION
            and snapshot[gpu]['free_memory_mib'] >= MIN_FREE_MIB]


def task_name(task):
    model, dataset, seed = task
    return f'{model}/{dataset}/seed_{seed}'


def command(task, gpu: int) -> list[str]:
    model, dataset, seed = task
    return [str(PYTHON_BY_MODEL[model]), '-u', str(WORKER),
            '--model', model, '--dataset', dataset,
            '--seed', str(seed), '--gpu', str(gpu)]


def write_status(state: dict) -> None:
    state['updated_unix'] = time.time()
    state['gpu_snapshot'], state['gpu_snapshot_error'] = gpu_snapshot()
    path = require_external_output(STATUS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def start_worker(task, gpu: int, attempt: int):
    model, dataset, seed = task
    log = require_external_output(
        LOG_ROOT / f'{model}_{dataset}_seed_{seed}_attempt_{attempt}.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    handle = log.open('ab', buffering=0)
    proc = subprocess.Popen(command(task, gpu), cwd=ROOT, env=env,
                            stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, log


def run_preflights(state: dict):
    for model, dataset in itertools.product(MODELS, DATASETS):
        while True:
            snapshot, error = gpu_snapshot()
            gpus = eligible_gpus(set(), snapshot)
            if gpus:
                gpu = gpus[0]
                break
            state['preflight_waiting_for_gpu'] = {'model': model, 'dataset': dataset}
            write_status(state)
            print(f'[preflight-wait] {model}/{dataset} gpu_snapshot={snapshot} '
                  f'error={error}', flush=True)
            time.sleep(POLL_SECONDS)
        log = require_external_output(LOG_ROOT / f'preflight_{model}_{dataset}.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'PYTHONPATH': str(ROOT),
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        env.pop('CUDA_VISIBLE_DEVICES', None)
        cmd = [str(PYTHON_BY_MODEL[model]), '-u', str(WORKER),
               '--model', model, '--dataset', dataset, '--seed', '666',
               '--gpu', str(gpu), '--preflight-only']
        print(f'[preflight-start] {model}/{dataset} gpu={gpu}', flush=True)
        with log.open('ab', buffering=0) as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode:
            raise RuntimeError(f'preflight failed: {model}/{dataset}; see {log}')
        state.setdefault('preflight_complete', []).append(f'{model}/{dataset}')
        state.pop('preflight_waiting_for_gpu', None)
        write_status(state)
        print(f'[preflight-done] {model}/{dataset} gpu={gpu}', flush=True)


def old_summary_path(dataset: str, model: str) -> Path:
    root = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_five_settings_canonical4s_v1')
    recipe = {
        'mirepnet': 'project_loso_mi8_30_subject_ea_idw45',
        'cbramod': 'project_loso_filtered_native4s_flatten',
        'ifnet': 'scratch_supervised_loso_v1',
        'eegnet': 'scratch_supervised_loso_v1',
        'adfcnn': 'scratch_supervised_loso_v1',
    }[model]
    return root / dataset / model / recipe / 'summary.json'


def aggregate_outputs():
    metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')
    overall = {}
    comparison_rows = []
    long_rows = []
    for model, dataset in itertools.product(MODELS, DATASETS):
        per_seed = {}
        all_rows = []
        for seed in SEEDS:
            seed_rows = []
            for subject in range(SUBJECTS[dataset]):
                path = (RESULT_ROOT / dataset / model / 'source_refreshed_loso_v1'
                        / f'seed_{seed}' / f'subject_{subject + 1:02d}' / 'result.npz')
                if not path.is_file():
                    continue
                with np.load(resolve_local_file(path), allow_pickle=False) as saved:
                    row = json.loads(str(saved['metrics_json'].item()))
                seed_rows.append(row)
                all_rows.append(row)
                long_rows.append(row)
            if seed_rows:
                per_seed[str(seed)] = {
                    metric: {'mean_subjects': float(np.nanmean([r[metric] for r in seed_rows])),
                             'std_subjects': float(np.nanstd([r[metric] for r in seed_rows], ddof=1))
                             if len(seed_rows) > 1 else 0.0}
                    for metric in metrics
                }
        expected = SUBJECTS[dataset] * len(SEEDS)
        summary = {
            'protocol': 'loso_source_refresh_004_5001_v1',
            'dataset': dataset, 'model': model,
            'expected_fold_seed_cells': expected,
            'completed_fold_seed_cells': len(all_rows),
            'complete': len(all_rows) == expected,
            'per_seed_subject_mean': per_seed,
        }
        if all_rows:
            summary['fold_seed_macro_mean'] = {
                metric: float(np.nanmean([r[metric] for r in all_rows]))
                for metric in metrics
            }
        old_path = old_summary_path(dataset, model)
        if old_path.is_file() and len(per_seed) == len(SEEDS):
            old = json.loads(resolve_local_file(old_path).read_text())
            old_per_seed = old.get('per_seed_subject_mean', {})
            summary['old_source_comparison'] = {}
            for seed in SEEDS:
                new_mean = per_seed[str(seed)]['accuracy']['mean_subjects']
                old_seed = old_per_seed.get(str(seed), {})
                old_accuracy = old_seed.get('accuracy', {})
                if isinstance(old_accuracy, dict):
                    old_mean = old_accuracy.get('mean_subjects')
                else:
                    old_mean = old_accuracy
                delta = None if old_mean is None else new_mean - float(old_mean)
                summary['old_source_comparison'][str(seed)] = {
                    'old_accuracy': old_mean, 'new_accuracy': new_mean,
                    'delta_percentage_points': None if delta is None else delta * 100,
                }
                comparison_rows.append({
                    'dataset': dataset, 'model': model, 'seed': seed,
                    'old_accuracy': old_mean, 'new_accuracy': new_mean,
                    'delta_percentage_points': None if delta is None else delta * 100,
                })
        overall[f'{model}/{dataset}'] = summary
        out = require_external_output(
            RESULT_ROOT / dataset / model / 'source_refreshed_loso_v1' / 'summary.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + '\n')
    combined = {
        'protocol': 'loso_source_refresh_004_5001_v1',
        'datasets': list(DATASETS), 'models': list(MODELS),
        'seeds': list(SEEDS), 'experiments': overall,
    }
    combined_path = require_external_output(RESULT_ROOT / 'summary.json')
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True, allow_nan=True) + '\n')
    csv_path = require_external_output(RESULT_ROOT / 'summary.csv')
    cols = ['dataset', 'model', 'recipe', 'test_subject', 'seed',
            'n_train', 'n_test', *metrics, 'elapsed_sec']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in cols}
                         for row in sorted(long_rows,
                                           key=lambda r: (r['dataset'], r['model'], r['seed'], r['test_subject'])))
    comparison_csv = require_external_output(RESULT_ROOT / 'old_vs_refreshed_source_accuracy.csv')
    with comparison_csv.open('w', newline='') as stream:
        cols = ['dataset', 'model', 'seed', 'old_accuracy', 'new_accuracy', 'delta_percentage_points']
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return combined_path


def main():
    require_external_output(LOG_ROOT).mkdir(parents=True, exist_ok=True)
    state = {
        'protocol': 'loso_source_refresh_004_5001_v1', 'status': 'preflight',
        'models': list(MODELS), 'datasets': list(DATASETS), 'seeds': list(SEEDS),
        'folds_per_dataset': SUBJECTS, 'expected_training_cells': sum(SUBJECTS.values())
        * len(MODELS) * len(SEEDS),
        'worker_tasks': len(MODELS) * len(DATASETS) * len(SEEDS),
        'allowed_gpus': list(GPU_INDICES), 'prohibited_gpus': [0],
        'max_concurrent_workers': MAX_CONCURRENT,
        'minimum_free_memory_mib': MIN_FREE_MIB,
        'maximum_gpu_utilization_pct': MAX_GPU_UTILIZATION,
        'started_unix': time.time(), 'active_jobs': [],
        'completed_jobs': [], 'failed_jobs': [],
    }
    stop_requested = False
    active = {}

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        state['stop_signal'] = signum
        state['status'] = 'stopping'
        for item in active.values():
            item['process'].terminate()
        write_status(state)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    write_status(state)
    run_preflights(state)

    order = [
        ('cbramod', 'BNCI2014004'), ('cbramod', 'BNCI2015001'),
        ('mirepnet', 'BNCI2014004'), ('mirepnet', 'BNCI2015001'),
        ('ifnet', 'BNCI2014004'), ('ifnet', 'BNCI2015001'),
        ('eegnet', 'BNCI2014004'), ('eegnet', 'BNCI2015001'),
        ('adfcnn', 'BNCI2014004'), ('adfcnn', 'BNCI2015001'),
    ]
    pending = [(model, dataset, seed) for model, dataset in order for seed in SEEDS]
    attempts = {task: 0 for task in pending}
    completed = []
    failed = []
    state['status'] = 'running'
    write_status(state)
    print(f'[queue-start] {len(pending)} model/dataset/seed workers; GPU0 excluded', flush=True)

    while pending or active:
        if stop_requested:
            break
        snapshot, error = gpu_snapshot()
        active_gpus = {item['gpu'] for item in active.values()}
        available = eligible_gpus(active_gpus, snapshot)
        while pending and available and len(active) < MAX_CONCURRENT:
            task = pending.pop(0)
            attempts[task] += 1
            gpu = available.pop(0)
            proc, log = start_worker(task, gpu, attempts[task])
            active[proc.pid] = {'process': proc, 'task': task,
                                'gpu': gpu, 'log': str(log),
                                'attempt': attempts[task], 'started_unix': time.time()}
            print(f'[worker-start] {task_name(task)} pid={proc.pid} gpu={gpu} '
                  f'attempt={attempts[task]}', flush=True)
        for pid, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            task = item['task']
            active.pop(pid)
            record = {**item, 'task': task_name(task), 'exit_code': code,
                      'finished_unix': time.time()}
            del record['process']
            if code == 0:
                completed.append(record)
                state['completed_jobs'] = [entry['task'] for entry in completed]
                print(f'[worker-done] {task_name(task)} elapsed='
                      f'{record["finished_unix"] - item["started_unix"]:.1f}s', flush=True)
            elif attempts[task] < MAX_ATTEMPTS:
                pending.append(task)
                state.setdefault('retries', []).append({**record, 'retry': attempts[task] + 1})
                print(f'[worker-retry] {task_name(task)} exit={code}; '
                      f'log={item["log"]}', flush=True)
            else:
                failed.append(record)
                state['failed_jobs'] = failed
                print(f'[worker-failed] {task_name(task)} exit={code}; '
                      f'log={item["log"]}', flush=True)
        state['active_jobs'] = [
            {'pid': pid, 'task': task_name(item['task']), 'gpu': item['gpu'],
             'log': item['log'], 'attempt': item['attempt'],
             'elapsed_seconds': round(time.time() - item['started_unix'], 1)}
            for pid, item in active.items()]
        state['pending_jobs'] = [task_name(task) for task in pending]
        state['gpu_snapshot'], state['gpu_snapshot_error'] = snapshot, error
        state['completed_jobs'] = [entry['task'] for entry in completed]
        state['failed_jobs'] = failed
        write_status(state)
        if pending or active:
            if not active and not available:
                print(f'[queue-wait] no eligible GPU; pending={len(pending)} '
                      f'snapshot={snapshot}', flush=True)
            time.sleep(POLL_SECONDS)

    if stop_requested:
        state['status'] = 'stopped'
    elif failed:
        state['status'] = 'failed'
    else:
        state['status'] = 'complete'
        summary = aggregate_outputs()
        state['summary_path'] = str(summary)
    state['dispatcher_elapsed_seconds'] = time.time() - state['started_unix']
    state['active_jobs'] = []
    state['pending_jobs'] = [task_name(task) for task in pending]
    write_status(state)
    print(f'[dispatcher-{state["status"]}] completed={len(completed)} '
          f'failed={len(failed)} summary={state.get("summary_path")}', flush=True)


if __name__ == '__main__':
    main()
