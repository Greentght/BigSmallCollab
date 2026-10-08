"""Queue all updated-source IFNet/EEGNet/ADFCNN 001 LOSO jobs on GPUs 1-9."""
import itertools
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.storage import require_external_output


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'experiments/finetune/run_loso_small_wideband_001.py'
PYTHON = Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python')
RESULT_ROOT = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_config_alignment_v2/wideband_npy_v3')
LOG_ROOT = RESULT_ROOT / 'execution_logs/small_models'
STATUS_PATH = LOG_ROOT / 'wideband_001_small_models_status.json'
MODELS = ('ifnet', 'eegnet', 'adfcnn')
DATASETS = ('BNCI2014001', 'BNCI2014001-4')
SEEDS = (0, 1, 2)
ALLOWED_GPUS = tuple(range(1, 10))
MAX_CONCURRENT = 6
MIN_FREE_MIB = 10000
MAX_UTILIZATION = 40
POLL_SECONDS = 15
MAX_ATTEMPTS = 2


def gpu_snapshot():
    proc = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=False)
    if proc.returncode:
        return {}, proc.stderr.strip()
    values = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 3:
            continue
        try:
            index, utilization, free_mib = map(int, parts)
        except ValueError:
            continue
        values[index] = {'utilization_pct': utilization, 'free_memory_mib': free_mib}
    return values, None


def eligible_gpus(active, snapshot):
    result = []
    for gpu in ALLOWED_GPUS:
        state = snapshot.get(gpu)
        if gpu in active or state is None:
            continue
        if state['utilization_pct'] <= MAX_UTILIZATION and state['free_memory_mib'] >= MIN_FREE_MIB:
            result.append(gpu)
    return result


def task_name(task):
    model, dataset, seed = task
    return f'{model}/{dataset}/seed_{seed}'


def write_status(state):
    state['updated_unix'] = time.time()
    state['gpu_snapshot'], state['gpu_snapshot_error'] = gpu_snapshot()
    status_path = require_external_output(STATUS_PATH)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, status_path)


def command(task, gpu):
    model, dataset, seed = task
    return [str(PYTHON), '-u', str(WORKER), '--model', model,
            '--datasets', dataset, '--seeds', str(seed), '--gpu', str(gpu)]


def aggregate_outputs():
    scalar_metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')
    overall = {}
    for model in MODELS:
        for dataset in DATASETS:
            output_dir = require_external_output(
                RESULT_ROOT / dataset / model / 'scratch_supervised_loso_wideband_v1')
            rows = []
            for seed in SEEDS:
                for subject in range(9):
                    result_path = output_dir / f'subject_{subject + 1:02d}' / f'seed_{seed}' / 'result.npz'
                    with np.load(result_path, allow_pickle=False) as saved:
                        rows.append(json.loads(str(saved['metrics_json'].item())))
            if len(rows) != 27:
                raise RuntimeError(f'{model}/{dataset}: expected 27 completed rows, found {len(rows)}')
            per_seed = {}
            for seed in SEEDS:
                seed_rows = [row for row in rows if int(row['seed']) == seed]
                if len(seed_rows) != 9:
                    raise RuntimeError(f'{model}/{dataset}/seed_{seed}: expected 9 folds')
                per_seed[str(seed)] = {
                    metric: {
                        'mean_subjects': float(np.nanmean([row[metric] for row in seed_rows])),
                        'std_subjects': float(np.nanstd([row[metric] for row in seed_rows], ddof=1)),
                    } for metric in scalar_metrics
                }
            summary = {
                'protocol': 'loso_001_all_models_wideband_v1',
                'dataset': dataset, 'model': model,
                'recipe': 'scratch_supervised_loso_wideband_v1',
                'expected_fold_seed_cells': 27, 'completed_fold_seed_cells': len(rows),
                'complete': True, 'per_seed_subject_mean': per_seed,
                'subject_fold_seed_macro_mean': {
                    metric: float(np.nanmean([row[metric] for row in rows]))
                    for metric in scalar_metrics
                },
            }
            (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            columns = ['dataset', 'model', 'recipe', 'test_subject', 'seed',
                       'n_train', 'n_test', *scalar_metrics, 'elapsed_sec', 'artifact_origin']
            with (output_dir / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in columns}
                                 for row in sorted(rows, key=lambda row: (row['seed'], row['test_subject'])))
            overall[f'{model}/{dataset}'] = summary
    combined = {
        'protocol': 'loso_001_all_models_wideband_v1',
        'datasets': list(DATASETS), 'models': list(MODELS),
        'source': '/data1/llx/BNCI2014001/broadband_0p1_75hz',
        'experiments': overall,
    }
    combined_path = require_external_output(
        RESULT_ROOT / 'wideband_14001_small_models_loso_summary.json')
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + '\n')
    return combined_path


def start_worker(task, gpu, attempt):
    log_path = require_external_output(
        LOG_ROOT / f'{task[0]}_{task[1]}_seed_{task[2]}_attempt_{attempt}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT),
                   'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4',
                   'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    handle = log_path.open('ab', buffering=0)
    process = subprocess.Popen(command(task, gpu), cwd=ROOT, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return process, log_path


def run_preflights():
    snapshot, error = gpu_snapshot()
    while True:
        candidates = eligible_gpus(set(), snapshot)
        if candidates:
            gpu = candidates[0]
            break
        print(f'[preflight-wait] no eligible GPU; snapshot={snapshot} error={error}', flush=True)
        time.sleep(POLL_SECONDS)
        snapshot, error = gpu_snapshot()
    for model in MODELS:
        log_path = require_external_output(LOG_ROOT / f'preflight_{model}.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, 'PYTHONPATH': str(ROOT),
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'OPENBLAS_NUM_THREADS': '4',
                       'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        cmd = [str(PYTHON), '-u', str(WORKER), '--model', model,
               '--datasets', *DATASETS, '--seeds', '0', '--gpu', str(gpu),
               '--preflight-only']
        print(f'[preflight-start] {model} gpu={gpu}', flush=True)
        with log_path.open('ab', buffering=0) as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode:
            raise RuntimeError(f'preflight failed for {model}; see {log_path}')
        print(f'[preflight-done] {model} gpu={gpu}', flush=True)


def main():
    require_external_output(LOG_ROOT).mkdir(parents=True, exist_ok=True)
    run_preflights()
    pending = list(itertools.product(MODELS, DATASETS, SEEDS))
    active = {}
    attempts = {task: 0 for task in pending}
    completed = []
    failed = []
    started_at = time.time()
    state = {
        'protocol': 'loso_001_all_models_wideband_v1',
        'status': 'running', 'models': list(MODELS), 'datasets': list(DATASETS),
        'seeds': list(SEEDS), 'folds_per_task': 9,
        'expected_training_cells': 162, 'worker_tasks': len(pending),
        'active_jobs': [], 'completed_jobs': [], 'failed_jobs': [],
        'allowed_gpus': list(ALLOWED_GPUS), 'prohibited_gpus': [0],
        'max_concurrent_workers': MAX_CONCURRENT,
        'minimum_free_memory_mib': MIN_FREE_MIB,
        'maximum_gpu_utilization_pct': MAX_UTILIZATION,
        'started_unix': started_at,
    }
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        state['stop_signal'] = signum
        state['status'] = 'stopping'
        write_status(state)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    write_status(state)
    print(f'[queue-start] {len(pending)} model/dataset/seed jobs; GPU0 excluded', flush=True)

    while pending or active:
        if stop_requested:
            for entry in active.values():
                entry['process'].terminate()
            for entry in active.values():
                entry['process'].wait()
            state['active_jobs'] = []
            state['status'] = 'stopped'
            write_status(state)
            return 130

        for task, entry in list(active.items()):
            code = entry['process'].poll()
            if code is None:
                continue
            del active[task]
            if code == 0:
                completed.append(task_name(task))
                print(f'[task-done] {task_name(task)} gpu={entry["gpu"]}', flush=True)
            elif attempts[task] < MAX_ATTEMPTS:
                print(f'[task-retry] {task_name(task)} exit={code}', flush=True)
                pending.append(task)
            else:
                failed.append({'task': task_name(task), 'exit_code': code,
                               'log': str(entry['log_path'])})
                print(f'[task-failed] {task_name(task)} exit={code}', flush=True)

        snapshot, error = gpu_snapshot()
        for gpu in eligible_gpus({entry['gpu'] for entry in active.values()}, snapshot):
            if not pending or len(active) >= MAX_CONCURRENT:
                break
            task = pending.pop(0)
            attempts[task] += 1
            process, log_path = start_worker(task, gpu, attempts[task])
            active[task] = {'process': process, 'gpu': gpu,
                            'log_path': str(log_path), 'attempt': attempts[task],
                            'started_unix': time.time()}
            print(f'[task-start] {task_name(task)} gpu={gpu} '
                  f'pid={process.pid} log={log_path}', flush=True)

        state['active_jobs'] = [
            {'task': task_name(task), 'pid': entry['process'].pid,
             'gpu': entry['gpu'], 'log': entry['log_path'],
             'attempt': entry['attempt'], 'elapsed_seconds': round(time.time()-entry['started_unix'], 1)}
            for task, entry in active.items()]
        state['completed_jobs'] = list(completed)
        state['failed_jobs'] = list(failed)
        state['pending_jobs'] = [task_name(task) for task in pending]
        state['gpu_snapshot_error'] = error
        write_status(state)
        time.sleep(POLL_SECONDS)

    if not failed:
        combined_path = aggregate_outputs()
        state['aggregate_summary'] = str(combined_path)
    state['status'] = 'failed' if failed else 'complete'
    state['active_jobs'] = []
    state['completed_jobs'] = list(completed)
    state['failed_jobs'] = list(failed)
    state['pending_jobs'] = []
    state['elapsed_seconds'] = round(time.time() - started_at, 1)
    write_status(state)
    print(f'[queue-{state["status"]}] completed={len(completed)} failed={len(failed)} '
          f'elapsed_hours={state["elapsed_seconds"]/3600:.2f}', flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
