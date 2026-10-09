#!/usr/bin/env python
"""Run the refreshed-source 004/5001 LOSO distillation queue on GPUs 1-9."""
from __future__ import annotations

import fcntl
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import require_external_output, resolve_local_file

RUNNER = ROOT / 'experiments/distill/run_loso_source_refresh_distillation.py'
PYTHON = {
    'mirepnet': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
    'cbramod': Path('/home/lixinli/anaconda3/envs/cbramod/bin/python'),
}
DATASETS = ('BNCI2014004', 'BNCI2015001')
TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
STAGES = ('logits_kd', 'kd_feature', 'warmup10_kd', 'warmup10_kd_feature')
SEEDS = (666, 667, 668)
SUBJECTS = {'BNCI2014004': 9, 'BNCI2015001': 12}
ALLOWED_GPUS = tuple(range(1, 10))
MAX_CONCURRENT = 2
MIN_FREE_MIB = 7500
MAX_UTILIZATION_PCT = 100
POLL_SECONDS = 20
MAX_ATTEMPTS = 2
RESULT_ROOT = Path('/data1/llx/BigSmallcollab/results/distill/loso_source_refresh_004_5001_v1')
LOG_ROOT = RESULT_ROOT / 'execution_logs'
STATUS_PATH = LOG_ROOT / 'dispatcher_status.json'
LOCK_PATH = LOG_ROOT / '.dispatcher.lock'


def gpu_snapshot() -> tuple[dict, str | None]:
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
    choices = [gpu for gpu in ALLOWED_GPUS
               if gpu not in active and gpu in snapshot
               and snapshot[gpu]['free_memory_mib'] >= MIN_FREE_MIB
               and snapshot[gpu]['utilization_pct'] <= MAX_UTILIZATION_PCT]
    return sorted(choices, key=lambda gpu: (
        -snapshot[gpu]['free_memory_mib'], snapshot[gpu]['utilization_pct'], gpu))


def task_name(task: dict) -> str:
    mode = task['mode']
    parts = [mode, task['dataset'], task['teacher']]
    if task.get('student'):
        parts.append(task['student'])
    if task.get('stage'):
        parts.append(task['stage'])
    parts.append(f"seed_{task['seed']}")
    return '/'.join(parts)


def python_for(task: dict) -> Path:
    # Training/smoke jobs build only the student. Target export loads the teacher.
    model = task['teacher'] if task['mode'] == 'prepare-cache' else 'mirepnet'
    interpreter = PYTHON[model]
    if not interpreter.is_file():
        raise FileNotFoundError(f'Python environment is missing: {interpreter}')
    return interpreter


def command(task: dict, gpu: int) -> list[str]:
    cmd = [str(python_for(task)), '-u', str(RUNNER), '--mode', task['mode'],
           '--dataset', task['dataset'], '--teacher', task['teacher'],
           '--seed', str(task['seed']), '--gpu', str(gpu)]
    if task.get('student'):
        cmd.extend(['--student', task['student']])
    if task.get('stage'):
        cmd.extend(['--stage', task['stage']])
    return cmd


def write_status(state: dict) -> None:
    state['updated_unix'] = time.time()
    state['gpu_snapshot'], state['gpu_snapshot_error'] = gpu_snapshot()
    path = require_external_output(STATUS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def stage_tasks(stage: str) -> list[dict]:
    return [{'mode': 'train', 'dataset': dataset, 'teacher': teacher,
             'student': student, 'stage': stage, 'seed': seed}
            for dataset, teacher, student, seed in itertools.product(
                DATASETS, TEACHERS, STUDENTS, SEEDS)]


def cache_tasks() -> list[dict]:
    return [{'mode': 'prepare-cache', 'dataset': dataset, 'teacher': teacher,
             'seed': seed}
            for dataset, teacher, seed in itertools.product(DATASETS, TEACHERS, SEEDS)]


def smoke_tasks() -> list[dict]:
    return [{'mode': 'smoke', 'dataset': dataset, 'teacher': teacher,
             'student': student, 'seed': SEEDS[0]}
            for dataset, teacher, student in itertools.product(DATASETS, TEACHERS, STUDENTS)]


def run_validate(state: dict) -> None:
    interpreter = PYTHON['mirepnet']
    log = require_external_output(LOG_ROOT / 'validate.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    with log.open('ab', buffering=0) as stream:
        result = subprocess.run([str(interpreter), '-u', str(RUNNER), '--mode', 'validate'],
                                cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'input/baseline validation failed; see {log}')
    state['prerequisites_validated'] = True
    state['validation_log'] = str(log)
    write_status(state)


def run_queue(tasks: list[dict], phase: str, state: dict) -> None:
    pending = list(tasks)
    attempts = {task_name(task): 0 for task in tasks}
    complete, failed, active = [], [], {}
    state['phase'] = phase
    state['phase_status'] = 'running'
    state['phase_expected_tasks'] = len(tasks)
    state['phase_completed_tasks'] = 0
    state['phase_failed_tasks'] = []
    state['phase_pending_tasks'] = [task_name(task) for task in pending]
    write_status(state)
    print(f'[queue-start] {phase} tasks={len(tasks)} GPUs={list(ALLOWED_GPUS)} GPU0 excluded', flush=True)

    while pending or active:
        snapshot, snapshot_error = gpu_snapshot()
        eligible = eligible_gpus({item['gpu'] for item in active.values()}, snapshot)
        while pending and eligible and len(active) < MAX_CONCURRENT:
            task = pending.pop(0)
            name = task_name(task)
            attempts[name] += 1
            gpu = eligible.pop(0)
            attempt = attempts[name]
            log = require_external_output(LOG_ROOT / f'{phase}_{name.replace("/", "_")}_attempt{attempt}.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, 'PYTHONPATH': str(ROOT),
                   'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
            env.pop('CUDA_VISIBLE_DEVICES', None)
            stream = log.open('ab', buffering=0)
            proc = subprocess.Popen(command(task, gpu), cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            active[proc.pid] = {'process': proc, 'stream': stream, 'task': task,
                                'gpu': gpu, 'log': str(log), 'attempt': attempt,
                                'started_unix': time.time()}
            print(f'[worker-start] {name} pid={proc.pid} gpu={gpu} attempt={attempt}', flush=True)

        for pid, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['stream'].close()
            active.pop(pid)
            task, name = item['task'], task_name(item['task'])
            record = {'task': name, 'pid': pid, 'gpu': item['gpu'],
                      'log': item['log'], 'attempt': item['attempt'],
                      'exit_code': code, 'started_unix': item['started_unix'],
                      'finished_unix': time.time()}
            if code == 0:
                complete.append(record)
                state['phase_completed_tasks'] = len(complete)
                print(f'[worker-done] {name} elapsed={record["finished_unix"]-record["started_unix"]:.1f}s', flush=True)
            elif attempts[name] < MAX_ATTEMPTS:
                pending.append(task)
                state.setdefault('retries', []).append(record)
                print(f'[worker-retry] {name} exit={code} log={item["log"]}', flush=True)
            else:
                failed.append(record)
                state['phase_failed_tasks'] = failed
                print(f'[worker-failed] {name} exit={code} log={item["log"]}', flush=True)

        state['active_jobs'] = [
            {'pid': pid, 'task': task_name(item['task']), 'gpu': item['gpu'],
             'log': item['log'], 'attempt': item['attempt'],
             'elapsed_seconds': round(time.time() - item['started_unix'], 1)}
            for pid, item in active.items()]
        state['phase_completed_tasks'] = len(complete)
        state['phase_pending_tasks'] = [task_name(task) for task in pending]
        state['phase_failed_tasks'] = failed
        state['gpu_snapshot'], state['gpu_snapshot_error'] = snapshot, snapshot_error
        write_status(state)
        if failed and bool(state.get('fail_fast', True)):
            for item in active.values():
                item['process'].terminate()
            for item in active.values():
                item['process'].wait(timeout=30)
                item['stream'].close()
            active.clear()
            break
        if pending or active:
            if not active and not eligible:
                print(f'[queue-wait] {phase} no eligible non-GPU0 card; pending={len(pending)} snapshot={snapshot}', flush=True)
            time.sleep(POLL_SECONDS)

    if failed:
        state['phase_status'] = 'failed'
        write_status(state)
        raise RuntimeError(f'{phase} failed {len(failed)} worker(s); see execution logs')
    state['phase_status'] = 'complete'
    state['phase_completed_tasks'] = len(complete)
    state.setdefault('completed_phases', []).append(phase)
    write_status(state)


def run_summary(state: dict) -> None:
    log = require_external_output(LOG_ROOT / 'summary.log')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    with log.open('ab', buffering=0) as stream:
        result = subprocess.run([str(PYTHON['mirepnet']), '-u', str(RUNNER),
                                 '--mode', 'summarize'], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'summary generation failed; see {log}')
    summary_path = RESULT_ROOT / 'summary.json'
    summary = json.loads(resolve_local_file(summary_path).read_text())
    incomplete = {stage: counts for stage, counts in summary['stage_counts'].items()
                  if not counts['complete']}
    if incomplete:
        raise RuntimeError(f'final summary contains incomplete stages: {incomplete}')
    state['summary_path'] = str(summary_path)
    state['summary_stage_counts'] = summary['stage_counts']


def main() -> None:
    LOG_ROOT_PATH = require_external_output(LOG_ROOT)
    LOG_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    lock_path = require_external_output(LOCK_PATH)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'another dispatcher already holds {lock_path}') from exc
        state = {
            'protocol': 'loso_source_refresh_004_5001_kd_feature_warmup10_v1',
            'source_protocol': 'loso_source_refresh_004_5001_v1',
            'status': 'preflight', 'datasets': list(DATASETS),
            'teachers': list(TEACHERS), 'students': list(STUDENTS),
            'stages': list(STAGES), 'seeds': list(SEEDS),
            'folds_per_dataset': SUBJECTS, 'expected_training_fold_seed_units': 1512,
            'allowed_gpus': list(ALLOWED_GPUS), 'prohibited_gpus': [0],
            'max_concurrent_workers': MAX_CONCURRENT,
            'minimum_free_memory_mib': MIN_FREE_MIB,
            'maximum_gpu_utilization_pct': MAX_UTILIZATION_PCT,
            'fail_fast': True, 'started_unix': time.time(), 'active_jobs': [],
            'completed_phases': [], 'retries': [],
        }
        write_status(state)
        try:
            run_validate(state)
            state['status'] = 'running'
            run_queue(cache_tasks(), 'teacher_cache', state)
            run_queue(smoke_tasks(), 'smoke', state)
            for stage in STAGES:
                run_queue(stage_tasks(stage), stage, state)
            run_summary(state)
            state['status'] = 'complete'
        except BaseException as exc:
            state['status'] = 'failed'
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            state['finished_unix'] = time.time()
            state['elapsed_seconds'] = state['finished_unix'] - state['started_unix']
            state['active_jobs'] = []
            write_status(state)
        print(f'[dispatcher-{state["status"]}] summary={state.get("summary_path")}', flush=True)


if __name__ == '__main__':
    main()
