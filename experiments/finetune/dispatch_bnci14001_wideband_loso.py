#!/usr/bin/env python
"""Run 001/001-4 MIRepNet and CBraMod LOSO from the new all-session NPY cache.

The dispatcher owns up to six independent seed workers, waits for available physical
GPUs 1--9, and keeps its state on disk. It does not need an interactive terminal.
"""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
PYTHONS = {
    'cbramod': Path('/home/lixinli/anaconda3/envs/cbramod/bin/python'),
    'mirepnet': Path('/home/lixinli/anaconda3/envs/mirepnet/bin/python'),
}
WORKER = ROOT / 'experiments/finetune/run_loso_config_alignment.py'
RESULT_ROOT = RESULTS_ROOT / 'reproductions/loso_config_alignment_v2'
INPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/model_inputs'
EXEC = RESULTS_ROOT / 'reproductions/loso_source_v3/execution_logs'
STATUS_PATH = EXEC / 'wideband_14001_loso_status.json'
LOCK_PATH = EXEC / 'wideband_14001_loso_dispatcher.lock'
PROFILE = 'wideband_npy_v3'
VARIANT = 'all_sessions_source_train_session'
DATASETS = ('BNCI2014001-4', 'BNCI2014001')
MODELS = ('cbramod', 'mirepnet')
COMBINATIONS = tuple((dataset, model) for model in MODELS for dataset in DATASETS)
SEEDS = (0, 1, 2)
SUBJECTS = tuple(range(1, 10))
TASKS = tuple((dataset, model, seed) for seed in SEEDS for dataset, model in COMBINATIONS)
EXPECTED_EPOCHS = {(dataset, model): (10 if model == 'mirepnet' and dataset == 'BNCI2014001' else 20)
                   for dataset, model in COMBINATIONS}
SOURCE_MANIFEST = DATA_CACHE_ROOT / 'loso_source_v3/BNCI2014001/manifest.json'
PREFLIGHT_LOGS = RESULT_ROOT / PROFILE / 'audit/preflight'
WORKER_LOGS = EXEC / 'wideband_14001_loso_workers'
ALLOWED_GPUS = tuple(range(1, 10))
MAX_CONCURRENT = 6
MIN_FREE_MIB = 10000
MAX_UTILIZATION = 40
POLL_SECONDS = 30
REFERENCES = (
    ('legacy_narrowband_npy', 'reference_aligned_npy', 'npy_source'),
    ('prior_wideband_moabb', 'reference_aligned', 'rebuilt_source'),
)
PREFLIGHTS_DONE: list[dict] = []


def task_key(task: tuple[str, str, int]) -> str:
    dataset, model, seed = task
    return f'{model}/{dataset}/seed_{seed}'


def task_record(task: tuple[str, str, int]) -> dict:
    dataset, model, seed = task
    return {'job_key': task_key(task), 'dataset': dataset, 'model': model, 'seed': seed}


def input_folder(dataset: str, model: str, profile: str = PROFILE,
                 variant: str = VARIANT) -> Path:
    return INPUT_ROOT / profile / dataset / model / variant


def result_folder(dataset: str, model: str, seed: int, subject: int,
                  profile: str = PROFILE) -> Path:
    return RESULT_ROOT / profile / dataset / model / f'seed_{seed}' / f'subject_{subject:02d}'


def expected_trials(dataset: str) -> int:
    return 2592 if dataset == 'BNCI2014001-4' else 1296


def expected_shape(model: str) -> list[int]:
    return [22, 4, 200] if model == 'cbramod' else [45, 1000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    # The launcher redirects stdout to wideband_14001_loso_dispatcher.log.
    print(f'[{utc_now()}] {message}', flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def local_data_path(path: Path) -> Path:
    """Read datasets and prior results through the external artifact store."""
    return resolve_local_file(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with local_data_path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(local_data_path(path).read_text())


def trial_table(folder: Path) -> list[tuple[str, int, int]]:
    with local_data_path(folder / 'trials.csv').open(newline='') as f:
        return [(row['trial_uid'], int(row['label_id']), int(row['subject_zero_based']))
                for row in csv.DictReader(f)]


def validate_inputs() -> tuple[dict, dict]:
    source = read_json(SOURCE_MANIFEST)
    source_hash = sha256_file(SOURCE_MANIFEST)
    if source.get('source_shape') != [5184, 22, 1001]:
        raise RuntimeError('The new source cache must retain all 5184 native trials')
    if source.get('trial_counts_by_session') != {'0train': 2592, '1test': 2592}:
        raise RuntimeError('The source cache must include both complete native sessions')
    manifests, hashes = {}, {}
    for dataset, model in COMBINATIONS:
        if not PYTHONS[model].is_file():
            raise FileNotFoundError(f'{model} Python unavailable: {PYTHONS[model]}')
        folder = input_folder(dataset, model)
        manifest_path = folder / 'manifest.json'
        manifest = read_json(manifest_path)
        count = expected_trials(dataset)
        for field, expected in (('profile', PROFILE), ('variant', VARIANT),
                                ('dataset', dataset), ('model', model),
                                ('trial_count', count), ('input_shape', expected_shape(model)),
                                ('selected_session', '0train'), ('source_manifest_sha256', source_hash)):
            if manifest.get(field) != expected:
                raise RuntimeError(f'{manifest_path}: {field} != {expected!r}')
        for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv'):
            record = manifest['files'][name]
            if sha256_file(folder / name) != record['sha256']:
                raise RuntimeError(f'Input content hash mismatch: {folder / name}')
        rows = trial_table(folder)
        if len(rows) != count or len({row[0] for row in rows}) != len(rows):
            raise RuntimeError(f'{dataset}/{model}: expected {count} unique input trial UIDs')
        if not all('|session-0train|' in row[0] for row in rows):
            raise RuntimeError(f'{dataset}/{model}: model inputs contain trials outside 0train')
        for subject in range(9):
            if sum(row[2] == subject for row in rows) != count // 9:
                raise RuntimeError(f'{dataset}/{model}: unexpected count for subject {subject + 1}')
        manifests[(dataset, model)] = manifest
        hashes[(dataset, model)] = sha256_file(manifest_path)
    return manifests, hashes


def gpu_snapshot() -> list[dict]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 3:
            rows.append({'gpu': int(fields[0]), 'utilization_pct': int(fields[1]),
                         'free_memory_mib': int(fields[2])})
    return rows


def choose_gpu(snapshot: list[dict], occupied: set[int]) -> int | None:
    candidates = [row for row in snapshot if row['gpu'] in ALLOWED_GPUS
                  and row['gpu'] != 0 and row['gpu'] not in occupied
                  and row['free_memory_mib'] >= MIN_FREE_MIB
                  and row['utilization_pct'] <= MAX_UTILIZATION]
    candidates.sort(key=lambda row: (row['utilization_pct'],
                                     -row['free_memory_mib'], row['gpu']))
    return candidates[0]['gpu'] if candidates else None


def fold_progress() -> tuple[int, list[dict]]:
    progress = []
    completed = 0
    for dataset, model, seed in TASKS:
        for subject in SUBJECTS:
            folder = result_folder(dataset, model, seed, subject)
            path = folder / 'status.json'
            if not path.is_file():
                continue
            item = read_json(path)
            status = item.get('status')
            if status == 'complete' and (folder / 'result.json').is_file():
                completed += 1
            epochs = EXPECTED_EPOCHS[(dataset, model)]
            progress.append({'dataset': dataset, 'model': model, 'seed': seed,
                             'held_out_subject': subject,
                             'status': status, 'epoch': item.get('epoch', epochs
                                                              if status == 'complete' else 0),
                             'epochs': item.get('epochs', epochs),
                             'accuracy': item.get('metrics', {}).get('accuracy'),
                             'gpu': item.get('gpu_physical_index'),
                             'elapsed_seconds': item.get('elapsed_seconds')})
    return completed, progress


def write_status(phase: str, pending: list[tuple], active: list[dict], **extra) -> None:
    complete_count, folds = fold_progress()
    atomic_json(STATUS_PATH, {
        'phase': phase, 'updated_at_utc': utc_now(), 'updated_at_epoch_s': time.time(),
        'profile': PROFILE, 'variant': VARIANT, 'datasets': list(DATASETS), 'models': list(MODELS),
        'seeds': list(SEEDS), 'jobs_expected': 12, 'folds_expected': 108, 'folds_complete': complete_count,
        'jobs_complete': len(extra.get('completed_jobs', [])),
        'preflights_expected': 4, 'preflights_complete': list(PREFLIGHTS_DONE),
        'preflights_complete_count': len(PREFLIGHTS_DONE),
        'pending_jobs': [task_record(task) for task in pending], 'fold_progress': folds,
        'active_jobs': [{**task_record(item['task']), 'gpu': item['gpu'],
                         'pid': item['proc'].pid, 'log': str(item['log_path']),
                         'elapsed_seconds': time.time() - item['started']}
                        for item in active],
        'gpu_policy': {'allowed_physical_gpus': list(ALLOWED_GPUS),
                       'prohibited_physical_gpus': [0], 'max_concurrent': MAX_CONCURRENT,
                       'minimum_free_memory_mib': MIN_FREE_MIB,
                       'maximum_utilization_pct': MAX_UTILIZATION},
        'gpu_snapshot': gpu_snapshot(), **extra,
    })


def worker_command(task: tuple[str, str, int], gpu: int, preflight: bool = False) -> list[str]:
    if gpu not in ALLOWED_GPUS or gpu == 0:
        raise ValueError(f'Prohibited GPU {gpu}')
    dataset, model, seed = task
    args = [str(PYTHONS[model]), '-u', str(WORKER), '--profile', PROFILE,
            '--dataset', dataset, '--model', model, '--seed', str(seed),
            '--gpu', str(gpu)]
    if preflight:
        args.extend(['--preflight-steps', '1'])
    return args


def start_worker(task: tuple[str, str, int], gpu: int, log_path: Path, preflight: bool = False) -> dict:
    log_path = require_external_output(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('ab', buffering=0)
    worker_env = {**os.environ, 'PYTHONPATH': str(ROOT),
                  'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                  'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    worker_env.pop('CUDA_VISIBLE_DEVICES', None)
    try:
        proc = subprocess.Popen(
            worker_command(task, gpu, preflight), cwd=ROOT,
            env=worker_env,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    except Exception:
        handle.close()
        raise
    log(f'started {"preflight" if preflight else "formal"} {task_key(task)} '
        f'pid={proc.pid} gpu={gpu}; log={log_path}')
    return {'task': task, 'gpu': gpu, 'proc': proc, 'started': time.time(),
            'log_path': log_path, 'log_handle': handle}


def run_preflight(manifest_hashes: dict) -> None:
    completed = PREFLIGHTS_DONE
    for dataset, model in COMBINATIONS:
        task = (dataset, model, 0)
        # Preserve successful preflights across storage migrations/restarts.
        # Validate the original receipt against the actual moved input bytes.
        prior_receipt = None
        for prior_log in sorted((PREFLIGHT_LOGS / model / dataset).glob('seed0_gpu*_try*.log')):
            for line in reversed(local_data_path(prior_log).read_text().splitlines()):
                if not line.startswith('{'):
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (candidate.get('preflight_status') == 'passed'
                        and candidate.get('profile') == PROFILE
                        and candidate.get('dataset') == dataset
                        and candidate.get('model') == model
                        and candidate.get('input_manifest_sha256') == manifest_hashes[(dataset, model)]
                        and candidate.get('input_shape') == expected_shape(model)
                        and candidate.get('batch_size') == (8 if model == 'mirepnet' else 16)):
                    prior_receipt = prior_log
                    break
            if prior_receipt:
                break
        if prior_receipt:
            completed.append(task_record(task))
            log(f'restored verified preflight: {model}/{dataset}; receipt={prior_receipt}')
            continue
        attempts = 0
        while True:
            gpu = choose_gpu(gpu_snapshot(), set())
            if gpu is None:
                write_status('waiting_for_preflight_gpu', list(TASKS), [],
                             preflights_complete=completed, next_preflight=task_record(task))
                time.sleep(POLL_SECONDS)
                continue
            path = PREFLIGHT_LOGS / model / dataset / f'seed0_gpu{gpu}_try{attempts + 1}.log'
            item = start_worker(task, gpu, path, True)
            while item['proc'].poll() is None:
                write_status('preflight_running', list(TASKS), [item], preflights_complete=completed)
                time.sleep(POLL_SECONDS)
            code = item['proc'].returncode
            item['log_handle'].close()
            if code == 0:
                completed.append(task_record(task))
                log(f'separate one-batch preflight passed: {model}/{dataset}')
                break
            attempts += 1
            if attempts > 1:
                raise RuntimeError(f'Preflight failed twice; code={code}; log={item["log_path"]}')
            log(f'preflight failed code={code}; retrying once when a GPU is available')
    log('all four preflights passed; formal seeds start from fresh RNG states')


def result_for(profile: str, dataset: str, model: str, seed: int, subject: int) -> dict:
    path = result_folder(dataset, model, seed, subject, profile) / 'result.json'
    result = read_json(path)
    test_trials = expected_trials(dataset) // 9
    expected = {'status': 'complete', 'profile': profile, 'dataset': dataset,
                'model': model, 'seed': seed, 'held_out_subject': subject,
                'train_trials': test_trials * 8, 'test_trials': test_trials,
                'input_shape': expected_shape(model)}
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(f'{path}: {field} != {value!r}')
    epochs = EXPECTED_EPOCHS[(dataset, model)]
    if result.get('resolved_training_config', {}).get('epochs') != epochs:
        raise RuntimeError(f'{path}: expected {epochs} trained epochs')
    return result


def validate_fresh_fold(task: tuple[str, str, int], subject: int, manifest_hash: str) -> dict:
    dataset, model, seed = task
    result = result_for(PROFILE, dataset, model, seed, subject)
    folder = result_folder(dataset, model, seed, subject)
    if result.get('input_manifest_sha256') != manifest_hash:
        raise RuntimeError(f'{folder}: training input differs from the verified new cache')
    with (folder / 'history.csv').open(newline='') as f:
        history = list(csv.DictReader(f))
    epochs = EXPECTED_EPOCHS[(dataset, model)]
    if [int(row['epoch']) for row in history] != list(range(1, epochs + 1)):
        raise RuntimeError(f'{folder}: incomplete or duplicate epoch history')
    for filename, hash_field in (('final_model.pt', 'final_model_sha256'),
                                  ('test_predictions.npz', 'test_predictions_sha256')):
        if sha256_file(folder / filename) != result.get(hash_field):
            raise RuntimeError(f'{folder}: {filename} differs from the trained result')
    if not (folder / 'training_state.pt').is_file():
        raise FileNotFoundError(f'{folder}: missing resumable training state')
    return result


def summarize_scores(scores: list[float]) -> dict:
    return {'seed_mean': statistics.mean(scores),
            'seed_sample_std': statistics.stdev(scores),
            'seed_scores': {str(seed): score for seed, score in zip(SEEDS, scores)}}


def compare_reference(dataset: str, model: str, fresh: dict, new_scores: list[float],
                      new_hashes: dict, new_rows: list[tuple]) -> dict:
    comparisons = {}
    for name, old_profile, old_variant in REFERENCES:
        old_input = input_folder(dataset, model, old_profile, old_variant)
        old_rows = trial_table(old_input)
        old_hashes = {filename: sha256_file(old_input / filename)
                      for filename in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv')}
        old_manifest = read_json(old_input / 'manifest.json')
        for filename, digest in old_hashes.items():
            if old_manifest['files'][filename]['sha256'] != digest:
                raise RuntimeError(f'Reference input content mismatch: {old_input / filename}')
        flags = {
            'trial_uid_order_matches': [row[0] for row in new_rows] == [row[0] for row in old_rows],
            'trial_label_order_matches': [row[1] for row in new_rows] == [row[1] for row in old_rows],
            'trial_subject_order_matches': [row[2] for row in new_rows] == [row[2] for row in old_rows],
            'y_array_sha256_matches': new_hashes['y.npy'] == old_hashes['y.npy'],
            'subject_array_sha256_matches': new_hashes['subjects.npy'] == old_hashes['subjects.npy'],
            'input_tensor_sha256_matches': new_hashes['X.npy'] == old_hashes['X.npy'],
        }
        old_results = {(seed, subject): result_for(old_profile, dataset, model, seed, subject)
                       for seed in SEEDS for subject in SUBJECTS}
        old_scores = [statistics.mean(old_results[(seed, subject)]['metrics']['accuracy']
                                     for subject in SUBJECTS) for seed in SEEDS]
        per_fold = []
        for seed in SEEDS:
            for subject in SUBJECTS:
                new, old = fresh[(seed, subject)], old_results[(seed, subject)]
                per_fold.append({
                    'seed': seed, 'held_out_subject': subject,
                    'new_accuracy': new['metrics']['accuracy'],
                    'reference_accuracy': old['metrics']['accuracy'],
                    'accuracy_delta_new_minus_reference': new['metrics']['accuracy'] - old['metrics']['accuracy'],
                    'initial_state_matches': new['initial_state_sha256'] == old['initial_state_sha256'],
                    'first_epoch_batch_order_matches': new['first_epoch_train_order_sha256'] == old['first_epoch_train_order_sha256'],
                    'train_test_counts_and_input_shape_match': all(new[field] == old[field]
                        for field in ('train_trials', 'test_trials', 'input_shape')),
                    'new_training_seconds': new['training_seconds'],
                    'reference_training_seconds': old['training_seconds'],
                })
        paired = (all(flags[field] for field in ('trial_uid_order_matches', 'trial_label_order_matches',
                  'trial_subject_order_matches', 'y_array_sha256_matches', 'subject_array_sha256_matches'))
                  and all(row['initial_state_matches'] and row['first_epoch_batch_order_matches']
                          and row['train_test_counts_and_input_shape_match'] for row in per_fold))
        comparisons[name] = {
            'reference_profile': old_profile, 'reference_accuracy': summarize_scores(old_scores),
            'paired_accuracy_delta': summarize_scores([new - old for new, old in zip(new_scores, old_scores)]),
            'input_pair_checks': flags, 'all_pair_checks_match': paired,
            'new_input_file_sha256': new_hashes, 'reference_input_file_sha256': old_hashes,
            'reference_training_seconds_total': sum(item['training_seconds'] for item in old_results.values()),
            'per_fold': per_fold,
        }
    return comparisons


def aggregate(manifest_hashes: dict, dispatch_started: float) -> dict:
    experiments = {}
    for dataset, model in COMBINATIONS:
        manifest_hash = manifest_hashes[(dataset, model)]
        fresh = {(seed, subject): validate_fresh_fold((dataset, model, seed), subject, manifest_hash)
                 for seed in SEEDS for subject in SUBJECTS}
        folder = input_folder(dataset, model)
        input_hashes = {name: sha256_file(folder / name)
                        for name in ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv')}
        rows = trial_table(folder)
        metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1')
        per_metric = {metric: summarize_scores([
            statistics.mean(fresh[(seed, subject)]['metrics'][metric] for subject in SUBJECTS)
            for seed in SEEDS]) for metric in metrics}
        comparisons = (compare_reference(dataset, model, fresh,
                                         [per_metric['accuracy']['seed_scores'][str(seed)] for seed in SEEDS],
                                         input_hashes, rows)
                       if dataset == 'BNCI2014001-4' and model == 'cbramod' else {})
        item = {
            'status': 'complete', 'dataset': dataset, 'model': model,
            'profile': PROFILE, 'variant': VARIANT, 'selected_session': '0train',
            'seeds': list(SEEDS), 'folds_complete': 27,
            'epochs_per_fold': EXPECTED_EPOCHS[(dataset, model)],
            'input_manifest_sha256': manifest_hash, 'input_file_sha256': input_hashes,
            'metrics': per_metric, 'accuracy': per_metric['accuracy'],
            'training_seconds_total': sum(row['training_seconds'] for row in fresh.values()),
            'training_wall_seconds_total': sum(row['wall_seconds'] for row in fresh.values()),
            'reference_comparisons': comparisons,
            'same_seed_reference_available': bool(comparisons),
            'reference_note': ('paired against the previous 0/1/2 runs'
                               if comparisons else 'no existing reference using these seeds and this aligned protocol'),
            'per_seed': {str(seed): {
                'subject_equal_mean': {metric: per_metric[metric]['seed_scores'][str(seed)] for metric in metrics},
                'training_seconds': sum(fresh[(seed, subject)]['training_seconds'] for subject in SUBJECTS),
                'folds': [{'held_out_subject': subject, 'metrics': fresh[(seed, subject)]['metrics'],
                           'training_seconds': fresh[(seed, subject)]['training_seconds']}
                          for subject in SUBJECTS],
            } for seed in SEEDS},
        }
        experiments[f'{model}/{dataset}'] = item
        path = RESULT_ROOT / PROFILE / dataset / model / 'paired_source_comparison.json'
        atomic_json(path, item)
    return {
        'status': 'complete', 'finished_at_utc': utc_now(), 'profile': PROFILE,
        'source_manifest': str(SOURCE_MANIFEST), 'source_manifest_sha256': sha256_file(SOURCE_MANIFEST),
        'source_cache_policy': 'all_5184_trials_and_both_sessions_saved',
        'training_session': '0train', 'seeds': list(SEEDS), 'folds_complete': 108,
        'datasets': list(DATASETS), 'models': list(MODELS), 'experiments': experiments,
        'training_seconds_total': sum(item['training_seconds_total'] for item in experiments.values()),
        'dispatcher_elapsed_seconds_including_gpu_waits': time.time() - dispatch_started,
    }


def main() -> int:
    EXEC.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('another wideband 14001 dispatcher owns the lock; leaving its workers running')
            return 0
        active: list[dict] = []
        pending = deque(TASKS)
        completed: list[str] = []
        failed: list[dict] = []
        retries = {task_key(task): 0 for task in TASKS}
        started = time.time()
        try:
            write_status('validating_inputs', list(pending), active)
            manifests, manifest_hashes = validate_inputs()
            log('verified complete all-session source and four model inputs; 0train only is selected for training')
            run_preflight(manifest_hashes)
            while pending or active:
                for item in list(active):
                    code = item['proc'].poll()
                    if code is None:
                        continue
                    active.remove(item)
                    item['log_handle'].close()
                    dataset, model, _ = item['task']
                    key = task_key(item['task'])
                    if code == 0:
                        # Require trained results, rather than treating exit code as completion.
                        for subject in SUBJECTS:
                            validate_fresh_fold(item['task'], subject, manifest_hashes[(dataset, model)])
                        completed.append(key)
                        log(f'{key} complete in {(time.time() - item["started"]) / 60:.1f} minutes')
                    elif retries[key] < 1:
                        retries[key] += 1
                        pending.appendleft(item['task'])
                        log(f'{key} failed code={code}; queued one checkpoint-resuming retry; log={item["log_path"]}')
                    else:
                        failed.append({**task_record(item['task']), 'return_code': code,
                                       'log': str(item['log_path'])})
                        log(f'{key} failed twice; remaining jobs continue; log={item["log_path"]}')
                while pending and len(active) < MAX_CONCURRENT:
                    gpu = choose_gpu(gpu_snapshot(), {item['gpu'] for item in active})
                    if gpu is None:
                        break
                    task = pending.popleft()
                    dataset, model, seed = task
                    log_path = WORKER_LOGS / model / dataset / f'seed_{seed}_gpu{gpu}_try{retries[task_key(task)] + 1}.log'
                    active.append(start_worker(task, gpu, log_path))
                write_status('running' if active else 'waiting_for_nonzero_gpu',
                             list(pending), active, completed_jobs=completed,
                             failed_jobs=failed, retry_count_by_job=retries,
                             input_manifest_sha256={f'{model}/{dataset}': digest
                                 for (dataset, model), digest in manifest_hashes.items()},
                             dispatcher_pid=os.getpid(), dispatcher_started_epoch_s=started,
                             input_shapes={f'{model}/{dataset}': manifest['input_shape']
                                           for (dataset, model), manifest in manifests.items()})
                if pending or active:
                    time.sleep(POLL_SECONDS)
            if failed or sorted(completed) != sorted(task_key(task) for task in TASKS):
                raise RuntimeError(f'Incomplete workers; failed={failed}; complete={completed}')
            write_status('aggregating', [], [], completed_jobs=completed)
            report = aggregate(manifest_hashes, started)
            report_path = RESULT_ROOT / PROFILE / 'wideband_14001_loso_summary.json'
            atomic_json(report_path, report)
            write_status('complete', [], [], completed_jobs=completed,
                         comparison=str(report_path), accuracy={key: item['accuracy']
                             for key, item in report['experiments'].items()},
                         training_seconds_total=report['training_seconds_total'],
                         input_manifest_sha256={f'{model}/{dataset}': digest
                             for (dataset, model), digest in manifest_hashes.items()})
            log(f'108 folds complete; report={report_path}')
            return 0
        except Exception as exc:
            write_status('failed', list(pending), active, completed_jobs=completed,
                         failed_jobs=failed, error=f'{type(exc).__name__}: {exc}',
                         active_workers_continue=bool(active))
            log(f'fatal: {type(exc).__name__}: {exc}')
            return 1
        finally:
            for item in active:
                item['log_handle'].close()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    raise SystemExit(main())
