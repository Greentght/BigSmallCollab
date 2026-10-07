#!/usr/bin/env python
"""Persistent non-GPU0 dispatcher for LOSO configuration-alignment runs.

The dispatcher waits for validated input manifests, runs small GPU preflights,
then completes the 001-4 source bridge before dispatching the formal 3-dataset
reference-aligned jobs. Each worker is resumable at epoch and fold boundaries.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
INPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/model_inputs'
RESULT_ROOT = RESULTS_ROOT / 'reproductions/loso_config_alignment_v2'
LOG_ROOT = RESULT_ROOT / 'execution_logs/jobs'
WORKER = ROOT / 'experiments/finetune/run_loso_config_alignment.py'
CONDA = '/home/lixinli/anaconda3/bin/conda'


def log(message: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {message}'
    print(line, flush=True)
    with (RESULT_ROOT / 'execution_logs/dispatcher.log').open('a') as f:
        f.write(line + '\n')


def read_status(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def gpu_snapshot() -> list[dict]:
    cmd = [
        'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
        '--format=csv,noheader,nounits',
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    devices = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 3:
            continue
        devices.append({'gpu': int(fields[0]), 'utilization_pct': int(fields[1]),
                        'free_memory_mib': int(fields[2])})
    return devices


def ready_gpu(active_gpus: set[int], allowed: list[int], max_util: int,
              min_free_mib: int) -> int | None:
    candidates = [x for x in gpu_snapshot()
                  if x['gpu'] != 0 and x['gpu'] in allowed
                  and x['gpu'] not in active_gpus
                  and x['utilization_pct'] <= max_util
                  and x['free_memory_mib'] >= min_free_mib]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x['utilization_pct'], -x['free_memory_mib'], x['gpu']))
    return candidates[0]['gpu']


def required_inputs() -> list[Path]:
    paths = []
    for dataset in ('BNCI2014001-4', 'BNCI2014004', 'BNCI2015001'):
        for model in ('cbramod', 'eegnet'):
            paths.append(INPUT_ROOT / 'reference_aligned' / dataset / model
                         / 'rebuilt_source' / 'manifest.json')
    for model in ('cbramod', 'eegnet'):
        for variant in ('legacy_cache', 'rebuilt_source'):
            paths.append(INPUT_ROOT / 'source_bridge' / 'BNCI2014001-4' / model
                         / variant / 'manifest.json')
    return paths


def input_identity(profile: str, dataset: str, model: str, variant: str) -> tuple[list, dict]:
    folder = INPUT_ROOT / profile / dataset / model / variant
    manifest = read_status(folder / 'manifest.json')
    trials = []
    with (folder / 'trials.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            trials.append((row['trial_uid'], int(row['subject_zero_based']),
                           int(row['label_id'])))
    if len(trials) != manifest.get('trial_count'):
        raise RuntimeError(f'{folder}: trial row count differs from manifest')
    for filename, record in manifest.get('files', {}).items():
        path = folder / filename
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise RuntimeError(f'{folder}: missing/truncated input file {filename}')
    return trials, manifest


def validate_input_identity() -> None:
    expected_count = {'BNCI2014001-4': 2592, 'BNCI2014004': 1400,
                      'BNCI2015001': 2400}
    expected_shapes = {
        'BNCI2014001-4': {'cbramod': [22, 4, 200], 'eegnet': [22, 1250]},
        'BNCI2014004': {'cbramod': [3, 5, 200], 'eegnet': [3, 1250]},
        'BNCI2015001': {'cbramod': [13, 4, 200], 'eegnet': [13, 1250]},
    }
    for dataset, count in expected_count.items():
        identities = {}
        manifests = {}
        for model in ('cbramod', 'eegnet'):
            identity, manifest = input_identity(
                'reference_aligned', dataset, model, 'rebuilt_source')
            if len(identity) != count:
                raise RuntimeError(f'{dataset}/{model}: expected {count} trials, got {len(identity)}')
            if manifest.get('input_shape') != expected_shapes[dataset][model]:
                raise RuntimeError(
                    f'{dataset}/{model}: expected input {expected_shapes[dataset][model]}, '
                    f"got {manifest.get('input_shape')}")
            identities[model] = identity
            manifests[model] = manifest
        if identities['cbramod'] != identities['eegnet']:
            raise RuntimeError(f'{dataset}: CBraMod and EEGNet formal trial UIDs/labels differ')
        if manifests['cbramod']['input_shape'][0] != manifests['eegnet']['input_shape'][0]:
            raise RuntimeError(f'{dataset}: model input channel counts differ')
        log(f'validated formal trial identity: {dataset} n={count} '
            f'channels={manifests["eegnet"]["input_shape"][0]}')

    bridge_by_model = {}
    for model in ('cbramod', 'eegnet'):
        bridge_by_model[model] = {}
        for variant in ('legacy_cache', 'rebuilt_source'):
            identity, manifest = input_identity(
                'source_bridge', 'BNCI2014001-4', model, variant)
            if len(identity) != 2592:
                raise RuntimeError(f'001-4 bridge {model}/{variant}: expected 2592 trials')
            shape = [22, 4, 200] if model == 'cbramod' else [22, 1000]
            if manifest.get('input_shape') != shape:
                raise RuntimeError(f'001-4 bridge {model}/{variant}: expected {shape}, '
                                   f"got {manifest.get('input_shape')}")
            bridge_by_model[model][variant] = (identity, manifest)
        if bridge_by_model[model]['legacy_cache'][0] != bridge_by_model[model]['rebuilt_source'][0]:
            raise RuntimeError(f'001-4 bridge {model}: source variants are not trial/label aligned')
    ref_identity = bridge_by_model['cbramod']['legacy_cache'][0]
    if bridge_by_model['eegnet']['legacy_cache'][0] != ref_identity:
        raise RuntimeError('001-4 bridge model trial orders differ')
    log('validated paired 001-4 bridge identity across both models and both source variants')


def wait_for_inputs(poll_seconds: int) -> None:
    req = required_inputs()
    while True:
        missing = [str(p) for p in req if not p.is_file()]
        if not missing:
            log(f'all {len(req)} required model-input manifests are present')
            return
        log(f'waiting for {len(missing)} model-input manifests; first missing: {missing[0]}')
        update_dispatch_status('waiting_for_inputs', [], [], missing=missing)
        time.sleep(poll_seconds)


def make_task(name: str, profile: str, dataset: str, model: str, seed: int,
              preflight_steps: int | None = None) -> dict:
    return {'name': name, 'profile': profile, 'dataset': dataset, 'model': model,
            'seed': seed, 'preflight_steps': preflight_steps, 'attempt': 0}


def command_for(task: dict, gpu: int) -> list[str]:
    args = [
        CONDA, 'run', '--no-capture-output', '-n', 'cbramod', 'python', '-u',
        str(WORKER), '--profile', task['profile'], '--dataset', task['dataset'],
        '--model', task['model'], '--seed', str(task['seed']), '--gpu', str(gpu),
    ]
    if task.get('preflight_steps') is not None:
        args += ['--preflight-steps', str(task['preflight_steps'])]
    return args


def update_dispatch_status(phase: str, pending: list[dict], active: list[dict], **extra) -> None:
    folder = RESULT_ROOT / 'execution_logs'
    folder.mkdir(parents=True, exist_ok=True)
    status = {
        'phase': phase,
        'updated_at_epoch_s': time.time(),
        'pending_jobs': [t['name'] for t in pending],
        'active_jobs': [
            {'name': a['task']['name'], 'gpu': a['gpu'], 'pid': a['proc'].pid,
             'elapsed_seconds': time.time() - a['started']}
            for a in active
        ],
        'gpu_snapshot': gpu_snapshot(),
        **extra,
    }
    tmp = folder / 'dispatcher_status.json.partial'
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, folder / 'dispatcher_status.json')


def run_queue(tasks: list[dict], phase: str, allowed_gpus: list[int], max_concurrent: int,
              max_util: int, min_free_mib: int, poll_seconds: int) -> None:
    pending = deque(tasks)
    active: list[dict] = []
    completed = []
    failed = []
    last_wait_log = 0.0
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for item in list(active):
            code = item['proc'].poll()
            if code is None:
                continue
            active.remove(item)
            task = item['task']
            (item['log_handle']).close()
            if code == 0:
                completed.append(task['name'])
                log(f"finished {phase}/{task['name']} gpu={item['gpu']} "
                    f"elapsed_min={(time.time()-item['started'])/60:.1f}")
            elif task['attempt'] < 1:
                task['attempt'] += 1
                pending.appendleft(task)
                log(f"worker failed code={code}; retrying once: {phase}/{task['name']} "
                    f"gpu={item['gpu']} log={item['log_path']}")
            else:
                failed.append({'name': task['name'], 'return_code': code,
                               'gpu': item['gpu'], 'log': str(item['log_path'])})
                log(f"worker failed permanently: {phase}/{task['name']} code={code}; "
                    f"log={item['log_path']}")

        while pending and len(active) < max_concurrent:
            occupied = {x['gpu'] for x in active}
            gpu = ready_gpu(occupied, allowed_gpus, max_util, min_free_mib)
            if gpu is None:
                if time.time() - last_wait_log >= 60:
                    log(f'{phase}: waiting for a nonzero GPU below {max_util}% '
                        f'with >= {min_free_mib} MiB free; pending={len(pending)} active={len(active)}')
                    last_wait_log = time.time()
                break
            task = pending.popleft()
            safe_name = task['name'].replace('/', '_')
            log_path = LOG_ROOT / f'{phase}_{safe_name}_try{task["attempt"] + 1}_gpu{gpu}.log'
            log_file = log_path.open('ab', buffering=0)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT)
            proc = subprocess.Popen(command_for(task, gpu), cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log_file,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    close_fds=True)
            active.append({'task': task, 'gpu': gpu, 'proc': proc,
                           'started': time.time(), 'log_handle': log_file,
                           'log_path': log_path})
            log(f"started {phase}/{task['name']} pid={proc.pid} gpu={gpu}; log={log_path}")

        update_dispatch_status(phase, list(pending), active,
                               completed_jobs=completed, failed_jobs=failed)
        if failed:
            raise RuntimeError(f'{phase} stopped after worker failures: {failed}')
        if pending or active:
            time.sleep(poll_seconds)


def aggregate_formal() -> dict:
    aggregate = {'status': 'complete', 'models': {}}
    for dataset in ('BNCI2014001-4', 'BNCI2014004', 'BNCI2015001'):
        for model in ('cbramod', 'eegnet'):
            per_seed = {}
            for seed in (0, 1, 2):
                path = RESULT_ROOT / 'reference_aligned' / dataset / model / f'seed_{seed}' / 'seed_summary.json'
                if not path.is_file():
                    aggregate['status'] = 'partial'
                    continue
                per_seed[str(seed)] = read_status(path).get('subject_equal_mean', {})
            metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1')
            combined = {}
            for metric in metrics:
                vals = [float(v[metric]) for v in per_seed.values() if metric in v]
                combined[metric] = {
                    'seed_mean': float(np.mean(vals)) if vals else None,
                    'seed_sample_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    'per_seed': {s: row.get(metric) for s, row in per_seed.items()},
                }
            aggregate['models'].setdefault(dataset, {})[model] = combined
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument('--max-concurrent', type=int, default=4)
    parser.add_argument('--max-util', type=int, default=40)
    parser.add_argument('--min-free-mib', type=int, default=10000)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    allowed = sorted({int(x) for x in args.gpus.split(',') if x.strip()})
    if 0 in allowed:
        raise SystemExit('GPU 0 cannot be assigned to this experiment')
    if not allowed or args.max_concurrent < 1:
        raise SystemExit('need at least one nonzero GPU and max-concurrent >= 1')
    (RESULT_ROOT / 'execution_logs').mkdir(parents=True, exist_ok=True)
    wait_for_inputs(args.poll_seconds)
    validate_input_identity()

    preflights = [
        make_task(f'{dataset}_{model}', 'reference_aligned', dataset, model, 0, 1)
        for dataset in ('001-4', '004', '5001')
        for model in ('cbramod', 'eegnet')
    ]
    run_queue(preflights, 'preflight', allowed, min(args.max_concurrent, 4),
              args.max_util, args.min_free_mib, args.poll_seconds)

    bridge = [
        make_task(f'{model}_seed666_0014', 'source_bridge', '001-4', model, 666)
        for model in ('cbramod', 'eegnet')
    ]
    bridge = [task for task in bridge if not (
        (RESULT_ROOT / 'source_bridge' / 'BNCI2014001-4' / task['model']
         / 'seed_666' / 'bridge_summary.json').is_file()
        and read_status(RESULT_ROOT / 'source_bridge' / 'BNCI2014001-4'
                        / task['model'] / 'seed_666' / 'bridge_summary.json').get('status') == 'complete'
    )]
    if not bridge:
        log('source-bridge stage already complete; skipping duplicate workers')
    else:
        log(f'source-bridge workers pending: {[task["name"] for task in bridge]}')
        run_queue(bridge, 'source_bridge', allowed, min(args.max_concurrent, 2),
                  args.max_util, args.min_free_mib, args.poll_seconds)

    formal = [
        make_task(f'{dataset}_{model}_seed{seed}', 'reference_aligned', dataset, model, seed)
        for dataset in ('001-4', '004', '5001')
        for model in ('cbramod', 'eegnet')
        for seed in (0, 1, 2)
    ]
    run_queue(formal, 'reference_aligned', allowed, args.max_concurrent,
              args.max_util, args.min_free_mib, args.poll_seconds)
    aggregate = aggregate_formal()
    aggregate['source_bridge'] = {}
    for model in ('cbramod', 'eegnet'):
        path = RESULT_ROOT / 'source_bridge' / 'BNCI2014001-4' / model / 'seed_666' / 'bridge_summary.json'
        aggregate['source_bridge'][model] = read_status(path) if path.exists() else None
    out = RESULT_ROOT / 'reference_aligned' / 'aggregate.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + '\n')
    update_dispatch_status('complete' if aggregate['status'] == 'complete' else 'partial', [], [],
                           aggregate=str(out))
    log(f"all experiment phases finished; aggregate={out}")


if __name__ == '__main__':
    main()
