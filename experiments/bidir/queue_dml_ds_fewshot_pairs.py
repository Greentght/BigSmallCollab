"""Dispatch subject-wise DML-DS across all supported model pairs."""
import csv
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.bidir.queue_dcm_loso import gpu_processes, job_is_running
from experiments.storage import require_external_output, require_report_output

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/fewshot_dml_ds_pairs.yaml'


def active_workers():
    output = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    active = {}
    for line in output.splitlines():
        parts = line.split()
        if 'experiments.bidir.run_dml_ds_fewshot_pairs' in parts and '--gpu' in parts:
            active[int(parts[0])] = int(parts[parts.index('--gpu') + 1])
    return active


def write_reports(root, cfg):
    import numpy as np

    rows = []
    splits = {}
    sources = {}
    datasets = dict(cfg['datasets'])
    original_root = Path('/data1/llx/BigSmallcollab/results/bidir/dcm_fewshot_0014_v1/runs')
    expanded_root = Path('/data1/llx/BigSmallcollab/results/bidir/dcm_fewshot_all_v1/runs')
    pairs = [f'{big}_{small}' for big, small in itertools.product(cfg['teachers'], cfg['students'])]
    for dataset, count in datasets.items():
        for pair, condition, seed, subject in itertools.product(pairs, cfg['conditions'], cfg['seeds'], range(count)):
            if pair in cfg['reuse_pairs']:
                dataset_root = original_root if dataset == 'BNCI2014001-4' else expanded_root / dataset
            else:
                dataset_root = root / 'runs' / dataset
            path = dataset_root / pair / condition / f'subject_{subject}' / f'seed_{seed}' / 'evaluation.json'
            result = json.loads(path.read_text())
            identity = result['identity']
            if (identity['dataset'], identity['pair'], identity['condition'], identity['seed'], identity['subject']) != (dataset, pair, condition, seed, subject):
                raise RuntimeError(f'Result identity mismatch: {path}')
            if identity['protocol'] != 'fewshot_stratified_random_30_70' or identity['smoke']:
                raise RuntimeError(f'Unexpected evaluation protocol: {path}')
            source = identity['source_manifest_sha256']
            if dataset in sources and sources[dataset] != source:
                raise RuntimeError(f'Source identity mismatch: {path}')
            sources[dataset] = source
            split = (identity['train_uid_sha256'], identity['test_uid_sha256'])
            split_key = (dataset, seed, subject)
            if split_key in splits and splits[split_key] != split:
                raise RuntimeError(f'Calibration/test split mismatch: {path}')
            splits[split_key] = split
            with np.load(path.parent / 'predictions.npz', allow_pickle=False) as predictions:
                uids = predictions['uids'].tolist()
                uid_hash = hashlib.sha256(json.dumps(uids, ensure_ascii=False).encode()).hexdigest()
                if len(uids) != result['test_count'] or len(set(uids)) != len(uids) or uid_hash != split[1]:
                    raise RuntimeError(f'Test UID mismatch: {path}')
                for role in ('big', 'small'):
                    accuracy = float(np.mean(predictions[role] == predictions['y']) * 100)
                    if not np.isclose(accuracy, result[role + '_accuracy'], rtol=0, atol=1e-9):
                        raise RuntimeError(f'Prediction accuracy mismatch: {path}')
            rows.append([dataset, pair, condition, seed, subject, result['big_accuracy'], result['small_accuracy'], str(path)])
    folder = ROOT / 'results/dml_ds_fewshot_pairs_v1'
    target = require_report_output(folder / 'subject_accuracy.csv')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'pair', 'condition', 'seed', 'subject_zero_based', 'big_accuracy_percent', 'small_accuracy_percent', 'source'])
        writer.writerows(rows)
    os.replace(temporary, target)
    target = require_report_output(folder / 'summary.csv')
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'pair', 'condition', 'big_accuracy_percent', 'small_accuracy_percent',
                         'big_delta_vs_none_pp', 'small_delta_vs_none_pp'])
        for dataset, pair in itertools.product(datasets, pairs):
            means = {}
            for condition in cfg['conditions']:
                selected = [row for row in rows if row[0] == dataset and row[1] == pair and row[2] == condition]
                means[condition] = [sum(row[index] for row in selected) / len(selected) for index in (5, 6)]
            for condition in cfg['conditions']:
                scores = means[condition]
                deltas = [scores[i] - means['none'][i] for i in (0, 1)]
                writer.writerow([dataset, pair, condition] + [f'{value:.2f}' for value in scores + deltas])
    os.replace(temporary, target)


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    root = require_external_output(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        pairs = [f'{big}_{small}' for big, small in itertools.product(cfg['teachers'], cfg['students'])
                 if f'{big}_{small}' not in cfg['reuse_pairs']]
        for dataset, count in cfg['datasets'].items():
            for subject, seed, condition, pair in itertools.product(range(count), cfg['seeds'], cfg['conditions'], pairs):
                folder = root / 'runs' / dataset / pair / condition / f'subject_{subject}' / f'seed_{seed}'
                pending.append((dataset, subject, seed, condition, pair, folder))
        pending.sort(key=lambda job: (job[-1] / 'evaluation.json').exists())
        files = [CONFIG, ROOT / 'experiments/bidir/run_dml_ds_fewshot_pairs.py', ROOT / 'collab/dcm.py', ROOT / 'collab/dml_ds_heads.py',
                 ROOT / 'data/split.py', ROOT / 'data/loso_source.py', ROOT / 'data/preproc.py',
                 ROOT / 'config.py', ROOT / 'models/base.py', ROOT / 'models/cbramod/criss_cross_transformer.py',
                 ROOT / 'experiments/finetune/run_neurogpt_loso.py', ROOT / 'experiments/distill/run_fewshot_structure_kd.py',
                 ROOT / 'experiments/finetune/labram_alexmi_broadband.py']
        files.extend(ROOT / f'configs/datasets/{dataset}.yaml' for dataset in cfg['datasets'])
        for model in ('mirepnet', 'cbramod', 'neurogpt', 'ifnet', 'eegnet', 'adfcnn'):
            files.extend([ROOT / f'configs/models/{model}.yaml', ROOT / f'models/{model}/adapter.py',
                          ROOT / ('models/eegnet/residual_eegnet.py' if model == 'eegnet' else 'models/mirepnet/mlm.py' if model == 'mirepnet' else f'models/{model}/{model}.py')])
        def fingerprint():
            return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        original = fingerprint()
        workers = {}
        failure = None
        while pending or workers:
            for pid, (gpu, process, job, log) in list(workers.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del workers[pid]
                    if code:
                        failure = f'Worker failed with exit {code}: {job}; see {job[-1] / "worker.log"}'
                        print(failure, flush=True)
                    else:
                        print(f'completed {job[:5]}', flush=True)
            if fingerprint() != original:
                failure = 'Experiment source or configuration changed while queue was running'
            if failure: pending.clear()
            devices = gpu_processes() if not failure else {}
            running = active_workers() if not failure else {}
            for gpu, registered in devices.items():
                reserved = sum(assigned == gpu and pid not in registered for pid, assigned in running.items())
                slots = max(0, 3 - len(registered) - reserved)
                while pending and slots and len(running) < 6:
                    available = next((i for i, job in enumerate(pending) if not job_is_running(job[-1])), None)
                    if available is None: break
                    job = pending.pop(available)
                    dataset, subject, seed, condition, pair, folder = job
                    folder.mkdir(parents=True, exist_ok=True)
                    log = (folder / 'worker.log').open('a')
                    command = [sys.executable, '-u', '-B', '-m', 'experiments.bidir.run_dml_ds_fewshot_pairs',
                               '--dataset', dataset,
                               '--pair', pair, '--condition', condition, '--subject', str(subject),
                               '--seed', str(seed), '--gpu', str(gpu)]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    workers[process.pid] = (gpu, process, job, log)
                    running[process.pid] = gpu
                    slots -= 1
                    print(f'started {job[:5]} on GPU {gpu}, pid={process.pid}', flush=True)
            if pending or workers:
                # Completed few-shot units often take less than 30 seconds.
                # Reap them promptly while keeping idle GPU checks bounded.
                time.sleep(3 if workers else 10)
        if failure: raise RuntimeError(failure)
        write_reports(root, cfg)
        print('All cells complete and reports written', flush=True)


if __name__ == '__main__':
    main()
