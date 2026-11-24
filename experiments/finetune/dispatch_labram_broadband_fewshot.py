"""Run and summarize broadband LaBraM subject-wise few-shot experiments.

The dispatcher owns only its own workers. It performs a one-epoch 001 smoke
test, then completes 001 before running the other requested datasets. The
per-GPU and per-cell locks are inherited by workers so restarting the manager
cannot start duplicate work while an orphaned worker is still running.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from experiments.storage import REPORTS_ROOT, RESULTS_ROOT, require_external_output, require_report_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    'BNCI2014001': 9,
    'BNCI2014001-4': 9,
    'BNCI2014004': 9,
    'BNCI2015001': 12,
    'AlexMI': 8,
}
SEEDS = (666, 667, 668)
CODE_FILES = (
    'experiments/finetune/dispatch_labram_broadband_fewshot.py',
    'experiments/finetune/run_labram_broadband_fewshot.py',
    'experiments/finetune/labram_alexmi_broadband.py',
    'experiments/distill/run_fewshot_structure_kd.py',
    'experiments/storage.py', 'data/loso_source.py', 'data/split.py',
    'data/preproc.py', 'data/channels.py', 'config.py', 'collab/seed.py',
    'models/base.py', 'models/__init__.py', 'models/labram/adapter.py',
    'models/labram/modeling_finetune.py', 'models/labram/montage.py',
    'models/labram/optim_factory.py', 'configs/models/labram.yaml',
    'configs/datasets/BNCI2014001.yaml', 'configs/datasets/BNCI2014001-4.yaml',
    'configs/datasets/BNCI2014004.yaml', 'configs/datasets/BNCI2015001.yaml',
    'configs/datasets/AlexMI.yaml',
)
DATA_MANIFESTS = {
    'BNCI2014001': Path('/data1/llx/BNCI2014001/wide_fre_band/manifest.json'),
    'BNCI2014001-4': Path('/data1/llx/BNCI2014001/wide_fre_band/manifest.json'),
    'BNCI2014004': Path('/data1/llx/BNCI2014004/wide_fre_band/manifest.json'),
    'BNCI2015001': Path('/data1/llx/BNCI2015001/wide_fre_band/manifest.json'),
    'AlexMI': Path('/data1/llx/AlexMI/wide_fre_band/manifest.json'),
}
PRETRAIN = Path('/data1/llx/pre_weight/labram-base.pth')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def campaign_identity(datasets):
    return {
        'datasets': list(datasets),
        'seeds': list(SEEDS),
        'split': 'within_subject_stratified_30_percent_train_70_percent_test',
        'code_sha256': {name: file_sha256(PROJECT_ROOT / name) for name in CODE_FILES},
        'source_manifest_sha256': {name: file_sha256(DATA_MANIFESTS[name]) for name in datasets},
        'pretrained_sha256': file_sha256(PRETRAIN),
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


def atomic_csv(path, rows):
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', newline='', delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, path)


def paths(run_id, dataset, subject, seed):
    relative = Path(run_id) / dataset / f'subject{subject + 1:02d}' / f'seed{seed}'
    record = RESULTS_ROOT / 'labram_broadband_fewshot' / relative / 'completed.json'
    report = REPORTS_ROOT / 'fewshot/labram_broadband' / relative / 'predictions.csv'
    return record, report


def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_result(run_id, dataset, subject, seed, campaign, *, smoke=False):
    record_path, report_path = paths(run_id, dataset, subject, seed)
    if not record_path.exists():
        return None
    with record_path.open(encoding='utf-8') as stream:
        record = json.load(stream)
    identity = record['identity']
    for key, value in (('dataset', dataset), ('subject', subject), ('seed', seed),
                       ('protocol', 'subject_session_stratified_30_70'), ('smoke', smoke),
                       ('source_manifest_sha256', campaign['source_manifest_sha256'][dataset]),
                       ('pretrained_sha256', campaign['pretrained_sha256'])):
        if identity.get(key) != value:
            raise ValueError(f'{record_path}: {key} identity mismatch')
    if record.get('identity_sha256') != digest_json(identity):
        raise ValueError(f'{record_path}: identity checksum mismatch')
    if any(campaign['code_sha256'].get(key) != value
           for key, value in identity['code_sha256'].items()):
        raise ValueError(f'{record_path}: implementation checksum mismatch')
    training = identity['training']
    expected_epochs = 1 if smoke else int(identity['model_config']['epochs'])
    if (training['epochs'] != expected_epochs or training['optimizer'] != 'AdamW'
            or training['evaluation'] != 'fixed_final_epoch'
            or training['dataloader_workers'] != 0
            or training['torch_intraop_threads'] != 1
            or training['torch_interop_threads'] != 1):
        raise ValueError(f'{record_path}: unexpected training configuration')
    train_uids = record['train_trial_uids']
    test_uids = record['test_trial_uids']
    n_train, n_test = int(record['n_train']), int(record['n_test'])
    if (len(train_uids) != n_train or len(test_uids) != n_test
            or len(set(train_uids)) != n_train or len(set(test_uids)) != n_test
            or set(train_uids) & set(test_uids) or n_train < 1
            or n_test != math.ceil((n_train + n_test) * 0.7)):
        raise ValueError(f'{record_path}: invalid 30/70 split or trial UIDs')
    if (identity['train_uid_sha256'] != digest_json(train_uids)
            or identity['test_uid_sha256'] != digest_json(test_uids)):
        raise ValueError(f'{record_path}: trial UID checksum mismatch')
    if Path(record['predictions_path']).resolve() != report_path.resolve():
        raise ValueError(f'{record_path}: unexpected prediction path')
    if record['predictions_sha256'] != file_sha256(report_path):
        raise ValueError(f'{report_path}: prediction checksum mismatch')
    with report_path.open(newline='', encoding='utf-8') as stream:
        predictions = list(csv.DictReader(stream))
    if [row['trial_uid'] for row in predictions] != test_uids:
        raise ValueError(f'{report_path}: test trial UID mismatch')
    num_classes = int(identity['num_classes'])
    if any(int(row[name]) < 0 or int(row[name]) >= num_classes
           for row in predictions for name in ('y_true', 'y_pred')):
        raise ValueError(f'{report_path}: invalid class label')
    labels = [int(row['y_true']) for row in predictions]
    predicted = [int(row['y_pred']) for row in predictions]
    if set(labels) != set(range(num_classes)):
        raise ValueError(f'{report_path}: test labels do not cover all classes')
    correct = sum(label == prediction for label, prediction in zip(labels, predicted))
    accuracy = correct / len(labels)
    recalls = [sum(label == cls and prediction == cls
                   for label, prediction in zip(labels, predicted)) / labels.count(cls)
               for cls in range(num_classes)]
    expected_agreement = sum(labels.count(cls) * predicted.count(cls)
                             for cls in range(num_classes)) / len(labels) ** 2
    metrics = {
        'accuracy_pct': 100.0 * accuracy,
        'balanced_accuracy_pct': 100.0 * sum(recalls) / len(recalls),
        'kappa': (accuracy - expected_agreement) / (1.0 - expected_agreement),
    }
    for metric, value in metrics.items():
        if not math.isclose(float(record[metric]), value, abs_tol=1e-8):
            raise ValueError(f'{record_path}: {metric} does not match predictions')
    return record


def summarize(run_id, datasets, identity):
    rows = []
    for dataset in datasets:
        seed_means = {metric: [] for metric in ('accuracy_pct', 'balanced_accuracy_pct', 'kappa')}
        finished = 0
        for seed in SEEDS:
            records = [verify_result(run_id, dataset, subject, seed, identity)
                       for subject in range(DATASETS[dataset])]
            available = [record for record in records if record is not None]
            finished += len(available)
            for metric, values in seed_means.items():
                values.append(sum(float(record[metric]) for record in available)
                              / len(available) if len(available) == DATASETS[dataset] else None)
        complete = all(value is not None for value in seed_means['accuracy_pct'])
        rows.append({
            'dataset': dataset, 'model': 'LaBraM', 'protocol': 'within_subject_30_70',
            'subjects': DATASETS[dataset],
            'seeds': ','.join(map(str, SEEDS)), 'completed_cells': finished,
            'expected_cells': DATASETS[dataset] * len(SEEDS), 'complete': complete,
            **{f'{metric}_seed_{seed}': (value if value is not None else '')
               for metric, values in seed_means.items()
               for seed, value in zip(SEEDS, values)},
            **{metric: (sum(values) / len(values) if complete else '')
               for metric, values in seed_means.items()},
            'source_manifest_sha256': identity['source_manifest_sha256'][dataset],
            'pretrained_sha256': identity['pretrained_sha256'],
        })
    path = REPORTS_ROOT / 'fewshot/labram_broadband' / run_id / 'summary.csv'
    atomic_csv(path, rows)
    return path


def gpu_uuids(requested, min_free_mb):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    available = {}
    for line in output.splitlines():
        index, uuid, free = (part.strip() for part in line.split(','))
        if int(index) in requested and int(free) >= min_free_mb:
            available[int(index)] = uuid
    return available


def try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def run(args):
    datasets = tuple(args.datasets)
    if len(datasets) != len(set(datasets)) or 'BNCI2014001' not in datasets:
        raise ValueError('datasets must be unique and include BNCI2014001 for the smoke test')
    datasets = ('BNCI2014001',) + tuple(dataset for dataset in datasets if dataset != 'BNCI2014001')
    controller = require_external_output(RESULTS_ROOT / 'labram_broadband_fewshot' / args.run_id / 'controller')
    controller.mkdir(parents=True, exist_ok=True)
    manager_lock = try_lock(controller / 'manager.lock')
    if manager_lock is None:
        raise RuntimeError('another manager is already running for this campaign')
    identity = campaign_identity(datasets)
    manifest_path = controller / 'campaign.json'
    if manifest_path.exists():
        with manifest_path.open(encoding='utf-8') as stream:
            if json.load(stream) != identity:
                raise ValueError('campaign source, code, weight or dataset identity has changed')
    else:
        atomic_json(manifest_path, identity)

    stopping = False

    def stop_after_workers(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_after_workers)
    signal.signal(signal.SIGINT, stop_after_workers)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                 'BIGSMALLCOLLAB_CPU_THREADS', 'TORCH_NUM_THREADS', 'TORCH_NUM_INTEROP_THREADS'):
        env[name] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    running = {}
    failed = False
    smoke_run = args.run_id + '_smoke'
    stages = [('smoke', 'BNCI2014001', [(0, 666)])]
    stages.extend((dataset, dataset, [(subject, seed) for seed in SEEDS
                                     for subject in range(DATASETS[dataset])])
                  for dataset in datasets)
    try:
        for stage, dataset, cells in stages:
            while True:
                for gpu, job in list(running.items()):
                    result = job['process'].poll()
                    if result is None:
                        continue
                    job['log'].close()
                    job['cell_lock'].close()
                    job['gpu_lock'].close()
                    del running[gpu]
                    if result != 0:
                        failed = True
                        print(f'[failed] {job["cell"]} on GPU {gpu}: exit {result}, log {job["log_path"]}', flush=True)
                    elif verify_result(job['run_id'], dataset, job['subject'], job['seed'], identity,
                                       smoke=stage == 'smoke') is None:
                        failed = True
                        print(f'[failed] {job["cell"]}: completion record missing', flush=True)
                run_id = smoke_run if stage == 'smoke' else args.run_id
                missing = []
                for subject, seed in cells:
                    verified = verify_result(run_id, dataset, subject, seed, identity,
                                             smoke=stage == 'smoke')
                    if verified is None:
                        missing.append((subject, seed))
                active = {(job['subject'], job['seed']) for job in running.values()}
                missing = [cell for cell in missing if cell not in active]
                if stage != 'smoke':
                    summarize(args.run_id, datasets, identity)
                atomic_json(controller / 'status.json', {
                    'stage': stage, 'dataset': dataset, 'stage_total': len(cells),
                    'completed': len(cells) - len(missing) - len(running),
                    'remaining': len(missing),
                    'running': [{'cell': job['cell'], 'gpu': gpu,
                                 'pid': job['process'].pid, 'log': job['log_path']}
                                for gpu, job in running.items()],
                    'failed': failed, 'stopping': stopping,
                })
                if failed or stopping:
                    if not running:
                        return 1 if failed else 75
                elif not missing and not running:
                    print(f'[completed] stage {stage}', flush=True)
                    break
                elif missing:
                    available = gpu_uuids(args.gpus, args.min_free_mb)
                    for gpu in args.gpus:
                        if gpu in running or gpu not in available or not missing:
                            continue
                        gpu_lock = try_lock(controller / f'gpu{gpu}.lock')
                        if gpu_lock is None:
                            continue
                        subject, seed = missing.pop(0)
                        cell_lock = try_lock(controller / 'cells' / stage / dataset /
                                             f'subject{subject + 1:02d}_seed{seed}.lock')
                        if cell_lock is None:
                            gpu_lock.close()
                            continue
                        if campaign_identity(datasets) != identity:
                            cell_lock.close()
                            gpu_lock.close()
                            raise ValueError('campaign source, code or pretrained weights changed')
                        log_path = controller / 'logs' / stage / dataset / f'subject{subject + 1:02d}_seed{seed}.log'
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open('a', encoding='utf-8')
                        command = [args.python, '-B', '-m', 'experiments.finetune.run_labram_broadband_fewshot',
                                   '--dataset', dataset, '--subject', str(subject), '--seed', str(seed),
                                   '--run-id', run_id, '--gpu', '0']
                        if stage == 'smoke':
                            command.append('--smoke')
                        child_env = dict(env, CUDA_VISIBLE_DEVICES=available[gpu])
                        try:
                            child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_env, stdout=log,
                                                     stderr=subprocess.STDOUT,
                                                     pass_fds=(cell_lock.fileno(), gpu_lock.fileno()))
                        except BaseException:
                            log.close()
                            cell_lock.close()
                            gpu_lock.close()
                            raise
                        running[gpu] = {'process': child, 'subject': subject, 'seed': seed,
                                        'run_id': run_id,
                                        'cell': f'{dataset}/subject{subject + 1:02d}/seed{seed}',
                                        'cell_lock': cell_lock, 'gpu_lock': gpu_lock,
                                        'log': log, 'log_path': str(log_path)}
                        print(f'[start] {stage} {running[gpu]["cell"]} gpu={gpu} pid={child.pid}', flush=True)
                time.sleep(args.poll_seconds)
        report = summarize(args.run_id, datasets, identity)
        atomic_json(controller / 'status.json', {'stage': 'completed', 'report': str(report),
                                                  'failed': False})
        return 0
    finally:
        manager_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--gpus', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--min-free-mb', type=int, default=4000)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--python', default='/home/lixinli/anaconda3/envs/labram/bin/python')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('', '.', '..'):
        parser.error('run-id must be one path component')
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or args.min_free_mb < 0 or args.poll_seconds <= 0:
        parser.error('invalid GPU list, free-memory limit or poll interval')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
