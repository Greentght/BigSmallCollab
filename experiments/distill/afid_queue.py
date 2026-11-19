"""Submit AFID few-shot cells to the shared GPU queue in verified stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.distill.queue import DEFAULT_ROOT, QueueStore, atomic_json, enqueue_snapshot, file_lock, read_json
from experiments.storage import PROJECT_ROOT, RESULTS_ROOT

TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
METHODS = ('none', 'kd', 'dml', 'afid_self_gate', 'afid')
SEEDS = (666, 667, 668)
CODE_FILES = (
    'experiments/distill/run_afid_fewshot.py', 'collab/afid_fewshot.py',
    'experiments/distill/afid_queue.py', 'experiments/distill/summarize_afid_fewshot.py',
    'experiments/distill/run_fewshot_structure_kd.py',
    'data/loso_source.py', 'data/split.py', 'data/preproc.py', 'data/channels.py',
    'collab/seed.py', 'models/base.py', 'models/__init__.py', 'config.py',
    'models/mirepnet/adapter.py', 'models/mirepnet/mlm.py',
    'models/cbramod/adapter.py', 'models/cbramod/cbramod.py',
    'models/cbramod/criss_cross_transformer.py', 'models/ifnet/adapter.py',
    'models/ifnet/ifnet.py', 'models/eegnet/adapter.py', 'models/eegnet/residual_eegnet.py',
    'models/adfcnn/adapter.py', 'models/adfcnn/adfcnn.py',
    'configs/models/mirepnet.yaml', 'configs/models/cbramod.yaml', 'configs/models/ifnet.yaml',
    'configs/models/eegnet.yaml', 'configs/models/adfcnn.yaml', 'configs/datasets/BNCI2014001.yaml',
)
INPUT_FILES = (Path('/data1/llx/BNCI2014001/wide_fre_band/manifest.json'),
               Path('/data1/llx/pre_weight/mirepnet.pth'),
               Path('/data1/llx/pre_weight/cbramod.pth'))


def hashes():
    files = [(name, PROJECT_ROOT / name) for name in CODE_FILES]
    files.extend((str(path), path) for path in INPUT_FILES)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files}


def cells(stage):
    if stage == 'smoke':
        return [(teacher, 'ifnet', 0, 666, 10) for teacher in TEACHERS]
    shots = (10,) if stage == 'shot10' else (5, 20)
    return [(teacher, student, subject, seed, shot)
            for shot in shots for teacher in TEACHERS for student in STUDENTS
            for seed in SEEDS for subject in range(9)]


def enqueue_stage(root, campaign, stage, code_hashes):
    if stage not in ('smoke', 'shot10', 'other_shots'):
        raise ValueError(f'unknown stage: {stage}')
    if hashes() != code_hashes:
        raise ValueError('AFID code changed since campaign submission')
    for teacher, student, subject, seed, shot in cells(stage):
        snapshot = dict(kind='afid_fewshot', campaign=campaign, stage=stage,
                        teacher=teacher, student=student, subject=subject,
                        seed=seed, shot=shot, code_hashes=code_hashes,
                        args=dict(teacher=teacher, student=student, subject=subject,
                                  seed=seed, shot=shot))
        enqueue_snapshot(snapshot, root=root,
                         label=f'AFID {stage} {teacher}/{student} S{subject+1:02d} seed{seed} K{shot}')


def enqueue_next_stage(root, completed):
    stage = completed['stage']
    if stage not in ('smoke', 'shot10', 'other_shots'):
        return
    store = QueueStore(root)
    marker = Path(root) / f"afid_{completed['campaign']}_{stage}.json"
    with file_lock(marker.with_suffix('.lock')):
        if marker.exists():
            return
        with store.lock():
            matching = {}
            for state in store.states():
                spec = read_json(store.job_dir(state['id']) / 'spec.json', {})
                snap = spec.get('snapshot', {})
                if (snap.get('kind') == 'afid_fewshot'
                        and snap.get('campaign') == completed['campaign']
                        and snap.get('stage') == stage
                        and snap.get('code_hashes') == completed['code_hashes']):
                    key = (snap['teacher'], snap['student'], snap['subject'], snap['seed'], snap['shot'])
                    matching[key] = state['status']
            if set(matching) != set(cells(stage)) or any(v != 'completed' for v in matching.values()):
                return
        if hashes() != completed['code_hashes']:
            raise ValueError('AFID code or input identity changed before stage transition')
        if stage != 'smoke':
            env = os.environ.copy()
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                env[name] = '1'
            subprocess.run([sys.executable, '-B', '-m', 'experiments.distill.summarize_afid_fewshot',
                            '--run-id', completed['campaign']], cwd=PROJECT_ROOT, env=env, check=True)
        if stage != 'other_shots':
            enqueue_stage(root, completed['campaign'],
                          'shot10' if stage == 'smoke' else 'other_shots', completed['code_hashes'])
        atomic_json(marker, {'campaign': completed['campaign'], 'stage': stage, 'status': 'completed'})


def worker(snapshot_path, progress, stop_file):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))['snapshot']
    if hashes() != snapshot['code_hashes']:
        raise ValueError('AFID code changed after queue submission')
    if Path(stop_file).exists():
        atomic_json(progress, {'status': 'paused'})
        return 75
    run_id = snapshot['campaign'] + ('_smoke' if snapshot['stage'] == 'smoke' else '')
    command = [sys.executable, '-B', '-m', 'experiments.distill.run_afid_fewshot',
               '--run-id', run_id, '--teacher', snapshot['teacher'],
               '--student', snapshot['student'], '--subject', str(snapshot['subject']),
               '--seed', str(snapshot['seed']), '--shots', str(snapshot['shot']), '--gpu', '0']
    if snapshot['stage'] == 'smoke':
        command += ['--epochs', '6', '--teacher-epochs', '1', '--smoke']
    atomic_json(progress, {'status': 'running', 'completed': 0, 'total': len(METHODS)})
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        return result.returncode
    root = (RESULTS_ROOT / 'fewshot/afid' / run_id / snapshot['teacher'] / snapshot['student'] /
            f"shot{snapshot['shot']}" / f"subject{snapshot['subject'] + 1:02d}" /
            f"seed{snapshot['seed']}")
    for method in METHODS:
        item = read_json(root / method / 'completed.json')
        if item is None or item.get('method') != method or int(item.get('n_test', 0)) != 101:
            raise RuntimeError(f'incomplete or invalid result {root / method}')
        if not Path(item['predictions_path']).is_file():
            raise RuntimeError(f'missing prediction file for {method}')
    atomic_json(progress, {'status': 'completed', 'completed': len(METHODS), 'total': len(METHODS)})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    submit = sub.add_parser('submit')
    submit.add_argument('--campaign', required=True)
    submit.add_argument('--root', default=str(DEFAULT_ROOT))
    advance = sub.add_parser('advance', help='retry a completed stage transition')
    advance.add_argument('--campaign', required=True)
    advance.add_argument('--stage', choices=('smoke', 'shot10', 'other_shots'), required=True)
    advance.add_argument('--root', default=str(DEFAULT_ROOT))
    run = sub.add_parser('worker')
    run.add_argument('--snapshot', required=True)
    run.add_argument('--progress', required=True)
    run.add_argument('--stop-file', required=True)
    args = parser.parse_args()
    if args.command == 'submit':
        if not args.campaign or Path(args.campaign).name != args.campaign:
            parser.error('campaign must be a single path component')
        enqueue_stage(Path(args.root), args.campaign, 'smoke', hashes())
        return 0
    if args.command == 'advance':
        store = QueueStore(Path(args.root))
        with store.lock():
            snapshots = [read_json(store.job_dir(s['id']) / 'spec.json', {}).get('snapshot', {})
                         for s in store.states()]
        matches = [s for s in snapshots if s.get('kind') == 'afid_fewshot'
                   and s.get('campaign') == args.campaign and s.get('stage') == args.stage]
        if not matches:
            raise ValueError('no matching AFID stage in queue')
        enqueue_next_stage(Path(args.root), matches[0])
        return 0
    return worker(args.snapshot, args.progress, args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
