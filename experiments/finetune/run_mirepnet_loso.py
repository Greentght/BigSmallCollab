#!/usr/bin/env python
"""Standalone MIRepNet 001/001-4 fine-tuning; every invocation starts a new run.

Defaults come from the model YAML. Only --resume permits reading completed
folds or interrupted training from a named run. Canonical EEG input caches are
validated and shared; every new fold starts from the official pretrained model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (REPORTS_ROOT, RESULTS_ROOT,
                                 require_external_output, require_report_output)

RUNS_ROOT = RESULTS_ROOT / 'finetune/mirepnet'
ENTRYPOINT = 'experiments/finetune/run_mirepnet_loso.py'
TRAINING_KEYS = ('epochs', 'lr', 'batch_size', 'weight_decay', 'optimizer')
DATASETS = {'001': 'BNCI2014001', '001-4': 'BNCI2014001-4',
            'BNCI2014001': 'BNCI2014001', 'BNCI2014001-4': 'BNCI2014001-4'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='001')
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='default: 666 667 668; executed sequentially on one GPU')
    parser.add_argument('--gpu', type=int, required=True,
                        help='allocated CUDA device index; GPU 0 is prohibited')
    parser.add_argument('--run-name', help='new experiment name; defaults to a timestamp plus random suffix')
    parser.add_argument('--resume', action='store_true', help='explicitly continue --run-name')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--optimizer', choices=('adam', 'adamw'))
    args = parser.parse_args(argv)
    if args.gpu <= 0:
        parser.error('--gpu must be a positive allocated GPU index')
    if args.resume and not args.run_name:
        parser.error('--resume requires --run-name')
    if args.seeds is not None and (min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error('--seeds must be distinct non-negative integers')
    for key in ('epochs', 'batch_size'):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    for key in ('lr', 'weight_decay'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0)):
            parser.error(f'invalid --{key.replace("_", "-")}')
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', args.run_name):
        parser.error('--run-name must use letters, digits, underscores, dots or hyphens (max 100 characters)')
    return args


def claim_run_directory(dataset, name, resume):
    name = name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:8]
    root = require_external_output(RUNS_ROOT / dataset / name)
    if resume:
        if not (root / 'run_manifest.json').is_file():
            raise ValueError(f'no resumable run manifest: {root}')
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError as exc:
            raise ValueError(f'run already exists: {root}; choose a new name, or explicitly use --resume') from exc
    return root


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_inputs(dataset, runner):
    from experiments.finetune import prepare_bnci14001_wideband_inputs as inputs
    from experiments.finetune import prepare_loso_alignment_inputs as validation
    print('[validate-input] checking source, preprocessing recipe and EEG file hashes', flush=True)
    source, _, source_manifest = inputs.validate_source()
    del source
    folder = runner.input_dir(runner.WIDEBAND_PROFILE, dataset, 'mirepnet', runner.WIDEBAND_VARIANT)
    manifest = runner.read_json(folder / 'manifest.json')
    required = {'X.npy', 'y.npy', 'subjects.npy', 'trials.csv'}
    if not required.issubset(manifest.get('files', {})):
        raise ValueError(f'incomplete input hash manifest: {folder}')
    if not validation.valid_existing_input(folder, runner.WIDEBAND_PROFILE, dataset,
            'mirepnet', runner.WIDEBAND_VARIANT, inputs.recipe('mirepnet', dataset), source_manifest):
        raise ValueError(f'input cache does not match source, preprocessing recipe or file hashes: {folder}')
    return {'input_manifest': str(folder / 'manifest.json'),
            'input_manifest_sha256': runner.sha256_file(folder / 'manifest.json'),
            'pretrained_checkpoint': str(runner.MIREPNET_WEIGHT_PATH),
            'pretrained_checkpoint_sha256': runner.sha256_file(runner.MIREPNET_WEIGHT_PATH)}


def prepare_plan(args, root, runner):
    import yaml
    dataset = DATASETS[args.dataset]
    snapshot_path = root / 'config_snapshot.json'
    manifest_path = root / 'run_manifest.json'
    supplied = {key: getattr(args, key) for key in TRAINING_KEYS if getattr(args, key) is not None}
    if args.resume:
        plan = json.loads(manifest_path.read_text())
        if (plan['dataset'] != dataset or
                runner.sha256_file(snapshot_path) != plan['config_snapshot_sha256']):
            raise ValueError('saved dataset or configuration snapshot changed')
        if args.seeds is not None and args.seeds != plan['seeds']:
            raise ValueError('--resume seeds differ from the saved experiment')
    else:
        snapshot = {'yaml_configs': {
            'models/mirepnet.yaml': yaml.safe_load((ROOT / 'configs/models/mirepnet.yaml').read_text()),
            f'datasets/{dataset}.yaml': yaml.safe_load((ROOT / f'configs/datasets/{dataset}.yaml').read_text()),
        }, 'protocol_spec': yaml.safe_load(runner.SPEC.read_text())}
        write_json(snapshot_path, snapshot)
        plan = {'entrypoint': ENTRYPOINT,
                'dataset': dataset, 'model': 'mirepnet', 'run_name': root.name,
                'seeds': args.seeds if args.seeds is not None else [666, 667, 668],
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'training_overrides': supplied,
                'config_snapshot_sha256': runner.sha256_file(snapshot_path),
                'resources': {'cpu_threads': 1, 'torch_interop_threads': 1, 'dataloader_workers': 0}}
        # Preserve the frozen plan even if cache validation is interrupted.
        write_json(manifest_path, plan)
    os.environ['BIGSMALL_RUN_SNAPSHOT'] = str(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text())
    spec = snapshot['protocol_spec']
    effective = runner.profile_config(runner.WIDEBAND_PROFILE, 'mirepnet', dataset, spec)
    effective.update(plan['training_overrides'])
    if plan['training_overrides']:
        effective['parameter_overrides'] = dict(plan['training_overrides'])
    if args.resume:
        for key, value in supplied.items():
            if value != effective[key]:
                raise ValueError(f'--resume cannot change {key}; start a new experiment')
    identity = validate_inputs(dataset, runner)
    if (('resolved_training_config' in plan and effective != plan['resolved_training_config']) or
            ('input_identity' in plan and identity != plan['input_identity'])):
        raise ValueError('saved training settings, EEG input or pretrained checkpoint changed')
    plan.update(resolved_training_config=effective, input_identity=identity)
    write_json(manifest_path, plan)
    return plan, spec


def export_summary(root, plan):
    seeds = []
    for seed in plan['seeds']:
        summary = json.loads((root / f'seed_{seed}/seed_summary.json').read_text())
        if summary['status'] != 'complete' or summary['folds_complete'] != 9 or summary['folds_expected'] != 9:
            raise ValueError(f'seed {seed}: expected all nine completed LOSO subjects')
        seeds.append({'seed': seed, 'accuracy_pct': summary['subject_equal_mean']['accuracy'] * 100})
    report = {'dataset': plan['dataset'], 'model': 'MIRepNet', 'run_name': root.name,
              'metric': 'Accuracy (%)', 'aggregation': 'equal-weight subject mean per seed, then equal-weight seed mean',
              'seed_results': seeds, 'accuracy_pct': sum(x['accuracy_pct'] for x in seeds) / len(seeds),
              'resolved_training_config': plan['resolved_training_config'], 'run_manifest': str(root / 'run_manifest.json')}
    output = require_report_output(REPORTS_ROOT / f'mirepnet_{plan["dataset"]}_{root.name}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f'[complete] Accuracy={report["accuracy_pct"]:.2f}% seeds={plan["seeds"]} report={output}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    # Apply numerical backend limits before importing NumPy, PyTorch or runners.
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                     'TORCH_NUM_THREADS', 'TORCH_NUM_INTEROP_THREADS'):
        os.environ[variable] = '1'
    import torch
    from experiments.finetune import run_loso_config_alignment as runner
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise ValueError(f'allocated CUDA device {args.gpu} is unavailable')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    root = claim_run_directory(DATASETS[args.dataset], args.run_name, args.resume)
    with (root / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'this experiment is already running: {root}') from exc
        try:
            plan, spec = prepare_plan(args, root, runner)
            print(f'[{"resume" if args.resume else "new-training"}] output={root}', flush=True)
            print(f'[config] {json.dumps(plan["resolved_training_config"], sort_keys=True)}', flush=True)
            for seed in plan['seeds']:
                worker = argparse.Namespace(profile=runner.WIDEBAND_PROFILE, dataset=plan['dataset'],
                                            model='mirepnet', seed=seed, epochs_override=None, preflight_steps=None)
                write_json(root / 'run_status.json', {'status': 'running', 'seed': seed, 'gpu': args.gpu})
                runner.run_worker(worker, spec, device, seed_output=root / f'seed_{seed}',
                                  training_overrides=plan['training_overrides'])
            export_summary(root, plan)
            write_json(root / 'run_status.json', {'status': 'complete', 'seeds': plan['seeds']})
        except BaseException as exc:
            write_json(root / 'run_status.json', {'status': 'interrupted', 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()
