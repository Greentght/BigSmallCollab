"""Train DML-DS and bidirectional cross-operator HFD at specified positions."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import fcntl
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

import config
from collab.dcm import mutual_losses
from collab.hfd_position import PositionHFD
from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import (
    ROOT, baseline_dir, implementation_hashes, make_models, source_and_split, validate_baseline,
)
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, atomic_save, state_digest, uid_hash
from experiments.bidir.run_hfd_position_baselines import BASE, DATASETS, OUTPUT, PAIRS
from experiments.distill.run_fewshot_structure_kd import _prepare_mirepnet_input
from experiments.storage import require_external_output

POSITIONS = {'C2': [(False, False)], 'C3': [(False, True)],
             'C4': [(True, False)], 'C5': [(True, True)],
             'C6': [(False, False), (False, True), (True, False), (True, True)]}
HFD_WEIGHT = 0.1
WARMUP = 5
RAMP = 5


def fingerprint():
    hashes = implementation_hashes()
    for path in ('collab/hfd_position.py', 'experiments/bidir/run_hfd_position_fewshot.py',
                 'experiments/bidir/run_hfd_position_baselines.py'):
        hashes[path] = sha256_file(ROOT / path)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--pair', choices=PAIRS, required=True)
    parser.add_argument('--subject', type=int, choices=range(9), required=True)
    parser.add_argument('--seed', type=int, choices=(666, 667, 668), required=True)
    parser.add_argument('--condition', choices=('C0', *POSITIONS), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-id', default='default')
    parser.add_argument('--expected-fingerprint')
    args = parser.parse_args()
    if not args.smoke_id.isalnum():
        parser.error('Smoke id must be alphanumeric')
    if args.condition == 'C0' and not args.pair.startswith('neurogpt_'):
        parser.error('C0 is reused for CBraMod and MIRepNet')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError('Expected one PyTorch inter-op thread')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    cfg = yaml.safe_load(BASE.read_text())
    big, small = args.pair.split('_')
    run_hash = fingerprint()
    if args.expected_fingerprint and args.expected_fingerprint != run_hash:
        raise ValueError('Position experiment implementation changed')
    root = OUTPUT / ('smoke' if args.smoke else 'runs')
    if args.smoke:
        root /= args.smoke_id
    folder = require_external_output(root / args.dataset / args.pair / args.condition /
                                     f'subject_{args.subject}' / f'seed_{args.seed}')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, split = source_and_split(args.dataset, args.subject, args.seed, cfg)
        channels = split['x_cal_raw'].shape[1]
        classes = int(config.load_dataset_config(args.dataset)['num_classes'])
        adapters, models, heads, original_hfd, optimizers = make_models(
            args.dataset, args.pair, channels, classes, cfg, args.seed, device)
        original_hfd.close()
        for optimizer in optimizers:
            optimizer.param_groups.pop()
        paths = [PositionHFD(big, small, *models, channels, args.seed, *position).to(device)
                 for position in POSITIONS.get(args.condition, [])]
        for role, optimizer in enumerate(optimizers):
            params = [p for path in paths for p in
                      (path.big_to_small if role == 0 else path.small_to_big).parameters()]
            if params:
                optimizer.add_param_group({'params': params})
        try:
            raw_train, raw_test = split['x_cal_raw'], split['x_test_raw']
            ea_sha = None
            if big == 'mirepnet':
                big_train, big_test, ea_sha = _prepare_mirepnet_input(
                    args.dataset, raw_train, raw_test, adapters[0])
            elif big == 'neurogpt':
                mean = raw_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
                std = raw_train.std(axis=(0, 2), keepdims=True, dtype=np.float64)
                big_train = torch.as_tensor((raw_train - mean) / np.maximum(std, 1e-6), dtype=torch.float32)
                big_test = torch.as_tensor((raw_test - mean) / np.maximum(std, 1e-6), dtype=torch.float32)
            else:
                big_train, big_test = raw_train, raw_test
            if big != 'neurogpt':
                big_train, big_test = adapters[0].preprocess(big_train), adapters[0].preprocess(big_test)
                baseline = validate_baseline(args.dataset, args.pair, args.subject, args.seed, source, split, cfg, ea_sha)
            else:
                baseline_path = baseline_dir(args.dataset, args.pair, args.subject, args.seed, cfg)
                baseline = {'result_sha256': sha256_file(baseline_path / 'evaluation.json')}
            inputs = [big_train, adapters[1].preprocess(raw_train)]
            test_inputs = [big_test, adapters[1].preprocess(raw_test)]
            labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
            epochs = 2 if args.smoke else int(cfg['teacher_training'][big]['epochs'])
            identity = {'dataset': args.dataset, 'pair': args.pair, 'subject': args.subject, 'seed': args.seed,
                        'condition': args.condition, 'epochs': epochs, 'smoke': args.smoke,
                        'positions': POSITIONS.get(args.condition, []), 'hfd_weight': HFD_WEIGHT,
                        'warmup': WARMUP, 'ramp': RAMP, 'implementation_sha256': run_hash,
                        'source_manifest_sha256': source.source_manifest_sha256,
                        'train_uid_sha256': uid_hash(split['trial_uid_cal'].tolist()),
                        'test_uid_sha256': uid_hash(split['trial_uid_test'].tolist()),
                        'mirepnet_ea_sha256': ea_sha,
                        'normalization': 'calibration_channel_zscore' if big == 'neurogpt' else 'existing_fixed_preprocessing',
                        'pretrained_sha256': sha256_file(config.weight_path(big)),
                        'initial_model_sha256': [state_digest(model) for model in models],
                        'initial_head_sha256': [state_digest(head) for head in heads],
                        'old_baseline_sha256': baseline['result_sha256']}
            result_path, state_path = folder / 'evaluation.json', folder / 'training_state.pt'
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result['identity'] != identity or result['predictions_sha256'] != sha256_file(folder / 'predictions.npz'):
                    raise ValueError('Completed result identity differs')
                return
            generator = torch.Generator().manual_seed(args.seed + 100003)
            start, history = 0, []
            if state_path.is_file():
                state = torch.load(state_path, map_location='cpu', weights_only=False)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                for objects, saved in ((models, state['models']), (heads, state['heads']),
                                       (paths, state['paths']), (optimizers, state['optimizers'])):
                    for obj, value in zip(objects, saved):
                        obj.load_state_dict(value)
                generator.set_state(state['generator'])
                random.setstate(state['python_rng'])
                np.random.set_state(state['numpy_rng'])
                torch.set_rng_state(state['torch_rng'])
                torch.cuda.set_rng_state(state['cuda_rng'], device)
                start, history = state['epoch'], state['history']
            train_settings = (cfg['teacher_training'][big], cfg['student_training'][small])
            batch = cfg['teacher_training'][big]['batch_size']
            for epoch in range(start, epochs):
                for model in models:
                    model.train()
                order = torch.randperm(len(labels), generator=generator)
                ramp = 1.0 if args.smoke else min(1.0, max(0.0, (epoch + 1 - WARMUP) / RAMP))
                sums, count = np.zeros(3), 0
                for rows in order.split(batch):
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    for path in paths:
                        path.clear()
                    logits = [adapter.forward(model, x[rows].to(device))[1]
                              for adapter, model, x in zip(adapters, models, inputs)]
                    outputs = [[z] + head(len(rows)) for z, head in zip(logits, heads)]
                    target = labels[rows].to(device)
                    ds = sum(mutual_losses(outputs, target, 'dml_ds', cfg['temperature'],
                                           cfg['alpha'], cfg['lambda']))
                    losses = []
                    if paths and ramp:
                        snapshots = [dict(path.features) for path in paths]
                        for path, snapshot in zip(paths, snapshots):
                            path.features = snapshot
                            losses.append(path.losses())
                    hfd_big = sum(x[0] for x in losses) / len(losses) if losses else ds.new_zeros(())
                    hfd_small = sum(x[1] for x in losses) / len(losses) if losses else ds.new_zeros(())
                    loss = ds + ramp * HFD_WEIGHT * (hfd_big + hfd_small)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    loss.backward()
                    for optimizer in optimizers:
                        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                        optimizer.step()
                    sums += [ds.item(), hfd_big.item(), hfd_small.item()]
                    count += 1
                    for path in paths:
                        path.clear()
                    if args.smoke and count == 2:
                        break
                factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / epochs))
                for optimizer, settings in zip(optimizers, train_settings):
                    for group in optimizer.param_groups:
                        group['lr'] = float(settings['lr']) * factor
                history.append({'epoch': epoch + 1, 'ramp': ramp, 'losses': (sums / count).tolist()})
                if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
                    atomic_save(state_path, {'identity': identity, 'epoch': epoch + 1,
                        'models': [m.state_dict() for m in models], 'heads': [h.state_dict() for h in heads],
                        'paths': [p.state_dict() for p in paths], 'optimizers': [o.state_dict() for o in optimizers],
                        'generator': generator.get_state(), 'python_rng': random.getstate(),
                        'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state(device), 'history': history})
                print(f'{args.dataset} {args.pair} {args.condition} subject={args.subject} seed={args.seed} '
                      f'epoch={epoch + 1}/{epochs} losses={(sums / count).tolist()}', flush=True)
            for head in heads:
                head.close()
            for path in paths:
                path.close()
            raw_evaluation = inputs if args.smoke else test_inputs
            truth = np.asarray(split['y_cal'] if args.smoke else split['y_test'])
            uids = split['trial_uid_cal'] if args.smoke else split['trial_uid_test']
            predictions = {}
            for role, adapter, model, data in zip(('big', 'small'), adapters, models, raw_evaluation):
                model.eval()
                parts = []
                with torch.no_grad():
                    for x in data.split(batch):
                        parts.append(adapter.forward(model, x.to(device))[1].argmax(dim=1).cpu())
                predictions[role] = torch.cat(parts).numpy()
            temp = folder / f'.predictions.{os.getpid()}.npz'
            np.savez_compressed(temp, uids=uids, y=truth, **predictions)
            os.replace(temp, folder / 'predictions.npz')
            result = {'identity': identity, 'evaluation_split': 'calibration' if args.smoke else 'test',
                      'predictions_sha256': sha256_file(folder / 'predictions.npz')}
            for role in predictions:
                result[role] = {'accuracy': float(np.mean(predictions[role] == truth) * 100),
                                'balanced_accuracy': float(balanced_accuracy_score(truth, predictions[role]) * 100)}
            atomic_json(folder / 'training.json', {'identity': identity, 'history': history})
            atomic_json(result_path, result)
            state_path.unlink(missing_ok=True)
        finally:
            for head in heads:
                head.close()
            for path in paths:
                path.close()


if __name__ == '__main__':
    main()
