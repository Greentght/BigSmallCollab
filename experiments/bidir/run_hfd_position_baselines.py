"""Independent CE and deep-supervision few-shot baselines for HFD position study."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['BIGSMALLCOLLAB_CPU_THREADS'] = '1'

import argparse
import fcntl
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score

import config
from data.loso_source import sha256_file
from experiments.bidir.run_dml_ds_cross_hfd_all import ROOT, make_models, source_and_split, baseline_dir
from experiments.bidir.run_dml_ds_cross_hfd_fewshot import atomic_json, atomic_save, state_digest, uid_hash
from experiments.distill.run_fewshot_structure_kd import _prepare_mirepnet_input
from experiments.storage import require_external_output

BASE = ROOT / 'configs/experiments/fewshot_dml_ds_pairs.yaml'
OUTPUT = Path('/data1/llx/BigSmallcollab/results/bidir/hfd_position_fewshot_v1')
PAIRS = ('cbramod_eegnet', 'mirepnet_adfcnn', 'neurogpt_ifnet')
DATASETS = ('BNCI2014004', 'BNCI2014001')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--pair', choices=PAIRS, required=True)
    parser.add_argument('--subject', type=int, choices=range(9), required=True)
    parser.add_argument('--seed', type=int, choices=(666, 667, 668), required=True)
    parser.add_argument('--condition', choices=('ce', 'ds'), required=True)
    parser.add_argument('--epochs', type=int, choices=(20, 100), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError('Expected one PyTorch inter-op thread')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    cfg = yaml.safe_load(BASE.read_text())
    big, small = args.pair.split('_')
    folder = require_external_output(OUTPUT / 'runs' / args.dataset / args.pair / args.condition /
                                     str(args.epochs) / f'subject_{args.subject}' / f'seed_{args.seed}')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, split = source_and_split(args.dataset, args.subject, args.seed, cfg)
        channels = split['x_cal_raw'].shape[1]
        classes = int(config.load_dataset_config(args.dataset)['num_classes'])
        adapters, models, heads, hfd, optimizers = make_models(
            args.dataset, args.pair, channels, classes, cfg, args.seed, device)
        hfd.close()
        try:
            train_raw, test_raw = split['x_cal_raw'], split['x_test_raw']
            ea_sha = None
            if big == 'mirepnet':
                big_train, big_test, ea_sha = _prepare_mirepnet_input(
                    args.dataset, train_raw, test_raw, adapters[0])
            elif big == 'neurogpt':
                # Fit per-channel population moments on calibration trials only.
                mean = train_raw.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
                std = train_raw.std(axis=(0, 2), keepdims=True, dtype=np.float64)
                big_train = torch.as_tensor((train_raw - mean) / np.maximum(std, 1e-6), dtype=torch.float32)
                big_test = torch.as_tensor((test_raw - mean) / np.maximum(std, 1e-6), dtype=torch.float32)
            else:
                big_train, big_test = train_raw, test_raw
            if big != 'neurogpt':
                big_train, big_test = adapters[0].preprocess(big_train), adapters[0].preprocess(big_test)
            inputs = [big_train, adapters[1].preprocess(train_raw)]
            test_inputs = [big_test, adapters[1].preprocess(test_raw)]
            labels = torch.as_tensor(split['y_cal'], dtype=torch.long)
            baseline = baseline_dir(args.dataset, args.pair, args.subject, args.seed, cfg)
            identity = {
                'dataset': args.dataset, 'pair': args.pair, 'subject': args.subject, 'seed': args.seed,
                'condition': args.condition, 'epochs': args.epochs, 'calibration_fraction': 0.3,
                'source_manifest_sha256': source.source_manifest_sha256,
                'train_uid_sha256': uid_hash(split['trial_uid_cal'].tolist()),
                'test_uid_sha256': uid_hash(split['trial_uid_test'].tolist()),
                'mirepnet_ea_sha256': ea_sha, 'pretrained_sha256': sha256_file(config.weight_path(big)),
                'initial_model_sha256': [state_digest(m) for m in models],
                'initial_head_sha256': [state_digest(h) for h in heads],
                'normalization': 'calibration_channel_zscore' if big == 'neurogpt' else 'existing_fixed_preprocessing',
                'baseline_result_sha256': sha256_file(baseline / 'evaluation.json'),
                'runner_sha256': sha256_file(Path(__file__)),
            }
            result_path = folder / 'evaluation.json'
            state_path = folder / 'training_state.pt'
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result['identity'] != identity or sha256_file(folder / 'predictions.npz') != result['predictions_sha256']:
                    raise ValueError('Existing completed result differs')
                return
            generator = torch.Generator().manual_seed(args.seed + 100003)
            start = 0
            if state_path.is_file():
                state = torch.load(state_path, map_location='cpu', weights_only=False)
                if state['identity'] != identity:
                    raise ValueError('Checkpoint identity differs')
                for objs, values in ((models, state['models']), (heads, state['heads']),
                                     (optimizers, state['optimizers'])):
                    for obj, value in zip(objs, values):
                        obj.load_state_dict(value)
                generator.set_state(state['generator'])
                random.setstate(state['python_rng'])
                np.random.set_state(state['numpy_rng'])
                torch.set_rng_state(state['torch_rng'])
                torch.cuda.set_rng_state(state['cuda_rng'], device)
                start = state['epoch']
            training = (cfg['teacher_training'][big], cfg['student_training'][small])
            for epoch in range(start, args.epochs):
                for model in models:
                    model.train()
                order = torch.randperm(len(labels), generator=generator)
                for rows in order.split(cfg['teacher_training'][big]['batch_size']):
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    logits = [adapter.forward(model, x[rows].to(device))[1]
                              for adapter, model, x in zip(adapters, models, inputs)]
                    if args.condition == 'ds':
                        outputs = [[z] + head(len(rows)) for z, head in zip(logits, heads)]
                        loss = sum(F.cross_entropy(side[0], labels[rows].to(device)) +
                                   cfg['alpha'] * torch.stack([
                                       F.cross_entropy(z, labels[rows].to(device)) for z in side[1:]]).mean()
                                   for side in outputs)
                    else:
                        loss = sum(F.cross_entropy(z, labels[rows].to(device)) for z in logits)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    loss.backward()
                    for optimizer in optimizers:
                        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                        optimizer.step()
                factor = 0.5 * (1 + np.cos(np.pi * (epoch + 1) / args.epochs))
                for optimizer, settings in zip(optimizers, training):
                    for group in optimizer.param_groups:
                        group['lr'] = float(settings['lr']) * factor
                if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
                    atomic_save(state_path, {'identity': identity, 'epoch': epoch + 1,
                        'models': [m.state_dict() for m in models], 'heads': [h.state_dict() for h in heads],
                        'optimizers': [o.state_dict() for o in optimizers], 'generator': generator.get_state(),
                        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)})
                print(f'{args.dataset} {args.pair} {args.condition} subject={args.subject} seed={args.seed} '
                      f'epoch={epoch + 1}/{args.epochs} loss={loss.item():.6f}', flush=True)
            for head in heads:
                head.close()
            predictions = {}
            batch = cfg['teacher_training'][big]['batch_size']
            for role, adapter, model, data in zip(('big', 'small'), adapters, models, test_inputs):
                model.eval()
                parts = []
                with torch.no_grad():
                    for x in data.split(batch):
                        parts.append(adapter.forward(model, x.to(device))[1].argmax(dim=1).cpu())
                predictions[role] = torch.cat(parts).numpy()
            y = np.asarray(split['y_test'])
            temporary = folder / f'.predictions.{os.getpid()}.npz'
            np.savez_compressed(temporary, uids=split['trial_uid_test'], y=y, **predictions)
            os.replace(temporary, folder / 'predictions.npz')
            result = {'identity': identity, 'predictions_sha256': sha256_file(folder / 'predictions.npz')}
            for role in predictions:
                result[role] = {'accuracy': float(np.mean(predictions[role] == y) * 100),
                                'balanced_accuracy': float(balanced_accuracy_score(y, predictions[role]) * 100)}
            atomic_json(result_path, result)
            state_path.unlink(missing_ok=True)
        finally:
            for head in heads:
                head.close()


if __name__ == '__main__':
    main()
