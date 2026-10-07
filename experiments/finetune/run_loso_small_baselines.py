"""Run scratch-trained IFNet, EEGNet, and ADFCNN on the frozen five-dataset LOSO.

This runner reuses the trial and fold manifest from ``run_loso_five_datasets``
but writes to a separate recipe directory. Each fold/seed starts with a fresh
random initialization and evaluates the final epoch once, with no validation
selection.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from models import get_adapter
from experiments.finetune import run_loso_five_datasets as protocol
from experiments.storage import external_path, require_external_output, resolve_local_file


MODELS = ('ifnet', 'eegnet', 'adfcnn')
DATASETS = protocol.DATASET_NAMES
RECIPE = 'scratch_supervised_loso_v1'
RECIPE_DIRNAME = 'scratch_supervised_loso_v1'


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='zero-based held-out subject IDs; default is all')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _config_for(model_name, dataset, ds_spec):
    cfg = config.load_model_config(model_name, dataset, 'loso')
    cfg.update(dataset_name=dataset,
               in_channels=int(ds_spec['native_channels']),
               samples=1000,
               sample_rate=int(ds_spec['canonical_fs_hz']))
    return cfg


def _profile(model_name, dataset, cfg):
    return {
        'name': RECIPE_DIRNAME,
        'parameter_source': f'configs/models/{model_name}.yaml::finetune.{dataset}.loso',
        'initialization': 'random_from_scratch_each_fold_and_seed',
        'optimizer': cfg.get('optimizer', cfg.get('optimizer_type', 'adamw')),
        'lr_schedule': 'cosine_annealing_per_epoch',
        'warmup_epochs': 0,
        'validation_policy': 'none; evaluate fixed final epoch',
        'resolved_config': {k: v for k, v in cfg.items()
                            if k not in ('dataset_name',)},
    }


def _optimizer(model, cfg):
    name = str(cfg.get('optimizer', cfg.get('optimizer_type', 'adamw'))).lower()
    lr = float(cfg.get('lr', 1e-3))
    wd = float(cfg.get('weight_decay', 1e-4))
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=float(cfg.get('momentum', 0.9)))
    raise ValueError(f'Unsupported optimizer {name!r}')


def _fit(adapter, model, x_train, y_train, cfg, seed):
    x_tensor = adapter.preprocess(x_train)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                        batch_size=int(cfg['batch_size']), shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg['epochs']))
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(int(cfg['epochs'])):
        model.train()
        loss_sum = 0.0
        seen = 0
        lr_used = float(optimizer.param_groups[0]['lr'])
        for xb, yb in loader:
            xb, yb = xb.to(adapter.device), yb.to(adapter.device)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward(model, xb)[1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(yb)
            seen += len(yb)
        scheduler.step()
        entry = {'epoch': epoch + 1,
                 'train_loss': loss_sum / max(seen, 1),
                 'n_train': seen,
                 'lr_used': lr_used,
                 'lr_next': float(optimizer.param_groups[0]['lr'])}
        history.append(entry)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(cfg['epochs']):
            print(f'  epoch={epoch + 1:03d}/{cfg["epochs"]} '
                  f'loss={entry["train_loss"]:.5f} lr={lr_used:.3g}', flush=True)
    return model, history


def _preflight(model_name, datasets, spec, snapshot, device):
    for dataset in datasets:
        ds_spec = spec['datasets'][dataset]
        protocol._verify_source_files(dataset, spec, snapshot)
        x, y, *_ = protocol._load_trials(dataset, spec)
        cfg = _config_for(model_name, dataset, ds_spec)
        batch = min(int(cfg['batch_size']), len(y))
        protocol._set_seed(1201)
        adapter = get_adapter(model_name, device=device, **cfg)
        xb = adapter.preprocess(x[:batch])
        model = adapter.build(len(ds_spec['classes']))
        model.train()
        yb = torch.as_tensor(y[:batch], dtype=torch.long, device=device)
        _, logits = adapter.forward(model, xb.to(device))
        if tuple(logits.shape) != (batch, len(ds_spec['classes'])):
            raise RuntimeError(f'{model_name}/{dataset}: invalid logits {tuple(logits.shape)}')
        loss = nn.CrossEntropyLoss()(logits, yb)
        loss.backward()
        _optimizer(model, cfg).step()
        print(f'[preflight-ok] {model_name} {dataset} '
              f'input={tuple(xb.shape)} logits={tuple(logits.shape)} '
              f'batch={batch}', flush=True)
        del model, adapter, xb, x, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(args, spec, snapshot, device):
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds must be non-empty and unique')
    for dataset in args.datasets:
        ds_spec = spec['datasets'][dataset]
        n_subjects = int(ds_spec['subjects'])
        folds = list(range(n_subjects)) if args.folds is None else args.folds
        if any(f < 0 or f >= n_subjects for f in folds):
            raise ValueError(f'{dataset}: fold IDs must be in [0,{n_subjects - 1}]')
        source_hashes = protocol._verify_source_files(dataset, spec, snapshot)
        x, y, subject_ids, source_uids, local_uids, _records, _meta, _labels_raw = \
            protocol._load_trials(dataset, spec)
        cfg = _config_for(args.model, dataset, ds_spec)
        profile = _profile(args.model, dataset, cfg)
        output_dir = protocol.RESULTS_ROOT / dataset / args.model / RECIPE_DIRNAME
        output_dir.mkdir(parents=True, exist_ok=True)
        source_payload = {
            'dataset_source': ds_spec['source_dataset'],
            'files': source_hashes,
            'trial_manifest_sha256': protocol._sha256(protocol.TRIALS_PATH),
            'spec_sha256': snapshot['spec_sha256'],
        }
        split_manifest = {
            'protocol': protocol.PROTOCOL,
            'dataset': dataset,
            'model': args.model,
            'recipe': RECIPE_DIRNAME,
            'training_profile': profile,
            'source_dataset': ds_spec['source_dataset'],
            'source_shape': ds_spec['source_shape'],
            'source_fs_hz': ds_spec['source_fs_hz'],
            'selected_session': ds_spec['selected_session'],
            'class_mapping': ds_spec['classes'],
            'num_classes': len(ds_spec['classes']),
            'num_subjects': n_subjects,
            'total_trials': len(y),
            'subject_trial_counts': ds_spec['per_subject_trials'],
            'window': {'canonical_samples': 1000,
                       'canonical_fs_hz': ds_spec['canonical_fs_hz'],
                       'window_label': ds_spec.get('window_label',
                                                   '4s_from_acquired_signal')},
            'test_policy': 'one full subject held out; fixed final epoch once',
            'validation_policy': 'none_in_fixed_recipe_baseline',
            'source_files': source_payload,
            'initialization': 'random_from_scratch_each_fold_and_seed',
            'seeds': list(args.seeds),
        }
        split_path = output_dir / 'split_manifest.json'
        if split_path.exists():
            old = json.loads(resolve_local_file(split_path).read_text())
            old_seeds = old.pop('seeds', [])
            expected = dict(split_manifest)
            expected.pop('seeds')
            if old != expected:
                raise RuntimeError(f'output directory belongs to a different run: {split_path}')
            split_manifest['seeds'] = sorted(set(old_seeds) | set(args.seeds))
        split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + '\n')

        print(f'[dataset] {args.model} {dataset} trials={len(y)} '
              f'subjects={n_subjects} device={device} cfg={cfg}', flush=True)
        for fold in folds:
            train_mask = subject_ids != fold
            test_mask = subject_ids == fold
            x_train, y_train = x[train_mask], y[train_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            uid_train, uid_test = source_uids[train_mask], source_uids[test_mask]
            local_test = local_uids[test_mask]
            for seed in args.seeds:
                cell, result_path, checkpoint_path, manifest_path, history_path = \
                    protocol._cell_paths(output_dir, fold, seed)
                cell.mkdir(parents=True, exist_ok=True)
                cached = protocol._cell_is_complete(
                    result_path, checkpoint_path, manifest_path, history_path,
                    cfg, source_payload, None, args.force)
                if cached is not None:
                    print(f'[skip] {args.model} {dataset} S{fold + 1} '
                          f'seed={seed}', flush=True)
                    continue

                protocol._set_seed(seed)
                adapter = get_adapter(args.model, device=device, **cfg)
                model = adapter.build(len(ds_spec['classes']))
                started = time.time()
                model, history = _fit(adapter, model, x_train, y_train, cfg, seed)
                feats, logits = adapter.infer(model, x_test)
                metrics, probs, pred, cm = protocol._metrics(y_test, logits)
                metrics.update(model=args.model, dataset=dataset,
                               test_subject=fold + 1, seed=seed,
                               n_train=len(y_train), n_test=len(y_test),
                               elapsed_sec=round(time.time() - started, 2),
                               artifact_origin='trained_from_scratch')
                result = {
                    'y': y_test, 'pred': pred, 'probs': probs,
                    'logits': logits, 'feats': feats,
                    'sample_uid': uid_test,
                    'local_sample_uid': local_test,
                    'confusion_matrix': cm,
                    'metrics_json': np.asarray(json.dumps(
                        metrics, sort_keys=True, allow_nan=True)),
                }
                manifest = {
                    'protocol': protocol.PROTOCOL,
                    'dataset': dataset,
                    'model': args.model,
                    'recipe': RECIPE_DIRNAME,
                    'training_profile': profile,
                    'artifact_origin': 'trained_from_scratch',
                    'dataset_source': ds_spec['source_dataset'],
                    'source_files': source_payload,
                    'source_trial_selection': {
                        'session': ds_spec['selected_session'],
                        'run': ds_spec.get('selected_run'),
                        'window': ds_spec['window'],
                    },
                    'initialization': 'random_from_scratch',
                    'pretrained_checkpoint': None,
                    'pretrained_sha256': None,
                    'model_config': cfg,
                    'environment_snapshot': protocol._environment_snapshot(device),
                    'num_classes': len(ds_spec['classes']),
                    'label_values': ds_spec['classes'],
                    'test_subject': fold + 1,
                    'train_subjects': sorted(np.unique(subject_ids[train_mask])
                                             .astype(int).tolist()),
                    'seed': seed,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'train_sample_uids': uid_train.tolist(),
                    'test_sample_uids': uid_test.tolist(),
                    'local_test_sample_uids': local_test.tolist(),
                    'window': split_manifest['window'],
                    'selection_policy': 'fixed_final_epoch_no_validation',
                    'preprocessing': ('4-16 and 16-40 Hz filter bank; concatenated '
                                      'along channels' if args.model == 'ifnet'
                                      and cfg.get('use_filter_bank', False)
                                      else 'canonical raw 4s; model-native input'),
                    'test_metrics': metrics,
                }
                protocol._save_cell(result_path, checkpoint_path, manifest_path,
                                    history_path, model, result, manifest, history)
                print(f'[done] {args.model} {dataset} S{fold + 1} '
                      f'seed={seed} acc={metrics["accuracy"]:.4f} '
                      f'bacc={metrics["balanced_accuracy"]:.4f} '
                      f'kappa={metrics["kappa"]:.4f}', flush=True)
                del model, adapter, feats, logits
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del x_train, y_train, x_test, y_test, uid_train, uid_test, local_test
        protocol._summary(output_dir, dataset, args.model, RECIPE_DIRNAME,
                          split_manifest['seeds'], n_subjects, profile)
        print(f'[summary] {output_dir / "summary.csv"}', flush=True)
        del x, y, subject_ids, source_uids, local_uids
        gc.collect()


def main():
    args = _args()
    spec, snapshot = protocol._load_spec()
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('--gpu specified but CUDA is unavailable')
    device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    if args.preflight_only:
        _preflight(args.model, args.datasets, spec, snapshot, device)
    else:
        _run(args, spec, snapshot, device)


if __name__ == '__main__':
    main()
