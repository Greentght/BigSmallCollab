#!/usr/bin/env python
"""Train one model/dataset/seed stream on the refreshed 004/5001 inputs."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from models import get_adapter
from experiments.finetune import run_loso_five_datasets as protocol
from experiments.storage import require_external_output, resolve_local_file

SPEC_PATH = ROOT / 'configs/reproductions/loso_source_refresh_004_5001_v1.yaml'
INPUT_ROOT = Path('/data1/llx/BigSmallcollab/cache/reproductions/loso_source_refresh_004_5001_v1/model_inputs')
RESULT_ROOT = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_source_refresh_004_5001_v1')
MODELS = ('mirepnet', 'cbramod', 'ifnet', 'eegnet', 'adfcnn')
DATASETS = ('BNCI2014004', 'BNCI2015001')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with resolve_local_file(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def input_paths(dataset: str, model: str) -> dict[str, Path]:
    folder = INPUT_ROOT / dataset / model
    return {name: folder / name for name in
            ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv', 'manifest.json')}


def load_input(dataset: str, model: str):
    paths = input_paths(dataset, model)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{dataset}/{model}: missing prepared files {missing}')
    manifest = json.loads(resolve_local_file(paths['manifest.json']).read_text())
    for name, record in manifest['files'].items():
        path = paths[name]
        if path.stat().st_size != int(record['bytes']) or sha256(path) != record['sha256']:
            raise RuntimeError(f'{dataset}/{model}: prepared input hash mismatch: {path}')
    x = np.load(resolve_local_file(paths['X.npy']), mmap_mode='r')
    y = np.load(resolve_local_file(paths['y.npy']), mmap_mode='r')
    subjects = np.load(resolve_local_file(paths['subjects.npy']), mmap_mode='r')
    trials = __import__('pandas').read_csv(resolve_local_file(paths['trials.csv']))
    if not (len(x) == len(y) == len(subjects) == len(trials)):
        raise RuntimeError(f'{dataset}/{model}: input arrays and trial metadata differ in length')
    if not np.array_equal(trials.label_id.to_numpy(dtype=np.int64), y):
        raise RuntimeError(f'{dataset}/{model}: trial labels disagree with y.npy')
    if not np.array_equal(trials.subject_zero_based.to_numpy(dtype=np.int64), subjects):
        raise RuntimeError(f'{dataset}/{model}: trial subjects disagree with subjects.npy')
    if not np.array_equal(np.unique(y), np.arange(len(manifest['class_mapping']))):
        raise RuntimeError(f'{dataset}/{model}: class IDs are not contiguous')
    return x, np.asarray(y), np.asarray(subjects), trials, manifest


def model_config(model: str, dataset: str, input_shape: tuple[int, ...]) -> dict:
    cfg = config.load_model_config(model, dataset, 'loso')
    channels = 45 if model == 'mirepnet' else int(input_shape[0])
    cfg.update(dataset_name=dataset, in_channels=channels, samples=1000,
               sample_rate=250, skip_preprocess=model in ('mirepnet', 'cbramod'))
    if model == 'cbramod':
        cfg.update(target_fs=200, l_freq=0.3, h_freq=75.0, notch_freq=60.0,
                   norm_method=None, apply_EA=False, feature_head='flatten',
                   scale=1.0, warmup_epochs=0, min_lr=0.0,
                   label_smoothing=0.1)
    elif model == 'mirepnet':
        cfg.update(optimizer='adam', weight_decay=1e-6)
    elif model == 'ifnet':
        cfg.update(use_filter_bank=True)
    return cfg


def optimizer_for(model: nn.Module, model_name: str, cfg: dict):
    name = str(cfg.get('optimizer', 'adamw')).lower()
    kwargs = {'lr': float(cfg['lr']), 'weight_decay': float(cfg.get('weight_decay', 0.0))}
    if model_name == 'cbramod':
        kwargs['eps'] = 1e-8
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f'{model_name}: unsupported optimizer {name}')


def atomic_torch_save(path: Path, value) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    torch.save(value, partial)
    os.replace(partial, path)


def rng_state(device: torch.device, loader_generator: torch.Generator) -> dict:
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device),
        'loader_generator': loader_generator.get_state(),
    }


def restore_rng(state: dict, device: torch.device,
                loader_generator: torch.Generator) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device=device)
    loader_generator.set_state(state['loader_generator'].cpu())


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = require_external_output(path)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def fold_output(dataset: str, model: str, seed: int, subject: int) -> Path:
    return RESULT_ROOT / dataset / model / 'source_refreshed_loso_v1' / \
        f'seed_{seed}' / f'subject_{subject + 1:02d}'


def make_fold_manifest(dataset: str, model: str, seed: int, subject: int,
                       cfg: dict, input_manifest: dict, file_hashes: dict,
                       weight_path: Path | None, weight_hash: str | None,
                       n_train: int, n_test: int, device: torch.device) -> dict:
    return {
        'protocol': 'loso_source_refresh_004_5001_v1',
        'dataset': dataset, 'model': model,
        'recipe': 'source_refreshed_loso_v1', 'seed': seed,
        'held_out_subject': subject + 1,
        'train_subjects': [s + 1 for s in range(len(input_manifest['subject_trial_counts']))
                           if s != subject],
        'n_train': n_train, 'n_test': n_test,
        'selected_session': input_manifest['selected_session'],
        'source_variant': input_manifest['source_variant'],
        'source_manifest_sha256': input_manifest['source_manifest_sha256'],
        'input_manifest_sha256': file_hashes['manifest.json']['sha256'],
        'input_files': file_hashes,
        'class_mapping': input_manifest['class_mapping'],
        'window': input_manifest['input_shape'],
        'preprocessing': input_manifest['preprocessing'],
        'model_config': cfg,
        'pretrained_checkpoint': str(weight_path) if weight_path else None,
        'pretrained_sha256': weight_hash,
        'evaluation': 'fixed_final_epoch_once_no_validation',
        'ea_regime': input_manifest['ea_regime'],
    }


def train_fold(dataset: str, model_name: str, seed: int, subject: int,
               x_all: np.ndarray, y_all: np.ndarray, subject_ids: np.ndarray,
               trials, input_manifest: dict, file_hashes: dict,
               cfg: dict, device: torch.device, preflight: bool = False) -> dict | None:
    n_subjects = len(np.unique(subject_ids))
    train_rows = np.flatnonzero(subject_ids != subject)
    test_rows = np.flatnonzero(subject_ids == subject)
    x_train_raw = np.asarray(x_all[train_rows], dtype=np.float32)
    y_train = y_all[train_rows].astype(np.int64, copy=False)
    x_test_raw = np.asarray(x_all[test_rows], dtype=np.float32)
    y_test = y_all[test_rows].astype(np.int64, copy=False)
    n_classes = len(input_manifest['class_mapping'])
    fold_cfg = dict(cfg)
    if preflight:
        fold_cfg['epochs'] = 1
    output = fold_output(dataset, model_name, seed, subject)
    output = require_external_output(output)
    model_manifest_path = output / 'manifest.json'
    expected_manifest = make_fold_manifest(
        dataset, model_name, seed, subject, fold_cfg, input_manifest,
        file_hashes, None, None, len(train_rows), len(test_rows), device)

    weight_path = None
    weight_hash = None
    if model_name in ('mirepnet', 'cbramod'):
        weight_path = Path(config.weight_path(model_name)).resolve()
        weight_hash = sha256(weight_path)
    expected_manifest['pretrained_checkpoint'] = str(weight_path) if weight_path else None
    expected_manifest['pretrained_sha256'] = weight_hash
    if model_manifest_path.exists() and not preflight:
        old = json.loads(resolve_local_file(model_manifest_path).read_text())
        for key, value in expected_manifest.items():
            if old.get(key) != value:
                raise RuntimeError(f'{model_manifest_path}: {key} differs from the active run')
        result_path = output / 'result.npz'
        checkpoint_path = output / 'final_model.pt'
        if result_path.is_file() and checkpoint_path.is_file() and old.get('status') == 'complete':
            with np.load(resolve_local_file(result_path), allow_pickle=False) as saved:
                metrics = json.loads(str(saved['metrics_json'].item()))
            print(f'[skip-complete] {dataset}/{model_name} S{subject+1} seed={seed}', flush=True)
            return metrics

    set_seed(seed)
    adapter = get_adapter(model_name, device=device, **fold_cfg)
    model = adapter.build(n_classes)
    model.train()
    x_train = adapter.preprocess(x_train_raw)
    x_test = adapter.preprocess(x_test_raw)
    batch_size = int(fold_cfg['batch_size'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train, torch.as_tensor(y_train, dtype=torch.long)),
                        batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=0, generator=generator)
    optimizer = optimizer_for(model, model_name, fold_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(fold_cfg['epochs']))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(fold_cfg.get('label_smoothing', 0.0)))
    if preflight:
        xb, yb = next(iter(loader))
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = adapter.forward(model, xb)[1]
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        print(f'[preflight-ok] {model_name}/{dataset} input={tuple(xb.shape)} '
              f'logits={tuple(logits.shape)} gpu={device.index}', flush=True)
        del adapter, model, x_train, x_test, x_train_raw, x_test_raw
        gc.collect()
        torch.cuda.empty_cache()
        return None

    output.mkdir(parents=True, exist_ok=True)
    if model_manifest_path.exists():
        existing = json.loads(model_manifest_path.read_text())
        if existing.get('status') == 'running':
            state_path = output / 'training_state.pt'
            if state_path.exists():
                saved = torch.load(resolve_local_file(state_path), map_location='cpu', weights_only=False)
                if saved.get('identity') != expected_manifest:
                    raise RuntimeError(f'{state_path}: cannot resume a different task')
                model.load_state_dict(saved['model_state'])
                optimizer.load_state_dict(saved['optimizer_state'])
                scheduler.load_state_dict(saved['scheduler_state'])
                history = saved['history']
                first_epoch = int(saved['next_epoch'])
                restore_rng(saved['rng_state'], device, generator)
            else:
                history, first_epoch = [], 0
        else:
            history, first_epoch = [], 0
    else:
        history, first_epoch = [], 0
    running_manifest = dict(expected_manifest, status='running',
                            started_utc_epoch_s=time.time())
    temp_manifest = model_manifest_path.with_suffix('.json.partial')
    temp_manifest.write_text(json.dumps(running_manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temp_manifest, model_manifest_path)
    state_path = output / 'training_state.pt'
    if first_epoch == 0:
        atomic_torch_save(state_path, {
            'identity': expected_manifest, 'next_epoch': 0, 'history': history,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'rng_state': rng_state(device, generator),
        })

    for epoch in range(first_epoch, int(fold_cfg['epochs'])):
        model.train()
        loss_total = 0.0
        seen = 0
        order = []
        lr_used = float(optimizer.param_groups[0]['lr'])
        started = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward(model, xb)[1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(yb)
            seen += len(yb)
        scheduler.step()
        history.append({
            'epoch': epoch + 1, 'train_loss': loss_total / max(seen, 1),
            'n_train': seen, 'lr_used': lr_used,
            'lr_next': float(optimizer.param_groups[0]['lr']),
            'epoch_seconds': time.time() - started,
        })
        write_history(output / 'history.csv', history)
        atomic_torch_save(state_path, {
            'identity': expected_manifest, 'next_epoch': epoch + 1,
            'history': history, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'rng_state': rng_state(device, generator),
        })
        status = dict(running_manifest, epoch=epoch + 1,
                      epochs=int(fold_cfg['epochs']),
                      elapsed_seconds=sum(row['epoch_seconds'] for row in history),
                      gpu_physical_index=device.index)
        temp_status = (output / 'status.json.partial')
        temp_status.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
        os.replace(temp_status, output / 'status.json')
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(fold_cfg['epochs']):
            print(f'[epoch] {model_name}/{dataset} S{subject+1} seed={seed} '
                  f'{epoch+1}/{fold_cfg["epochs"]} loss={history[-1]["train_loss"]:.5f} '
                  f'lr={lr_used:.3g}', flush=True)

    model.eval()
    logits_list, features_list = [], []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            xb = x_test[start:start+batch_size].to(device)
            features, logits = adapter.forward(model, xb)
            features_list.append(features.float().cpu())
            logits_list.append(logits.float().cpu())
    logits_np = torch.cat(logits_list).numpy()
    features_np = torch.cat(features_list).numpy()
    metrics, probabilities, predictions, cm = protocol._metrics(y_test, logits_np)
    metrics.update(model=model_name, dataset=dataset, test_subject=subject + 1,
                   seed=seed, n_train=len(train_rows), n_test=len(test_rows),
                   elapsed_sec=round(sum(row['epoch_seconds'] for row in history), 2),
                   artifact_origin='trained_on_refreshed_moabb_source')
    test_uid = trials.trial_uid.astype(str).to_numpy()[test_rows]
    result_path = output / 'result.npz'
    temp_result = result_path.with_suffix('.npz.partial')
    with temp_result.open('wb') as stream:
        np.savez_compressed(
            stream, y=y_test, predictions=predictions, probabilities=probabilities,
            logits=logits_np, features=features_np, trial_uid=test_uid,
            confusion_matrix=cm,
            metrics_json=np.asarray(json.dumps(metrics, sort_keys=True, allow_nan=True)))
    os.replace(temp_result, result_path)
    final_model_path = output / 'final_model.pt'
    atomic_torch_save(final_model_path, model.state_dict())
    model_manifest = dict(expected_manifest, status='complete',
                          completed_utc_epoch_s=time.time(),
                          training_seconds=sum(row['epoch_seconds'] for row in history),
                          final_model_sha256=sha256(final_model_path),
                          result_sha256=sha256(result_path),
                          test_metrics=metrics)
    temp_manifest = model_manifest_path.with_suffix('.json.partial')
    temp_manifest.write_text(json.dumps(model_manifest, indent=2, sort_keys=True,
                                        allow_nan=True) + '\n')
    os.replace(temp_manifest, model_manifest_path)
    if state_path.exists():
        state_path.unlink()
    print(f'[complete] {model_name}/{dataset} S{subject+1} seed={seed} '
          f'acc={metrics["accuracy"]:.4f} elapsed={metrics["elapsed_sec"]:.1f}s', flush=True)
    del adapter, model, x_train, x_test, x_train_raw, x_test_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def summarize_task(dataset: str, model: str, seed: int,
                   n_subjects: int) -> None:
    root = RESULT_ROOT / dataset / model / 'source_refreshed_loso_v1' / f'seed_{seed}'
    rows = []
    for subject in range(n_subjects):
        path = fold_output(dataset, model, seed, subject) / 'result.npz'
        if not path.is_file():
            continue
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            rows.append(json.loads(str(saved['metrics_json'].item())))
    if not rows:
        return
    metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')
    summary = {
        'protocol': 'loso_source_refresh_004_5001_v1', 'dataset': dataset,
        'model': model, 'seed': seed,
        'complete': len(rows) == n_subjects,
        'completed_folds': len(rows), 'expected_folds': n_subjects,
        'subject_equal_mean': {name: float(np.nanmean([row[name] for row in rows]))
                               for name in metrics},
    }
    path = require_external_output(root / 'seed_summary.json')
    temp = path.with_suffix('.json.partial')
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + '\n')
    os.replace(temp, path)


def run(args) -> None:
    if args.gpu == 0:
        raise SystemExit('GPU 0 is prohibited for this experiment')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; refusing silent CPU fallback')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    x, y, subjects, trials, input_manifest = load_input(args.dataset, args.model)
    n_subjects = int(np.max(subjects)) + 1
    expected_subjects = 9 if args.dataset == 'BNCI2014004' else 12
    if n_subjects != expected_subjects:
        raise RuntimeError(f'{args.dataset}: expected {expected_subjects} subjects, got {n_subjects}')
    paths = input_paths(args.dataset, args.model)
    file_hashes = {name: {'path': str(path), 'sha256': sha256(path),
                          'bytes': path.stat().st_size}
                   for name, path in paths.items() if name != 'manifest.json'}
    file_hashes['manifest.json'] = {
        'path': str(paths['manifest.json']), 'sha256': sha256(paths['manifest.json']),
        'bytes': paths['manifest.json'].stat().st_size}
    cfg = model_config(args.model, args.dataset, tuple(x.shape[1:]))
    folds = range(n_subjects) if args.folds is None else args.folds
    for subject in folds:
        if subject < 0 or subject >= n_subjects:
            raise ValueError(f'held-out subject must be in [0,{n_subjects - 1}]')
        train_fold(args.dataset, args.model, args.seed, subject, x, y,
                   subjects, trials, input_manifest, file_hashes, cfg,
                   device, preflight=args.preflight_only)
        if args.preflight_only:
            break
    if not args.preflight_only:
        summarize_task(args.dataset, args.model, args.seed, n_subjects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--folds', type=int, nargs='+')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
