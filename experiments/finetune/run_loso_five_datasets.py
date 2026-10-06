"""Run the fixed five-setting MIRepNet / CBraMod LOSO reproduction.

Run inside each model's own environment. Use ``--preflight-only`` before
training; ``--seeds`` can be supplied one seed at a time and completed cells
are resumed after their manifests and resolved configs are checked.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

import config
from data.eeg_dataset import EEGDataset, _DATA_ROOT
from models import get_adapter


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / 'configs/reproductions/loso_five_datasets_v1.yaml'
SNAPSHOT_PATH = ROOT / 'configs/reproductions/manifests/loso_five_datasets_v1_sources.json'
TRIALS_PATH = ROOT / 'configs/reproductions/manifests/loso_five_datasets_v1_trials.csv'
PROTOCOL = 'loso_five_settings_canonical4s_v1'
RESULTS_ROOT = ROOT / 'results/reproductions' / PROTOCOL
LEGACY_004 = ROOT / 'results/reproductions/loso_benchmark_session3_4s_v1'
DATASET_NAMES = ('BNCI2014001', 'BNCI2014001-4', 'BNCI2014004',
                 'BNCI2015001', 'AlexMI')
MODEL_NAMES = ('mirepnet', 'cbramod')
METRICS = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASET_NAMES,
                        default=list(DATASET_NAMES))
    parser.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='zero-based held-out subject IDs; omit for all subjects')
    parser.add_argument('--recipe', choices=('main', 'eegfm_full'), default='main',
                        help='eegfm_full is an optional CBraMod-only transferred reference recipe')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--preflight-only', action='store_true',
                        help='validate all selected inputs and run one disposable optimizer step per dataset')
    parser.add_argument('--no-reuse-004', action='store_true',
                        help='train the 004 main cells again instead of importing their verified artifacts')
    parser.add_argument('--force', action='store_true',
                        help='retrain selected cells even if a matching complete cell exists')
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _environment_snapshot(device):
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'mne', 'pandas'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    info = {'python': sys.version.split()[0], 'packages': packages,
            'torch_cuda': torch.version.cuda, 'device': str(device)}
    if str(device).startswith('cuda'):
        info['device_name'] = torch.cuda.get_device_name(torch.device(device))
    return info


def _load_spec():
    spec = yaml.safe_load(SPEC_PATH.read_text())
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    actual_hash = _sha256(SPEC_PATH)
    if snapshot['spec_sha256'] != actual_hash:
        raise RuntimeError('experiment YAML differs from its recorded source snapshot')
    return spec, snapshot


def _verify_source_files(dataset, spec, snapshot):
    ds = spec['datasets'][dataset]
    source = ds['source_dataset']
    source_info = snapshot['sources'][source]
    source_root = Path(_DATA_ROOT) / source
    current = {}
    for name, record in source_info['files'].items():
        path = source_root / name
        digest = _sha256(path)
        if digest != record['sha256']:
            raise RuntimeError(f'{dataset}: source file hash changed: {path}')
        current[name] = {'path': str(path), 'sha256': digest,
                         'bytes': path.stat().st_size}
    return current


def _load_trials(dataset, spec):
    ds = spec['datasets'][dataset]
    subjects = list(range(ds['subjects']))
    kwargs = {'dataset_name': dataset, 'sub': subjects}
    if 'loader_data_mode' in ds:
        kwargs['data_mode'] = ds['loader_data_mode']
    if dataset == 'BNCI2015001':
        os.environ['MI2015001_SESSION'] = 'session_A'
    source = ds['source_dataset']
    meta_path = Path(_DATA_ROOT) / source / ('meta004.csv' if source == 'BNCI2014004' else 'meta.csv')
    meta = pd.read_csv(meta_path, dtype={'session': str, 'run': str})
    y_raw = np.load(Path(_DATA_ROOT) / source / 'labels.npy', allow_pickle=True).astype(str)
    args = SimpleNamespace(**kwargs)
    loaded = EEGDataset(args=args)
    x = np.asarray(loaded.X, dtype=np.float32)
    y = np.asarray(loaded.y, dtype=np.int64)

    records = []
    with TRIALS_PATH.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['task_dataset'] == dataset:
                records.append(row)
    records.sort(key=lambda r: (int(r['subject']), int(r['selected_trial_index'])))
    if len(records) != ds['selected_trials'] or len(y) != len(records):
        raise RuntimeError(f'{dataset}: loader/trial manifest count mismatch: {len(x)}, {len(y)}, {len(records)}')
    if x.ndim != 3 or x.shape[1] != ds['native_channels'] or x.shape[2] < 1000:
        raise RuntimeError(f'{dataset}: invalid loaded signal shape {x.shape}')

    subject_ids = np.asarray([int(r['zero_subject']) for r in records], dtype=np.int64)
    selected_rows = np.asarray([int(r['selected_trial_index']) for r in records], dtype=np.int64)
    source_rows = np.asarray([int(r['raw_row']) for r in records], dtype=np.int64)
    source_uids = np.column_stack((subject_ids, source_rows))
    local_uids = np.column_stack((subject_ids, selected_rows))
    manifest_label = np.asarray([int(r['label_id']) for r in records], dtype=np.int64)
    if not np.array_equal(y, manifest_label):
        mismatch = np.flatnonzero(y != manifest_label)[:8].tolist()
        raise RuntimeError(f'{dataset}: loader label IDs disagree with task manifest at {mismatch}')
    if set(np.unique(y).tolist()) != set(range(len(ds['classes']))):
        raise RuntimeError(f'{dataset}: loaded classes do not match fixed mapping {ds["classes"]}')
    for subject in range(ds['subjects']):
        n = int((subject_ids == subject).sum())
        if n != ds['per_subject_trials'][subject]:
            raise RuntimeError(f'{dataset} S{subject+1}: expected {ds["per_subject_trials"][subject]} trials, got {n}')
    if dataset == 'BNCI2015001' and os.environ.get('MI2015001_SESSION') != 'session_A':
        raise RuntimeError('BNCI2015001 session is not locked to session_A')
    if len({tuple(row) for row in source_uids.tolist()}) != len(source_uids):
        raise RuntimeError(f'{dataset}: duplicate source UID in selected trial set')
    return x[:, :, :1000], y, subject_ids, source_uids, local_uids, records, meta, y_raw


def _config(model, dataset, recipe, ds_spec, spec):
    cfg = config.load_model_config(model, dataset, 'loso')
    cfg.update(dataset_name=dataset, in_channels=ds_spec['native_channels'], samples=1000)
    if model == 'mirepnet':
        cfg.update(skip_preprocess=True)
        return cfg

    cfg.update(target_fs=200, l_freq=0.3, h_freq=75.0, notch_freq=60.0,
               norm_method=None, apply_EA=False, scale=1.0,
               feature_head='flatten')
    if recipe == 'main':
        cfg['dropout'] = 0.1
    else:
        if dataset not in spec['optional_cbramod_reference_recipe']['per_dataset']:
            raise ValueError(f'No EEG-FM-Benchmark reference recipe is specified for {dataset}')
        ref = spec['optional_cbramod_reference_recipe']
        override = ref['per_dataset'][dataset]
        cfg.update(epochs=int(ref['epochs']), lr=float(ref['lr']),
                   batch_size=int(override['batch_size']),
                   weight_decay=float(override['weight_decay']),
                   dropout=float(ref['dropout']),
                   label_smoothing=float(ref['label_smoothing']),
                   optimizer='adamw', optimizer_eps=float(ref['optimizer_eps']),
                   warmup_epochs=int(ref['warmup_epochs']), min_lr=float(ref['min_lr']),
                   norm_method=override['norm_method'])
    return cfg


def _profile(model, recipe, dataset, spec, cfg):
    if recipe == 'main':
        return {'name': spec['main_recipes'][model]['name'],
                'parameter_source': spec['main_recipes'][model]['parameter_source'],
                'resolved_config_sha256': _json_hash(cfg)}
    ref = spec['optional_cbramod_reference_recipe']
    return {'name': ref['name'], 'source': ref['per_dataset'][dataset]['source'],
            'original_task': ref['original_task'],
            'resolved_config_sha256': _json_hash(cfg),
            'schedule': ref['lr_schedule'],
            'transfer_note': 'Fewshot parameters applied to full-source-subject outer LOSO training'}


def _make_adapter(model, cfg, device, protocol, dataset, recipe):
    adapter = get_adapter(model, device=device, **cfg)
    adapter.name = f'{protocol}_{dataset}_{recipe}_{model}'
    return adapter


def _prepare_mirepnet(x, subject_ids, adapter):
    from data.preproc import bandpass
    x = bandpass(np.asarray(x, dtype=np.float64), 250, 8.0, 30.0)
    return adapter.ea_pad_per_subject(x, subject_ids)


def _metrics(y, logits):
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    logits = np.asarray(logits)
    n_classes = logits.shape[1]
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(n_classes))
    try:
        if n_classes == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, labels=np.arange(n_classes),
                                multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    metrics = {
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'kappa': float(cohen_kappa_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=np.arange(n_classes),
                                   average='macro', zero_division=0)),
        'auroc': float(auc),
        'confusion_matrix': cm.tolist(),
    }
    return metrics, probs, pred, cm


def _train_mirepnet(adapter, model, x, y, cfg):
    x_tensor = adapter.preprocess(x)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                        batch_size=int(cfg['batch_size']), shuffle=True,
                        num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                                 weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg['epochs']))
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(int(cfg['epochs'])):
        model.train()
        lr_used = optimizer.param_groups[0]['lr']
        loss_sum = 0.0
        seen = 0
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
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / max(seen, 1),
                        'n_train': seen, 'lr_used': lr_used,
                        'lr_next': optimizer.param_groups[0]['lr']})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(cfg['epochs']):
            print(f'  epoch={epoch+1:02d}/{cfg["epochs"]} '
                  f'loss={history[-1]["train_loss"]:.5f} lr={lr_used:.3g}', flush=True)
    return model, history


def _train_cbramod(adapter, model, x, y, cfg, seed, recipe):
    x_tensor = adapter.preprocess(x)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                        batch_size=int(cfg['batch_size']), shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg['lr']),
        weight_decay=float(cfg['weight_decay']),
        eps=float(cfg.get('optimizer_eps', 1e-8)))
    if recipe == 'eegfm_full':
        lr_schedule = adapter._cosine_schedule(
            float(cfg['lr']), float(cfg['min_lr']), int(cfg['epochs']),
            len(loader), warmup_epochs=int(cfg['warmup_epochs']))
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg['epochs']), eta_min=0.0)
        lr_schedule = None
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg['label_smoothing']))
    history = []
    global_step = 0
    for epoch in range(int(cfg['epochs'])):
        model.train()
        start_lrs = []
        loss_total = 0.0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(adapter.device), yb.to(adapter.device)
            if lr_schedule is not None:
                adapter._apply_lr_schedule(optimizer, lr_schedule, global_step)
            start_lrs.append(float(optimizer.param_groups[0]['lr']))
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward(model, xb)[1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_total += float(loss.detach()) * len(yb)
            seen += len(yb)
        if scheduler is not None:
            scheduler.step()
        entry = {'epoch': epoch + 1, 'train_loss': loss_total / max(seen, 1),
                 'n_train': seen, 'lr_min_used': min(start_lrs),
                 'lr_max_used': max(start_lrs),
                 'lr_next': float(optimizer.param_groups[0]['lr'])}
        history.append(entry)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(cfg['epochs']):
            print(f'  epoch={epoch+1:02d}/{cfg["epochs"]} '
                  f'loss={entry["train_loss"]:.5f} '
                  f'lr={entry["lr_max_used"]:.3g}', flush=True)
    return model, history


def _cell_paths(output_dir, subject, seed):
    cell = output_dir / f'subject_{subject + 1:02d}' / f'seed_{seed}'
    return cell, cell/'result.npz', cell/'model.pt', cell/'manifest.json', cell/'train_history.csv'


def _write_history(path, history):
    keys = list(history[0]) if history else ['epoch']
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def _save_cell(result_path, checkpoint_path, manifest_path, history_path,
               model, result, manifest, history):
    checkpoint_tmp = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(model.state_dict(), checkpoint_tmp)
    os.replace(checkpoint_tmp, checkpoint_path)
    result_tmp = result_path.with_suffix('.npz.tmp')
    with result_tmp.open('wb') as stream:
        np.savez_compressed(stream, **result)
    os.replace(result_tmp, result_path)
    _write_history(history_path, history)
    manifest_tmp = manifest_path.with_suffix('.json.tmp')
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + '\n')
    os.replace(manifest_tmp, manifest_path)


def _cell_is_complete(result_path, checkpoint_path, manifest_path, history_path,
                      cfg, source_hashes, weight_sha, force):
    if force or not all(p.is_file() for p in (result_path, checkpoint_path, manifest_path)):
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('model_config') != cfg:
        raise RuntimeError(f'completed cell model config mismatch: {manifest_path}')
    if manifest.get('source_files') != source_hashes or manifest.get('pretrained_sha256') != weight_sha:
        raise RuntimeError(f'completed cell data or pretrained checkpoint mismatch: {manifest_path}')
    if not manifest.get('artifact_origin', '').startswith('reused_') and not history_path.is_file():
        return None
    with np.load(result_path) as saved:
        return json.loads(str(saved['metrics_json'].item()))


def _import_004_cell(dataset, model, subject, seed, cfg, y_test, uid_test,
                     local_uid_test, train_uids, output_dir, source_file_payload, weight_sha,
                     profile_meta, n_classes):
    legacy_dir = LEGACY_004 / model / f'subject_{subject + 1:02d}' / f'seed_{seed}'
    legacy_result = legacy_dir/'result.npz'
    legacy_checkpoint = legacy_dir/'model.pt'
    legacy_manifest = legacy_dir/'manifest.json'
    if not all(p.is_file() for p in (legacy_result, legacy_checkpoint, legacy_manifest)):
        raise FileNotFoundError(f'completed legacy 004 cell missing: {legacy_dir}')
    old = json.loads(legacy_manifest.read_text())
    if old.get('model_config') != cfg:
        raise RuntimeError(f'legacy 004 config does not match current main recipe: {legacy_manifest}')
    legacy_split = json.loads((LEGACY_004/'split_manifest.json').read_text())
    for name, item in legacy_split['source_files'].items():
        if source_file_payload['files'].get(name, {}).get('sha256') != item['sha256']:
            raise RuntimeError(f'legacy 004 source hash mismatch: {name}')
    if old.get('pretrained_sha256') != weight_sha:
        raise RuntimeError('legacy 004 pretrained checkpoint hash mismatch')

    with np.load(legacy_result) as saved:
        old_y = saved['y']
        old_uid = saved['sample_uid'].astype(np.int64)
        if not np.array_equal(old_y, y_test):
            raise RuntimeError(f'legacy 004 labels no longer match S{subject+1}')
        if (len(old_uid) != len(uid_test) or len(local_uid_test) != len(uid_test)
                or not np.all(old_uid[:, 0] == subject)):
            raise RuntimeError(f'legacy 004 trial IDs invalid for S{subject+1}')
        local = old_uid[:, 1]
        if set(local.tolist()) != set(range(len(uid_test))):
            raise RuntimeError(f'legacy 004 local UID order invalid for S{subject+1}')
        # Old UIDs were (zero-based subject, session-local trial index).
        local_order = np.argsort(local_uid_test[:, 1])
        local_by_index = local_uid_test[local_order]
        if (not np.array_equal(local_by_index[:, 0], np.full(len(uid_test), subject))
                or not np.array_equal(local_by_index[:, 1], np.arange(len(uid_test)))):
            raise RuntimeError('new 004 source-row mapping is not indexed by legacy local trial')
        uid_by_local = uid_test[local_order]
        new_uid = uid_by_local[local]
        copied = {key: saved[key].copy() for key in saved.files
                  if key not in ('sample_uid', 'metrics_json')}
        old_metrics = json.loads(str(saved['metrics_json'].item()))
    copied['sample_uid'] = new_uid
    copied['legacy_sample_uid'] = old_uid
    metrics, probs, pred, cm = _metrics(old_y, copied['logits'])
    metrics.update(model=model, dataset=dataset, test_subject=subject+1, seed=seed,
                   n_train=len(train_uids), n_test=len(y_test),
                   elapsed_sec=old_metrics.get('elapsed_sec'),
                   artifact_origin='reused_004_main')
    copied.update(y=old_y.copy(), pred=pred, probs=probs, confusion_matrix=cm,
                  metrics_json=np.asarray(json.dumps(metrics, sort_keys=True, allow_nan=True)))
    cell, result_path, checkpoint_path, manifest_path, history_path = _cell_paths(
        output_dir, subject, seed)
    cell.mkdir(parents=True, exist_ok=True)
    tmp = result_path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **copied)
    os.replace(tmp, result_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    try:
        os.link(legacy_checkpoint, checkpoint_path)
    except OSError:
        shutil.copy2(legacy_checkpoint, checkpoint_path)
    manifest = {
        'protocol': PROTOCOL, 'dataset': dataset, 'model': model,
        'recipe': 'main', 'training_profile': profile_meta,
        'artifact_origin': 'reused_004_main_verified',
        'reuse_source': {'protocol': 'loso_benchmark_session3_4s_v1',
                         'result_sha256': _sha256(legacy_result),
                         'checkpoint_sha256': _sha256(legacy_checkpoint),
                         'manifest_path': str(legacy_manifest)},
        'legacy_provenance_gaps': ['per_epoch_loss_and_lr_history',
                                   'historical_software_environment_snapshot'],
        'dataset_source': spec_source(dataset), 'source_files': source_file_payload,
        'pretrained_sha256': weight_sha,
        'model_config': cfg, 'environment_snapshot': None,
        'num_classes': n_classes, 'test_subject': subject+1,
        'train_subjects': sorted({int(uid[0])+1 for uid in train_uids}),
        'seed': seed, 'n_train': len(train_uids), 'n_test': len(y_test),
        'window': 'canonical_4s; 004_first_1000_at_250hz',
        'source_dataset': spec_source(dataset),
        'label_values': _dataset_label_map(dataset),
        'train_sample_uids': train_uids.tolist(),
        'test_sample_uids': new_uid.tolist(),
        'selection_policy': 'fixed_final_epoch_no_validation',
        'ea_policy': 'per_subject; held-out 004 covariance uses unlabeled full-fold trials'
                     if model == 'mirepnet' else 'disabled',
        'test_metrics': metrics,
    }
    manifest_tmp = manifest_path.with_suffix('.json.tmp')
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True)+'\n')
    os.replace(manifest_tmp, manifest_path)
    print(f'[reuse] {model} {dataset} S{subject+1} seed={seed} '
          f'acc={metrics["accuracy"]:.4f}', flush=True)
    return metrics


def spec_source(dataset):
    return SPEC['datasets'][dataset]['source_dataset']


def _dataset_label_map(dataset):
    return SPEC['datasets'][dataset]['classes']


def _summary(output_dir, dataset, model, recipe, seeds, n_subjects, profile_meta):
    rows = []
    for seed in seeds:
        for subject in range(n_subjects):
            _, result_path, _, _, _ = _cell_paths(output_dir, subject, seed)
            if result_path.is_file():
                with np.load(result_path) as saved:
                    rows.append(json.loads(str(saved['metrics_json'].item())))
    scalar_metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')
    csv_path = output_dir/'summary.csv'
    columns = ['dataset', 'model', 'recipe', 'test_subject', 'seed',
               'n_train', 'n_test', *scalar_metrics, 'elapsed_sec', 'artifact_origin']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r['seed'], r['test_subject'])):
            writer.writerow({key: row.get(key) for key in columns})
    per_seed = {}
    for seed in seeds:
        seed_rows = [r for r in rows if int(r['seed']) == seed]
        if seed_rows:
            per_seed[str(seed)] = {
                m: {'mean_subjects': float(np.nanmean([r[m] for r in seed_rows])),
                    'std_subjects': float(np.nanstd([r[m] for r in seed_rows], ddof=1))
                    if len(seed_rows) > 1 else 0.0}
                for m in scalar_metrics
            }
    summary = {
        'protocol': PROTOCOL, 'dataset': dataset, 'model': model, 'recipe': recipe,
        'training_profile': profile_meta,
        'expected_fold_seed_cells': n_subjects*len(seeds),
        'completed_fold_seed_cells': len(rows),
        'per_seed_subject_mean': per_seed,
        'complete': len(rows) == n_subjects*len(seeds),
    }
    if rows:
        summary['subject_fold_seed_macro_mean'] = {
            m: float(np.nanmean([r[m] for r in rows])) for m in scalar_metrics}
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True)+'\n')


def _run_preflight(model, datasets, recipe, device, spec):
    for dataset in datasets:
        ds = spec['datasets'][dataset]
        cfg = _config(model, dataset, recipe, ds, spec)
        x, y, subject_ids, _, _, _, _, _ = _load_trials(dataset, spec)
        batch_size = min(int(cfg['batch_size']), len(y))
        xb_raw, yb = x[:batch_size], y[:batch_size]
        _set_seed(123)
        adapter = _make_adapter(model, cfg, device, PROTOCOL, dataset, recipe)
        if model == 'mirepnet':
            xb = _prepare_mirepnet(xb_raw, subject_ids[:batch_size], adapter)
        else:
            xb = adapter.preprocess(xb_raw)
        model_instance = adapter.build(len(ds['classes']))
        xb_t = torch.as_tensor(xb, dtype=torch.float32).to(device)
        yb_t = torch.as_tensor(yb[:batch_size], dtype=torch.long).to(device)
        model_instance.train()
        optimizer = (torch.optim.Adam(model_instance.parameters(), lr=float(cfg['lr']),
                                      weight_decay=float(cfg['weight_decay']))
                     if model == 'mirepnet' else
                     torch.optim.AdamW(model_instance.parameters(), lr=float(cfg['lr']),
                                       weight_decay=float(cfg['weight_decay']), eps=1e-8))
        logits = adapter.forward(model_instance, xb_t)[1]
        if logits.shape != (batch_size, len(ds['classes'])):
            raise RuntimeError(f'{model}/{dataset}: head produced {tuple(logits.shape)}')
        loss = nn.CrossEntropyLoss()(logits, yb_t)
        loss.backward()
        optimizer.step()
        print(f'[preflight-ok] {model} {dataset} input={tuple(xb_t.shape)} '
              f'classes={len(ds["classes"])} batch={batch_size}', flush=True)
        del model_instance, adapter, optimizer, xb_t, yb_t, xb, x, y, subject_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(args, spec, snapshot, device):
    model = args.model
    if args.recipe == 'eegfm_full' and model != 'cbramod':
        raise ValueError('eegfm_full recipe is only defined for CBraMod')
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError('--datasets contains duplicates')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('--seeds must be non-empty and unique')

    weight_path = Path(config.weight_path(model)).resolve()
    weight_sha = _sha256(weight_path)
    if snapshot['pretrained_weights'][model]['sha256'] != weight_sha:
        raise RuntimeError(f'{model} pretrained checkpoint differs from frozen plan snapshot')
    weight_record = {'path': str(weight_path), 'sha256': weight_sha,
                     'bytes': weight_path.stat().st_size}

    for dataset in args.datasets:
        ds = spec['datasets'][dataset]
        n_subjects = int(ds['subjects'])
        folds = list(range(n_subjects)) if args.folds is None else args.folds
        if any(f < 0 or f >= n_subjects for f in folds):
            raise ValueError(f'{dataset}: --folds must be zero-based IDs in [0,{n_subjects-1}]')
        source_hashes = _verify_source_files(dataset, spec, snapshot)
        x, y, subject_ids, source_uids, local_uids, records, meta, labels_raw = _load_trials(dataset, spec)
        cfg = _config(model, dataset, args.recipe, ds, spec)
        profile_meta = _profile(model, args.recipe, dataset, spec, cfg)
        recipe_dirname = spec['main_recipes'][model]['name'] if args.recipe == 'main' else spec['optional_cbramod_reference_recipe']['name']
        output_dir = RESULTS_ROOT / dataset / model / recipe_dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        source_file_payload = {'dataset_source': ds['source_dataset'], 'files': source_hashes,
                               'trial_manifest_sha256': _sha256(TRIALS_PATH),
                               'spec_sha256': snapshot['spec_sha256']}
        split_manifest = {
            'protocol': PROTOCOL, 'dataset': dataset, 'model': model,
            'recipe': recipe_dirname, 'training_profile': profile_meta,
            'source_dataset': ds['source_dataset'],
            'source_shape': ds['source_shape'], 'source_fs_hz': ds['source_fs_hz'],
            'selected_session': ds['selected_session'],
            'class_mapping': ds['classes'], 'num_classes': len(ds['classes']),
            'num_subjects': n_subjects, 'total_trials': len(y),
            'subject_trial_counts': ds['per_subject_trials'],
            'window': {'canonical_samples': 1000, 'canonical_fs_hz': 250,
                       'window_label': ds.get('window_label', '4s_from_acquired_signal')},
            'test_policy': 'one full subject held out; fixed final epoch once',
            'validation_policy': 'none_in_fixed_recipe_baseline',
            'source_files': source_file_payload,
            'pretrained_checkpoint': weight_record,
            'resolved_model_config': cfg,
            'resolved_config_sha256': _json_hash(cfg),
            'seeds': list(args.seeds),
        }
        split_path = output_dir/'split_manifest.json'
        if split_path.exists():
            old = json.loads(split_path.read_text())
            previous_seeds = old.pop('seeds', [])
            expected = dict(split_manifest); expected.pop('seeds')
            if old != expected:
                raise RuntimeError(f'output directory belongs to another split or recipe: {split_path}')
            split_manifest['seeds'] = sorted(set(previous_seeds)|set(args.seeds))
        split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True)+'\n')

        counts = [int((subject_ids == s).sum()) for s in range(n_subjects)]
        print(f'[dataset] {dataset} trials={len(y)} subjects={n_subjects} '
              f'per_subject={counts} device={device} recipe={args.recipe}', flush=True)
        for fold in folds:
            train_mask = subject_ids != fold
            test_mask = subject_ids == fold
            if np.any(train_mask & test_mask) or set(np.unique(subject_ids[train_mask]).tolist()) != set(range(n_subjects))-{fold}:
                raise RuntimeError(f'{dataset} fold {fold}: invalid train/test subject partition')
            x_tr, y_tr = x[train_mask], y[train_mask]
            x_te, y_te = x[test_mask], y[test_mask]
            sub_tr = subject_ids[train_mask]
            uid_tr, uid_te = source_uids[train_mask], source_uids[test_mask]
            local_uid_te = local_uids[test_mask]
            for seed in args.seeds:
                cell, rp, cp, mp, hp = _cell_paths(output_dir, fold, seed)
                cell.mkdir(parents=True, exist_ok=True)
                cached = _cell_is_complete(rp, cp, mp, hp, cfg, source_file_payload,
                                           weight_sha, args.force)
                if cached is not None:
                    print(f'[skip] {model} {dataset} S{fold+1} seed={seed}', flush=True)
                    continue
                if (dataset == 'BNCI2014004' and args.recipe == 'main'
                        and not args.no_reuse_004 and not args.force):
                    metric = _import_004_cell(
                        dataset, model, fold, seed, cfg, y_te, uid_te, local_uid_te, uid_tr,
                        output_dir, source_file_payload, weight_sha, profile_meta,
                        len(ds['classes']))
                    continue

                _set_seed(seed)
                adapter = _make_adapter(model, cfg, device, PROTOCOL, dataset, args.recipe)
                if model == 'mirepnet':
                    x_train = _prepare_mirepnet(x_tr, sub_tr, adapter)
                    x_test = _prepare_mirepnet(
                        x_te, np.full(len(y_te), fold, dtype=np.int64), adapter)
                else:
                    x_train, x_test = x_tr, x_te
                model_instance = adapter.build(len(ds['classes']))
                started = time.time()
                if model == 'mirepnet':
                    model_instance, history = _train_mirepnet(
                        adapter, model_instance, x_train, y_tr, cfg)
                else:
                    model_instance, history = _train_cbramod(
                        adapter, model_instance, x_train, y_tr, cfg, seed, args.recipe)
                feats, logits = adapter.infer(model_instance, x_test)
                metrics, probs, pred, cm = _metrics(y_te, logits)
                metrics.update(model=model, dataset=dataset, test_subject=fold+1,
                               seed=seed, n_train=len(y_tr), n_test=len(y_te),
                               elapsed_sec=round(time.time()-started, 2),
                               artifact_origin='trained')
                result = {'y': y_te, 'pred': pred, 'probs': probs,
                          'logits': logits, 'feats': feats,
                          'sample_uid': uid_te, 'confusion_matrix': cm,
                          'metrics_json': np.asarray(json.dumps(metrics, sort_keys=True, allow_nan=True))}
                manifest = {
                    'protocol': PROTOCOL, 'dataset': dataset, 'model': model,
                    'recipe': recipe_dirname, 'training_profile': profile_meta,
                    'artifact_origin': 'trained', 'dataset_source': ds['source_dataset'],
                    'source_files': source_file_payload,
                    'source_trial_selection': {'session': ds['selected_session'],
                                               'run': ds.get('selected_run'),
                                               'window': ds['window']},
                    'pretrained_checkpoint': weight_record,
                    'pretrained_sha256': weight_sha, 'model_config': cfg,
                    'resolved_config_sha256': _json_hash(cfg),
                    'environment_snapshot': _environment_snapshot(device),
                    'num_classes': len(ds['classes']), 'label_values': ds['classes'],
                    'test_subject': fold+1,
                    'train_subjects': sorted(np.unique(sub_tr).astype(int).tolist()),
                    'seed': seed, 'n_train': len(y_tr), 'n_test': len(y_te),
                    'train_sample_uids': uid_tr.tolist(),
                    'test_sample_uids': uid_te.tolist(),
                    'window': split_manifest['window'],
                    'selection_policy': 'fixed_final_epoch_no_validation',
                    'ea_policy': ('per_source_subject; held-out subject covariance '
                                  'uses all unlabeled held-out trials (transductive)'
                                  if model == 'mirepnet' else 'disabled'),
                    'test_metrics': metrics,
                }
                _save_cell(rp, cp, mp, hp, model_instance, result, manifest, history)
                print(f'[done] {model} {dataset} S{fold+1} seed={seed} '
                      f'acc={metrics["accuracy"]:.4f} '
                      f'bacc={metrics["balanced_accuracy"]:.4f} '
                      f'kappa={metrics["kappa"]:.4f}', flush=True)
                del model_instance, adapter, x_train, x_test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del x_tr, y_tr, x_te, y_te, sub_tr, uid_tr, uid_te, local_uid_te
        _summary(output_dir, dataset, model, recipe_dirname,
                 split_manifest['seeds'], n_subjects, profile_meta)
        del x, y, subject_ids, source_uids, local_uids, records, meta, labels_raw
        gc.collect()
        print(f'[summary] {output_dir/"summary.csv"}', flush=True)


def main():
    global SPEC
    args = _parse_args()
    SPEC, snapshot = _load_spec()
    if args.gpu is not None and not torch.cuda.is_available():
        raise RuntimeError('--gpu specified but CUDA is unavailable')
    device = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    if args.preflight_only:
        if args.recipe == 'eegfm_full' and args.model != 'cbramod':
            raise ValueError('eegfm_full recipe is only defined for CBraMod')
        weight = Path(config.weight_path(args.model)).resolve()
        if _sha256(weight) != snapshot['pretrained_weights'][args.model]['sha256']:
            raise RuntimeError('pretrained checkpoint differs from the frozen source snapshot')
        for dataset in args.datasets:
            _verify_source_files(dataset, SPEC, snapshot)
        _run_preflight(args.model, args.datasets, args.recipe, device, SPEC)
        print('[preflight-complete] no formal result cells written', flush=True)
        return
    _run(args, SPEC, snapshot, device)


if __name__ == '__main__':
    main()
