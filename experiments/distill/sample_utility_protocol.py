"""Data, manifests, schedules, and durable storage for the sample utility pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

import config
from experiments.finetune import run_loso_five_datasets as canonical
from experiments.finetune import run_loso_small_baselines as small_baselines
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
                                 require_external_output, resolve_local_file)
from experiments.distill.sample_utility_splits import fold_masks
from models import get_adapter


PROTOCOL_ID = 'sample_utility_adaptive_rl_loso_pilot_v1'
CONFIG_PATH = Path('configs/experiments/sample_utility_adaptive_rl_loso_pilot.yaml')
CONFIG_SHA = None
CACHE_ROOT = require_external_output(DATA_CACHE_ROOT / PROTOCOL_ID / 'BNCI2014004')
WEIGHT_ROOT = require_external_output(WEIGHTS_ROOT / PROTOCOL_ID)
RESULT_ROOT = require_external_output(RESULTS_ROOT / 'distill' / PROTOCOL_ID)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_uids(uids) -> str:
    arr = np.ascontiguousarray(np.asarray(uids, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _require_values(section: str, actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f'pilot {section}.{key} must be {value!r}, got {actual.get(key)!r}')


def _validate_runtime_settings(raw: dict, student_cfg: dict, teacher_cfg: dict) -> None:
    _require_values('loss', raw.get('loss', {}), {
        'ce_weight': 1.0, 'lam_kd': 0.5, 'temperature': 2.0,
        'lam_feature': 0.0, 'kd_denominator': 'actual_batch_size',
    })
    _require_values('controller', raw.get('controller', {}), {
        'hidden_dims': [128, 64], 'activation': 'silu',
        'output_min': 0.05, 'output_max': 0.95,
        'optimizer': 'adam', 'lr': 0.001, 'weight_decay': 0.0,
        'betas': [0.9, 0.999], 'eps': 1e-8, 'scheduler': 'none',
        'update_every_student_batches': 1, 'feedback_batch_size': 16,
        'entropy_coefficient': 0.0, 'gradient_clip': None,
    })
    _require_values('rl', raw.get('rl', {}), {
        'baseline_initial': 0.0, 'baseline_ema_decay': 0.9,
        'advantage_normalization': 'none', 'log_prob_reduction': 'sum',
    })
    _require_values('execution', raw.get('execution', {}), {
        'precision': 'fp32', 'amp': False,
        'student_optimizer_foreach': False, 'student_optimizer_fused': False,
        'gradient_accumulation': 1, 'checkpoint_every_epochs': 1, 'num_workers': 0,
    })
    _require_values('report', raw.get('report', {}), {
        'primary_metric': 'balanced_accuracy', 'bootstrap_draws': 10000,
        'bootstrap_seed': 666,
        'primary_pvalue_family': ['adaptive_vs_kd_all', 'rl_vs_kd_all'],
        'primary_pvalue_correction': 'holm',
    })
    _require_values('student_config', student_cfg, {
        'batch_size': 16, 'epochs': 100, 'lr': 0.001, 'weight_decay': 0.01,
    })
    _require_values('teacher_config', teacher_cfg, {
        'batch_size': 8, 'epochs': 10, 'lr': 0.001,
        'weight_decay': 1e-6, 'optimizer': 'adam',
    })


def atomic_json(path: str | Path, value) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, default=str, allow_nan=True) + '\n')
    os.replace(temp, path)


def atomic_torch(path: str | Path, value) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_npz(path: str | Path, **arrays) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.stem}.tmp-{os.getpid()}{path.suffix}')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def atomic_csv(path: str | Path, rows: list[dict]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_resolved_config(path: str | Path, device: torch.device):
    global CONFIG_SHA
    path = Path(path)
    if not path.is_absolute():
        path = canonical.ROOT / path
    raw = yaml.safe_load(resolve_local_file(path).read_text())
    expected = {
        'protocol_id': PROTOCOL_ID, 'dataset': 'BNCI2014004',
        'teacher': 'mirepnet', 'student': 'ifnet', 'seed': 666,
        'epochs': 100, 'warmup_epochs': 10,
        'conditions': ['BASE_CE', 'DELAYED_KD_ALL', 'ADAPTIVE_WEIGHT_KD',
                       'RL_GATE_KD', 'ADAPTIVE_SHUFFLE', 'RL_SHUFFLE'],
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f'pilot config {key} must be {value!r}, got {raw.get(key)!r}')
    # The repository's canonical source manifest predates its current YAML and
    # fails canonical._load_spec's strict YAML digest check. Keep that historical
    # file untouched; freeze both hashes in this pilot and validate the exact
    # dataset fields and all referenced source-file hashes below.
    spec_path = canonical.SPEC_PATH
    snapshot_path = canonical.SNAPSHOT_PATH
    spec = yaml.safe_load(resolve_local_file(spec_path).read_text())
    snapshot = json.loads(resolve_local_file(snapshot_path).read_text())
    ds = spec['datasets']['BNCI2014004']
    locked = {'source_dataset': 'BNCI2014004', 'selected_session': 'session_3',
              'loader_data_mode': 'session3', 'subjects': 9,
              'selected_trials': 1400, 'native_channels': 3,
              'canonical_fs_hz': 250, 'per_subject_trials':
              [160, 120, 160, 160, 160, 160, 160, 160, 160],
              'classes': {'left_hand': 0, 'right_hand': 1}}
    for key, expected_value in locked.items():
        if ds.get(key) != expected_value:
            raise RuntimeError(f'canonical BNCI2014004 {key} changed: {ds.get(key)!r}')
    student_cfg = small_baselines._config_for('ifnet', 'BNCI2014004', ds)
    teacher_cfg = canonical._config('mirepnet', 'BNCI2014004', 'main', ds, spec)
    _validate_runtime_settings(raw, student_cfg, teacher_cfg)
    source_hashes = canonical._verify_source_files('BNCI2014004', spec, snapshot)
    pretrained_path = Path(config.weight_path('mirepnet')).resolve()
    pretrained_sha = sha256_file(pretrained_path)
    expected_teacher_sha = snapshot['pretrained_weights']['mirepnet']['sha256']
    if pretrained_sha != expected_teacher_sha:
        raise RuntimeError('MIRepNet pretraining file differs from the frozen source snapshot')
    resolved = dict(raw)
    resolved['canonical_spec_sha256'] = sha256_file(spec_path)
    resolved['canonical_manifest_recorded_spec_sha256'] = snapshot['spec_sha256']
    resolved['canonical_manifest_spec_hash_matches'] = (
        resolved['canonical_spec_sha256'] == snapshot['spec_sha256'])
    resolved['canonical_source_snapshot_sha256'] = sha256_file(snapshot_path)
    resolved['trial_manifest_sha256'] = sha256_file(canonical.TRIALS_PATH)
    resolved['source_hashes'] = source_hashes
    resolved['pretrained_checkpoint'] = {'path': str(pretrained_path), 'sha256': pretrained_sha}
    resolved['student_config'] = student_cfg
    resolved['teacher_config'] = teacher_cfg
    resolved['optimizer_defaults'] = {
        'student': {'betas': (0.9, 0.999), 'eps': 1e-8,
                    'amsgrad': False, 'foreach': False, 'fused': False},
        'controller': {'betas': (0.9, 0.999), 'eps': 1e-8, 'amsgrad': False},
    }
    resolved['torch'] = str(torch.__version__)
    CONFIG_SHA = hash_json(resolved)
    resolved['resolved_config_sha256'] = CONFIG_SHA
    resolved['runtime_device'] = str(device)
    return resolved, spec, snapshot


def load_data(spec, snapshot, verified_source_hashes=None):
    source_hashes = (canonical._verify_source_files('BNCI2014004', spec, snapshot)
                     if verified_source_hashes is None else verified_source_hashes)
    x, y, subjects, uids, local_uids, records, meta, raw_labels = canonical._load_trials(
        'BNCI2014004', spec)
    expected = spec['datasets']['BNCI2014004']['per_subject_trials']
    counts = [int(np.sum(subjects == sid)) for sid in range(9)]
    if counts != list(expected) or len(y) != 1400 or x.shape != (1400, 3, 1000):
        raise RuntimeError(f'canonical BNCI2014004 data changed: shape={x.shape}, counts={counts}')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise RuntimeError('EEG source contains non-finite values or unsupported labels')
    return x, y, subjects, uids, local_uids, records, meta, raw_labels, source_hashes


def split_fold(x, y, subjects, uids, local_uids, records, fold: int):
    if not 0 <= int(fold) < 9:
        raise ValueError(f'fold must be in [0,8], got {fold}')
    feedback, masks = fold_masks(subjects, fold)
    uidsets = {name: {tuple(row) for row in np.asarray(uids[mask]).tolist()}
               for name, mask in masks.items()}
    if any(uidsets[a] & uidsets[b] for a, b in (('train', 'feedback'),
                                                  ('train', 'test'),
                                                  ('feedback', 'test'))):
        raise RuntimeError('train, feedback, and test UIDs overlap')
    if set.union(*uidsets.values()) != {tuple(row) for row in uids.tolist()}:
        raise RuntimeError('split UIDs do not cover the selected dataset')
    result = {}
    for name, mask in masks.items():
        result[name] = {'x': x[mask], 'y': y[mask], 'uids': uids[mask],
                        'local_uids': local_uids[mask],
                        'subjects': subjects[mask],
                        'records': [records[i] for i in np.flatnonzero(mask)]}
    if len(result['train']['y']) not in (1080, 1120):
        raise RuntimeError(f'unexpected D_train size for fold {fold}: {len(result["train"]["y"])}')
    return feedback, result


def fold_paths(fold: int, feedback: int, split_sha: str):
    name = f'target_{fold + 1:02d}_feedback_{feedback + 1:02d}_seed_666_{split_sha[:16]}'
    cache = CACHE_ROOT / name
    weights = WEIGHT_ROOT / name
    results = RESULT_ROOT / 'folds' / name
    for path in (cache, weights, results):
        require_external_output(path)
    return {'key': name, 'cache': cache, 'weights': weights, 'results': results}


def split_manifest(fold: int, feedback: int, split, source_hashes, resolved):
    payload = {
        'protocol_id': PROTOCOL_ID, 'dataset': 'BNCI2014004', 'seed': 666,
        'target_subject_index_zero_based': int(fold), 'target_subject_id': int(fold) + 1,
        'feedback_subject_index_zero_based': int(feedback), 'feedback_subject_id': int(feedback) + 1,
        'train_subject_ids': sorted(np.unique(split['train']['subjects']).astype(int).tolist()),
        'selected_session': 'session_3', 'canonical_window_samples': 1000,
        'class_mapping': {'left_hand': 0, 'right_hand': 1},
        'source_hashes': source_hashes,
        'trial_manifest_sha256': resolved['trial_manifest_sha256'],
        'preprocessing': {'teacher': 'bandpass_8_30Hz_then_per_source_subject_EA_then_IDW45',
                          'student': 'IFNet_filterbank_4_16Hz_16_40Hz'},
        'split_counts': {name: int(len(row['y'])) for name, row in split.items()},
        'split_uid_sha256': {name: hash_uids(row['uids']) for name, row in split.items()},
        'split_uids': {name: np.asarray(row['uids'], dtype=np.int64).tolist()
                       for name, row in split.items()},
        'split_labels': {name: np.asarray(row['y'], dtype=np.int64).tolist()
                         for name, row in split.items()},
        'resolved_config_sha256': resolved['resolved_config_sha256'],
    }
    payload['split_sha256'] = hash_json({k: v for k, v in payload.items()
                                         if k != 'split_sha256'})
    return payload


def epoch_schedule(n: int, batch_size: int, epochs: int, seed: int):
    generator = np.random.default_rng(int(seed))
    schedule = []
    for _ in range(epochs):
        order = generator.permutation(n).astype(np.int64)
        schedule.append([order[start:start + batch_size]
                         for start in range(0, n, batch_size)])
    return schedule


def feedback_schedule(n: int, batch_size: int, steps: int, seed: int):
    """Cycle complete feedback-set permutations, retaining each tail batch."""
    generator = np.random.default_rng(int(seed))
    result = []
    while len(result) < steps:
        order = generator.permutation(n).astype(np.int64)
        result.extend(order[start:start + batch_size]
                      for start in range(0, n, batch_size))
    return result[:steps]


def save_schedule(path: Path, schedule):
    flat, offsets = [], [0]
    for epoch in schedule:
        for batch in epoch:
            flat.extend(np.asarray(batch, dtype=np.int64).tolist())
            offsets.append(len(flat))
    atomic_npz(path, indices=np.asarray(flat, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64))


def seed_all(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_teacher(split, resolved, spec, device):
    cfg = dict(resolved['teacher_config'])
    adapter = get_adapter('mirepnet', device=device, **cfg)
    x_prepared = canonical._prepare_mirepnet(split['train']['x'],
                                             split['train']['subjects'], adapter)
    x_tensor = adapter.preprocess(x_prepared)
    y_tensor = torch.as_tensor(split['train']['y'], dtype=torch.long)
    teacher_seed = int(resolved['seed'])
    torch_gen = torch.Generator().manual_seed(teacher_seed)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                        batch_size=int(cfg['batch_size']),
                        shuffle=True, generator=torch_gen, num_workers=0,
                        drop_last=False)
    model = adapter.build(2)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                                 weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg['epochs']))
    history, start = [], time.monotonic()
    for epoch in range(int(cfg['epochs'])):
        model.train()
        losses, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward(model, xb)[1]
            loss = torch.nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            losses += float(loss.detach()) * len(yb)
            seen += len(yb)
        scheduler.step()
        history.append({'epoch': epoch + 1, 'loss': losses / seen,
                        'lr_next': optimizer.param_groups[0]['lr']})
    model.eval()
    features, logits = [], []
    with torch.no_grad():
        for start_idx in range(0, len(x_tensor), 16):
            feat, lg = adapter.forward(model, x_tensor[start_idx:start_idx + 16].to(device))
            features.append(feat.cpu().numpy())
            logits.append(lg.cpu().numpy())
    cache = {'feats': np.concatenate(features).astype(np.float32),
             'logits': np.concatenate(logits).astype(np.float32),
             'y': np.asarray(split['train']['y'], dtype=np.int64),
             'uids': np.asarray(split['train']['uids'], dtype=np.int64)}
    if cache['feats'].shape[0] != len(cache['uids']) or not np.isfinite(cache['logits']).all():
        raise RuntimeError('Teacher train cache has invalid shape or values')
    provenance = {
        'model': 'mirepnet', 'seed': teacher_seed, 'epochs': int(cfg['epochs']), 'config': cfg,
        'initial_checkpoint_sha256': resolved['pretrained_checkpoint']['sha256'],
        'train_uid_sha256': hash_uids(cache['uids']),
        'train_subject_ids': sorted(np.unique(split['train']['subjects']).astype(int).tolist()),
        'feedback_subject_id': int(np.unique(split['feedback']['subjects'])[0]) + 1,
        'target_subject_id': int(np.unique(split['test']['subjects'])[0]) + 1,
        'training_history': history, 'elapsed_seconds': time.monotonic() - start,
        'feature_dim': int(cache['feats'].shape[1]), 'logit_dim': int(cache['logits'].shape[1]),
    }
    return model, cache, provenance


def save_teacher(path: Path, model, cache, provenance):
    atomic_torch(path / 'teacher_final.pt', {
        'model': model.state_dict(), 'provenance': provenance,
    })
    atomic_npz(path / 'teacher_train.npz', **cache)
    atomic_json(path / 'teacher_provenance.json', provenance)


def validate_teacher(path: Path, train_uids, train_labels=None):
    ckpt = path / 'teacher_final.pt'
    cache_path = path / 'teacher_train.npz'
    provenance_path = path / 'teacher_provenance.json'
    if not all(item.is_file() for item in (ckpt, cache_path, provenance_path)):
        return None
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('train_uid_sha256') != hash_uids(train_uids):
        raise RuntimeError(f'Teacher cache belongs to another seven-source split: {path}')
    with np.load(cache_path, allow_pickle=False) as data:
        cached = {key: np.asarray(data[key]) for key in data.files}
    if not np.array_equal(cached['uids'], train_uids):
        raise RuntimeError(f'Teacher cache UID order mismatch: {path}')
    if train_labels is not None and not np.array_equal(cached['y'], train_labels):
        raise RuntimeError(f'Teacher cache labels do not match its train UIDs: {path}')
    if (cached['feats'].shape[0] != len(train_uids)
            or cached['logits'].shape != (len(train_uids), 2)):
        raise RuntimeError(f'Teacher cache shape does not match D_train: {path}')
    if not np.isfinite(cached['feats']).all() or not np.isfinite(cached['logits']).all():
        raise RuntimeError(f'Teacher cache contains NaN/Inf: {path}')
    return cached, provenance, torch.load(resolve_local_file(ckpt), map_location='cpu')
