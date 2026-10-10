#!/usr/bin/env python
"""Current broadband 001/001-4 LOSO teachers with resumable fold checkpoints.

The worker deliberately accepts physical CUDA indices and refuses GPU 0. A
formal worker owns one (dataset, model, seed) stream and processes LOSO subjects
in sorted order. Historical alignment/source-bridge CLI profiles are retired;
the current protocol reads formal model LOSO configurations.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.storage import (BNCI14001_SOURCE_ROOT, DATA_CACHE_ROOT, RESULTS_ROOT,
                                 require_external_output, resolve_local_file)
SPEC = ROOT / 'configs/protocols/loso_001.yaml'
INPUT_ROOT = DATA_CACHE_ROOT / 'eegfm_alignment_v2/model_inputs'
RESULT_ROOT = RESULTS_ROOT / 'reproductions/loso_config_alignment_v2'
REFERENCE_ROOT = Path('/home/lixinli/EEG-FM-Benchmark')
WEIGHT_PATH = Path('/data1/llx/pre_weight/cbramod.pth')
MIREPNET_WEIGHT_PATH = Path('/data1/llx/pre_weight/mirepnet.pth')
WIDEBAND_PROFILE = 'wideband_npy_v3'
WIDEBAND_VARIANT = 'all_sessions_source_train_session'
DATASET_ALIASES = {
    '001': 'BNCI2014001', '001-4': 'BNCI2014001-4',
    '004': 'BNCI2014004', '5001': 'BNCI2015001',
    'BNCI2014001': 'BNCI2014001',
    'BNCI2014001-4': 'BNCI2014001-4', 'BNCI2014004': 'BNCI2014004',
    'BNCI2015001': 'BNCI2015001',
}


class ReferenceEEGNet(nn.Module):
    """The EEG-FM-Benchmark EEGNet architecture and asymmetric padding."""

    def __init__(self, n_classes: int, chans: int, samples: int, dropout: float):
        super().__init__()
        f1, depth, f2, temporal = 8, 2, 16, 64
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((temporal // 2 - 1, temporal - temporal // 2, 0, 0)),
            nn.Conv2d(1, f1, (1, temporal), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth), nn.ELU(), nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16),
                      groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (samples // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x.unsqueeze(1))
        x = self.block2(x)
        return self.classifier(x.flatten(start_dim=1))


def read_json(path: Path) -> dict:
    return json.loads(resolve_local_file(path).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with resolve_local_file(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_dict_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode())
        h.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


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


def capture_rng(device: torch.device | None = None) -> dict:
    current_cuda = (torch.cuda.get_rng_state(device).cpu()
                    if torch.cuda.is_available() and device is not None else None)
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'torch_cuda_current': current_cuda,
    }


def restore_rng(state: dict, device: torch.device | None = None) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if (torch.cuda.is_available() and device is not None
            and state.get('torch_cuda_current') is not None):
        torch.cuda.set_rng_state(state['torch_cuda_current'].cpu(), device=device)
    elif torch.cuda.is_available() and state.get('torch_cuda'):
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def canonical_dataset(value: str) -> str:
    try:
        return DATASET_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f'Unsupported dataset: {value}') from exc


def profile_config(profile: str, model: str, dataset: str, spec: dict) -> dict:
    if profile != WIDEBAND_PROFILE:
        raise ValueError(
            f'{profile} is retired. For 001/001-4 use wideband_npy_v3 and '
            'configs/protocols/loso_001.yaml; for 004/5001 use '
            'experiments/finetune/run_loso_source_refresh_004_5001.py.')
    if profile == WIDEBAND_PROFILE:
        if dataset not in ('BNCI2014001', 'BNCI2014001-4'):
            raise ValueError(f'{profile} supports 001 and 001-4 only')
        if model == 'mirepnet':
            import config
            from experiments.finetune import prepare_loso_alignment_inputs as inputs
            c = config.load_model_config(model, dataset, 'loso')
            inputs.validate_001_dataset(
                dataset, BNCI14001_SOURCE_ROOT, INPUT_ROOT / WIDEBAND_PROFILE,
                RESULT_ROOT / WIDEBAND_PROFILE, spec)
            inputs.validate_001_model(model, dataset, c)
            return {
                'optimizer': c['optimizer'], 'lr': float(c['lr']),
                'weight_decay': float(c['weight_decay']),
                'batch_size': int(c['batch_size']), 'epochs': int(c['epochs']),
                # MIRepNet's native architecture owns dropout; make_model does
                # not override its 0.5 embedding/transformer dropout.
                'dropout': 0.5, 'dropout_policy': 'native_architecture_unchanged',
                'class_weights': False, 'label_smoothing': 0.0,
                'min_lr': 0.0, 'warmup_epochs': 0,
                'lr_schedule': 'epoch_cosine', 'duration_seconds': 4.0,
                'parameter_source': 'configs/models/mirepnet.yaml::' + dataset + '.loso',
            }
        if model != 'cbramod':
            raise ValueError(f'{profile} supports MIRepNet and CBraMod only')
        import config
        from experiments.finetune import prepare_loso_alignment_inputs as inputs
        c = config.load_model_config(model, dataset, 'loso')
        inputs.validate_001_dataset(
            dataset, BNCI14001_SOURCE_ROOT, INPUT_ROOT / WIDEBAND_PROFILE,
            RESULT_ROOT / WIDEBAND_PROFILE, spec)
        inputs.validate_001_model(model, dataset, c)
        # These two provenance labels are retained in the resolved dictionary
        # to keep completed checkpoint fingerprints valid. Numeric values
        # are now read only from the formal model YAML, as documented by SPEC.
        return {
            'optimizer': c['optimizer'], 'lr': float(c['lr']),
            'weight_decay': float(c['weight_decay']),
            'batch_size': int(c['batch_size']), 'epochs': int(c['epochs']),
            'dropout': float(c['dropout']),
            'class_weights': bool(c['class_weights']),
            'label_smoothing': float(c['label_smoothing']),
            'min_lr': float(c['min_lr']), 'warmup_epochs': int(c['warmup_epochs']),
            'lr_schedule': c['lr_schedule'],
            'duration_seconds': float(c['duration_seconds']),
            'parameter_source': 'reference_aligned.cbramod.BNCI2014001-4',
            'parameter_transfer': (
                'four_class_recipe_transferred_to_binary_loso'
                if dataset == 'BNCI2014001' else 'same_four_class_recipe'),
        }


def input_dir(profile: str, dataset: str, model: str, variant: str) -> Path:
    return INPUT_ROOT / profile / dataset / model / variant


def load_fold_data(profile: str, dataset: str, model: str, variant: str):
    folder = input_dir(profile, dataset, model, variant)
    required = ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv', 'manifest.json')
    missing = [n for n in required if not (folder / n).is_file()]
    if missing:
        raise FileNotFoundError(f'{folder}: missing prepared inputs {missing}')
    manifest = read_json(folder / 'manifest.json')
    x = np.load(resolve_local_file(folder / 'X.npy'), mmap_mode='r')
    y = np.load(resolve_local_file(folder / 'y.npy'), mmap_mode='r')
    subjects = np.load(resolve_local_file(folder / 'subjects.npy'), mmap_mode='r')
    import pandas as pd
    trials = pd.read_csv(resolve_local_file(folder / 'trials.csv'))
    if not (len(x) == len(y) == len(subjects) == len(trials)):
        raise RuntimeError(f'{folder}: sample arrays and trial manifest differ')
    if not np.array_equal(trials.label_id.to_numpy(), y):
        raise RuntimeError(f'{folder}: trial labels differ from y.npy')
    if not np.array_equal(trials.subject_zero_based.to_numpy(), subjects):
        raise RuntimeError(f'{folder}: trial subjects differ from subjects.npy')
    if trials.trial_uid.duplicated().any():
        raise RuntimeError(f'{folder}: trial_uid is not unique')
    if profile == WIDEBAND_PROFILE:
        expected_classes = 2 if dataset == 'BNCI2014001' else 4
        expected_shape = (45, 1000) if model == 'mirepnet' else (22, 4, 200)
        if tuple(x.shape[1:]) != expected_shape:
            raise RuntimeError(f'{folder}: expected input shape {expected_shape}, got {x.shape[1:]}')
        if not np.array_equal(np.unique(y), np.arange(expected_classes)):
            raise RuntimeError(f'{folder}: expected {expected_classes} contiguous classes')
        if not np.array_equal(np.unique(subjects), np.arange(9)):
            raise RuntimeError(f'{folder}: expected all nine zero-based subjects')
    return x, np.asarray(y), np.asarray(subjects), trials, manifest


def make_model(model: str, dataset: str, n_classes: int, input_shape: tuple,
               dropout: float, profile: str) -> nn.Module:
    if model == 'mirepnet':
        if profile != WIDEBAND_PROFILE:
            raise ValueError('MIRepNet is supported only by wideband_npy_v3')
        if tuple(input_shape) != (45, 1000):
            raise RuntimeError(f'MIRepNet expects [45,1000], received {input_shape}')
        if not MIREPNET_WEIGHT_PATH.is_file():
            raise FileNotFoundError(f'MIRepNet checkpoint not found: {MIREPNET_WEIGHT_PATH}')
        import config
        from models.mirepnet.adapter import MIRepNetAdapter
        model_cfg = config.load_model_config(model, dataset, 'loso')
        model_cfg.update(dataset_name=dataset, skip_preprocess=True,
                         pretrain=str(MIREPNET_WEIGHT_PATH))
        return MIRepNetAdapter(device='cpu', **model_cfg).build(n_classes)
    if model == 'eegnet':
        if len(input_shape) != 2:
            raise RuntimeError(f'EEGNet expects [C,T], received {input_shape}')
        if profile == 'source_bridge':
            from models.eegnet.residual_eegnet import ResidualEEGNet
            return ResidualEEGNet(in_channels=int(input_shape[0]),
                                  samples=int(input_shape[1]),
                                  num_classes=n_classes)
        return ReferenceEEGNet(n_classes, int(input_shape[0]), int(input_shape[1]), dropout)
    if model == 'cbramod':
        from models.cbramod.adapter import _CBraModModel
        expected_hash = None
        provenance_path = RESULT_ROOT / 'reference_resolution.json'
        if provenance_path.exists():
            provenance = read_json(provenance_path)
            expected_hash = provenance.get('pretrained_checkpoints', {}).get('cbramod', {}).get('sha256')
        if not WEIGHT_PATH.is_file():
            raise FileNotFoundError(f'CBraMod checkpoint not found: {WEIGHT_PATH}')
        actual_hash = sha256_file(WEIGHT_PATH)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f'CBraMod checkpoint hash mismatch: {actual_hash} != {expected_hash}')
        n_ch, n_patch, patch = map(int, input_shape)
        if patch != 200:
            raise RuntimeError(f'CBraMod patch width must be 200, got {input_shape}')
        return _CBraModModel(n_classes, n_ch=n_ch, n_patch=n_patch,
                             dropout=dropout, pretrain=str(WEIGHT_PATH),
                             feature_head='flatten')
    raise ValueError(f'Unsupported model {model}')


def pretrained_checkpoint(model: str) -> Path | None:
    return {'cbramod': WEIGHT_PATH, 'mirepnet': MIREPNET_WEIGHT_PATH}.get(model)


def validate_wideband_checkpoint(saved: dict, identity: dict, path: Path) -> None:
    """Refuse reuse when source inputs or pretrained weights have changed."""
    for key, value in identity.items():
        if saved.get(key) != value:
            raise RuntimeError(f'{path}: {key} changed or is missing from checkpoint')


def optimizer_for(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    kwargs = {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'], 'eps': 1e-8}
    opt = cfg['optimizer'].lower()
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f'Unsupported reference optimizer: {opt}')


def reference_lr_schedule(cfg: dict, n_steps: int) -> np.ndarray:
    epochs = int(cfg['epochs'])
    warmup_steps = min(int(cfg['warmup_epochs']), epochs) * int(n_steps)
    warmup = (np.linspace(0.0, float(cfg['lr']), warmup_steps)
              if warmup_steps > 0 else np.array([], dtype=np.float64))
    cosine_steps = epochs * int(n_steps) - warmup_steps
    cosine = np.array([
        float(cfg['min_lr']) + 0.5 * (float(cfg['lr']) - float(cfg['min_lr']))
        * (1.0 + np.cos(np.pi * i / cosine_steps))
        for i in range(cosine_steps)
    ], dtype=np.float64)
    values = np.concatenate((warmup, cosine))
    expected = epochs * int(n_steps)
    if len(values) != expected:
        raise RuntimeError(f'LR table length {len(values)} != expected {expected}')
    return values


def apply_step_lr(optimizer, schedule: np.ndarray, global_step: int) -> float:
    idx = min(int(global_step), len(schedule) - 1)
    lr = float(schedule[idx])
    for group in optimizer.param_groups:
        group['lr'] = lr * float(group.get('lr_scale', 1.0))
    return lr


def checkpoint_save(path: Path, state: dict) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    torch.save(state, tmp)
    os.replace(tmp, path)


def compact_completed_state(state: dict) -> dict:
    """Keep fold identity and terminal RNG, without training tensor payloads.

    This is also the conversion contract for previously completed full resume
    checkpoints. Preserve every other field, including historical fingerprints.
    """
    if state.get('completed') is not True:
        raise ValueError('Only a completed checkpoint can be compacted')
    if not isinstance(state.get('rng_state'), dict):
        raise ValueError('Completed checkpoint must retain its terminal RNG state')
    return {key: value for key, value in state.items()
            if key not in ('model_state', 'optimizer_state', 'history')}


def completed_fold(output: Path, identity: dict, epochs: int) -> tuple[dict, dict] | None:
    """Validate retained results before skipping a completed fold.

    A compact completion sidecar takes precedence over an obsolete full resume
    state. Incomplete folds still use their complete model/optimizer checkpoint.
    Missing completion provenance must fail explicitly instead of retraining.
    """
    compact_path = output / 'completed_state.pt'
    resume_path = output / 'training_state.pt'
    state_path = compact_path if compact_path.is_file() else resume_path
    result_file = output / 'result.json'
    if not state_path.is_file():
        if result_file.is_file() and read_json(result_file).get('status') == 'complete':
            raise RuntimeError(
                f'{output}: completed result lacks completed_state.pt; refusing retraining')
        return None
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    validate_wideband_checkpoint(state, identity, state_path)
    if state.get('completed') is not True:
        if state_path == compact_path:
            raise RuntimeError(f'{compact_path}: compact state is not completed')
        return None
    if int(state.get('next_epoch', -1)) != int(epochs):
        raise RuntimeError(f'{state_path}: completed epoch count differs from configuration')
    rng = state.get('rng_state')
    if not isinstance(rng, dict) or any(
            key not in rng for key in ('python', 'numpy', 'torch_cpu')):
        raise RuntimeError(f'{state_path}: terminal fold RNG is incomplete')
    if not result_file.is_file():
        raise RuntimeError(f'{output}: completed state lacks result.json')
    result = read_json(result_file)
    expected = {
        'status': 'complete', 'profile': identity['profile'],
        'dataset': identity['dataset'], 'model': identity['model'],
        'seed': identity['seed'], 'held_out_subject': identity['subject'],
        'input_manifest_sha256': identity['input_manifest_sha256'],
        'reference_pretrained_sha256': identity['pretrained_checkpoint_sha256'],
        'input_shape': identity['input_shape'],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f'{result_file}: {key} changed or is missing')
    actual_fingerprint = hashlib.sha256(json.dumps(
        result.get('resolved_training_config'), sort_keys=True).encode()).hexdigest()
    if actual_fingerprint != identity['config_fingerprint']:
        raise RuntimeError(f'{result_file}: training configuration changed')
    if result.get('initial_state_sha256') != state.get('initial_state_sha256'):
        raise RuntimeError(f'{result_file}: initialization differs from completion state')
    if not isinstance(result.get('metrics'), dict):
        raise RuntimeError(f'{result_file}: completed result lacks metrics')
    for name, hash_key in (
            ('final_model.pt', 'final_model_sha256'),
            ('test_predictions.npz', 'test_predictions_sha256')):
        path = output / name
        expected_hash = result.get(hash_key)
        if not path.is_file() or not expected_hash or sha256_file(path) != expected_hash:
            raise RuntimeError(f'{path}: retained completed artifact is missing or changed')
    return result, state


def result_path(base: Path, subject: int) -> Path:
    return base / f'subject_{subject + 1:02d}'


def set_status(folder: Path, status: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / 'status.json.partial'
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, folder / 'status.json')


def train_one_fold(profile: str, dataset: str, model_name: str, seed: int,
                   subject: int, variant: str, x_all: np.ndarray, y_all: np.ndarray,
                   subjects_all: np.ndarray, trials, manifest: dict, cfg: dict,
                   device: torch.device, output: Path, epochs_override: int | None = None,
                   max_batches: int | None = None, stream_resume_rng: bool = True) -> dict:
    out = require_external_output(output)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = np.flatnonzero(subjects_all != subject)
    test_rows = np.flatnonzero(subjects_all == subject)
    if not len(train_rows) or not len(test_rows):
        raise RuntimeError(f'{dataset}: empty train or test fold for subject {subject + 1}')
    expected_subject_trials = {
        'BNCI2014001': [144] * 9,
        'BNCI2014001-4': [288] * 9,
        'BNCI2014004': [160, 120, 160, 160, 160, 160, 160, 160, 160],
        'BNCI2015001': [200] * 12,
    }
    expected_count = expected_subject_trials[dataset][subject]
    if len(test_rows) != expected_count:
        raise RuntimeError(
            f'{dataset} subject {subject + 1}: expected {expected_count} test trials, '
            f'got {len(test_rows)}')
    train_labels = y_all[train_rows].astype(np.int64, copy=False)
    test_labels = y_all[test_rows].astype(np.int64, copy=False)
    classes = np.unique(y_all)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise RuntimeError(f'{dataset}: class IDs are not contiguous: {classes}')
    fold_cfg = dict(cfg)
    if epochs_override is not None:
        fold_cfg['epochs'] = int(epochs_override)
    epochs = int(fold_cfg['epochs'])
    batch_size = int(fold_cfg['batch_size'])
    n_classes = len(classes)
    input_manifest_path = input_dir(profile, dataset, model_name, variant) / 'manifest.json'
    input_manifest_hash = sha256_file(input_manifest_path)
    pretrained_path = pretrained_checkpoint(model_name)
    pretrained_hash = sha256_file(pretrained_path) if pretrained_path else None
    source_identity = ({
        'dataset': dataset, 'model': model_name,
        'input_manifest_sha256': input_manifest_hash,
        'pretrained_checkpoint_sha256': pretrained_hash,
        'input_shape': list(x_all.shape[1:]), 'num_classes': n_classes,
    } if profile == WIDEBAND_PROFILE else {})
    checkpoint_identity = {
        'profile': profile, 'dataset': dataset, 'model': model_name,
        'subject': subject + 1, 'variant': variant, 'seed': seed,
        'config_fingerprint': hashlib.sha256(
            json.dumps(fold_cfg, sort_keys=True).encode()).hexdigest(),
        **source_identity,
    }
    if profile == WIDEBAND_PROFILE and max_batches is None:
        completion = completed_fold(out, checkpoint_identity, epochs)
        if completion is not None:
            final, saved = completion
            if stream_resume_rng:
                restore_rng(saved['rng_state'], device)
            return final
    fold_seed = int(seed)
    if profile == 'source_bridge':
        # Independent-fold seed: both source variants reset to this exact state.
        fold_seed = int(seed)
        set_seed(fold_seed)

    x_train = torch.from_numpy(np.array(x_all[train_rows], dtype=np.float32, copy=True))
    y_train = torch.from_numpy(train_labels.copy()).long()
    row_train = torch.from_numpy(train_rows.astype(np.int64))
    x_test = torch.from_numpy(np.array(x_all[test_rows], dtype=np.float32, copy=True))
    y_test = torch.from_numpy(test_labels.copy()).long()
    train_ds = TensorDataset(x_train, y_train, row_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, drop_last=False)
    steps_per_epoch = len(train_loader)
    lr_table = reference_lr_schedule(fold_cfg, steps_per_epoch) if fold_cfg['lr_schedule'] in (
        'reference_step_table', 'step_table') else None

    # Reference run_finetuning builds one unused model after setting each seed,
    # then process_all_splits_dl builds a fresh model inside each split.
    start_time = time.time()
    set_seed(fold_seed) if profile == 'source_bridge' else None
    model = make_model(model_name, dataset, n_classes, tuple(x_all.shape[1:]),
                       fold_cfg['dropout'], profile)
    init_hash = state_dict_sha256(model.state_dict())
    model.to(device)
    optimizer = optimizer_for(model, fold_cfg)
    if fold_cfg['class_weights']:
        counts = np.bincount(train_labels, minlength=n_classes)
        class_weight = len(train_labels) / (n_classes * counts.astype(np.float64))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weight, dtype=torch.float32, device=device),
            label_smoothing=float(fold_cfg['label_smoothing']))
    else:
        class_weight = None
        criterion = nn.CrossEntropyLoss(label_smoothing=float(fold_cfg['label_smoothing']))

    if max_batches is not None:
        model.train()
        losses = []
        batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[-1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
            batches += 1
            if batches >= max_batches:
                break
        return {
            'preflight_status': 'passed', 'profile': profile,
            'dataset': dataset, 'model': model_name,
            'input_shape': list(x_all.shape[1:]), 'fold_subject': subject + 1,
            'train_trials': len(train_rows), 'test_trials': len(test_rows),
            'batch_size': batch_size, 'batches_run': batches,
            'initial_state_sha256': init_hash,
            'loss_last_batch': losses[-1],
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'pretrained_hash': pretrained_hash,
            'pretrained_checkpoint': str(pretrained_path) if pretrained_path else None,
            'input_manifest_sha256': input_manifest_hash,
            'gpu': device.index,
        }

    ckpt_path = out / 'training_state.pt'
    history_path = out / 'history.csv'
    history: list[dict] = []
    global_step = 0
    first_epoch = 0
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if saved.get('profile') != profile or saved.get('subject') != subject + 1:
            raise RuntimeError(f'{ckpt_path}: checkpoint identity mismatch')
        if saved.get('variant') != variant or saved.get('seed') != seed:
            raise RuntimeError(f'{ckpt_path}: source variant/seed mismatch')
        if source_identity:
            validate_wideband_checkpoint(saved, source_identity, ckpt_path)
        if saved.get('config_fingerprint') != hashlib.sha256(
                json.dumps(fold_cfg, sort_keys=True).encode()).hexdigest():
            raise RuntimeError(f'{ckpt_path}: training configuration changed')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        history = saved.get('history', [])
        init_hash = saved.get('initial_state_sha256', init_hash)
        global_step = int(saved.get('global_step', 0))
        first_epoch = int(saved.get('next_epoch', 0))
        restore_rng(saved['rng_state'], device)
        if saved.get('completed'):
            final = read_json(out / 'result.json')
            return final
    else:
        # Initial RNG checkpoint is after model construction, as with the
        # reference's per-split import_model call.
        checkpoint_save(ckpt_path, {
            'profile': profile, 'dataset': dataset, 'model': model_name,
            'subject': subject + 1, 'variant': variant, 'seed': seed,
            'config_fingerprint': hashlib.sha256(json.dumps(fold_cfg, sort_keys=True).encode()).hexdigest(),
            'next_epoch': 0, 'global_step': 0, 'history': [], 'completed': False,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'rng_state': capture_rng(device), 'initial_state_sha256': init_hash,
            **source_identity,
        })

    class_counts = np.bincount(train_labels, minlength=n_classes).tolist()
    set_status(out, {
        'status': 'running', 'profile': profile, 'dataset': dataset,
        'model': model_name, 'seed': seed, 'fold_seed': fold_seed,
        'held_out_subject': subject + 1, 'variant': variant,
        'epoch': first_epoch, 'epochs': epochs, 'train_trials': len(train_rows),
        'test_trials': len(test_rows), 'train_class_counts': class_counts,
        'input_manifest_sha256': input_manifest_hash,
        'started_or_resumed_utc_epoch_s': time.time(),
        'gpu_physical_index': device.index,
    })
    start_epoch_time = time.time()
    first_batch_ids = None
    for epoch in range(first_epoch, epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_lr = float(optimizer.param_groups[0]['lr'])
        order_rows = []
        for bidx, (xb, yb, global_rows) in enumerate(train_loader):
            order_rows.extend(global_rows.numpy().tolist())
            xb, yb = xb.to(device, non_blocking=False), yb.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[-1]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(yb)
            seen += len(yb)
            global_step += 1
        if seen != len(train_rows):
            raise RuntimeError(f'epoch {epoch + 1}: saw {seen}/{len(train_rows)} training trials')
        if first_batch_ids is None:
            first_batch_ids = order_rows[:min(64, len(order_rows))]
        if fold_cfg['lr_schedule'] in ('reference_step_table', 'step_table'):
            # The benchmark applies this at epoch end, indexing by cumulative
            # global_step (this intentionally preserves its epoch/step mismatch).
            next_lr = apply_step_lr(optimizer, lr_table, global_step)
        elif fold_cfg['lr_schedule'] == 'epoch_cosine':
            next_lr = float(fold_cfg['lr']) * 0.5 * (
                1.0 + np.cos(np.pi * (epoch + 1) / epochs))
            for group in optimizer.param_groups:
                group['lr'] = float(next_lr)
        else:
            raise RuntimeError(f"unknown LR schedule {fold_cfg['lr_schedule']}")

        # In the reference, evaluate_model creates an iterator over the
        # sequential test loader after each epoch. With num_workers=0 that
        # consumes one global torch RNG draw for DataLoader's base seed. Match
        # that draw while keeping the requested final-only test evaluation.
        test_iterator = iter(test_loader)
        del test_iterator
        row_hash = hashlib.sha256(np.asarray(order_rows, dtype=np.int64).tobytes()).hexdigest()
        history.append({
            'epoch': epoch + 1, 'train_loss': running_loss / seen,
            'lr_used': epoch_lr, 'lr_after_reference_update': float(next_lr),
            'global_step': global_step, 'train_trials_seen': seen,
            'train_order_sha256': row_hash,
            'epoch_seconds': time.time() - start_epoch_time,
        })
        with history_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        checkpoint_save(ckpt_path, {
            'profile': profile, 'dataset': dataset, 'model': model_name,
            'subject': subject + 1, 'variant': variant, 'seed': seed,
            'config_fingerprint': hashlib.sha256(json.dumps(fold_cfg, sort_keys=True).encode()).hexdigest(),
            'next_epoch': epoch + 1, 'global_step': global_step,
            'history': history, 'completed': False,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'rng_state': capture_rng(device), 'initial_state_sha256': init_hash,
            'train_order_first_epoch_sha256': history[0]['train_order_sha256'],
            **source_identity,
        })
        set_status(out, {
            'status': 'running', 'profile': profile, 'dataset': dataset,
            'model': model_name, 'seed': seed, 'held_out_subject': subject + 1,
            'variant': variant, 'epoch': epoch + 1, 'epochs': epochs,
            'train_trials': len(train_rows), 'test_trials': len(test_rows),
            'last_epoch_seconds': history[-1]['epoch_seconds'],
            'elapsed_seconds': time.time() - start_time,
            'gpu_physical_index': device.index,
        })
        start_epoch_time = time.time()

    # One final evaluation, after the fixed final epoch only.
    model.eval()
    logits_list = []
    y_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[-1]
            logits_list.append(logits.float().cpu())
            y_list.append(yb)
    logits_np = torch.cat(logits_list).numpy()
    y_true = torch.cat(y_list).numpy()
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    pred = probs.argmax(axis=1)
    metrics = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'kappa': float(cohen_kappa_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, pred, labels=np.arange(n_classes)).tolist(),
    }
    trial_uids = trials.trial_uid.astype(str).to_numpy()
    np.savez_compressed(out / 'test_predictions.npz',
                        row_ids=test_rows, trial_uids=trial_uids[test_rows],
                        labels=y_true, logits=logits_np, probabilities=probs,
                        predictions=pred)
    model_file = out / 'final_model.pt'
    checkpoint_save(model_file, model.state_dict())
    reference_files = [
        REFERENCE_ROOT / 'models/DL/EEGNet/Model_EEGNet.py',
        REFERENCE_ROOT / 'models/DL/EEGNet/Loader_EEGNet.py',
        REFERENCE_ROOT / 'utils/trainer.py',
        REFERENCE_ROOT / 'utils/optimizer.py',
        ROOT / 'models/cbramod/adapter.py', ROOT / 'experiments/finetune/run_loso_config_alignment.py',
        ROOT / 'models/eegnet/residual_eegnet.py',
    ]
    if model_name == 'mirepnet':
        reference_files.extend([
            ROOT / 'models/mirepnet/adapter.py', ROOT / 'models/mirepnet/mlm.py',
            ROOT / 'configs/models/mirepnet.yaml',
        ])
    result = {
        'status': 'complete', 'profile': profile, 'dataset': dataset,
        'model': model_name, 'seed': seed, 'fold_seed': fold_seed,
        'held_out_subject': subject + 1,
        'train_trials': len(train_rows), 'test_trials': len(test_rows),
        'train_class_counts': class_counts, 'test_class_counts': np.bincount(test_labels, minlength=n_classes).tolist(),
        'metrics': metrics, 'input_shape': list(x_all.shape[1:]),
        'input_manifest_sha256': input_manifest_hash,
        'input_manifest': str(input_manifest_path),
        'input_files': manifest.get('files'),
        'class_weights': None if class_weight is None else class_weight.tolist(),
        'resolved_training_config': fold_cfg,
        'optimizer_param_groups': [
            {'params': len(g['params']), 'weight_decay': g['weight_decay'], 'lr': g['lr']}
            for g in optimizer.param_groups
        ],
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initial_state_sha256': init_hash,
        'first_epoch_train_order_sha256': history[0]['train_order_sha256'] if history else None,
        'reference_pretrained_sha256': pretrained_hash,
        'pretrained_checkpoint': str(pretrained_path) if pretrained_path else None,
        'training_seconds': float(sum(row['epoch_seconds'] for row in history)),
        'wall_seconds': time.time() - start_time,
        'final_model_sha256': sha256_file(model_file),
        'test_predictions_sha256': sha256_file(out / 'test_predictions.npz'),
        'reference_source_hashes': {
            str(p.relative_to(REFERENCE_ROOT) if p.is_relative_to(REFERENCE_ROOT) else p): sha256_file(p)
            for p in reference_files if p.is_file()
        },
        'rng_scope_note': (
            'dataset/model/seed stream continues over sorted folds; one unused model build and '
            'per-epoch sequential test-loader base-seed draw emulated; final-only evaluation; '
            'new seeds 0/1/2 are not paired with older 666/667/668 runs'
            if profile == WIDEBAND_PROFILE else
            'dataset/model/seed stream continues over sorted folds; reference per-epoch test loader '
            'base-seed draw emulated; only final test metrics are evaluated and used'
            if profile in ('reference_aligned', 'reference_aligned_npy') else
            'each fold independently seeded; paired source variants reset to identical seed'
        ),
    }
    result_tmp = out / 'result.json.partial'
    result_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(result_tmp, out / 'result.json')
    completed_state = {
        'profile': profile, 'dataset': dataset, 'model': model_name,
        'subject': subject + 1, 'variant': variant, 'seed': seed,
        'config_fingerprint': hashlib.sha256(json.dumps(fold_cfg, sort_keys=True).encode()).hexdigest(),
        'next_epoch': epochs, 'global_step': global_step,
        'completed': True, 'rng_state': capture_rng(device),
        'initial_state_sha256': init_hash,
        **source_identity,
    }
    checkpoint_save(out / 'completed_state.pt', compact_completed_state(completed_state))
    # Results, final weights and the terminal RNG are durable before retiring
    # the temporary optimizer/model checkpoint used only for interrupted folds.
    ckpt_path.unlink(missing_ok=True)
    set_status(out, {**result, 'status': 'complete', 'completed_utc_epoch_s': time.time()})
    return result


def write_seed_summary(base: Path, results: list[dict], expected_folds: int) -> None:
    complete = [x for x in results if x.get('status') == 'complete']
    if not complete:
        return
    keys = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1')
    summary = {
        'status': 'complete' if len(complete) == expected_folds else 'partial',
        'folds_complete': len(complete), 'folds_expected': expected_folds,
        'subject_equal_mean': {
            key: float(np.mean([r['metrics'][key] for r in complete])) for key in keys
        },
        'fold_metrics': [
            {'held_out_subject': r['held_out_subject'], **r['metrics']}
            for r in complete
        ],
    }
    tmp = base / 'seed_summary.json.partial'
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, base / 'seed_summary.json')


def run_worker(args, spec: dict, device: torch.device) -> None:
    dataset, model = canonical_dataset(args.dataset), args.model
    cfg = profile_config(args.profile, model, dataset, spec)
    variants = (['legacy_cache', 'rebuilt_source'] if args.profile == 'source_bridge'
                else [WIDEBAND_VARIANT] if args.profile == WIDEBAND_PROFILE
                else ['npy_source'] if args.profile == 'reference_aligned_npy'
                else ['rebuilt_source'])
    if args.profile == 'source_bridge' and dataset != 'BNCI2014001-4':
        raise ValueError('source_bridge supports BNCI2014001-4 only')
    subject_outputs = []
    root = RESULT_ROOT / args.profile / dataset / model / f'seed_{args.seed}'
    # Match run_finetuning.run_single_seed: one seed, a throwaway model build,
    # then a fresh model at the start of every ordered LOSO split.
    set_seed(args.seed)
    x0, y0, s0, _, _ = load_fold_data(args.profile, dataset, model, variants[0])
    active_input_shape = list(x0.shape[1:])
    active_num_classes = int(len(np.unique(y0)))
    dummy = make_model(model, dataset, int(len(np.unique(y0))), tuple(x0.shape[1:]),
                       cfg['dropout'], args.profile)
    dummy.to(device)
    del dummy
    del x0, y0, s0
    subjects = range(int(spec['datasets'][dataset]['subjects']))
    if args.profile == 'source_bridge':
        # The bridge intentionally resets the stream per held-out fold and per
        # source variant, allowing one-to-one paired initialization/batches.
        for subject in subjects:
            variants_results = []
            for variant in variants:
                x, y, sub, trials, manifest = load_fold_data(args.profile, dataset, model, variant)
                out = root / f'subject_{subject + 1:02d}' / variant
                final = train_one_fold(
                    args.profile, dataset, model, args.seed, subject, variant,
                    x, y, sub, trials, manifest, cfg, device, out,
                    epochs_override=args.epochs_override,
                    max_batches=args.preflight_steps,
                )
                variants_results.append(final)
                if args.preflight_steps is not None:
                    print(json.dumps(final, sort_keys=True), flush=True)
                    return
            hashes = {r['initial_state_sha256'] for r in variants_results}
            orders = {r.get('first_epoch_train_order_sha256') for r in variants_results}
            if len(hashes) != 1 or len(orders) != 1:
                raise RuntimeError(f'paired inputs diverged in init/order for subject {subject + 1}')
            pair = {
                'status': 'complete', 'dataset': dataset, 'model': model,
                'seed': args.seed, 'held_out_subject': subject + 1,
                'initialization_matched': True, 'first_epoch_order_matched': True,
                'variants': {v: r['metrics'] for v, r in zip(variants, variants_results)},
                'accuracy_delta_rebuilt_minus_legacy': (
                    variants_results[1]['metrics']['accuracy'] - variants_results[0]['metrics']['accuracy']),
                'initial_state_sha256': variants_results[0]['initial_state_sha256'],
                'first_epoch_train_order_sha256': variants_results[0]['first_epoch_train_order_sha256'],
            }
            pair_path = root / f'subject_{subject + 1:02d}' / 'paired_result.json'
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + '\n')
            print(f"[complete] bridge {dataset}/{model} subject={subject + 1} "
                  f"old={variants_results[0]['metrics']['accuracy']:.4f} "
                  f"rebuilt={variants_results[1]['metrics']['accuracy']:.4f}", flush=True)
        pairs = []
        for subject in subjects:
            pair_file = root / f'subject_{subject + 1:02d}' / 'paired_result.json'
            if pair_file.exists():
                pairs.append(read_json(pair_file))
        metrics = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1')
        bridge_summary = {
            'status': 'complete' if len(pairs) == len(list(subjects)) else 'partial',
            'dataset': dataset, 'model': model, 'seed': args.seed,
            'folds_complete': len(pairs), 'folds_expected': len(list(subjects)),
            'subject_equal_mean_by_variant': {
                variant: {
                    key: float(np.mean([p['variants'][variant][key] for p in pairs]))
                    for key in metrics
                } for variant in variants
            },
            'mean_accuracy_delta_rebuilt_minus_legacy': float(np.mean([
                p['accuracy_delta_rebuilt_minus_legacy'] for p in pairs
            ])) if pairs else None,
            'folds': pairs,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / 'bridge_summary.json').write_text(
            json.dumps(bridge_summary, indent=2, sort_keys=True) + '\n')
        return

    # Resume the formal seed stream from the last fully completed fold. If a
    # previous fold is interrupted mid-epoch its own checkpoint resumes first.
    restore_from = None
    expected_subjects = list(subjects)
    pending: list[int] = []
    active_cfg = dict(cfg)
    if args.epochs_override is not None:
        active_cfg['epochs'] = int(args.epochs_override)
    active_manifest = input_dir(args.profile, dataset, model, variants[0]) / 'manifest.json'
    pretrained_path = pretrained_checkpoint(model)
    stream_identity = {
        'profile': args.profile, 'dataset': dataset, 'model': model,
        'seed': args.seed, 'variant': variants[0],
        'input_manifest_sha256': sha256_file(active_manifest),
        'pretrained_checkpoint_sha256': sha256_file(pretrained_path) if pretrained_path else None,
        'input_shape': active_input_shape, 'num_classes': active_num_classes,
        'config_fingerprint': hashlib.sha256(
            json.dumps(active_cfg, sort_keys=True).encode()).hexdigest(),
    }
    for subject in expected_subjects:
        out = result_path(root, subject)
        ckpt = out / 'training_state.pt'
        identity = {**stream_identity, 'subject': subject + 1}
        completion = completed_fold(out, identity, int(active_cfg['epochs']))
        if completion is not None:
            final, state = completion
            restore_from = state['rng_state']
            subject_outputs.append(final)
            continue
        if ckpt.exists():
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            if args.profile == WIDEBAND_PROFILE:
                validate_wideband_checkpoint(state, identity, ckpt)
            pending = expected_subjects[subject:]
            break
        pending = expected_subjects[subject:]
        break
    if restore_from is not None:
        restore_rng(restore_from, device)
    if not pending:
        write_seed_summary(root, subject_outputs, len(expected_subjects))
        print(f'[complete] {args.profile}/{dataset}/{model}/seed_{args.seed} already complete', flush=True)
        return

    for subject in pending:
        variants_results = []
        for variant in variants:
            x, y, sub, trials, manifest = load_fold_data(args.profile, dataset, model, variant)
            out = result_path(root, subject)
            res = train_one_fold(
                args.profile, dataset, model, args.seed, subject, variant,
                x, y, sub, trials, manifest, cfg, device, out,
                epochs_override=args.epochs_override,
                max_batches=args.preflight_steps,
                stream_resume_rng=True,
            )
            variants_results.append(res)
            if args.preflight_steps is not None:
                print(json.dumps(res, sort_keys=True), flush=True)
                return
        subject_outputs.append(variants_results[-1])
        print(f"[complete] {args.profile}/{dataset}/{model} seed={args.seed} "
              f"subject={subject + 1}/{len(expected_subjects)} "
              f"acc={variants_results[-1]['metrics']['accuracy']:.4f} "
              f"elapsed={variants_results[-1]['wall_seconds'] / 60:.1f}min", flush=True)
        write_seed_summary(root, subject_outputs, len(expected_subjects))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('reference_aligned', 'reference_aligned_npy',
                                               'source_bridge', WIDEBAND_PROFILE), required=True)
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_ALIASES))
    parser.add_argument('--model', choices=('cbramod', 'eegnet', 'mirepnet'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True, help='physical CUDA index; GPU 0 is prohibited')
    parser.add_argument('--epochs-override', type=int)
    parser.add_argument('--preflight-steps', type=int,
                        help='run a short non-result forward/backward preflight and exit')
    args = parser.parse_args()
    if args.profile != WIDEBAND_PROFILE:
        raise SystemExit(
            f'{args.profile} is retired. For 001/001-4 use --profile '
            'wideband_npy_v3 and configs/protocols/loso_001.yaml. '
            'For 004/5001 use experiments/finetune/run_loso_source_refresh_004_5001.py.')
    if args.gpu == 0:
        raise SystemExit('GPU 0 is prohibited for this experiment')
    if args.preflight_steps is not None and args.preflight_steps < 1:
        raise SystemExit('--preflight-steps must be >= 1')
    if args.epochs_override is not None and args.epochs_override < 1:
        raise SystemExit('--epochs-override must be >= 1')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; refusing silent CPU fallback')
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f'GPU index {args.gpu} outside visible range 0..{torch.cuda.device_count()-1}')
    canonical = canonical_dataset(args.dataset)
    if args.profile == WIDEBAND_PROFILE:
        if canonical not in ('BNCI2014001', 'BNCI2014001-4'):
            raise SystemExit('wideband_npy_v3 supports only 001 and 001-4')
        if args.model not in ('mirepnet', 'cbramod'):
            raise SystemExit('wideband_npy_v3 supports only MIRepNet and CBraMod')
        if args.seed not in (0, 1, 2):
            raise SystemExit('wideband_npy_v3 uses seeds 0, 1, 2')
    lock_dir = RESULT_ROOT / args.profile / canonical / args.model / f'seed_{args.seed}'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / 'worker.lock').open('a')
    while True:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            print(f'[lock-wait] {args.profile}/{canonical}/{args.model}/seed_{args.seed} '
                  'already has a worker; waiting for its resumable outputs', flush=True)
            time.sleep(30)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    spec = yaml.safe_load(SPEC.read_text())
    try:
        run_worker(args, spec, device)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == '__main__':
    main()
