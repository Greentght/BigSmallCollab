"""Run the seven-source BNCI2014004 adaptive-weight / RL gate LOSO pilot."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)

from collab.sample_utility import (SampleUtilityMLP, bernoulli_policy_loss,
                                  build_controller_input, sample_bernoulli_actions,
                                  student_loss)
from collab.lookahead import (FunctionalIFNet, FunctionalState,
                              feedback_logits, functional_training_step)
from experiments.distill import sample_utility_protocol as protocol
from experiments.finetune import run_loso_small_baselines as small_baselines
from experiments.storage import require_external_output, resolve_local_file
from models import get_adapter


CONDITIONS = ('BASE_CE', 'DELAYED_KD_ALL', 'ADAPTIVE_WEIGHT_KD',
              'RL_GATE_KD', 'ADAPTIVE_SHUFFLE', 'RL_SHUFFLE')
CONTROLLER_CONDITIONS = {'ADAPTIVE_WEIGHT_KD', 'RL_GATE_KD'}
CONTROL_CONDITIONS = {'BASE_CE', 'DELAYED_KD_ALL'}
SHUFFLE_SOURCE = {'ADAPTIVE_SHUFFLE': 'ADAPTIVE_WEIGHT_KD',
                  'RL_SHUFFLE': 'RL_GATE_KD'}
OFFSET = {'controller_init': 10000, 'action': 20000, 'feedback': 30000,
          'adaptive_shuffle': 40000, 'rl_shuffle': 50000}
BATCH_SIZE = 16
TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 10
LAMBDA_KD = 0.5
TEMPERATURE = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(protocol.CONFIG_PATH))
    parser.add_argument('--stage', choices=('preflight', 'teacher', 'warmup',
                                            'controls', 'learned', 'shuffle',
                                            'evaluate', 'report', 'all'),
                        default='preflight')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='zero-based target subject indices; default is all nine')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-continuation-epochs', type=int, default=3)
    parser.add_argument('--smoke-id', default=None,
                        help='unique external output namespace for a smoke run')
    parser.add_argument('--assert-complete', action='store_true')
    return parser.parse_args()


def device_for(gpu):
    if gpu is None:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requested but CUDA is unavailable')
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f'invalid GPU index {gpu}')
    torch.cuda.set_device(gpu)
    return torch.device(f'cuda:{gpu}')


def seed_everything(seed, device):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['torch_cuda'], device)


@contextmanager
def preserve_torch_rng(device):
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def environment_snapshot(device):
    packages = {}
    for package in ('torch', 'numpy', 'scipy', 'scikit-learn', 'mne', 'pandas'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'not-installed'
    return {'python': sys.version.split()[0], 'packages': packages,
            'torch_cuda': torch.version.cuda, 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}


def git_snapshot():
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                              text=True).strip(),
            'tracked_changes': subprocess.check_output(
                ['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()}


def make_paths(fold, feedback, split_sha, smoke, smoke_id=None):
    paths = protocol.fold_paths(fold, feedback, split_sha)
    if smoke:
        if not smoke_id:
            raise ValueError('smoke runs require a unique --smoke-id')
        for key in ('cache', 'weights', 'results'):
            paths[key] = paths[key] / 'smoke' / smoke_id
    return paths


def _schedule_hash(schedule, uids):
    h = hashlib.sha256()
    for epoch, batches in enumerate(schedule, start=1):
        h.update(np.asarray([epoch], dtype=np.int64).tobytes())
        for batch in batches:
            h.update(np.asarray(uids[np.asarray(batch, dtype=np.int64)], dtype=np.int64).tobytes())
    return h.hexdigest()


def _hash_rng_schedule(schedule):
    h = hashlib.sha256()
    for batch in schedule:
        h.update(np.asarray(batch, dtype=np.int64).tobytes())
    return h.hexdigest()


def _store_train_schedule(path, schedule, uids):
    flat, batch_offsets, epoch_offsets = [], [0], [0]
    uid_hash = hashlib.sha256()
    for epoch in schedule:
        for batch in epoch:
            row = np.asarray(uids[np.asarray(batch, dtype=np.int64)], dtype=np.int64)
            flat.extend(np.asarray(batch, dtype=np.int64).tolist())
            batch_offsets.append(len(flat))
            uid_hash.update(row.tobytes())
        epoch_offsets.append(len(batch_offsets) - 1)
    protocol.atomic_npz(path, indices=np.asarray(flat, dtype=np.int64),
                        batch_offsets=np.asarray(batch_offsets, dtype=np.int64),
                        epoch_offsets=np.asarray(epoch_offsets, dtype=np.int64),
                        uid_schedule_sha256=np.asarray(uid_hash.hexdigest()))


def _store_feedback_schedule(path, batches, uids):
    flat, offsets = [], [0]
    uid_hash = hashlib.sha256()
    for batch in batches:
        row = np.asarray(uids[np.asarray(batch, dtype=np.int64)], dtype=np.int64)
        flat.extend(batch.tolist())
        offsets.append(len(flat))
        uid_hash.update(row.tobytes())
    protocol.atomic_npz(path, indices=np.asarray(flat, dtype=np.int64),
                        offsets=np.asarray(offsets, dtype=np.int64),
                        uid_schedule_sha256=np.asarray(uid_hash.hexdigest()))


def _lock_fold(path):
    path = require_external_output(path)
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / '.fold.lock').open('a+')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _student_adapter(device, cfg):
    adapter_cfg = dict(cfg['student_config'])
    adapter = get_adapter('ifnet', device=device, **adapter_cfg)
    return adapter


def _new_student(device, cfg):
    adapter = _student_adapter(device, cfg)
    seed_everything(666, device)
    model = adapter.build(2)
    model.train()
    opt_cfg = cfg['optimizer_defaults']['student']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg['student_config']['lr']),
        weight_decay=float(cfg['student_config']['weight_decay']),
        betas=tuple(opt_cfg['betas']), eps=float(opt_cfg['eps']),
        amsgrad=bool(opt_cfg['amsgrad']),
        foreach=bool(opt_cfg['foreach']), fused=bool(opt_cfg['fused']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg['student_config']['epochs']))
    return adapter, model, optimizer, scheduler


def _load_or_make_student_inputs(paths, split, cfg, device):
    adapter = _student_adapter(device, cfg)
    cache_path = paths['cache'] / 'student_inputs.npz'
    if cache_path.is_file():
        with np.load(resolve_local_file(cache_path), allow_pickle=False) as source:
            cached = {key: np.asarray(source[key]) for key in source.files}
        for name in ('train', 'feedback'):
            if not np.array_equal(cached[f'{name}_uids'], split[name]['uids']):
                raise RuntimeError(f'preprocessed {name} UID mismatch at {cache_path}')
            if not np.array_equal(cached[f'{name}_y'], split[name]['y']):
                raise RuntimeError(f'preprocessed {name} labels mismatch at {cache_path}')
        return adapter, cached
    arrays = {}
    for name in ('train', 'feedback'):
        arrays[f'{name}_x'] = adapter.preprocess(split[name]['x']).cpu().numpy().astype(np.float32)
        arrays[f'{name}_y'] = np.asarray(split[name]['y'], dtype=np.int64)
        arrays[f'{name}_uids'] = np.asarray(split[name]['uids'], dtype=np.int64)
        arrays[f'{name}_local_uids'] = np.asarray(split[name]['local_uids'], dtype=np.int64)
        arrays[f'{name}_subjects'] = np.asarray(split[name]['subjects'], dtype=np.int64)
    protocol.atomic_npz(cache_path, **arrays)
    return adapter, arrays


def _export_reference_features(paths, warmup, train_x, train_y, train_uids, device):
    cache_path = paths['cache'] / 'reference_student_train.npz'
    if cache_path.is_file():
        with np.load(resolve_local_file(cache_path), allow_pickle=False) as source:
            data = {key: np.asarray(source[key]) for key in source.files}
        if not np.array_equal(data['uids'], train_uids):
            raise RuntimeError('S10 feature cache UID mismatch')
        if not np.array_equal(data['y'], train_y):
            raise RuntimeError('S10 feature cache labels do not match its train UIDs')
        if data['warmup_state_sha256'].item() != warmup['warmup_state_sha256']:
            raise RuntimeError('S10 features belong to another warm-up checkpoint')
        return data
    adapter, model, _, _ = _new_student(device, {
        'student_config': warmup['student_config'],
        'optimizer_defaults': warmup['optimizer_defaults'],
    })
    model.load_state_dict(warmup['model'], strict=True)
    view = FunctionalIFNet(model)
    state = view.snapshot(model)
    feats, logits = [], []
    tensor = torch.as_tensor(train_x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(tensor), 64):
            feature, prediction = view.forward(state, tensor[start:start + 64].to(device), training=False,
                                               differentiable_projection=True)
            feats.append(feature.cpu().numpy())
            logits.append(prediction.cpu().numpy())
    data = {'feats': np.concatenate(feats).astype(np.float32),
            'logits': np.concatenate(logits).astype(np.float32),
            'y': np.asarray(train_y, dtype=np.int64),
            'uids': np.asarray(train_uids, dtype=np.int64),
            'warmup_state_sha256': np.asarray(warmup['warmup_state_sha256'])}
    protocol.atomic_npz(cache_path, **data)
    return data


def create_warmup(paths, split, train_x, train_uids, cfg, device, *, resume):
    checkpoint = paths['weights'] / 'warmup_epoch10.pt'
    if checkpoint.is_file():
        state = torch.load(resolve_local_file(checkpoint), map_location='cpu')
        if state['train_uid_sha256'] != protocol.hash_uids(train_uids):
            raise RuntimeError(f'warm-up checkpoint has the wrong D_train: {checkpoint}')
        if state['resolved_config_sha256'] != cfg['resolved_config_sha256']:
            raise RuntimeError(f'warm-up checkpoint config mismatch: {checkpoint}')
        _export_reference_features(paths, state, train_x, split['train']['y'],
                                   train_uids, device)
        return state
    adapter, model, optimizer, scheduler = _new_student(device, cfg)
    generator_seed = 666
    schedule = protocol.epoch_schedule(len(train_uids), BATCH_SIZE, 100, generator_seed)
    schedule_hash = _schedule_hash(schedule, train_uids)
    _store_train_schedule(paths['cache'] / 'train_schedule.npz', schedule, train_uids)
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(split['train']['y'], dtype=torch.long, device=device)
    history, started = [], time.monotonic()
    for epoch in range(10):
        model.train()
        loss_sum, seen = 0.0, 0
        for ids in schedule[epoch]:
            xb, yb = x[ids], y[ids]
            optimizer.zero_grad(set_to_none=True)
            _, logits = adapter.forward(model, xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            seen += len(ids)
        lr_used = float(optimizer.param_groups[0]['lr'])
        scheduler.step()
        history.append({'epoch': epoch + 1, 'ce_loss': loss_sum / seen,
                        'lr_used': lr_used,
                        'lr_next': float(scheduler.get_last_lr()[0])})
        print(f'[warmup] {paths["key"]} epoch={epoch+1:02d}/10 ce={history[-1]["ce_loss"]:.5f}',
              flush=True)
    payload = {
        'epoch': 10, 'model': _to_cpu(model.state_dict()),
        'optimizer': _to_cpu(optimizer.state_dict()),
        'scheduler': _to_cpu(scheduler.state_dict()),
        'rng': rng_state(device), 'train_uid_sha256': protocol.hash_uids(train_uids),
        'schedule_sha256': schedule_hash, 'schedule_seed': generator_seed,
        'resolved_config_sha256': cfg['resolved_config_sha256'],
        'student_config': cfg['student_config'],
        'optimizer_defaults': cfg['optimizer_defaults'], 'history': history,
        'elapsed_seconds': time.monotonic() - started,
    }
    payload['warmup_state_sha256'] = state_hash(payload['model'])
    protocol.atomic_torch(checkpoint, payload)
    protocol.atomic_csv(paths['results'] / 'warmup_history.csv', history)
    _export_reference_features(paths, payload, train_x, split['train']['y'],
                               train_uids, device)
    return payload


def _initial_controller(input_dim, device, fold):
    seed = 666 + OFFSET['controller_init'] + 100 * fold
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = SampleUtilityMLP(input_dim).to(device)
    return model.state_dict(), seed


def _controller_optimizer(controller, cfg):
    return torch.optim.Adam(controller.parameters(), lr=float(cfg['controller']['lr']),
                            betas=tuple(cfg['controller']['betas']),
                            eps=float(cfg['controller']['eps']),
                            weight_decay=float(cfg['controller']['weight_decay']))


def _make_student_optimizer(model, warmup, device):
    opt_cfg = warmup['optimizer_defaults']['student']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(warmup['student_config']['lr']),
        weight_decay=float(warmup['student_config']['weight_decay']),
        betas=tuple(opt_cfg['betas']), eps=float(opt_cfg['eps']),
        amsgrad=bool(opt_cfg['amsgrad']),
        foreach=bool(opt_cfg['foreach']), fused=bool(opt_cfg['fused']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(warmup['student_config']['epochs']))
    optimizer.load_state_dict(warmup['optimizer'])
    scheduler.load_state_dict(warmup['scheduler'])
    return optimizer, scheduler


def _new_feedback_sequence(split, paths, fold, steps, cfg):
    seed = 666 + OFFSET['feedback'] + 100 * fold
    batches = protocol.feedback_schedule(len(split['feedback']['y']),
                                        int(cfg['controller']['feedback_batch_size']),
                                        steps, seed)
    _store_feedback_schedule(paths['cache'] / 'feedback_schedule.npz',
                             batches, split['feedback']['uids'])
    return batches, seed


def _controller_inputs(controller, view, model, state, xb, batch_ids,
                       teacher, reference, teacher_prob, progress):
    with torch.no_grad():
        _, current_logits = view.forward(state, xb, training=False,
                                         differentiable_projection=True)
        current_prob = F.softmax(current_logits, dim=1)
    return build_controller_input(
        torch.as_tensor(teacher['feats'][batch_ids], device=xb.device),
        torch.as_tensor(reference['feats'][batch_ids], device=xb.device),
        torch.as_tensor(teacher_prob[batch_ids], device=xb.device),
        current_prob, progress)


def _virtual_step(view, model, optimizer, xb, yb, teacher_logits,
                  weights, *, create_graph):
    state = view.snapshot(model, requires_grad=True)
    def objective(logits):
        return student_loss(logits, yb, teacher_logits, weights,
                            lam_kd=LAMBDA_KD, temperature=TEMPERATURE)[0]
    return functional_training_step(view, model, optimizer, state, xb, objective,
                                    create_graph=create_graph)


def _feedback_loss(view, state, xb, yb):
    logits = feedback_logits(view, state, xb)
    return F.cross_entropy(logits, yb)


def _capture_condition_context(paths, condition, cfg, device, adapter, warmup):
    model_adapter, model, _, _ = _new_student(device, cfg)
    model.load_state_dict(warmup['model'], strict=True)
    optimizer, scheduler = _make_student_optimizer(model, warmup, device)
    view = FunctionalIFNet(model)
    controller = None
    controller_opt = None
    if condition in CONTROLLER_CONDITIONS:
        input_dim = int(warmup['controller_input_dim'])
        initial, initial_seed = _initial_controller(input_dim, device,
                                                    int(warmup['fold']))
        controller = SampleUtilityMLP(input_dim).to(device)
        controller.load_state_dict(initial, strict=True)
        controller_opt = _controller_optimizer(controller, cfg)
        warmup['_controller_init_sha256'] = state_hash(initial)
        warmup['_controller_init_seed'] = initial_seed
    return model_adapter, model, optimizer, scheduler, view, controller, controller_opt


def _condition_fingerprint(paths, condition, warmup, schedule_sha):
    return protocol.hash_json({
        'condition': condition, 'warmup_sha256': warmup['warmup_state_sha256'],
        'schedule_sha256': schedule_sha, 'split_key': paths['key'],
        'config_sha256': warmup['resolved_config_sha256'],
        'controller_input_dim': warmup.get('controller_input_dim'),
    })


def _load_replay(path, expected_uids, condition):
    if not path.is_file():
        raise FileNotFoundError(f'replay source is incomplete: {path}')
    with np.load(resolve_local_file(path), allow_pickle=False) as source:
        replay = {key: np.asarray(source[key]) for key in source.files}
    replay_uids = np.asarray(replay['uids'], dtype=np.int64)
    expected_uids = np.asarray(expected_uids, dtype=np.int64)
    if (len(replay_uids) != len(expected_uids)
            or len({tuple(uid) for uid in replay_uids.tolist()}) != len(replay_uids)
            or {tuple(uid) for uid in replay_uids.tolist()}
            != {tuple(uid) for uid in expected_uids.tolist()}):
        raise RuntimeError(f'{condition} replay UID set mismatch: {path}')
    return replay


def _apply_real_step(adapter, model, optimizer, xb, yb, tlogits, weights):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, logits = adapter.forward(model, xb)
    total, parts = student_loss(logits, yb, tlogits, weights,
                                lam_kd=LAMBDA_KD, temperature=TEMPERATURE)
    total.backward()
    optimizer.step()
    return {'loss': float(total.detach()), 'ce': float(parts['ce'].detach()),
            'kd': float(parts['kd'].detach()),
            'raw_kd_mean': float(parts['kd_rows'].mean().detach())}


def _write_step_file(path, records):
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with gzip.open(temp, 'wt', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def _save_replay(path, uids, weights, probabilities, actions, rewards,
                 baselines, batch_indices):
    arrays = {'uids': np.asarray(uids, dtype=np.int64),
              'weights': np.asarray(weights, dtype=np.float32),
              'probabilities': np.asarray(probabilities, dtype=np.float32),
              'actions': np.asarray(actions, dtype=np.float32),
              'rewards': np.asarray(rewards, dtype=np.float32),
              'baselines': np.asarray(baselines, dtype=np.float32),
              'batch_index': np.asarray(batch_indices, dtype=np.int32)}
    protocol.atomic_npz(path, **arrays)


def _weighted_epoch_summaries(weights, labels, subjects):
    result = {'weight_mean': float(np.mean(weights)), 'weight_std': float(np.std(weights)),
              'weight_p05': float(np.quantile(weights, 0.05)),
              'weight_p50': float(np.quantile(weights, 0.50)),
              'weight_p95': float(np.quantile(weights, 0.95))}
    for label in (0, 1):
        mask = labels == label
        result[f'weight_class_{label}_mean'] = float(np.mean(weights[mask])) if mask.any() else float('nan')
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        result[f'weight_source_S{int(subject)+1:02d}_mean'] = float(np.mean(weights[mask]))
    return result


def train_condition(condition, paths, fold, feedback, split, train_data,
                    teacher_cache, reference_cache, warmup, cfg, device,
                    *, resume, continuation_epochs=90):
    cell = paths['results'] / condition
    require_external_output(cell).mkdir(parents=True, exist_ok=True)
    run_fingerprint = protocol.hash_json({
        'base': _condition_fingerprint(paths, condition, warmup,
                                       warmup['schedule_sha256']),
        'continuation_epochs': int(continuation_epochs),
    })
    final_manifest = cell / 'manifest.json'
    if final_manifest.is_file():
        old = json.loads(final_manifest.read_text())
        if old.get('status') == 'complete' and old.get('run_fingerprint') == run_fingerprint:
            return old
        raise RuntimeError(f'condition directory belongs to another or incomplete run: {cell}')

    adapter, model, optimizer, scheduler, view, controller, controller_opt = \
        _capture_condition_context(paths, condition, cfg, device,
                                   train_data['adapter'], warmup)
    if controller is not None:
        warmup['controller_input_dim'] = int(reference_cache['feats'].shape[1]
                                             + teacher_cache['feats'].shape[1] + 5)
        # Reconstruct from the exact same initialization after dimensions are fixed.
        controller = SampleUtilityMLP(warmup['controller_input_dim']).to(device)
        init_state, init_seed = _initial_controller(warmup['controller_input_dim'], device, fold)
        controller.load_state_dict(init_state, strict=True)
        controller_opt = _controller_optimizer(controller, cfg)
    student_x = torch.as_tensor(train_data['arrays']['train_x'], dtype=torch.float32, device=device)
    student_y = torch.as_tensor(split['train']['y'], dtype=torch.long, device=device)
    feedback_x = torch.as_tensor(train_data['arrays']['feedback_x'], dtype=torch.float32, device=device)
    feedback_y = torch.as_tensor(split['feedback']['y'], dtype=torch.long, device=device)
    teacher_logits_all = torch.as_tensor(teacher_cache['logits'], dtype=torch.float32, device=device)
    teacher_probs_all = F.softmax(teacher_logits_all / 1.0, dim=1)
    teacher_probs_all = teacher_probs_all.detach()
    reference = reference_cache
    steps_per_epoch = len(warmup['train_schedule'][WARMUP_EPOCHS])
    total_steps = int(continuation_epochs) * steps_per_epoch
    feedback_batches, feedback_seed = _new_feedback_sequence(split, paths, fold, total_steps,
                                                              cfg)
    feedback_schedule_hash = _hash_rng_schedule(feedback_batches)
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(666 + OFFSET['action'] + 100 * fold)
    if not state_path_exists(cell / 'checkpoint_resume.pt'):
        restore_rng(warmup['rng'], device)

    # Warm-up already ended with a fully initialized AdamW state.
    start_epoch = WARMUP_EPOCHS
    reward_baseline = float(cfg['rl']['baseline_initial'])
    state_path = cell / 'checkpoint_resume.pt'
    if state_path.is_file():
        if not resume:
            raise RuntimeError(f'partial run exists; use --resume: {state_path}')
        saved = torch.load(resolve_local_file(state_path), map_location='cpu')
        if saved['run_fingerprint'] != run_fingerprint:
            raise RuntimeError(f'resume fingerprint mismatch: {state_path}')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if controller is not None:
            controller.load_state_dict(saved['controller'], strict=True)
            controller_opt.load_state_dict(saved['controller_optimizer'])
            reward_baseline = float(saved['reward_baseline'])
        action_generator.set_state(saved['action_generator_state'])
        if saved.get('feedback_schedule_sha256') != feedback_schedule_hash:
            raise RuntimeError('feedback schedule differs from the saved checkpoint')
        restore_rng(saved['rng'], device)
        start_epoch = int(saved['completed_epoch'])
        if not WARMUP_EPOCHS <= start_epoch <= WARMUP_EPOCHS + continuation_epochs:
            raise RuntimeError('invalid resumed epoch')
    elif resume and not state_path.is_file():
        raise FileNotFoundError(f'no partial checkpoint to resume: {state_path}')

    history = list(saved['history']) if state_path.is_file() else []
    if len(history) != start_epoch - WARMUP_EPOCHS:
        raise RuntimeError('checkpoint history length differs from resume epoch')
    if state_path.is_file():
        protocol.atomic_json(cell / 'history.json', history)
        protocol.atomic_csv(cell / 'history.csv', history)
    global_offset = (start_epoch - WARMUP_EPOCHS) * steps_per_epoch
    started = time.monotonic()
    train_uid = np.asarray(split['train']['uids'], dtype=np.int64)
    train_labels = np.asarray(split['train']['y'], dtype=np.int64)
    train_subjects = np.asarray(split['train']['subjects'], dtype=np.int64)
    for epoch in range(start_epoch, WARMUP_EPOCHS + continuation_epochs):
        epoch_started = time.monotonic()
        model.train()
        batch_records, replay_weights, replay_q, replay_actions = [], [], [], []
        replay_rewards, replay_baselines, replay_batch_ids, replay_uids = [], [], [], []
        sums = {'loss': 0.0, 'ce': 0.0, 'kd': 0.0, 'raw_kd_mean': 0.0}
        seen = 0
        virtual_steps = feedback_queries = feedback_query_trials = 0
        lr_used = float(optimizer.param_groups[0]['lr'])
        for batch_no, batch_ids in enumerate(warmup['train_schedule'][epoch]):
            global_step = (epoch - WARMUP_EPOCHS) * steps_per_epoch + batch_no
            batch_ids = np.asarray(batch_ids, dtype=np.int64)
            xb = student_x[batch_ids]
            yb = student_y[batch_ids]
            tlogits = teacher_logits_all[batch_ids]
            student_rng = rng_state(device)
            current_state = view.snapshot(model)
            weights = None
            q = torch.ones(len(batch_ids), device=device)
            actions = q.clone()
            reward_value = 0.0
            baseline_value = reward_baseline
            controller_grad_norm = 0.0
            feedback_loss_value = None
            feedback_ce_value = None
            if controller is not None:
                progress = global_step / max(total_steps - 1, 1)
                u = _controller_inputs(controller, view, model, current_state, xb,
                                       batch_ids, teacher_cache, reference,
                                       teacher_probs_all.cpu().numpy(), progress)
                if condition == 'ADAPTIVE_WEIGHT_KD':
                    w_old = controller(u)
                    fb_ids = feedback_batches[global_step]
                    fidx = np.asarray(fb_ids, dtype=np.int64)
                    fxb, fyb = feedback_x[fidx], feedback_y[fidx]
                    with preserve_torch_rng(device):
                        virtual, _, _, _ = _virtual_step(view, model, optimizer,
                                                         xb, yb, tlogits, w_old,
                                                         create_graph=True)
                    virtual_steps += 1
                    meta_loss = _feedback_loss(view, virtual, fxb, fyb)
                    feedback_queries += 1
                    feedback_query_trials += len(fidx)
                    controller_opt.zero_grad(set_to_none=True)
                    grads = torch.autograd.grad(meta_loss, tuple(controller.parameters()),
                                                allow_unused=False)
                    if not all(torch.isfinite(grad).all() for grad in grads):
                        raise FloatingPointError('non-finite adaptive meta-gradient')
                    controller_grad_norm = float(torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])))
                    for parameter, grad in zip(controller.parameters(), grads):
                        parameter.grad = grad
                    controller_opt.step()
                    weights = controller(u).detach()
                    feedback_loss_value = float(meta_loss.detach())
                    actions = weights
                    q = w_old.detach()
                else:
                    q = controller(u)
                    baseline_value = reward_baseline
                    actions = sample_bernoulli_actions(q, action_generator)
                    fb_ids = feedback_batches[global_step]
                    fidx = np.asarray(fb_ids, dtype=np.int64)
                    fxb, fyb = feedback_x[fidx], feedback_y[fidx]
                    # fork_rng restores the Student stream, so both branches use
                    # the same train-mode dropout draw and leave no state behind.
                    with preserve_torch_rng(device):
                        ce_state, _, _, _ = _virtual_step(view, model, optimizer,
                                                          xb, yb, tlogits,
                                                          torch.zeros_like(actions),
                                                          create_graph=False)
                    virtual_steps += 1
                    ce_feedback = _feedback_loss(view, ce_state, fxb, fyb)
                    with preserve_torch_rng(device):
                        action_state, _, _, _ = _virtual_step(view, model, optimizer,
                                                              xb, yb, tlogits, actions,
                                                              create_graph=False)
                    virtual_steps += 1
                    action_feedback = _feedback_loss(view, action_state, fxb, fyb)
                    feedback_queries += 2
                    feedback_query_trials += 2 * len(fidx)
                    reward = (ce_feedback - action_feedback).detach()
                    policy_loss = bernoulli_policy_loss(q, actions, reward,
                                                        torch.as_tensor(baseline_value, device=device))
                    controller_opt.zero_grad(set_to_none=True)
                    controller_grads = torch.autograd.grad(policy_loss,
                                                           tuple(controller.parameters()),
                                                           allow_unused=False)
                    if not all(torch.isfinite(grad).all() for grad in controller_grads):
                        raise FloatingPointError('non-finite RL policy gradient')
                    controller_grad_norm = float(torch.linalg.vector_norm(
                        torch.stack([torch.linalg.vector_norm(g.detach()) for g in controller_grads])))
                    for parameter, grad in zip(controller.parameters(), controller_grads):
                        parameter.grad = grad
                    controller_opt.step()
                    reward_value = float(reward)
                    feedback_ce_value = float(ce_feedback.detach())
                    feedback_loss_value = float(action_feedback.detach())
                    decay = float(cfg['rl']['baseline_ema_decay'])
                    reward_baseline = decay * baseline_value + (1.0 - decay) * reward_value
                    weights = actions.detach()
            elif condition == 'DELAYED_KD_ALL':
                weights = torch.ones(len(batch_ids), device=device)
            elif condition == 'BASE_CE':
                weights = torch.zeros(len(batch_ids), device=device)

            # Isolated branches, feature/controller work, and action sampling do
            # not advance the Student's dropout stream.
            restore_rng(student_rng, device)
            train_metrics = _apply_real_step(adapter, model, optimizer, xb, yb,
                                             tlogits, weights)
            count = len(batch_ids)
            seen += count
            for key in sums:
                sums[key] += train_metrics[key] * count
            w_np = weights.detach().cpu().numpy().astype(np.float32)
            q_np = q.detach().cpu().numpy().astype(np.float32)
            a_np = actions.detach().cpu().numpy().astype(np.float32)
            replay_uids.append(train_uid[batch_ids])
            replay_weights.append(w_np)
            replay_q.append(q_np)
            replay_actions.append(a_np)
            replay_rewards.append(np.full(count, reward_value, dtype=np.float32))
            replay_baselines.append(np.full(count, baseline_value, dtype=np.float32))
            replay_batch_ids.extend([batch_no] * count)
            batch_records.append({
                'epoch': epoch + 1, 'step_in_epoch': batch_no,
                'global_postwarmup_step': global_step,
                'train_uids': train_uid[batch_ids].tolist(),
                'weights': w_np.tolist(), 'probabilities': q_np.tolist(),
                'actions': a_np.tolist(), 'reward': reward_value,
                'baseline_before_action': baseline_value,
                'feedback_ce_loss': feedback_ce_value,
                'feedback_action_loss': feedback_loss_value,
                'controller_grad_norm': controller_grad_norm,
                **train_metrics, 'lr': lr_used,
            })
        scheduler.step()
        means = {key: value / seen for key, value in sums.items()}
        epoch_replay_weights = np.concatenate(replay_weights)
        epoch_replay_actions = np.concatenate(replay_actions)
        entry = {'epoch': epoch + 1, **means, 'n_train': seen,
                 'lr_used': lr_used, 'lr_next': float(scheduler.get_last_lr()[0]),
                 'epoch_elapsed_sec': time.monotonic() - epoch_started,
                 'controller_steps': len(batch_records),
                 'temporary_student_steps': virtual_steps,
                 'feedback_queries': feedback_queries,
                 'feedback_query_trials': feedback_query_trials,
                 'effective_lam_kd': LAMBDA_KD * float(epoch_replay_weights.mean()),
                 'rl_keep_rate': float(epoch_replay_actions.mean()) if condition == 'RL_GATE_KD' else None,
                 'feedback_loss_mean': float(np.nanmean([row['feedback_action_loss'] for row in batch_records]))
                 if controller is not None else None,
                 'reward_mean': float(np.mean([row['reward'] for row in batch_records]))
                 if condition == 'RL_GATE_KD' else None,
                 'reward_std': float(np.std([row['reward'] for row in batch_records]))
                 if condition == 'RL_GATE_KD' else None}
        if condition == 'ADAPTIVE_WEIGHT_KD':
            ordered_uids = np.concatenate(replay_uids)
            index_by_uid = {tuple(uid): i for i, uid in enumerate(train_uid.tolist())}
            ordered_indices = np.asarray([index_by_uid[tuple(uid)] for uid in ordered_uids.tolist()],
                                         dtype=np.int64)
            teacher_correct = (teacher_cache['logits'].argmax(axis=1)[ordered_indices]
                               == train_labels[ordered_indices])
            entry.update(_weighted_epoch_summaries(epoch_replay_weights,
                                                   train_labels[ordered_indices],
                                                   train_subjects[ordered_indices]))
            entry['weight_teacher_correct_mean'] = float(np.mean(epoch_replay_weights[teacher_correct]))
            entry['weight_teacher_wrong_mean'] = float(np.mean(epoch_replay_weights[~teacher_correct]))
        elif condition == 'RL_GATE_KD':
            entry.update({'probability_mean': float(np.mean(np.concatenate(replay_q))),
                          'probability_std': float(np.std(np.concatenate(replay_q))),
                          'action_mean': float(epoch_replay_actions.mean())})
        history.append(entry)
        _write_step_file(cell / f'step_metrics_epoch_{epoch+1:03d}.jsonl.gz', batch_records)
        _save_replay(cell / f'replay_epoch_{epoch+1:03d}.npz',
                     np.concatenate(replay_uids), epoch_replay_weights,
                     np.concatenate(replay_q), epoch_replay_actions,
                     np.concatenate(replay_rewards), np.concatenate(replay_baselines),
                     replay_batch_ids)
        checkpoint_payload = {
            'run_fingerprint': run_fingerprint, 'condition': condition,
            'completed_epoch': epoch + 1,
            'model': _to_cpu(model.state_dict()),
            'optimizer': _to_cpu(optimizer.state_dict()),
            'scheduler': _to_cpu(scheduler.state_dict()),
            'controller': _to_cpu(controller.state_dict()) if controller is not None else None,
            'controller_optimizer': _to_cpu(controller_opt.state_dict()) if controller_opt is not None else None,
            'reward_baseline': reward_baseline,
            'action_generator_state': action_generator.get_state(),
            'rng': rng_state(device), 'history': history,
            'resolved_config_sha256': cfg['resolved_config_sha256'],
            'train_uid_sha256': protocol.hash_uids(train_uid),
            'warmup_state_sha256': warmup['warmup_state_sha256'],
            'schedule_sha256': warmup['schedule_sha256'],
            'feedback_schedule_sha256': feedback_schedule_hash,
            'feedback_schedule_seed': feedback_seed,
        }
        protocol.atomic_torch(state_path, _to_cpu(checkpoint_payload))
        protocol.atomic_json(cell / 'history.json', history)
        protocol.atomic_csv(cell / 'history.csv', history)
        global_offset += steps_per_epoch
        print(f'[{condition}] {paths["key"]} epoch={epoch+1:03d}/{WARMUP_EPOCHS+continuation_epochs} '
              f'ce={means["ce"]:.5f} kd={means["kd"]:.5f} '
              f'sec={entry["epoch_elapsed_sec"]:.1f}', flush=True)

    final_model_hash = state_hash(model.state_dict())
    protocol.atomic_torch(cell / 'checkpoint_final.pt', {
        'model': _to_cpu(model.state_dict()), 'condition': condition,
        'final_epoch': WARMUP_EPOCHS + continuation_epochs,
        'run_fingerprint': run_fingerprint,
        'warmup_state_sha256': warmup['warmup_state_sha256'],
        'final_model_sha256': final_model_hash,
    })
    if controller is not None:
        protocol.atomic_torch(cell / 'controller_final.pt', {
            'controller': _to_cpu(controller.state_dict()),
            'controller_initial_sha256': state_hash(_initial_controller(
                warmup['controller_input_dim'], device, fold)[0]),
            'input_dim': warmup['controller_input_dim'],
            'controller_optimizer': _to_cpu(controller_opt.state_dict()),
        })
    manifest = {'status': 'complete', 'condition': condition,
                'run_fingerprint': run_fingerprint,
                'target_subject_id': fold + 1, 'feedback_subject_id': feedback + 1,
                'train_uid_sha256': protocol.hash_uids(train_uid),
                'warmup_state_sha256': warmup['warmup_state_sha256'],
                'schedule_sha256': warmup['schedule_sha256'],
                'feedback_schedule_sha256': feedback_schedule_hash,
                'final_model_sha256': final_model_hash,
                'final_epoch': WARMUP_EPOCHS + continuation_epochs,
                'elapsed_seconds': time.monotonic() - started,
                'resolved_config_sha256': cfg['resolved_config_sha256'],
                'controller_initial_sha256': (state_hash(_initial_controller(
                    warmup['controller_input_dim'], device, fold)[0])
                    if controller is not None else None)}
    protocol.atomic_json(final_manifest, manifest)
    return manifest


def _cpu_torch_rng_state(device):
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def state_path_exists(path):
    return Path(path).is_file()


def train_shuffle(condition, source_condition, paths, fold, split, train_data,
                  warmup, cfg, device, *, resume, continuation_epochs=90):
    source_cell = paths['results'] / source_condition
    output_cell = paths['results'] / condition
    replay_paths = [source_cell / f'replay_epoch_{epoch:03d}.npz'
                    for epoch in range(WARMUP_EPOCHS + 1,
                                       WARMUP_EPOCHS + continuation_epochs + 1)]
    for replay_path in replay_paths:
        if not replay_path.is_file():
            raise FileNotFoundError(f'shuffle requires complete learned replay: {replay_path}')
    output_cell.mkdir(parents=True, exist_ok=True)
    replay_digest = hashlib.sha256()
    for replay_path in replay_paths:
        with replay_path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                replay_digest.update(block)
    source_replay_sha256 = replay_digest.hexdigest()
    fingerprint = protocol.hash_json({
        'base': _condition_fingerprint(paths, condition, warmup, warmup['schedule_sha256']),
        'source_replay_sha256': source_replay_sha256,
        'continuation_epochs': int(continuation_epochs),
    })
    final_manifest = output_cell / 'manifest.json'
    if final_manifest.is_file():
        old = json.loads(final_manifest.read_text())
        if old.get('status') == 'complete' and old.get('run_fingerprint') == fingerprint:
            return old
        raise RuntimeError(f'shuffle output belongs to another run: {output_cell}')
    adapter, model, optimizer, scheduler, view, _, _ = _capture_condition_context(
        paths, condition, cfg, device, train_data['adapter'], warmup)
    student_x = torch.as_tensor(train_data['arrays']['train_x'], dtype=torch.float32, device=device)
    student_y = torch.as_tensor(split['train']['y'], dtype=torch.long, device=device)
    uids = np.asarray(split['train']['uids'], dtype=np.int64)
    replay_by_epoch = {}
    for epoch, replay_path in enumerate(replay_paths, start=WARMUP_EPOCHS + 1):
        replay = _load_replay(replay_path, uids, source_condition)
        if len(replay['weights']) != len(uids):
            raise RuntimeError(f'incomplete source replay for epoch {epoch}')
        replay_by_epoch[epoch] = replay
    source_index = {tuple(uid): i for i, uid in enumerate(uids.tolist())}
    if len(source_index) != len(uids):
        raise RuntimeError('train UIDs must be unique before shuffle')
    shuffle_seed = 666 + OFFSET['adaptive_shuffle' if condition == 'ADAPTIVE_SHUFFLE'
                                  else 'rl_shuffle'] + 100 * fold
    shuffle_rng = np.random.default_rng(shuffle_seed)
    state_path = output_cell / 'checkpoint_resume.pt'
    start_epoch = WARMUP_EPOCHS
    history = []
    if not state_path.is_file():
        restore_rng(warmup['rng'], device)
    if state_path.is_file():
        if not resume:
            raise RuntimeError(f'partial shuffle exists; use --resume: {state_path}')
        saved = torch.load(resolve_local_file(state_path), map_location='cpu')
        if saved['run_fingerprint'] != fingerprint:
            raise RuntimeError('shuffle resume fingerprint mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        shuffle_rng.bit_generator.state = saved['shuffle_rng_state']
        restore_rng(saved['rng'], device)
        start_epoch = int(saved['completed_epoch'])
        history = saved['history']
    elif resume:
        raise FileNotFoundError(state_path)
    history_by_epoch = {row['epoch']: row for row in history}
    started = time.monotonic()
    schedule = warmup['train_schedule']
    for epoch in range(start_epoch, WARMUP_EPOCHS + continuation_epochs):
        replay = replay_by_epoch[epoch + 1]
        model.train()
        step_records, output_weights, source_weights, source_actions = [], [], [], []
        source_q, actions_out, uid_rows, batch_rows = [], [], [], []
        ce_sum = kd_sum = 0.0
        seen = 0
        for batch_no, batch_ids in enumerate(schedule[epoch]):
            batch_ids = np.asarray(batch_ids, dtype=np.int64)
            uid_batch = uids[batch_ids]
            source_pos = np.asarray([source_index[tuple(uid)] for uid in uid_batch.tolist()], dtype=np.int64)
            src_w = replay['weights'][source_pos]
            src_a = replay['actions'][source_pos]
            src_q = replay['probabilities'][source_pos]
            permutation = shuffle_rng.permutation(len(batch_ids))
            if condition == 'ADAPTIVE_SHUFFLE':
                used = src_w[permutation]
            else:
                used = src_a[permutation]
            if not np.array_equal(np.sort(used), np.sort(src_w if condition == 'ADAPTIVE_SHUFFLE' else src_a)):
                raise RuntimeError('batch shuffle changed the source weight/action multiset')
            xb, yb = student_x[batch_ids], student_y[batch_ids]
            tlogits = torch.as_tensor(train_data['teacher_logits'][batch_ids], device=device)
            metrics = _apply_real_step(adapter, model, optimizer, xb, yb, tlogits,
                                       torch.as_tensor(used, dtype=torch.float32, device=device))
            ce_sum += metrics['ce'] * len(batch_ids)
            kd_sum += metrics['kd'] * len(batch_ids)
            seen += len(batch_ids)
            uid_rows.append(uid_batch)
            source_weights.extend(src_w.tolist())
            source_actions.extend(src_a.tolist())
            source_q.extend(src_q.tolist())
            output_weights.extend(used.tolist())
            actions_out.extend(used.tolist())
            batch_rows.extend([batch_no] * len(batch_ids))
            step_records.append({'epoch': epoch + 1, 'step': batch_no,
                                 'uids': uid_batch.tolist(),
                                 'source_weights': src_w.tolist(),
                                 'applied_weights': np.asarray(used).tolist(),
                                 'source_actions': src_a.tolist(),
                                 'permutation': permutation.tolist(), **metrics})
        lr_used = float(optimizer.param_groups[0]['lr'])
        scheduler.step()
        entry = {'epoch': epoch + 1, 'ce_loss': ce_sum / seen,
                 'kd_loss': kd_sum / seen, 'n_train': seen,
                 'effective_lam_kd': LAMBDA_KD * float(np.mean(output_weights)),
                 'source_weight_sum': float(np.sum(source_weights)),
                 'applied_weight_sum': float(np.sum(output_weights)),
                 'source_keep_count': (int(np.sum(source_actions))
                                       if condition == 'RL_SHUFFLE' else None),
                 'applied_keep_count': (int(np.sum(actions_out))
                                        if condition == 'RL_SHUFFLE' else None),
                 'lr_used': lr_used}
        if condition == 'RL_SHUFFLE':
            entry['rl_keep_rate'] = float(np.mean(actions_out))
        history.append(entry)
        _write_step_file(output_cell / f'step_metrics_epoch_{epoch+1:03d}.jsonl.gz', step_records)
        _save_replay(output_cell / f'replay_epoch_{epoch+1:03d}.npz',
                     np.concatenate(uid_rows), np.asarray(output_weights),
                     np.asarray(source_q), np.asarray(actions_out),
                     np.zeros(len(output_weights), dtype=np.float32),
                     np.zeros(len(output_weights), dtype=np.float32), batch_rows)
        if abs(entry['source_weight_sum'] - entry['applied_weight_sum']) > 1e-5:
            raise RuntimeError('shuffle did not preserve total weight')
        if (condition == 'RL_SHUFFLE'
                and entry['source_keep_count'] != entry['applied_keep_count']):
            raise RuntimeError('shuffle did not preserve the per-batch action count')
        saved = {'run_fingerprint': fingerprint, 'completed_epoch': epoch + 1,
                 'model': _to_cpu(model.state_dict()),
                 'optimizer': _to_cpu(optimizer.state_dict()),
                 'scheduler': _to_cpu(scheduler.state_dict()),
                 'shuffle_rng_state': shuffle_rng.bit_generator.state,
                 'rng': rng_state(device), 'history': history,
                 'warmup_state_sha256': warmup['warmup_state_sha256'],
                 'source_condition': source_condition}
        protocol.atomic_torch(state_path, _to_cpu(saved))
        protocol.atomic_json(output_cell / 'history.json', history)
        protocol.atomic_csv(output_cell / 'history.csv', history)
        print(f'[{condition}] {paths["key"]} epoch={epoch+1:03d}/{WARMUP_EPOCHS+continuation_epochs} '
              f'ce={entry["ce_loss"]:.5f} kd={entry["kd_loss"]:.5f}', flush=True)
    final_hash = state_hash(model.state_dict())
    protocol.atomic_torch(output_cell / 'checkpoint_final.pt', {
        'model': _to_cpu(model.state_dict()), 'condition': condition,
        'final_epoch': WARMUP_EPOCHS + continuation_epochs,
        'run_fingerprint': fingerprint, 'final_model_sha256': final_hash,
        'warmup_state_sha256': warmup['warmup_state_sha256'],
        'resolved_config_sha256': cfg['resolved_config_sha256'],
        'source_replay_conditions': [source_condition],
    })
    manifest = {'status': 'complete', 'condition': condition,
                'run_fingerprint': fingerprint, 'source_condition': source_condition,
                'target_subject_id': fold + 1,
                'feedback_subject_id': int(np.unique(split['feedback']['subjects'])[0]) + 1,
                'train_uid_sha256': protocol.hash_uids(uids),
                'warmup_state_sha256': warmup['warmup_state_sha256'],
                'final_epoch': WARMUP_EPOCHS + continuation_epochs,
                'final_model_sha256': final_hash,
                'resolved_config_sha256': cfg['resolved_config_sha256']}
    protocol.atomic_json(final_manifest, manifest)
    return manifest


def _load_warmup(paths, train_uids, cfg, device, *, resume):
    checkpoint = paths['weights'] / 'warmup_epoch10.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'warm-up dependency missing: {checkpoint}')
    state = torch.load(resolve_local_file(checkpoint), map_location='cpu')
    if state['train_uid_sha256'] != protocol.hash_uids(train_uids):
        raise RuntimeError('warm-up checkpoint D_train UID mismatch')
    if state['resolved_config_sha256'] != cfg['resolved_config_sha256']:
        raise RuntimeError('warm-up checkpoint resolved config mismatch')
    state['train_schedule'] = protocol.epoch_schedule(len(train_uids), BATCH_SIZE,
                                                      TOTAL_EPOCHS, 666)
    if _schedule_hash(state['train_schedule'], train_uids) != state['schedule_sha256']:
        raise RuntimeError('regenerated training schedule hash mismatch')
    state['controller_input_dim'] = None
    return state


def _prepare_fold(args, resolved, spec, snapshot, source, device, fold):
    feedback, split = protocol.split_fold(*source[:6], fold)
    fold_manifest = protocol.split_manifest(fold, feedback, split,
                                            source[8], resolved)
    paths = make_paths(fold, feedback, fold_manifest['split_sha256'],
                       args.smoke_only, args.smoke_id)
    manifest_path = paths['cache'] / 'split_manifest.json'
    for path in (paths['cache'], paths['weights'], paths['results']):
        require_external_output(path).mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get('split_sha256') != fold_manifest['split_sha256']:
            raise RuntimeError(f'fold cache split mismatch: {manifest_path}')
    else:
        protocol.atomic_json(manifest_path, fold_manifest)
    adapter, arrays = _load_or_make_student_inputs(paths, split, resolved, device)
    steps_per_epoch = int(np.ceil(len(split['train']['y']) / BATCH_SIZE))
    schedule = protocol.epoch_schedule(len(split['train']['y']), BATCH_SIZE,
                                       TOTAL_EPOCHS, 666)
    schedule_sha = _schedule_hash(schedule, split['train']['uids'])
    _store_train_schedule(paths['cache'] / 'train_schedule.npz', schedule,
                          split['train']['uids'])
    source_teacher = protocol.validate_teacher(paths['weights'], split['train']['uids'],
                                               split['train']['y'])
    if source_teacher is None:
        teacher_cache = None
    else:
        teacher_cache = source_teacher[0]
    teacher_logits = (teacher_cache['logits'] if teacher_cache is not None else None)
    data_ctx = {'adapter': adapter, 'arrays': arrays,
                'teacher_logits': teacher_logits}
    return feedback, split, fold_manifest, paths, data_ctx, teacher_cache, schedule, schedule_sha


def _teacher_stage(paths, split, resolved, spec, device):
    existing = protocol.validate_teacher(paths['weights'], split['train']['uids'],
                                         split['train']['y'])
    if existing is not None:
        return existing[0], existing[1]
    seed_everything(666, device)
    model, cache, provenance = protocol.make_teacher(split, resolved, spec, device)
    protocol.save_teacher(paths['weights'], model, cache, provenance)
    return cache, provenance


def _warmup_stage(paths, split, data_ctx, schedule_sha, resolved, device, *, resume):
    warmup = create_warmup(paths, split, data_ctx['arrays']['train_x'],
                           split['train']['uids'], resolved, device, resume=resume)
    warmup['schedule_sha256'] = schedule_sha
    warmup['train_schedule'] = protocol.epoch_schedule(
        len(split['train']['y']), BATCH_SIZE, TOTAL_EPOCHS, 666)
    return warmup


def _condition_scope(args):
    return set(args.conditions) if args.conditions else set(CONDITIONS)


def run_train_stage(stage, args, resolved, spec, snapshot, source, device):
    folds = list(range(9)) if args.folds is None else list(args.folds)
    if len(set(folds)) != len(folds) or any(f < 0 or f > 8 for f in folds):
        raise ValueError('--folds must contain unique zero-based indices from 0 to 8')
    if args.smoke_only:
        if stage not in ('all', 'preflight'):
            raise ValueError('--smoke-only controls its own stages; do not combine --stage')
        folds = [0]
    conditions = _condition_scope(args)
    if not conditions.issubset(set(CONDITIONS)):
        raise ValueError('unknown condition')
    cont_epochs = args.smoke_continuation_epochs if args.smoke_only else 90
    if not 1 <= cont_epochs <= 90:
        raise ValueError('smoke continuation epochs must be in [1,90]')

    if source is None:
        raise ValueError('training stages require one prevalidated source dataset object')
    resolved_out = resolved
    protocol.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_root = (protocol.RESULT_ROOT / 'smoke' / args.smoke_id
                     if args.smoke_only else protocol.RESULT_ROOT)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    protocol.atomic_json(snapshot_root / 'environment.json', environment_snapshot(device))
    protocol.atomic_json(snapshot_root / 'execution_snapshot.json', git_snapshot())
    resolved_yaml = yaml.safe_dump(resolved_out, sort_keys=True, allow_unicode=True)
    resolved_path = snapshot_root / 'resolved.yaml'
    if resolved_path.is_file():
        previous = yaml.safe_load(resolved_path.read_text())
        if previous.get('resolved_config_sha256') != resolved_out['resolved_config_sha256']:
            raise RuntimeError(f'resolved config already exists with another hash: {resolved_path}')
    else:
        resolved_path.write_text(resolved_yaml)

    for fold in folds:
        feedback, split, manifest, paths, data_ctx, teacher_cache, schedule, schedule_sha = \
            _prepare_fold(args, resolved_out, spec, snapshot, source, device, fold)
        lock = _lock_fold(paths['results'].parent)
        try:
            steps = len(schedule[WARMUP_EPOCHS]) * cont_epochs
            if stage == 'preflight':
                print(f'[preflight-ok] {paths["key"]} train={len(split["train"]["y"])} '
                      f'feedback={len(split["feedback"]["y"])} test={len(split["test"]["y"])} '
                      f'batches/epoch={len(schedule[0])} split={manifest["split_sha256"]}',
                      flush=True)
                continue
            if stage in ('teacher', 'all') or args.smoke_only:
                teacher_cache, provenance = _teacher_stage(paths, split, resolved_out, spec, device)
                data_ctx['teacher_logits'] = teacher_cache['logits']
                print(f'[teacher-ready] {paths["key"]} feature_dim={teacher_cache["feats"].shape[1]} '
                      f'train_uid={provenance["train_uid_sha256"]}', flush=True)
                if stage == 'teacher' and not args.smoke_only:
                    continue
            if stage in ('warmup', 'all') or args.smoke_only:
                warmup = _warmup_stage(paths, split, data_ctx, schedule_sha,
                                       resolved_out, device, resume=args.resume)
                data_ctx['reference'] = _export_reference_features(
                    paths, warmup, data_ctx['arrays']['train_x'], split['train']['y'],
                    split['train']['uids'], device)
                warmup['controller_input_dim'] = (teacher_cache['feats'].shape[1]
                                                  + data_ctx['reference']['feats'].shape[1] + 5)
                warmup['fold'] = fold
                if stage == 'warmup' and not args.smoke_only:
                    continue
            else:
                warmup = _load_warmup(paths, split['train']['uids'], resolved_out,
                                      device, resume=args.resume)
                warmup['schedule_sha256'] = schedule_sha
                warmup['train_schedule'] = schedule
                data_ctx['reference'] = _export_reference_features(
                    paths, warmup, data_ctx['arrays']['train_x'], split['train']['y'],
                    split['train']['uids'], device)
                warmup['controller_input_dim'] = (teacher_cache['feats'].shape[1]
                                                  + data_ctx['reference']['feats'].shape[1] + 5)
                warmup['fold'] = fold
            if stage in ('controls', 'all') or args.smoke_only:
                for condition in ('BASE_CE', 'DELAYED_KD_ALL'):
                    if condition in conditions:
                        train_condition(condition, paths, fold, feedback, split,
                                        data_ctx, teacher_cache, data_ctx['reference'],
                                        warmup, resolved_out, device, resume=args.resume,
                                        continuation_epochs=cont_epochs)
                if stage == 'controls' and not args.smoke_only:
                    continue
            if stage in ('learned', 'all') or args.smoke_only:
                for condition in ('ADAPTIVE_WEIGHT_KD', 'RL_GATE_KD'):
                    if condition in conditions:
                        warmup['controller_input_dim'] = (teacher_cache['feats'].shape[1]
                                                          + data_ctx['reference']['feats'].shape[1] + 5)
                        train_condition(condition, paths, fold, feedback, split,
                                        data_ctx, teacher_cache, data_ctx['reference'],
                                        warmup, resolved_out, device, resume=args.resume,
                                        continuation_epochs=cont_epochs)
                if stage == 'learned' and not args.smoke_only:
                    continue
            if stage in ('shuffle', 'all') or args.smoke_only:
                for condition, source_condition in SHUFFLE_SOURCE.items():
                    if condition in conditions:
                        train_shuffle(condition, source_condition, paths, fold,
                                      split, data_ctx, warmup, resolved_out, device,
                                      resume=args.resume,
                                      continuation_epochs=cont_epochs)
            if args.smoke_only:
                print(f'[smoke-complete] {paths["key"]} continuation_epochs={cont_epochs}; '
                      'target evaluation was not run', flush=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


def _load_final_model(paths, condition, cfg, device):
    checkpoint = paths['results'] / condition / 'checkpoint_final.pt'
    manifest = paths['results'] / condition / 'manifest.json'
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError(f'final condition missing: {checkpoint}')
    info = json.loads(manifest.read_text())
    if info.get('status') != 'complete' or info.get('final_epoch') != 100:
        raise RuntimeError(f'condition is not a complete epoch100 run: {manifest}')
    saved = torch.load(resolve_local_file(checkpoint), map_location='cpu')
    if saved['run_fingerprint'] != info['run_fingerprint']:
        raise RuntimeError(f'checkpoint/manifest mismatch: {checkpoint}')
    seed_everything(666, device)
    adapter = _student_adapter(device, cfg)
    model = adapter.build(2)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    return model


def _assert_training_complete(resolved, spec, snapshot, device, source):
    """Fail before target inference unless all nine fold matrices are complete."""
    _x, _y, _subjects, _uids, _local_uids, _records, _meta, _raw, source_hashes = source
    checked = 0
    for fold in range(9):
        feedback, split = protocol.split_fold(*source[:6], fold)
        split_info = protocol.split_manifest(fold, feedback, split,
                                             source_hashes, resolved)
        paths = make_paths(fold, feedback, split_info['split_sha256'], False)
        teacher = protocol.validate_teacher(paths['weights'], split['train']['uids'],
                                            split['train']['y'])
        if teacher is None:
            raise FileNotFoundError(f'missing seven-source Teacher for {paths["key"]}')
        provenance = teacher[1]
        expected_ids = {'target_subject_id': fold + 1,
                        'feedback_subject_id': feedback + 1,
                        'train_uid_sha256': protocol.hash_uids(split['train']['uids'])}
        if any(provenance.get(key) != value for key, value in expected_ids.items()):
            raise RuntimeError(f'Teacher provenance mismatch for {paths["key"]}')
        warmup = _load_warmup(paths, split['train']['uids'], resolved,
                              device, resume=False)
        if warmup.get('epoch') != WARMUP_EPOCHS:
            raise RuntimeError(f'warm-up is not complete for {paths["key"]}')
        reference_dim = _reference_feature_dim(
            paths, warmup, split['train']['uids'], split['train']['y'])

        for condition in CONDITIONS:
            cell = paths['results'] / condition
            manifest_path = cell / 'manifest.json'
            checkpoint_path = cell / 'checkpoint_final.pt'
            if not manifest_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f'incomplete {condition} checkpoint for {paths["key"]}')
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get('status') != 'complete'
                    or manifest.get('condition') != condition
                    or manifest.get('final_epoch') != TOTAL_EPOCHS
                    or manifest.get('target_subject_id') != fold + 1
                    or manifest.get('feedback_subject_id') != feedback + 1
                    or manifest.get('train_uid_sha256') != protocol.hash_uids(split['train']['uids'])
                    or manifest.get('warmup_state_sha256') != warmup['warmup_state_sha256']
                    or manifest.get('resolved_config_sha256') != resolved['resolved_config_sha256']):
                raise RuntimeError(f'condition manifest identity mismatch: {manifest_path}')
            checkpoint = torch.load(resolve_local_file(checkpoint_path), map_location='cpu')
            model_hash = state_hash(checkpoint['model'])
            if (checkpoint.get('run_fingerprint') != manifest.get('run_fingerprint')
                    or checkpoint.get('final_epoch') != TOTAL_EPOCHS
                    or checkpoint.get('final_model_sha256') != model_hash
                    or manifest.get('final_model_sha256') != model_hash):
                raise RuntimeError(f'condition checkpoint identity mismatch: {checkpoint_path}')
            history_path = cell / 'history.json'
            if not history_path.is_file():
                raise FileNotFoundError(f'missing condition history: {history_path}')
            history = json.loads(history_path.read_text())
            if [row.get('epoch') for row in history] != list(range(WARMUP_EPOCHS + 1,
                                                                    TOTAL_EPOCHS + 1)):
                raise RuntimeError(f'condition history is incomplete: {history_path}')
            for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
                for prefix in ('replay', 'step_metrics'):
                    path = cell / f'{prefix}_epoch_{epoch:03d}.npz' if prefix == 'replay' else \
                        cell / f'{prefix}_epoch_{epoch:03d}.jsonl.gz'
                    if not path.is_file():
                        raise FileNotFoundError(f'missing per-epoch artifact: {path}')
            if condition in CONTROLLER_CONDITIONS:
                controller_path = cell / 'controller_final.pt'
                if not controller_path.is_file():
                    raise FileNotFoundError(f'missing learned controller: {controller_path}')
                controller = torch.load(resolve_local_file(controller_path), map_location='cpu')
                if controller.get('input_dim') != teacher[0]['feats'].shape[1] + reference_dim + 5:
                    raise RuntimeError(f'controller input dimension mismatch: {controller_path}')
            checked += 1
    if checked != 54:
        raise RuntimeError(f'expected 54 complete Student runs, validated {checked}')


def _reference_feature_dim(paths, warmup, train_uids, train_y):
    reference_path = paths['cache'] / 'reference_student_train.npz'
    if not reference_path.is_file():
        raise FileNotFoundError(f'missing S10 feature cache: {reference_path}')
    with np.load(resolve_local_file(reference_path), allow_pickle=False) as reference:
        if (not np.array_equal(reference['uids'], train_uids)
                or not np.array_equal(reference['y'], train_y)):
            raise RuntimeError('S10 feature UID/label order differs from D_train')
        if reference['warmup_state_sha256'].item() != warmup['warmup_state_sha256']:
            raise RuntimeError('S10 feature cache belongs to another warm-up')
        return int(reference['feats'].shape[1])


def _assert_evaluation_complete(rows, source, resolved):
    if len(rows) != 54:
        raise RuntimeError(f'expected 54 target evaluation rows, got {len(rows)}')
    keyed = {(int(row['target_subject']), row['condition']): row for row in rows}
    if len(keyed) != 54 or set(keyed) != {
            (fold + 1, condition) for fold in range(9) for condition in CONDITIONS}:
        raise RuntimeError('target metrics do not contain exactly one row per fold/condition')
    _x, _y, _subjects, _uids, _local_uids, _records, _meta, _raw, source_hashes = source
    for fold in range(9):
        feedback, split = protocol.split_fold(*source[:6], fold)
        split_info = protocol.split_manifest(fold, feedback, split, source_hashes, resolved)
        paths = make_paths(fold, feedback, split_info['split_sha256'], False)
        for condition in CONDITIONS:
            row = keyed[(fold + 1, condition)]
            cell = paths['results'] / condition
            pred_path, metrics_path = cell / 'predictions.npz', cell / 'metrics.json'
            if not pred_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f'missing final target artifacts for {row}')
            with np.load(resolve_local_file(pred_path), allow_pickle=False) as saved:
                if (not np.array_equal(saved['uids'], split['test']['uids'])
                        or not np.array_equal(saved['y'], split['test']['y'])):
                    raise RuntimeError(f'target prediction UID/label mismatch: {pred_path}')
                if any(len(saved[key]) != len(split['test']['y'])
                       for key in ('predictions', 'logits', 'probabilities')):
                    raise RuntimeError(f'target prediction length mismatch: {pred_path}')
            metrics = json.loads(metrics_path.read_text())
            for key in ('condition', 'target_subject', 'feedback_subject', 'n_test'):
                if metrics.get(key) != row[key]:
                    raise RuntimeError(f'target metric identity mismatch: {metrics_path}')
            if metrics.get('test_uid_sha256') != protocol.hash_uids(split['test']['uids']):
                raise RuntimeError(f'target metric UID hash mismatch: {metrics_path}')
            if row.get('test_uid_sha256') != metrics['test_uid_sha256']:
                raise RuntimeError(f'CSV/JSON target UID hash mismatch: {metrics_path}')


def evaluate_all(args, resolved, spec, snapshot, device):
    if args.smoke_only:
        raise ValueError('target evaluation is forbidden for smoke runs')
    source = protocol.load_data(spec, snapshot, resolved['source_hashes'])
    selected = list(range(9)) if args.folds is None else args.folds
    if set(selected) != set(range(9)) or len(selected) != 9:
        raise ValueError('target evaluation requires all nine folds as one fixed matrix')
    _assert_training_complete(resolved, spec, snapshot, device, source)
    _x, y, subjects, uids, local_uids, _records, _meta, _raw, source_hashes = source
    rows = []
    for fold in selected:
        feedback, split = protocol.split_fold(_x, y, subjects, uids, local_uids,
                                              _records, fold)
        fm = protocol.split_manifest(fold, feedback, split, source_hashes, resolved)
        paths = make_paths(fold, feedback, fm['split_sha256'], False)
        test_x = torch.as_tensor(split['test']['x'], dtype=torch.float32)
        for condition in CONDITIONS:
            model = _load_final_model(paths, condition, resolved, device)
            adapter = _student_adapter(device, resolved)
            pred_parts = []
            view = FunctionalIFNet(model)
            state = view.snapshot(model)
            with torch.no_grad():
                for start in range(0, len(test_x), 64):
                    xb = adapter.preprocess(split['test']['x'][start:start + 64]).to(device)
                    _, logits = view.forward(state, xb, training=False,
                                             differentiable_projection=True)
                    pred_parts.append(logits.cpu().numpy())
            logits = np.concatenate(pred_parts).astype(np.float32)
            probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
            predictions = probabilities.argmax(axis=1)
            labels = np.asarray(split['test']['y'], dtype=np.int64)
            counts = np.bincount(predictions, minlength=2)
            metrics = {
                'dataset': 'BNCI2014004', 'target_subject': fold + 1,
                'feedback_subject': feedback + 1, 'seed': 666,
                'condition': condition, 'n_test': len(labels),
                'test_uid_sha256': protocol.hash_uids(split['test']['uids']),
                'accuracy': float(accuracy_score(labels, predictions)),
                'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
                'kappa': float(cohen_kappa_score(labels, predictions)),
                'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
                'prediction_count_class_0': int(counts[0]),
                'prediction_count_class_1': int(counts[1]),
                'collapse': bool(np.count_nonzero(counts) == 1),
                'max_prediction_class_fraction': float(counts.max() / len(labels)),
            }
            result_dir = paths['results'] / condition
            protocol.atomic_npz(result_dir / 'predictions.npz',
                                y=labels, logits=logits, probabilities=probabilities,
                                predictions=predictions,
                                uids=np.asarray(split['test']['uids'], dtype=np.int64))
            protocol.atomic_json(result_dir / 'metrics.json', metrics)
            rows.append(metrics)
            print(f'[evaluated] target=S{fold+1:02d} condition={condition} '
                  f'BA={metrics["balanced_accuracy"]:.4f} Acc={metrics["accuracy"]:.4f}',
                  flush=True)
            del model, view, state, adapter
    protocol.atomic_csv(protocol.RESULT_ROOT / 'metrics_by_subject.csv', rows)
    return rows


def _bootstrap_mean_delta(delta, draws=10000, seed=666):
    delta = np.asarray(delta, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _holm(pvalues):
    order = sorted(pvalues, key=lambda item: item[1])
    count = len(order)
    adjusted, running = {}, 0.0
    for index, (label, pvalue) in enumerate(order):
        running = max(running, (count - index) * pvalue)
        adjusted[label] = min(1.0, running)
    return adjusted


def build_report(rows, resolved):
    frame = pd.DataFrame(rows)
    if len(frame) != 54 or frame.groupby('condition')['target_subject'].nunique().to_dict() != {
            condition: 9 for condition in CONDITIONS}:
        raise RuntimeError('report requires exactly 54 completed target evaluations')
    comparisons = [
        ('ADAPTIVE_WEIGHT_KD', 'DELAYED_KD_ALL', 'adaptive_vs_kd_all'),
        ('RL_GATE_KD', 'DELAYED_KD_ALL', 'rl_vs_kd_all'),
        ('ADAPTIVE_WEIGHT_KD', 'ADAPTIVE_SHUFFLE', 'adaptive_vs_shuffle'),
        ('RL_GATE_KD', 'RL_SHUFFLE', 'rl_vs_shuffle'),
        ('RL_GATE_KD', 'ADAPTIVE_WEIGHT_KD', 'rl_vs_adaptive'),
    ]
    out = []
    primary_p = []
    for metric in ('balanced_accuracy', 'accuracy', 'kappa'):
        for method, baseline, name in comparisons:
            a = (frame[frame['condition'] == method].set_index('target_subject')[metric]
                 .sort_index())
            b = (frame[frame['condition'] == baseline].set_index('target_subject')[metric]
                 .sort_index())
            delta = (a - b).to_numpy()
            lo, hi = _bootstrap_mean_delta(
                delta, draws=int(resolved['report']['bootstrap_draws']),
                seed=int(resolved['report']['bootstrap_seed']))
            record = {'metric': metric, 'method': method, 'baseline': baseline,
                      'comparison_id': name,
                      'mean_method': float(a.mean()), 'mean_baseline': float(b.mean()),
                      'mean_delta': float(delta.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
                      'win': int(np.sum(delta > 1e-12)), 'tie': int(np.sum(np.abs(delta) <= 1e-12)),
                      'loss': int(np.sum(delta < -1e-12)),
                      'subject_deltas': json.dumps(delta.tolist())}
            if metric == 'balanced_accuracy' and name in ('adaptive_vs_kd_all', 'rl_vs_kd_all'):
                pvalue = float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue) if np.any(delta != 0) else 1.0
                record['raw_p'] = pvalue
                primary_p.append((name, pvalue))
            else:
                record['raw_p'] = None
            out.append(record)
    adjusted = _holm(primary_p)
    for row in out:
        key = row['comparison_id']
        if row['metric'] == 'balanced_accuracy' and key in adjusted:
            row['holm_p'] = adjusted[key]
        else:
            row['holm_p'] = None
    protocol.atomic_csv(protocol.RESULT_ROOT / 'paired_comparisons.csv', out)
    summary = frame.groupby('condition')[['accuracy', 'balanced_accuracy', 'kappa']].mean()
    lines = [
        '# BNCI2014004 样本价值蒸馏 pilot 报告', '',
        '协议：session_3；LOSO 目标被试仅在六组 100 epoch checkpoint 齐全后评测；seed=666。',
        '该结果是九折单 seed pilot。反馈源被试参与控制器反馈学习；MIRepNet 预训练数据暴露情况未知。', '',
        f'配置 hash：`{resolved["resolved_config_sha256"]}`', '',
        '## 逐条件均值', '',
        summary.to_markdown(floatfmt='.4f'), '',
        '## 配对比较', '',
        '| 指标 | 方法 - 对照 | mean delta | 95% bootstrap CI | Win/Tie/Loss | Holm p |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in out:
        lines.append(f'| {row["metric"]} | {row["method"]} - {row["baseline"]} | '
                     f'{row["mean_delta"]:+.4f} | [{row["ci_low"]:+.4f}, {row["ci_high"]:+.4f}] | '
                     f'{row["win"]}/{row["tie"]}/{row["loss"]} | '
                     f'{row["holm_p"] if row["holm_p"] is not None else "—"} |')
    lines += ['', '主要检验为前两项 Balanced Accuracy 配对 Wilcoxon，并对两项 p 值使用 Holm 校正。',
              '置信区间覆盖零时只报告方向。总体结果不把单 seed 视为跨 seed 稳定证据。', '']
    report_path = protocol.RESULT_ROOT / 'report.md'
    require_external_output(report_path).write_text('\n'.join(lines))
    return out


def _load_results_for_report():
    path = protocol.RESULT_ROOT / 'metrics_by_subject.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path).to_dict(orient='records')


def main():
    args = parse_args()
    device = device_for(args.gpu)
    resolved, spec, snapshot = protocol.load_resolved_config(args.config, device)
    protocol.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.smoke_only and not args.smoke_id:
        commit = subprocess.check_output(['git', 'rev-parse', '--short=12', 'HEAD'],
                                          cwd=ROOT, text=True).strip()
        args.smoke_id = f'{resolved["resolved_config_sha256"][:12]}_{commit}'
    if args.stage == 'report':
        if args.smoke_only:
            raise ValueError('report cannot consume smoke outputs')
        rows = _load_results_for_report()
        if args.assert_complete:
            source = protocol.load_data(spec, snapshot, resolved['source_hashes'])
            _assert_training_complete(resolved, spec, snapshot, device, source)
            _assert_evaluation_complete(rows, source, resolved)
        build_report(rows, resolved)
        return
    if args.stage == 'evaluate':
        evaluate_all(args, resolved, spec, snapshot, device)
        return
    if args.stage == 'all' and not args.smoke_only:
        if args.folds is not None and set(args.folds) != set(range(9)):
            raise ValueError('--stage all requires all nine folds')
    stages = [args.stage]
    if args.smoke_only:
        stages = ['all']
    source = protocol.load_data(spec, snapshot, resolved['source_hashes'])
    for stage in stages:
        run_train_stage(stage, args, resolved, spec, snapshot, source, device)


if __name__ == '__main__':
    main()
