"""CBraMod -> IFNet joint trainable-adapter V2 LOSO experiment."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS',
              'TORCH_NUM_THREADS', 'TORCH_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
import yaml
from torch.nn import functional as F

from collab.cbramod_ifnet_trainable_adapter import (
    STUDENT_DIMENSION, TEACHER_DIMENSION, TeacherToStudentAdapter,
    derive_adapter_seed, initialize_adapter, json_hash, state_hash,
)
from experiments.distill import run_cbramod_mcc_diffkd_loso as cbramod_run
from experiments.distill import run_cbramod_t2s_full_feature_loso as v1_run
from experiments.distill import run_student_aware_diffusion_loso as reference
from experiments.storage import REPORTS_ROOT, require_report_output, resolve_local_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/loso/cbramod_ifnet_trainable_adapter_v2.yaml'
SOURCE_CONFIG = ROOT / 'configs/experiments/loso/cbramod_t2s_full_feature_loso_v1.yaml'
SEEDS = (666, 667, 668)
FOLDS = tuple(range(9))
CONDITIONS = ('JOINT_A_CE_FEATURE', 'JOINT_A_MCC_FEATURE')
TEACHER_FOR = {'JOINT_A_CE_FEATURE': 'T_CE', 'JOINT_A_MCC_FEATURE': 'T_MCC'}
DIAGNOSTIC_EPOCHS = (1, 10, 50, 100)
BOOTSTRAP_DRAWS = 10000


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(resolve_local_file(path)).read_text(encoding='utf-8'))


def _atomic_json(path: Path, value: dict) -> None:
    cbramod_run._atomic_json(Path(path), value)


def _atomic_torch(path: Path, value: dict) -> None:
    cbramod_run._atomic_torch(Path(path), value)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    cbramod_run._atomic_csv(Path(path), rows)


def _atomic_npz(path: Path, **arrays) -> None:
    cbramod_run._atomic_npz(Path(path), **arrays)


def _write_progress(path: Path | None, value: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n',
                         encoding='utf-8')
    os.replace(temporary, path)


def _validate_config(config: dict) -> None:
    expected = {
        'experiment_id': 'CBRAMOD_IFNET_TRAINABLE_TEACHER_ADAPTER_LOSO_V2',
        'dataset': 'BNCI2014004', 'selected_session': 'session_3',
        'source_variant': 'wide_fre_band', 'seeds': list(SEEDS),
        'conditions': {'JOINT_A_CE_FEATURE': 'T_CE',
                      'JOINT_A_MCC_FEATURE': 'T_MCC'},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'V2 config {key} differs from the locked protocol')
    if (config['adapter']['input_dimension'] != 2400
            or config['adapter']['hidden_dimension'] != 512
            or config['adapter']['output_dimension'] != 512
            or config['student']['epochs'] != 100
            or config['student']['batch_size'] != 16
            or config['student']['feature_weight'] != 0.5
            or config['protocol']['folds'] != 9):
        raise ValueError('V2 adapter/student settings differ from the locked protocol')


def _load_config(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    _validate_config(config)
    return config


def _device(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('formal CBraMod trainable-adapter training requires CUDA')
    torch.cuda.set_device(int(gpu))
    return torch.device('cuda', int(gpu))


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(1)


def _roots(run_id: str, seed: int, fold: int) -> dict:
    return v1_run._roots(run_id, seed, fold)


def _source_context(config: dict, seed: int, fold: int, run_id: str,
                    device: torch.device, *, include_test: bool = False) -> dict:
    source_config = v1_run._read_config(SOURCE_CONFIG)
    v1_run._validate(source_config, seed)
    roots = _roots(run_id, seed, fold)
    (historical, resolved, run_config, identity, data, student, old_roots,
     old_resolved, teacher_config) = v1_run._load_fold(
        source_config, seed, fold, roots, device, include_test=include_test)
    source_id = config['teacher_source']['run_id_prefix'] + f'_seed{seed}'
    source_roots = _roots(source_id, seed, fold)
    target_path = source_roots['run_cache'] / config['teacher_source']['feature_cache']
    target_manifest_path = target_path.with_suffix('.json')
    if not target_path.is_file() or not target_manifest_path.is_file():
        raise FileNotFoundError(f'missing V1 full-feature teacher target: {target_path}')
    target_meta = _read_json(target_manifest_path)
    if (target_meta.get('status') != 'complete'
            or target_meta.get('file_sha256') != _sha(target_path)
            or target_meta.get('feature_tap') !=
               'CBraMod backbone [B,3,4,200], reshape only to [B,2400]; no pooling/mean'):
        raise RuntimeError(f'V1 teacher-target identity/hash mismatch: {target_path}')
    complete = _read_json(source_roots['run_result'] / 'fold_complete.json')
    if complete.get('status') != 'trained':
        raise RuntimeError(f'V1 source fold is incomplete: seed={seed}, fold={fold + 1}')
    for name in ('T_CE', 'T_MCC'):
        teacher_manifest = _read_json(source_roots['run_result'] / f'teacher_{name}.json')
        checkpoint = source_roots['run_weights'] / f'teacher_{name}.pt'
        if (not checkpoint.is_file() or teacher_manifest.get('status') != 'complete'
                or teacher_manifest.get('checkpoint_sha256') != _sha(checkpoint)
                or teacher_manifest.get('model_sha256') != target_meta.get(
                    'teacher_state_sha256', {}).get(name)):
            raise RuntimeError(f'V1 {name} checkpoint does not match its target cache')
        payload = torch.load(resolve_local_file(checkpoint), map_location='cpu',
                             weights_only=False)
        if cbramod_run._state_hash(payload['model']) != teacher_manifest['model_sha256']:
            raise RuntimeError(f'V1 {name} model-state hash mismatch')
    with np.load(resolve_local_file(target_path), allow_pickle=False) as saved:
        expected_shapes = {
            'train_uids': tuple(identity['uids']['train'].shape),
            'train_y': tuple(identity['labels']['train'].shape),
            'feedback_uids': tuple(identity['uids']['feedback'].shape),
            'feedback_y': tuple(identity['labels']['feedback'].shape),
        }
        for key, shape in expected_shapes.items():
            if tuple(saved[key].shape) != shape:
                raise RuntimeError(f'V1 cache {key} shape differs from split')
        if (not np.array_equal(saved['train_uids'], identity['uids']['train'])
                or not np.array_equal(saved['train_y'], identity['labels']['train'])
                or not np.array_equal(saved['feedback_uids'], identity['uids']['feedback'])
                or not np.array_equal(saved['feedback_y'], identity['labels']['feedback'])):
            raise RuntimeError(f'V1 teacher targets are not UID/label aligned: {target_path}')
        for name in ('T_CE', 'T_MCC'):
            if tuple(saved[f'{name}_train_features_full'].shape) != (
                    len(identity['uids']['train']), TEACHER_DIMENSION):
                raise RuntimeError(f'V1 {name} train feature shape mismatch')
    source_eval = source_roots['run_result'] / 'fold_evaluation_metrics.csv'
    source_eval_manifest = _read_json(source_roots['run_result'] / 'fold_evaluation.json')
    if (not source_eval.is_file() or source_eval_manifest.get('status') != 'complete'
            or source_eval_manifest.get('metrics_sha256') != _sha(source_eval)):
        raise RuntimeError('V1 target metrics are missing or do not match their completion record')
    v1_fold = _read_json(source_roots['run_result'] / 'fold_complete.json')
    initial_hash = student['initial_student']['model_sha256']
    if (v1_fold.get('initial_IFNet_sha256') != initial_hash
            or v1_fold.get('student_schedule_sha256') != student['student_schedule_sha256']):
        raise RuntimeError('V1 student initialization or UID schedule is not paired')
    return {
        'roots': roots, 'source_roots': source_roots, 'target_path': target_path,
        'target_meta': target_meta, 'identity': identity, 'data': data,
        'student': student, 'historical': historical, 'resolved': resolved,
        'source_eval_path': source_eval, 'source_eval_manifest': source_eval_manifest,
        'old_resolved': old_resolved, 'teacher_config': teacher_config,
        'source_target_sha256': _sha(target_path),
    }


def _fold_fingerprint(config_path: Path, config: dict, seed: int, fold: int,
                      context: dict) -> str:
    return json_hash({
        'experiment_id': config['experiment_id'], 'config_sha256': _sha(config_path),
        'runner_sha256': _sha(Path(__file__)),
        'adapter_core_sha256': _sha(ROOT / 'collab/cbramod_ifnet_trainable_adapter.py'),
        'source_runner_sha256': _sha(ROOT / 'experiments/distill/run_cbramod_t2s_full_feature_loso.py'),
        'seed': seed, 'fold': fold,
        'source_split_manifest_sha256': context['identity']['sha256'],
        'source_resolved_config_sha256': context['resolved']['resolved_config_sha256'],
        'source_teacher_target_sha256': context['source_target_sha256'],
        'teacher_state_sha256': context['target_meta']['teacher_state_sha256'],
        'initial_IFNet_sha256': context['student']['initial_student']['model_sha256'],
        'student_schedule_sha256': context['student']['student_schedule_sha256'],
        'feature_shape': [3, 4, 200], 'teacher_dimension': TEACHER_DIMENSION,
        'student_dimension': STUDENT_DIMENSION,
    })


def _adapter_and_state(seed: int, fold: int, config: dict,
                       device: torch.device):
    adapter_seed = derive_adapter_seed(config['experiment_id'], seed, fold)
    adapter, adapter_hash = initialize_adapter(adapter_seed)
    state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
    if sum(value.numel() for value in state.values()) != 1_491_968:
        raise RuntimeError('trainable adapter parameter count differs from the locked design')
    return adapter.to(device), state, adapter_hash, adapter_seed


def _effective_rank(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh((x.T @ x) / max(1, len(x) - 1))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def _centroid_distance(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(values), np.asarray(labels)
    centers = [x[y == cls].mean(axis=0) for cls in (0, 1) if np.any(y == cls)]
    inter = float(np.linalg.norm(centers[0] - centers[1])) if len(centers) == 2 else float('nan')
    within = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            within.extend(np.linalg.norm(x[mask] - x[mask].mean(axis=0), axis=1).tolist())
    return inter, float(np.mean(within)) if within else float('nan')


def _evaluate_partition(model, student_adapter, adapter, x: np.ndarray, y: np.ndarray,
                        teacher_features: np.ndarray, device: torch.device) -> dict:
    state = reference._cpu_tree(model.state_dict())
    before = reference._state_hash(state)
    rng = reference._capture_rng(device)
    model_mode, adapter_mode = model.training, adapter.training
    try:
        model.eval(); adapter.eval()
        logits_parts, student_parts, teacher_parts = [], [], []
        with torch.no_grad():
            for start in range(0, len(x), 64):
                sl = slice(start, start + 64)
                xb = torch.as_tensor(x[sl], dtype=torch.float32, device=device)
                native, logits = student_adapter.forward(model, xb)
                z_student = F.layer_norm(native, (512,), weight=None, bias=None, eps=1e-5)
                target = torch.as_tensor(teacher_features[sl], dtype=torch.float32, device=device)
                z_adapter = adapter(target)
                logits_parts.append(logits.float().cpu().numpy())
                student_parts.append(z_student.float().cpu().numpy())
                teacher_parts.append(z_adapter.float().cpu().numpy())
        logits = np.concatenate(logits_parts)
        z_student = np.concatenate(student_parts)
        z_adapter = np.concatenate(teacher_parts)
        mse = float(np.mean((z_student - z_adapter) ** 2))
        metrics = cbramod_run._metric_values(y, logits)
        inter_s, intra_s = _centroid_distance(z_student, y)
        inter_a, intra_a = _centroid_distance(z_adapter, y)
        return {
            **metrics, 'feature_mse': mse,
            'student_feature_variance': float(np.var(z_student, axis=0).mean()),
            'adapter_feature_variance': float(np.var(z_adapter, axis=0).mean()),
            'student_effective_rank': _effective_rank(z_student),
            'adapter_effective_rank': _effective_rank(z_adapter),
            'student_interclass_centroid_distance': inter_s,
            'student_intraclass_radius': intra_s,
            'adapter_interclass_centroid_distance': inter_a,
            'adapter_intraclass_radius': intra_a,
            '_adapter_features': z_adapter,
        }
    finally:
        if reference._state_hash(model.state_dict()) != before:
            model.load_state_dict(state, strict=True)
        reference._restore_rng(rng, device)
        model.train(model_mode)
        adapter.train(adapter_mode)
        if reference._state_hash(model.state_dict()) != before:
            raise RuntimeError('read-only feedback/train diagnostic mutated IFNet state')


def _gradient_cosine(loss_a: torch.Tensor, loss_b: torch.Tensor,
                     parameter: torch.nn.Parameter) -> float:
    grad_a = torch.autograd.grad(loss_a, parameter, retain_graph=True,
                                 allow_unused=True)[0]
    grad_b = torch.autograd.grad(loss_b, parameter, retain_graph=True,
                                 allow_unused=True)[0]
    if grad_a is None or grad_b is None:
        return float('nan')
    a, b = grad_a.detach().flatten(), grad_b.detach().flatten()
    denom = a.norm() * b.norm()
    if float(denom) <= 0:
        return float('nan')
    return float(torch.dot(a, b).div(denom).cpu())


def _save_training_state(path: Path, *, model, adapter, optimizer_student,
                         optimizer_adapter, scheduler_student, scheduler_adapter,
                         epoch: int, fingerprint: str, adapter_init_hash: str,
                         device: torch.device) -> None:
    _atomic_torch(path, {
        'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'adapter': {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
        'optimizer_student': optimizer_student.state_dict(),
        'optimizer_adapter': optimizer_adapter.state_dict(),
        'scheduler_student': scheduler_student.state_dict(),
        'scheduler_adapter': scheduler_adapter.state_dict(),
        'epoch': int(epoch), 'run_fingerprint': fingerprint,
        'adapter_init_hash': adapter_init_hash,
        'rng_state': reference._capture_rng(device),
    })


def _fit_condition(condition: str, config: dict, context: dict, seed: int,
                   fold: int, fingerprint: str, adapter_init_state: dict,
                   adapter_init_hash: str, adapter_seed: int,
                   device: torch.device, *, smoke: bool = False,
                   progress_path: Path | None = None,
                   stop_file: Path | None = None) -> dict:
    roots, identity, data, student = (context['roots'], context['identity'],
                                      context['data'], context['student'])
    teacher = TEACHER_FOR[condition]
    target_meta = context['target_meta']
    with np.load(resolve_local_file(context['target_path']), allow_pickle=False) as saved:
        train_features = np.asarray(saved[f'{teacher}_train_features_full'], dtype=np.float32).copy()
        feedback_features = np.asarray(saved[f'{teacher}_feedback_features_full'], dtype=np.float32).copy()
    if train_features.shape != (len(identity['uids']['train']), TEACHER_DIMENSION):
        raise RuntimeError('cached teacher train features have an unexpected shape')
    if feedback_features.shape != (len(identity['uids']['feedback']), TEACHER_DIMENSION):
        raise RuntimeError('cached teacher feedback features have an unexpected shape')
    expected_teacher_hash = target_meta['teacher_state_sha256'][teacher]
    condition_fp = json_hash({
        'run_fingerprint': fingerprint, 'condition': condition, 'teacher': teacher,
        'teacher_state_sha256': expected_teacher_hash,
        'teacher_target_sha256': context['source_target_sha256'],
        'initial_adapter_hash': adapter_init_hash,
        'initial_IFNet_hash': student['initial_student']['model_sha256'],
        'schedule_sha256': student['student_schedule_sha256'],
        'epochs': 1 if smoke else config['student']['epochs'],
    })
    final_student = roots['run_weights'] / f'student_{condition}.pt'
    final_adapter = roots['run_weights'] / f'adapter_{condition}.pt'
    manifest_path = roots['run_result'] / f'{condition}.json'
    state_path = roots['run_weights'] / f'training_state_{condition}.pt'
    if not smoke and (manifest_path.exists() or final_student.exists() or final_adapter.exists()):
        if not (manifest_path.is_file() and final_student.is_file() and final_adapter.is_file()):
            raise RuntimeError(f'{condition} has partial terminal artifacts; refusing implicit retrain')
        manifest = _read_json(manifest_path)
        if (manifest.get('status') != 'complete'
                or manifest.get('method_fingerprint') != condition_fp
                or manifest.get('student_checkpoint_sha256') != _sha(final_student)
                or manifest.get('adapter_checkpoint_sha256') != _sha(final_adapter)):
            raise RuntimeError(f'{condition} terminal identity/hash mismatch')
        return manifest

    student_model, optimizer_student, scheduler_student, _ = reference._new_paired_student(
        student, student['initial_student'], seed, device)
    teacher_adapter, actual_init_state, actual_init_hash, actual_adapter_seed = _adapter_and_state(
        seed, fold, config, device)
    if actual_adapter_seed != adapter_seed or actual_init_hash != adapter_init_hash:
        raise RuntimeError('adapter initialization changed between paired conditions')
    teacher_adapter.load_state_dict(adapter_init_state, strict=True)
    teacher_adapter.train()
    optimizer_adapter = torch.optim.AdamW(
        teacher_adapter.parameters(), lr=float(config['adapter']['lr']),
        betas=tuple(config['adapter']['betas']), eps=float(config['adapter']['eps']),
        weight_decay=float(config['adapter']['weight_decay']), foreach=False, fused=False)
    scheduler_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adapter, T_max=int(config['adapter']['scheduler_t_max']),
        eta_min=float(config['adapter']['scheduler_eta_min']))
    model_optimizer_cfg = config['student']
    optimizer_student = torch.optim.AdamW(
        student_model.parameters(), lr=float(model_optimizer_cfg['lr']),
        betas=tuple(model_optimizer_cfg['betas']), eps=float(model_optimizer_cfg['eps']),
        weight_decay=float(model_optimizer_cfg['weight_decay']), foreach=False, fused=False)
    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=int(model_optimizer_cfg['scheduler_t_max']),
        eta_min=float(model_optimizer_cfg['scheduler_eta_min']))

    train_x = torch.as_tensor(data['student_x']['train'], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(identity['labels']['train'], dtype=torch.long, device=device)
    feature_tensor = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    history_path = roots['run_result'] / f'{condition}_history.csv'
    diagnostics_path = roots['run_result'] / f'{condition}_diagnostics.csv'
    history, diagnostics = [], []
    start_epoch = 0
    if state_path.is_file():
        saved = torch.load(resolve_local_file(state_path), map_location='cpu', weights_only=False)
        if (saved.get('run_fingerprint') != fingerprint
                or saved.get('adapter_init_hash') != adapter_init_hash):
            raise RuntimeError(f'{condition} resume state belongs to another run')
        student_model.load_state_dict(saved['model'], strict=True)
        teacher_adapter.load_state_dict(saved['adapter'], strict=True)
        optimizer_student.load_state_dict(saved['optimizer_student'])
        optimizer_adapter.load_state_dict(saved['optimizer_adapter'])
        scheduler_student.load_state_dict(saved['scheduler_student'])
        scheduler_adapter.load_state_dict(saved['scheduler_adapter'])
        start_epoch = int(saved['epoch'])
        reference._restore_rng(saved['rng_state'], device)
        if history_path.is_file():
            history = pd.read_csv(resolve_local_file(history_path)).to_dict('records')
        if diagnostics_path.is_file():
            diagnostics = pd.read_csv(resolve_local_file(diagnostics_path)).to_dict('records')
    schedule = student['student_schedule']
    epochs = 1 if smoke else int(config['student']['epochs'])
    stem_parameter = next(student_model.stem.parameters())
    reference_path = roots['run_cache'] / f'{condition}_adapter_epoch1_reference.npz'
    train_reference = feedback_reference = None
    if reference_path.is_file() and not smoke:
        with np.load(resolve_local_file(reference_path), allow_pickle=False) as saved:
            train_reference = np.asarray(saved['train'], dtype=np.float32).copy()
            feedback_reference = np.asarray(saved['feedback'], dtype=np.float32).copy()
    for epoch_index in range(start_epoch, epochs):
        student_model.train(); teacher_adapter.train()
        sums = {'ce': 0.0, 'feature': 0.0, 'total': 0.0, 'student_grad': 0.0,
                'adapter_grad': 0.0}
        count = steps = 0
        angle_value = float('nan')
        epoch_batches = schedule[epoch_index][:2] if smoke else schedule[epoch_index]
        for batch_index, ids_np in enumerate(epoch_batches):
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            xb, yb, teacher_batch = train_x[ids], train_y[ids], feature_tensor[ids]
            optimizer_student.zero_grad(set_to_none=True)
            optimizer_adapter.zero_grad(set_to_none=True)
            native, logits = student['student_adapter'].forward(student_model, xb)
            if native.shape != (len(ids_np), STUDENT_DIMENSION):
                raise RuntimeError(f'IFNet feature tap changed: {tuple(native.shape)}')
            z_student = F.layer_norm(native, (STUDENT_DIMENSION,), weight=None,
                                     bias=None, eps=float(config['student']['layer_norm_epsilon']))
            # Teacher feature is fixed; z_adapter remains attached to A_theta.
            z_adapter = teacher_adapter(teacher_batch.detach())
            feature_loss = F.mse_loss(z_student, z_adapter, reduction='mean')
            ce = F.cross_entropy(logits, yb)
            weighted_feature = float(config['student']['feature_weight']) * feature_loss
            loss = ce + weighted_feature
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{condition} non-finite loss at epoch {epoch_index + 1}')
            if batch_index == 0 and (epoch_index + 1) in DIAGNOSTIC_EPOCHS:
                angle_value = _gradient_cosine(ce, weighted_feature, stem_parameter)
            loss.backward()
            student_norm = torch.linalg.vector_norm(torch.stack([
                p.grad.detach().norm() for p in student_model.parameters() if p.grad is not None]))
            adapter_norm = torch.nn.utils.clip_grad_norm_(
                teacher_adapter.parameters(), float(config['adapter']['gradient_clip_norm']))
            if (not torch.isfinite(student_norm) or not torch.isfinite(adapter_norm)
                    or float(adapter_norm) <= 0):
                raise FloatingPointError(f'{condition} invalid student/adapter gradient')
            if epoch_index == start_epoch and batch_index == 0:
                if not any(p.grad is not None and torch.isfinite(p.grad).all()
                           and float(p.grad.norm()) > 0 for p in student_model.stem.parameters()):
                    raise RuntimeError('feature loss did not reach the IFNet stem')
                if not any(p.grad is not None and torch.isfinite(p.grad).all()
                           and float(p.grad.norm()) > 0 for p in teacher_adapter.parameters()):
                    raise RuntimeError('feature loss did not reach the trainable adapter')
            optimizer_student.step(); optimizer_adapter.step()
            n = len(ids_np)
            sums['ce'] += float(ce.detach().cpu()) * n
            sums['feature'] += float(feature_loss.detach().cpu()) * n
            sums['total'] += float(loss.detach().cpu()) * n
            sums['student_grad'] += float(student_norm.detach().cpu())
            sums['adapter_grad'] += float(adapter_norm.detach().cpu())
            count += n; steps += 1
        scheduler_student.step(); scheduler_adapter.step()
        row = {
            'condition': condition, 'teacher': teacher, 'seed': seed,
            'target_subject': fold + 1, 'epoch': epoch_index + 1,
            'train_ce': sums['ce'] / count, 'train_feature_mse': sums['feature'] / count,
            'weighted_feature_mse': float(config['student']['feature_weight']) * sums['feature'] / count,
            'train_total': sums['total'] / count,
            'mean_student_gradient_norm': sums['student_grad'] / steps,
            'mean_adapter_gradient_norm_preclip': sums['adapter_grad'] / steps,
            'ce_feature_stem_gradient_cosine': angle_value,
            'student_lr_next': float(scheduler_student.get_last_lr()[0]),
            'adapter_lr_next': float(scheduler_adapter.get_last_lr()[0]),
            'adapter_init_hash': adapter_init_hash,
            'train_uid_sha256': identity['uid_sha256']['train'],
            'schedule_sha256': student['student_schedule_sha256'],
        }
        history.append(row)
        if smoke:
            continue
        if (epoch_index + 1) in DIAGNOSTIC_EPOCHS:
            train_diag = _evaluate_partition(
                student_model, student['student_adapter'], teacher_adapter,
                data['student_x']['train'], identity['labels']['train'],
                train_features, device)
            feedback_diag = _evaluate_partition(
                student_model, student['student_adapter'], teacher_adapter,
                data['student_x']['feedback'], identity['labels']['feedback'],
                feedback_features, device)
            if train_reference is None:
                train_reference = train_diag['_adapter_features']
            if feedback_reference is None:
                feedback_reference = feedback_diag['_adapter_features']
                _atomic_npz(reference_path, train=train_reference,
                            feedback=feedback_reference)
            for partition, result, reference_features in (
                ('D_train', train_diag, train_reference),
                ('D_feedback', feedback_diag, feedback_reference)):
                adapted = result.pop('_adapter_features')
                drift = float(np.sqrt(np.mean((adapted - reference_features) ** 2)))
                diagnostics.append({
                    'condition': condition, 'teacher': teacher, 'seed': seed,
                    'target_subject': fold + 1, 'epoch': epoch_index + 1,
                    'partition': partition, **result,
                    'adapter_target_drift_rmse_from_epoch1': drift,
                    'adapter_init_hash': adapter_init_hash,
                })
        _atomic_csv(history_path, history)
        if diagnostics:
            _atomic_csv(diagnostics_path, diagnostics)
        # Checkpoints are written every ten epochs and on a requested pause.
        stop_requested = bool(stop_file and stop_file.exists())
        if (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs or stop_requested:
            _save_training_state(
                state_path, model=student_model, adapter=teacher_adapter,
                optimizer_student=optimizer_student, optimizer_adapter=optimizer_adapter,
                scheduler_student=scheduler_student, scheduler_adapter=scheduler_adapter,
                epoch=epoch_index + 1, fingerprint=fingerprint,
                adapter_init_hash=adapter_init_hash, device=device)
        if stop_requested:
            _write_progress(progress_path, {
                'status': 'paused', 'seed': seed, 'fold': fold,
                'current_condition': condition, 'current_epoch': epoch_index + 1,
                'records': [{'condition': condition, 'epoch': epoch_index + 1}],
                'total_cells': 2,
            })
            if smoke:
                return {'status': 'paused'}
            raise SystemExit(75)

    if smoke:
        # A smoke test proves only that the intended graph has finite gradients.
        return {'status': 'smoke_passed', 'condition': condition,
                'adapter_init_hash': adapter_init_hash,
                'adapter_parameter_count': sum(p.numel() for p in teacher_adapter.parameters()),
                'student_gradient_reached': True, 'adapter_gradient_reached': True}

    student_state = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}
    adapter_state = {k: v.detach().cpu().clone() for k, v in teacher_adapter.state_dict().items()}
    _atomic_torch(final_student, {'model': student_state, 'model_sha256': reference._state_hash(student_state),
                                 'method_fingerprint': condition_fp, 'condition': condition})
    _atomic_torch(final_adapter, {'model': adapter_state, 'model_sha256': state_hash(adapter_state),
                                 'method_fingerprint': condition_fp, 'condition': condition,
                                 'initial_state_sha256': adapter_init_hash,
                                 'adapter_seed': adapter_seed})
    manifest = {
        'status': 'complete', 'condition': condition, 'teacher': teacher,
        'run_fingerprint': fingerprint, 'method_fingerprint': condition_fp,
        'seed': seed, 'fold_zero_based': fold, 'target_subject': fold + 1,
        'final_epoch': epochs, 'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'student_schedule_sha256': student['student_schedule_sha256'],
        'train_uid_sha256': identity['uid_sha256']['train'],
        'teacher_target_sha256': context['source_target_sha256'],
        'teacher_state_sha256': expected_teacher_hash,
        'adapter_architecture': config['adapter']['architecture'],
        'adapter_parameter_count': sum(p.numel() for p in teacher_adapter.parameters()),
        'adapter_seed': adapter_seed, 'adapter_initial_sha256': adapter_init_hash,
        'adapter_final_sha256': state_hash(adapter_state),
        'adapter_checkpoint': str(final_adapter), 'adapter_checkpoint_sha256': _sha(final_adapter),
        'student_checkpoint': str(final_student), 'student_checkpoint_sha256': _sha(final_student),
        'loss': {'classification': 'cross_entropy', 'feature': 'MSE(LN(h_S), A_theta(stopgrad(f_T)))',
                 'feature_weight': config['student']['feature_weight'],
                 'teacher_logits_kd': False, 'start_epoch': 1},
        'history': str(history_path), 'diagnostics': str(diagnostics_path),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _atomic_json(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    return manifest


def _train_fold(config_path: Path, config: dict, seed: int, fold: int,
                run_id: str, gpu: int, *, smoke: bool = False,
                progress_path: Path | None = None, stop_file: Path | None = None) -> dict:
    if seed not in SEEDS or fold not in FOLDS:
        raise ValueError('seed/fold is outside the locked three-seed nine-fold scope')
    device = _device(gpu)
    context = _source_context(config, seed, fold, run_id, device, include_test=False)
    fingerprint = _fold_fingerprint(config_path, config, seed, fold, context)
    roots, identity, student = context['roots'], context['identity'], context['student']
    if not smoke:
        resolved_path = roots['run_result'] / 'resolved_run.json'
        record = {
            'status': 'training', 'experiment_id': config['experiment_id'],
            'run_fingerprint': fingerprint, 'config': str(config_path),
            'config_sha256': _sha(config_path), 'seed': seed, 'fold_zero_based': fold,
            'target_subject': fold + 1,
            'feedback_subject': int(identity['manifest']['feedback_subject_id']),
            'train_subject_ids': identity['manifest']['train_subject_ids'],
            'split_manifest': str(roots['split_dir']),
            'split_manifest_sha256': identity['sha256'],
            'train_uid_sha256': identity['uid_sha256']['train'],
            'feedback_uid_sha256': identity['uid_sha256']['feedback'],
            'source_teacher_run': str(context['source_roots']['run_result']),
            'source_teacher_target': str(context['target_path']),
            'source_teacher_target_sha256': context['source_target_sha256'],
            'initial_IFNet_sha256': student['initial_student']['model_sha256'],
            'student_schedule_sha256': student['student_schedule_sha256'],
            'target_evaluation': 'deferred until all 27 folds complete',
        }
        path = roots['run_result'] / 'resolved_run.json'
        if path.exists():
            existing = _read_json(path)
            if existing.get('run_fingerprint') != fingerprint:
                raise RuntimeError('existing V2 fold record belongs to another run')
        else:
            _atomic_json(path, record)
        complete_path = roots['run_result'] / 'fold_complete.json'
        if complete_path.exists():
            complete = _read_json(complete_path)
            if complete.get('run_fingerprint') != fingerprint or complete.get('status') != 'trained':
                raise RuntimeError('existing V2 fold completion belongs to another identity')
            return {'status': 'trained', 'resumed_existing': True, 'fold': fold}

    initial_adapter, adapter_init_state, adapter_init_hash, adapter_seed = _adapter_and_state(
        seed, fold, config, device)
    del initial_adapter
    results = {}
    for condition in CONDITIONS:
        result = _fit_condition(
            condition, config, context, seed, fold, fingerprint,
            adapter_init_state, adapter_init_hash, adapter_seed, device,
            smoke=smoke, progress_path=progress_path, stop_file=stop_file)
        results[condition] = result
        if smoke:
            continue
        _write_progress(progress_path, {
            'status': 'running', 'seed': seed, 'fold': fold,
            'records': [{'condition': name, 'status': 'complete'}
                        for name in results], 'total_cells': 2,
        })
    if smoke:
        result = {
            'status': 'passed', 'seed': seed, 'fold': fold,
            'run_fingerprint': fingerprint,
            'teacher_target_sha256': context['source_target_sha256'],
            'conditions': results, 'target_evaluated': False,
        }
        smoke_root = roots['run_result'] / 'smoke'
        smoke_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(smoke_root / f'smoke_seed{seed}_fold{fold + 1:02d}.json', result)
        _write_progress(progress_path, {'status': 'completed', 'seed': seed,
                                        'fold': fold, 'records': [{'smoke': 'passed'}],
                                        'total_cells': 1})
        return result
    _atomic_json(roots['run_result'] / 'fold_complete.json', {
        'status': 'trained', 'run_fingerprint': fingerprint,
        'seed': seed, 'fold_zero_based': fold, 'target_subject': fold + 1,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'student_schedule_sha256': student['student_schedule_sha256'],
        'conditions': {name: results[name]['student_checkpoint_sha256'] for name in CONDITIONS},
        'adapter_checkpoints': {name: results[name]['adapter_checkpoint_sha256'] for name in CONDITIONS},
        'source_teacher_target_sha256': context['source_target_sha256'],
        'target_evaluation': 'deferred',
    })
    _write_progress(progress_path, {'status': 'completed', 'seed': seed,
                                    'fold': fold, 'records': [
                                        {'condition': name, 'status': 'complete'}
                                        for name in CONDITIONS], 'total_cells': 2})
    return {'status': 'trained', 'fold': fold, 'run_fingerprint': fingerprint}


def _target_evaluate_fold(config: dict, seed: int, fold: int,
                          run_id: str, device: torch.device) -> tuple[list[dict], list[dict]]:
    context = _source_context(config, seed, fold, run_id, device, include_test=True)
    roots, identity, data, student = (context['roots'], context['identity'],
                                      context['data'], context['student'])
    complete = _read_json(roots['run_result'] / 'fold_complete.json')
    if complete.get('status') != 'trained':
        raise RuntimeError(f'cannot evaluate incomplete training fold seed={seed},fold={fold+1}')
    metrics_path = roots['run_result'] / 'fold_evaluation_metrics.csv'
    predictions_path = roots['run_result'] / 'fold_predictions.csv'
    eval_path = roots['run_result'] / 'fold_evaluation.json'
    if all(path.is_file() for path in (metrics_path, predictions_path, eval_path)):
        meta = _read_json(eval_path)
        if (meta.get('status') != 'complete'
                or meta.get('run_fingerprint') != complete.get('run_fingerprint')
                or meta.get('metrics_sha256') != _sha(metrics_path)
                or meta.get('predictions_sha256') != _sha(predictions_path)):
            raise RuntimeError('saved V2 target evaluation identity/hash mismatch')
        return (pd.read_csv(resolve_local_file(metrics_path)).to_dict('records'),
                pd.read_csv(resolve_local_file(predictions_path)).to_dict('records'))
    if any(p.exists() for p in (metrics_path, predictions_path, eval_path)):
        raise RuntimeError('partial V2 target evaluation exists; refusing to overwrite')
    source_roots = context['source_roots']
    if identity['uid_sha256']['test'] != context['source_eval_manifest'].get('test_uid_sha256'):
        raise RuntimeError('test UID set differs from the previously locked V1 split')
    source_metrics = pd.read_csv(resolve_local_file(source_roots['run_result'] /
                                                     'fold_evaluation_metrics.csv'))
    matched_source = source_metrics[source_metrics['model'].isin(
        ['IFNet_CE', 'CE_T2S_FEATURE', 'MCC_T2S_FEATURE'])]
    if set(matched_source['model']) != {'IFNet_CE', 'CE_T2S_FEATURE', 'MCC_T2S_FEATURE'}:
        raise RuntimeError('V1 CE/fixed-P comparison rows are missing')
    metric_rows, prediction_rows = [], []
    for _, row in matched_source.iterrows():
        metric_rows.append({
            'model': row['model'], 'seed': seed, 'target_subject': fold + 1,
            'accuracy': float(row['accuracy']), 'balanced_accuracy': float(row['balanced_accuracy']),
            'kappa': float(row['kappa']), 'cross_entropy': float(row['cross_entropy']),
            'source': 'cbramod_t2s_full_feature_loso_v1',
        })
    target_rows = data['student_rows']['test']
    raw = np.asarray(data['student_input_mmap'][target_rows], dtype=np.float32)
    x_test = reference._preprocess_student(data['student_adapter'], raw)
    y_test = np.asarray(identity['labels']['test'], dtype=np.int64)
    uids = np.asarray(identity['uids']['test'], dtype=np.int64)
    for condition in CONDITIONS:
        manifest = _read_json(roots['run_result'] / f'{condition}.json')
        checkpoint = roots['run_weights'] / f'student_{condition}.pt'
        if (manifest.get('status') != 'complete'
                or manifest.get('student_checkpoint_sha256') != _sha(checkpoint)
                or manifest.get('run_fingerprint') != complete.get('run_fingerprint')):
            raise RuntimeError(f'{condition} checkpoint is missing or mismatched')
        payload = torch.load(resolve_local_file(checkpoint), map_location='cpu', weights_only=False)
        model = student['student_adapter'].build(2)
        model.load_state_dict(payload['model'], strict=True)
        model.to(device)
        logits, actual_hash = cbramod_run._predict_student_preserving_state(
            model, student['student_adapter'], x_test, device)
        if actual_hash != payload['model_sha256']:
            raise RuntimeError(f'{condition} model state changed during target prediction')
        values = cbramod_run._metric_values(y_test, logits)
        metric_rows.append({'model': condition, 'seed': seed,
            'target_subject': fold + 1, **values, 'source': 'new V2 target evaluation'})
        for index, (uid, label, row_logits) in enumerate(zip(uids, y_test, logits)):
            prediction_rows.append({
                'model': condition, 'seed': seed, 'target_subject': fold + 1,
                'trial_uid': f'{int(uid[0])}:{int(uid[1])}', 'trial_order': index,
                'label': int(label), 'logit_0': float(row_logits[0]),
                'logit_1': float(row_logits[1]), 'prediction': int(np.argmax(row_logits)),
            })
    _atomic_csv(metrics_path, metric_rows)
    _atomic_csv(predictions_path, prediction_rows)
    _atomic_json(eval_path, {
        'status': 'complete', 'run_fingerprint': complete['run_fingerprint'],
        'seed': seed, 'fold_zero_based': fold, 'target_subject': fold + 1,
        'test_uid_sha256': identity['uid_sha256']['test'], 'n_trials': len(y_test),
        'metrics_sha256': _sha(metrics_path), 'predictions_sha256': _sha(predictions_path),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    return metric_rows, prediction_rows


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 512):
        n = min(512, BOOTSTRAP_DRAWS - start)
        ids = rng.integers(0, len(x), size=(n, len(x)))
        means[start:start + n] = x[ids].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_summary(config: dict, run_id_prefix: str) -> dict:
    metric_rows, prediction_rows = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            roots = _roots(run_id_prefix + f'_seed{seed}', seed, fold)
            eval_path = roots['run_result'] / 'fold_evaluation.json'
            if not eval_path.is_file():
                raise FileNotFoundError(f'missing target evaluation: seed={seed}, fold={fold+1}')
            meta = _read_json(eval_path)
            metric_path, prediction_path = (roots['run_result'] / 'fold_evaluation_metrics.csv',
                                            roots['run_result'] / 'fold_predictions.csv')
            if meta.get('metrics_sha256') != _sha(metric_path) or meta.get('predictions_sha256') != _sha(prediction_path):
                raise RuntimeError('target evaluation files differ from completion hash')
            metric_rows.extend(pd.read_csv(resolve_local_file(metric_path)).to_dict('records'))
            prediction_rows.extend(pd.read_csv(resolve_local_file(prediction_path)).to_dict('records'))
    metrics = pd.DataFrame(metric_rows)
    expected_models = {'IFNet_CE', 'CE_T2S_FEATURE', 'MCC_T2S_FEATURE', *CONDITIONS}
    if set(metrics['model'].unique()) != expected_models or len(metrics) != 27 * len(expected_models):
        raise RuntimeError('target metrics do not contain the complete 27-fold condition matrix')
    rows = []
    for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy'):
        pivot = metrics.pivot(index=['seed', 'target_subject'], columns='model', values=metric)
        by_subject = pivot.groupby('target_subject').mean()
        for model in sorted(expected_models):
            seed_means = pivot[model].groupby('seed').mean()
            subject_values = by_subject[model].to_numpy()
            lo, hi = _bootstrap(subject_values, config['report']['bootstrap_seed'])
            rows.append({
                'metric': metric, 'model': model,
                'mean_percent': float(pivot[model].mean() * 100) if metric != 'cross_entropy' else float(pivot[model].mean()),
                'subject_sd_percent': float(by_subject[model].std(ddof=1) * 100) if metric != 'cross_entropy' else float(by_subject[model].std(ddof=1)),
                'subject_bootstrap_ci_low': lo * (100 if metric != 'cross_entropy' else 1),
                'subject_bootstrap_ci_high': hi * (100 if metric != 'cross_entropy' else 1),
                'seed_666_mean': float(seed_means.get(666, np.nan) * (100 if metric != 'cross_entropy' else 1)),
                'seed_667_mean': float(seed_means.get(667, np.nan) * (100 if metric != 'cross_entropy' else 1)),
                'seed_668_mean': float(seed_means.get(668, np.nan) * (100 if metric != 'cross_entropy' else 1)),
                'n_seed': 3, 'n_subject': 9,
            })
    paired_rows = []
    comparison_map = {
        'JOINT_A_CE_FEATURE_vs_FIXED_P_CE_FEATURE': ('JOINT_A_CE_FEATURE', 'CE_T2S_FEATURE'),
        'JOINT_A_MCC_FEATURE_vs_FIXED_P_MCC_FEATURE': ('JOINT_A_MCC_FEATURE', 'MCC_T2S_FEATURE'),
        'JOINT_A_CE_FEATURE_vs_IFNet_CE': ('JOINT_A_CE_FEATURE', 'IFNet_CE'),
        'JOINT_A_MCC_FEATURE_vs_IFNet_CE': ('JOINT_A_MCC_FEATURE', 'IFNet_CE'),
        'JOINT_A_MCC_FEATURE_vs_JOINT_A_CE_FEATURE': ('JOINT_A_MCC_FEATURE', 'JOINT_A_CE_FEATURE'),
    }
    for metric in ('accuracy', 'balanced_accuracy'):
        pivot = metrics.pivot(index=['seed', 'target_subject'], columns='model', values=metric)
        subj = pivot.groupby('target_subject').mean()
        for comparison, (left, right) in comparison_map.items():
            differences = (subj[left] - subj[right]).to_numpy() * 100
            lo, hi = _bootstrap(differences, config['report']['bootstrap_seed'])
            paired_rows.append({
                'metric': metric, 'comparison': comparison,
                'mean_difference_pp': float(differences.mean()),
                'subject_sd_pp': float(differences.std(ddof=1)),
                'bootstrap_95ci_low_pp': lo, 'bootstrap_95ci_high_pp': hi,
                'win': int((differences > 1e-6).sum()),
                'tie': int((np.abs(differences) <= 1e-6).sum()),
                'loss': int((differences < -1e-6).sum()), 'n_seed': 3, 'n_subject': 9,
            })
            for subject, value in zip(range(1, 10), differences):
                paired_rows.append({'metric': metric, 'comparison': comparison,
                    'subject': subject, 'difference_pp': float(value), 'n_seed': 3})
    report_root = require_report_output(REPORTS_ROOT / config['report']['report_directory'] / 'summary.csv').parent
    report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_root / 'metric_summary.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(report_root / 'paired_differences.csv', index=False)
    metrics.to_csv(report_root / 'fold_metrics.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(report_root / 'trial_predictions.csv', index=False)
    summary = {
        'experiment_id': config['experiment_id'], 'run_id_prefix': run_id_prefix,
        'dataset': 'BNCI2014004', 'selected_session': 'session_3',
        'seeds': list(SEEDS), 'folds_per_seed': 9, 'n_target_subjects': 9,
        'aggregation': 'mean over three seeds within each target subject; paired subject bootstrap',
        'test_protocol': 'student CBraMod source recipe and IFNet native no-EA recipe; no target data used in fitting',
        'metrics': rows, 'paired_differences': [r for r in paired_rows if 'mean_difference_pp' in r],
        'target_evaluations': 27, 'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _atomic_json(report_root / 'aggregate.json', summary)
    _write_workbook(report_root / 'summary.xlsx', pd.DataFrame(rows), pd.DataFrame(paired_rows))
    return summary


def _write_workbook(path: Path, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active; ws.title = 'Summary'
    for row in [list(summary.columns), *summary.itertuples(index=False, name=None)]:
        ws.append(list(row))
    ws.freeze_panes = 'A2'; ws.sheet_view.showGridLines = False
    header_fill = PatternFill('solid', fgColor='1F4E78')
    for cell in ws[1]:
        cell.fill = header_fill; cell.font = Font(color='FFFFFF', bold=True)
    for column in ws.columns:
        letter = get_column_letter(column[0].column)
        ws.column_dimensions[letter].width = min(40, max(14, max(len(str(c.value or '')) for c in column) + 2))
        for cell in column[1:]: cell.alignment = Alignment(horizontal='center')
    ws2 = wb.create_sheet('Paired differences')
    for row in [list(paired.columns), *paired.itertuples(index=False, name=None)]: ws2.append(list(row))
    ws2.freeze_panes = 'A2'; ws2.sheet_view.showGridLines = False
    for cell in ws2[1]: cell.fill = header_fill; cell.font = Font(color='FFFFFF', bold=True)
    for column in ws2.columns:
        letter = get_column_letter(column[0].column)
        ws2.column_dimensions[letter].width = min(48, max(14, max(len(str(c.value or '')) for c in column) + 2))
        for cell in column[1:]: cell.alignment = Alignment(horizontal='center')
    note = wb.create_sheet('Notes')
    note.append(['CBraMod → IFNet, 004 session_3, three seeds × nine 7/1/1 folds.'])
    note.append(['New methods: joint training of IFNet and a trainable 2400→512→512 GELU adapter; feature weight 0.5.'])
    note.append(['Test labels are read only after all 27 training folds complete. CBraMod apply_EA=false; IFNet no EA.'])
    note.append(['Means are reported over 27 fold-seed records; bootstrap resamples nine target subjects after averaging seeds within subject.'])
    note.column_dimensions['A'].width = 120
    for row in note.iter_rows():
        for cell in row: cell.alignment = Alignment(wrap_text=True, vertical='top')
    wb.save(path)


def _evaluate_all(config: dict, run_id_prefix: str, gpu: int,
                  progress_path: Path | None = None) -> dict:
    # Fail closed before reading target labels: all 54 formal trajectories must be terminal.
    for seed in SEEDS:
        for fold in FOLDS:
            roots = _roots(run_id_prefix + f'_seed{seed}', seed, fold)
            complete_path = roots['run_result'] / 'fold_complete.json'
            if not complete_path.is_file() or _read_json(complete_path).get('status') != 'trained':
                raise RuntimeError(f'target evaluation refused: training incomplete seed={seed}, fold={fold+1}')
            for condition in CONDITIONS:
                manifest = _read_json(roots['run_result'] / f'{condition}.json')
                if manifest.get('status') != 'complete' or manifest.get('final_epoch') != 100:
                    raise RuntimeError(f'target evaluation refused: {condition} is not terminal epoch 100')
    device = _device(gpu)
    completed = []
    for seed in SEEDS:
        for fold in FOLDS:
            _target_evaluate_fold(config, seed, fold,
                                  run_id_prefix + f'_seed{seed}', device)
            completed.append({'seed': seed, 'fold': fold, 'status': 'evaluated'})
            _write_progress(progress_path, {'status': 'running', 'records': completed,
                                            'total_cells': 27})
    summary = _write_summary(config, run_id_prefix)
    _write_progress(progress_path, {'status': 'completed', 'records': completed,
                                    'total_cells': 27})
    return summary


def _watch_training(config_path: Path, config: dict, run_id_prefix: str,
                    queue_root: Path, poll_seconds: float) -> dict:
    """After all 27 fold jobs finish, enqueue the deferred target evaluation."""
    from experiments.distill.queue import QueueStore, validated_root
    store = QueueStore(validated_root(queue_root))
    status_path = (Path('/data1/llx/BigSmallcollab/results/distill') /
                   f'{run_id_prefix}_orchestration.json')
    while True:
        matches = []
        for state in store.states():
            spec = json.loads((store.job_dir(state['id']) / 'spec.json').read_text())
            snapshot = spec.get('snapshot', {})
            if (snapshot.get('kind') == 'cbramod_ifnet_trainable_adapter_loso_v2'
                    and snapshot.get('stage') == 'train'
                    and snapshot.get('run_id_prefix') == run_id_prefix):
                matches.append(state)
        if len(matches) >= 27:
            failed = [s for s in matches if s['status'] in {'failed', 'cancelled', 'paused'}]
            if failed:
                result = {'status': 'training_incomplete', 'run_id_prefix': run_id_prefix,
                          'job_count': len(matches), 'failed_jobs': [s['id'] for s in failed],
                          'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                _atomic_json(status_path, result)
                return result
            if len(matches) == 27 and all(s['status'] == 'completed' for s in matches):
                jobs = _enqueue(config_path, config, 'evaluate', queue_root=queue_root,
                                run_id_prefix=run_id_prefix)
                result = {'status': 'evaluation_queued', 'run_id_prefix': run_id_prefix,
                          'training_job_count': len(matches), 'evaluation_jobs': jobs,
                          'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                _atomic_json(status_path, result)
                return result
        _atomic_json(status_path, {
            'status': 'waiting_for_training', 'run_id_prefix': run_id_prefix,
            'training_job_count': len(matches),
            'completed_jobs': sum(s['status'] == 'completed' for s in matches),
            'active_or_pending_jobs': sum(s['status'] in {'pending', 'starting', 'running', 'stopping'}
                                          for s in matches),
            'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        })
        time.sleep(poll_seconds)


def _enqueue(config_path: Path, config: dict, stage: str, *, queue_root: Path | None = None,
             run_id_prefix: str | None = None) -> list[str]:
    from experiments.distill import queue
    config_path = Path(config_path).resolve()
    config_sha = _sha(config_path)
    if run_id_prefix is None:
        run_id_prefix = config['storage']['external_run_id_prefix']
    root = Path(queue_root or (ROOT / config['storage']['queue_root'])).resolve()
    jobs = []
    if stage == 'smoke':
        matrix = [(666, 0)]
    elif stage == 'train':
        matrix = [(seed, fold) for seed in SEEDS for fold in FOLDS]
    elif stage == 'evaluate':
        matrix = [(0, 0)]
    else:
        raise ValueError(f'unknown enqueue stage {stage}')
    for seed, fold in matrix:
        run_id = run_id_prefix + f'_seed{seed}' if stage != 'smoke' else run_id_prefix + '_smoke_seed666'
        snapshot = {
            'kind': 'cbramod_ifnet_trainable_adapter_loso_v2',
            'stage': stage, 'config': str(config_path), 'config_sha256': config_sha,
            'run_id_prefix': run_id_prefix, 'run_id': run_id,
            'seed': seed, 'fold': fold,
            'args': {
                'dataset': 'BNCI2014004', 'teacher': 'CBraMod', 'student': 'IFNet',
                'seeds': [seed] if stage != 'evaluate' else list(SEEDS),
                'keys': [fold] if stage != 'evaluate' else list(FOLDS),
                'methods': list(CONDITIONS) if stage == 'train' else [stage],
                'epochs': 100 if stage == 'train' else 2,
                'batch_size': 16, 'lr': 0.001, 'weight_decay': 0.01,
            },
        }
        job_id = queue.enqueue_snapshot(
            snapshot, root=root,
            argv=['cbramod_ifnet_trainable_adapter_loso_v2', '--stage', stage,
                  '--seed', str(seed), '--fold', str(fold)],
            label=f'CBraMod->IFNet trainable adapter V2 {stage} seed={seed} fold={fold+1}')
        jobs.append(job_id)
    return jobs


def preflight(config_path: Path, config: dict) -> dict:
    """Read-only audit of every expected source fold without touching test labels."""
    missing, verified = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            try:
                run_id = config['storage']['external_run_id_prefix'] + f'_seed{seed}'
                context = _source_context(config, seed, fold, run_id,
                                          torch.device('cpu'), include_test=False)
                verified.append({'seed': seed, 'fold': fold + 1,
                                 'target_sha256': context['source_target_sha256'],
                                 'initial_IFNet_sha256': context['student']['initial_student']['model_sha256'],
                                 'schedule_sha256': context['student']['student_schedule_sha256']})
            except Exception as exc:
                missing.append({'seed': seed, 'fold': fold + 1,
                                'error': f'{type(exc).__name__}: {exc}'})
    report = {'status': 'ready' if not missing else 'blocked',
              'experiment_id': config['experiment_id'], 'config_sha256': _sha(config_path),
              'expected_folds': 27, 'verified_folds': len(verified),
              'missing': missing, 'verified': verified,
              'target_test_labels_read': False}
    return report


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--config-sha256')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--run-id-prefix', default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--watch-training', action='store_true')
    parser.add_argument('--watch-poll-seconds', type=float, default=30.0)
    parser.add_argument('--enqueue', choices=('smoke', 'train', 'evaluate'))
    parser.add_argument('--queue-root', type=Path)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    if args.config_sha256 and _sha(config_path) != args.config_sha256:
        raise RuntimeError('config hash differs from the queued immutable snapshot')
    if args.preflight:
        report = preflight(config_path, config)
        report_root = require_report_output(REPORTS_ROOT / config['report']['report_directory'] / 'preflight.json').parent
        report_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(report_root / 'preflight.json', report)
        print(json.dumps({'status': report['status'], 'verified_folds': report['verified_folds'],
                          'missing': report['missing']}, ensure_ascii=False))
        return 0 if report['status'] == 'ready' else 2
    if args.enqueue:
        jobs = _enqueue(config_path, config, args.enqueue, queue_root=args.queue_root,
                        run_id_prefix=args.run_id_prefix)
        print(json.dumps({'stage': args.enqueue, 'jobs': jobs}, ensure_ascii=False))
        return 0
    run_id_prefix = args.run_id_prefix or config['storage']['external_run_id_prefix']
    if args.watch_training:
        if not args.queue_root:
            queue_root = ROOT / config['storage']['queue_root']
        else:
            queue_root = args.queue_root
        result = _watch_training(config_path, config, run_id_prefix,
                                 queue_root, args.watch_poll_seconds)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.evaluate:
        summary = _evaluate_all(config, run_id_prefix, args.gpu, args.progress)
        print(json.dumps({'status': 'evaluated', 'target_evaluations': summary['target_evaluations']},
                         ensure_ascii=False))
        return 0
    if args.seed is None or args.fold is None:
        parser.error('--seed and --fold are required for a fold training worker')
    run_id = args.run_id or (run_id_prefix + f'_seed{args.seed}')
    try:
        result = _train_fold(config_path, config, args.seed, args.fold, run_id,
                             args.gpu, smoke=args.smoke, progress_path=args.progress,
                             stop_file=args.stop_file)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except SystemExit as exc:
        if exc.code == 75:
            return 75
        raise
    except Exception as exc:
        trace = traceback.format_exc()
        print(trace, file=sys.stderr, flush=True)
        if args.progress:
            _write_progress(args.progress, {'status': 'failed', 'seed': args.seed,
                'fold': args.fold, 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    raise SystemExit(_main())
