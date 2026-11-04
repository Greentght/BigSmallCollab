"""CBraMod -> IFNet dual trainable-adapter FEATURE LOSO V3."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS',
              'TORCH_NUM_THREADS', 'TORCH_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import yaml
from torch.nn import functional as F

from collab import cbramod_t2s_projection as projection_core
from collab.cbramod_ifnet_dual_adapter import (
    StudentToSharedAdapter, derive_student_adapter_seed,
    initialize_student_adapter, state_hash as adapter_state_hash,
)
from experiments.distill import run_cbramod_ifnet_trainable_adapter_loso as base_v2
from experiments.distill import run_cbramod_mcc_diffkd_loso as cbramod_run
from experiments.distill import run_student_aware_diffusion_loso as reference
from experiments.storage import REPORTS_ROOT, require_report_output, resolve_local_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/loso/cbramod_ifnet_dual_adapter_v3.yaml'
SEEDS = (666, 667, 668)
FOLDS = tuple(range(9))
CONDITIONS = ('FIXED_P_STUDENT_ADAPTER_FEATURE', 'DUAL_ADAPTER_FEATURE')
DIAGNOSTIC_EPOCHS = (0, 1, 10, 50, 100)
BOOTSTRAP_DRAWS = 10000
MODEL_NAMES = ('T_CE', 'IFNet_CE', 'FIXED_P_FEATURE',
               'TEACHER_ADAPTER_FEATURE', *CONDITIONS)


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


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


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(resolve_local_file(path)).read_text(encoding='utf-8'))
    if (config.get('experiment_id') != 'CBRAMOD_IFNET_DUAL_ADAPTER_FEATURE_LOSO_V3'
            or config.get('dataset') != 'BNCI2014004'
            or tuple(config.get('seeds', ())) != SEEDS
            or config.get('feature', {}).get('shared_dimension') != 512
            or config.get('feature', {}).get('feature_weight') != 0.5
            or config.get('student', {}).get('epochs') != 100
            or config.get('student', {}).get('batch_size') != 16
            or tuple(config.get('conditions', ())) != CONDITIONS):
        raise ValueError('V3 config differs from the locked dual-adapter protocol')
    return config


def _roots(run_id: str, seed: int, fold: int) -> dict:
    return base_v2._roots(run_id, seed, fold)


def _source_context(config: dict, seed: int, fold: int, run_id: str,
                    device: torch.device, *, include_test: bool = False) -> dict:
    v2_config_path = (ROOT / config['sources']['v2_config']).resolve()
    v2_config = base_v2._load_config(v2_config_path)
    if v2_config['storage']['external_run_id_prefix'] != config['sources']['v2_run_id_prefix']:
        raise RuntimeError('V2 reference run ID prefix changed')
    return base_v2._source_context(v2_config, seed, fold, run_id, device,
                                   include_test=include_test)


def _load_fixed_projection(context: dict, fold_complete: dict) -> tuple[np.ndarray, dict]:
    roots = context['source_roots']
    path = roots['run_cache'] / 'fixed_teacher_to_IFNet_P.npz'
    manifest_path = roots['run_cache'] / 'fixed_teacher_to_IFNet_P.json'
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'missing V1 fixed teacher projection: {path}')
    meta = _read_json(manifest_path)
    actual_sha = _sha(path)
    if (meta.get('status') != 'complete' or meta.get('sha256') != actual_sha
            or tuple(meta.get('shape', ())) != (512, 2400)
            or fold_complete.get('projection_sha256') != actual_sha
            or meta.get('fingerprint') != fold_complete.get('run_fingerprint')):
        # A deterministic restore may have recorded its original file hash separately;
        # the V1 fold completion still binds to the current, verified projection file.
        raise RuntimeError('V1 fixed projection does not match its fold completion identity')
    with np.load(resolve_local_file(path), allow_pickle=False) as saved:
        matrix = np.asarray(saved['P_float32'], dtype=np.float32)
    if matrix.shape != (512, 2400):
        raise RuntimeError(f'V1 projection has wrong shape: {matrix.shape}')
    return matrix, {'path': str(path), 'sha256': actual_sha,
                    'original_sha256': meta.get('restored_original_sha256', actual_sha),
                    'seed': meta.get('seed'), 'fingerprint': meta.get('fingerprint')}


def _v2_teacher_adapter_initial(context: dict, config: dict, seed: int, fold: int,
                                device: torch.device) -> tuple[dict, str, int, dict]:
    v2_config_path = (ROOT / config['sources']['v2_config']).resolve()
    v2_config = base_v2._load_config(v2_config_path)
    _, initial_state, init_hash, adapter_seed = base_v2._adapter_and_state(
        seed, fold, v2_config, device)
    run_id = config['sources']['v2_run_id_prefix'] + f'_seed{seed}'
    v2_roots = _roots(run_id, seed, fold)
    condition = config['sources']['v2_teacher_adapter_condition']
    manifest_path = v2_roots['run_result'] / f'{condition}.json'
    checkpoint_path = v2_roots['run_weights'] / config['sources']['v2_teacher_adapter_checkpoint']
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f'missing completed V2 teacher adapter seed={seed}, fold={fold + 1}')
    manifest = _read_json(manifest_path)
    payload = torch.load(resolve_local_file(checkpoint_path), map_location='cpu', weights_only=False)
    expected_teacher_sha = context['target_meta']['teacher_state_sha256']['T_CE']
    if (manifest.get('status') != 'complete'
            or manifest.get('condition') != condition
            or manifest.get('adapter_initial_sha256') != init_hash
            or manifest.get('adapter_seed') != adapter_seed
            or manifest.get('adapter_checkpoint_sha256') != _sha(checkpoint_path)
            or payload.get('initial_state_sha256') != init_hash
            or payload.get('adapter_seed') != adapter_seed
            or manifest.get('teacher_state_sha256') != expected_teacher_sha
            or manifest.get('teacher_target_sha256') != context['source_target_sha256']):
        raise RuntimeError('V2 T_CE adapter initialization/provenance mismatch')
    if base_v2.state_hash(payload['model']) != manifest.get('adapter_final_sha256'):
        raise RuntimeError('V2 trained teacher-adapter checkpoint hash mismatch')
    return initial_state, init_hash, adapter_seed, {
        'manifest': str(manifest_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': _sha(checkpoint_path),
        'final_state_sha256': base_v2.state_hash(payload['model']),
        'trained_adapter_is_reference_only': True,
    }


def _student_adapter_init(config: dict, seed: int, fold: int) -> tuple[dict, str, int]:
    actual_seed = derive_student_adapter_seed(config['experiment_id'], seed, fold)
    module, digest = initialize_student_adapter(actual_seed)
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    if sum(value.numel() for value in state.values()) != 525312:
        raise RuntimeError('student adapter parameter count differs from locked V3 design')
    return state, digest, actual_seed


def _fold_fingerprint(config_path: Path, config: dict, seed: int, fold: int,
                      context: dict, projection: dict, teacher_init_hash: str,
                      teacher_reference: dict, student_init_hash: str,
                      student_adapter_seed: int) -> str:
    return _json_hash({
        'experiment_id': config['experiment_id'], 'config_sha256': _sha(config_path),
        'runner_sha256': _sha(Path(__file__)),
        'student_adapter_sha256': _sha(ROOT / 'collab/cbramod_ifnet_dual_adapter.py'),
        'seed': seed, 'fold': fold,
        'source_split_sha256': context['identity']['sha256'],
        'source_target_sha256': context['source_target_sha256'],
        'teacher_state_sha256': context['target_meta']['teacher_state_sha256']['T_CE'],
        'fixed_projection': projection, 'teacher_adapter_initial_sha256': teacher_init_hash,
        'teacher_adapter_reference': teacher_reference,
        'student_adapter_initial_sha256': student_init_hash,
        'student_adapter_seed': student_adapter_seed,
        'initial_IFNet_sha256': context['student']['initial_student']['model_sha256'],
        'student_schedule_sha256': context['student']['student_schedule_sha256'],
        'conditions': list(CONDITIONS), 'epochs': 100, 'feature_weight': 0.5,
    })


def preflight(config_path: Path, config: dict) -> dict:
    missing, verified = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            run_id = config['storage']['external_run_id_prefix'] + f'_seed{seed}'
            try:
                context = _source_context(config, seed, fold, run_id,
                                          torch.device('cpu'), include_test=False)
                complete = _read_json(context['source_roots']['run_result'] / 'fold_complete.json')
                matrix, pmeta = _load_fixed_projection(context, complete)
                _, thash, tseed, v2ref = _v2_teacher_adapter_initial(
                    context, config, seed, fold, torch.device('cpu'))
                _, shash, sseed = _student_adapter_init(config, seed, fold)
                v1eval = _read_json(context['source_roots']['run_result'] / 'fold_evaluation.json')
                if v1eval.get('status') != 'complete':
                    raise RuntimeError('V1 matched target evaluation missing')
                v2_run_id = config['sources']['v2_run_id_prefix'] + f'_seed{seed}'
                v2_roots = _roots(v2_run_id, seed, fold)
                v2eval = _read_json(v2_roots['run_result'] / 'fold_evaluation.json')
                if v2eval.get('status') != 'complete':
                    raise RuntimeError('V2 matched target evaluation missing')
                verified.append({
                    'seed': seed, 'fold': fold + 1,
                    'teacher_target_sha256': context['source_target_sha256'],
                    'fixed_projection_sha256': pmeta['sha256'],
                    'teacher_adapter_initial_sha256': thash,
                    'teacher_adapter_seed': tseed,
                    'teacher_adapter_reference_sha256': v2ref['checkpoint_sha256'],
                    'student_adapter_initial_sha256': shash,
                    'student_adapter_seed': sseed,
                    'initial_IFNet_sha256': context['student']['initial_student']['model_sha256'],
                    'schedule_sha256': context['student']['student_schedule_sha256'],
                })
            except Exception as exc:
                missing.append({'seed': seed, 'fold': fold + 1,
                                'error': f'{type(exc).__name__}: {exc}'})
    return {'status': 'ready' if not missing else 'blocked',
            'experiment_id': config['experiment_id'], 'config_sha256': _sha(config_path),
            'expected_folds': 27, 'verified_folds': len(verified),
            'verified': verified, 'missing': missing, 'target_test_labels_read': False}


def _effective_rank(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh((x.T @ x) / max(1, len(x) - 1))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0.0
    p = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).copy()
    y = np.asarray(right, dtype=np.float64).copy()
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        return float('nan')
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx, yy = x.T @ x, y.T @ y
    denominator = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    return float(np.sum(cross * cross) / denominator) if denominator > 0 else float('nan')


def _pairwise_cosine(values: np.ndarray) -> tuple[float, int]:
    x = np.asarray(values[:128], dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    zero = norms <= 1e-12
    valid = ~zero
    if valid.sum() < 2:
        return float('nan'), int(zero.sum())
    x = x[valid] / norms[valid, None]
    cos = x @ x.T
    mask = ~np.eye(len(cos), dtype=bool)
    return float(cos[mask].mean()), int(zero.sum())


def _centroid_stats(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(values), np.asarray(labels)
    centers = [x[y == c].mean(axis=0) for c in (0, 1) if np.any(y == c)]
    inter = float(np.linalg.norm(centers[0] - centers[1])) if len(centers) == 2 else float('nan')
    within = []
    for c in (0, 1):
        mask = y == c
        if mask.any(): within.extend(np.linalg.norm(x[mask] - x[mask].mean(axis=0), axis=1))
    return inter, float(np.mean(within)) if within else float('nan')


def _readonly_partition(model, student_adapter, student_map, teacher_map,
                        x: np.ndarray, y: np.ndarray, f_teacher: np.ndarray,
                        condition: str, projection: np.ndarray,
                        device: torch.device) -> tuple[dict, dict]:
    state = reference._cpu_tree(model.state_dict())
    before = reference._state_hash(state)
    rng = reference._capture_rng(device)
    model_mode, map_mode = model.training, student_map.training
    teacher_mode = teacher_map.training if teacher_map is not None else None
    try:
        model.eval(); student_map.eval()
        if teacher_map is not None: teacher_map.eval()
        logits_out, raw_student, z_student, z_teacher = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(x), 64):
                end = min(start + 64, len(x))
                xb = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
                h_s, logits = student_adapter.forward(model, xb)
                z_s = student_map(h_s)
                f_t = torch.as_tensor(f_teacher[start:end], dtype=torch.float32, device=device)
                if condition == 'DUAL_ADAPTER_FEATURE':
                    z_t = teacher_map(f_t)
                else:
                    p = torch.as_tensor(projection, dtype=torch.float32, device=device)
                    z_t = F.layer_norm(f_t @ p.T, (512,), weight=None, bias=None, eps=1e-5)
                logits_out.append(logits.float().cpu().numpy())
                raw_student.append(h_s.float().cpu().numpy())
                z_student.append(z_s.float().cpu().numpy())
                z_teacher.append(z_t.float().cpu().numpy())
        result = {key: np.concatenate(value) for key, value in {
            'logits': logits_out, 'h_student': raw_student,
            'z_student': z_student, 'z_teacher': z_teacher}.items()}
        logit_metrics = cbramod_run._metric_values(y, result['logits'])
        pair_mse = float(np.mean((result['z_student'] - result['z_teacher']) ** 2))
        d_st, r_st = _centroid_stats(result['z_student'], y)
        d_t, r_t = _centroid_stats(result['z_teacher'], y)
        cos_st, zero_st = _pairwise_cosine(result['z_student'])
        cos_t, zero_t = _pairwise_cosine(result['z_teacher'])
        diag = {
            **logit_metrics, 'feature_mse': pair_mse,
            'student_cross_trial_variance': float(np.var(result['z_student'], axis=0).mean()),
            'teacher_cross_trial_variance': float(np.var(result['z_teacher'], axis=0).mean()),
            'native_student_cross_trial_variance': float(np.var(result['h_student'], axis=0).mean()),
            'student_effective_rank': _effective_rank(result['z_student']),
            'teacher_effective_rank': _effective_rank(result['z_teacher']),
            'native_student_effective_rank': _effective_rank(result['h_student']),
            'student_interclass_distance': d_st, 'student_intraclass_radius': r_st,
            'teacher_interclass_distance': d_t, 'teacher_intraclass_radius': r_t,
            'linear_cka_teacher_student': _linear_cka(result['z_teacher'], result['z_student']),
            'linear_cka_teacher_raw_feature': _linear_cka(result['z_teacher'], f_teacher),
            'linear_cka_student_raw_feature': _linear_cka(result['h_student'], f_teacher),
            'student_pairwise_cosine_mean': cos_st, 'student_zero_norm_count': zero_st,
            'teacher_pairwise_cosine_mean': cos_t, 'teacher_zero_norm_count': zero_t,
        }
        return diag, result
    finally:
        if reference._state_hash(model.state_dict()) != before:
            model.load_state_dict(state, strict=True)
        reference._restore_rng(rng, device)
        model.train(model_mode); student_map.train(map_mode)
        if teacher_map is not None: teacher_map.train(teacher_mode)
        if reference._state_hash(model.state_dict()) != before:
            raise RuntimeError('read-only diagnostic changed IFNet state')


def _make_optimizers(model, teacher_map, student_map, config):
    s = config['student']
    opt_s = torch.optim.AdamW(model.parameters(), lr=float(s['lr']),
        betas=tuple(s['betas']), eps=float(s['eps']), weight_decay=float(s['weight_decay']),
        foreach=False, fused=False)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_s, T_max=int(s['scheduler_t_max']), eta_min=float(s['scheduler_eta_min']))
    opts, schs = {}, {}
    for name, module in (('teacher', teacher_map), ('student', student_map)):
        if module is None: continue
        cfg = config['adapter_optimizers'][name]
        opts[name] = torch.optim.AdamW(module.parameters(), lr=float(cfg['lr']),
            betas=tuple(cfg['betas']), eps=float(cfg['eps']),
            weight_decay=float(cfg['weight_decay']), foreach=False, fused=False)
        schs[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[name], T_max=int(cfg['scheduler_t_max']), eta_min=float(cfg['scheduler_eta_min']))
    return opt_s, sch_s, opts, schs


def _capture_adapter_states(teacher_map, student_map):
    return {
        'teacher': None if teacher_map is None else {
            key: value.detach().cpu().clone() for key, value in teacher_map.state_dict().items()},
        'student': {key: value.detach().cpu().clone()
                    for key, value in student_map.state_dict().items()},
    }


def _save_resume(path: Path, *, model, teacher_map, student_map,
                 opt_student, sch_student, opt_adapters, sch_adapters,
                 epoch: int, fingerprint: str, init_hashes: dict,
                 device: torch.device) -> None:
    _atomic_torch(path, {
        'model': reference._cpu_tree(model.state_dict()),
        'adapters': _capture_adapter_states(teacher_map, student_map),
        'optimizer_student': opt_student.state_dict(),
        'scheduler_student': sch_student.state_dict(),
        'optimizer_adapters': {k: v.state_dict() for k, v in opt_adapters.items()},
        'scheduler_adapters': {k: v.state_dict() for k, v in sch_adapters.items()},
        'epoch': int(epoch), 'run_fingerprint': fingerprint,
        'adapter_initial_hashes': init_hashes,
        'rng_state': reference._capture_rng(device),
    })


def _evaluate_diagnostics(model, context, condition, teacher_map,
                          student_map, projection, device):
    identity, data = context['identity'], context['data']
    features = context['_train_teacher_features']
    feedback_features = context['_feedback_teacher_features']
    outputs = {}
    rows = []
    for partition, x, y, f_t in (
        ('D_train', data['student_x']['train'], identity['labels']['train'], features),
        ('D_feedback', data['student_x']['feedback'], identity['labels']['feedback'], feedback_features),
    ):
        metrics, outputs[partition] = _readonly_partition(
            model, data['student_adapter'], student_map, teacher_map,
            x, y, f_t, condition, projection, device)
        rows.append({'partition': partition, **metrics})
    return rows, outputs


def _fit_condition(condition: str, config: dict, context: dict, seed: int,
                   fold: int, fold_fingerprint: str,
                   teacher_init_state: dict, teacher_init_hash: str,
                   teacher_init_seed: int, student_init_state: dict,
                   student_init_hash: str, student_init_seed: int,
                   projection: np.ndarray, device: torch.device, *,
                   smoke: bool = False, progress_path: Path | None = None,
                   stop_file: Path | None = None) -> dict:
    roots, identity, data, student = (context['roots'], context['identity'],
                                      context['data'], context['student'])
    is_dual = condition == 'DUAL_ADAPTER_FEATURE'
    v3cfg = config['student']
    teacher_reference = context['teacher_adapter_reference']
    init_hashes = {'teacher': teacher_init_hash if is_dual else None,
                   'student': student_init_hash}
    method_fingerprint = _json_hash({
        'fold_fingerprint': fold_fingerprint, 'condition': condition,
        'teacher_adapter_initial_sha256': init_hashes['teacher'],
        'student_adapter_initial_sha256': student_init_hash,
        'teacher_adapter_v2_reference': teacher_reference if is_dual else None,
        'fixed_projection_sha256': context['fixed_projection_meta']['sha256'] if not is_dual else None,
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'schedule_sha256': student['student_schedule_sha256'],
        'epochs': 1 if smoke else int(v3cfg['epochs']),
    })
    final_student = roots['run_weights'] / f'student_{condition}.pt'
    final_student_adapter = roots['run_weights'] / f'student_adapter_{condition}.pt'
    final_teacher_adapter = roots['run_weights'] / f'teacher_adapter_{condition}.pt' if is_dual else None
    manifest_path = roots['run_result'] / f'{condition}.json'
    state_path = roots['run_weights'] / f'training_state_{condition}.pt'
    if not smoke and any(path.exists() for path in
                         (final_student, final_student_adapter, manifest_path)
                         + ((final_teacher_adapter,) if final_teacher_adapter else ())):
        required = [final_student, final_student_adapter, manifest_path]
        if final_teacher_adapter:
            required.append(final_teacher_adapter)
        if not all(path.is_file() for path in required):
            raise RuntimeError(f'{condition} has partial terminal artifacts; refusing implicit retrain')
        old = _read_json(manifest_path)
        if (old.get('status') != 'complete'
                or old.get('method_fingerprint') != method_fingerprint
                or old.get('student_checkpoint_sha256') != _sha(final_student)
                or old.get('student_adapter_checkpoint_sha256') != _sha(final_student_adapter)
                or (final_teacher_adapter is not None
                    and old.get('teacher_adapter_checkpoint_sha256') != _sha(final_teacher_adapter))):
            raise RuntimeError(f'{condition} terminal identity/hash mismatch')
        return old

    model, _, _, _ = reference._new_paired_student(
        student, student['initial_student'], seed, device)
    student_map, actual_s_hash = initialize_student_adapter(student_init_seed)
    if actual_s_hash != student_init_hash:
        raise RuntimeError('student adapter initialization changed')
    student_map.load_state_dict(student_init_state, strict=True)
    student_map.to(device).train()
    teacher_map = None
    if is_dual:
        teacher_map, actual_t_state, actual_t_hash, actual_t_seed = base_v2._adapter_and_state(
            seed, fold, context['v2_config'], device)
        if actual_t_hash != teacher_init_hash or actual_t_seed != teacher_init_seed:
            raise RuntimeError('V2 teacher adapter initialization changed')
        teacher_map.load_state_dict(teacher_init_state, strict=True)
        teacher_map.to(device).train()

    opt_student, sch_student, opt_adapters, sch_adapters = _make_optimizers(
        model, teacher_map, student_map, config)
    with np.load(resolve_local_file(context['target_path']), allow_pickle=False) as saved:
        teacher_features = np.asarray(saved['T_CE_train_features_full'], dtype=np.float32).copy()
        feedback_features = np.asarray(saved['T_CE_feedback_features_full'], dtype=np.float32).copy()
    if (teacher_features.shape != (len(identity['uids']['train']), 2400)
            or feedback_features.shape != (len(identity['uids']['feedback']), 2400)):
        raise RuntimeError('T_CE full features do not match the source split')
    context['_train_teacher_features'] = teacher_features
    context['_feedback_teacher_features'] = feedback_features
    context['fixed_projection_meta'] = context['_fixed_projection_meta']

    train_x = torch.as_tensor(data['student_x']['train'], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(identity['labels']['train'], dtype=torch.long, device=device)
    teacher_tensor = torch.as_tensor(teacher_features, dtype=torch.float32, device=device)
    p_tensor = torch.as_tensor(projection, dtype=torch.float32, device=device)
    schedule = student['student_schedule']
    history_path = roots['run_result'] / f'{condition}_history.csv'
    diagnostic_path = roots['run_result'] / f'{condition}_diagnostics.csv'
    initial_diag_path = roots['run_cache'] / f'{condition}_epoch0_diagnostic_features.npz'
    history, diagnostics = [], []
    start_epoch = 0
    if state_path.is_file() and not smoke:
        saved = torch.load(resolve_local_file(state_path), map_location='cpu', weights_only=False)
        if (saved.get('run_fingerprint') != fold_fingerprint
                or saved.get('adapter_initial_hashes') != init_hashes):
            raise RuntimeError(f'{condition} resume identity/hash mismatch')
        model.load_state_dict(saved['model'], strict=True)
        student_map.load_state_dict(saved['adapters']['student'], strict=True)
        if teacher_map is not None:
            teacher_map.load_state_dict(saved['adapters']['teacher'], strict=True)
        opt_student.load_state_dict(saved['optimizer_student'])
        sch_student.load_state_dict(saved['scheduler_student'])
        for key, value in saved['optimizer_adapters'].items(): opt_adapters[key].load_state_dict(value)
        for key, value in saved['scheduler_adapters'].items(): sch_adapters[key].load_state_dict(value)
        start_epoch = int(saved['epoch'])
        reference._restore_rng(saved['rng_state'], device)
        if history_path.is_file(): history = pd.read_csv(resolve_local_file(history_path)).to_dict('records')
        if diagnostic_path.is_file(): diagnostics = pd.read_csv(resolve_local_file(diagnostic_path)).to_dict('records')

    def current_diagnostics(epoch: int):
        rows, outs = _evaluate_diagnostics(model, context, condition, teacher_map,
                                           student_map, projection, device)
        base = {}
        if initial_diag_path.is_file():
            with np.load(resolve_local_file(initial_diag_path), allow_pickle=False) as arr:
                base = {k: np.asarray(arr[k]) for k in arr.files}
        else:
            for part, result in outs.items():
                base[f'{part}_student'] = result['z_student'][:128].copy()
                base[f'{part}_teacher'] = result['z_teacher'][:128].copy()
            _atomic_npz(initial_diag_path, **base)
        for row in rows:
            part = row['partition']; result = outs[part]
            row.update({
                'condition': condition, 'seed': seed, 'target_subject': fold + 1,
                'epoch': epoch, 'n_trials': len(result['z_student']),
                'student_adapter_drift_from_epoch0': float(np.sqrt(np.mean(
                    (result['z_student'][:128] - base[f'{part}_student']) ** 2))),
                'teacher_adapter_drift_from_epoch0': float(np.sqrt(np.mean(
                    (result['z_teacher'][:128] - base[f'{part}_teacher']) ** 2))),
                'adapter_initial_hashes': json.dumps(init_hashes, sort_keys=True),
            })
        return rows

    if not smoke and 0 not in {row.get('epoch') for row in diagnostics}:
        diagnostics.extend(current_diagnostics(0))
        _atomic_csv(diagnostic_path, diagnostics)

    if smoke:
        epochs = 1
    else:
        epochs = int(v3cfg['epochs'])
    stem_parameter = next(model.stem.parameters())
    for epoch_index in range(start_epoch, epochs):
        model.train(); student_map.train()
        if teacher_map is not None: teacher_map.train()
        totals = {'ce': 0.0, 'feature': 0.0, 'total': 0.0}
        student_norm_sum = 0.0; adapter_norm_sum = {'student': 0.0, 'teacher': 0.0}
        gradient_cos = float('nan'); steps = count = 0
        batches = schedule[epoch_index][:2] if smoke else schedule[epoch_index]
        for batch_index, batch_ids in enumerate(batches):
            ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
            xb, yb, ft = train_x[ids], train_y[ids], teacher_tensor[ids]
            opt_student.zero_grad(set_to_none=True)
            for opt in opt_adapters.values(): opt.zero_grad(set_to_none=True)
            hs, logits = data['student_adapter'].forward(model, xb)
            if hs.shape != (len(ids), 512):
                raise RuntimeError(f'IFNet feature tap changed: {tuple(hs.shape)}')
            zs = student_map(hs)
            if condition == 'DUAL_ADAPTER_FEATURE':
                zt = teacher_map(ft.detach())
            else:
                zt = F.layer_norm(ft @ p_tensor.T, (512,), weight=None, bias=None, eps=1e-5)
            feature_loss = F.mse_loss(zs, zt, reduction='mean')
            ce = F.cross_entropy(logits, yb, label_smoothing=0.0)
            total = ce + float(config['feature']['feature_weight']) * feature_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f'{condition} non-finite loss at epoch {epoch_index + 1}')
            if batch_index == 0 and (epoch_index + 1) in (1, 10, 50, 100):
                gradient_cos = base_v2._gradient_cosine(ce, feature_loss, stem_parameter)
            total.backward()
            student_grads = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
            student_norm = torch.linalg.vector_norm(torch.stack(student_grads)) if student_grads else torch.tensor(float('nan'), device=device)
            if not torch.isfinite(student_norm):
                raise FloatingPointError(f'{condition} non-finite IFNet gradient')
            for name, module in (('teacher', teacher_map), ('student', student_map)):
                if module is None: continue
                norm = torch.nn.utils.clip_grad_norm_(
                    module.parameters(), float(config['adapter_optimizers'][name]['gradient_clip_norm']))
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'{condition} non-finite {name} adapter gradient')
                adapter_norm_sum[name] += float(norm.detach().cpu())
            if smoke and batch_index == 0:
                if not any(p.grad is not None and torch.isfinite(p.grad).all()
                           and float(p.grad.norm()) > 0 for p in model.stem.parameters()):
                    raise RuntimeError('feature gradient did not reach the IFNet stem')
                if not any(p.grad is not None and torch.isfinite(p.grad).all()
                           and float(p.grad.norm()) > 0 for p in student_map.parameters()):
                    raise RuntimeError('feature gradient did not reach student adapter')
                if teacher_map is not None and not any(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        and float(p.grad.norm()) > 0 for p in teacher_map.parameters()):
                    raise RuntimeError('feature gradient did not reach teacher adapter')
            opt_student.step()
            if 'teacher' in opt_adapters: opt_adapters['teacher'].step()
            opt_adapters['student'].step()
            n = len(ids)
            totals['ce'] += float(ce.detach().cpu()) * n
            totals['feature'] += float(feature_loss.detach().cpu()) * n
            totals['total'] += float(total.detach().cpu()) * n
            student_norm_sum += float(student_norm.detach().cpu())
            count += n; steps += 1
        sch_student.step()
        for scheduler in sch_adapters.values(): scheduler.step()
        history.append({
            'condition': condition, 'seed': seed, 'target_subject': fold + 1,
            'epoch': epoch_index + 1, 'train_ce': totals['ce'] / max(1, count),
            'feature_mse': totals['feature'] / max(1, count),
            'weighted_feature_mse': float(config['feature']['feature_weight']) * totals['feature'] / max(1, count),
            'total_loss': totals['total'] / max(1, count),
            'student_gradient_norm_mean': student_norm_sum / max(1, steps),
            'teacher_adapter_gradient_norm_mean': adapter_norm_sum['teacher'] / max(1, steps),
            'student_adapter_gradient_norm_mean': adapter_norm_sum['student'] / max(1, steps),
            'ce_feature_stem_gradient_cosine': gradient_cos,
            'student_lr_next': sch_student.get_last_lr()[0],
            'student_adapter_lr_next': sch_adapters['student'].get_last_lr()[0],
            'teacher_adapter_lr_next': sch_adapters['teacher'].get_last_lr()[0] if 'teacher' in sch_adapters else None,
            'train_uid_sha256': identity['uid_sha256']['train'],
            'schedule_sha256': student['student_schedule_sha256'],
        })
        if smoke: continue
        if epoch_index + 1 in DIAGNOSTIC_EPOCHS:
            diagnostics.extend(current_diagnostics(epoch_index + 1))
        _atomic_csv(history_path, history)
        if diagnostics: _atomic_csv(diagnostic_path, diagnostics)
        stop_requested = bool(stop_file and stop_file.exists())
        if (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs or stop_requested:
            _save_resume(state_path, model=model, teacher_map=teacher_map,
                student_map=student_map, opt_student=opt_student, sch_student=sch_student,
                opt_adapters=opt_adapters, sch_adapters=sch_adapters,
                epoch=epoch_index + 1, fingerprint=fold_fingerprint,
                init_hashes=init_hashes, device=device)
        if stop_requested:
            _write_progress(progress_path, {'status': 'paused', 'condition': condition,
                'epoch': epoch_index + 1, 'seed': seed, 'fold': fold})
            raise SystemExit(75)

    if smoke:
        return {'status': 'smoke_passed', 'condition': condition,
                'student_adapter_initial_sha256': student_init_hash,
                'teacher_adapter_initial_sha256': teacher_init_hash if is_dual else None,
                'student_adapter_parameter_count': sum(p.numel() for p in student_map.parameters()),
                'teacher_adapter_parameter_count': sum(p.numel() for p in teacher_map.parameters()) if teacher_map else 0,
                'student_gradient_reached': True,
                'student_adapter_gradient_reached': True,
                'teacher_adapter_gradient_reached': True if is_dual else None}

    student_state = reference._cpu_tree(model.state_dict())
    student_map_state = reference._cpu_tree(student_map.state_dict())
    _atomic_torch(final_student, {'model': student_state,
        'model_sha256': reference._state_hash(student_state),
        'condition': condition, 'method_fingerprint': method_fingerprint})
    _atomic_torch(final_student_adapter, {'model': student_map_state,
        'model_sha256': adapter_state_hash(student_map_state), 'condition': condition,
        'method_fingerprint': method_fingerprint,
        'initial_state_sha256': student_init_hash, 'seed': student_init_seed})
    teacher_adapter_hash = None
    if teacher_map is not None:
        teacher_state = reference._cpu_tree(teacher_map.state_dict())
        teacher_adapter_hash = adapter_state_hash(teacher_state)
        _atomic_torch(final_teacher_adapter, {'model': teacher_state,
            'model_sha256': teacher_adapter_hash, 'condition': condition,
            'method_fingerprint': method_fingerprint,
            'initial_state_sha256': teacher_init_hash, 'seed': teacher_init_seed,
            'v2_initialization_reference': teacher_reference})
    manifest = {
        'status': 'complete', 'experiment_id': config['experiment_id'],
        'condition': condition, 'seed': seed, 'fold_zero_based': fold,
        'target_subject': fold + 1, 'final_epoch': epochs,
        'run_fingerprint': fold_fingerprint, 'method_fingerprint': method_fingerprint,
        'teacher_state_sha256': context['target_meta']['teacher_state_sha256']['T_CE'],
        'teacher_target_sha256': context['source_target_sha256'],
        'fixed_projection_sha256': context['fixed_projection_meta']['sha256'],
        'teacher_adapter_reference': teacher_reference if is_dual else None,
        'teacher_adapter_initial_sha256': teacher_init_hash if is_dual else None,
        'teacher_adapter_final_sha256': teacher_adapter_hash,
        'student_adapter_initial_sha256': student_init_hash,
        'student_adapter_final_sha256': adapter_state_hash(student_map_state),
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'student_schedule_sha256': student['student_schedule_sha256'],
        'train_uid_sha256': identity['uid_sha256']['train'],
        'loss': {'classification': 'cross_entropy', 'feature': 'MSE(z_S,z_T)',
                 'feature_weight': float(config['feature']['feature_weight']),
                 'teacher_backbone_frozen': True, 'mcc': False, 'logits_kd': False},
        'student_checkpoint': str(final_student),
        'student_checkpoint_sha256': _sha(final_student),
        'student_adapter_checkpoint': str(final_student_adapter),
        'student_adapter_checkpoint_sha256': _sha(final_student_adapter),
        'teacher_adapter_checkpoint': str(final_teacher_adapter) if final_teacher_adapter else None,
        'teacher_adapter_checkpoint_sha256': _sha(final_teacher_adapter) if final_teacher_adapter else None,
        'history': str(history_path), 'diagnostics': str(diagnostic_path),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _atomic_json(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    return manifest


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == 'cuda': torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(1)


def _device(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('formal dual-adapter LOSO training requires CUDA')
    torch.cuda.set_device(int(gpu))
    return torch.device('cuda', int(gpu))


def _train_fold(config_path: Path, config: dict, seed: int, fold: int,
                run_id: str, gpu: int, *, smoke: bool = False,
                progress_path: Path | None = None,
                stop_file: Path | None = None) -> dict:
    if seed not in SEEDS or fold not in FOLDS:
        raise ValueError('seed/fold outside the locked 3-seed 9-fold range')
    device = _device(gpu)
    _seed_all(seed, device)
    context = _source_context(config, seed, fold, run_id, device, include_test=False)
    roots, identity, student = context['roots'], context['identity'], context['student']
    source_complete = _read_json(context['source_roots']['run_result'] / 'fold_complete.json')
    projection, projection_meta = _load_fixed_projection(context, source_complete)
    teacher_init, teacher_init_hash, teacher_init_seed, teacher_ref = _v2_teacher_adapter_initial(
        context, config, seed, fold, device)
    student_init, student_init_hash, student_init_seed = _student_adapter_init(config, seed, fold)
    context['fixed_projection_meta'] = projection_meta
    context['_fixed_projection_meta'] = projection_meta
    context['teacher_adapter_reference'] = teacher_ref
    v2_config_path = (ROOT / config['sources']['v2_config']).resolve()
    context['v2_config'] = base_v2._load_config(v2_config_path)
    fingerprint = _fold_fingerprint(config_path, config, seed, fold, context,
        projection_meta, teacher_init_hash, teacher_ref, student_init_hash, student_init_seed)
    complete_path = roots['run_result'] / 'fold_complete.json'
    if not smoke and complete_path.is_file():
        old = _read_json(complete_path)
        if old.get('status') != 'trained' or old.get('run_fingerprint') != fingerprint:
            raise RuntimeError('existing V3 completion record does not match this fold')
        _write_progress(progress_path, {'status': 'completed', 'seed': seed,
            'fold': fold, 'records': list(CONDITIONS), 'resumed_existing': True})
        return old
    resolved_path = roots['run_result'] / 'resolved_run.json'
    if not smoke:
        record = {
            'status': 'training', 'experiment_id': config['experiment_id'],
            'run_fingerprint': fingerprint, 'config': str(config_path),
            'config_sha256': _sha(config_path), 'seed': seed,
            'fold_zero_based': fold, 'target_subject': fold + 1,
            'feedback_subject': int(identity['manifest']['feedback_subject_id']),
            'train_subject_ids': identity['manifest']['train_subject_ids'],
            'split_manifest_sha256': identity['sha256'],
            'train_uid_sha256': identity['uid_sha256']['train'],
            'feedback_uid_sha256': identity['uid_sha256']['feedback'],
            'teacher_run': str(context['source_roots']['run_result']),
            'teacher_target_sha256': context['source_target_sha256'],
            'fixed_projection': projection_meta,
            'V2_teacher_adapter_reference': teacher_ref,
            'teacher_adapter_initial_sha256': teacher_init_hash,
            'student_adapter_initial_sha256': student_init_hash,
            'initial_IFNet_sha256': student['initial_student']['model_sha256'],
            'student_schedule_sha256': student['student_schedule_sha256'],
            'target_evaluation': 'deferred until all 27 folds complete',
        }
        if resolved_path.exists():
            old = _read_json(resolved_path)
            if old.get('run_fingerprint') != fingerprint:
                raise RuntimeError('existing V3 resolved record belongs to another fold identity')
        else:
            _atomic_json(resolved_path, record)

    records = []
    for condition in CONDITIONS:
        result = _fit_condition(condition, config, context, seed, fold, fingerprint,
            teacher_init, teacher_init_hash, teacher_init_seed, student_init,
            student_init_hash, student_init_seed, projection, device,
            smoke=smoke, progress_path=progress_path, stop_file=stop_file)
        records.append(result)
        _write_progress(progress_path, {'status': 'running', 'seed': seed, 'fold': fold,
            'records': [{'condition': r['condition'], 'status': r['status']} for r in records],
            'total_conditions': len(CONDITIONS)})
    if smoke:
        result = {'status': 'passed', 'seed': seed, 'fold': fold,
                  'run_fingerprint': fingerprint, 'conditions': records,
                  'target_evaluated': False,
                  'teacher_target_sha256': context['source_target_sha256'],
                  'fixed_projection_sha256': projection_meta['sha256']}
        smoke_dir = roots['run_result'] / 'smoke'; smoke_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(smoke_dir / f'smoke_seed{seed}_fold{fold + 1:02d}.json', result)
        _write_progress(progress_path, {'status': 'completed', 'seed': seed,
            'fold': fold, 'records': [{'smoke': 'passed'}]})
        return result
    completion = {
        'status': 'trained', 'experiment_id': config['experiment_id'],
        'run_fingerprint': fingerprint, 'seed': seed, 'fold_zero_based': fold,
        'target_subject': fold + 1,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'student_schedule_sha256': student['student_schedule_sha256'],
        'conditions': {r['condition']: r['student_checkpoint_sha256'] for r in records},
        'student_adapter_checkpoints': {r['condition']: r['student_adapter_checkpoint_sha256'] for r in records},
        'teacher_adapter_checkpoint': records[1]['teacher_adapter_checkpoint_sha256'],
        'source_teacher_target_sha256': context['source_target_sha256'],
        'fixed_projection_sha256': projection_meta['sha256'],
        'target_evaluation': 'deferred',
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _atomic_json(complete_path, completion)
    _write_progress(progress_path, {'status': 'completed', 'seed': seed,
        'fold': fold, 'records': [{'condition': x, 'status': 'complete'} for x in CONDITIONS]})
    return completion


def _append_predictions(rows, model_name, seed, fold, uids, labels, logits):
    for index, (uid, label, row_logits) in enumerate(zip(uids, labels, logits)):
        rows.append({'model': model_name, 'seed': seed, 'target_subject': fold + 1,
            'trial_uid': f'{int(uid[0])}:{int(uid[1])}', 'trial_order': index,
            'label': int(label), 'logit_0': float(row_logits[0]),
            'logit_1': float(row_logits[1]), 'prediction': int(np.argmax(row_logits))})


def _reuse_source_rows(context: dict, seed: int, fold: int) -> tuple[list[dict], list[dict]]:
    identity = context['identity']
    v1_roots = context['source_roots']
    v1_metrics_path = v1_roots['run_result'] / 'fold_evaluation_metrics.csv'
    v1_preds_path = v1_roots['run_result'] / 'fold_predictions.csv'
    v1_eval_path = v1_roots['run_result'] / 'fold_evaluation.json'
    if not all(Path(resolve_local_file(x)).is_file() for x in (v1_metrics_path, v1_preds_path, v1_eval_path)):
        raise FileNotFoundError('V1 target evaluation files are missing')
    v1_eval = _read_json(v1_eval_path)
    if (v1_eval.get('status') != 'complete'
            or v1_eval.get('test_uid_sha256') != identity['uid_sha256']['test']
            or v1_eval.get('metrics_sha256') != _sha(v1_metrics_path)
            or v1_eval.get('predictions_sha256') != _sha(v1_preds_path)):
        raise RuntimeError('V1 target evaluation provenance/hash mismatch')
    v1_metrics = pd.read_csv(resolve_local_file(v1_metrics_path))
    wanted_metrics = v1_metrics[v1_metrics['model'].isin(['T_CE', 'IFNet_CE', 'CE_T2S_FEATURE'])]
    if set(wanted_metrics['model']) != {'T_CE', 'IFNet_CE', 'CE_T2S_FEATURE'}:
        raise RuntimeError('V1 target metrics do not contain the required teacher/baselines')
    metric_rows = []
    rename = {'CE_T2S_FEATURE': 'FIXED_P_FEATURE'}
    for row in wanted_metrics.to_dict('records'):
        name = rename.get(row['model'], row['model'])
        metric_rows.append({'model': name, 'seed': seed, 'target_subject': fold + 1,
            'accuracy': float(row['accuracy']), 'balanced_accuracy': float(row['balanced_accuracy']),
            'kappa': float(row['kappa']), 'cross_entropy': float(row['cross_entropy']),
            'source': 'cbramod_t2s_full_feature_loso_v1'})
    v1_preds = pd.read_csv(resolve_local_file(v1_preds_path))
    select = {'T_CE', 'IFNet_CE', 'CE_T2S_FEATURE'}
    pred_rows = []
    for row in v1_preds[v1_preds['model'].isin(select)].to_dict('records'):
        row['model'] = rename.get(row['model'], row['model'])
        pred_rows.append(row)

    v2_run_id = context['v2_config']['storage']['external_run_id_prefix'] + f'_seed{seed}'
    v2_roots = _roots(v2_run_id, seed, fold)
    v2_metrics_path = v2_roots['run_result'] / 'fold_evaluation_metrics.csv'
    v2_preds_path = v2_roots['run_result'] / 'fold_predictions.csv'
    v2_eval_path = v2_roots['run_result'] / 'fold_evaluation.json'
    if not all(Path(resolve_local_file(x)).is_file() for x in (v2_metrics_path, v2_preds_path, v2_eval_path)):
        raise FileNotFoundError('V2 target evaluation files are missing')
    v2_eval = _read_json(v2_eval_path)
    if (v2_eval.get('status') != 'complete'
            or v2_eval.get('test_uid_sha256') != identity['uid_sha256']['test']
            or v2_eval.get('metrics_sha256') != _sha(v2_metrics_path)
            or v2_eval.get('predictions_sha256') != _sha(v2_preds_path)):
        raise RuntimeError('V2 target evaluation provenance/hash mismatch')
    v2_metrics = pd.read_csv(resolve_local_file(v2_metrics_path))
    match = v2_metrics[v2_metrics['model'] == 'JOINT_A_CE_FEATURE']
    if len(match) != 1:
        raise RuntimeError('V2 target evaluation lacks a unique CE teacher-adapter row')
    row = match.iloc[0].to_dict()
    metric_rows.append({'model': 'TEACHER_ADAPTER_FEATURE', 'seed': seed,
        'target_subject': fold + 1,
        **{key: float(row[key]) for key in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')},
        'source': 'cbramod_ifnet_trainable_adapter_v2'})
    v2_preds = pd.read_csv(resolve_local_file(v2_preds_path))
    match_preds = v2_preds[v2_preds['model'] == 'JOINT_A_CE_FEATURE']
    if len(match_preds) != len(v1_preds[v1_preds['model'] == 'T_CE']):
        raise RuntimeError('V2 prediction trial count differs from V1 target test')
    for row in match_preds.to_dict('records'):
        row['model'] = 'TEACHER_ADAPTER_FEATURE'
        pred_rows.append(row)
    if len(pred_rows) != sum(x['n_trials'] if 'n_trials' in x else 0 for x in metric_rows):
        # Metric records reused from older runners may not carry n_trials. UID sets are checked below.
        pass
    for name in ('T_CE', 'IFNet_CE', 'FIXED_P_FEATURE', 'TEACHER_ADAPTER_FEATURE'):
        uids = [r['trial_uid'] for r in pred_rows if r['model'] == name]
        if len(uids) != len(set(uids)) or not uids:
            raise RuntimeError(f'{name} reference predictions contain duplicate/missing UIDs')
    base_uids = {r['trial_uid'] for r in pred_rows if r['model'] == 'T_CE'}
    for name in ('IFNet_CE', 'FIXED_P_FEATURE', 'TEACHER_ADAPTER_FEATURE'):
        values = {r['trial_uid'] for r in pred_rows if r['model'] == name}
        if values != base_uids:
            raise RuntimeError(f'{name} trial UIDs differ from T_CE')
    return metric_rows, pred_rows


def _evaluate_fold(config: dict, seed: int, fold: int, run_id: str,
                   device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    context = _source_context(config, seed, fold, run_id, device, include_test=True)
    roots, identity, data, student = (context['roots'], context['identity'],
                                      context['data'], context['student'])
    complete = _read_json(roots['run_result'] / 'fold_complete.json')
    if complete.get('status') != 'trained':
        raise RuntimeError(f'incomplete training fold seed={seed}, subject={fold + 1}')
    metrics_path = roots['run_result'] / 'fold_evaluation_metrics.csv'
    predictions_path = roots['run_result'] / 'fold_predictions.csv'
    diagnostics_path = roots['run_result'] / 'target_feature_diagnostics.csv'
    eval_path = roots['run_result'] / 'fold_evaluation.json'
    if all(p.is_file() for p in (metrics_path, predictions_path, diagnostics_path, eval_path)):
        old = _read_json(eval_path)
        if (old.get('status') != 'complete'
                or old.get('run_fingerprint') != complete.get('run_fingerprint')
                or old.get('metrics_sha256') != _sha(metrics_path)
                or old.get('predictions_sha256') != _sha(predictions_path)
                or old.get('diagnostics_sha256') != _sha(diagnostics_path)):
            raise RuntimeError('saved V3 evaluation hash/provenance mismatch')
        return (pd.read_csv(resolve_local_file(metrics_path)).to_dict('records'),
                pd.read_csv(resolve_local_file(predictions_path)).to_dict('records'),
                pd.read_csv(resolve_local_file(diagnostics_path)).to_dict('records'))
    if any(p.exists() for p in (metrics_path, predictions_path, diagnostics_path, eval_path)):
        raise RuntimeError('partial V3 target evaluation exists; refusing overwrite')

    source_metrics, pred_rows = _reuse_source_rows(context, seed, fold)
    x_test_raw = np.asarray(data['student_input_mmap'][data['student_rows']['test']], dtype=np.float32)
    x_test = reference._preprocess_student(data['student_adapter'], x_test_raw)
    y_test = np.asarray(identity['labels']['test'], dtype=np.int64)
    uids = np.asarray(identity['uids']['test'])
    if len(x_test) != len(y_test) or len(uids) != len(y_test):
        raise RuntimeError('target IFNet inputs, labels and UIDs have inconsistent sizes')
    source_complete = _read_json(context['source_roots']['run_result'] / 'fold_complete.json')
    projection, projection_meta = _load_fixed_projection(context, source_complete)
    if complete.get('fixed_projection_sha256') != projection_meta['sha256']:
        raise RuntimeError('V3 training projection identity differs at evaluation')
    v2_config = context['v2_config']
    teacher_init, teacher_init_hash, teacher_init_seed, teacher_ref = _v2_teacher_adapter_initial(
        context, config, seed, fold, device)
    student_init, student_init_hash, student_init_seed = _student_adapter_init(config, seed, fold)
    context.update(fixed_projection_meta=projection_meta,
                   _fixed_projection_meta=projection_meta,
                   teacher_adapter_reference=teacher_ref, v2_config=v2_config)

    target_rows = np.asarray(data['teacher_rows']['test'], dtype=np.int64)
    x_teacher_test = np.asarray(data['teacher_input_mmap'][target_rows], dtype=np.float32)
    teacher_adapter, teachers = base_v2.v1_run._load_teacher_models(
        context['source_roots'], context['teacher_config'], device)
    teacher_features, teacher_logits = base_v2.v1_run._extract_full_teacher(
        teachers['T_CE'], teacher_adapter, x_teacher_test, device, check_forward=False)
    by_model = {row['model']: row for row in source_metrics}
    v1_teacher_pred = [r for r in pred_rows if r['model'] == 'T_CE']
    if len(v1_teacher_pred) != len(teacher_logits):
        raise RuntimeError('recomputed T_CE feature/logit count differs from saved V1 predictions')
    if not np.allclose(np.asarray([[r['logit_0'], r['logit_1']] for r in v1_teacher_pred]),
                       teacher_logits, rtol=1e-4, atol=1e-4):
        raise RuntimeError('recomputed T_CE target logits differ from the V1 reference')

    metric_rows, target_diag_rows = list(source_metrics), []
    for condition in CONDITIONS:
        result_manifest = _read_json(roots['run_result'] / f'{condition}.json')
        model_path = roots['run_weights'] / f'student_{condition}.pt'
        student_adapter_path = roots['run_weights'] / f'student_adapter_{condition}.pt'
        if (result_manifest.get('status') != 'complete'
                or result_manifest.get('run_fingerprint') != complete.get('run_fingerprint')
                or result_manifest.get('student_checkpoint_sha256') != _sha(model_path)
                or result_manifest.get('student_adapter_checkpoint_sha256') != _sha(student_adapter_path)):
            raise RuntimeError(f'{condition} target checkpoint identity mismatch')
        model, _, _, _ = reference._new_paired_student(student,
            student['initial_student'], seed, device)
        model_payload = torch.load(resolve_local_file(model_path), map_location='cpu', weights_only=False)
        model.load_state_dict(model_payload['model'], strict=True); model.to(device).eval()
        smap = StudentToSharedAdapter().to(device)
        smap_payload = torch.load(resolve_local_file(student_adapter_path), map_location='cpu', weights_only=False)
        smap.load_state_dict(smap_payload['model'], strict=True); smap.eval()
        tmap = None
        if condition == 'DUAL_ADAPTER_FEATURE':
            tpath = roots['run_weights'] / f'teacher_adapter_{condition}.pt'
            tpayload = torch.load(resolve_local_file(tpath), map_location='cpu', weights_only=False)
            if tpayload.get('initial_state_sha256') != teacher_init_hash:
                raise RuntimeError('saved teacher adapter did not start from the V2 initial state')
            tmap = base_v2.TeacherToStudentAdapter().to(device)
            tmap.load_state_dict(tpayload['model'], strict=True); tmap.eval()
        logits, model_hash = cbramod_run._predict_student_preserving_state(
            model, student['student_adapter'], x_test, device)
        if model_hash != model_payload['model_sha256']:
            raise RuntimeError(f'{condition} IFNet state changed during target prediction')
        values = cbramod_run._metric_values(y_test, logits)
        metric_rows.append({'model': condition, 'seed': seed,
            'target_subject': fold + 1, **values, 'source': 'new V3 target evaluation'})
        _append_predictions(pred_rows, condition, seed, fold, uids, y_test, logits)
        diag, outputs = _readonly_partition(model, student['student_adapter'], smap, tmap,
            x_test, y_test, teacher_features, condition, projection, device)
        target_diag_rows.append({'condition': condition, 'seed': seed,
            'target_subject': fold + 1, 'partition': 'D_test', 'epoch': 100,
            **diag, 'test_uid_sha256': identity['uid_sha256']['test']})
        del model, smap, tmap

    # Verify all six predictors have identical test trial identities and labels.
    expected_uids = {r['trial_uid'] for r in pred_rows if r['model'] == 'T_CE'}
    expected_labels = {r['trial_uid']: r['label'] for r in pred_rows if r['model'] == 'T_CE'}
    if len(expected_uids) != len(y_test):
        raise RuntimeError('saved teacher reference has missing or duplicate target UIDs')
    for name in MODEL_NAMES:
        rows = [r for r in pred_rows if r['model'] == name]
        if ({r['trial_uid'] for r in rows} != expected_uids
                or {r['trial_uid']: r['label'] for r in rows} != expected_labels):
            raise RuntimeError(f'{name} test predictions do not share the same UID/label set')
    _atomic_csv(metrics_path, metric_rows)
    _atomic_csv(predictions_path, pred_rows)
    _atomic_csv(diagnostics_path, target_diag_rows)
    _atomic_json(eval_path, {
        'status': 'complete', 'run_fingerprint': complete['run_fingerprint'],
        'seed': seed, 'fold_zero_based': fold, 'target_subject': fold + 1,
        'test_uid_sha256': identity['uid_sha256']['test'], 'n_trials': len(y_test),
        'models': list(MODEL_NAMES), 'metrics_sha256': _sha(metrics_path),
        'predictions_sha256': _sha(predictions_path),
        'diagnostics_sha256': _sha(diagnostics_path),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    return metric_rows, pred_rows, target_diag_rows


def _bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    means = np.asarray(values, dtype=np.float64)[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _style_workbook(path: Path, summary: pd.DataFrame,
                    paired: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fill_header = PatternFill('solid', fgColor='1F4E78')
    fill_teacher = PatternFill('solid', fgColor='FAC090')
    fill_baseline = PatternFill('solid', fgColor='93CDDD')
    fill_both = PatternFill('solid', fgColor='FFC7CE')
    fill_teacher_only = PatternFill('solid', fgColor='C6EFCE')
    fill_student_only = PatternFill('solid', fgColor='FFF2CC')
    wb = Workbook()
    score_map = {(r['metric'], r['model']): float(r['mean_percent'])
                 for r in summary.to_dict('records')}
    for metric, title in (('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced Accuracy')):
        ws = wb.active if len(wb.worksheets) == 1 and wb.active.max_row == 1 else wb.create_sheet()
        ws.title = title
        metric_title = 'Accuracy (%)' if metric == 'accuracy' else 'Balanced Accuracy (%)'
        rows = [
            ['student', 'method', '004 / session_3'],
            ['CBraMod', 'T_CE teacher reference', score_map[(metric, 'T_CE')]],
            ['IFNet', 'none', score_map[(metric, 'IFNet_CE')]],
            ['IFNet', 'fixed_P_feature', score_map[(metric, 'FIXED_P_FEATURE')]],
            ['IFNet', 'teacher_adapter_feature', score_map[(metric, 'TEACHER_ADAPTER_FEATURE')]],
            ['IFNet', 'fixed_P_student_adapter_feature', score_map[(metric, 'FIXED_P_STUDENT_ADAPTER_FEATURE')]],
            ['IFNet', 'dual_adapter_feature', score_map[(metric, 'DUAL_ADAPTER_FEATURE')]],
        ]
        ws.append([metric_title, 'n_seed=3; n_subject=9', 'mean (%)'])
        for row in rows: ws.append(row)
        ws.freeze_panes = 'A2'; ws.sheet_view.showGridLines = False
        for c in ws[1]: c.fill = fill_header; c.font = Font(color='FFFFFF', bold=True)
        ws.row_dimensions[1].height = 28
        for idx in range(2, ws.max_row + 1):
            ws.cell(idx, 3).number_format = '0.00'
            for c in ws[idx]: c.alignment = Alignment(horizontal='center', vertical='center')
            if idx == 2:
                for c in ws[idx]: c.fill = fill_teacher
            elif idx == 3:
                for c in ws[idx]: c.fill = fill_baseline
            else:
                value = float(ws.cell(idx, 3).value)
                teacher = score_map[(metric, 'T_CE')]
                student = score_map[(metric, 'IFNet_CE')]
                gt_t, gt_s = value > teacher, value > student
                fill = fill_both if gt_t and gt_s else fill_teacher_only if gt_t else fill_student_only if gt_s else None
                if fill:
                    for c in ws[idx]: c.fill = fill
        widths = {'A': 20, 'B': 42, 'C': 22}
        for col, width in widths.items(): ws.column_dimensions[col].width = width
        legend_row = ws.max_row + 2
        ws.cell(legend_row, 1, 'Legend')
        for col, label, fill in ((1, '大于二者', fill_both),
                                 (2, '大于教师', fill_teacher_only),
                                 (3, '大于学生', fill_student_only)):
            cell = ws.cell(legend_row + 1, col, label); cell.fill = fill
            cell.alignment = Alignment(horizontal='center')
        ws.cell(legend_row + 3, 1,
            '004 session_3；三 seed × 九个目标被试；seed 内被试等权平均后再对 seed 等权平均。')
        ws.merge_cells(start_row=legend_row + 3, start_column=1, end_row=legend_row + 3, end_column=3)
        ws.cell(legend_row + 3, 1).alignment = Alignment(wrap_text=True, vertical='top')
        ws.row_dimensions[legend_row + 3].height = 32
    wpd = wb.create_sheet('Paired differences')
    paired_view = paired[paired['mean_difference_pp'].notna()].copy()
    cols = ['metric', 'comparison', 'mean_difference_pp', 'ci_low_pp', 'ci_high_pp',
            'win', 'tie', 'loss', 'n_seed', 'n_subject']
    wpd.append(cols)
    for row in paired_view[cols].itertuples(index=False, name=None): wpd.append(list(row))
    wpd.freeze_panes = 'A2'; wpd.sheet_view.showGridLines = False
    for c in wpd[1]: c.fill = fill_header; c.font = Font(color='FFFFFF', bold=True)
    for col in wpd.columns:
        letter = get_column_letter(col[0].column)
        wpd.column_dimensions[letter].width = min(48, max(14, max(len(str(x.value or '')) for x in col) + 2))
        for cell in col[1:]: cell.alignment = Alignment(horizontal='center')
    for idx in range(2, wpd.max_row + 1):
        for col in (3, 4, 5): wpd.cell(idx, col).number_format = '0.00'
    wdg = wb.create_sheet('Diagnostics')
    if len(diagnostics):
        diag = diagnostics.copy()
        for row in [list(diag.columns), *diag.itertuples(index=False, name=None)]: wdg.append(list(row))
    else:
        wdg.append(['No diagnostics available'])
    wdg.freeze_panes = 'A2'; wdg.sheet_view.showGridLines = False
    for c in wdg[1]: c.fill = fill_header; c.font = Font(color='FFFFFF', bold=True)
    for col in wdg.columns:
        letter = get_column_letter(col[0].column)
        wdg.column_dimensions[letter].width = min(32, max(14, max(len(str(x.value or '')) for x in col) + 2))
    notes = wb.create_sheet('Notes')
    for row in [
        ['CBraMod → IFNet, BNCI2014004 session_3, 3 seeds × 9 LOSO folds, 7/1/1 source/feedback/target.'],
        ['CBraMod and IFNet use their historical model-specific inputs; apply_EA=false for CBraMod, no EA for IFNet.'],
        ['Only D_train updates IFNet and adapters. D_feedback is diagnostic. Target evaluation runs after all training endpoints are fixed.'],
        ['T_CE is the reused pure CE adapted teacher. MCC and logits KD are absent; training loss is CE + 0.5 paired feature MSE.'],
        ['Dual adapter: trainable 2400→512→512 teacher map and 512→512→512 student map; inference loads IFNet only.'],
        ['Fold-level paired differences average three seeds within each target subject, then bootstrap nine subjects (10000 draws).'],
        ['Seed means and subject SD are reported separately. CKA with zero centered variance is NA; low rank alone is not called collapse.'],
    ]: notes.append(row)
    notes.column_dimensions['A'].width = 130
    for row in notes.iter_rows():
        for cell in row: cell.alignment = Alignment(wrap_text=True, vertical='top')
    wb.save(path)


def _write_summary(config: dict, run_id_prefix: str) -> dict:
    metrics, predictions, diagnostics = [], [], []
    for seed in SEEDS:
        for fold in FOLDS:
            roots = _roots(run_id_prefix + f'_seed{seed}', seed, fold)
            eval_path = roots['run_result'] / 'fold_evaluation.json'
            mpath = roots['run_result'] / 'fold_evaluation_metrics.csv'
            ppath = roots['run_result'] / 'fold_predictions.csv'
            dpath = roots['run_result'] / 'target_feature_diagnostics.csv'
            if not all(Path(resolve_local_file(p)).is_file() for p in (eval_path, mpath, ppath, dpath)):
                raise FileNotFoundError(f'missing evaluation artifact seed={seed},fold={fold + 1}')
            record = _read_json(eval_path)
            if (record.get('status') != 'complete' or record.get('metrics_sha256') != _sha(mpath)
                    or record.get('predictions_sha256') != _sha(ppath)
                    or record.get('diagnostics_sha256') != _sha(dpath)):
                raise RuntimeError('evaluation artifact hash mismatch during summary')
            metrics.extend(pd.read_csv(resolve_local_file(mpath)).to_dict('records'))
            predictions.extend(pd.read_csv(resolve_local_file(ppath)).to_dict('records'))
            diagnostics.extend(pd.read_csv(resolve_local_file(dpath)).to_dict('records'))
            for condition in CONDITIONS:
                path = roots['run_result'] / f'{condition}_diagnostics.csv'
                if path.is_file():
                    diagnostics.extend(pd.read_csv(resolve_local_file(path)).to_dict('records'))
    metric_df = pd.DataFrame(metrics)
    expected = set(MODEL_NAMES)
    if (set(metric_df['model'].unique()) != expected
            or len(metric_df) != 27 * len(expected)
            or metric_df.duplicated(['model', 'seed', 'target_subject']).any()):
        raise RuntimeError('V3 evaluation matrix is incomplete, duplicated or has unexpected models')
    for name, group in metric_df.groupby('model'):
        if (len(group) != 27 or set(group['seed'].astype(int)) != set(SEEDS)
                or any(group[group['seed'].astype(int) == s]['target_subject'].nunique() != 9 for s in SEEDS)):
            raise RuntimeError(f'{name} does not have exactly 3×9 unique fold metrics')
    prediction_df = pd.DataFrame(predictions)
    if prediction_df.duplicated(['model', 'seed', 'target_subject', 'trial_uid']).any():
        raise RuntimeError('V3 predictions contain duplicate model/seed/fold/UID records')
    for (seed, subject), group in prediction_df.groupby(['seed', 'target_subject']):
        sets = {name: set(g['trial_uid']) for name, g in group.groupby('model')}
        if set(sets) != expected or any(value != sets['T_CE'] for value in sets.values()):
            raise RuntimeError(f'trial UID mismatch across models at seed={seed},subject={subject}')

    summary_rows = []
    bootstrap_rng = np.random.default_rng(int(config['report']['bootstrap_seed']))
    bootstrap_indices = bootstrap_rng.integers(0, 9, size=(BOOTSTRAP_DRAWS, 9))
    for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy'):
        pivot = metric_df.pivot(index=['seed', 'target_subject'], columns='model', values=metric)
        by_subject = pivot.groupby('target_subject').mean()
        for model_name in MODEL_NAMES:
            subject_values = by_subject[model_name].to_numpy(dtype=np.float64)
            seed_means = pivot[model_name].groupby('seed').mean()
            scale = 100.0 if metric != 'cross_entropy' else 1.0
            low, high = _bootstrap_ci(subject_values, bootstrap_indices)
            summary_rows.append({
                'metric': metric, 'model': model_name,
                'mean_percent': float(subject_values.mean() * scale),
                'subject_sd_percent': float(subject_values.std(ddof=1) * scale),
                'subject_bootstrap_ci_low': low * scale,
                'subject_bootstrap_ci_high': high * scale,
                'seed_666_mean': float(seed_means.loc[666] * scale),
                'seed_667_mean': float(seed_means.loc[667] * scale),
                'seed_668_mean': float(seed_means.loc[668] * scale),
                'seed_mean_sd': float(seed_means.reindex(SEEDS).std(ddof=1) * scale),
                'n_seed': 3, 'n_subject': 9,
            })
    comparisons = [
        ('DUAL_ADAPTER_FEATURE_vs_TEACHER_ADAPTER_FEATURE', 'DUAL_ADAPTER_FEATURE', 'TEACHER_ADAPTER_FEATURE'),
        ('DUAL_ADAPTER_FEATURE_vs_IFNet_CE', 'DUAL_ADAPTER_FEATURE', 'IFNet_CE'),
        ('FIXED_P_STUDENT_ADAPTER_FEATURE_vs_FIXED_P_FEATURE',
         'FIXED_P_STUDENT_ADAPTER_FEATURE', 'FIXED_P_FEATURE'),
        ('DUAL_ADAPTER_FEATURE_vs_FIXED_P_STUDENT_ADAPTER_FEATURE',
         'DUAL_ADAPTER_FEATURE', 'FIXED_P_STUDENT_ADAPTER_FEATURE'),
    ]
    paired_rows = []
    for metric in ('accuracy', 'balanced_accuracy'):
        pivot = metric_df.pivot(index=['seed', 'target_subject'], columns='model', values=metric)
        by_subject = pivot.groupby('target_subject').mean()
        for name, left, right in comparisons:
            differences = (by_subject[left] - by_subject[right]).to_numpy(dtype=np.float64) * 100.0
            low, high = _bootstrap_ci(differences, bootstrap_indices)
            paired_rows.append({
                'metric': metric, 'comparison': name,
                'mean_difference_pp': float(differences.mean()),
                'subject_sd_pp': float(differences.std(ddof=1)),
                'ci_low_pp': low, 'ci_high_pp': high,
                'win': int((differences > float(config['report']['tie_threshold_pp'])).sum()),
                'tie': int((np.abs(differences) <= float(config['report']['tie_threshold_pp'])).sum()),
                'loss': int((differences < -float(config['report']['tie_threshold_pp'])).sum()),
                'n_seed': 3, 'n_subject': 9,
            })
            for subject_id, value in zip(range(1, 10), differences):
                paired_rows.append({'metric': metric, 'comparison': name,
                    'subject': subject_id, 'difference_pp': float(value), 'n_seed': 3})

    report_dir = require_report_output(REPORTS_ROOT / config['report']['report_directory'] / 'summary.xlsx').parent
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_df, paired_df = pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    metric_df.to_csv(report_dir / 'fold_metrics.csv', index=False)
    prediction_df.to_csv(report_dir / 'trial_predictions.csv', index=False)
    summary_df.to_csv(report_dir / 'metric_summary.csv', index=False)
    paired_df.to_csv(report_dir / 'paired_differences.csv', index=False)
    diagnostics_df.to_csv(report_dir / 'feature_diagnostics.csv', index=False)
    aggregate = {
        'experiment_id': config['experiment_id'], 'run_id_prefix': run_id_prefix,
        'dataset': config['dataset'], 'selected_session': config['selected_session'],
        'seeds': list(SEEDS), 'folds_per_seed': 9, 'n_target_subjects': 9,
        'aggregation': 'average seeds within each target subject; then equal-weight subject summary',
        'paired_bootstrap': {'unit': 'target_subject', 'replicates': BOOTSTRAP_DRAWS,
                             'seed': config['report']['bootstrap_seed'],
                             'same_indices_across_comparisons': True},
        'target_evaluations': 27, 'models': list(MODEL_NAMES),
        'metrics': summary_rows,
        'paired_differences': [r for r in paired_rows if 'mean_difference_pp' in r],
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'target_protocol': 'no EA; target trial labels used only after all training endpoints were fixed',
    }
    _atomic_json(report_dir / 'aggregate.json', aggregate)
    _style_workbook(report_dir / 'summary.xlsx', summary_df, paired_df, diagnostics_df)
    _write_report_markdown(report_dir / 'summary.md', aggregate)
    return aggregate


def _write_report_markdown(path: Path, aggregate: dict) -> None:
    scores = {(x['metric'], x['model']): x['mean_percent'] for x in aggregate['metrics']}
    paired = {x['comparison']: x for x in aggregate['paired_differences']
              if x['metric'] == 'balanced_accuracy'}
    lines = [
        '# CBraMod → IFNet 双侧 adapter LOSO V3', '',
        '数据：BNCI2014004/session_3；3 seeds（666/667/668）× 9 个目标被试；每折7/1/1。',
        '本报告只评估 FEATURE；T_CE 复用；目标被试不参与训练/adapter拟合。', '',
        '| 条件 | Accuracy (%) | BA (%) |', '|---|---:|---:|',
    ]
    for model in MODEL_NAMES:
        lines.append(f"| {model} | {scores[('accuracy',model)]:.2f} | {scores[('balanced_accuracy',model)]:.2f} |")
    lines.extend(['', '配对差以目标被试为单位，seed先在被试内平均；bootstrap 95% CI：', '',
                  '| 比较（BA差，pp） | 均值 | 95% CI | W/T/L |', '|---|---:|---:|---:|'])
    for name, row in paired.items():
        lines.append(f"| {name} | {row['mean_difference_pp']:.2f} | [{row['ci_low_pp']:.2f}, {row['ci_high_pp']:.2f}] | {row['win']}/{row['tie']}/{row['loss']} |")
    lines.extend(['', '双侧 MSE、跨 trial 方差、有效秩、CKA、adapter 漂移和梯度诊断见工作簿与 `feature_diagnostics.csv`。',
                  '最终推理仅使用 IFNet 原分类路径。', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def _evaluate_all(config: dict, run_id_prefix: str, gpu: int,
                  progress_path: Path | None = None) -> dict:
    run_id_prefix = str(run_id_prefix)
    # Do not inspect target labels before all 27 training fold records are terminal.
    for seed in SEEDS:
        for fold in FOLDS:
            roots = _roots(run_id_prefix + f'_seed{seed}', seed, fold)
            complete_path = roots['run_result'] / 'fold_complete.json'
            if not complete_path.is_file() or _read_json(complete_path).get('status') != 'trained':
                raise RuntimeError(f'target evaluation requested before training finished: {seed}/{fold+1}')
    device = _device(gpu)
    completed = []
    for seed in SEEDS:
        for fold in FOLDS:
            run_id = run_id_prefix + f'_seed{seed}'
            _evaluate_fold(config, seed, fold, run_id, device)
            completed.append({'seed': seed, 'fold': fold + 1})
            _write_progress(progress_path, {'status': 'running', 'records': completed,
                                             'total_folds': 27})
    result = _write_summary(config, run_id_prefix)
    _write_progress(progress_path, {'status': 'completed', 'records': completed,
                                     'total_folds': 27})
    return result


def _enqueue(config_path: Path, config: dict, stage: str,
             queue_root: Path | None = None) -> list[str]:
    from experiments.distill import queue
    config_path = config_path.resolve(); config_sha = _sha(config_path)
    root = Path(queue_root or ROOT / config['storage']['queue_root']).resolve()
    matrix = [(666, 0)] if stage == 'smoke' else (
        [(seed, fold) for seed in SEEDS for fold in FOLDS] if stage == 'train' else [(0, 0)])
    if stage not in ('smoke', 'train', 'evaluate'):
        raise ValueError(f'unknown queue stage {stage}')
    jobs = []
    run_prefix = config['storage']['external_run_id_prefix']
    for seed, fold in matrix:
        run_id = run_prefix + ('_smoke_seed666' if stage == 'smoke' else f'_seed{seed}')
        snapshot = {
            'kind': 'cbramod_ifnet_dual_adapter_loso_v3',
            'stage': stage, 'config': str(config_path), 'config_sha256': config_sha,
            'run_id_prefix': run_prefix, 'run_id': run_id, 'seed': seed, 'fold': fold,
            'args': {'dataset': 'BNCI2014004', 'teacher': 'CBraMod', 'student': 'IFNet',
                     'seeds': list(SEEDS) if stage == 'evaluate' else [seed],
                     'keys': list(FOLDS) if stage == 'evaluate' else [fold],
                     'methods': list(CONDITIONS), 'epochs': 100,
                     'batch_size': 16, 'lr': 0.001, 'weight_decay': 0.01},
        }
        jobs.append(queue.enqueue_snapshot(snapshot, root=root,
            argv=['cbramod_ifnet_dual_adapter_loso_v3', '--stage', stage,
                  '--seed', str(seed), '--fold', str(fold)],
            label=f'CBraMod->IFNet dual-adapter V3 {stage} seed={seed} fold={fold+1}'))
    return jobs


def _watch_training(config_path: Path, config: dict, queue_root: Path,
                    poll_seconds: float = 30.0) -> dict:
    from experiments.distill.queue import QueueStore, validated_root
    store = QueueStore(validated_root(queue_root))
    run_prefix = config['storage']['external_run_id_prefix']
    status_path = Path('/data1/llx/BigSmallcollab/results/distill') / f'{run_prefix}_orchestration.json'
    while True:
        matches, evaluations = [], []
        for state in store.states():
            spec = _read_json(store.job_dir(state['id']) / 'spec.json')
            snap = spec.get('snapshot', {})
            if (snap.get('kind') == 'cbramod_ifnet_dual_adapter_loso_v3'
                    and snap.get('run_id_prefix') == run_prefix):
                if snap.get('stage') == 'train': matches.append(state)
                elif snap.get('stage') == 'evaluate': evaluations.append(state)
        failed = [x for x in matches if x['status'] in ('failed', 'cancelled', 'paused')]
        if failed:
            result = {'status': 'training_incomplete', 'run_id_prefix': run_prefix,
                'training_job_count': len(matches), 'failed_jobs': [x['id'] for x in failed],
                'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            _atomic_json(status_path, result); return result
        if len(matches) == 27 and all(x['status'] == 'completed' for x in matches):
            if evaluations:
                result = {'status': 'evaluation_already_queued', 'run_id_prefix': run_prefix,
                    'evaluation_jobs': [x['id'] for x in evaluations],
                    'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            else:
                ids = _enqueue(config_path, config, 'evaluate', queue_root)
                result = {'status': 'evaluation_queued', 'run_id_prefix': run_prefix,
                    'training_job_count': len(matches), 'evaluation_jobs': ids,
                    'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            _atomic_json(status_path, result); return result
        _atomic_json(status_path, {'status': 'waiting_for_training',
            'run_id_prefix': run_prefix, 'training_job_count': len(matches),
            'completed_jobs': sum(x['status'] == 'completed' for x in matches),
            'active_or_pending_jobs': sum(x['status'] in {'pending','starting','running','stopping'} for x in matches),
            'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        time.sleep(float(poll_seconds))


def main(argv=None) -> int:
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
    parser.add_argument('--enqueue', choices=('smoke', 'train', 'evaluate'))
    parser.add_argument('--watch-training', action='store_true')
    parser.add_argument('--watch-poll-seconds', type=float, default=30.0)
    parser.add_argument('--queue-root', type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    config_sha = _sha(config_path)
    if args.config_sha256 and args.config_sha256 != config_sha:
        raise RuntimeError('queued V3 config hash mismatch')
    if args.preflight:
        report = preflight(config_path, config)
        out = require_report_output(REPORTS_ROOT / config['report']['report_directory'] / 'preflight.json')
        out.parent.mkdir(parents=True, exist_ok=True); _atomic_json(out, report)
        print(json.dumps({'status': report['status'], 'verified_folds': report['verified_folds'],
                          'missing': report['missing']}, ensure_ascii=False))
        return 0 if report['status'] == 'ready' else 2
    if args.enqueue:
        jobs = _enqueue(config_path, config, args.enqueue, args.queue_root)
        print(json.dumps({'stage': args.enqueue, 'jobs': jobs}, ensure_ascii=False)); return 0
    run_prefix = args.run_id_prefix or config['storage']['external_run_id_prefix']
    if args.watch_training:
        result = _watch_training(config_path, config,
            args.queue_root or ROOT / config['storage']['queue_root'], args.watch_poll_seconds)
        print(json.dumps(result, ensure_ascii=False)); return 0
    if args.evaluate:
        result = _evaluate_all(config, run_prefix, args.gpu, args.progress)
        print(json.dumps({'status': 'evaluated', 'target_evaluations': result['target_evaluations']}, ensure_ascii=False))
        return 0
    if args.seed is None or args.fold is None:
        parser.error('--seed and --fold are required for a training worker')
    run_id = args.run_id or (run_prefix + f'_seed{args.seed}')
    try:
        result = _train_fold(config_path, config, args.seed, args.fold, run_id,
            args.gpu, smoke=args.smoke, progress_path=args.progress, stop_file=args.stop_file)
        print(json.dumps(result, ensure_ascii=False)); return 0
    except SystemExit as exc:
        if exc.code == 75: return 75
        raise
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        if args.progress:
            _write_progress(args.progress, {'status': 'failed', 'seed': args.seed,
                'fold': args.fold, 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
