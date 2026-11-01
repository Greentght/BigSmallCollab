"""CBraMod full [3,4,200] feature projection into IFNet's native 512-D space."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'BIGSMALLCOLLAB_CPU_THREADS', 'TORCH_NUM_THREADS',
              'TORCH_NUM_INTEROP_THREADS'):
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

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch.nn import functional as F
import yaml

from collab import cbramod_t2s_projection as projection_core
from collab import mcc_diffkd as diffusion_core
from experiments.distill import run_cbramod_mcc_diffkd_loso as channel_run
from experiments.distill import run_student_aware_diffusion_loso as reference
from experiments.storage import (REPORTS_ROOT, require_external_output,
                                 require_report_output, resolve_local_file)
from models import get_adapter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/loso/cbramod_t2s_full_feature_loso_v1.yaml'
SEEDS = (666, 667, 668)
CONDITIONS = ('CE_T2S_FEATURE', 'MCC_T2S_FEATURE',
              'CE_T2S_DIFFKD', 'MCC_T2S_DIFFKD')
TEACHER_FOR = {'CE_T2S_FEATURE': 'T_CE', 'MCC_T2S_FEATURE': 'T_MCC',
               'CE_T2S_DIFFKD': 'T_CE', 'MCC_T2S_DIFFKD': 'T_MCC'}
IS_DIFFUSION = {'CE_T2S_FEATURE': False, 'MCC_T2S_FEATURE': False,
                'CE_T2S_DIFFKD': True, 'MCC_T2S_DIFFKD': True}


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), default=str).encode('utf-8')).hexdigest()


def _read_json(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        return json.load(stream)


def _atomic_json(path: Path, value: dict) -> None:
    channel_run._atomic_json(Path(path), value)


def _atomic_npz(path: Path, **arrays) -> None:
    channel_run._atomic_npz(Path(path), **arrays)


def _atomic_torch(path: Path, value: dict) -> None:
    channel_run._atomic_torch(Path(path), value)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    channel_run._atomic_csv(Path(path), rows)


def _write_progress(path: Path | None, value: dict) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n',
                         encoding='utf-8')
    os.replace(temporary, target)


def _read_config(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _validate(config: dict, seed: int) -> None:
    if seed not in SEEDS or seed not in config.get('seeds', []):
        raise ValueError(f'unsupported seed {seed}')
    if config.get('experiment_id') != 'CBRAMOD_T2S_FULL_FEATURE_LOSO_V1':
        raise ValueError('unexpected experiment ID')
    if config.get('dataset') != 'BNCI2014004' or config.get('selected_session') != 'session_3':
        raise ValueError('this runner is fixed to BNCI2014004 session_3')
    if config.get('conditions') != list(CONDITIONS):
        raise ValueError('formal condition order/names were changed')
    if config.get('reference_run', {}).get('resolved_config_sha256_by_seed') != channel_run.RESOLVED_HASHES:
        raise ValueError('audited historical resolved hashes differ')
    fs = config.get('feature_space', {})
    if (fs.get('teacher_native_shape') != [3, 4, 200]
            or fs.get('teacher_flat_dimension') != 2400
            or fs.get('student_native_dimension') != 512
            or fs.get('teacher_projection_shape') != [512, 2400]
            or fs.get('teacher_projection') != 'fixed_gaussian_thin_qr_row_orthogonal'):
        raise ValueError('fixed full-feature projection specification was changed')
    if config.get('diffusion', {}).get('ddim_timesteps') != [250, 200, 150, 100, 50, 0]:
        raise ValueError('fixed DDIM schedule was changed')


def _device(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError('formal CBraMod T2S training requires CUDA')
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
    roots = channel_run._roots(run_id, seed, fold)
    for key in ('run_result', 'run_cache', 'run_weights', 'run_seed_weights'):
        roots[key] = require_external_output(roots[key])
        roots[key].mkdir(parents=True, exist_ok=True)
    return roots


def _resolved(config: dict, seed: int):
    historical_config = dict(config)
    historical_config['seed'] = seed
    historical_config['reference_run'] = {
        'protocol_id': config['reference_run']['protocol_id'],
        'seed': seed,
        'resolved_config_sha256': config['reference_run']['resolved_config_sha256_by_seed'][seed],
    }
    return reference._load_resolved_reference(historical_config)


def _load_fold(config: dict, seed: int, fold: int, roots: dict,
               device: torch.device, include_test: bool):
    historical, cfg = _resolved(config, seed)
    run_config = dict(config)
    run_config['seed'] = seed
    run_config['reference_run'] = {
        'protocol_id': config['reference_run']['protocol_id'],
        'seed': seed, 'resolved_config_sha256': cfg['resolved_config_sha256']}
    identity = reference._load_fold_identity(
        run_config, fold, roots, cfg['resolved_config_sha256'],
        include_test_labels=include_test)
    data = channel_run._load_data(identity, roots, historical['student_config'], device)
    student = channel_run._load_student_reference(
        run_config, seed, fold, identity, roots, historical, cfg, data, device)
    old_id = config['teachers']['historical_run_id_prefix'] + f'_seed{seed}'
    old_roots = channel_run._roots(old_id, seed, fold)
    old_resolved_path = old_roots['run_result'] / 'resolved_run.json'
    old_resolved = _read_json(old_resolved_path)
    if (old_resolved.get('resolved_config_sha256') != cfg['resolved_config_sha256']
            or old_resolved.get('train_uid_sha256') != identity['uid_sha256']['train']
            or old_resolved.get('feedback_uid_sha256') != identity['uid_sha256']['feedback']):
        raise RuntimeError(f'historical teacher provenance does not match seed={seed}, fold={fold + 1}')
    for name in ('T0', 'T_CE', 'T_MCC'):
        manifest = _read_json(old_roots['run_result'] / f'teacher_{name}.json')
        if manifest.get('status') != 'complete':
            raise RuntimeError(f'missing complete old {name} record: seed={seed}, fold={fold + 1}')
    teacher_cfg = old_resolved.get('cbramod_config')
    if not isinstance(teacher_cfg, dict) or not teacher_cfg.get('pretrain'):
        raise RuntimeError('per-fold historical resolved_run.json has no CBraMod config')
    if teacher_cfg.get('feature_head') != 'flatten' or teacher_cfg.get('skip_preprocess') is not True:
        raise RuntimeError('historical CBraMod recipe is not the audited full flatten path')
    return historical, cfg, run_config, identity, data, student, old_roots, old_resolved, teacher_cfg


def _fingerprint(config_path: Path, config: dict, seed: int, fold: int,
                 cfg: dict, identity: dict, student: dict, old_resolved: dict,
                 teacher_cfg: dict) -> str:
    return _hash_json({
        'experiment_id': config['experiment_id'], 'config_sha256': _sha(config_path),
        'implementation_sha256': _sha(__file__),
        'projection_helper_sha256': _sha(ROOT / 'collab/cbramod_t2s_projection.py'),
        'teacher_core_sha256': _sha(ROOT / 'experiments/distill/run_cbramod_mcc_diffkd_loso.py'),
        'diffusion_core_sha256': _sha(ROOT / 'collab/mcc_diffkd.py'),
        'seed': seed, 'fold': fold, 'split_manifest_sha256': identity['sha256'],
        'resolved_config_sha256': cfg['resolved_config_sha256'],
        'historical_teacher_resolved_sha256': _sha(
            channel_run._roots(config['teachers']['historical_run_id_prefix'] + f'_seed{seed}', seed, fold)['run_result']
            / 'resolved_run.json'),
        'historical_teacher_config': teacher_cfg,
        'historical_teacher_fingerprint': old_resolved.get('run_fingerprint'),
        'pretrained_checkpoint_sha256': _sha(teacher_cfg['pretrain']),
        'source_manifest_sha256': identity['source_manifest_sha256'],
        'source_trials_sha256': identity['source_trial_manifest_sha256'],
        'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'IFNet_schedule_sha256': student['student_schedule_sha256'],
        'full_teacher_shape': [3, 4, 200], 'projection_shape': [512, 2400],
    })


def _extract_full_teacher(model, adapter, x: np.ndarray, device: torch.device,
                          check_forward: bool = False, batch_size: int = 32):
    model.eval()
    feature_parts, logit_parts = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            backbone = model.backbone(xb)
            if backbone.ndim != 4 or tuple(backbone.shape[1:]) != (3, 4, 200):
                raise RuntimeError(f'CBraMod full feature shape changed: {tuple(backbone.shape)}')
            flattened = model.flatten(backbone)
            if flattened.shape[1:] != (2400,):
                raise RuntimeError(f'CBraMod classifier input is not 2400-D: {tuple(flattened.shape)}')
            logits = model.classifier(model.dropout(flattened))
            if logits.shape[1:] != (2,):
                raise RuntimeError(f'CBraMod classifier output changed: {tuple(logits.shape)}')
            if check_forward and start == 0:
                _, adapter_logits = adapter.forward(model, xb)
                if not torch.allclose(logits, adapter_logits, rtol=1e-5, atol=1e-5):
                    raise RuntimeError('manual full-feature logits differ from the adapter forward')
            feature_parts.append(flattened.detach().float().cpu().numpy())
            logit_parts.append(logits.detach().float().cpu().numpy())
    return np.concatenate(feature_parts), np.concatenate(logit_parts)


def _projection_record(roots: dict, seed: int, fold: int, fingerprint: str):
    path = roots['run_cache'] / 'fixed_teacher_to_IFNet_P.npz'
    meta_path = roots['run_cache'] / 'fixed_teacher_to_IFNet_P.json'
    actual_seed = projection_core.derive_seed('teacher_projection', seed, fold)
    if path.exists() or meta_path.exists():
        if not (path.exists() and meta_path.exists()):
            raise RuntimeError('incomplete fixed T2S projection artifact')
        meta = _read_json(meta_path)
        if (meta.get('fingerprint') != fingerprint or meta.get('seed') != actual_seed
                or meta.get('sha256') != _sha(path)):
            raise RuntimeError('fixed T2S projection identity/hash mismatch')
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            matrix64 = np.asarray(saved['P_float64'], dtype=np.float64)
            matrix32 = np.asarray(saved['P_float32'], dtype=np.float32)
        return matrix64, matrix32, meta
    matrix64 = projection_core.fixed_teacher_projection(actual_seed)
    matrix32 = matrix64.astype(np.float32)
    _atomic_npz(path, P_float64=matrix64, P_float32=matrix32)
    meta = {
        'status': 'complete', 'fingerprint': fingerprint, 'seed': actual_seed,
        'file': str(path), 'sha256': _sha(path), 'shape': [512, 2400],
        'construction': 'CPU float64 Gaussian [2400,512] thin QR; canonical R signs; transpose',
        'row_orthogonality_max_abs_error': float(np.max(np.abs(
            matrix64 @ matrix64.T - np.eye(512)))),
        'random_stream': 'SHA-256(experiment_id/stage/seed/fold)',
    }
    _atomic_json(meta_path, meta)
    return matrix64, matrix32, meta


def _teacher_targets(roots: dict, identity: dict, data: dict, teachers: dict,
                     adapter, teacher_cfg: dict, projection64: np.ndarray,
                     projection32: np.ndarray, fingerprint: str, device: torch.device,
                     smoke: bool):
    path = roots['run_cache'] / 'full_teacher_targets.npz'
    meta_path = roots['run_cache'] / 'full_teacher_targets.json'
    if path.exists() or meta_path.exists():
        if not (path.exists() and meta_path.exists()):
            raise RuntimeError('incomplete full teacher targets')
        meta = _read_json(meta_path)
        if (meta.get('fingerprint') != fingerprint or meta.get('file_sha256') != _sha(path)
                or meta.get('train_uid_sha256') != identity['uid_sha256']['train']):
            raise RuntimeError('full teacher targets identity/hash mismatch')
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]) for key in saved.files}
        diagnostic_path = roots['run_result'] / 'teacher_projection_diagnostics.csv'
        if not diagnostic_path.is_file():
            _write_teacher_projection_diagnostics(
                diagnostic_path, identity, arrays, projection64)
        return arrays, meta

    arrays = {
        'train_uids': identity['uids']['train'], 'train_y': identity['labels']['train'],
        'feedback_uids': identity['uids']['feedback'], 'feedback_y': identity['labels']['feedback'],
    }
    for name in ('T0', 'T_CE', 'T_MCC'):
        for part in ('train', 'feedback'):
            features, logits = _extract_full_teacher(
                teachers[name], adapter, data['teacher_x'][part], device,
                check_forward=smoke or name == 'T0')
            labels = np.asarray(identity['labels'][part], dtype=np.int64)
            projected, normalized = projection_core.project_teacher_features(
                features, projection32, device)
            arrays[f'{name}_{part}_features_full'] = features.astype(np.float32)
            arrays[f'{name}_{part}_logits'] = logits.astype(np.float32)
            arrays[f'{name}_{part}_features_projected'] = projected.astype(np.float32)
            arrays[f'{name}_{part}_features_normalized'] = normalized.astype(np.float32)
            weight = teachers[name].classifier.weight.detach().float().cpu().numpy()
            bias = teachers[name].classifier.bias.detach().float().cpu().numpy()
            arrays[f'{name}_classifier_weight'] = weight.astype(np.float32)
            arrays[f'{name}_classifier_bias'] = bias.astype(np.float32)
            diag = projection_core.projection_readout_diagnostic(
                features, logits, weight, bias, projection64, labels)
            projected_logits = diag.pop('projected_logits')
            arrays[f'{name}_{part}_projected_readout_logits'] = projected_logits
    diagnostic_path = roots['run_result'] / 'teacher_projection_diagnostics.csv'
    diagnostics = _write_teacher_projection_diagnostics(
        diagnostic_path, identity, arrays, projection64)
    _atomic_npz(path, **arrays)
    meta = {
        'status': 'complete', 'fingerprint': fingerprint, 'file': str(path),
        'file_sha256': _sha(path), 'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'feature_tap': 'CBraMod backbone [B,3,4,200], reshape only to [B,2400]; no pooling/mean',
        'projected_feature': 'fixed P [512,2400], then no-affine LayerNorm eps=1e-5',
        'teacher_state_sha256': {name: channel_run._read_json(
            roots['run_result'] / f'teacher_{name}.json')['model_sha256']
            for name in ('T0', 'T_CE', 'T_MCC')},
        'diagnostics': str(diagnostic_path),
    }
    _atomic_json(meta_path, meta)
    return arrays, meta


def _write_teacher_projection_diagnostics(path: Path, identity: dict,
                                          arrays: dict, projection64: np.ndarray):
    rows = []
    for name in ('T0', 'T_CE', 'T_MCC'):
        weight = arrays[f'{name}_classifier_weight']
        bias = arrays[f'{name}_classifier_bias']
        for part in ('train', 'feedback'):
            features = arrays[f'{name}_{part}_features_full']
            logits = arrays[f'{name}_{part}_logits']
            labels = np.asarray(identity['labels'][part], dtype=np.int64)
            diag = projection_core.projection_readout_diagnostic(
                features, logits, weight, bias, projection64, labels)
            projected_logits = diag.pop('projected_logits')
            rows.append({'teacher': name, 'partition': part,
                **channel_run._metric_values(labels, logits),
                **{f'projection_{key}': value for key, value in diag.items()},
                **{f'projected_readout_{key}': value for key, value in
                   channel_run._metric_values(labels, projected_logits).items()}})
    _atomic_csv(path, rows)
    return rows


def _prior_initial(seed: int, fold: int, config: dict):
    actual_seed = projection_core.derive_seed('prior_initial', seed, fold)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(actual_seed)
        model = diffusion_core.EpsilonPredictor(
            512, int(config['diffusion']['hidden_dimensions']),
            int(config['diffusion']['time_embedding_dimensions']))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, reference._state_hash(state), actual_seed


def _fit_prior(name: str, config: dict, identity: dict, roots: dict,
               targets: dict, fingerprint: str, initial_state: dict,
               initial_hash: str, prior_seed: int, device: torch.device,
               smoke: bool = False):
    final_path = roots['run_weights'] / f'prior_{name}.pt'
    manifest_path = roots['run_result'] / f'prior_{name}.json'
    history_path = roots['run_result'] / f'prior_{name}_history.csv'
    train = np.asarray(targets[f'{name}_train_features_normalized'], dtype=np.float32)
    feedback = np.asarray(targets[f'{name}_feedback_features_normalized'], dtype=np.float32)
    updates = 4 if smoke else int(config['diffusion']['training_updates'])
    method_fp = _hash_json({'run_fingerprint': fingerprint, 'teacher': name,
        'initial_prior_sha256': initial_hash, 'updates': updates,
        'teacher_targets_sha256': _sha(roots['run_cache'] / 'full_teacher_targets.npz')})
    if final_path.exists() or manifest_path.exists():
        if not (final_path.exists() and manifest_path.exists()):
            raise RuntimeError(f'{name} prior checkpoint/manifest is incomplete')
        manifest = _read_json(manifest_path)
        if (manifest.get('method_fingerprint') != method_fp
                or manifest.get('checkpoint_sha256') != _sha(final_path)):
            raise RuntimeError(f'{name} prior identity/hash mismatch')
        model = diffusion_core.EpsilonPredictor(512,
            int(config['diffusion']['hidden_dimensions']),
            int(config['diffusion']['time_embedding_dimensions'])).to(device)
        payload = torch.load(resolve_local_file(final_path), map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False); p.grad = None
        return model, manifest

    diffusion = config['diffusion']
    model = diffusion_core.EpsilonPredictor(512, int(diffusion['hidden_dimensions']),
                                             int(diffusion['time_embedding_dimensions'])).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(diffusion['lr']),
        betas=tuple(diffusion['betas']), eps=float(diffusion['eps']),
        weight_decay=float(diffusion['weight_decay']))
    _, alpha, sigma = diffusion_core.ddpm_schedule(
        int(diffusion['steps']), float(diffusion['beta_start']),
        float(diffusion['beta_end']), device)
    train_tensor = torch.as_tensor(train, dtype=torch.float32, device=device)
    batch_size = min(int(diffusion['batch_size_cap']), len(train_tensor))
    generator = torch.Generator(device=device).manual_seed(int(prior_seed))
    fixed = {}
    for part, values in (('train', train), ('feedback', feedback)):
        rng = np.random.default_rng(int(prior_seed) + (0 if part == 'train' else 1))
        count = min(128, len(values))
        ids = rng.choice(len(values), count, replace=False)
        times = rng.integers(1, int(diffusion['steps']) + 1, count)
        noise = rng.standard_normal((count, 512)).astype(np.float32)
        fixed[part] = (np.asarray(values, dtype=np.float32)[ids], times, noise)

    def measure(part):
        clean_np, times_np, noise_np = fixed[part]
        clean = torch.as_tensor(clean_np, dtype=torch.float32, device=device)
        times = torch.as_tensor(times_np, dtype=torch.long, device=device)
        eps = torch.as_tensor(noise_np, dtype=torch.float32, device=device)
        noisy = alpha[times, None] * clean + sigma[times, None] * eps
        with torch.no_grad():
            pred = model(noisy, times)
            return float(F.mse_loss(pred, eps).cpu()), float(eps.square().mean().cpu())

    history = []
    for update in range(0, updates + 1):
        if update > 0:
            ids = torch.randint(len(train_tensor), (batch_size,), device=device, generator=generator)
            times = torch.randint(1, int(diffusion['steps']) + 1, (batch_size,),
                                  dtype=torch.long, device=device, generator=generator)
            eps = torch.randn((batch_size, 512), dtype=torch.float32,
                              device=device, generator=generator)
            noisy = alpha[times, None] * train_tensor[ids] + sigma[times, None] * eps
            prediction = model(noisy, times)
            loss = F.mse_loss(prediction, eps)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name} prior has non-finite loss at update {update}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(diffusion['gradient_clip_norm']))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'{name} prior has non-finite gradient at update {update}')
            optimizer.step()
        if update == 0 or update % int(diffusion['diagnostic_interval']) == 0 or update == updates:
            tr, tr0 = measure('train'); fb, fb0 = measure('feedback')
            history.append({'teacher': name, 'update': update,
                'train_noise_mse': tr, 'train_zero_predictor_mse': tr0,
                'feedback_noise_mse': fb, 'feedback_zero_predictor_mse': fb0,
                **({'batch_loss': float(loss.detach().cpu()),
                    'gradient_norm': float(grad_norm.detach().cpu())} if update else {})})
            _atomic_csv(history_path, history)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_torch(final_path, {'model': state, 'method_fingerprint': method_fp,
                               'model_sha256': reference._state_hash(state)})
    manifest = {'status': 'complete', 'teacher': name, 'method_fingerprint': method_fp,
        'run_fingerprint': fingerprint, 'checkpoint': str(final_path),
        'checkpoint_sha256': _sha(final_path), 'model_sha256': reference._state_hash(state),
        'initial_model_sha256': initial_hash, 'final_update': updates,
        'feature_dimension': 512, 'training_partition': 'D_train only',
        'conditioning': 'none', 'noise_schedule': {'steps': diffusion['steps'],
            'beta_start': diffusion['beta_start'], 'beta_end': diffusion['beta_end']},
        'history': str(history_path)}
    _atomic_json(manifest_path, manifest)
    return model, manifest


def _student_feedback_preserving_state(model, adapter, x, y, device):
    state = reference._cpu_tree(model.state_dict())
    before = reference._state_hash(state)
    metrics = channel_run._student_feedback(model, adapter, x, y, device)
    if reference._state_hash(model.state_dict()) != before:
        model.load_state_dict(state, strict=True)
    if reference._state_hash(model.state_dict()) != before:
        raise RuntimeError('IFNet feedback diagnostic changed model state')
    return metrics


def _fit_student(condition: str, config: dict, identity: dict, roots: dict,
                 models: dict, data: dict, initial: dict, targets: dict,
                 prior: torch.nn.Module | None, fingerprint: str,
                 device: torch.device, smoke: bool = False):
    final_path = roots['run_weights'] / f'student_{condition}.pt'
    manifest_path = roots['run_result'] / f'student_{condition}.json'
    teacher = TEACHER_FOR[condition]
    use_diffusion = IS_DIFFUSION[condition]
    epochs = 2 if smoke else int(models['student_cfg']['epochs'])
    target_sha = _sha(roots['run_cache'] / 'full_teacher_targets.npz')
    projection_sha = _sha(roots['run_cache'] / 'fixed_teacher_to_IFNet_P.npz')
    prior_sha = _sha(roots['run_weights'] / f'prior_{teacher}.pt') if use_diffusion else None
    method_fp = _hash_json({'run_fingerprint': fingerprint, 'condition': condition,
        'teacher': teacher, 'uses_diffusion': use_diffusion,
        'initial_IFNet_sha256': initial['model_sha256'], 'target_sha256': target_sha,
        'projection_sha256': projection_sha, 'prior_sha256': prior_sha, 'epochs': epochs})
    if final_path.exists() or manifest_path.exists():
        if not (final_path.exists() and manifest_path.exists()):
            raise RuntimeError(f'{condition} student checkpoint/manifest is incomplete')
        manifest = _read_json(manifest_path)
        if (manifest.get('method_fingerprint') != method_fp
                or manifest.get('checkpoint_sha256') != _sha(final_path)):
            raise RuntimeError(f'{condition} student checkpoint identity/hash mismatch')
        return manifest

    paired_seed = int(config['randomness']['student_dropout_seed_base']) + 103 * identity['manifest']['target_subject_id'] + int(config['seed'])
    _seed_all(paired_seed, device)
    model, optimizer, scheduler, _ = reference._new_paired_student(
        models, initial, int(config['seed']), device)
    train_x = torch.as_tensor(data['student_x']['train'], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(identity['labels']['train'], dtype=torch.long, device=device)
    targets_tensor = torch.as_tensor(targets[f'{teacher}_train_features_normalized'],
                                     dtype=torch.float32, device=device)
    prior_generator_seed = int(config['randomness']['student_diffusion_noise_seed_base']) + \
        109 * identity['manifest']['target_subject_id'] + int(config['seed'])
    noise_generator = torch.Generator(device=device).manual_seed(prior_generator_seed) if use_diffusion else None
    alpha = sigma = None
    if use_diffusion:
        _, alpha, sigma = diffusion_core.ddpm_schedule(
            int(config['diffusion']['steps']), float(config['diffusion']['beta_start']),
            float(config['diffusion']['beta_end']), device)
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False); p.grad = None
    history, feedback_rows = [], []
    first_gradient = None
    schedule = models['student_schedule']
    for epoch_index in range(epochs):
        model.train()
        ce_sum = feature_sum = total_sum = grad_sum = 0.0
        count_sum = step_count = 0
        for ids_np in schedule[epoch_index]:
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            x_batch, y_batch = train_x[ids], train_y[ids]
            optimizer.zero_grad(set_to_none=True)
            native, logits = models['student_adapter'].forward(model, x_batch)
            if native.ndim != 2 or native.shape[1] != 512:
                raise RuntimeError(f'IFNet native feature changed: {tuple(native.shape)}')
            native.retain_grad()
            z_student = projection_core.layer_norm_no_affine(
                native, float(config['feature_space']['layer_norm_epsilon']))
            if use_diffusion:
                noise = torch.randn(z_student.shape, dtype=torch.float32,
                                    device=device, generator=noise_generator)
                matched = diffusion_core.ddim_denoise(
                    z_student, noise, prior, alpha, sigma,
                    tuple(config['diffusion']['ddim_timesteps']))
            else:
                matched = z_student
            feature_loss = F.mse_loss(matched, targets_tensor[ids].detach(), reduction='mean')
            ce = F.cross_entropy(logits, y_batch)
            loss = ce + float(config['feature_space']['feature_weight']) * feature_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{condition} non-finite objective at epoch {epoch_index + 1}')
            loss.backward()
            if first_gradient is None:
                native_norm = float(native.grad.norm().detach().cpu()) if native.grad is not None else 0.0
                stem_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                    and float(p.grad.norm().detach().cpu()) > 0 for p in model.parameters())
                first_gradient = {'native_feature_grad_norm': native_norm,
                                  'student_parameter_gradient_reached': bool(stem_grad)}
                if native_norm <= 0 or not stem_grad:
                    raise RuntimeError(f'{condition} feature gradient did not reach IFNet')
                if use_diffusion and any(p.grad is not None for p in prior.parameters()):
                    raise RuntimeError(f'{condition} prior parameters received gradients')
            grad_parts = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_parts))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'{condition} non-finite student gradient')
            optimizer.step()
            batch_count = len(ids)
            ce_sum += float(ce.detach().cpu()) * batch_count
            feature_sum += float(feature_loss.detach().cpu()) * batch_count
            total_sum += float(loss.detach().cpu()) * batch_count
            grad_sum += float(grad_norm.detach().cpu())
            count_sum += batch_count; step_count += 1
        lr_used = float(optimizer.param_groups[0]['lr'])
        scheduler.step()
        history.append({'condition': condition, 'teacher': teacher,
            'epoch': epoch_index + 1, 'train_ce': ce_sum / count_sum,
            'train_feature_mse': feature_sum / count_sum,
            'weighted_feature_mse': float(config['feature_space']['feature_weight']) * feature_sum / count_sum,
            'train_total': total_sum / count_sum, 'mean_gradient_norm': grad_sum / step_count,
            'lr_used': lr_used, 'lr_next': float(scheduler.get_last_lr()[0]),
            'paired_dropout_seed': paired_seed, 'noise_seed': prior_generator_seed if use_diffusion else None,
            'train_uid_sha256': identity['uid_sha256']['train'],
            'schedule_sha256': models['student_schedule_sha256'], **(first_gradient or {})})
        if (epoch_index + 1) in (1, 10, 50, 100) or (smoke and epoch_index + 1 == epochs):
            metrics = _student_feedback_preserving_state(
                model, models['student_adapter'], data['student_x']['feedback'],
                identity['labels']['feedback'], device)
            feedback_rows.append({'condition': condition, 'epoch': epoch_index + 1,
                **{f'feedback_{key}': value for key, value in metrics.items()}})
        _atomic_csv(roots['run_result'] / f'{condition}_history.csv', history)
        if feedback_rows:
            _atomic_csv(roots['run_result'] / f'{condition}_feedback.csv', feedback_rows)
    model.eval()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_torch(final_path, {'model': state, 'method_fingerprint': method_fp,
                               'model_sha256': reference._state_hash(state),
                               'condition': condition})
    manifest = {'status': 'complete', 'condition': condition, 'teacher': teacher,
        'uses_diffusion': use_diffusion, 'method_fingerprint': method_fp,
        'run_fingerprint': fingerprint, 'checkpoint': str(final_path),
        'checkpoint_sha256': _sha(final_path), 'model_sha256': reference._state_hash(state),
        'initial_IFNet_sha256': initial['model_sha256'], 'final_epoch': epochs,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'schedule_sha256': models['student_schedule_sha256'],
        'loss': {'ce_weight': 1.0, 'feature_weight': config['feature_space']['feature_weight'],
                 'feature_dimension': 512, 'feature': 'MSE(z_S, stopgrad(z_T)); z_T=P_T f_T',
                 'logits_kd': False, 'start_epoch': 1},
        'teacher_targets_sha256': target_sha, 'projection_sha256': projection_sha,
        'prior_checkpoint_sha256': prior_sha,
        'paired_dropout_seed': paired_seed,
        'student_diffusion_noise_seed': prior_generator_seed if use_diffusion else None,
        'history': str(roots['run_result'] / f'{condition}_history.csv'),
        'feedback_diagnostics': str(roots['run_result'] / f'{condition}_feedback.csv')}
    _atomic_json(manifest_path, manifest)
    return manifest


def _train_fold(config_path: Path, config: dict, seed: int, run_id: str,
                fold: int, device: torch.device, smoke: bool = False) -> dict:
    roots = _roots(run_id, seed, fold)
    (historical, cfg, run_config, identity, data, student, old_roots,
     old_resolved, teacher_cfg) = _load_fold(config, seed, fold, roots, device, include_test=False)
    fingerprint = _fingerprint(config_path, config, seed, fold, cfg, identity,
                               student, old_resolved, teacher_cfg)
    complete_path = roots['run_result'] / 'fold_complete.json'
    if complete_path.exists():
        complete = _read_json(complete_path)
        if complete.get('status') != 'trained' or complete.get('run_fingerprint') != fingerprint:
            raise RuntimeError(f'fold {fold + 1}: existing completion belongs to another identity')
        return {'status': 'trained', 'fold': fold, 'resumed_existing': True,
                'run_fingerprint': fingerprint}
    resolved_path = roots['run_result'] / 'resolved_run.json'
    if resolved_path.exists():
        saved = _read_json(resolved_path)
        if saved.get('run_fingerprint') != fingerprint:
            raise RuntimeError(f'fold {fold + 1}: resolved run identity changed')
    else:
        _atomic_json(resolved_path, {
            'status': 'training', 'experiment_id': config['experiment_id'],
            'run_fingerprint': fingerprint, 'config': str(config_path),
            'config_sha256': _sha(config_path), 'seed': seed, 'fold_zero_based': fold,
            'target_subject_id': int(identity['manifest']['target_subject_id']),
            'feedback_subject_id': int(identity['manifest']['feedback_subject_id']),
            'train_subject_ids': identity['manifest']['train_subject_ids'],
            'split_manifest': identity['path'], 'split_manifest_sha256': identity['sha256'],
            'train_uid_sha256': identity['uid_sha256']['train'],
            'feedback_uid_sha256': identity['uid_sha256']['feedback'],
            'historical_resolved_config_sha256': cfg['resolved_config_sha256'],
            'historical_teacher_resolved_run': str(old_roots['run_result'] / 'resolved_run.json'),
            'historical_teacher_resolved_run_sha256': _sha(old_roots['run_result'] / 'resolved_run.json'),
            'cbramod_config_from_historical_fold': teacher_cfg,
            'ifnet_config': student['student_cfg'],
            'student_schedule_sha256': student['student_schedule_sha256'],
            'initialization_audit': student['initialization_audit'],
            'source_manifest_sha256': identity['source_manifest_sha256'],
            'source_trial_manifest_sha256': identity['source_trial_manifest_sha256'],
            'teacher_rebuild': 'from canonical pretrained checkpoint using exact archived per-fold config',
            'target_evaluation': 'deferred until all three seeds finish training'})

    teacher_adapter, t0, t0_manifest = channel_run._fit_source_teacher(
        run_config, identity, roots, teacher_cfg, data, fingerprint, device, smoke=smoke)
    teachers = {'T0': t0}
    teacher_manifests = {'T0': t0_manifest}
    for name in ('T_CE', 'T_MCC'):
        model, manifest = channel_run._adapt_teacher(
            name, run_config, identity, roots, teacher_cfg, teacher_adapter,
            t0, data, fingerprint, device, smoke=smoke)
        teachers[name] = model
        teacher_manifests[name] = manifest

    expected_old = {}
    for name in ('T0', 'T_CE', 'T_MCC'):
        expected_old[name] = _read_json(old_roots['run_result'] / f'teacher_{name}.json')['model_sha256']
    rebuild_matches = {name: teacher_manifests[name].get('model_sha256') == expected_old[name]
                       for name in expected_old}
    _atomic_json(roots['run_result'] / 'teacher_rebuild_identity.json', {
        'seed': seed, 'fold': fold, 'historical_model_state_sha256': expected_old,
        'rebuilt_model_state_sha256': {name: teacher_manifests[name].get('model_sha256')
                                       for name in teacher_manifests},
        'exact_historical_state_match': rebuild_matches,
        'all_match': all(rebuild_matches.values()),
        'smoke': smoke,
        'interpretation': ('smoke weights are intentionally short and are not compared'
                           if smoke else 'compare tensor state hashes, not serialized checkpoint files')})

    projection64, projection32, projection_meta = _projection_record(
        roots, seed, fold, fingerprint)
    targets, target_meta = _teacher_targets(
        roots, identity, data, teachers, teacher_adapter, teacher_cfg,
        projection64, projection32, fingerprint, device, smoke)
    prior_initial, prior_initial_hash, prior_seed = _prior_initial(seed, fold, config)
    init_path = roots['run_weights'] / 'prior_common_initial.pt'
    _atomic_torch(init_path, {'model': prior_initial, 'model_sha256': prior_initial_hash,
                              'seed': prior_seed, 'run_fingerprint': fingerprint})
    priors, prior_manifests = {}, {}
    for name in ('T_CE', 'T_MCC'):
        prior, prior_manifest = _fit_prior(
            name, config, identity, roots, targets, fingerprint,
            prior_initial, prior_initial_hash, prior_seed, device, smoke)
        priors[name] = prior
        prior_manifests[name] = prior_manifest

    student_manifests = {}
    for condition in CONDITIONS:
        teacher = TEACHER_FOR[condition]
        student_manifests[condition] = _fit_student(
            condition, run_config, identity, roots, student, data,
            student['initial_student'], targets,
            priors[teacher] if IS_DIFFUSION[condition] else None,
            fingerprint, device, smoke)

    if smoke:
        gradient_checks = {}
        for condition in CONDITIONS:
            manifest = _read_json(roots['run_result'] / f'student_{condition}.json')
            history = pd.read_csv(resolve_local_file(
                roots['run_result'] / f'{condition}_history.csv'))
            first = history.iloc[0]
            reached = (float(first['native_feature_grad_norm']) > 0.0
                       and bool(first['student_parameter_gradient_reached']))
            if not reached:
                raise RuntimeError(f'{condition} smoke did not confirm feature gradients reached IFNet')
            gradient_checks[condition] = reached
        smoke_record = {
            'status': 'passed', 'experiment_id': config['experiment_id'],
            'run_fingerprint': fingerprint, 'seed': seed, 'fold': fold,
            'target_subject_id': fold + 1, 'target_evaluated': False,
            'full_feature_native_shape': [3, 4, 200],
            'full_feature_flat_dimension': 2400,
            'projected_shape': list(targets['T_CE_train_features_normalized'].shape[1:]),
            'projection_sha256': projection_meta['sha256'],
            'teacher_target_sha256': target_meta['file_sha256'],
            'teacher_rebuild_hash_matches': rebuild_matches,
            'prior_updates': {k: v['final_update'] for k, v in prior_manifests.items()},
            'student_epochs': {k: v['final_epoch'] for k, v in student_manifests.items()},
            'feature_gradients_reached_IFNet': gradient_checks,
            'feature_loss_manifest': {k: _read_json(
                roots['run_result'] / f'student_{k}.json')['loss'] for k in CONDITIONS}}
        _atomic_json(roots['run_result'] / 'smoke_result.json', smoke_record)
        return {'status': 'smoke_passed', 'fold': fold, 'run_fingerprint': fingerprint}

    completion = {
        'status': 'trained', 'experiment_id': config['experiment_id'],
        'run_fingerprint': fingerprint, 'seed': seed, 'fold_zero_based': fold,
        'target_subject_id': int(identity['manifest']['target_subject_id']),
        'train_uid_sha256': identity['uid_sha256']['train'],
        'teacher_checkpoints': {k: v['checkpoint_sha256'] for k, v in teacher_manifests.items()},
        'teacher_state_hashes_match_historical': rebuild_matches,
        'teacher_targets_sha256': target_meta['file_sha256'],
        'projection_sha256': projection_meta['sha256'],
        'prior_checkpoints': {k: v['checkpoint_sha256'] for k, v in prior_manifests.items()},
        'student_checkpoints': {k: v['checkpoint_sha256'] for k, v in student_manifests.items()},
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'student_schedule_sha256': student['student_schedule_sha256'],
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    _atomic_json(complete_path, completion)
    current = _read_json(resolved_path)
    current.update(status='trained', fold_complete_sha256=_sha(complete_path))
    _atomic_json(resolved_path, current)
    return {'status': 'trained', 'fold': fold, 'run_fingerprint': fingerprint,
            'teacher_hashes_match': rebuild_matches}


def _run_seed(config_path: Path, config: dict, seed: int, run_id: str, gpu: int,
              folds: list[int] | None, smoke: bool, progress_path: Path | None,
              stop_file: Path | None) -> int:
    device = _device(gpu)
    _validate(config, seed)
    selected = [0] if smoke else list(range(9)) if folds is None else list(folds)
    if any(f < 0 or f >= 9 for f in selected) or len(set(selected)) != len(selected):
        raise ValueError('folds must be unique zero-based values in [0,8]')
    records = []
    for fold in selected:
        if stop_file is not None and Path(stop_file).exists():
            _write_progress(progress_path, {'status': 'paused', 'seed': seed,
                'run_id': run_id, 'records': records, 'next_fold': fold})
            return 75
        _write_progress(progress_path, {'status': 'running', 'seed': seed, 'run_id': run_id,
            'records': records, 'active_fold': fold + 1})
        record = _train_fold(config_path, config, seed, run_id, fold, device, smoke)
        records.append({**record, 'target_subject': fold + 1})
        _write_progress(progress_path, {'status': 'running', 'seed': seed, 'run_id': run_id,
                                        'records': records})
    _write_progress(progress_path, {'status': 'completed', 'seed': seed, 'run_id': run_id,
        'records': records, 'evaluation': 'deferred until all three seed jobs complete'})
    return 0


def _load_teacher_models(roots: dict, teacher_cfg: dict, device: torch.device):
    t0_manifest = _read_json(roots['run_result'] / 'teacher_T0.json')
    adapter, initialized = channel_run._build_teacher(
        teacher_cfg, device, int(t0_manifest['initialization_seed']))
    models = {}
    for name in ('T0', 'T_CE', 'T_MCC'):
        manifest = _read_json(roots['run_result'] / f'teacher_{name}.json')
        checkpoint = Path(manifest['checkpoint'])
        if not checkpoint.is_file() or manifest.get('checkpoint_sha256') != _sha(checkpoint):
            raise RuntimeError(f'{name} teacher checkpoint is missing or corrupted')
        model = copy.deepcopy(initialized)
        payload = torch.load(resolve_local_file(checkpoint), map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        models[name] = model
    return adapter, models


def _load_student_checkpoint(models: dict, initial: dict, seed: int,
                             checkpoint: Path, device: torch.device):
    model, _, _, _ = reference._new_paired_student(models, initial, seed, device)
    payload = torch.load(resolve_local_file(checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return model


def _evaluate_fold(config: dict, seed: int, run_id: str, fold: int,
                   device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    roots = _roots(run_id, seed, fold)
    complete_path = roots['run_result'] / 'fold_complete.json'
    if not complete_path.is_file():
        raise RuntimeError(f'cannot evaluate incomplete fold seed={seed}, subject={fold + 1}')
    complete = _read_json(complete_path)
    if complete.get('status') != 'trained':
        raise RuntimeError(f'fold not trained: seed={seed}, subject={fold + 1}')
    metrics_path = roots['run_result'] / 'fold_evaluation_metrics.csv'
    predictions_path = roots['run_result'] / 'fold_predictions.csv'
    projection_path = roots['run_result'] / 'test_projection_diagnostics.csv'
    eval_path = roots['run_result'] / 'fold_evaluation.json'
    if all(path.is_file() for path in (metrics_path, predictions_path, projection_path, eval_path)):
        saved = _read_json(eval_path)
        if (saved.get('status') != 'complete'
                or saved.get('run_fingerprint') != complete['run_fingerprint']
                or saved.get('metrics_sha256') != _sha(metrics_path)
                or saved.get('predictions_sha256') != _sha(predictions_path)
                or saved.get('projection_diagnostics_sha256') != _sha(projection_path)):
            raise RuntimeError('saved target evaluation identity/hash mismatch')
        train_feedback_path = roots['run_result'] / 'teacher_projection_diagnostics.csv'
        if not train_feedback_path.is_file():
            raise RuntimeError('train/feedback teacher projection diagnostics are missing')
        train_feedback = pd.read_csv(resolve_local_file(train_feedback_path)).to_dict('records')
        for row in train_feedback:
            row.update(seed=seed, target_subject=fold + 1)
        test_diag = pd.read_csv(resolve_local_file(projection_path)).to_dict('records')
        return (pd.read_csv(resolve_local_file(metrics_path)).to_dict('records'),
                pd.read_csv(resolve_local_file(predictions_path)).to_dict('records'),
                train_feedback + test_diag)
    if any(path.exists() for path in (metrics_path, predictions_path, projection_path, eval_path)):
        raise RuntimeError(f'incomplete saved target evaluation for seed={seed}, subject={fold + 1}')

    (historical, cfg, run_config, identity, data, student, old_roots,
     old_resolved, teacher_cfg) = _load_fold(config, seed, fold, roots, device, include_test=True)
    if identity['uid_sha256']['test'] != complete.get('test_uid_sha256', identity['uid_sha256']['test']):
        raise RuntimeError('target UID identity changed after training')
    target_rows_teacher = data['teacher_rows']['test']
    target_rows_student = data['student_rows']['test']
    x_teacher_test = np.asarray(data['teacher_input_mmap'][target_rows_teacher], dtype=np.float32)
    x_student_raw = np.asarray(data['student_input_mmap'][target_rows_student], dtype=np.float32)
    x_student_test = reference._preprocess_student(data['student_adapter'], x_student_raw)
    y_test = np.asarray(identity['labels']['test'], dtype=np.int64)
    uids = np.asarray(identity['uids']['test'])
    if len(x_teacher_test) != len(y_test) or len(x_student_test) != len(y_test):
        raise RuntimeError('target data, labels and UID counts differ')

    adapter, teachers = _load_teacher_models(roots, teacher_cfg, device)
    projection_path = roots['run_cache'] / 'fixed_teacher_to_IFNet_P.npz'
    projection_meta = _read_json(roots['run_cache'] / 'fixed_teacher_to_IFNet_P.json')
    if projection_meta.get('sha256') != _sha(projection_path):
        raise RuntimeError('fixed teacher projection changed before target evaluation')
    with np.load(resolve_local_file(projection_path), allow_pickle=False) as saved:
        p64 = np.asarray(saved['P_float64'], dtype=np.float64)
        p32 = np.asarray(saved['P_float32'], dtype=np.float32)

    metric_rows, prediction_rows, projection_rows = [], [], []

    def append_model(name: str, logits: np.ndarray, group: str, teacher_source=None):
        values = channel_run._metric_values(y_test, logits)
        metric_rows.append({'model': name, 'group': group, 'seed': seed,
            'target_subject': fold + 1, 'n_trials': len(y_test), **values,
            'teacher_source': teacher_source})
        for index, (uid, label, row_logits) in enumerate(zip(uids, y_test, logits)):
            prediction_rows.append({'model': name, 'seed': seed,
                'target_subject': fold + 1,
                'trial_uid': f'{int(uid[0])}:{int(uid[1])}',
                'trial_order': index, 'label': int(label),
                'logit_0': float(row_logits[0]), 'logit_1': float(row_logits[1]),
                'prediction': int(np.argmax(row_logits))})

    for name in ('T0', 'T_CE', 'T_MCC'):
        features, logits = _extract_full_teacher(
            teachers[name], adapter, x_teacher_test, device, check_forward=(name == 'T0'))
        append_model(name, logits, 'teacher', None)
        weight = teachers[name].classifier.weight.detach().float().cpu().numpy()
        bias = teachers[name].classifier.bias.detach().float().cpu().numpy()
        diag = projection_core.projection_readout_diagnostic(
            features, logits, weight, bias, p64, y_test)
        projected_logits = diag.pop('projected_logits')
        projection_rows.append({'teacher': name, 'partition': 'test', 'seed': seed,
            'target_subject': fold + 1, **diag,
            **{f'projected_readout_{key}': value for key, value in
               channel_run._metric_values(y_test, projected_logits).items()}})

    baseline_model = _load_student_checkpoint(
        student, student['initial_student'], seed,
        Path(student['baseline_checkpoint']), device)
    baseline_logits, baseline_state_hash = channel_run._predict_student_preserving_state(
        baseline_model, student['student_adapter'], x_student_test, device)
    if baseline_state_hash != student['baseline_manifest']['final_model_sha256']:
        raise RuntimeError('IFNet CE reference changed while generating predictions')
    append_model('IFNet_CE', baseline_logits, 'student_baseline', None)

    for condition in CONDITIONS:
        manifest = _read_json(roots['run_result'] / f'{condition}.json')
        checkpoint = Path(manifest['checkpoint'])
        if (not checkpoint.is_file() or manifest.get('checkpoint_sha256') != _sha(checkpoint)
                or manifest.get('run_fingerprint') != complete['run_fingerprint']):
            raise RuntimeError(f'{condition} checkpoint is missing or does not match this fold')
        model = _load_student_checkpoint(student, student['initial_student'], seed,
                                         checkpoint, device)
        logits, actual_state_hash = channel_run._predict_student_preserving_state(
            model, student['student_adapter'], x_student_test, device)
        if actual_state_hash != manifest['model_sha256']:
            raise RuntimeError(f'{condition} IFNet state changed during target evaluation')
        append_model(condition, logits, 'student_method', TEACHER_FOR[condition])

    _atomic_csv(metrics_path, metric_rows)
    _atomic_csv(predictions_path, prediction_rows)
    _atomic_csv(projection_path, projection_rows)
    eval_record = {'status': 'complete', 'run_fingerprint': complete['run_fingerprint'],
        'seed': seed, 'fold_zero_based': fold, 'target_subject': fold + 1,
        'test_uid_sha256': identity['uid_sha256']['test'],
        'n_trials': len(y_test), 'models': [row['model'] for row in metric_rows],
        'metrics_sha256': _sha(metrics_path), 'predictions_sha256': _sha(predictions_path),
        'projection_diagnostics_sha256': _sha(projection_path),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    _atomic_json(eval_path, eval_record)
    train_feedback_path = roots['run_result'] / 'teacher_projection_diagnostics.csv'
    train_feedback = pd.read_csv(resolve_local_file(train_feedback_path)).to_dict('records')
    for row in train_feedback:
        row.update(seed=seed, target_subject=fold + 1)
    return metric_rows, prediction_rows, train_feedback + projection_rows


def _summary_rows(metrics: list[dict]) -> list[dict]:
    frame = pd.DataFrame(metrics)
    rows = []
    for model, part in frame.groupby('model', sort=False):
        seed_rows = []
        for seed, seed_part in part.groupby('seed'):
            if seed_part['target_subject'].nunique() != 9:
                raise RuntimeError(f'{model}: expected 9 target subjects for seed {seed}')
            seed_rows.append({'seed': int(seed), **{
                metric: float(seed_part[metric].astype(float).mean())
                for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')}})
        values = {}
        for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy'):
            metric_seed = np.asarray([row[metric] for row in seed_rows], dtype=np.float64)
            values[f'{metric}_mean'] = float(metric_seed.mean())
            values[f'{metric}_seed_sd'] = float(metric_seed.std(ddof=1)) if len(metric_seed) > 1 else 0.0
        rows.append({'model': model, 'seed_count': len(seed_rows), 'folds_per_seed': 9,
            'fold_records': int(len(part)), **values,
            'seed_means': {str(row['seed']): {m: row[m] for m in
                ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')}
                for row in seed_rows}})
    return rows


def _paired_summary(metrics: list[dict], seed: int = 20261101,
                    replicates: int = 10000) -> list[dict]:
    frame = pd.DataFrame(metrics)
    comparisons = [
        ('MCC_T2S_FEATURE', 'CE_T2S_FEATURE', 'MCC_T2S_FEATURE - CE_T2S_FEATURE'),
        ('MCC_T2S_DIFFKD', 'CE_T2S_DIFFKD', 'MCC_T2S_DIFFKD - CE_T2S_DIFFKD'),
        ('CE_T2S_DIFFKD', 'CE_T2S_FEATURE', 'CE_T2S_DIFFKD - CE_T2S_FEATURE'),
        ('MCC_T2S_DIFFKD', 'MCC_T2S_FEATURE', 'MCC_T2S_DIFFKD - MCC_T2S_FEATURE'),
        *((name, 'IFNet_CE', f'{name} - IFNet_CE') for name in CONDITIONS),
    ]
    rng = np.random.default_rng(int(seed))
    bootstrap_indices = rng.integers(0, 9, size=(int(replicates), 9))
    rows = []
    for left, right, label in comparisons:
        joined = frame[frame['model'].isin([left, right])].pivot(
            index=['seed', 'target_subject'], columns='model', values=['accuracy', 'balanced_accuracy'])
        for metric in ('accuracy', 'balanced_accuracy'):
            delta = (joined[(metric, left)] - joined[(metric, right)]).unstack('seed')
            if delta.shape != (9, 3) or delta.isna().any().any():
                raise RuntimeError(f'incomplete paired difference: {label}, {metric}')
            subject_delta = delta.mean(axis=1).to_numpy(dtype=np.float64) * 100.0
            bootstrap = subject_delta[bootstrap_indices].mean(axis=1)
            rows.append({'comparison': label, 'left_model': left, 'right_model': right,
                'metric': metric, 'seed_count': 3, 'folds_per_seed': 9,
                'mean_paired_difference_pp': float(subject_delta.mean()),
                'bootstrap_95ci_low_pp': float(np.quantile(bootstrap, 0.025)),
                'bootstrap_95ci_high_pp': float(np.quantile(bootstrap, 0.975)),
                'win': int((subject_delta > 1e-6).sum()),
                'tie': int((np.abs(subject_delta) <= 1e-6).sum()),
                'loss': int((subject_delta < -1e-6).sum()),
                'per_subject_mean_difference_pp': {
                    str(i + 1): float(subject_delta[i]) for i in range(9)},
                'per_seed_mean_difference_pp': {
                    str(s): float((delta[s].mean()) * 100.0) for s in delta.columns}})
    return rows


def _write_report(config: dict, metrics: list[dict], predictions: list[dict],
                  projection_rows: list[dict], rebuild_rows: list[dict],
                  run_ids: list[str]) -> dict:
    report_dir = require_report_output(REPORTS_ROOT / config['storage']['report_directory'])
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(metrics)
    paired = _paired_summary(metrics, int(config['report']['bootstrap_seed']),
                             int(config['report']['bootstrap_replicates']))
    summary_frame = pd.DataFrame([{k: v for k, v in row.items() if k != 'seed_means'} for row in summary])
    paired_frame = pd.DataFrame([{k: v for k, v in row.items()
                                  if k not in ('per_subject_mean_difference_pp',
                                               'per_seed_mean_difference_pp')}
                                 for row in paired])
    projection_frame = pd.DataFrame(projection_rows)
    rebuild_frame = pd.DataFrame(rebuild_rows)
    prediction_frame = pd.DataFrame(predictions)
    for filename, frame in (('metric_summary.csv', summary_frame),
                            ('paired_differences.csv', paired_frame),
                            ('projection_diagnostics.csv', projection_frame),
                            ('teacher_rebuild_identity.csv', rebuild_frame),
                            ('trial_predictions.csv', prediction_frame)):
        target = require_report_output(report_dir / filename)
        temp = target.with_suffix(target.suffix + '.tmp')
        frame.to_csv(temp, index=False)
        os.replace(temp, target)

    summary_by_model = {row['model']: row for row in summary}
    table_models = [
        ('CBraMod', 'source T0', 'T0', None),
        ('CBraMod', 'CE adapted', 'T_CE', None),
        ('CBraMod', 'MCC adapted', 'T_MCC', None),
        ('IFNet', 'none', 'IFNet_CE', None),
        ('IFNet', 'CE_T2S_FEATURE', 'CE_T2S_FEATURE', 'T_CE'),
        ('IFNet', 'MCC_T2S_FEATURE', 'MCC_T2S_FEATURE', 'T_MCC'),
        ('IFNet', 'CE_T2S_DIFFKD', 'CE_T2S_DIFFKD', 'T_CE'),
        ('IFNet', 'MCC_T2S_DIFFKD', 'MCC_T2S_DIFFKD', 'T_MCC'),
    ]
    table_rows = []
    for student_name, method, model_name, teacher_name in table_models:
        stat = summary_by_model[model_name]
        row = {'student': student_name, 'method': method, 'model': model_name,
            'seeds': stat['seed_count'], 'folds_per_seed': stat['folds_per_seed'],
            'Accuracy (%)': stat['accuracy_mean'] * 100,
            'Accuracy seed SD (pp)': stat['accuracy_seed_sd'] * 100,
            'Balanced Accuracy (%)': stat['balanced_accuracy_mean'] * 100,
            'Balanced Accuracy seed SD (pp)': stat['balanced_accuracy_seed_sd'] * 100,
            'matched_teacher': teacher_name}
        table_rows.append(row)
    table_frame = pd.DataFrame(table_rows)
    table_path = require_report_output(report_dir / 'summary.xlsx')
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'LOSO 汇总'
        sheet.sheet_view.showGridLines = False
        headers = ['student', 'method', 'Accuracy (%)', 'Accuracy seed SD (pp)',
                   'Balanced Accuracy (%)', 'BA seed SD (pp)', 'seeds', 'folds/seed']
        sheet.append(headers)
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        for row in table_rows:
            sheet.append([row['student'], row['method'], row['Accuracy (%)'],
                          row['Accuracy seed SD (pp)'], row['Balanced Accuracy (%)'],
                          row['Balanced Accuracy seed SD (pp)'], row['seeds'],
                          row['folds_per_seed']])
        teacher_fill = PatternFill('solid', fgColor='FAC090')
        none_fill = PatternFill('solid', fgColor='93CDDD')
        red_fill = PatternFill('solid', fgColor='FFC7CE')
        green_fill = PatternFill('solid', fgColor='C6EFCE')
        yellow_fill = PatternFill('solid', fgColor='FFF2CC')
        for row_index, row in enumerate(table_rows, start=2):
            for column in range(1, len(headers) + 1):
                sheet.cell(row_index, column).alignment = Alignment(horizontal='center', vertical='center')
            if row['student'] == 'CBraMod':
                for column in range(1, len(headers) + 1):
                    sheet.cell(row_index, column).fill = teacher_fill
            elif row['method'] == 'none':
                for column in range(1, len(headers) + 1):
                    sheet.cell(row_index, column).fill = none_fill
            else:
                teacher_stat = summary_by_model[row['matched_teacher']]
                baseline_stat = summary_by_model['IFNet_CE']
                for column, metric in ((3, 'accuracy'), (5, 'balanced_accuracy')):
                    value = row['Accuracy (%)'] if metric == 'accuracy' else row['Balanced Accuracy (%)']
                    above_teacher = value > teacher_stat[f'{metric}_mean'] * 100
                    above_student = value > baseline_stat[f'{metric}_mean'] * 100
                    if above_teacher and above_student:
                        sheet.cell(row_index, column).fill = red_fill
                    elif above_teacher:
                        sheet.cell(row_index, column).fill = green_fill
                    elif above_student:
                        sheet.cell(row_index, column).fill = yellow_fill
            for column in (3, 4, 5, 6):
                sheet.cell(row_index, column).number_format = '0.00'
        sheet.freeze_panes = 'C2'
        for idx, width in enumerate((18, 28, 18, 20, 26, 18, 12, 14), 1):
            sheet.column_dimensions[get_column_letter(idx)].width = width
        legend_row = len(table_rows) + 3
        legend = [
            ('大于小模型', 'FFF2CC'), ('大于大模型', 'C6EFCE'), ('大于二者', 'FFC7CE'),
            ('教师参考', 'FAC090'), ('IFNet none 基线', '93CDDD')]
        for offset, (label, color) in enumerate(legend):
            cell = sheet.cell(legend_row + offset, 1, label)
            cell.fill = PatternFill('solid', fgColor=color)
            cell.font = Font(bold=True)
        notes = [
            '指标：Accuracy 与 Balanced Accuracy；百分比按 0–100 展示，保留两位小数。',
            '聚合：每个 seed 内对 9 个 LOSO 目标被试等权平均，再对 seed 666/667/668 等权平均；seed SD 基于三个 seed 均值。',
            '方法：完整 CBraMod [3,4,200] reshape 为 2400 维，经固定 512×2400 正交投影和无仿射 LayerNorm；IFNet 原生 512 维特征同样 LayerNorm。',
            'FEATURE = CE + 0.5 配对特征 MSE；DIFFKD = CE + 0.5 五步 DDIM 后配对特征 MSE；教师 logits KD 未使用。',
            'T_CE 与 T_MCC 均从同折同一个 T0 继续适配 10 epoch；T0 按历史每折 resolved 配置重建，并逐模型状态 hash 核验。',
            '配对差以 subject 为统计单位；bootstrap 95% CI 对 9 个 subject 采样，固定 10000 次；仅作探索性估计。',
        ]
        note_row = legend_row + len(legend) + 1
        for i, note in enumerate(notes):
            sheet.merge_cells(start_row=note_row + i, start_column=1,
                              end_row=note_row + i, end_column=len(headers))
            cell = sheet.cell(note_row + i, 1, note)
            cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
            sheet.row_dimensions[note_row + i].height = 30
        workbook.save(table_path)
    except ImportError:
        table_path = None

    aggregate = {'status': 'complete', 'experiment_id': config['experiment_id'],
        'dataset': config['dataset'], 'session': config['selected_session'],
        'seeds': list(SEEDS), 'subjects_per_seed': 9,
        'aggregation': config['report']['aggregate'],
        'target_evaluation': 'after all 27 formal training folds completed',
        'seed_mean_summary': summary, 'paired_differences': paired,
        'run_ids': run_ids,
        'teacher_rebuild_identity': rebuild_rows,
        'teacher_rebuild_exact_match_counts': {name: sum(
            bool(row['exact_historical_state_match']) for row in rebuild_rows
            if row['teacher'] == name) for name in ('T0', 'T_CE', 'T_MCC')},
        'rebuild_identity_note': 'state hashes compared against historical teacher manifests; mismatches are explicit',
        'files': {'metric_summary': str(report_dir / 'metric_summary.csv'),
            'paired_differences': str(report_dir / 'paired_differences.csv'),
            'projection_diagnostics': str(report_dir / 'projection_diagnostics.csv'),
            'teacher_rebuild_identity': str(report_dir / 'teacher_rebuild_identity.csv'),
            'trial_predictions': str(report_dir / 'trial_predictions.csv'),
            'summary_workbook': str(table_path) if table_path else None}}
    aggregate_path = require_report_output(report_dir / 'aggregate.json')
    _atomic_json(aggregate_path, aggregate)
    _write_conclusions(report_dir / 'conclusions.md', summary, paired, rebuild_rows)
    return aggregate


def _write_conclusions(path: Path, summary: list[dict], paired: list[dict],
                       rebuild_rows: list[dict]):
    stats = {row['model']: row for row in summary}
    by_cmp = {(row['comparison'], row['metric']): row for row in paired}
    method_order = ['CE_T2S_FEATURE', 'MCC_T2S_FEATURE', 'CE_T2S_DIFFKD', 'MCC_T2S_DIFFKD']
    exact_counts = {name: sum(bool(row['exact_historical_state_match'])
        for row in rebuild_rows if row['teacher'] == name) for name in ('T0', 'T_CE', 'T_MCC')}
    lines = [
        '# CBraMod → IFNet full-feature projection LOSO V1',
        'Seeds 666/667/668; 9 target subjects per seed; 27 seed-fold records per model.',
        f"IFNet CE: Acc {stats['IFNet_CE']['accuracy_mean']*100:.2f}% ± {stats['IFNet_CE']['accuracy_seed_sd']*100:.2f} pp; BA {stats['IFNet_CE']['balanced_accuracy_mean']*100:.2f}% ± {stats['IFNet_CE']['balanced_accuracy_seed_sd']*100:.2f} pp (SD across three seed means).",
    ]
    for model in method_order:
        lines.append(f"{model}: Acc {stats[model]['accuracy_mean']*100:.2f}% ± {stats[model]['accuracy_seed_sd']*100:.2f} pp; BA {stats[model]['balanced_accuracy_mean']*100:.2f}% ± {stats[model]['balanced_accuracy_seed_sd']*100:.2f} pp.")
    feature = by_cmp[('MCC_T2S_FEATURE - CE_T2S_FEATURE', 'balanced_accuracy')]
    diffkd = by_cmp[('MCC_T2S_DIFFKD - CE_T2S_DIFFKD', 'balanced_accuracy')]
    lines.append(f"MCC−CE BA difference: FEATURE {feature['mean_paired_difference_pp']:+.2f} pp (95% CI {feature['bootstrap_95ci_low_pp']:+.2f} to {feature['bootstrap_95ci_high_pp']:+.2f}); DIFFKD {diffkd['mean_paired_difference_pp']:+.2f} pp ({diffkd['bootstrap_95ci_low_pp']:+.2f} to {diffkd['bootstrap_95ci_high_pp']:+.2f}).")
    lines.append(f"Exact historical teacher state hashes: T0 {exact_counts['T0']}/27, T_CE {exact_counts['T_CE']}/27, T_MCC {exact_counts['T_MCC']}/27.")
    lines.append('DiffKD−FEATURE paired differences are in paired_differences.csv; bootstrap samples subjects, not 27 seed-folds.')
    Path(path).write_text('\n'.join(lines[:10]) + '\n', encoding='utf-8')


def _evaluate_all(config: dict, device: torch.device) -> dict:
    metrics, predictions, projection_rows, rebuild_rows, run_ids = [], [], [], [], []
    for seed in SEEDS:
        run_id = f"{config['storage']['external_run_id_prefix']}{seed}"
        run_ids.append(run_id)
        for fold in range(9):
            rows, preds, diag = _evaluate_fold(config, seed, run_id, fold, device)
            metrics.extend(rows); predictions.extend(preds); projection_rows.extend(diag)
            roots = _roots(run_id, seed, fold)
            audit = _read_json(roots['run_result'] / 'teacher_rebuild_identity.json')
            if audit.get('smoke'):
                raise RuntimeError('smoke teacher weights cannot enter the formal aggregate')
            for name in ('T0', 'T_CE', 'T_MCC'):
                rebuild_rows.append({'seed': seed, 'target_subject': fold + 1,
                    'teacher': name,
                    'historical_model_state_sha256': audit['historical_model_state_sha256'][name],
                    'rebuilt_model_state_sha256': audit['rebuilt_model_state_sha256'][name],
                    'exact_historical_state_match': bool(audit['exact_historical_state_match'][name])})
    expected_models = {'T0', 'T_CE', 'T_MCC', 'IFNet_CE', *CONDITIONS}
    grouped = {}
    for row in metrics:
        grouped.setdefault(row['model'], set()).add((int(row['seed']), int(row['target_subject'])))
    expected_pairs = {(seed, subject) for seed in SEEDS for subject in range(1, 10)}
    if set(grouped) != expected_models or any(value != expected_pairs for value in grouped.values()):
        raise RuntimeError('final evaluation is missing a model or one of its 27 seed-fold records')
    return _write_report(config, metrics, predictions, projection_rows, rebuild_rows, run_ids)


def _queue_root(queue_root: Path | None) -> Path:
    if queue_root is not None:
        return Path(queue_root).resolve()
    return ROOT / 'results/loso/distill/queue'


def enqueue_run(config_path: Path, queue_root: Path | None = None,
                smoke: bool = False, smoke_id: str | None = None) -> list[str]:
    from experiments.distill.queue import DEFAULT_ROOT, enqueue_snapshot
    config_path = Path(config_path).resolve()
    config = _read_config(config_path)
    selected_seeds = [666] if smoke else list(SEEDS)
    for seed in selected_seeds:
        _validate(config, seed)
    root = Path(queue_root or DEFAULT_ROOT).resolve()
    jobs = []
    for seed in selected_seeds:
        run_id = (f"cbramod_t2s_full_feature_loso_v1_smoke_{smoke_id or int(time.time())}_seed{seed}"
                  if smoke else f"{config['storage']['external_run_id_prefix']}{seed}")
        snapshot = {'kind': 'cbramod_t2s_full_feature_loso_v1',
            'stage': 'smoke' if smoke else 'train_seed',
            'config': str(config_path), 'config_sha256': _sha(config_path),
            'seed': seed, 'run_id': run_id, 'folds': [0] if smoke else list(range(9)),
            'args': {'dataset': 'BNCI2014004', 'teacher': 'cbramod', 'student': 'ifnet',
                'protocol': 'loso_7_1_1', 'keys': [0] if smoke else list(range(9)),
                'seeds': [seed], 'methods': list(CONDITIONS), 'source_variant': 'wide_fre_band',
                'selected_session': 'session_3', 'epochs': 100,
                'teacher_feature_shape': [3, 4, 200], 'feature_projection': [512, 2400]}}
        jobs.append(enqueue_snapshot(snapshot, root=root,
            label=(f'CBraMod full-feature T2S smoke seed={seed} target=01' if smoke else
                   f'CBraMod full-feature T2S seed={seed}, 9-fold LOSO')))
    if not smoke:
        report_dir = require_report_output(REPORTS_ROOT / config['storage']['report_directory'] / 'dispatch')
        report_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', '-m', 'experiments.distill.run_cbramod_t2s_full_feature_loso',
            '--watch-jobs', '--queue-root', str(root), '--job-ids', *jobs,
            '--config', str(config_path), '--gpu', '0']
        environment = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                     'BIGSMALLCOLLAB_CPU_THREADS', 'TORCH_NUM_THREADS',
                     'TORCH_NUM_INTEROP_THREADS'):
            environment[name] = '1'
        watcher_log = report_dir / 'watcher.log'
        with watcher_log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=environment, close_fds=True)
        _atomic_json(report_dir / 'dispatch.json', {
            'status': 'submitted', 'experiment_id': config['experiment_id'],
            'config': str(config_path), 'config_sha256': _sha(config_path),
            'seeds': selected_seeds, 'training_job_ids': jobs,
            'queue_root': str(root), 'watcher_pid': process.pid,
            'teacher_retraining': 'rebuild 27 T0 and 54 adapted teachers from historical resolved configs',
            'evaluation': 'target evaluation starts after all three seed jobs complete'})
    return jobs


def _watch_jobs(config: dict, queue_root: Path, job_ids: list[str], gpu: int) -> int:
    from experiments.distill.queue import QueueStore, validated_root
    store = QueueStore(validated_root(queue_root))
    report_dir = require_report_output(REPORTS_ROOT / config['storage']['report_directory'] / 'dispatch')
    while True:
        states = [store.load(job_id) for job_id in job_ids]
        failed = [s for s in states if s['status'] in ('failed', 'cancelled')]
        if failed:
            _atomic_json(report_dir / 'watcher_state.json', {
                'status': 'failed', 'jobs': [{'id': s['id'], 'status': s['status'],
                    'message': s.get('message')} for s in failed]})
            return 1
        if all(s['status'] == 'completed' for s in states):
            if any(s.get('child') for s in states):
                time.sleep(20)
                continue
            device = _device(gpu)
            result = _evaluate_all(config, device)
            # At this point all training workers ended and predictions/provenance are self-contained.
            # Keep the teacher checkpoints and full feature caches for the planned projection audit.
            for seed in SEEDS:
                run_id = f"{config['storage']['external_run_id_prefix']}{seed}"
                for fold in range(9):
                    roots = _roots(run_id, seed, fold)
                    for condition in CONDITIONS:
                        student_path = roots['run_weights'] / f'student_{condition}.pt'
                        if student_path.exists():
                            student_path.unlink()
                    for teacher in ('T_CE', 'T_MCC'):
                        prior_path = roots['run_weights'] / f'prior_{teacher}.pt'
                        if prior_path.exists():
                            prior_path.unlink()
                    init_path = roots['run_weights'] / 'prior_common_initial.pt'
                    if init_path.exists():
                        init_path.unlink()
            _atomic_json(report_dir / 'watcher_state.json', {
                'status': 'complete', 'jobs': job_ids, 'aggregate_completion': result,
                'student_and_prior_checkpoints_pruned': True,
                'retained': 'teacher checkpoints, full feature targets, projection, metrics and provenance'})
            return 0
        time.sleep(30)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--folds', nargs='+', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--enqueue-smoke', action='store_true')
    parser.add_argument('--smoke-id', default=None)
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--queue-root', type=Path, default=None)
    parser.add_argument('--progress', type=Path, default=None)
    parser.add_argument('--stop-file', type=Path, default=None)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--watch-jobs', action='store_true')
    parser.add_argument('--job-ids', nargs='*', default=[])
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        if _sha(config_path) == '':
            raise RuntimeError('could not hash config')
        config = _read_config(config_path)
        if args.enqueue_smoke:
            print(json.dumps({'smoke_job_ids': enqueue_run(
                config_path, args.queue_root, smoke=True, smoke_id=args.smoke_id)}, indent=2), flush=True)
            return 0
        if args.enqueue:
            print(json.dumps({'training_job_ids': enqueue_run(
                config_path, args.queue_root, smoke=False)}, indent=2), flush=True)
            return 0
        if args.aggregate:
            result = _evaluate_all(config, _device(args.gpu))
            print(json.dumps({'status': result['status'], 'files': result['files']}, indent=2), flush=True)
            return 0
        if args.watch_jobs:
            if not args.job_ids:
                raise ValueError('--watch-jobs requires --job-ids')
            return _watch_jobs(config, args.queue_root or _queue_root(None), args.job_ids, args.gpu)
        _validate(config, args.seed)
        run_id = args.run_id or f"{config['storage']['external_run_id_prefix']}{args.seed}"
        return _run_seed(config_path, config, args.seed, run_id, args.gpu,
                         args.folds, args.smoke, args.progress, args.stop_file)
    except Exception as exc:
        traceback.print_exc()
        print(f'[cbramod-t2s-full-feature] {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
