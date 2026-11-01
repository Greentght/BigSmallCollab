"""Mean-pool CBraMod's three channel features, then align one IFNet vector."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS',
              'TORCH_NUM_THREADS', 'TORCH_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
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

from collab import mcc_diffkd as core
from collab.student_feature_prior import apply_pca_whitener, fit_pca_whitener
from experiments.distill import run_cbramod_mcc_diffkd_loso as channel_run
from experiments.distill import run_student_aware_diffusion_loso as reference
from experiments.storage import REPORTS_ROOT, RESULTS_ROOT, require_external_output, require_report_output, resolve_local_file
from models import get_adapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/experiments/loso/cbramod_meanpool_mcc_diffkd_loso_v1.yaml'
OLD_CONDITIONS = ('CE_FEATURE', 'MCC_FEATURE', 'CE_DIFFKD', 'MCC_DIFFKD')
CONDITIONS = tuple('MEANPOOL_' + name for name in OLD_CONDITIONS)
TEACHER_FOR = {'CE_FEATURE': 'T_CE', 'MCC_FEATURE': 'T_MCC',
               'CE_DIFFKD': 'T_CE', 'MCC_DIFFKD': 'T_MCC'}
IS_DIFFUSION = {'CE_FEATURE': False, 'MCC_FEATURE': False,
                'CE_DIFFKD': True, 'MCC_DIFFKD': True}
SEEDS = (666, 667, 668)
OLD_REPORT = REPORTS_ROOT / 'distill/cbramod_mcc_diffkd_loso_v1/trial_predictions_3seed.csv'


def _sha(path) -> str:
    h = hashlib.sha256()
    with Path(resolve_local_file(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), default=str).encode()
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        return json.load(stream)


def _atomic_json(path: Path, value) -> None:
    channel_run._atomic_json(Path(path), value)


def _atomic_npz(path: Path, **arrays) -> None:
    channel_run._atomic_npz(Path(path), **arrays)


def _atomic_torch(path: Path, value: dict) -> None:
    channel_run._atomic_torch(Path(path), value)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    channel_run._atomic_csv(Path(path), rows)


def _read_config(path: Path) -> dict:
    with Path(resolve_local_file(path)).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _validate(config: dict, seed: int) -> None:
    if seed not in SEEDS or seed not in config.get('seeds', []):
        raise ValueError(f'unsupported mean-pool seed {seed}')
    if (config.get('experiment_id') != 'CBRAMOD_MEANPOOL_MCC_DIFFKD_LOSO_V1'
            or config.get('dataset') != 'BNCI2014004'
            or config.get('conditions') != list(OLD_CONDITIONS)
            or config.get('feature_space', {}).get('channel_pooling') != 'arithmetic_mean_before_alignment'
            or config.get('protocol', {}).get('folds') != 9
            or config.get('diffusion', {}).get('conditioning') != 'none'
            or config.get('diffusion', {}).get('ddim_timesteps') != [250, 200, 150, 100, 50, 0]):
        raise ValueError('fixed mean-pool CBraMod configuration was changed')
    if config['reference_run']['resolved_config_sha256_by_seed'].get(seed) != channel_run.RESOLVED_HASHES[seed]:
        raise ValueError(f'seed {seed}: resolved IFNet reference identity mismatch')


def _roots(run_id: str, seed: int, fold: int) -> dict:
    roots = channel_run._roots(run_id, seed, fold)
    for key in ('run_result', 'run_cache', 'run_weights', 'run_seed_weights'):
        require_external_output(roots[key]).mkdir(parents=True, exist_ok=True)
    return roots


def _resolved(config: dict, seed: int):
    return channel_run._resolved(config, seed)


def _fold_context(config: dict, seed: int, fold: int, roots: dict,
                  device: torch.device, include_test: bool = False):
    historical, cfg = _resolved(config, seed)
    run_config = dict(config)
    run_config['seed'] = seed
    run_config['reference_run'] = {
        'protocol_id': config['reference_run']['protocol_id'], 'seed': seed,
        'resolved_config_sha256': cfg['resolved_config_sha256']}
    identity = reference._load_fold_identity(
        run_config, fold, roots, cfg['resolved_config_sha256'],
        include_test_labels=include_test)
    data = channel_run._load_data(identity, roots, historical['student_config'], device)
    student = channel_run._load_student_reference(
        run_config, seed, fold, identity, roots, historical, cfg, data, device)
    return historical, cfg, run_config, identity, data, student


def _old_artifacts(config: dict, seed: int, fold: int, roots: dict,
                   identity: dict) -> tuple[dict, dict]:
    old_id = config['teachers']['source_feature_cache_run_id_prefix'] + f'_seed{seed}'
    old_roots = channel_run._roots(old_id, seed, fold)
    target_path = old_roots['run_cache'] / 'teacher_channel_targets.npz'
    target_meta_path = old_roots['run_cache'] / 'teacher_channel_targets.json'
    complete_path = old_roots['run_result'] / 'fold_complete.json'
    if not all(p.is_file() for p in (target_path, target_meta_path, complete_path)):
        raise FileNotFoundError(f'missing reusable completed CBraMod artifacts for seed={seed}, fold={fold + 1}')
    metadata, complete = _read_json(target_meta_path), _read_json(complete_path)
    target_sha = _sha(target_path)
    if (metadata.get('file_sha256') != target_sha
            or complete.get('status') != 'trained'
            or int(complete.get('seed', -1)) != seed
            or int(complete.get('fold_zero_based', -1)) != fold
            or int(complete.get('target_subject_id', -1)) != fold + 1
            or complete.get('train_uid_sha256') != identity['uid_sha256']['train']
            or complete.get('teacher_targets_sha256') != target_sha
            or metadata.get('train_uid_sha256') != identity['uid_sha256']['train']
            or metadata.get('feedback_uid_sha256') != identity['uid_sha256']['feedback']
            or metadata.get('training_samples') != 'D_train only; feedback used for diagnostics; no test target exported'
            or metadata.get('feature_tap') != 'per-channel patch mean; no cross-channel pooling or flattening'):
        raise RuntimeError(f'cached CBraMod teacher-target provenance mismatch: {target_path}')
    for name in ('T0', 'T_CE', 'T_MCC'):
        if metadata.get('teacher_checkpoint_sha256', {}).get(name) != complete.get('teacher_checkpoints', {}).get(name):
            raise RuntimeError(f'{name} cached target provenance differs from completed fold record')
        manifest_path = old_roots['run_result'] / f'teacher_{name}.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'missing teacher provenance record: {manifest_path}')
        teacher_manifest = _read_json(manifest_path)
        if (teacher_manifest.get('status') != 'complete'
                or teacher_manifest.get('checkpoint_sha256') != complete['teacher_checkpoints'][name]
                or teacher_manifest.get('train_uid_sha256') != identity['uid_sha256']['train']):
            raise RuntimeError(f'{name} manifest disagrees with retained completed-fold provenance')
    with np.load(resolve_local_file(target_path), allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    if (not np.array_equal(arrays['train_uids'], identity['uids']['train'])
            or not np.array_equal(arrays['train_y'], identity['labels']['train'])
            or not np.array_equal(arrays['feedback_uids'], identity['uids']['feedback'])
            or not np.array_equal(arrays['feedback_y'], identity['labels']['feedback'])):
        raise RuntimeError('cached CBraMod feature targets have different UIDs/labels')
    for name in ('T0', 'T_CE', 'T_MCC'):
        for part in ('train', 'feedback'):
            native = arrays[f'{name}_{part}_features_native']
            if native.shape != (len(identity['uids'][part]), 3, 200) or not np.isfinite(native).all():
                raise RuntimeError(f'{name}/{part} native feature shape/value mismatch: {native.shape}')
    for name in ('T_CE', 'T_MCC'):
        if arrays[f'{name}_train_logits'].shape != (len(identity['uids']['train']), 2):
            raise RuntimeError(f'{name}: cached source logits have wrong shape')
    provenance = {'source_run_id': old_id, 'source_cache_path': str(target_path),
                  'source_targets_sha256': target_sha,
                  'source_fold_complete_sha256': _sha(complete_path),
                  'source_teacher_checkpoint_sha256': metadata['teacher_checkpoint_sha256']}
    return arrays, provenance


def _make_mean_targets(config: dict, seed: int, fold: int, roots: dict,
                       identity: dict, run_fingerprint: str) -> tuple[dict, dict, dict]:
    source, provenance = _old_artifacts(config, seed, fold, roots, identity)
    pca_path = roots['run_cache'] / 'fixed_meanpool_T0_pca.npz'
    target_path = roots['run_cache'] / 'teacher_meanpool_targets.npz'
    meta_path = roots['run_cache'] / 'teacher_meanpool_targets.json'
    if target_path.exists() or meta_path.exists() or pca_path.exists():
        if not all(p.exists() for p in (target_path, meta_path, pca_path)):
            raise RuntimeError('incomplete mean-pooled target artifacts')
        meta = _read_json(meta_path)
        if (meta.get('run_fingerprint') != run_fingerprint
                or meta.get('file_sha256') != _sha(target_path)
                or meta.get('source_targets_sha256') != provenance['source_targets_sha256']):
            raise RuntimeError('mean-pooled target cache identity/hash mismatch')
        with np.load(resolve_local_file(target_path), allow_pickle=False) as saved:
            targets = {key: np.asarray(saved[key]) for key in saved.files}
        with np.load(resolve_local_file(pca_path), allow_pickle=False) as saved:
            pca = {key: np.asarray(saved[key]) for key in saved.files}
        pca.update(meta['pca'])
        return targets, pca, meta

    # Crucial change: [trial, channel, feature] -> [trial, feature] BEFORE
    # fitting the shared teacher coordinate and before any student alignment.
    pooled = {f'{name}_{part}': source[f'{name}_{part}_features_native'].mean(axis=1).astype(np.float32)
              for name in ('T0', 'T_CE', 'T_MCC') for part in ('train', 'feedback')}
    fs = config['feature_space']
    pca = fit_pca_whitener(
        pooled['T0_train'], max_dimensions=int(fs['pca_max_dimensions']),
        rank_relative_threshold=float(fs['rank_relative_threshold']),
        eigenvalue_floor_relative=float(fs['whitening_eigenvalue_floor_relative']))
    if int(pca['dimension']) < 1:
        raise RuntimeError('mean-pooled T0 D_train feature PCA has zero effective rank')
    dim = int(pca['dimension'])
    arrays = {'train_uids': source['train_uids'], 'train_y': source['train_y'],
              'feedback_uids': source['feedback_uids'], 'feedback_y': source['feedback_y'],
              'T0_train_features_pooled': pooled['T0_train'],
              'T0_feedback_features_pooled': pooled['T0_feedback']}
    for name in ('T_CE', 'T_MCC'):
        for part in ('train', 'feedback'):
            arrays[f'{name}_{part}_features_pooled'] = pooled[f'{name}_{part}']
            arrays[f'{name}_{part}_features_whitened'] = apply_pca_whitener(
                pooled[f'{name}_{part}'], pca).astype(np.float32)
            arrays[f'{name}_{part}_logits'] = source[f'{name}_{part}_logits'].astype(np.float32)
    pca_arrays = {key: pca[key] for key in ('mean', 'components', 'eigenvalues', 'scales')}
    _atomic_npz(pca_path, **pca_arrays)
    _atomic_npz(target_path, **arrays)
    pca_summary = {key: (int(value) if key in ('effective_rank', 'dimension') else float(value))
                    for key, value in pca.items()
                    if key not in ('mean', 'components', 'eigenvalues', 'scales')}
    meta = {'status': 'complete', 'run_fingerprint': run_fingerprint,
            'file_sha256': _sha(target_path), 'pca_file_sha256': _sha(pca_path),
            'train_uid_sha256': identity['uid_sha256']['train'],
            'feedback_uid_sha256': identity['uid_sha256']['feedback'],
            'source_targets_sha256': provenance['source_targets_sha256'],
            'source_teacher_checkpoint_sha256': provenance['source_teacher_checkpoint_sha256'],
            'source_cache_path': provenance['source_cache_path'],
            'pooling': 'arithmetic mean over CBraMod channel axis (axis=1), native [N,3,200] to [N,200]',
            'pca_fit_population': 'T0 mean-pooled D_train only',
            'student_targets': 'mean-pool T_CE/T_MCC per trial, then apply the fixed T0-derived PCA whitener',
            'pca': pca_summary, 'feature_dimension': dim,
            'feedback_used_for_diagnostics_only': True, 'test_targets_exported': False}
    _atomic_json(meta_path, meta)
    return arrays, pca, meta


def _prior_initial(dimension: int, seed: int, config: dict) -> tuple[dict, str]:
    dcfg = config['diffusion']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = core.EpsilonPredictor(dimension, int(dcfg['hidden_dimensions']),
                                      int(dcfg['time_embedding_dimensions']))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, reference._state_hash(state)


def _fit_prior(name: str, config: dict, identity: dict, roots: dict,
               targets: dict, fingerprint: str, initial: dict, initial_hash: str,
               prior_seed: int, device: torch.device, smoke: bool = False):
    final_path = roots['run_weights'] / f'meanpool_prior_{name}.pt'
    manifest_path = roots['run_result'] / f'meanpool_prior_{name}.json'
    history_path = roots['run_result'] / f'meanpool_prior_{name}_history.csv'
    train_np = np.asarray(targets[f'{name}_train_features_whitened'], dtype=np.float32)
    feedback_np = np.asarray(targets[f'{name}_feedback_features_whitened'], dtype=np.float32)
    dimension = int(train_np.shape[1])
    updates = 4 if smoke else int(config['diffusion']['training_updates'])
    method_fp = _json_hash({'run_fingerprint': fingerprint, 'teacher': name,
                            'source_targets_sha256': _sha(roots['run_cache'] / 'teacher_meanpool_targets.npz'),
                            'initial_prior_sha256': initial_hash, 'updates': updates,
                            'conditioning': 'none'})
    if final_path.is_file() or manifest_path.is_file():
        if not (final_path.is_file() and manifest_path.is_file()):
            raise RuntimeError(f'{name} mean-pool prior checkpoint/manifest is incomplete')
        manifest = _read_json(manifest_path)
        if manifest.get('method_fingerprint') != method_fp or manifest.get('checkpoint_sha256') != _sha(final_path):
            raise RuntimeError(f'{name} mean-pool prior identity/hash mismatch')
        model = core.EpsilonPredictor(dimension, int(config['diffusion']['hidden_dimensions']),
                                      int(config['diffusion']['time_embedding_dimensions'])).to(device)
        payload = torch.load(resolve_local_file(final_path), map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False); p.grad = None
        return model, manifest
    dcfg = config['diffusion']
    model = core.EpsilonPredictor(dimension, int(dcfg['hidden_dimensions']),
                                  int(dcfg['time_embedding_dimensions'])).to(device)
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(dcfg['lr']),
        betas=tuple(dcfg['betas']), eps=float(dcfg['eps']), weight_decay=float(dcfg['weight_decay']))
    _, alpha, sigma = core.ddpm_schedule(int(dcfg['steps']), float(dcfg['beta_start']),
                                         float(dcfg['beta_end']), device)
    train = torch.as_tensor(train_np, dtype=torch.float32, device=device)
    batch_size = min(int(dcfg['batch_size_cap']), len(train))
    generator = torch.Generator(device=device).manual_seed(int(prior_seed))
    diag_rng = np.random.default_rng(int(prior_seed) + 1)
    fixed = {}
    for part, values in (('train', train_np), ('feedback', feedback_np)):
        count = min(128, len(values)); ids = diag_rng.choice(len(values), count, replace=False)
        t = diag_rng.integers(1, 1001, count)
        eps = diag_rng.standard_normal((count, dimension)).astype(np.float32)
        fixed[part] = (values[ids], t, eps)
    def measure(part):
        clean, times, eps_np = fixed[part]
        t = torch.as_tensor(times, dtype=torch.long, device=device)
        eps = torch.as_tensor(eps_np, dtype=torch.float32, device=device)
        x = torch.as_tensor(clean, dtype=torch.float32, device=device)
        noisy = alpha[t, None] * x + sigma[t, None] * eps
        with torch.no_grad():
            pred = model(noisy, t)
            return float(F.mse_loss(pred, eps).cpu()), float(eps.square().mean().cpu())
    history = []
    start_mse, zero_mse = measure('train')
    fb_mse, fb_zero = measure('feedback')
    history.append({'teacher': name, 'update': 0, 'train_mse': start_mse,
                    'train_zero_mse': zero_mse, 'feedback_mse': fb_mse,
                    'feedback_zero_mse': fb_zero})
    model.train()
    for update in range(1, updates + 1):
        ids = torch.randint(len(train), (batch_size,), device=device, generator=generator)
        t = torch.randint(1, int(dcfg['steps']) + 1, (batch_size,), device=device,
                          dtype=torch.long, generator=generator)
        eps = torch.randn((batch_size, dimension), device=device, dtype=torch.float32,
                          generator=generator)
        noisy = alpha[t, None] * train[ids] + sigma[t, None] * eps
        loss = F.mse_loss(model(noisy, t), eps)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{name} prior non-finite noise loss at update {update}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(dcfg['gradient_clip_norm']))
        if not torch.isfinite(norm):
            raise FloatingPointError(f'{name} prior non-finite gradient at update {update}')
        optimizer.step()
        if update % int(dcfg['diagnostic_interval']) == 0 or update == updates:
            a, b = measure('train'); c, d = measure('feedback')
            history.append({'teacher': name, 'update': update,
                'train_batch_loss': float(loss.detach().cpu()), 'gradient_norm': float(norm.detach().cpu()),
                'train_mse': a, 'train_zero_mse': b,
                'feedback_mse': c, 'feedback_zero_mse': d})
            _atomic_csv(history_path, history)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_torch(final_path, {'model': state, 'method_fingerprint': method_fp,
                               'model_sha256': reference._state_hash(state)})
    manifest = {'status': 'complete', 'teacher': name, 'conditioning': 'none',
        'method_fingerprint': method_fp, 'run_fingerprint': fingerprint,
        'checkpoint': str(final_path), 'checkpoint_sha256': _sha(final_path),
        'model_sha256': reference._state_hash(state), 'initial_model_sha256': initial_hash,
        'final_update': updates, 'training_partition': 'mean-pooled D_train teacher features only',
        'feature_dimension': dimension, 'history': str(history_path)}
    _atomic_json(manifest_path, manifest)
    return model, manifest


def _projection(dimension: int, seed: int, roots: dict, fingerprint: str):
    path = roots['run_cache'] / 'fixed_IFNet_meanpool_Q.npz'
    meta_path = roots['run_cache'] / 'fixed_IFNet_meanpool_Q.json'
    if path.is_file() or meta_path.is_file():
        if not (path.is_file() and meta_path.is_file()):
            raise RuntimeError('incomplete mean-pool IFNet projection artifacts')
        meta = _read_json(meta_path)
        if (meta.get('fingerprint') != fingerprint or meta.get('seed') != seed
                or meta.get('dimension') != dimension or meta.get('file_sha256') != _sha(path)):
            raise RuntimeError('mean-pool IFNet projection identity/hash mismatch')
        with np.load(resolve_local_file(path), allow_pickle=False) as saved:
            q = np.asarray(saved['Q'], dtype=np.float32)
        return torch.as_tensor(q), meta
    q64 = core.fixed_student_projection(dimension, seed)
    q = q64.astype(np.float32)
    _atomic_npz(path, Q=q, Q_float64=q64)
    meta = {'fingerprint': fingerprint, 'file_sha256': _sha(path), 'seed': seed,
            'dimension': dimension, 'shape': list(q.shape),
            'construction': 'CPU float64 Gaussian [512,d], thin QR, canonical positive R diagonal',
            'row_orthogonality_error': float(np.max(np.abs(q64 @ q64.T - np.eye(dimension)))),
            'layernorm_centered_rank': int(np.linalg.matrix_rank(
                q64 @ (np.eye(512) - np.ones((512, 512)) / 512), tol=1e-10))}
    _atomic_json(meta_path, meta)
    return torch.as_tensor(q), meta


def _student_feedback(model, adapter, x: np.ndarray, y: np.ndarray, device):
    return channel_run._metric_values(np.asarray(y), channel_run._predict_student(model, adapter, x, device))


def _fit_student(condition: str, config: dict, identity: dict, roots: dict,
                 models: dict, data: dict, initial: dict, targets: dict,
                 projection: torch.Tensor, prior, fingerprint: str,
                 device: torch.device, smoke: bool = False):
    short_name = condition.removeprefix('MEANPOOL_')
    teacher = TEACHER_FOR[short_name]
    diffuse = IS_DIFFUSION[short_name]
    epochs = 2 if smoke else int(models['student_cfg']['epochs'])
    final_path = roots['run_weights'] / f'student_{condition}.pt'
    manifest_path = roots['run_result'] / f'student_{condition}.json'
    method_fp = _json_hash({'run_fingerprint': fingerprint, 'condition': condition,
        'source_targets_sha256': _sha(roots['run_cache'] / 'teacher_meanpool_targets.npz'),
        'Q_sha256': _sha(roots['run_cache'] / 'fixed_IFNet_meanpool_Q.npz'),
        'prior_sha256': (_sha(roots['run_weights'] / f'meanpool_prior_{teacher}.pt') if diffuse else None),
        'initial_IFNet_sha256': initial['model_sha256'], 'epochs': epochs,
        'feature_weight': float(config['student']['feature_weight'])})
    if final_path.exists() or manifest_path.exists():
        if not (final_path.exists() and manifest_path.exists()):
            raise RuntimeError(f'{condition} checkpoint/manifest incomplete')
        manifest = _read_json(manifest_path)
        if manifest.get('method_fingerprint') != method_fp or manifest.get('checkpoint_sha256') != _sha(final_path):
            raise RuntimeError(f'{condition} checkpoint identity/hash mismatch')
        return manifest
    model, optimizer, scheduler, _ = reference._new_paired_student(
        models, initial, int(config['seed']), device)
    x = torch.as_tensor(data['student_x']['train'], dtype=torch.float32, device=device)
    y = torch.as_tensor(identity['labels']['train'], dtype=torch.long, device=device)
    t = torch.as_tensor(targets[f'{teacher}_train_features_whitened'], dtype=torch.float32, device=device)
    q = projection.to(device)
    if prior is not None:
        prior.eval()
        for p in prior.parameters():
            p.requires_grad_(False); p.grad = None
    noise_seed = int(config['randomness']['student_diffusion_noise_seed_base']) + 1009 * int(identity['manifest']['target_subject_id']) + int(config['seed'])
    noise_rng = torch.Generator(device=device).manual_seed(noise_seed) if diffuse else None
    alpha = sigma = None
    if diffuse:
        _, alpha, sigma = core.ddpm_schedule(int(config['diffusion']['steps']),
            float(config['diffusion']['beta_start']), float(config['diffusion']['beta_end']), device)
    schedule = models['student_schedule']
    history, feedback_rows = [], []
    first_grad = None
    for epoch in range(epochs):
        model.train(); seen = steps = 0
        ce_sum = feat_sum = total_sum = grad_sum = 0.0
        for ids_np in schedule[epoch]:
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            native, logits = models['student_adapter'].forward(model, x[ids])
            native.retain_grad()
            projected = core.student_features_to_vector(native, q, epsilon=1e-5)
            if diffuse:
                noise = torch.randn(projected.shape, dtype=torch.float32, device=device, generator=noise_rng)
                aligned = core.ddim_denoise(projected, noise, prior, alpha, sigma,
                    tuple(config['diffusion']['ddim_timesteps']))
            else:
                aligned = projected
            feat = F.mse_loss(aligned, t[ids].detach(), reduction='mean')
            ce = F.cross_entropy(logits, y[ids])
            loss = ce + float(config['student']['feature_weight']) * feat
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{condition} non-finite loss in epoch {epoch + 1}')
            loss.backward()
            if first_grad is None:
                ng = float(native.grad.norm().detach().cpu()) if native.grad is not None else 0.0
                reached = any(p.grad is not None and torch.isfinite(p.grad).all()
                              and float(p.grad.norm().detach().cpu()) > 0 for p in model.parameters())
                first_grad = {'native_feature_gradient_norm': ng,
                              'student_parameter_gradient_reached': bool(reached)}
                if ng <= 0 or not reached:
                    raise RuntimeError(f'{condition}: alignment gradient did not reach IFNet')
            grad_norm = torch.linalg.vector_norm(torch.stack(
                [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'{condition} non-finite IFNet gradient')
            optimizer.step()
            n = len(ids); seen += n; steps += 1
            ce_sum += float(ce.detach().cpu()) * n
            feat_sum += float(feat.detach().cpu()) * n
            total_sum += float(loss.detach().cpu()) * n
            grad_sum += float(grad_norm.detach().cpu())
        lr_used = float(optimizer.param_groups[0]['lr']); scheduler.step()
        history.append({'condition': condition, 'teacher': teacher, 'epoch': epoch + 1,
            'train_ce': ce_sum / seen, 'train_feature_mse': feat_sum / seen,
            'weighted_feature_mse': float(config['student']['feature_weight']) * feat_sum / seen,
            'train_total': total_sum / seen, 'mean_gradient_norm': grad_sum / steps,
            'lr_used': lr_used, 'lr_next': float(scheduler.get_last_lr()[0]),
            'train_uid_sha256': identity['uid_sha256']['train'],
            'schedule_sha256': models['student_schedule_sha256'], **(first_grad or {})})
        if epoch + 1 in (1, 10, 50, 100) or (smoke and epoch + 1 == epochs):
            metrics = _student_feedback(model, models['student_adapter'], data['student_x']['feedback'],
                                        identity['labels']['feedback'], device)
            feedback_rows.append({'condition': condition, 'epoch': epoch + 1,
                                  **{f'feedback_{k}': v for k, v in metrics.items()}})
        _atomic_csv(roots['run_result'] / f'{condition}_history.csv', history)
        if feedback_rows:
            _atomic_csv(roots['run_result'] / f'{condition}_feedback.csv', feedback_rows)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _atomic_torch(final_path, {'model': state, 'method_fingerprint': method_fp,
        'model_sha256': reference._state_hash(state), 'condition': condition})
    manifest = {'status': 'complete', 'condition': condition, 'teacher': teacher,
        'uses_diffusion': diffuse, 'method_fingerprint': method_fp,
        'run_fingerprint': fingerprint, 'checkpoint': str(final_path),
        'checkpoint_sha256': _sha(final_path), 'model_sha256': reference._state_hash(state),
        'initial_model_sha256': initial['model_sha256'], 'final_epoch': epochs,
        'train_uid_sha256': identity['uid_sha256']['train'],
        'schedule_sha256': models['student_schedule_sha256'],
        'loss': {'ce_weight': 1.0, 'feature_weight': float(config['student']['feature_weight']),
                 'feature': 'MSE over one channel-mean-pooled teacher vector and one IFNet vector',
                 'logits_kd': False, 'start_epoch': 1},
        'projection_sha256': _sha(roots['run_cache'] / 'fixed_IFNet_meanpool_Q.npz'),
        'prior_sha256': _sha(roots['run_weights'] / f'meanpool_prior_{teacher}.pt') if diffuse else None,
        'history': str(roots['run_result'] / f'{condition}_history.csv')}
    _atomic_json(manifest_path, manifest)
    return manifest


def _fingerprint(config_path: Path, seed: int, fold: int, identity: dict,
                 student: dict, source_sha: str) -> str:
    return _json_hash({'config_sha256': _sha(config_path), 'seed': seed, 'fold': fold,
        'split_manifest_sha256': identity['sha256'], 'train_uid_sha256': identity['uid_sha256']['train'],
        'feedback_uid_sha256': identity['uid_sha256']['feedback'],
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'schedule_sha256': student['student_schedule_sha256'],
        'source_channel_targets_sha256': source_sha,
        'pooling': 'arithmetic mean across three channels before PCA/alignment'})


def _train_fold(config_path: Path, config: dict, seed: int, run_id: str,
                fold: int, device: torch.device, smoke: bool = False) -> dict:
    roots = _roots(run_id, seed, fold)
    historical, cfg, run_config, identity, data, student = _fold_context(
        config, seed, fold, roots, device, include_test=False)
    source_path = channel_run._roots(
        config['teachers']['source_feature_cache_run_id_prefix'] + f'_seed{seed}', seed, fold)['run_cache'] / 'teacher_channel_targets.npz'
    source_sha = _sha(source_path)
    fingerprint = _fingerprint(config_path, seed, fold, identity, student, source_sha)
    complete_path = roots['run_result'] / 'fold_complete.json'
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get('status') == 'trained' and complete.get('run_fingerprint') == fingerprint:
            return {'status': 'trained', 'fold': fold, 'resumed_existing': True}
        raise RuntimeError(f'fold {fold + 1}: completion belongs to another run')
    _atomic_json(roots['run_result'] / 'resolved_run.json', {
        'experiment_id': config['experiment_id'], 'seed': seed, 'fold_zero_based': fold,
        'run_fingerprint': fingerprint, 'config': str(config_path),
        'config_sha256': _sha(config_path), 'resolved_reference_sha256': cfg['resolved_config_sha256'],
        'source_teacher_targets_sha256': source_sha, 'source_partition': 'verified reused D_train/feedback cache',
        'pooling': 'mean over channel axis before PCA and feature alignment',
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'schedule_sha256': student['student_schedule_sha256'], 'status': 'training'})
    targets, pca, target_meta = _make_mean_targets(config, seed, fold, roots, identity, fingerprint)
    dim = int(pca['dimension'])
    q_seed = int(config['feature_space']['student_projection_seed_base']) + 100 * (seed - 666) + fold
    projection, q_meta = _projection(dim, q_seed, roots, fingerprint)
    prior_seed = int(config['randomness']['prior_seed_base']) + 31 * fold + seed
    prior_init, prior_init_hash = _prior_initial(dim, prior_seed, config)
    _atomic_torch(roots['run_weights'] / 'meanpool_prior_initial.pt',
                  {'model': prior_init, 'model_sha256': prior_init_hash, 'seed': prior_seed})
    priors, prior_meta = {}, {}
    for name in ('T_CE', 'T_MCC'):
        priors[name], prior_meta[name] = _fit_prior(name, config, identity, roots,
            targets, fingerprint, prior_init, prior_init_hash, prior_seed, device, smoke)
    student_meta = {}
    for condition in CONDITIONS:
        short = condition.removeprefix('MEANPOOL_')
        teacher = TEACHER_FOR[short]
        student_meta[condition] = _fit_student(condition, run_config, identity, roots,
            student, data, student['initial_student'], targets, projection,
            priors[teacher] if IS_DIFFUSION[short] else None, fingerprint, device, smoke)
    if smoke:
        _atomic_json(roots['run_result'] / 'smoke_result.json', {
            'status': 'passed', 'run_fingerprint': fingerprint, 'seed': seed, 'fold': fold,
            'target_evaluated': False, 'pooled_feature_dimension': dim,
            'pca_sha256': _sha(roots['run_cache'] / 'fixed_meanpool_T0_pca.npz'),
            'target_cache_sha256': _sha(roots['run_cache'] / 'teacher_meanpool_targets.npz'),
            'projection_sha256': q_meta['file_sha256'],
            'prior_updates': {k: v['final_update'] for k, v in prior_meta.items()},
            'student_epochs': {k: v['final_epoch'] for k, v in student_meta.items()}})
        shutil.rmtree(roots['run_weights'], ignore_errors=True)
        shutil.rmtree(roots['run_cache'], ignore_errors=True)
        return {'status': 'smoke_passed', 'fold': fold, 'run_fingerprint': fingerprint}
    complete = {'status': 'trained', 'experiment_id': config['experiment_id'],
        'run_fingerprint': fingerprint, 'seed': seed, 'fold_zero_based': fold,
        'target_subject_id': fold + 1, 'train_uid_sha256': identity['uid_sha256']['train'],
        'source_channel_targets_sha256': source_sha,
        'source_teacher_checkpoint_sha256': target_meta['source_teacher_checkpoint_sha256'],
        'pooled_target_cache_sha256': target_meta['file_sha256'],
        'pca_sha256': _sha(roots['run_cache'] / 'fixed_meanpool_T0_pca.npz'),
        'projection_sha256': q_meta['file_sha256'],
        'prior_checkpoints': {k: v['checkpoint_sha256'] for k, v in prior_meta.items()},
        'student_checkpoints': {k: v['checkpoint_sha256'] for k, v in student_meta.items()},
        'initial_IFNet_sha256': student['initial_student']['model_sha256'],
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    _atomic_json(complete_path, complete)
    return {'status': 'trained', 'fold': fold, 'run_fingerprint': fingerprint}


def _report_csv(path: Path, rows: list[dict]):
    out = require_report_output(path); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    fields = list(rows[0]) if rows else []
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, out)


def _load_old_prediction_rows(seed: int, subject: int, uids: np.ndarray,
                              labels: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(resolve_local_file(OLD_REPORT))
    part = frame[(frame['seed'].astype(int) == seed)
                 & (frame['target_subject'].astype(int) == subject)].copy()
    expected_uids = [f'{int(uid[0])}:{int(uid[1])}' for uid in uids]
    if (len(part) != len(uids) or part['trial_uid'].astype(str).tolist() != expected_uids
            or not np.array_equal(part['label'].to_numpy(dtype=np.int64), labels)):
        raise RuntimeError(f'old channel-run prediction UIDs/labels differ: seed={seed}, subject={subject}')
    required = ['T0', 'T_CE', 'T_MCC', 'IFNet_CE', *OLD_CONDITIONS]
    if any(f'{name}_logit_0' not in part or f'{name}_logit_1' not in part for name in required):
        raise RuntimeError('old channel-run target logits report is missing required models')
    return part


def _evaluate_fold(config: dict, seed: int, run_id: str, fold: int,
                   device: torch.device) -> tuple[list[dict], list[dict]]:
    roots = _roots(run_id, seed, fold)
    complete = _read_json(roots['run_result'] / 'fold_complete.json')
    if complete.get('status') != 'trained':
        raise RuntimeError(f'fold {fold + 1}: mean-pool training is not complete')
    historical, cfg = _resolved(config, seed)
    run_config = dict(config); run_config['seed'] = seed
    run_config['reference_run'] = {'protocol_id': config['reference_run']['protocol_id'],
        'seed': seed, 'resolved_config_sha256': cfg['resolved_config_sha256']}
    identity = reference._load_fold_identity(run_config, fold, roots,
        cfg['resolved_config_sha256'], include_test_labels=True)
    evaluation_path = roots['run_result'] / 'fold_evaluation.json'
    metrics_path = roots['run_result'] / 'fold_evaluation_metrics.csv'
    predictions_path = roots['run_result'] / 'fold_predictions.csv'
    if any(p.is_file() for p in (evaluation_path, metrics_path, predictions_path)):
        if not all(p.is_file() for p in (evaluation_path, metrics_path, predictions_path)):
            raise RuntimeError(f'fold {fold + 1}: incomplete saved target evaluation')
        saved = _read_json(evaluation_path)
        if (saved.get('status') != 'complete'
                or saved.get('run_fingerprint') != complete['run_fingerprint']
                or saved.get('test_uid_sha256') != identity['uid_sha256']['test']
                or saved.get('metrics_sha256') != _sha(metrics_path)
                or saved.get('predictions_sha256') != _sha(predictions_path)):
            raise RuntimeError(f'fold {fold + 1}: saved target evaluation provenance/hash mismatch')
        return (pd.read_csv(resolve_local_file(metrics_path)).to_dict('records'),
                pd.read_csv(resolve_local_file(predictions_path)).to_dict('records'))
    data = channel_run._load_data(identity, roots, historical['student_config'], device)
    rows = data['student_rows']['test']
    raw = np.asarray(data['student_input_mmap'][rows], dtype=np.float32)
    prep_adapter = get_adapter('ifnet', device='cpu', **historical['student_config'])
    x = reference._preprocess_student(prep_adapter, raw)
    y = np.asarray(identity['labels']['test'], dtype=np.int64)
    uids = np.asarray(identity['uids']['test'], dtype=np.int64)
    old = _load_old_prediction_rows(seed, fold + 1, uids, y)
    logits = {}
    for name in ('T0', 'T_CE', 'T_MCC', 'IFNet_CE', *OLD_CONDITIONS):
        logits[name] = old[[f'{name}_logit_0', f'{name}_logit_1']].to_numpy(dtype=np.float32)
    student_adapter = get_adapter('ifnet', device=device, **historical['student_config'])
    for condition in CONDITIONS:
        path = roots['run_weights'] / f'student_{condition}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'missing mean-pool student checkpoint: {path}')
        payload = torch.load(resolve_local_file(path), map_location='cpu', weights_only=False)
        model = student_adapter.build(2); model.load_state_dict(payload['model'], strict=True)
        if reference._state_hash(model.state_dict()) != payload.get('model_sha256'):
            raise RuntimeError(f'{condition}: student checkpoint hash mismatch')
        logits[condition], _ = channel_run._predict_student_preserving_state(model, student_adapter, x, device)
    subject = fold + 1
    metrics, predictions = [], []
    for name, values in logits.items():
        metrics.append({'seed': seed, 'target_subject': subject, 'model': name,
                        'alignment': ('meanpool' if name.startswith('MEANPOOL_') else
                                      'three_channel' if name in OLD_CONDITIONS else 'reference'),
                        'n_trials': len(y), **channel_run._metric_values(y, values)})
    for i, (uid, label) in enumerate(zip(uids, y)):
        row = {'seed': seed, 'target_subject': subject,
               'subject_zero_based': int(uid[0]), 'legacy_raw_row': int(uid[1]),
               'trial_uid': f'{int(uid[0])}:{int(uid[1])}', 'label': int(label)}
        for name, values in logits.items():
            row[f'{name}_logit_0'] = float(values[i, 0]); row[f'{name}_logit_1'] = float(values[i, 1])
            row[f'{name}_prediction'] = int(np.argmax(values[i]))
        predictions.append(row)
    _atomic_csv(roots['run_result'] / 'fold_evaluation_metrics.csv', metrics)
    _atomic_csv(roots['run_result'] / 'fold_predictions.csv', predictions)
    _atomic_json(evaluation_path, {
        'status': 'complete', 'seed': seed, 'target_subject': subject,
        'run_fingerprint': complete['run_fingerprint'], 'test_uid_sha256': identity['uid_sha256']['test'],
        'n_trials': len(y), 'metrics_sha256': _sha(roots['run_result'] / 'fold_evaluation_metrics.csv'),
        'predictions_sha256': _sha(roots['run_result'] / 'fold_predictions.csv'),
        'old_reference_report_sha256': _sha(OLD_REPORT),
        'target_labels_used_for_evaluation_only': True})
    return metrics, predictions


def _summarize_seed(config: dict, seed: int, run_id: str, device):
    metrics, predictions = [], []
    for fold in range(9):
        m, p = _evaluate_fold(config, seed, run_id, fold, device)
        metrics.extend(m); predictions.extend(p)
    model_names = list(dict.fromkeys(row['model'] for row in metrics))
    seed_means = []
    for model in model_names:
        part = [r for r in metrics if r['model'] == model]
        if len(part) != 9 or len({r['target_subject'] for r in part}) != 9:
            raise RuntimeError(f'{model}: seed {seed} is missing or duplicates LOSO subjects')
        seed_means.append({'seed': seed, 'model': model, 'n_subjects': 9,
            **{metric: float(np.mean([r[metric] for r in part]))
               for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')}})
    report = REPORTS_ROOT / config['storage']['report_directory'] / f'seed_{seed}'
    _report_csv(report / 'subject_metrics.csv', metrics)
    _report_csv(report / 'trial_predictions.csv', predictions)
    _report_csv(report / 'seed_means.csv', seed_means)
    channel_run._report_json(report / 'completion.json', {
        'status': 'complete', 'seed': seed, 'n_subjects': 9,
        'n_predictions': len(predictions), 'source_channel_report_sha256': _sha(OLD_REPORT),
        'aggregation': 'equal-weight mean over nine target subjects within this seed',
        'metrics_sha256': _sha(report / 'subject_metrics.csv'),
        'predictions_sha256': _sha(report / 'trial_predictions.csv')})
    # Keep only compact results and provenance after evaluation is safely saved.
    for fold in range(9):
        shutil.rmtree(_roots(run_id, seed, fold)['run_weights'], ignore_errors=True)
    return metrics, predictions, seed_means


def _aggregate(config: dict):
    report_root = REPORTS_ROOT / config['storage']['report_directory']
    subject_rows, prediction_rows, seed_means = [], [], []
    for seed in SEEDS:
        report = report_root / f'seed_{seed}'
        completion = _read_json(report / 'completion.json')
        if completion.get('status') != 'complete' or completion.get('n_subjects') != 9:
            raise RuntimeError(f'seed {seed} evaluation is incomplete')
        metrics = pd.read_csv(resolve_local_file(report / 'subject_metrics.csv')).to_dict('records')
        predictions = pd.read_csv(resolve_local_file(report / 'trial_predictions.csv')).to_dict('records')
        if len(metrics) != 9 * (3 + 1 + 4 + 4) or len(predictions) != 1400:
            raise RuntimeError(f'seed {seed} report row count mismatch: metrics={len(metrics)}, preds={len(predictions)}')
        subject_rows.extend(metrics); prediction_rows.extend(predictions)
        for model in dict.fromkeys(r['model'] for r in metrics):
            part = [r for r in metrics if r['model'] == model]
            seed_means.append({'seed': seed, 'model': model, 'n_subjects': 9,
                **{metric: float(np.mean([float(r[metric]) for r in part]))
                   for metric in ('accuracy', 'balanced_accuracy', 'kappa', 'cross_entropy')}})
    summary = []
    for model in dict.fromkeys(r['model'] for r in seed_means):
        part = [r for r in seed_means if r['model'] == model]
        if {int(r['seed']) for r in part} != set(SEEDS):
            raise RuntimeError(f'{model}: missing seed mean')
        summary.append({'model': model, 'n_seed': 3, 'seeds': '666,667,668', 'n_subjects_per_seed': 9,
            **{f'{m}_mean': float(np.mean([r[m] for r in part])) for m in ('accuracy','balanced_accuracy','kappa','cross_entropy')},
            **{f'{m}_sd_across_seed_means': float(np.std([r[m] for r in part], ddof=1)) for m in ('accuracy','balanced_accuracy')}})
    pairs = []
    for metric in ('accuracy', 'balanced_accuracy'):
        for old_name in OLD_CONDITIONS:
            new_name = 'MEANPOOL_' + old_name
            deltas_by_seed = []
            all_deltas = []
            for seed in SEEDS:
                lookup = {(int(r['target_subject']), r['model']): float(r[metric])
                          for r in subject_rows if int(r['seed']) == seed}
                deltas = [lookup[(subject, new_name)] - lookup[(subject, old_name)] for subject in range(1,10)]
                deltas_by_seed.append(float(np.mean(deltas))); all_deltas.extend(deltas)
                pairs.append({'comparison': f'{new_name} minus {old_name}', 'metric': metric,
                    'seed': seed, 'mean_subject_difference_pp': float(np.mean(deltas) * 100),
                    'win': sum(d > 1e-12 for d in deltas), 'tie': sum(abs(d) <= 1e-12 for d in deltas),
                    'loss': sum(d < -1e-12 for d in deltas), 'n_subjects': 9})
            pairs.append({'comparison': f'{new_name} minus {old_name}', 'metric': metric,
                'seed': 'all', 'mean_subject_difference_pp': float(np.mean(deltas_by_seed) * 100),
                'sd_across_seed_mean_differences_pp': float(np.std(deltas_by_seed, ddof=1) * 100),
                'win': sum(d > 1e-12 for d in all_deltas), 'tie': sum(abs(d) <= 1e-12 for d in all_deltas),
                'loss': sum(d < -1e-12 for d in all_deltas), 'n_seed': 3, 'n_seed_subject_pairs': 27})
    _report_csv(report_root / 'summary_3seed.csv', summary)
    _report_csv(report_root / 'seed_means_3seed.csv', seed_means)
    _report_csv(report_root / 'subject_metrics_3seed.csv', subject_rows)
    _report_csv(report_root / 'trial_predictions_3seed.csv', prediction_rows)
    _report_csv(report_root / 'paired_meanpool_vs_three_channel_3seed.csv', pairs)
    result = {'status': 'complete', 'experiment_id': config['experiment_id'],
        'seeds': list(SEEDS), 'subjects_per_seed': 9, 'n_student_fold_conditions': 108,
        'pooling': 'arithmetic mean across three teacher channels before PCA and alignment',
        'teacher_adaptation_and_logits': 'reused verified prior-run T0/T_CE/T_MCC artifacts; no teacher retraining',
        'paired_channel_report': str(OLD_REPORT), 'paired_channel_report_sha256': _sha(OLD_REPORT),
        'aggregation': 'equal-weight subject mean within each seed, then equal-weight mean across seeds',
        'comparison': 'mean-pool minus original three-channel result; paired by seed and LOSO subject',
        'summary_sha256': _sha(report_root / 'summary_3seed.csv'),
        'paired_differences_sha256': _sha(report_root / 'paired_meanpool_vs_three_channel_3seed.csv'),
        'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    channel_run._report_json(report_root / 'completion_3seed.json', result)
    _write_workbook(report_root / 'cbramod_meanpool_mcc_diffkd_loso_3seed.xlsx', summary, pairs)
    return result


def _write_workbook(path: Path, summary: list[dict], pairs: list[dict]):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook(); ws = wb.active; ws.title = 'Summary'
    columns = ['Model', 'Accuracy (%)', 'Accuracy SD (seed means)',
               'Balanced Accuracy (%)', 'BA SD (seed means)', 'Seeds', 'Subjects/seed']
    ws.append(columns)
    pct = {'accuracy_mean', 'balanced_accuracy_mean', 'accuracy_sd_across_seed_means',
           'balanced_accuracy_sd_across_seed_means'}
    for r in summary:
        ws.append([r['model'], r['accuracy_mean']*100, r['accuracy_sd_across_seed_means']*100,
                   r['balanced_accuracy_mean']*100, r['balanced_accuracy_sd_across_seed_means']*100,
                   r['seeds'], r['n_subjects_per_seed']])
    for cell in ws[1]:
        cell.fill = PatternFill('solid', fgColor='1F4E78'); cell.font = Font(color='FFFFFF', bold=True)
    ws.freeze_panes = 'B2'; ws.sheet_view.showGridLines = False
    for col in range(2,6):
        for row in range(2,ws.max_row+1): ws.cell(row,col).number_format = '0.00'
    ws2=wb.create_sheet('Paired differences'); ws2.append(list(pairs[0]) if pairs else ['comparison'])
    for r in pairs: ws2.append([r.get(k,'') for k in ws2[1]])
    for sheet in (ws,ws2):
        for cell in sheet[1]:
            cell.fill=PatternFill('solid',fgColor='1F4E78'); cell.font=Font(color='FFFFFF',bold=True)
            cell.alignment=Alignment(horizontal='center',wrap_text=True)
        sheet.sheet_view.showGridLines=False
        for column in sheet.columns:
            letter=get_column_letter(column[0].column)
            sheet.column_dimensions[letter].width=min(38,max(14,max(len(str(c.value or '')) for c in column)+2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row: cell.alignment=Alignment(horizontal='center',vertical='center')
    out=require_report_output(path); out.parent.mkdir(parents=True,exist_ok=True); wb.save(out)


def _write_progress(path: Path | None, status: str, records: list[dict], **extra):
    if path is None: return
    channel_run._write_progress(path, {'status':status,'records':records,**extra})


def run_seed(config_path: Path, config: dict, seed: int, run_id: str, gpu: int,
             folds: list[int] | None, smoke: bool, progress: Path | None,
             stop_file: Path | None) -> int:
    _validate(config, seed)
    device=channel_run._device(gpu)
    selected=[0] if smoke else list(folds if folds is not None else range(9))
    if not selected or len(set(selected)) != len(selected) or any(f<0 or f>8 for f in selected):
        raise ValueError('folds must be a unique non-empty subset of zero-based 0..8')
    records=[]
    for fold in selected:
        fold_result=_train_fold(config_path,config,seed,run_id,fold,device,smoke=smoke)
        if smoke:
            records.append({'fold':fold+1,'status':'smoke_passed'})
        else:
            records.extend({'fold':fold+1,'condition':c,'status':'trained'} for c in CONDITIONS)
        _write_progress(progress,'running',records,seed=seed,total_cells=36 if not smoke else 4)
        if stop_file is not None and Path(stop_file).exists():
            _write_progress(progress,'paused',records,seed=seed)
            return 75
    if smoke:
        _write_progress(progress,'completed',records,seed=seed,target_evaluated=False)
        return 0
    if len(selected)!=9:
        raise ValueError('formal seed run must train all nine folds before target evaluation')
    if stop_file is not None and Path(stop_file).exists():
        _write_progress(progress,'paused',records,seed=seed); return 75
    _summarize_seed(config,seed,run_id,device)
    _atomic_json(RESULTS_ROOT/'distill'/run_id/f'seed_{seed}'/'seed_complete.json',{
        'status':'complete','seed':seed,'run_id':run_id,
        'completed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    _write_progress(progress,'completed',records,seed=seed,evaluation='complete')
    return 0


def enqueue_run(config_path: Path, queue_root: Path | None = None,
                smoke: bool = False, smoke_id: str | None = None) -> list[str]:
    from experiments.distill.queue import DEFAULT_ROOT, enqueue_snapshot
    config_path=Path(config_path).resolve(); config=_read_config(config_path)
    selected=[666] if smoke else list(SEEDS)
    for seed in selected: _validate(config,seed)
    root=Path(queue_root or DEFAULT_ROOT).resolve(); jobs=[]
    for seed in selected:
        run_id=(f"cbramod_meanpool_mcc_diffkd_loso_v1_smoke_{smoke_id or int(time.time())}_seed{seed}"
                if smoke else f'cbramod_meanpool_mcc_diffkd_loso_v1_seed{seed}')
        snapshot={'kind':'cbramod_meanpool_mcc_diffkd_loso_v1',
            'stage':'smoke' if smoke else 'train_seed','config':str(config_path),
            'config_sha256':_sha(config_path),'seed':seed,'run_id':run_id,
            'folds':[0] if smoke else list(range(9)),
            'args':{'dataset':'BNCI2014004','teacher':'cbramod','student':'ifnet',
                'protocol':'loso_7_1_1','keys':[0] if smoke else list(range(9)),
                'seeds':[seed],'methods':list(CONDITIONS),'source_variant':'wide_fre_band',
                'selected_session':'session_3','epochs':100,'channel_pooling':'mean'}}
        jobs.append(enqueue_snapshot(snapshot,root=root,
            label=f'CBraMod mean-pool feature alignment seed={seed}, '+('smoke' if smoke else '9-fold LOSO')))
    if not smoke:
        report_dir=REPORTS_ROOT/config['storage']['report_directory']/'dispatch'
        report_dir.mkdir(parents=True,exist_ok=True)
        log=report_dir/'watcher.log'
        command=[sys.executable,'-B','-m','experiments.distill.run_cbramod_meanpool_mcc_diffkd_loso',
            '--watch-jobs','--queue-root',str(root),'--job-ids',*jobs,'--config',str(config_path)]
        env=os.environ.copy()
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                     'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BIGSMALLCOLLAB_CPU_THREADS','TORCH_NUM_THREADS'):
            env[name]='1'
        with log.open('a',encoding='utf-8') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,env=env,close_fds=True)
        channel_run._report_json(report_dir/'dispatch.json',{'status':'submitted',
            'experiment_id':config['experiment_id'],'config':str(config_path),
            'config_sha256':_sha(config_path),'seeds':selected,'training_job_ids':jobs,
            'queue_root':str(root),'watcher_pid':proc.pid,'teacher_retraining':False})
    return jobs


def _watch_jobs(config_path: Path, queue_root: Path, job_ids: list[str]) -> int:
    from experiments.distill.queue import QueueStore, validated_root
    store=QueueStore(validated_root(queue_root))
    while True:
        states=[store.load(job) for job in job_ids]
        failed=[s for s in states if s['status'] in ('failed','cancelled')]
        report=REPORTS_ROOT/_read_config(config_path)['storage']['report_directory']/'dispatch'
        if failed:
            channel_run._report_json(report/'watcher_state.json',{'status':'failed',
                'jobs':[{'id':s['id'],'status':s['status'],'message':s.get('message')} for s in failed]})
            return 1
        if all(s['status']=='completed' for s in states):
            result=_aggregate(_read_config(config_path))
            channel_run._report_json(report/'watcher_state.json',{'status':'complete',
                'jobs':job_ids,'aggregate_completion':result})
            return 0
        time.sleep(30)


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--seed',type=int,default=666); parser.add_argument('--run-id',default=None)
    parser.add_argument('--gpu',type=int,default=0); parser.add_argument('--folds',nargs='+',type=int,default=None)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--enqueue-smoke',action='store_true')
    parser.add_argument('--smoke-id',default=None); parser.add_argument('--enqueue',action='store_true')
    parser.add_argument('--queue-root',type=Path,default=None); parser.add_argument('--progress',type=Path,default=None)
    parser.add_argument('--stop-file',type=Path,default=None); parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--watch-jobs',action='store_true'); parser.add_argument('--job-ids',nargs='*',default=[])
    args=parser.parse_args(argv)
    try:
        config_path=args.config.resolve(); config=_read_config(config_path)
        if args.enqueue_smoke:
            jobs=enqueue_run(config_path,args.queue_root,smoke=True,smoke_id=args.smoke_id)
            print(json.dumps({'smoke_job_ids':jobs},indent=2),flush=True); return 0
        if args.enqueue:
            jobs=enqueue_run(config_path,args.queue_root,smoke=False)
            print(json.dumps({'training_job_ids':jobs},indent=2),flush=True); return 0
        if args.aggregate:
            print(json.dumps(_aggregate(config),indent=2),flush=True); return 0
        if args.watch_jobs:
            if not args.job_ids: raise ValueError('--watch-jobs requires --job-ids')
            return _watch_jobs(config_path,args.queue_root or Path('/home/lixinli/BigSmallCollab/results/loso/distill/queue'),args.job_ids)
        run_id=args.run_id or f'cbramod_meanpool_mcc_diffkd_loso_v1_seed{args.seed}'
        return run_seed(config_path,config,args.seed,run_id,args.gpu,args.folds,args.smoke,args.progress,args.stop_file)
    except Exception as exc:
        traceback.print_exc(); print(f'[cbramod-meanpool-mcc-diffkd] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
