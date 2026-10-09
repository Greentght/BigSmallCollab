#!/usr/bin/env python
"""LOSO logits/feature KD on the refreshed BNCI 004 and 5001 inputs.

Teacher targets are exported from the completed source-refreshed LOSO teacher
checkpoints and contain source-training trials only. Student configurations
and CE comparisons come from the matching source-refreshed baseline artifacts.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models import get_adapter
from experiments.distill import run_loso_distillation as core
from experiments.finetune import run_loso_five_datasets as metric_protocol
from experiments.storage import require_external_output, resolve_local_file

CONFIG_PATH = ROOT / 'configs/reproductions/loso_source_refresh_distillation_004_5001_v1.yaml'
INPUT_ROOT = Path('/data1/llx/BigSmallcollab/cache/reproductions/loso_source_refresh_004_5001_v1/model_inputs')
BASELINE_ROOT = Path('/data1/llx/BigSmallcollab/results/reproductions/loso_source_refresh_004_5001_v1')
TEACHER_CACHE_ROOT = Path('/data1/llx/BigSmallcollab/cache/reproductions/loso_source_refresh_004_5001_distillation_v1/teacher_targets')
RESULT_ROOT = Path('/data1/llx/BigSmallcollab/results/distill/loso_source_refresh_004_5001_v1')
DATASETS = ('BNCI2014004', 'BNCI2015001')
TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
STAGES = ('logits_kd', 'kd_feature', 'warmup10_kd', 'warmup10_kd_feature')
MODELS = (*TEACHERS, *STUDENTS)
SUBJECTS = {'BNCI2014004': 9, 'BNCI2015001': 12}
METRICS = ('accuracy', 'balanced_accuracy', 'kappa', 'macro_f1', 'auroc')
_HASH_CACHE: dict[str, tuple[tuple[int, int, int, int], str]] = {}


def sha256(path: Path) -> str:
    path = resolve_local_file(path).resolve()
    stat = path.stat()
    signature = (stat.st_dev, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HASH_CACHE.get(str(path))
    if cached and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    result = digest.hexdigest()
    _HASH_CACHE[str(path)] = (signature, result)
    return result


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_hash(value) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_plan(path: Path = CONFIG_PATH) -> dict:
    plan = yaml.safe_load(resolve_local_file(path).read_text())
    if tuple(plan['datasets']) != DATASETS or tuple(plan['stages']) != STAGES:
        raise RuntimeError('distillation config has unexpected dataset or stage order')
    if plan['loss']['teacher_correct_mask'] or float(plan['loss']['ce_weight']) != 1.0:
        raise RuntimeError('this protocol requires unmasked KD and unit-weight CE')
    plan['_config_sha256'] = sha256(path)
    return plan


def input_paths(dataset: str, model: str) -> dict[str, Path]:
    folder = INPUT_ROOT / dataset / model
    return {name: folder / name for name in
            ('X.npy', 'y.npy', 'subjects.npy', 'trials.csv', 'manifest.json')}


def load_input(dataset: str, model: str, verify_hashes: bool = False) -> dict:
    paths = input_paths(dataset, model)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{dataset}/{model}: missing model inputs {missing}')
    manifest_path = paths['manifest.json']
    manifest = json.loads(resolve_local_file(manifest_path).read_text())
    for name, record in manifest['files'].items():
        path = paths[name]
        if path.stat().st_size != int(record['bytes']):
            raise RuntimeError(f'{path}: size differs from input manifest')
        if verify_hashes and sha256(path) != record['sha256']:
            raise RuntimeError(f'{path}: hash differs from input manifest')
    x = np.load(resolve_local_file(paths['X.npy']), mmap_mode='r', allow_pickle=False)
    y = np.load(resolve_local_file(paths['y.npy']), mmap_mode='r', allow_pickle=False)
    subject_ids = np.load(resolve_local_file(paths['subjects.npy']), mmap_mode='r', allow_pickle=False)
    with resolve_local_file(paths['trials.csv']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not (len(x) == len(y) == len(subject_ids) == len(rows)):
        raise RuntimeError(f'{dataset}/{model}: input arrays and trial metadata differ in length')
    uids = np.asarray([row['trial_uid'] for row in rows], dtype=str)
    csv_y = np.asarray([int(row['label_id']) for row in rows], dtype=np.int64)
    csv_subjects = np.asarray([int(row['subject_zero_based']) for row in rows], dtype=np.int64)
    if not np.array_equal(y, csv_y) or not np.array_equal(subject_ids, csv_subjects):
        raise RuntimeError(f'{dataset}/{model}: trial metadata disagrees with y/subjects arrays')
    if len(np.unique(uids)) != len(uids):
        raise RuntimeError(f'{dataset}/{model}: duplicate trial UID')
    mapping = manifest['class_mapping']
    if not np.array_equal(np.unique(y), np.arange(len(mapping))):
        raise RuntimeError(f'{dataset}/{model}: class IDs do not match the input manifest')
    expected_n = 1400 if dataset == 'BNCI2014004' else 2400
    if len(y) != expected_n or len(np.unique(subject_ids)) != SUBJECTS[dataset]:
        raise RuntimeError(f'{dataset}/{model}: unexpected trial or subject count')
    return {'x': x, 'y': np.asarray(y, dtype=np.int64),
            'subjects': np.asarray(subject_ids, dtype=np.int64), 'uids': uids,
            'rows': rows, 'manifest': manifest,
            'manifest_sha256': sha256(manifest_path),
            'file_records': manifest['files']}


def validate_shared_trials(dataset: str, teacher_data: dict, student_data: dict) -> None:
    for key in ('uids', 'y', 'subjects'):
        if not np.array_equal(teacher_data[key], student_data[key]):
            raise RuntimeError(f'{dataset}: teacher and student {key} differ')
    if teacher_data['manifest']['class_mapping'] != student_data['manifest']['class_mapping']:
        raise RuntimeError(f'{dataset}: teacher and student class mappings differ')


def baseline_cell(dataset: str, model: str, fold: int, seed: int) -> Path:
    return (BASELINE_ROOT / dataset / model / 'source_refreshed_loso_v1'
            / f'seed_{seed}' / f'subject_{fold + 1:02d}')


def validate_baseline(dataset: str, model: str, fold: int, seed: int,
                      data: dict, verify_checkpoint_hash: bool = False) -> tuple[dict, dict, dict]:
    cell = baseline_cell(dataset, model, fold, seed)
    manifest_path, result_path = cell / 'manifest.json', cell / 'result.npz'
    checkpoint_path, history_path = cell / 'final_model.pt', cell / 'history.csv'
    if not all(p.is_file() for p in (manifest_path, result_path, checkpoint_path, history_path)):
        raise FileNotFoundError(f'{dataset}/{model} S{fold+1} seed={seed}: incomplete baseline {cell}')
    manifest = json.loads(resolve_local_file(manifest_path).read_text())
    checks = {'protocol': 'loso_source_refresh_004_5001_v1', 'dataset': dataset,
              'model': model, 'recipe': 'source_refreshed_loso_v1', 'seed': seed,
              'held_out_subject': fold + 1, 'status': 'complete',
              'input_manifest_sha256': data['manifest_sha256']}
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f'{manifest_path}: {key} differs: {manifest.get(key)!r} != {expected!r}')
    test_idx = data['subjects'] == fold
    train_idx = ~test_idx
    if manifest['n_train'] != int(train_idx.sum()) or manifest['n_test'] != int(test_idx.sum()):
        raise RuntimeError(f'{manifest_path}: train/test counts differ from the source trial list')
    if manifest['class_mapping'] != data['manifest']['class_mapping']:
        raise RuntimeError(f'{manifest_path}: class mapping mismatch')
    cfg = manifest['model_config']
    if int(cfg['epochs']) < 1 or (model in STUDENTS and int(cfg['epochs']) != 100):
        raise RuntimeError(f'{manifest_path}: baseline epoch count is incompatible with this protocol')
    with history_path.open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if len(history) != int(cfg['epochs']) or int(history[-1]['epoch']) != int(cfg['epochs']):
        raise RuntimeError(f'{history_path}: incomplete baseline history')
    with np.load(resolve_local_file(result_path), allow_pickle=True) as saved:
        test_uids = np.asarray(saved['trial_uid']).astype(str)
        test_y = saved['y'].astype(np.int64, copy=False)
        if not np.array_equal(test_uids, data['uids'][test_idx]) or not np.array_equal(test_y, data['y'][test_idx]):
            raise RuntimeError(f'{result_path}: test UIDs or labels disagree with source trials')
        metrics = json.loads(str(saved['metrics_json'].item()))
    if verify_checkpoint_hash and sha256(checkpoint_path) != manifest.get('final_model_sha256'):
        raise RuntimeError(f'{checkpoint_path}: checkpoint hash mismatch')
    record = {'path': str(cell), 'manifest_sha256': sha256(manifest_path),
              'result_sha256': sha256(result_path),
              'final_model_sha256': manifest['final_model_sha256'],
              'input_manifest_sha256': data['manifest_sha256'],
              'model_config_sha256': json_hash(cfg)}
    return cfg, record, metrics


def validate_prerequisites(verify_hashes: bool = True) -> dict:
    inventory, common = {}, {}
    for dataset in DATASETS:
        for model in MODELS:
            data = load_input(dataset, model, verify_hashes=verify_hashes)
            inventory[f'{dataset}/{model}'] = {
                'input_manifest_sha256': data['manifest_sha256'],
                'source_manifest_sha256': data['manifest']['source_manifest_sha256'],
                'n_trials': len(data['y']), 'x_shape': list(data['x'].shape),
                'input_files': data['file_records']}
            identity = (data['uids'], data['y'], data['subjects'])
            if dataset not in common:
                common[dataset] = identity
            elif not all(np.array_equal(a, b) for a, b in zip(common[dataset], identity)):
                raise RuntimeError(f'{dataset}: model caches do not have identical trial order')
            for seed in (666, 667, 668):
                for fold in range(SUBJECTS[dataset]):
                    _, _, _ = validate_baseline(dataset, model, fold, seed, data,
                                                verify_checkpoint_hash=verify_hashes)
        expected_sessions = {'BNCI2014004': 'session_3', 'BNCI2015001': 'session_A'}
        if inventory[f'{dataset}/mirepnet']['n_trials'] != (1400 if dataset == 'BNCI2014004' else 2400):
            raise RuntimeError(f'{dataset}: expected selected LOSO trial count changed')
        if data['manifest']['selected_session'] != expected_sessions[dataset]:
            raise RuntimeError(f'{dataset}: selected session changed')
    return {'protocol': 'loso_source_refresh_004_5001_v1',
            'inputs_and_baselines_complete': True, 'inventory': inventory,
            'validated_unix': time.time()}


def cache_dir(dataset: str, teacher: str, fold: int, seed: int) -> Path:
    return (TEACHER_CACHE_ROOT / dataset / teacher / f'seed_{seed}'
            / f'subject_{fold + 1:02d}')


def teacher_checkpoint(dataset: str, teacher: str, fold: int, seed: int) -> tuple[Path, Path, dict]:
    cell = baseline_cell(dataset, teacher, fold, seed)
    manifest_path = cell / 'manifest.json'
    checkpoint_path = cell / 'final_model.pt'
    manifest = json.loads(resolve_local_file(manifest_path).read_text())
    if manifest.get('final_model_sha256') != sha256(checkpoint_path):
        raise RuntimeError(f'{checkpoint_path}: teacher checkpoint hash mismatch')
    return checkpoint_path, manifest_path, manifest


def _atomic_npz(path: Path, arrays: dict) -> str:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + '.', suffix='.npz.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256(temp_path)
        os.replace(temp_path, path)
        return digest
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare_teacher_fold(dataset: str, teacher: str, fold: int, seed: int,
                         gpu: int, data: dict | None = None) -> dict:
    if gpu == 0:
        raise RuntimeError('GPU 0 is prohibited for this experiment')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    if data is None:
        data = load_input(dataset, teacher, verify_hashes=False)
    cell = cache_dir(dataset, teacher, fold, seed)
    cache_path, manifest_path = cell / 'targets.npz', cell / 'manifest.json'
    lock_path = require_external_output(cell / '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ckpt_path, teacher_manifest_path, baseline = teacher_checkpoint(dataset, teacher, fold, seed)
        train_idx, test_idx = data['subjects'] != fold, data['subjects'] == fold
        if manifest_path.is_file() and cache_path.is_file():
            old = json.loads(resolve_local_file(manifest_path).read_text())
            if (old.get('teacher_checkpoint_sha256') == baseline['final_model_sha256']
                    and old.get('input_manifest_sha256') == data['manifest_sha256']
                    and old.get('train_uid_sha256') == array_hash(data['uids'][train_idx])
                    and old.get('seed') == seed and old.get('fold_zero_based') == fold
                    and cache_path.stat().st_size == old.get('cache_bytes')):
                return old
        if int(baseline['n_train']) != int(train_idx.sum()) or int(baseline['n_test']) != int(test_idx.sum()):
            raise RuntimeError(f'{teacher}/{dataset} S{fold+1}: teacher baseline partition mismatch')
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        cfg = baseline['model_config']
        adapter = get_adapter(teacher, device=device, **cfg)
        model = adapter.build(len(data['manifest']['class_mapping']))
        try:
            state = torch.load(resolve_local_file(ckpt_path), map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(resolve_local_file(ckpt_path), map_location='cpu')
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        x_train = np.asarray(data['x'][train_idx], dtype=np.float32)
        x_tensor = adapter.preprocess(x_train)
        logits_out, feats_out = [], []
        batch = int(cfg.get('batch_size', 32))
        with torch.inference_mode():
            for start in range(0, len(x_tensor), batch):
                feat, logits = adapter.forward(model, x_tensor[start:start+batch].to(device))
                feats_out.append(feat.detach().float().cpu().numpy())
                logits_out.append(logits.detach().float().cpu().numpy())
        logits = np.concatenate(logits_out).astype(np.float32, copy=False)
        feats = np.concatenate(feats_out).astype(np.float32, copy=False)
        y = data['y'][train_idx].astype(np.int64, copy=False)
        uids = data['uids'][train_idx]
        if logits.shape != (len(y), len(data['manifest']['class_mapping'])) or feats.ndim != 2:
            raise RuntimeError(f'{teacher}/{dataset} S{fold+1}: invalid target shapes {logits.shape}/{feats.shape}')
        if not np.isfinite(logits).all() or not np.isfinite(feats).all():
            raise RuntimeError(f'{teacher}/{dataset} S{fold+1}: teacher targets contain NaN/Inf')
        with np.load(resolve_local_file(baseline_cell(dataset, teacher, fold, seed) / 'result.npz'), allow_pickle=True) as saved:
            if not np.array_equal(np.asarray(saved['trial_uid']).astype(str), data['uids'][test_idx]):
                raise RuntimeError('held-out teacher test UIDs do not match the refreshed source trial list')
        cache_hash = _atomic_npz(cache_path, {'logits': logits, 'feats': feats,
                                               'y': y, 'trial_uid': uids})
        metadata = {
            'protocol': 'loso_source_refresh_004_5001_kd_feature_warmup10_v1',
            'source_protocol': 'loso_source_refresh_004_5001_v1',
            'dataset': dataset, 'teacher': teacher, 'seed': seed,
            'fold_zero_based': fold, 'test_subject': fold + 1,
            'teacher_recipe': 'source_refreshed_loso_v1',
            'teacher_checkpoint': str(ckpt_path),
            'teacher_checkpoint_sha256': baseline['final_model_sha256'],
            'teacher_manifest_sha256': sha256(teacher_manifest_path),
            'input_manifest_sha256': data['manifest_sha256'],
            'input_X_sha256': data['file_records']['X.npy']['sha256'],
            'source_manifest_sha256': data['manifest']['source_manifest_sha256'],
            'source_variant': data['manifest']['source_variant'],
            'selected_session': data['manifest']['selected_session'],
            'model_config': cfg, 'model_config_sha256': json_hash(cfg),
            'n_train': int(train_idx.sum()), 'n_test': int(test_idx.sum()),
            'num_classes': len(data['manifest']['class_mapping']),
            'class_mapping': data['manifest']['class_mapping'],
            'feature_dim': int(feats.shape[1]), 'logits_shape': list(logits.shape),
            'feats_shape': list(feats.shape),
            'train_uid_sha256': array_hash(uids), 'train_labels_sha256': array_hash(y),
            'train_uids': uids.tolist(),
            'target_policy': 'frozen_teacher_inference_on_source_training_trials_only',
            'cache_file': str(cache_path), 'cache_file_sha256': cache_hash,
            'cache_bytes': cache_path.stat().st_size,
            'runner_sha256': sha256(Path(__file__)),
            'status': 'complete', 'completed_unix': time.time(),
        }
        temp_manifest = manifest_path.with_suffix('.json.tmp')
        temp_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n')
        os.replace(temp_manifest, manifest_path)
        del model, adapter, x_tensor, x_train, feats_out, logits_out, logits, feats
        gc.collect()
        torch.cuda.empty_cache()
        return metadata


def prepare_teacher_task(dataset: str, teacher: str, seed: int, gpu: int) -> None:
    data = load_input(dataset, teacher, verify_hashes=False)
    for fold in range(SUBJECTS[dataset]):
        metadata = prepare_teacher_fold(dataset, teacher, fold, seed, gpu, data=data)
        print(f'[target-ready] {teacher}/{dataset} S{fold+1} seed={seed} '
              f'features={metadata["feature_dim"]} cache_mib={metadata["cache_bytes"] / 2**20:.1f}', flush=True)
    del data
    gc.collect()
    torch.cuda.empty_cache()


def load_teacher_targets(dataset: str, teacher: str, fold: int, seed: int,
                         data: dict, include_features: bool) -> tuple[dict, dict]:
    cell = cache_dir(dataset, teacher, fold, seed)
    cache_path, manifest_path = cell / 'targets.npz', cell / 'manifest.json'
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'teacher target cache missing: {cell}')
    metadata = json.loads(resolve_local_file(manifest_path).read_text())
    expected = {'dataset': dataset, 'teacher': teacher, 'seed': seed,
                'fold_zero_based': fold, 'status': 'complete',
                'input_manifest_sha256': data['manifest_sha256'],
                'input_X_sha256': data['file_records']['X.npy']['sha256']}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'{manifest_path}: target cache {key} mismatch')
    if cache_path.stat().st_size != metadata['cache_bytes']:
        raise RuntimeError(f'{cache_path}: byte count mismatch')
    with np.load(resolve_local_file(cache_path), allow_pickle=False) as saved:
        names = ('logits', 'feats', 'y', 'trial_uid') if include_features else ('logits', 'y', 'trial_uid')
        targets = {name: saved[name].copy() for name in names}
    train_idx = data['subjects'] != fold
    if (not np.array_equal(targets['y'], data['y'][train_idx])
            or not np.array_equal(targets['trial_uid'].astype(str), data['uids'][train_idx])):
        raise RuntimeError(f'{cache_path}: target UID/label order differs from student source trials')
    if targets['logits'].shape != (int(train_idx.sum()), metadata['num_classes']):
        raise RuntimeError(f'{cache_path}: teacher logits shape mismatch')
    if include_features and targets['feats'].shape != (int(train_idx.sum()), metadata['feature_dim']):
        raise RuntimeError(f'{cache_path}: teacher feature shape mismatch')
    return targets, {**metadata, 'manifest_sha256': sha256(manifest_path)}


def result_cell(dataset: str, teacher: str, student: str, stage: str,
                fold: int, seed: int) -> Path:
    return (RESULT_ROOT / stage / dataset / f'{teacher}__{student}'
            / f'subject_{fold + 1:02d}' / f'seed_{seed}')


def _set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_task(dataset: str, teacher: str, student: str, stage_name: str,
               seed: int, gpu: int, plan: dict) -> None:
    if gpu == 0:
        raise RuntimeError('GPU 0 is prohibited for this experiment')
    if stage_name not in STAGES or teacher not in TEACHERS or student not in STUDENTS:
        raise ValueError('invalid stage, teacher, or student')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    teacher_data = load_input(dataset, teacher, verify_hashes=False)
    student_data = load_input(dataset, student, verify_hashes=False)
    validate_shared_trials(dataset, teacher_data, student_data)
    stage = plan['stages'][stage_name]
    include_features = float(stage['lam_feat']) > 0
    for fold in range(SUBJECTS[dataset]):
        tr_mask = student_data['subjects'] != fold
        te_mask = ~tr_mask
        teacher_targets, target_meta = load_teacher_targets(
            dataset, teacher, fold, seed, teacher_data, include_features)
        cfg, baseline_record, baseline_metrics = validate_baseline(
            dataset, student, fold, seed, student_data, verify_checkpoint_hash=False)
        if (str(cfg.get('optimizer', 'adamw')).lower() != str(plan['student_training']['optimizer']).lower()
                or not np.isclose(float(cfg['lr']), float(plan['student_training']['learning_rate']))
                or int(cfg['batch_size']) != int(plan['student_training']['batch_size'][student])
                or not np.isclose(float(cfg['weight_decay']),
                                  float(plan['student_training']['weight_decay'][student]))
                or int(cfg['epochs']) != int(plan['student_training']['epochs'])):
            raise RuntimeError(f'{dataset}/{student}: baseline training config differs from KD plan')
        if not np.array_equal(teacher_targets['y'], student_data['y'][tr_mask]):
            raise RuntimeError('teacher labels and student training labels do not align')
        cell = require_external_output(result_cell(dataset, teacher, student, stage_name, fold, seed))
        cell.mkdir(parents=True, exist_ok=True)
        with (cell / '.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _set_seed(seed)
            adapter = get_adapter(student, device=device, **cfg)
            model = adapter.build(len(student_data['manifest']['class_mapping']))
            initial_hash = core._state_hash(model)
            projection = core._projector(
                student, model, int(target_meta['feature_dim']), seed,
                adapter.device, include_features)
            run_config = {
                'protocol': plan['protocol_id'], 'source_protocol': plan['source_protocol'],
                'config_sha256': plan['_config_sha256'],
                'runner_sha256': sha256(Path(__file__)),
                'core_runner_sha256': sha256(ROOT / 'experiments/distill/run_loso_distillation.py'),
                'dataset': dataset, 'teacher': teacher, 'student': student,
                'stage': stage_name, 'stage_parameters': stage,
                'loss_parameters': plan['loss'], 'seed': seed,
                'test_subject': fold + 1, 'model_config': cfg,
                'source_variant': student_data['manifest']['source_variant'],
                'selected_session': student_data['manifest']['selected_session'],
                'source_manifest_sha256': student_data['manifest']['source_manifest_sha256'],
                'input_manifest_sha256': student_data['manifest_sha256'],
                'input_X_sha256': student_data['file_records']['X.npy']['sha256'],
                'teacher_input_manifest_sha256': teacher_data['manifest_sha256'],
                'teacher_target_cache': str(cache_dir(dataset, teacher, fold, seed) / 'targets.npz'),
                'teacher_target_manifest_sha256': target_meta['manifest_sha256'],
                'teacher_checkpoint_sha256': target_meta['teacher_checkpoint_sha256'],
                'student_baseline': baseline_record,
                'student_baseline_metrics': baseline_metrics,
                'initialization': 'fresh_random_seeded; baseline final weights not loaded',
                'initial_state_sha256': initial_hash,
                'train_uid_sha256': array_hash(student_data['uids'][tr_mask]),
                'test_uid_sha256': array_hash(student_data['uids'][te_mask]),
                'selection_policy': 'fixed_final_epoch_no_validation',
            }
            fingerprint = json_hash(run_config)
            if core._completed(cell, fingerprint, student_data['uids'][te_mask],
                               student_data['y'][te_mask], include_features):
                print(f'[skip] {stage_name} {dataset} {teacher}->{student} '
                      f'S{fold+1} seed={seed}', flush=True)
                del adapter, model, projection
                continue
            x_train = np.asarray(student_data['x'][tr_mask], dtype=np.float32)
            x_test = np.asarray(student_data['x'][te_mask], dtype=np.float32)
            x_tensor = adapter.preprocess(x_train)
            print(f'[start] {stage_name} {dataset} {teacher}->{student} '
                  f'S{fold+1} seed={seed} train={int(tr_mask.sum())} '
                  f'teacher_dim={target_meta["feature_dim"]} gpu={gpu}', flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            history, elapsed = core._fit(
                adapter, model, projection, x_tensor, student_data['y'][tr_mask],
                teacher_targets, cfg, stage, float(plan['loss']['temperature']),
                seed, cell, fingerprint, initial_hash,
                int(plan['execution']['checkpoint_every_epochs']))
            test_features, logits = adapter.infer(model, x_test)
            metrics, probs, pred, cm = metric_protocol._metrics(student_data['y'][te_mask], logits)
            metrics.update(dataset=dataset, model=student, teacher=teacher,
                           student=student, recipe=stage_name, seed=seed,
                           test_subject=fold + 1, n_train=int(tr_mask.sum()),
                           n_test=int(te_mask.sum()), elapsed_sec=round(elapsed, 2),
                           artifact_origin='distilled_from_source_refreshed_teacher')
            for metric in METRICS:
                metrics[f'baseline_{metric}'] = float(baseline_metrics[metric])
                metrics[f'delta_{metric}'] = float(metrics[metric]) - float(baseline_metrics[metric])
            result = {'y': student_data['y'][te_mask], 'pred': pred, 'probs': probs,
                      'logits': logits, 'feats': test_features,
                      'sample_uid': student_data['uids'][te_mask],
                      'local_sample_uid': np.asarray([int(r['source_row']) for r, keep in zip(student_data['rows'], te_mask) if keep], dtype=np.int64),
                      'confusion_matrix': cm,
                      'metrics_json': np.asarray(json.dumps(metrics, sort_keys=True, allow_nan=True))}
            manifest = {**run_config, 'run_fingerprint': fingerprint,
                        'status': 'complete', 'completed_unix': time.time(),
                        'environment_snapshot': metric_protocol._environment_snapshot(device),
                        'teacher_feature_dim': int(target_meta['feature_dim']),
                        'student_feature_dim': int(getattr(model, core.FEATURE_ATTR[student])),
                        'student_parameters': sum(p.numel() for p in model.parameters()),
                        'projector_parameters': sum(p.numel() for p in projection.parameters()) if projection is not None else 0,
                        'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
                        'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
                        'n_train': int(tr_mask.sum()), 'n_test': int(te_mask.sum()),
                        'label_values': sorted(student_data['manifest']['class_mapping'].values()),
                        'test_metrics': metrics}
            if projection is not None:
                core._torch_write(cell / 'projector.pt', projection.state_dict())
            cell_paths = (cell / 'result.npz', cell / 'model.pt',
                          cell / 'manifest.json', cell / 'train_history.csv')
            metric_protocol._save_cell(*cell_paths, model, result, manifest, history)
            core._json_write(cell / 'progress.json', {
                'status': 'complete', 'completed_epoch': int(cfg['epochs']),
                'run_fingerprint': fingerprint, 'updated_unix': time.time(),
                'test_metrics': metrics})
            (cell / 'resume.pt').unlink(missing_ok=True)
            print(f'[done] {stage_name} {dataset} {teacher}->{student} '
                  f'S{fold+1} seed={seed} acc={metrics["accuracy"]:.4f} '
                  f'delta={metrics["delta_accuracy"]:+.4f} elapsed={elapsed:.1f}s', flush=True)
            del adapter, model, projection, x_train, x_test, x_tensor, result
            gc.collect()
            torch.cuda.empty_cache()
    del teacher_data, student_data
    gc.collect()


def smoke_task(dataset: str, teacher: str, student: str, seed: int,
               gpu: int, plan: dict) -> None:
    """Run one real minibatch through all four losses and the feature projector."""
    if gpu == 0 or not torch.cuda.is_available():
        raise RuntimeError('smoke requires CUDA on an allowed nonzero GPU')
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    teacher_data = load_input(dataset, teacher, verify_hashes=False)
    student_data = load_input(dataset, student, verify_hashes=False)
    validate_shared_trials(dataset, teacher_data, student_data)
    fold = 0
    cache, metadata = load_teacher_targets(dataset, teacher, fold, seed, teacher_data, True)
    cfg, _, _ = validate_baseline(dataset, student, fold, seed, student_data)
    _set_seed(seed)
    device = torch.device(f'cuda:{gpu}')
    adapter = get_adapter(student, device=device, **cfg)
    model = adapter.build(len(student_data['manifest']['class_mapping']))
    projector = core._projector(student, model, int(metadata['feature_dim']), seed,
                                adapter.device, True)
    train_idx = student_data['subjects'] != fold
    batch_size = int(cfg['batch_size'])
    xb = adapter.preprocess(np.asarray(student_data['x'][train_idx][:batch_size], dtype=np.float32)).to(device)
    labels = torch.as_tensor(student_data['y'][train_idx][:batch_size], dtype=torch.long, device=device)
    teacher_logits = torch.as_tensor(cache['logits'][:batch_size], device=device)
    teacher_features = torch.as_tensor(cache['feats'][:batch_size], device=device)
    for name in STAGES:
        model.zero_grad(set_to_none=True)
        projector.zero_grad(set_to_none=True)
        features, logits = adapter.forward(model, xb)
        total, _, kd, feature, active = core._losses(
            logits, features, labels, teacher_logits, teacher_features, projector,
            plan['stages'][name], float(plan['loss']['temperature']),
            int(plan['stages'][name]['distill_warmup_epochs']))
        if not torch.isfinite(total) or logits.shape != (len(labels), len(student_data['manifest']['class_mapping'])):
            raise RuntimeError(f'{name}: invalid logits or loss')
        total.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError(f'{name}: no finite student gradient')
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in projector.parameters()):
            raise RuntimeError(f'{name}: no finite projector gradient')
        if name.startswith('warmup10') and active:
            raise RuntimeError('warmup epoch 1 must be CE only')
        print(f'[smoke-ok] {name} {dataset} {teacher}->{student} '
              f'input={tuple(xb.shape)} student_dim={features.shape[1]} '
              f'teacher_dim={teacher_features.shape[1]} '
              f'peak_mib={torch.cuda.max_memory_allocated(device)/2**20:.1f}', flush=True)
    del adapter, model, projector, xb, labels, teacher_logits, teacher_features
    gc.collect()
    torch.cuda.empty_cache()


def summarize(plan: dict) -> dict:
    rows, latest = [], None
    counts = {}
    subjects_by_dataset = SUBJECTS
    runner_hash = sha256(Path(__file__))
    for stage in STAGES:
        completed = 0
        for dataset in DATASETS:
            for teacher in TEACHERS:
                for student in STUDENTS:
                    for fold in range(SUBJECTS[dataset]):
                        for seed in (666, 667, 668):
                            cell = result_cell(dataset, teacher, student, stage, fold, seed)
                            progress = cell / 'progress.json'
                            if progress.is_file():
                                details = json.loads(resolve_local_file(progress).read_text())
                                updated = progress.stat().st_mtime
                                if latest is None or updated > latest['mtime']:
                                    latest = {'path': str(progress), 'mtime': updated, 'details': details}
                            manifest_path, result_path = cell / 'manifest.json', cell / 'result.npz'
                            history_path = cell / 'train_history.csv'
                            if not all(p.is_file() for p in (manifest_path, result_path, history_path, cell / 'model.pt')):
                                continue
                            manifest = json.loads(resolve_local_file(manifest_path).read_text())
                            if (manifest.get('status') != 'complete'
                                    or manifest.get('protocol') != plan['protocol_id']
                                    or manifest.get('stage') != stage
                                    or manifest.get('dataset') != dataset
                                    or manifest.get('teacher') != teacher
                                    or manifest.get('student') != student
                                    or manifest.get('test_subject') != fold + 1
                                    or manifest.get('seed') != seed
                                    or manifest.get('runner_sha256') != runner_hash):
                                raise RuntimeError(f'inconsistent distillation output: {manifest_path}')
                            if float(manifest.get('stage_parameters', {}).get('lam_feat', 0)) > 0 and not (cell / 'projector.pt').is_file():
                                raise RuntimeError(f'feature stage is missing its projector: {cell}')
                            with history_path.open(newline='') as stream:
                                history = list(csv.DictReader(stream))
                            if len(history) != 100 or int(history[-1]['epoch']) != 100:
                                raise RuntimeError(f'incomplete 100-epoch history: {history_path}')
                            with np.load(resolve_local_file(result_path), allow_pickle=False) as saved:
                                metrics = json.loads(str(saved['metrics_json'].item()))
                                if not np.isfinite(saved['logits']).all():
                                    raise RuntimeError(f'non-finite logits: {result_path}')
                            metrics.update(stage=stage, teacher=teacher, student=student,
                                           dataset=dataset, test_subject=fold + 1, seed=seed)
                            rows.append(metrics)
                            completed += 1
        expected = sum(SUBJECTS[d] for d in DATASETS) * len(TEACHERS) * len(STUDENTS) * 3
        counts[stage] = {'completed': completed, 'expected': expected,
                         'complete': completed == expected}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    columns = ['stage', 'dataset', 'teacher', 'student', 'test_subject', 'seed',
               'n_train', 'n_test', *METRICS,
               *(f'baseline_{m}' for m in METRICS),
               *(f'delta_{m}' for m in METRICS), 'elapsed_sec']
    csv_path = require_external_output(RESULT_ROOT / 'all_results.csv')
    temp_csv = csv_path.with_suffix('.csv.tmp')
    with temp_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    os.replace(temp_csv, csv_path)
    groups = []
    for stage in STAGES:
        for dataset in DATASETS:
            for teacher in TEACHERS:
                for student in STUDENTS:
                    selected = [row for row in rows if row['stage'] == stage and row['dataset'] == dataset
                                and row['teacher'] == teacher and row['student'] == student]
                    expected = SUBJECTS[dataset] * 3
                    group = {'stage': stage, 'dataset': dataset, 'teacher': teacher,
                             'student': student, 'completed': len(selected),
                             'expected': expected, 'complete': len(selected) == expected}
                    if selected:
                        for metric in (*METRICS, *(f'delta_{m}' for m in METRICS)):
                            by_seed = [float(np.nanmean([r[metric] for r in selected if r['seed'] == seed]))
                                       for seed in (666, 667, 668)
                                       if any(r['seed'] == seed for r in selected)]
                            if by_seed:
                                group[f'{metric}_mean'] = float(np.mean(by_seed))
                                group[f'{metric}_std_seeds'] = float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else 0.0
                                group[f'{metric}_per_seed_subject_mean'] = {
                                    str(seed): float(np.nanmean([r[metric] for r in selected if r['seed'] == seed]))
                                    for seed in (666, 667, 668) if any(r['seed'] == seed for r in selected)}
                    groups.append(group)
    summary = {'protocol': plan['protocol_id'], 'source_protocol': plan['source_protocol'],
               'stage_counts': counts, 'groups': groups, 'latest_progress': latest,
               'completed_training_fold_seed_units': len(rows),
               'expected_training_fold_seed_units': 1512}
    summary_path = require_external_output(RESULT_ROOT / 'summary.json')
    tmp = summary_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + '\n')
    os.replace(tmp, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('validate', 'prepare-cache', 'train', 'smoke', 'summarize'), required=True)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--teacher', choices=TEACHERS)
    parser.add_argument('--student', choices=STUDENTS)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    plan = load_plan()
    if args.mode == 'validate':
        result = validate_prerequisites(verify_hashes=True)
        out = require_external_output(RESULT_ROOT / 'preflight' / 'prerequisite_inventory.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(f'[validate-ok] {len(result["inventory"])} model input caches and all LOSO baseline cells verified', flush=True)
    elif args.mode == 'prepare-cache':
        if not all((args.dataset, args.teacher, args.seed is not None, args.gpu is not None)):
            parser.error('prepare-cache requires --dataset --teacher --seed --gpu')
        prepare_teacher_task(args.dataset, args.teacher, args.seed, args.gpu)
    elif args.mode == 'train':
        if not all((args.dataset, args.teacher, args.student, args.stage, args.seed is not None, args.gpu is not None)):
            parser.error('train requires --dataset --teacher --student --stage --seed --gpu')
        train_task(args.dataset, args.teacher, args.student, args.stage, args.seed, args.gpu, plan)
    elif args.mode == 'smoke':
        if not all((args.dataset, args.teacher, args.student, args.seed is not None, args.gpu is not None)):
            parser.error('smoke requires --dataset --teacher --student --seed --gpu')
        smoke_task(args.dataset, args.teacher, args.student, args.seed, args.gpu, plan)
    else:
        summary = summarize(plan)
        print(json.dumps(summary['stage_counts'], indent=2), flush=True)


if __name__ == '__main__':
    main()
