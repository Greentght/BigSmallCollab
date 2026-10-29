"""Queue and run the CBraMod channel-wise feature alignment rerun.

Few-shot reads the validated wideband input/teacher caches from the current
structure-KD pipeline and trains all five student arms. LOSO reuses the
existing source-fold targets and scratch baselines, training only the two
channel-feature arms. Training tensors and checkpoints stay under /data1;
user-facing metrics stay in the checkout's results/ directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

# Set backend limits before loading NumPy, SciPy, or PyTorch.
for _thread_var in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'TORCH_NUM_THREADS', 'BIGSMALLCOLLAB_CPU_THREADS'):
    os.environ[_thread_var] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
# Configure PyTorch inter-op before importing NumPy, SciPy, sklearn, or model
# adapters; several of those can start parallel work during import.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # A parent process may have configured PyTorch already.
    pass

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

import config as model_config
from collab.cbramod_channel_feature import (
    channel_cosine_loss, mean_pool_temporal_patches,
)
from collab.seed import set_seed
from data import loso_source
from experiments.distill import run_fewshot_structure_kd as fewshot
from experiments.distill import run_loso_distillation as loso_runner
from experiments.distill import loso_teacher_cache as loso_teachers
from experiments.distill.loso_teacher_prepare import load_verified_alexmi_spec
from experiments.finetune import run_loso_five_datasets as loso_protocol
from experiments.storage import (
    DATA_CACHE_ROOT, REPORTS_ROOT, require_external_output, require_report_output,
    resolve_local_file,
)
from models import get_adapter


DEFAULT_CONFIG = PROJECT_ROOT / 'configs/experiments/cbramod_channel_feature_alignment.yaml'
FEWSHOT_ARMS = ('ce', 'kd', 'kd_feature', 'warmup10_kd', 'warmup10_kd_feature')
LOSO_ARMS = ('kd_feature', 'warmup10_kd_feature')
DISPLAY_ARMS = {
    'ce': 'CE',
    'kd': 'KD',
    'kd_feature': 'KD + CBraMod channel feature',
    'warmup10_kd': 'warmup10 KD',
    'warmup10_kd_feature': 'warmup10 KD + CBraMod channel feature',
}
CODE_FILES = (
    'experiments/distill/run_cbramod_channel_feature.py',
    'experiments/distill/run_fewshot_structure_kd.py',
    'experiments/distill/run_loso_distillation.py',
    'experiments/distill/loso_teacher_cache.py',
    'experiments/distill/loso_teacher_prepare.py',
    'experiments/finetune/run_loso_five_datasets.py',
    'data/loso_source.py', 'data/split.py', 'collab/seed.py',
    'collab/cbramod_channel_feature.py',
    'models/base.py', 'models/cbramod/adapter.py',
    'models/cbramod/cbramod.py', 'models/cbramod/criss_cross_transformer.py',
    'models/ifnet/adapter.py', 'models/ifnet/ifnet.py',
    'models/eegnet/adapter.py', 'models/eegnet/residual_eegnet.py',
    'models/adfcnn/adapter.py', 'models/adfcnn/adfcnn.py',
    'configs/experiments/cbramod_channel_feature_alignment.yaml',
    'configs/experiments/fewshot/structure_kd.yaml',
    'configs/models/cbramod.yaml', 'configs/models/ifnet.yaml',
    'configs/models/eegnet.yaml', 'configs/models/adfcnn.yaml',
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    resolved = Path(resolve_local_file(path))
    digest = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _sha256_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    return {name: _sha256_file(PROJECT_ROOT / name) for name in CODE_FILES}


def _atomic_json(path: str | Path, value: Any, *, report=False) -> None:
    path = Path(require_report_output(path) if report else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _atomic_csv(path: str | Path, rows: list[dict[str, Any]], *, report=True) -> None:
    path = Path(require_report_output(path) if report else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: str | Path, *, external=True, **arrays: Any) -> None:
    path = Path(require_external_output(path) if external else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected a YAML mapping in {path}')
    if value.get('teacher') != 'cbramod':
        raise ValueError('this rerun is frozen to CBraMod as the teacher')
    if tuple(value.get('datasets', ())) != (
            'BNCI2014004', 'BNCI2014001', 'BNCI2014001-4', 'BNCI2015001', 'AlexMI'):
        raise ValueError('the dataset list/order differs from the accepted five-dataset plan')
    if tuple(value.get('students', ())) != ('ifnet', 'eegnet', 'adfcnn'):
        raise ValueError('the student list/order differs from the accepted plan')
    if tuple(value.get('seeds', ())) != (666, 667, 668):
        raise ValueError('the run requires seeds 666, 667, and 668')
    alignment = value['alignment']
    if (alignment.get('temporal_patch_pool') != 'mean'
            or alignment.get('projector') != 'one_shared_linear_student_dim_to_200'
            or alignment.get('loss') != 'mean_over_batch_and_eeg_channels_of_1_minus_cosine'):
        raise ValueError('unsupported CBraMod channel-alignment definition')
    if tuple(value['fewshot']['train_all_arms']) != FEWSHOT_ARMS:
        raise ValueError('few-shot must train exactly the five agreed arms')
    if tuple(value['loso']['train_arms']) != LOSO_ARMS:
        raise ValueError('LOSO must train only the two new feature arms')
    return value


def _set_runtime_threads() -> None:
    torch.set_num_threads(1)


def _device(gpu: int | None) -> torch.device:
    if gpu is None:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError(f'GPU {gpu} was assigned but CUDA is unavailable')
    torch.cuda.set_device(int(gpu))
    return torch.device(f'cuda:{int(gpu)}')


def _progress(path: str | Path, status: str, records: set[str], total: int,
              *, error: str | None = None) -> None:
    payload = {'status': status, 'records': sorted(records),
               'completed_cells': len(records), 'total_cells': int(total),
               'updated_at': datetime.now(timezone.utc).isoformat()}
    if error:
        payload['error'] = error
    _atomic_json(path, payload)


def _initial_student(adapter, num_classes: int, seed: int):
    set_seed(int(seed))
    model = adapter.build(int(num_classes))
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    digest = _sha256_state(state)
    del model
    return state, digest


def _projection(adapter, model, student: str, seed: int, device: torch.device):
    feature_attr = {'ifnet': 'feat_dim', 'eegnet': 'flatten_size',
                    'adfcnn': 'feat_size'}[student]
    feature_dim = int(getattr(model, feature_attr))
    feature_device = ([device.index] if device.type == 'cuda' else [])
    projection_seed = int(seed) + 7919
    with torch.random.fork_rng(devices=feature_device):
        torch.manual_seed(projection_seed)
        projection = nn.Linear(feature_dim, 200).to(device)
    return projection, feature_dim, projection_seed


def _train_fewshot_cell(adapter, student: str, num_classes: int,
                        X_cal: np.ndarray, y_cal: np.ndarray,
                        teacher_logits: np.ndarray, teacher_features: np.ndarray,
                        X_test: np.ndarray, seed: int, arm: str,
                        training: dict[str, Any], device: torch.device,
                        initial_state: dict[str, torch.Tensor], initial_hash: str):
    uses_kd = arm in {'kd', 'kd_feature', 'warmup10_kd', 'warmup10_kd_feature'}
    uses_feature = arm in {'kd_feature', 'warmup10_kd_feature'}
    warmup = 10 if arm in {'warmup10_kd', 'warmup10_kd_feature'} else 0
    set_seed(int(seed))
    model = adapter.build(int(num_classes))
    missing, unexpected = model.load_state_dict(initial_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'student initial state mismatch: missing={missing}, unexpected={unexpected}')
    projection = None
    if uses_feature:
        projection, feature_dim, projection_seed = _projection(
            adapter, model, student, seed, device)
    else:
        feature_dim, projection_seed = int(getattr(
            model, {'ifnet': 'feat_dim', 'eegnet': 'flatten_size',
                    'adfcnn': 'feat_size'}[student])), None

    x_tensor = torch.as_tensor(np.asarray(X_cal), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y_cal), dtype=torch.long)
    tl_tensor = torch.as_tensor(np.asarray(teacher_logits), dtype=torch.float32)
    tf_tensor = torch.as_tensor(np.asarray(teacher_features), dtype=torch.float32)
    sample_ids = torch.arange(len(y_tensor), dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor, tl_tensor, tf_tensor, sample_ids),
        batch_size=int(training['batch_size']), shuffle=True, drop_last=False,
        num_workers=0, generator=generator)
    parameters = list(model.parameters())
    if projection is not None:
        parameters.extend(projection.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training['lr']),
        weight_decay=float(training['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training['epochs']))
    temperature = 2.0
    lambda_kd = 0.5 if uses_kd else 0.0
    lambda_feature = 1.0 if uses_feature else 0.0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    batch_hashes: list[str] = []

    for epoch in range(int(training['epochs'])):
        model.train()
        if projection is not None:
            projection.train()
        total_sum = ce_sum = kd_sum = feature_sum = 0.0
        correct = seen = 0
        order = hashlib.sha256()
        lr_used = float(optimizer.param_groups[0]['lr'])
        for xb, yb, teacher_lg, teacher_f, indices in loader:
            order.update(indices.numpy().tobytes())
            xb, yb = xb.to(device), yb.to(device)
            teacher_lg, teacher_f = teacher_lg.to(device), teacher_f.to(device)
            feature, logits = adapter.forward(model, xb)
            ce_loss = F.cross_entropy(logits, yb)
            kd_loss = ce_loss.new_zeros(())
            feature_loss = ce_loss.new_zeros(())
            active = epoch >= warmup
            loss = ce_loss
            if active and uses_kd:
                kd_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_lg.detach() / temperature, dim=1),
                    reduction='batchmean') * temperature ** 2
                loss = loss + lambda_kd * kd_loss
            if active and uses_feature:
                feature_loss = channel_cosine_loss(
                    projection(feature), teacher_f.detach())
                loss = loss + lambda_feature * feature_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and
                   not bool(torch.isfinite(parameter.grad).all().item())
                   for parameter in parameters):
                raise FloatingPointError(f'non-finite gradient at epoch {epoch + 1}')
            optimizer.step()
            count = int(yb.shape[0])
            seen += count
            total_sum += float(loss.detach()) * count
            ce_sum += float(ce_loss.detach()) * count
            kd_sum += float(kd_loss.detach()) * count
            feature_sum += float(feature_loss.detach()) * count
            correct += int((logits.detach().argmax(1) == yb).sum())
        scheduler.step()
        batch_hashes.append(order.hexdigest())
        entry = {
            'epoch': epoch + 1,
            'total_loss': total_sum / max(seen, 1),
            'ce_loss': ce_sum / max(seen, 1),
            'kd_loss': kd_sum / max(seen, 1),
            'feature_loss': feature_sum / max(seen, 1),
            'train_accuracy_pct': 100.0 * correct / max(seen, 1),
            'distillation_active': bool(active),
            'lr_used': lr_used,
            'lr_next': float(optimizer.param_groups[0]['lr']),
            'batch_order_sha256': order.hexdigest(),
        }
        history.append(entry)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(training['epochs']):
            print(f'[fewshot] {arm} epoch={epoch + 1}/{training["epochs"]} '
                  f'loss={entry["total_loss"]:.5f} active={active}', flush=True)

    test_features, test_logits = adapter.infer(model, X_test)
    predictions = np.asarray(test_logits).argmax(1).astype(np.int64)
    final_state_hash = _sha256_state(model.state_dict())
    projector_state = ({k: v.detach().cpu().clone()
                        for k, v in projection.state_dict().items()}
                       if projection is not None else None)
    output = {
        'predictions': predictions,
        'logits': np.asarray(test_logits, dtype=np.float32),
        'features': np.asarray(test_features, dtype=np.float32),
        'history': history,
        'projector_state': projector_state,
        'student_feature_dim': feature_dim,
        'projector_seed': projection_seed,
        'initial_state_sha256': initial_hash,
        'final_state_sha256': final_state_hash,
        'batch_order_sha256': _sha256_bytes('\n'.join(batch_hashes).encode()),
        'runtime_seconds': time.monotonic() - started,
        'lambda_kd': lambda_kd,
        'lambda_feature': lambda_feature,
        'temperature': temperature if uses_kd else '',
        'warmup_epochs': warmup,
    }
    del model, projection, optimizer, scheduler, loader, x_tensor
    return output


def _ensure_fewshot_teacher_train_test_features(
        resolved: dict[str, Any], dataset: str, subject: int, seed: int,
        raw: dict[str, Any], inputs: dict[str, dict[str, np.ndarray]],
        input_manifests: dict[str, dict[str, Any]],
        teacher_targets: dict[str, np.ndarray], teacher_manifest: dict[str, Any],
        device: torch.device) -> tuple[dict[str, np.ndarray], Path]:
    """Export held-out teacher features from the exact cached fine-tuned state."""
    train_uid = np.asarray(inputs['cbramod']['sample_uid_cal'], dtype=np.int64)
    test_uid = np.asarray(inputs['cbramod']['sample_uid_test'], dtype=np.int64)
    checkpoint_sha = str(teacher_manifest.get('teacher_checkpoint_sha256', ''))
    if len(checkpoint_sha) != 64:
        raise RuntimeError('few-shot teacher target manifest lacks checkpoint SHA-256')
    identity = {
        'schema_version': 1, 'protocol': 'fewshot', 'dataset': dataset,
        'teacher': 'cbramod', 'subject_index_zero_based': int(subject),
        'seed': int(seed), 'teacher_checkpoint_sha256': checkpoint_sha,
        'teacher_target_manifest_sha256': _sha256_file(
            teacher_manifest['manifest_path']),
        'teacher_input_manifest_sha256': _sha256_file(
            input_manifests['cbramod']['manifest_path']),
        'train_uid_sha256': _sha256_array(train_uid),
        'test_uid_sha256': _sha256_array(test_uid),
        'feature_shape': [int(raw['x_cal_raw'].shape[1]), 4, 200],
        'export_code_sha256': _sha256_file(__file__),
    }
    identity_sha = _sha256_bytes(_canonical(identity).encode())
    directory = Path(DATA_CACHE_ROOT) / 'artifact-fewshot' / 'teacher_feature_exports' / \
        dataset / 'cbramod' / f'subject_{int(subject)+1:02d}' / f'seed_{int(seed)}' / identity_sha[:20]
    directory = Path(require_external_output(directory))
    with fewshot._cache_lock(directory / '.lock'):
        cached, reason = fewshot._read_cache(directory, identity, 'train_test_features')
        if cached is not None:
            arrays, manifest = cached
            if (not np.array_equal(arrays['train_uid'], train_uid)
                    or not np.array_equal(arrays['test_uid'], test_uid)
                    or not np.array_equal(arrays['train_y'], teacher_targets['y'])
                    or not np.array_equal(arrays['test_y'], inputs['cbramod']['y_test'])):
                raise RuntimeError(f'cached few-shot teacher train/test UIDs differ: {directory}')
            return arrays, Path(manifest['manifest_path'])

        teacher_cfg = copy.deepcopy(resolved['teacher_configs'][f'{dataset}/cbramod'])
        teacher_cfg.update(
            dataset_name=dataset, skip_preprocess=True,
            in_channels=int(inputs['cbramod']['X_cal'].shape[1]), samples=1000)
        teacher_adapter = get_adapter('cbramod', device=str(device), **teacher_cfg)
        set_seed(int(seed))
        model = teacher_adapter.build(int(resolved['dataset_configs'][dataset]['num_classes']))
        model = teacher_adapter.finetune(
            model, inputs['cbramod']['X_cal'], inputs['cbramod']['y_cal'],
            int(resolved['dataset_configs'][dataset]['num_classes']))
        model.eval()
        train_features, train_logits = teacher_adapter.infer(
            model, inputs['cbramod']['X_cal'])
        test_features, test_logits = teacher_adapter.infer(
            model, inputs['cbramod']['X_test'])
        state_sha = fewshot._sha256_state_dict(model.state_dict())
        del model, teacher_adapter
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if state_sha != checkpoint_sha:
            raise RuntimeError(
                're-finetuned few-shot CBraMod state differs from cached teacher targets; '
                'refusing to combine train/test features from different checkpoints')
        train_features = np.asarray(train_features, dtype=np.float32)
        train_logits = np.asarray(train_logits, dtype=np.float32)
        if (not np.allclose(train_features, teacher_targets['features'], rtol=1e-6, atol=1e-7)
                or not np.allclose(train_logits, teacher_targets['logits'], rtol=1e-6, atol=1e-7)):
            raise RuntimeError('re-exported train features/logits disagree with the validated cache')
        arrays = {
            'train_features': train_features, 'train_logits': train_logits,
            'train_y': np.asarray(inputs['cbramod']['y_cal'], dtype=np.int64),
            'train_uid': train_uid,
            'test_features': np.asarray(test_features, dtype=np.float32),
            'test_logits': np.asarray(test_logits, dtype=np.float32),
            'test_y': np.asarray(inputs['cbramod']['y_test'], dtype=np.int64),
            'test_uid': test_uid,
        }
        manifest = fewshot._write_cache(directory, 'train_test_features', identity,
                                        arrays, {
            'teacher_checkpoint_sha256': state_sha,
            'target_scope': 'calibration_train_and_heldout_test',
            'feature_origin': 'raw_CBraMod_backbone_output_before_task_head',
        })
        return arrays, Path(manifest['manifest_path'])


def _load_compatible_fewshot_teacher_cache(
        resolved: dict[str, Any], dataset: str, subject: int, seed: int,
        inputs: dict[str, dict[str, np.ndarray]],
        input_manifest: dict[str, dict[str, Any]]):
    """Reuse old teacher targets only after validating their actual inputs.

    Cache identities include a broad source-code hash and the exact model-input
    manifest hash. Those hashes can change for unrelated student/MiRepNet edits
    even when the CBraMod recipe and every model-input array are unchanged. This
    narrow compatibility path accepts only the known unrelated MiRepNet-YAML
    difference, finds the exact referenced input manifest, and compares every
    cached teacher input array against the current ones before loading targets.
    """
    teacher_inputs = inputs['cbramod']
    current_input_manifest = input_manifest['cbramod']
    current_input_identity = current_input_manifest['identity']
    teacher_cfg = copy.deepcopy(
        resolved['teacher_configs'][f'{dataset}/cbramod'])
    weight_path = Path(resolve_local_file(
        model_config.weight_path('cbramod'))).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f'CBraMod pretrained weights missing: {weight_path}')
    weight_sha = _sha256_file(weight_path)
    current_codes = resolved['code_sha256']
    root = Path(resolved['storage']['cache_root'])
    base = (root / 'teacher_targets' / dataset / 'cbramod' /
            f'subject_{int(subject)+1:02d}' / f'seed_{int(seed)}')
    reasons: list[str] = []

    for target_manifest_path in sorted(base.glob('*/teacher_targets.manifest.json')):
        try:
            saved_manifest = json.loads(target_manifest_path.read_text(encoding='utf-8'))
            identity = saved_manifest.get('identity', {})
            expected = {
                'dataset': dataset, 'teacher': 'cbramod',
                'subject_index_zero_based': int(subject), 'seed': int(seed),
                'protocol': 'fewshot', 'split_policy': 'fewshot_stratified_random',
                'calibration_fraction': 0.30, 'test_fraction': 0.70,
                'train_uid_sha256': current_input_identity['train_uid_sha256'],
                'test_uid_sha256': current_input_identity['test_uid_sha256'],
                'source_identity_sha256': current_input_identity['source_identity_sha256'],
                'teacher_config': teacher_cfg,
                'pretrained_weight_path': str(weight_path),
                'pretrained_weight_sha256': weight_sha,
            }
            mismatch = [key for key, value in expected.items()
                        if identity.get(key) != value]
            if mismatch:
                reasons.append(f'{target_manifest_path.parent.name}: identity fields {mismatch}')
                continue

            old_codes = identity.get('code_sha256')
            if not isinstance(old_codes, dict):
                reasons.append(f'{target_manifest_path.parent.name}: code hashes missing')
                continue
            changed_codes = [name for name, digest in old_codes.items()
                             if current_codes.get(name) != digest]
            missing_codes = [name for name in current_codes if name not in old_codes]
            # This file is deliberately excluded from the CBraMod compatibility
            # decision: it only configures the other teacher architecture.
            if (set(changed_codes) - {'configs/models/mirepnet.yaml'} or missing_codes):
                reasons.append(
                    f'{target_manifest_path.parent.name}: code hash changes '
                    f'{changed_codes}, missing={missing_codes}')
                continue

            referenced_input_sha = identity.get('model_input_manifest_sha256')
            input_base = (root / 'model_inputs' / dataset / 'cbramod' /
                          f'subject_{int(subject)+1:02d}' / f'seed_{int(seed)}')
            matched_input = None
            for candidate_path in input_base.glob('*/model_inputs.manifest.json'):
                if _sha256_file(candidate_path) == referenced_input_sha:
                    matched_input = candidate_path
                    break
            if matched_input is None:
                reasons.append(f'{target_manifest_path.parent.name}: referenced input manifest missing')
                continue
            old_input_manifest = json.loads(matched_input.read_text(encoding='utf-8'))
            old_input_identity = old_input_manifest.get('identity', {})
            for key in ('dataset', 'subject_index_zero_based', 'seed', 'protocol',
                        'train_uid_sha256', 'test_uid_sha256', 'source_identity_sha256',
                        'model', 'model_config', 'preprocessing', 'source_path',
                        'source_variant'):
                if old_input_identity.get(key) != current_input_identity.get(key):
                    raise RuntimeError(
                        f'{dataset} subject={subject+1} seed={seed}: candidate teacher '
                        f'input manifest differs in {key}: {matched_input}')
            old_inputs, input_reason = fewshot._read_cache(
                matched_input.parent, old_input_identity, 'model_inputs')
            if old_inputs is None:
                reasons.append(
                    f'{target_manifest_path.parent.name}: referenced inputs invalid: {input_reason}')
                continue
            if set(old_inputs[0]) != set(teacher_inputs) or any(
                    not np.array_equal(old_inputs[0][name], teacher_inputs[name])
                    for name in teacher_inputs):
                raise RuntimeError(
                    f'{dataset} subject={subject+1} seed={seed}: cached CBraMod inputs '
                    f'differ from the current source/preprocessing; refusing target reuse')

            saved_targets, target_reason = fewshot._read_cache(
                target_manifest_path.parent, identity, 'teacher_targets')
            saved_eval, eval_reason = fewshot._read_cache(
                target_manifest_path.parent, identity, 'teacher_eval_predictions')
            if saved_targets is None or saved_eval is None:
                reasons.append(
                    f'{target_manifest_path.parent.name}: target={target_reason}; eval={eval_reason}')
                continue
            targets, eval_arrays = saved_targets[0], saved_eval[0]
            if (not np.array_equal(targets['sample_uid'], teacher_inputs['sample_uid_cal'])
                    or not np.array_equal(targets['y'], teacher_inputs['y_cal'])
                    or not np.array_equal(eval_arrays['uid'], teacher_inputs['sample_uid_test'])
                    or not np.array_equal(eval_arrays['y_true'], teacher_inputs['y_test'])):
                raise RuntimeError(
                    f'{dataset} subject={subject+1} seed={seed}: cached CBraMod target '
                    f'UID/label order differs from the current split')
            if (targets['features'].shape != (len(teacher_inputs['y_cal']),
                                                int(teacher_inputs['X_cal'].shape[1]) * 800)
                    or targets['logits'].shape[0] != len(teacher_inputs['y_cal'])
                    or not np.isfinite(targets['features']).all()
                    or not np.isfinite(targets['logits']).all()):
                raise RuntimeError(
                    f'{dataset} subject={subject+1} seed={seed}: invalid cached CBraMod outputs')
            manifest = saved_targets[1]
            manifest['cache_action'] = 'reused_semantically_validated_legacy'
            manifest['cache_identity_validation'] = {
                'current_input_manifest_sha256': _sha256_file(
                    current_input_manifest['manifest_path']),
                'referenced_input_manifest_sha256': referenced_input_sha,
                'input_arrays_exact_match': True,
                'teacher_config_exact_match': True,
                'changed_unrelated_code_hashes': changed_codes,
            }
            print(f'[cache] reuse validated CBraMod teacher targets {dataset} '
                  f'S{subject+1:02d}/seed={seed} '
                  f'checkpoint={saved_manifest.get("teacher_checkpoint_sha256", "")[:12]}',
                  flush=True)
            return targets, eval_arrays, manifest
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - inspect the next candidate safely
            reasons.append(f'{target_manifest_path.parent.name}: {type(exc).__name__}: {exc}')
    return None, '; '.join(reasons) if reasons else 'no prior CBraMod target manifests'


def _ensure_loso_teacher_test_features(dataset: str, fold: int, seed: int,
                                       x: np.ndarray, y: np.ndarray,
                                       subject_ids: np.ndarray, uids: np.ndarray,
                                       spec: dict[str, Any], snapshot: dict[str, Any],
                                       device: torch.device) -> Path:
    test_mask = np.asarray(subject_ids) == int(fold)
    train_mask = ~test_mask
    test_uid = np.asarray(uids[test_mask], dtype=np.int64)
    test_y = np.asarray(y[test_mask], dtype=np.int64)
    if not len(test_y) or np.any(np.asarray(uids[train_mask])[:, 0] == int(fold)):
        raise ValueError(f'{dataset} subject {fold+1}: invalid LOSO source split')
    teacher_cell = loso_teachers.teacher_cell('cbramod', dataset, fold, seed, spec)
    checkpoint_path = teacher_cell / 'model.pt'
    teacher_manifest_path = teacher_cell / 'manifest.json'
    if not checkpoint_path.is_file() or not teacher_manifest_path.is_file():
        raise FileNotFoundError(f'completed CBraMod teacher checkpoint missing: {teacher_cell}')
    teacher_manifest = json.loads(resolve_local_file(teacher_manifest_path).read_text())
    if (not np.array_equal(np.asarray(teacher_manifest['test_sample_uids'], dtype=np.int64), test_uid)
            or not np.array_equal(np.asarray(teacher_manifest['train_sample_uids'], dtype=np.int64),
                                  np.asarray(uids[train_mask], dtype=np.int64))):
        raise RuntimeError(f'CBraMod teacher checkpoint uses another LOSO split: {teacher_cell}')
    checkpoint_sha = _sha256_file(checkpoint_path)
    identity = {
        'schema_version': 1, 'protocol': 'loso', 'dataset': dataset,
        'teacher': 'cbramod', 'fold_zero_based': int(fold),
        'test_subject': int(fold) + 1, 'seed': int(seed),
        'teacher_checkpoint_sha256': checkpoint_sha,
        'teacher_manifest_sha256': _sha256_file(teacher_manifest_path),
        'source_spec_sha256': snapshot['spec_sha256'],
        'test_uid_sha256': _sha256_array(test_uid),
        'test_labels_sha256': _sha256_array(test_y),
        'feature_shape': [int(x.shape[1]), 4, 200],
        'export_code_sha256': _sha256_file(__file__),
    }
    identity_sha = _sha256_bytes(_canonical(identity).encode())
    directory = Path(DATA_CACHE_ROOT) / 'artifact-loso' / 'teacher_test_features' / \
        dataset / 'cbramod' / f'seed_{seed}' / f'subject_{int(fold)+1:02d}' / identity_sha[:20]
    directory = Path(require_external_output(directory))
    with loso_teachers._lock(directory, exclusive=True):
        manifest_path = directory / 'test_features.manifest.json'
        arrays_path = directory / 'test_features.npz'
        if manifest_path.is_file() and arrays_path.is_file():
            manifest = json.loads(resolve_local_file(manifest_path).read_text())
            if manifest.get('identity') != identity:
                raise RuntimeError(f'LOSO test feature cache identity mismatch: {directory}')
            if _sha256_file(arrays_path) != manifest.get('cache_file_sha256'):
                raise RuntimeError(f'LOSO test feature cache checksum mismatch: {arrays_path}')
            with np.load(resolve_local_file(arrays_path), allow_pickle=False) as saved:
                if (not np.array_equal(saved['sample_uid'], test_uid)
                        or not np.array_equal(saved['y'], test_y)):
                    raise RuntimeError(f'LOSO test feature cache UID mismatch: {arrays_path}')
            return manifest_path

        cfg = copy.deepcopy(teacher_manifest['model_config'])
        adapter = get_adapter('cbramod', device=str(device), **cfg)
        model = adapter.build(len(spec['datasets'][dataset]['classes']))
        state = torch.load(resolve_local_file(checkpoint_path), map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.eval()
        features, logits = adapter.infer(model, np.asarray(x[test_mask], dtype=np.float32))
        features = np.asarray(features, dtype=np.float32)
        logits = np.asarray(logits, dtype=np.float32)
        channel_features = mean_pool_temporal_patches(features, int(x.shape[1]))
        arrays = {
            'features_raw': features, 'features_channel_mean': channel_features,
            'logits': logits, 'y': test_y,
            'y_pred': logits.argmax(1).astype(np.int64), 'sample_uid': test_uid,
        }
        _atomic_npz(arrays_path, **arrays)
        manifest = {
            'schema_version': 1, 'identity': identity,
            'identity_sha256': identity_sha,
            'cache_file_sha256': _sha256_file(arrays_path),
            'teacher_feature_shape_raw': list(features.shape),
            'teacher_feature_shape_channel_mean': list(channel_features.shape),
            'scope': 'heldout_test_only_no_labels_used_in_training',
        }
        _atomic_json(manifest_path, manifest)
        print(f'[teacher-test-features] exported {dataset} S{fold+1:02d} '
              f'seed={seed} checkpoint={checkpoint_sha[:12]}', flush=True)
        del model, adapter
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return manifest_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _write_fewshot_cell_report(path: Path, row: dict[str, Any]) -> None:
    existing = _read_rows(path)
    rows = [old for old in existing if old.get('cell_id') != row['cell_id']]
    rows.append(row)
    rows.sort(key=lambda value: (int(value['seed']), int(value['subject']), value['method']))
    _atomic_csv(path, rows)


def _run_fewshot(snapshot: dict[str, Any], progress_path: Path,
                 gpu: int | None, stop_file: str | None) -> int:
    cfg = _load_config(snapshot['config'])
    dataset, student = snapshot['dataset'], snapshot['student']
    seeds, subjects = snapshot['seeds'], snapshot['subjects']
    arms = cfg['fewshot']['train_all_arms']
    device = _device(gpu)
    # This frozen reference config is the same resolver used when the broadband
    # calibration/teacher caches were prepared. Its code/config identity is kept
    # intact so valid saved teacher targets can be reused as-is.
    cache_resolved = fewshot.resolve_experiment(
        fewshot.DEFAULT_CONFIG, datasets=[dataset], teachers=['cbramod'],
        students=[student], seeds=seeds, methods=['kd_proto'],
        run_id=f'channel_cache_reference_{snapshot["run_id"]}_{dataset}_{student}')
    report_dir = Path(snapshot['report_dir']) / 'fewshot' / dataset / student
    report_dir = Path(require_report_output(report_dir / 'per_subject_seed.csv')).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv = report_dir / 'per_subject_seed.csv'
    external_root = require_external_output(
        Path(cfg['fewshot']['artifact_root']) / snapshot['run_id'] /
        dataset / student)
    total = len(seeds) * len(subjects) * len(arms)
    old_progress = {}
    if progress_path.is_file():
        try:
            old_progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    records = set(old_progress.get('records', []))
    _progress(progress_path, 'running', records, total)
    source = loso_source.load_dataset(dataset, model=None)
    ds_cfg = cache_resolved['dataset_configs'][dataset]
    training_base = dict(cfg['fewshot'])
    for seed in seeds:
        for subject in subjects:
            if stop_file and Path(stop_file).exists():
                _progress(progress_path, 'paused', records, total)
                return 75
            raw, inputs, input_manifests = fewshot._prepare_model_inputs(
                cache_resolved, dataset, 'cbramod', student, int(subject), int(seed),
                source, logger=print)
            compatible = _load_compatible_fewshot_teacher_cache(
                cache_resolved, dataset, int(subject), int(seed), inputs,
                input_manifests)
            if compatible[0] is None:
                print(f'[cache] no semantically compatible CBraMod teacher cache '
                      f'{dataset} S{int(subject)+1:02d}/seed={seed}; '
                      f'will fit teacher because its required train outputs are missing '
                      f'or incompatible: {compatible[1]}', flush=True)
                targets, teacher_eval, teacher_manifest = fewshot._ensure_teacher_targets(
                    cache_resolved, dataset, 'cbramod', int(subject), int(seed), inputs,
                    input_manifests['cbramod'], int(ds_cfg['num_classes']), device,
                    logger=print)
            else:
                targets, teacher_eval, teacher_manifest = compatible
            teacher_feature_export, teacher_export_manifest_path = \
                _ensure_fewshot_teacher_train_test_features(
                    cache_resolved, dataset, int(subject), int(seed), raw, inputs,
                    input_manifests, targets, teacher_manifest, device)
            student_inputs = inputs[student]
            fewshot._check_uid_alignment(raw, student_inputs, targets)
            if not np.array_equal(teacher_eval['uid'], student_inputs['sample_uid_test']):
                raise ValueError('cached teacher evaluation UID order differs from student test split')
            channels = int(raw['x_cal_raw'].shape[1])
            channel_features = mean_pool_temporal_patches(targets['features'], channels)
            if len(channel_features) != len(targets['y']):
                raise ValueError('teacher channel feature and calibration label counts differ')
            if not np.array_equal(targets['sample_uid'], student_inputs['sample_uid_cal']):
                raise ValueError('teacher feature UIDs differ from student calibration UIDs')
            teacher_y = np.asarray(teacher_eval['y_true'], dtype=np.int64)
            teacher_pred = np.asarray(teacher_eval['y_pred'], dtype=np.int64)
            teacher_acc = float(accuracy_score(teacher_y, teacher_pred) * 100.0)
            teacher_ba = float(balanced_accuracy_score(teacher_y, teacher_pred) * 100.0)

            student_cfg = copy.deepcopy(
                cache_resolved['student_configs'][f'{dataset}/{student}'])
            training = {
                'epochs': int(training_base['epochs']),
                'lr': float(training_base['lr']),
                'batch_size': int(training_base['batch_size'][student]),
                'weight_decay': float(training_base['weight_decay'][student]),
                'optimizer': 'adamw', 'scheduler': 'cosine_per_epoch',
            }
            for key, value in training_base.get('dataset_overrides', {}).get(
                    dataset, {}).get(student, {}).items():
                training[key] = value
            student_cfg.update(epochs=training['epochs'], lr=training['lr'],
                               batch_size=training['batch_size'],
                               weight_decay=training['weight_decay'],
                               optimizer='adamw', lr_schedule='cosine_epoch_end',
                               num_workers=0, dataset_name=dataset,
                               in_channels=int(raw['x_cal_raw'].shape[1]),
                               samples=int(raw['x_cal_raw'].shape[-1]))
            adapter = fewshot._PreparedStudentAdapter(
                get_adapter(student, device=str(device), **student_cfg))
            initial_state, initial_hash = fewshot._initial_student_state(
                adapter, int(ds_cfg['num_classes']), int(seed))
            model_input_sha = _sha256_file(
                Path(input_manifests[student]['manifest_path']))
            teacher_input_sha = _sha256_file(
                Path(input_manifests['cbramod']['manifest_path']))
            target_manifest_sha = _sha256_file(
                Path(teacher_manifest['manifest_path']))
            train_uid_sha = _sha256_array(student_inputs['sample_uid_cal'])
            test_uid_sha = _sha256_array(student_inputs['sample_uid_test'])
            split_sha = _sha256_bytes((train_uid_sha + test_uid_sha).encode())

            for arm in arms:
                cell_id = f'{dataset}/{student}/subject_{int(subject)+1:02d}/seed_{seed}/{arm}'
                if cell_id in records:
                    continue
                if stop_file and Path(stop_file).exists():
                    _progress(progress_path, 'paused', records, total)
                    return 75
                cell_dir = external_root / f'subject_{int(subject)+1:02d}' / \
                    f'seed_{seed}' / arm
                cell_dir = Path(require_external_output(cell_dir))
                manifest_path = cell_dir / 'manifest.json'
                result_path = cell_dir / 'result.npz'
                uses_feature = arm in {'kd_feature', 'warmup10_kd_feature'}
                lambda_kd = 0.5 if arm != 'ce' else 0.0
                lambda_feature = 1.0 if uses_feature else 0.0
                warmup = 10 if arm.startswith('warmup10_') else 0
                identity = {
                    'protocol': 'fewshot', 'run_id': snapshot['run_id'],
                    'dataset': dataset, 'teacher': 'cbramod', 'student': student,
                    'subject_index_zero_based': int(subject), 'seed': int(seed),
                    'arm': arm, 'epochs': training['epochs'], 'lr': training['lr'],
                    'batch_size': training['batch_size'],
                    'weight_decay': training['weight_decay'],
                    'optimizer': training['optimizer'], 'scheduler': training['scheduler'],
                    'temperature': 2.0 if lambda_kd else None,
                    'lambda_kd': lambda_kd, 'lambda_feature': lambda_feature,
                    'warmup_epochs': warmup,
                    'feature_transform': 'mean_over_4_temporal_patches_per_channel',
                    'student_projection': 'one_shared_linear_to_200',
                    'source_variant': ('wide_fre_band' if dataset != 'AlexMI'
                                       else 'current_AlexMI_loader'),
                    'model_input_manifest_sha256': model_input_sha,
                    'teacher_input_manifest_sha256': teacher_input_sha,
                    'teacher_target_manifest_sha256': target_manifest_sha,
                    'teacher_train_test_feature_export_manifest_sha256': _sha256_file(
                        teacher_export_manifest_path),
                    'train_uid_sha256': train_uid_sha,
                    'test_uid_sha256': test_uid_sha,
                    'split_uid_sha256': split_sha,
                    'initial_state_sha256': initial_hash,
                    'code_sha256': snapshot['code_sha256'],
                    'config_sha256': snapshot['config_sha256'],
                }
                fingerprint = _sha256_bytes(_canonical(identity).encode())
                if manifest_path.is_file() and result_path.is_file():
                    existing = json.loads(resolve_local_file(manifest_path).read_text())
                    if existing.get('run_fingerprint') != fingerprint:
                        raise RuntimeError(f'few-shot artifact belongs to another run: {cell_dir}')
                    records.add(cell_id)
                    _progress(progress_path, 'running', records, total)
                    continue

                result = _train_fewshot_cell(
                    adapter, student, int(ds_cfg['num_classes']),
                    student_inputs['X_cal'], student_inputs['y_cal'],
                    targets['logits'], channel_features, student_inputs['X_test'],
                    int(seed), arm, training, device, initial_state, initial_hash)
                y_true = np.asarray(student_inputs['y_test'], dtype=np.int64)
                y_pred = result['predictions']
                metrics = {
                    'accuracy_pct': float(accuracy_score(y_true, y_pred) * 100.0),
                    'balanced_accuracy_pct': float(
                        balanced_accuracy_score(y_true, y_pred) * 100.0),
                    'kappa': float(cohen_kappa_score(y_true, y_pred)),
                }
                row = {
                    'cell_id': cell_id, 'dataset': dataset, 'protocol': 'fewshot',
                    'teacher': 'cbramod', 'student': student,
                    'method': arm, 'display_method': DISPLAY_ARMS[arm],
                    'subject': int(subject) + 1, 'subject_index': int(subject),
                    'seed': int(seed), 'source_variant': identity['source_variant'],
                    'n_calibration': int(len(student_inputs['y_cal'])),
                    'n_test': int(len(y_true)), 'num_classes': int(ds_cfg['num_classes']),
                    **metrics, 'teacher_accuracy_pct': teacher_acc,
                    'teacher_balanced_accuracy_pct': teacher_ba,
                    'epochs': training['epochs'], 'lr': training['lr'],
                    'batch_size': training['batch_size'],
                    'weight_decay': training['weight_decay'],
                    'optimizer': training['optimizer'], 'scheduler': training['scheduler'],
                    'temperature': result['temperature'],
                    'lambda_kd': result['lambda_kd'],
                    'lambda_feature': result['lambda_feature'],
                    'warmup_epochs': result['warmup_epochs'],
                    'teacher_feature_channels': channels,
                    'projector_output_dim': 200 if uses_feature else '',
                    'student_feature_dim': result['student_feature_dim'],
                    'projector_seed': result['projector_seed'] or '',
                    'initial_state_sha256': result['initial_state_sha256'],
                    'final_state_sha256': result['final_state_sha256'],
                    'batch_order_sha256': result['batch_order_sha256'],
                    'train_uid_sha256': train_uid_sha,
                    'test_uid_sha256': test_uid_sha,
                    'split_uid_sha256': split_sha,
                    'teacher_checkpoint_sha256': teacher_manifest.get(
                        'teacher_checkpoint_sha256', ''),
                    'teacher_target_manifest_sha256': target_manifest_sha,
                    'teacher_train_test_feature_export_manifest_sha256': _sha256_file(
                        teacher_export_manifest_path),
                    'model_input_manifest_sha256': model_input_sha,
                    'artifact_dir': str(cell_dir),
                    'runtime_seconds': round(result['runtime_seconds'], 3),
                }
                _atomic_npz(
                    result_path,
                    y_true=y_true, y_pred=y_pred,
                    logits=result['logits'], features=result['features'],
                    sample_uid=np.asarray(student_inputs['sample_uid_test'], dtype=np.int64),
                    train_uid=np.asarray(student_inputs['sample_uid_cal'], dtype=np.int64))
                history_rows = result['history']
                _atomic_csv(cell_dir / 'train_history.csv', history_rows, report=False)
                if result['projector_state'] is not None:
                    projector_path = Path(require_external_output(cell_dir / 'projector.pt'))
                    temporary = projector_path.with_name(f'.{projector_path.name}.{os.getpid()}.tmp')
                    torch.save(result['projector_state'], temporary)
                    os.replace(temporary, projector_path)
                manifest = {**identity, 'run_fingerprint': fingerprint,
                            'status': 'complete', 'metrics': metrics,
                            'teacher_accuracy_pct': teacher_acc,
                            'teacher_balanced_accuracy_pct': teacher_ba,
                            'projector_parameters': (
                                result['student_feature_dim'] * 200 + 200
                                if uses_feature else 0),
                            'test_prediction_sha256': _sha256_file(result_path)}
                _write_fewshot_cell_report(output_csv, row)
                _atomic_json(manifest_path, manifest)
                records.add(cell_id)
                _progress(progress_path, 'running', records, total)
                print(f'[fewshot-done] {dataset} {student} S{int(subject)+1:02d} '
                      f'seed={seed} arm={arm} acc={metrics["accuracy_pct"]:.2f}% '
                      f'cell={len(records)}/{total}', flush=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del raw, inputs, input_manifests, targets, teacher_eval
    _progress(progress_path, 'completed', records, total)
    return 0


def _loso_plan(config_path: Path, cfg: dict[str, Any],
               spec: dict[str, Any], run_id: str) -> dict[str, Any]:
    loso_cfg = cfg['loso']
    feature_lambda = float(loso_cfg['feature']['lambda'])
    kd_lambda = float(loso_cfg['kd']['lambda'])
    temperature = float(loso_cfg['kd']['temperature'])
    stages = {
        'logits_kd': {'lam_kd': kd_lambda, 'lam_feat': 0.0,
                      'distill_warmup_epochs': 0},
        'kd_feature': {'lam_kd': kd_lambda, 'lam_feat': feature_lambda,
                       'distill_warmup_epochs': 0},
        'warmup10_kd': {'lam_kd': kd_lambda, 'lam_feat': 0.0,
                        'distill_warmup_epochs': int(loso_cfg['warmup_epochs'])},
        'warmup10_kd_feature': {
            'lam_kd': kd_lambda, 'lam_feat': feature_lambda,
            'distill_warmup_epochs': int(loso_cfg['warmup_epochs'])},
    }
    cbramod_cfg = spec['optional_cbramod_reference_recipe']
    alex_cfg = spec['main_recipes']['cbramod']
    return {
        'spec_version': 1,
        'protocol_id': 'cbramod_channel_feature_alignment_loso_v1',
        'source_protocol': loso_protocol.PROTOCOL,
        'datasets': list(cfg['datasets']), 'seeds': list(cfg['seeds']),
        'teachers': {
            'cbramod': {
                'default_recipe': cbramod_cfg['name'],
                'recipe_overrides': {'AlexMI': alex_cfg['name']},
            },
        },
        'students': list(cfg['students']),
        'loss': {
            'ce_weight': 1.0, 'temperature': temperature,
            'feature_metric': 'cosine',
            'feature_transform': 'cbramod_channel_mean_patch_shared_student',
            'teacher_correct_mask': False,
        },
        'stages': stages,
        'student_training': {
            'parameter_source': 'configs/cbramod_channel_feature_alignment.yaml',
            'epochs': int(loso_cfg['epochs']), 'lr': float(loso_cfg['lr']),
            'optimizer': loso_cfg['optimizer'], 'scheduler': loso_cfg['scheduler'],
            'batch_size': loso_cfg['batch_size'],
            'weight_decay': loso_cfg['weight_decay'],
            'initialization': 'fresh_random_paired_seed',
        },
        'output_root': str(Path(loso_cfg['artifact_root']) / run_id),
        'execution': {
            'checkpoint_every_epochs': 5, 'torch_num_threads': 1,
            'dataloader_workers': 0,
        },
        '_file_sha256': _sha256_file(config_path),
    }


def _run_loso(snapshot: dict[str, Any], progress_path: Path,
              gpu: int | None, stop_file: str | None) -> int:
    cfg = _load_config(snapshot['config'])
    config_path = Path(snapshot['config']).resolve()
    spec, source_snapshot = load_verified_alexmi_spec(loso_protocol)
    if source_snapshot.get('spec_sha256') != snapshot['source_spec_sha256']:
        raise RuntimeError('frozen LOSO source specification changed after enqueue')
    plan = _loso_plan(config_path, cfg, spec, snapshot['run_id'])
    stage = snapshot['stage']
    dataset, student = snapshot['dataset'], snapshot['student']
    folds, seeds = snapshot['folds'], snapshot['seeds']
    loso_protocol._verify_source_files(dataset, spec, source_snapshot)

    # Prior LOSO teacher-target files were produced from an older code
    # fingerprint and may point at the checkout's historical artifact path.
    # Preserve those files and export targets from the already-fitted frozen
    # teacher checkpoints into this run's external cache instead. This is
    # inference only; it never fine-tunes or saves another teacher model.
    loso_teachers.CACHE_ROOT = Path(require_external_output(
        Path(DATA_CACHE_ROOT) / 'artifact-loso' / 'teacher_targets' /
        'cbramod_channel_feature_alignment' / snapshot['run_id']))
    loso_teachers.export(SimpleNamespace(
        teacher='cbramod', datasets=[dataset], seeds=seeds,
        folds=folds, gpu=gpu))

    source_x, source_y, subject_ids, source_uids, _, *_ = \
        loso_protocol._load_trials(dataset, spec)
    for fold in folds:
        for seed in seeds:
            _ensure_loso_teacher_test_features(
                dataset, int(fold), int(seed), source_x, source_y,
                subject_ids, source_uids, spec, source_snapshot, _device(gpu))
    del source_x, source_y, subject_ids, source_uids
    args = SimpleNamespace(
        config=config_path, teacher='cbramod', stage=stage,
        datasets=[dataset], students=[student], seeds=seeds, folds=folds)
    device = _device(gpu)
    plan['_file_sha256'] = _sha256_file(config_path)
    loso_runner._run(
        args, plan, spec, source_snapshot,
        str(device), queue_progress_path=progress_path,
        stop_file=stop_file, write_summary=False)
    saved = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if saved.get('status') == 'paused':
        return 75
    if saved.get('status') != 'completed':
        raise RuntimeError('LOSO runner returned without completing the requested folds')
    return 0


def run_snapshot(snapshot_path: str | Path, *, progress_path: str | Path,
                 gpu: int | None, stop_file: str | None = None) -> int:
    _set_runtime_threads()
    wrapper = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    snapshot = wrapper.get('snapshot', wrapper)
    if snapshot.get('kind') != 'cbramod_channel_feature_alignment':
        raise ValueError('unsupported CBraMod channel-feature queue snapshot')
    if _sha256_file(snapshot['config']) != snapshot['config_sha256']:
        raise RuntimeError('experiment YAML changed after queue submission')
    for name, expected in snapshot['code_sha256'].items():
        if _sha256_file(PROJECT_ROOT / name) != expected:
            raise RuntimeError(f'training code changed after queue submission: {name}')
    progress_path = Path(progress_path)
    try:
        if snapshot['protocol'] == 'fewshot':
            return _run_fewshot(snapshot, progress_path, gpu, stop_file)
        if snapshot['protocol'] == 'loso':
            return _run_loso(snapshot, progress_path, gpu, stop_file)
        raise ValueError(f"unsupported protocol {snapshot.get('protocol')!r}")
    except BaseException as exc:
        try:
            current = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            records = set(current.get('records', []))
            _progress(progress_path, 'failed', records,
                      int(current.get('total_cells', 0)),
                      error=f'{type(exc).__name__}: {exc}')
        except Exception:
            pass
        traceback.print_exc()
        raise


def enqueue(config_path: str | Path, queue_root: str | Path,
            run_id: str | None = None) -> str:
    config_path = Path(config_path).expanduser().resolve()
    cfg = _load_config(config_path)
    config_sha = _sha256_file(config_path)
    code_sha = _code_hashes()
    run_id = run_id or (
        f"{cfg['run_id_prefix']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    report_dir = Path(REPORTS_ROOT) / 'fewshot' / 'cbramod_channel_feature' / run_id
    report_dir = Path(require_report_output(report_dir / 'config_resolved.json')).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    spec, source_snapshot = load_verified_alexmi_spec(loso_protocol)
    resolved = {
        'name': cfg['name'], 'run_id': run_id, 'config': str(config_path),
        'config_sha256': config_sha,
        'protocols': {
            'fewshot': {
                'datasets': cfg['datasets'], 'students': cfg['students'],
                'seeds': cfg['seeds'], 'arms': cfg['fewshot']['train_all_arms'],
                'source': cfg['fewshot']['source'],
                'split': 'stratified 30/70 per subject; random_state=seed',
                'epochs': cfg['fewshot']['epochs'], 'lr': cfg['fewshot']['lr'],
                'optimizer': cfg['fewshot']['optimizer'],
                'scheduler': cfg['fewshot']['scheduler'],
                'batch_size': cfg['fewshot']['batch_size'],
                'weight_decay': cfg['fewshot']['weight_decay'],
                'kd': cfg['fewshot']['kd'], 'feature': cfg['fewshot']['feature'],
                'warmup_epochs': cfg['fewshot']['warmup_epochs'],
                'student_fits': 47 * 3 * 3 * 5,
            },
            'loso': {
                'datasets': cfg['datasets'], 'students': cfg['students'],
                'seeds': cfg['seeds'], 'arms_trained': cfg['loso']['train_arms'],
                'arms_reused': cfg['loso']['reuse_arms'],
                'source_spec_sha256': source_snapshot['spec_sha256'],
                'epochs': cfg['loso']['epochs'], 'lr': cfg['loso']['lr'],
                'batch_size': cfg['loso']['batch_size'],
                'weight_decay': cfg['loso']['weight_decay'],
                'kd': cfg['loso']['kd'], 'feature': cfg['loso']['feature'],
                'warmup_epochs': cfg['loso']['warmup_epochs'],
                'student_fits': 47 * 3 * 3 * 2,
            },
        },
        'alignment': cfg['alignment'], 'code_sha256': code_sha,
        'expected_student_fits_total': 47 * 3 * 3 * 5 + 47 * 3 * 3 * 2,
        'teacher_cache_units_fewshot': 47 * 3,
        'teacher_retraining': 'only on a strict cache miss; existing train features are reused',
        'report_dir': str(report_dir),
        'training_artifact_roots': {
            'fewshot': cfg['fewshot']['artifact_root'],
            'loso': str(Path(cfg['loso']['artifact_root']) / run_id),
        },
    }
    _atomic_json(report_dir / 'config_resolved.json', resolved, report=True)

    from experiments.distill.queue import enqueue_snapshot
    for dataset in cfg['datasets']:
        n_subjects = int(model_config.load_dataset_config(dataset)['num_subjects'])
        subjects = list(range(n_subjects))
        for student in cfg['students']:
            snapshot = {
                'schema_version': 1, 'kind': 'cbramod_channel_feature_alignment',
                'protocol': 'fewshot', 'run_id': run_id,
                'config': str(config_path), 'config_sha256': config_sha,
                'code_sha256': code_sha, 'report_dir': str(report_dir),
                'dataset': dataset, 'student': student,
                'seeds': list(cfg['seeds']), 'subjects': subjects,
                'args': {
                    'protocol': 'fewshot', 'dataset': dataset,
                    'teacher': 'cbramod', 'student': student,
                    'seeds': list(cfg['seeds']), 'methods': list(FEWSHOT_ARMS),
                    'keys': subjects,
                },
            }
            from experiments.distill.queue import enqueue_snapshot
            enqueue_snapshot(snapshot, root=queue_root,
                             argv=['run_cbramod_channel_feature.py', '--enqueue'],
                             label=f'CBraMod channel feature few-shot {dataset} {student}')

    for dataset in cfg['datasets']:
        n_folds = int(spec['datasets'][dataset]['subjects'])
        folds = list(range(n_folds))
        for student in cfg['students']:
            for arm in cfg['loso']['train_arms']:
                snapshot = {
                    'schema_version': 1, 'kind': 'cbramod_channel_feature_alignment',
                    'protocol': 'loso', 'run_id': run_id,
                    'config': str(config_path), 'config_sha256': config_sha,
                    'code_sha256': code_sha, 'report_dir': str(report_dir),
                    'source_spec': source_snapshot,
                    'source_spec_sha256': source_snapshot['spec_sha256'],
                    'dataset': dataset, 'student': student, 'stage': arm,
                    'seeds': list(cfg['seeds']), 'folds': folds,
                    'args': {
                        'protocol': 'loso', 'dataset': dataset,
                        'teacher': 'cbramod', 'student': student,
                        'seeds': list(cfg['seeds']), 'methods': [arm],
                        'keys': folds,
                    },
                }
                enqueue_snapshot(
                    snapshot, root=queue_root,
                    argv=['run_cbramod_channel_feature.py', '--enqueue'],
                    label=f'CBraMod channel feature LOSO {dataset} {student} {arm}')
    _atomic_json(report_dir / 'queue_submitted.json', {
        'run_id': run_id, 'expected_jobs': 45,
        'fewshot_jobs': 15, 'loso_jobs': 30,
        'expected_student_fits': resolved['expected_student_fits_total'],
        'queue_root': str(queue_root),
        'submitted_at': datetime.now(timezone.utc).isoformat(),
    }, report=True)
    print(f'[queue-submitted] run_id={run_id} jobs=45 '
          f'student_fits={resolved["expected_student_fits_total"]} '
          f'report={report_dir}', flush=True)
    return run_id


def _seed_mean(rows: list[dict[str, Any]], metric: str, expected_subjects: int,
               seeds: list[int]) -> tuple[str, str, int, bool]:
    seed_means = {}
    for seed in seeds:
        selected = [row for row in rows if int(row['seed']) == seed]
        by_subject = {int(row['subject']): float(row[metric]) for row in selected}
        if len(by_subject) != expected_subjects:
            return '', '', len(rows), False
        seed_means[str(seed)] = float(np.mean([by_subject[i] for i in range(1, expected_subjects + 1)]))
    values = list(seed_means.values())
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return _canonical(seed_means), f'{np.mean(values):.6f}|{std:.6f}', len(rows), True


def summarize_fewshot(report_dir: Path, cfg: dict[str, Any]) -> Path:
    rows_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for dataset in cfg['datasets']:
        for student in cfg['students']:
            path = report_dir / 'fewshot' / dataset / student / 'per_subject_seed.csv'
            for row in _read_rows(path):
                rows_by_group.setdefault((dataset, student, row['method']), []).append(row)
    summary, missing = [], []
    for dataset in cfg['datasets']:
        expected_subjects = int(model_config.load_dataset_config(dataset)['num_subjects'])
        for student in cfg['students']:
            for arm in FEWSHOT_ARMS:
                rows = rows_by_group.get((dataset, student, arm), [])
                complete = (len(rows) == expected_subjects * len(cfg['seeds']) and
                            len({(int(r['subject']), int(r['seed'])) for r in rows}) == len(rows))
                row = {
                    'protocol': 'fewshot', 'dataset': dataset, 'teacher': 'cbramod',
                    'student': student, 'method': arm,
                    'display_method': DISPLAY_ARMS[arm],
                    'n_subjects_per_seed': expected_subjects,
                    'seeds': ','.join(map(str, cfg['seeds'])),
                    'n_records': len(rows), 'expected_records': expected_subjects * len(cfg['seeds']),
                    'complete': complete,
                }
                for metric in ('accuracy_pct', 'balanced_accuracy_pct'):
                    if complete:
                        per_seed = {}
                        for seed in cfg['seeds']:
                            selected = [r for r in rows if int(r['seed']) == seed]
                            per_seed[str(seed)] = float(np.mean([float(r[metric]) for r in selected]))
                        values = list(per_seed.values())
                        row[f'{metric}_mean'] = float(np.mean(values))
                        row[f'{metric}_std_seed_means'] = float(np.std(values, ddof=1))
                        row[f'{metric}_per_seed'] = _canonical(per_seed)
                    else:
                        row[f'{metric}_mean'] = ''
                        row[f'{metric}_std_seed_means'] = ''
                        row[f'{metric}_per_seed'] = ''
                summary.append(row)
                if not complete:
                    missing.append({**row, 'missing_records': row['expected_records'] - len(rows)})
    path = report_dir / 'fewshot_summary.csv'
    _atomic_csv(path, summary)
    _atomic_csv(report_dir / 'fewshot_missing.csv', missing)
    return path


def summarize_loso(report_dir: Path, cfg: dict[str, Any],
                   source_spec: dict[str, Any]) -> Path:
    resolved = json.loads((report_dir / 'config_resolved.json').read_text())
    plan = _loso_plan(Path(resolved['config']), cfg, source_spec, resolved['run_id'])
    # The main YAML's SHA is used by the training manifests, not DEFAULT_CONFIG's.
    plan['_file_sha256'] = _sha256_file(report_dir / 'config_resolved.json')
    # Replace the hash with the original experiment YAML digest recorded at enqueue.
    plan['_file_sha256'] = resolved['config_sha256']
    spec, snapshot = load_verified_alexmi_spec(loso_protocol)
    counts, new_rows, _ = loso_runner._summarize(plan, spec)
    print('[loso-summary-scan] ' + json.dumps(counts, sort_keys=True), flush=True)
    old_path = Path('/data1/llx/BigSmallcollab/results/distill/'
                    'loso_five_settings_kd_feature_warmup10_v1/all_results.csv')
    old_rows = []
    if old_path.is_file():
        with old_path.open(newline='', encoding='utf-8') as stream:
            old_rows = list(csv.DictReader(stream))
    reused_stage_coverage = {}
    expected_reused_records = sum(
        int(spec['datasets'][dataset]['subjects']) for dataset in cfg['datasets']) * \
        len(cfg['students']) * len(cfg['seeds'])
    for old_stage in ('logits_kd', 'warmup10_kd'):
        selected = [row for row in old_rows if row.get('teacher') == 'cbramod'
                    and row.get('stage') == old_stage]
        keys = {(row['dataset'], row['student'], int(row['test_subject']),
                 int(row['seed'])) for row in selected}
        reused_stage_coverage[old_stage] = {
            'records': len(selected), 'unique_records': len(keys),
            'expected_records': expected_reused_records,
            'complete': (len(selected) == expected_reused_records
                         and len(keys) == expected_reused_records),
            'source': str(old_path),
        }

    datasets, students, seeds = cfg['datasets'], cfg['students'], cfg['seeds']
    summary, missing = [], []
    for dataset in datasets:
        expected_subjects = int(spec['datasets'][dataset]['subjects'])
        expected = expected_subjects * len(seeds)
        for student in students:
            candidate = [r for r in new_rows if r['dataset'] == dataset
                         and r['teacher'] == 'cbramod' and r['student'] == student
                         and r['stage'] == 'kd_feature']
            ce_rows = [{**r, 'subject': int(r['test_subject']),
                        'accuracy_pct': float(r['baseline_accuracy']) * 100.0}
                       for r in candidate]
            reused_sources = {
                'kd': 'logits_kd', 'warmup10_kd': 'warmup10_kd',
            }
            arms = [
                ('ce', ce_rows, 'scratch_supervised_loso_v1 baseline in new-run manifests'),
            ]
            for arm, old_stage in reused_sources.items():
                rows = [r for r in old_rows if r.get('dataset') == dataset
                        and r.get('teacher') == 'cbramod' and r.get('student') == student
                        and r.get('stage') == old_stage]
                normalized = [{**r,
                               'subject': int(r['test_subject']),
                               'accuracy_pct': float(r['accuracy']) * 100.0}
                              for r in rows]
                arms.append((arm, normalized, str(old_path)))
            for arm, source_rows, source_label in arms:
                unique = {(int(r['subject']), int(r['seed'])) for r in source_rows}
                complete = len(source_rows) == expected and len(unique) == expected
                row = {
                    'protocol': 'loso', 'dataset': dataset, 'teacher': 'cbramod',
                    'student': student, 'method': arm,
                    'display_method': DISPLAY_ARMS[arm],
                    'n_subjects_per_seed': expected_subjects,
                    'seeds': ','.join(map(str, seeds)),
                    'n_records': len(source_rows), 'expected_records': expected,
                    'complete': complete, 'source': source_label,
                    'feature_transform': ('cbramod_channel_mean_patch_shared_student'
                        if arm in LOSO_ARMS else 'reused_existing_result'),
                }
                for metric_name in ('accuracy_pct',):
                    if complete:
                        per_seed = {}
                        for seed in seeds:
                            selected = [r for r in source_rows if int(r['seed']) == seed]
                            by_subject = {int(r['subject']): float(r[metric_name])
                                          for r in selected}
                            if len(by_subject) != expected_subjects:
                                complete = False
                                break
                            per_seed[str(seed)] = float(np.mean([
                                by_subject[index] for index in range(1, expected_subjects + 1)]))
                        if complete:
                            values = list(per_seed.values())
                            row[f'{metric_name}_mean'] = float(np.mean(values))
                            row[f'{metric_name}_std_seed_means'] = float(
                                np.std(values, ddof=1))
                            row[f'{metric_name}_per_seed'] = _canonical(per_seed)
                        else:
                            row[f'{metric_name}_mean'] = ''
                            row[f'{metric_name}_std_seed_means'] = ''
                            row[f'{metric_name}_per_seed'] = ''
                    else:
                        row[f'{metric_name}_mean'] = ''
                        row[f'{metric_name}_std_seed_means'] = ''
                        row[f'{metric_name}_per_seed'] = ''
                row['complete'] = complete
                summary.append(row)
                if not complete:
                    missing.append({**row, 'missing_records': expected - len(source_rows)})

            for arm in LOSO_ARMS:
                source_rows = [r for r in new_rows if r['dataset'] == dataset
                               and r['teacher'] == 'cbramod' and r['student'] == student
                               and r['stage'] == arm]
                complete = len(source_rows) == expected and len({
                    (int(r['test_subject']), int(r['seed'])) for r in source_rows}) == expected
                row = {
                    'protocol': 'loso', 'dataset': dataset, 'teacher': 'cbramod',
                    'student': student, 'method': arm,
                    'display_method': DISPLAY_ARMS[arm],
                    'n_subjects_per_seed': expected_subjects,
                    'seeds': ','.join(map(str, seeds)),
                    'n_records': len(source_rows), 'expected_records': expected,
                    'complete': complete,
                    'source': str(Path(plan['output_root']) / arm),
                    'feature_transform': plan['loss']['feature_transform'],
                }
                if complete:
                    per_seed = {}
                    for seed in seeds:
                        selected = [r for r in source_rows if int(r['seed']) == seed]
                        per_seed[str(seed)] = float(np.mean([
                            float(r['accuracy']) * 100.0 for r in selected]))
                    values = list(per_seed.values())
                    row['accuracy_pct_mean'] = float(np.mean(values))
                    row['accuracy_pct_std_seed_means'] = float(np.std(values, ddof=1))
                    row['accuracy_pct_per_seed'] = _canonical(per_seed)
                else:
                    row['accuracy_pct_mean'] = ''
                    row['accuracy_pct_std_seed_means'] = ''
                    row['accuracy_pct_per_seed'] = ''
                summary.append(row)
                if not complete:
                    missing.append({**row, 'missing_records': expected - len(source_rows)})
    path = report_dir / 'loso_summary.csv'
    _atomic_csv(path, summary)
    _atomic_csv(report_dir / 'loso_missing.csv', missing)
    _atomic_json(report_dir / 'loso_source_manifest.json', {
        'new_channel_feature_results': str(Path(plan['output_root']) / 'all_results.csv'),
        'teacher_test_feature_exports': str(
            Path(DATA_CACHE_ROOT) / 'artifact-loso' / 'teacher_test_features'),
        'reused_kd_results': str(old_path),
        'reused_ce_results': 'scratch supervised baseline references in new-run manifests',
        'reused_warmup_kd_results': str(old_path),
        'source_spec_sha256': snapshot['spec_sha256'],
        'new_run_stage_scan': counts,
        'reused_stage_coverage': reused_stage_coverage,
    }, report=True)
    return path


def summarize_run(report_dir: str | Path, config_path: str | Path) -> dict[str, str]:
    report_dir, config_path = Path(report_dir), Path(config_path)
    cfg = _load_config(config_path)
    fewshot_path = summarize_fewshot(report_dir, cfg)
    spec, _ = load_verified_alexmi_spec(loso_protocol)
    loso_path = summarize_loso(report_dir, cfg, spec)
    return {'fewshot_summary': str(fewshot_path), 'loso_summary': str(loso_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--queue-root', default=None)
    parser.add_argument('--run-id')
    parser.add_argument('--snapshot')
    parser.add_argument('--progress')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--stop-file')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--report-dir')
    args = parser.parse_args(argv)
    _set_runtime_threads()
    if args.enqueue:
        from experiments.distill.queue import DEFAULT_ROOT
        return enqueue(args.config, args.queue_root or DEFAULT_ROOT, args.run_id)
    if args.summarize:
        if not args.report_dir:
            parser.error('--report-dir is required with --summarize')
        result = summarize_run(args.report_dir, args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.snapshot:
        if not args.progress:
            parser.error('--progress is required with --snapshot')
        return run_snapshot(args.snapshot, progress_path=args.progress,
                            gpu=args.gpu, stop_file=args.stop_file)
    parser.error('use --enqueue, --summarize, or a queue --snapshot')


if __name__ == '__main__':
    raise SystemExit(main())
