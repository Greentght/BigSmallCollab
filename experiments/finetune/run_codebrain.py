"""Run the CodeBrain big-model baseline in BigSmallCollab's few-shot protocol.

Examples (run from the repository root in the ``cbramod`` environment):

    PYTHONPATH=$PWD/.deps/codebrain:$PWD conda run -n cbramod python \
      experiments/finetune/run_codebrain.py --action validate-config \
      --run-id codebrain_yaml_rerun_20260927
    PYTHONPATH=$PWD/.deps/codebrain:$PWD conda run -n cbramod python \
      experiments/finetune/run_codebrain.py --action preflight \
      --run-id codebrain_yaml_rerun_20260927 --gpu 5
    PYTHONPATH=$PWD/.deps/codebrain:$PWD conda run -n cbramod python \
      experiments/finetune/run_codebrain.py --action smoke \
      --run-id codebrain_yaml_rerun_20260927 --gpu 5
    PYTHONPATH=$PWD/.deps/codebrain:$PWD conda run -n cbramod python \
      experiments/finetune/run_codebrain.py --action train --seeds 666 \
      --run-id codebrain_yaml_rerun_20260927 --gpu 5

The same run-id is used to resume a run, and every exported hub artifact is
namespaced under ``results/codebrain/<run-id>/artifacts/``. Then run seeds 667
and 668 with the same run-id. A changed YAML/source/data recipe needs a new
run-id. The script does not select checkpoints from test data: the final epoch
is evaluated once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

import config
import data
from collab import artifacts
from data import split as split_utils
from data.channels import (
    AlexMI_chn_names,
    BNCI2014001_chn_names,
    BNCI2014004_chn_names,
    BNCI2015001_chn_names,
)
from models.codebrain.adapter import (
    CodeBrainClassifier,
    CodeBrainInputAdapter,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    OFFICIAL_WEIGHT_SHA256,
    OFFICIAL_WEIGHT_REVISION,
    OFFICIAL_WEIGHT_URL,
    build_backbone,
    load_codebrain_backbone,
    sha256_file,
)
from experiments.storage import external_path, require_external_output, resolve_local_file


DATASETS = ('BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
CHANNELS = {
    'BNCI2014001': list(BNCI2014001_chn_names),
    'BNCI2014004': list(BNCI2014004_chn_names),
    'BNCI2015001': list(BNCI2015001_chn_names),
    'AlexMI': list(AlexMI_chn_names),
}
CLASS_NAMES = {
    'BNCI2014001': ['left_hand', 'right_hand'],
    'BNCI2014004': ['left_hand', 'right_hand'],
    'BNCI2015001': ['feet', 'right_hand'],
    'AlexMI': ['feet', 'right_hand'],
}
SESSIONS = {
    'BNCI2014001': 'sessionT',
    'BNCI2014004': 'session3',
    'BNCI2015001': 'session_A',
    'AlexMI': (
        'canonical project cache: source 3 s @ 512 Hz resampled to 750 samples '
        '@ 250 Hz, then first 1 s repeated to 1000 samples'
    ),
}
SPLIT_POLICY = split_utils.FEWSHOT_SPLIT_POLICY
REFERENCE_MODELS = ('mirepnet', 'cbramod')
ARTIFACT_MODEL = {'pretrained': 'codebrain', 'random': 'codebrain_random'}
RESULTS_ROOT = Path('/data1/llx/BigSmallcollab/results') / 'codebrain'
MODEL_YAML = Path(__file__).resolve().parents[2] / 'configs' / 'models' / 'codebrain.yaml'
DATASET_YAMLS = {
    dataset: Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / f'{dataset}.yaml'
    for dataset in DATASETS
}
SOURCE_FILES = {
    'runner': Path(__file__).resolve(),
    'adapter': Path(__file__).resolve().parents[2] / 'models' / 'codebrain' / 'adapter.py',
    'config_loader': Path(__file__).resolve().parents[2] / 'config.py',
    'split_policy': Path(__file__).resolve().parents[2] / 'data' / 'split.py',
    'dataset_loader': Path(__file__).resolve().parents[2] / 'data' / 'eeg_dataset.py',
    'artifact_hub': Path(__file__).resolve().parents[2] / 'collab' / 'artifacts.py',
    'upstream_sssm': Path(__file__).resolve().parents[2] / 'models' / 'codebrain' / 'upstream' / 'CodeBrain-main' / 'Models' / 'SSSM.py',
    'upstream_sgconv': Path(__file__).resolve().parents[2] / 'models' / 'codebrain' / 'upstream' / 'CodeBrain-main' / 'Models' / 'SGConv.py',
    'upstream_license': Path(__file__).resolve().parents[2] / 'models' / 'codebrain' / 'upstream' / 'CodeBrain-main' / 'LICENSE',
}


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot JSON encode {type(value).__name__}')


def _write_json(path: Path, obj: Any) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write('\n')
    tmp.replace(path)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         default=_json_default).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _manifest_signature_payload(manifest: dict) -> dict:
    """Use one field set to create and verify the immutable run signature."""
    return {key: value for key, value in manifest.items()
            if key != 'run_signature_sha256'}


def _yaml_sha256(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return raw.decode('utf-8'), hashlib.sha256(raw).hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        'python': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'torch': str(torch.__version__),
        'numpy': str(np.__version__),
        'scipy': str(scipy.__version__),
    }


def _apply_runtime_backend(model_cfg: dict, device: torch.device) -> dict:
    """Apply an optional YAML backend policy before any model forward pass."""
    policy = model_cfg.get('backend')
    if policy is not None and policy.get('device') is not None:
        if policy['device'] != device.type:
            raise RuntimeError(
                f"Configured backend device {policy['device']!r} does not match "
                f'execution device {device.type!r}'
            )
    if policy is not None and device.type == 'cpu':
        torch.set_num_threads(int(policy['torch_num_threads']))
        torch.backends.mkldnn.enabled = bool(policy['mkldnn_enabled'])
        torch.use_deterministic_algorithms(bool(policy['deterministic_algorithms']))
    state = {
        'configured_policy': policy,
        'applied': bool(policy is not None and device.type == 'cpu'),
        'device_type': device.type,
        'torch_num_threads': int(torch.get_num_threads()),
        'torch_num_interop_threads': int(torch.get_num_interop_threads()),
        'mkldnn_enabled': bool(torch.backends.mkldnn.enabled),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
    }
    if (policy is not None and 'torch_num_interop_threads' in policy
            and state['torch_num_interop_threads'] != policy['torch_num_interop_threads']):
        raise RuntimeError(
            'Configured torch_num_interop_threads does not match the actual runtime: '
            f"{policy['torch_num_interop_threads']} != {state['torch_num_interop_threads']}"
        )
    return state


def _validate_model_config(dataset: str, values: dict) -> dict:
    required = (
        'lr', 'weight_decay', 'batch_size', 'dropout', 'scale_divisor',
        'min_lr', 'max_grad_norm', 'scheduler', 'label_smoothing', 'epochs',
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f'{dataset}: CodeBrain YAML missing resolved values {missing}')
    resolved = dict(values)
    for key in ('lr', 'weight_decay', 'dropout', 'scale_divisor', 'min_lr',
                'max_grad_norm', 'label_smoothing'):
        resolved[key] = float(resolved[key])
    for key in ('batch_size', 'epochs'):
        resolved[key] = int(resolved[key])
    resolved['scheduler'] = str(resolved['scheduler']).strip().lower()
    if 'encoder_lr' in resolved:
        resolved['encoder_lr'] = float(resolved['encoder_lr'])
    if 'head_lr' in resolved:
        resolved['head_lr'] = float(resolved['head_lr'])
    resolved['group_min_lr_ratio'] = float(resolved.get('group_min_lr_ratio', 0.1))
    if 'backend' in resolved:
        backend = resolved['backend']
        if not isinstance(backend, dict):
            raise ValueError(f'{dataset}: backend must be a mapping when configured')
        required_backend = (
            'torch_num_threads', 'mkldnn_enabled', 'deterministic_algorithms'
        )
        missing_backend = [key for key in required_backend if key not in backend]
        if missing_backend:
            raise ValueError(f'{dataset}: backend missing {missing_backend}')
        backend = {
            'torch_num_threads': int(backend['torch_num_threads']),
            'mkldnn_enabled': backend['mkldnn_enabled'],
            'deterministic_algorithms': backend['deterministic_algorithms'],
        }
        if backend['torch_num_threads'] <= 0:
            raise ValueError(f'{dataset}: backend torch_num_threads must be positive')
        if (type(backend['mkldnn_enabled']) is not bool
                or type(backend['deterministic_algorithms']) is not bool):
            raise ValueError(f'{dataset}: backend flags must be booleans')
        if 'device' in resolved['backend']:
            backend_device = str(resolved['backend']['device']).strip().lower()
            if backend_device != 'cpu':
                raise ValueError(
                    f"{dataset}: configured backend device must be 'cpu', "
                    f'got {backend_device!r}'
                )
            backend['device'] = backend_device
        if 'torch_num_interop_threads' in resolved['backend']:
            interop_threads = int(resolved['backend']['torch_num_interop_threads'])
            runtime_interop_threads = int(torch.get_num_interop_threads())
            if interop_threads <= 0:
                raise ValueError(
                    f'{dataset}: backend torch_num_interop_threads must be positive'
                )
            if interop_threads != runtime_interop_threads:
                raise ValueError(
                    f'{dataset}: backend torch_num_interop_threads={interop_threads} '
                    'does not match actual runtime '
                    f'{runtime_interop_threads}'
                )
            backend['torch_num_interop_threads'] = interop_threads
        resolved['backend'] = backend
    if resolved['lr'] <= 0 or resolved['weight_decay'] < 0:
        raise ValueError(f'{dataset}: invalid lr/weight_decay in CodeBrain YAML')
    if resolved['batch_size'] <= 0 or resolved['epochs'] <= 0:
        raise ValueError(f'{dataset}: batch_size and epochs must be positive')
    if not 0 <= resolved['dropout'] < 1:
        raise ValueError(f'{dataset}: dropout must be in [0, 1)')
    if resolved['scale_divisor'] <= 0 or resolved['min_lr'] < 0:
        raise ValueError(f'{dataset}: scale_divisor must be positive and min_lr nonnegative')
    if resolved['max_grad_norm'] <= 0:
        raise ValueError(f'{dataset}: max_grad_norm must be positive')
    if not 0 <= resolved['label_smoothing'] < 1:
        raise ValueError(f'{dataset}: label_smoothing must be in [0, 1)')
    if resolved['scheduler'] not in (
        'cosine_per_batch', 'cosine_per_group_to_10pct', 'none'
    ):
        raise ValueError(
            f'{dataset}: unsupported CodeBrain scheduler {resolved["scheduler"]!r}; '
            'choose cosine_per_batch, cosine_per_group_to_10pct, or none'
        )
    if resolved['scheduler'] == 'cosine_per_batch' and resolved['min_lr'] > resolved['lr']:
        raise ValueError(f'{dataset}: min_lr cannot exceed lr for cosine_per_batch')
    if resolved['scheduler'] == 'cosine_per_group_to_10pct':
        if 'encoder_lr' not in resolved or 'head_lr' not in resolved:
            raise ValueError(
                f'{dataset}: grouped cosine scheduler requires encoder_lr and head_lr'
            )
        if resolved['encoder_lr'] <= 0 or resolved['head_lr'] <= 0:
            raise ValueError(f'{dataset}: encoder_lr and head_lr must be positive')
        if not 0 < resolved['group_min_lr_ratio'] <= 1:
            raise ValueError(f'{dataset}: group_min_lr_ratio must be in (0, 1]')
    return resolved


def _build_run_context(run_id: str, device: torch.device | None = None) -> dict:
    run_id = str(run_id).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', run_id):
        raise ValueError(
            '--run-id must be 1-80 characters, start with an alphanumeric, '
            'and contain only letters, digits, dot, underscore, or hyphen'
        )
    if run_id in {'.', '..'}:
        raise ValueError(f'Invalid --run-id {run_id!r}')

    model_yaml_text, model_yaml_sha256 = _yaml_sha256(MODEL_YAML)
    execution_device = device or torch.device('cpu')
    dataset_yaml_texts, dataset_yaml_hashes = {}, {}
    dataset_configs, model_configs = {}, {}
    for dataset in DATASETS:
        dataset_yaml_texts[dataset], dataset_yaml_hashes[dataset] = _yaml_sha256(
            DATASET_YAMLS[dataset]
        )
        dataset_cfg = config.load_dataset_config(dataset)
        val_split = float(dataset_cfg.get('val_split', float('nan')))
        if abs(val_split - 0.7) > 1e-12:
            raise ValueError(
                f'{dataset}: this CodeBrain rerun is fixed to val_split=0.7 '
                f'(30% train/70% test), but dataset YAML has {val_split!r}'
            )
        if int(dataset_cfg.get('num_classes', 0)) != 2:
            raise ValueError(f'{dataset}: expected a two-class dataset config')
        dataset_configs[dataset] = dataset_cfg
        model_configs[dataset] = _validate_model_config(
            dataset, config.load_model_config('codebrain', dataset, 'fewshot')
        )

    if (execution_device.type != 'cpu'
            and any(model_configs[dataset].get('backend') is not None
                    for dataset in DATASETS)):
        raise ValueError(
            'A CodeBrain CPU backend policy is configured, but the selected '
            f'device is {execution_device}; set CUDA_VISIBLE_DEVICES=\'\' so '
            '_device resolves the requested CPU recipe'
        )

    runtime_backend_states = {
        dataset: _apply_runtime_backend(model_configs[dataset], execution_device)
        for dataset in DATASETS
    }

    selected_session = os.environ.get('MI2015001_SESSION', 'session_A')
    if selected_session != 'session_A':
        raise ValueError(
            'This CodeBrain run-id is fixed to BNCI2015001 session_A; '
            f'MI2015001_SESSION={selected_session!r}'
        )

    checkpoint_value = (
        model_configs[DATASETS[0]].get('pretrain') or config.weight_path('codebrain')
    )
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'CodeBrain pretrained checkpoint not found: {checkpoint_path}')
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != OFFICIAL_WEIGHT_SHA256:
        raise ValueError(
            'Resolved CodeBrain checkpoint does not match the public checkpoint: '
            f'expected {OFFICIAL_WEIGHT_SHA256}, got {checkpoint_sha256} '
            f'({checkpoint_path})'
        )

    source_hashes = {}
    for label, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing CodeBrain source file {path}')
        source_hashes[label] = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    for relpath, expected in {
        'upstream_sssm': 'e2fe5f7364907129507f3a9e946df9f9e44e10c3511efdcf772fab46d6fa278f',
        'upstream_sgconv': '94f12f897ab4e8784a45d64fb65664184c872e6928159a6759fda2920c488be8',
        'upstream_license': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    }.items():
        if source_hashes[relpath]['sha256'] != expected:
            raise ValueError(
                f'CodeBrain vendored source changed: {relpath} '
                f'{source_hashes[relpath]["sha256"]} != {expected}'
            )

    versions = _runtime_versions()
    recipes = {}
    for dataset in DATASETS:
        recipes[dataset] = {
            'model_config': model_configs[dataset],
            'dataset_config_sha256': dataset_yaml_hashes[dataset],
            'dataset_split': {
                'val_split': float(dataset_configs[dataset]['val_split']),
                'split_policy': SPLIT_POLICY,
                'session': SESSIONS[dataset],
            },
        }
        recipes[dataset]['recipe_sha256'] = _digest({
            **recipes[dataset],
            'model_yaml_sha256': model_yaml_sha256,
            'code_sha256': {k: v['sha256'] for k, v in source_hashes.items()},
            'pretrained_weight_sha256': checkpoint_sha256,
            'runtime_versions': versions,
            'gradient_clip_error_if_nonfinite': True,
            'execution_device': str(execution_device),
            'runtime_backend_state': runtime_backend_states[dataset],
        })

    root = RESULTS_ROOT / run_id
    artifact_root = root / 'artifacts'
    manifest = {
        'schema_version': 1,
        'run_id': run_id,
        'protocol': 'fewshot',
        'datasets': list(DATASETS),
        'seeds_expected': [666, 667, 668],
        'session_policy': {'BNCI2015001': 'session_A'},
        'runtime_backend_policy_by_dataset': {
            dataset: model_configs[dataset].get('backend') for dataset in DATASETS
        },
        'execution_device': str(execution_device),
        'runtime_backend_state_by_dataset': runtime_backend_states,
        'gradient_clip_error_if_nonfinite': True,
        'session_environment_value': os.environ.get('MI2015001_SESSION'),
        'split_policy': SPLIT_POLICY,
        'val_split': 0.7,
        'model_config_yaml': str(MODEL_YAML.resolve()),
        'model_config_yaml_sha256': model_yaml_sha256,
        'dataset_config_yaml_sha256': dataset_yaml_hashes,
        'source_hashes': source_hashes,
        'pretrained_weight': {
            'path': str(checkpoint_path),
            'url': OFFICIAL_WEIGHT_URL,
            'revision': OFFICIAL_WEIGHT_REVISION,
            'sha256': checkpoint_sha256,
        },
        'runtime_versions': versions,
        'runtime_backend_by_dataset': runtime_backend_states,
        'recipes': recipes,
    }
    manifest['run_signature_sha256'] = _digest(_manifest_signature_payload(manifest))
    context = {
        'run_id': run_id,
        'run_root': root,
        'artifact_root': artifact_root,
        'manifest': manifest,
        'model_yaml_text': model_yaml_text,
        'dataset_yaml_texts': dataset_yaml_texts,
        'model_yaml_sha256': model_yaml_sha256,
        'dataset_yaml_hashes': dataset_yaml_hashes,
        'model_configs': model_configs,
        'dataset_configs': dataset_configs,
        'source_hashes': source_hashes,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'runtime_versions': versions,
    }
    _check_run_id_collision(context)
    return context


def _check_snapshot(path: Path, expected_text: str, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f'Run-id snapshot is missing: {path}')
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha256 or resolve_local_file(path).read_text(encoding='utf-8') != expected_text:
        raise RuntimeError(f'Run-id snapshot hash collision: {path}')


def _check_run_id_collision(context: dict) -> None:
    root = context['run_root']
    if not root.exists():
        return
    manifest_path = root / 'run_manifest.json'
    if not manifest_path.is_file():
        entries = list(root.iterdir())
        if entries:
            raise FileExistsError(
                f'--run-id {context["run_id"]!r} already has output but no '
                f'matching run manifest: {root}; choose a new --run-id'
            )
        return
    prior = json.loads(resolve_local_file(manifest_path).read_text(encoding='utf-8'))
    expected = context['manifest']['run_signature_sha256']
    actual = prior.get('run_signature_sha256')
    if actual != expected:
        raise RuntimeError(
            f'--run-id {context["run_id"]!r} recipe hash conflict: '
            f'existing={actual!r}, requested={expected!r}; choose a new --run-id'
        )
    prior_payload = _manifest_signature_payload(prior)
    expected_payload = _manifest_signature_payload(context['manifest'])
    if (
        prior.get('run_id') != context['run_id']
        or prior_payload != expected_payload
        or _digest(prior_payload) != actual
    ):
        raise RuntimeError(f'Run manifest was modified under run-id: {manifest_path}')
    snapshot_root = root / 'config_snapshot'
    _check_snapshot(
        snapshot_root / 'models' / 'codebrain.yaml',
        context['model_yaml_text'], context['model_yaml_sha256'],
    )
    for dataset, text in context['dataset_yaml_texts'].items():
        _check_snapshot(
            snapshot_root / 'datasets' / f'{dataset}.yaml', text,
            context['dataset_yaml_hashes'][dataset],
        )


def _persist_run_context(context: dict) -> None:
    _check_run_id_collision(context)
    root = context['run_root']
    snapshot_root = root / 'config_snapshot'
    model_snapshot = snapshot_root / 'models' / 'codebrain.yaml'
    model_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not model_snapshot.exists():
        model_snapshot.write_text(context['model_yaml_text'], encoding='utf-8')
    for dataset, text in context['dataset_yaml_texts'].items():
        path = snapshot_root / 'datasets' / f'{dataset}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text, encoding='utf-8')
    resolved_snapshot = {
        'run_id': context['run_id'],
        'run_signature_sha256': context['manifest']['run_signature_sha256'],
        'model_yaml_sha256': context['model_yaml_sha256'],
        'dataset_yaml_sha256': context['dataset_yaml_hashes'],
        'resolved_model_config_by_dataset': context['model_configs'],
        'dataset_split_by_dataset': {
            dataset: {
                'val_split': float(context['dataset_configs'][dataset]['val_split']),
                'split_policy': SPLIT_POLICY,
                'session': SESSIONS[dataset],
            }
            for dataset in DATASETS
        },
        'runtime_versions': context['runtime_versions'],
        'source_hashes': context['source_hashes'],
        'pretrained_weight': context['manifest']['pretrained_weight'],
        'runtime_backend_policy_by_dataset': context['manifest'].get(
            'runtime_backend_policy_by_dataset', {}
        ),
        'runtime_backend_state_by_dataset': context['manifest'].get(
            'runtime_backend_state_by_dataset', {}
        ),
        'gradient_clip_error_if_nonfinite': True,
    }
    resolved_path = root / 'config_resolved_all.json'
    if resolved_path.exists():
        current = json.loads(resolve_local_file(resolved_path).read_text(encoding='utf-8'))
        if current != resolved_snapshot:
            raise RuntimeError(f'Resolved config collision under run-id: {resolved_path}')
    else:
        _write_json(resolved_path, resolved_snapshot)
    manifest_path = root / 'run_manifest.json'
    if not manifest_path.exists():
        _write_json(manifest_path, context['manifest'])
    _check_run_id_collision(context)


def _run_recipe(context: dict, dataset: str) -> dict:
    return context['manifest']['recipes'][dataset]


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(gpu: int | None) -> torch.device:
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        return torch.device(f'cuda:{int(gpu)}')
    if gpu is not None and not torch.cuda.is_available():
        raise RuntimeError(f'--gpu {gpu} was requested but CUDA is unavailable')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _subject_split(dataset: str, subject: int, seed: int):
    """Use the canonical project split and verify all cached comparators."""
    dataset_cfg = config.load_dataset_config(dataset)
    val_split = float(dataset_cfg.get('val_split', float('nan')))
    if abs(val_split - 0.7) > 1e-12:
        raise ValueError(
            f'{dataset} dataset config changed val_split from required 0.7 to {val_split}'
        )
    if dataset == 'BNCI2015001':
        selected_session = os.environ.get('MI2015001_SESSION', 'session_A')
        if selected_session != 'session_A':
            raise ValueError(
                'CodeBrain baseline is fixed to BNCI2015001 session_A; '
                f'MI2015001_SESSION={selected_session!r}'
            )
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset, subject, val_split=val_split, seed=seed, return_uid=True
    )
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    X_te = np.asarray(X_te, dtype=np.float32)
    y_te = np.asarray(y_te, dtype=np.int64)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    uid_te = np.asarray(uid_te, dtype=np.int64)
    if X_tr.ndim != 3 or X_te.ndim != 3:
        raise ValueError(f'{dataset} subject {subject}: expected [N,C,T] raw arrays')
    if X_tr.shape[1] != len(CHANNELS[dataset]) or X_te.shape[1] != len(CHANNELS[dataset]):
        raise ValueError(
            f'{dataset} native channel metadata has {len(CHANNELS[dataset])} names, '
            f'but raw data has {X_tr.shape[1]}/{X_te.shape[1]} channels'
        )
    if not np.array_equal(np.unique(np.concatenate([y_tr, y_te])), [0, 1]):
        raise ValueError(
            f'{dataset} subject {subject}: expected project label IDs [0, 1], '
            f'got {np.unique(np.concatenate([y_tr, y_te])).tolist()}'
        )
    if len(np.unique(uid_tr, axis=0)) != len(uid_tr):
        raise ValueError(f'{dataset} subject {subject}: duplicate train UID')
    if len(np.unique(uid_te, axis=0)) != len(uid_te):
        raise ValueError(f'{dataset} subject {subject}: duplicate test UID')
    if set(map(tuple, uid_tr)).intersection(set(map(tuple, uid_te))):
        raise ValueError(f'{dataset} subject {subject}: train/test UID overlap')

    reference = {}
    for model_name in REFERENCE_MODELS:
        for split, y, uid in (('train', y_tr, uid_tr), ('test', y_te, uid_te)):
            path = artifacts.artifact_path(dataset, model_name, subject, seed, split)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f'Missing {model_name} {split} artifact required for alignment: {path}'
                )
            cached = artifacts.load(dataset, model_name, subject, seed, split)
            if 'sample_uid' not in cached:
                raise ValueError(f'{model_name} {path} has no sample_uid')
            if 'split_policy' not in cached:
                raise ValueError(f'{model_name} {path} has no split_policy')
            if not np.array_equal(cached['y'], y):
                raise ValueError(
                    f'{model_name} labels mismatch {dataset} subject {subject} '
                    f'seed {seed} split {split}'
                )
            if not np.array_equal(cached['sample_uid'], uid):
                raise ValueError(
                    f'{model_name} UID mismatch {dataset} subject {subject} '
                    f'seed {seed} split {split}'
                )
            if cached['split_policy'] != SPLIT_POLICY:
                raise ValueError(
                    f'{model_name} split_policy={cached["split_policy"]!r}; '
                    f'expected {SPLIT_POLICY!r}'
                )
            reference[f'{model_name}_{split}'] = {
                'path': path,
                'uid_match': True,
                'y_match': True,
                'split_policy_match': True,
                'split_policy': cached['split_policy'],
            }

    return {
        'X_tr': X_tr,
        'y_tr': y_tr,
        'X_te': X_te,
        'y_te': y_te,
        'uid_tr': uid_tr,
        'uid_te': uid_te,
        'reference': reference,
    }


def _counts(y: np.ndarray, num_classes: int = 2) -> dict[str, int]:
    count = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes)
    return {str(i): int(count[i]) for i in range(num_classes)}


def _training_signal_stats(X_tr: np.ndarray) -> dict:
    values = np.asarray(X_tr, dtype=np.float64)
    return {
        'raw_train_abs_quantiles_uv': {
            'q50': float(np.quantile(np.abs(values), 0.50)),
            'q90': float(np.quantile(np.abs(values), 0.90)),
            'q99': float(np.quantile(np.abs(values), 0.99)),
            'q100': float(np.max(np.abs(values))),
        },
        'raw_train_channel_std_median_uv': float(np.median(values.std(axis=-1))),
    }


def _manifest(context: dict, dataset: str, subject: int, seed: int, split: dict,
              Xp_tr: np.ndarray, Xp_te: np.ndarray) -> dict:
    val_split = float(config.load_dataset_config(dataset)['val_split'])
    return {
        'run_id': context['run_id'],
        'run_recipe_sha256': _run_recipe(context, dataset)['recipe_sha256'],
        'model_config_yaml_sha256': context['model_yaml_sha256'],
        'dataset_config_yaml_sha256': context['dataset_yaml_hashes'][dataset],
        'dataset': dataset,
        'subject_index_zero_based': int(subject),
        'subject': f'S{int(subject) + 1}',
        'seed': int(seed),
        'protocol': 'fewshot',
        'session': SESSIONS[dataset],
        'session_environment': (
            os.environ.get('MI2015001_SESSION', 'session_A')
            if dataset == 'BNCI2015001' else None
        ),
        'val_split_test_fraction': val_split,
        'train_fraction': 1.0 - val_split,
        'split_policy': SPLIT_POLICY,
        'train_count': int(len(split['y_tr'])),
        'test_count': int(len(split['y_te'])),
        'train_uid': split['uid_tr'].tolist(),
        'test_uid': split['uid_te'].tolist(),
        'train_uid_sha256': _array_hash(split['uid_tr']),
        'test_uid_sha256': _array_hash(split['uid_te']),
        'train_y_sha256': _array_hash(split['y_tr']),
        'test_y_sha256': _array_hash(split['y_te']),
        'train_class_counts': _counts(split['y_tr']),
        'test_class_counts': _counts(split['y_te']),
        'train_test_uid_disjoint': True,
        'raw_train_shape': list(split['X_tr'].shape),
        'raw_test_shape': list(split['X_te'].shape),
        'raw_train_X_sha256': _array_hash(split['X_tr']),
        'raw_test_X_sha256': _array_hash(split['X_te']),
        'model_train_shape': list(Xp_tr.shape),
        'model_test_shape': list(Xp_te.shape),
        'native_channels': CHANNELS[dataset],
        'class_label_names_by_project_id': {
            '0': CLASS_NAMES[dataset][0],
            '1': CLASS_NAMES[dataset][1],
        },
        'train_signal_statistics_only': _training_signal_stats(split['X_tr']),
        'reference_artifact_checks': split['reference'],
        'runtime_backend': context.get('runtime_backend_by_dataset', {}).get(dataset),
    }


def _resolved_config(context: dict, dataset: str, init_mode: str, seed: int,
                     subject: int, *, smoke: bool = False,
                     actual_epochs: int | None = None) -> dict:
    model_cfg = context['model_configs'][dataset]
    dataset_cfg = context['dataset_configs'][dataset]
    input_cfg = CodeBrainInputAdapter(
        scale_divisor=model_cfg['scale_divisor']
    ).config()
    configured_epochs = int(model_cfg['epochs'])
    checkpoint = context['manifest']['pretrained_weight']
    return {
        'run_id': context['run_id'],
        'run_root': str(context['run_root'].resolve()),
        'artifact_root': str(context['artifact_root'].resolve()),
        'model': 'CodeBrain EEGSSM + three-layer MLP task head',
        'initialization': init_mode,
        'dataset': dataset,
        'subject_index_zero_based': int(subject),
        'seed': int(seed),
        'protocol': 'fewshot',
        'split': {
            'function': 'data.subject_split',
            'val_split_test_fraction': float(dataset_cfg['val_split']),
            'train_fraction': 1.0 - float(dataset_cfg['val_split']),
            'seeded': True,
            'split_policy': SPLIT_POLICY,
            'dataset_yaml_sha256': context['dataset_yaml_hashes'][dataset],
            'session_policy': SESSIONS[dataset],
            'session_environment_value': os.environ.get('MI2015001_SESSION')
            if dataset == 'BNCI2015001' else None,
        },
        'model_config_yaml_sha256': context['model_yaml_sha256'],
        'dataset_config_yaml_sha256': context['dataset_yaml_hashes'][dataset],
        'recipe_sha256': _run_recipe(context, dataset)['recipe_sha256'],
        'source_code_sha256': context['source_hashes'],
        'runtime_versions': context['runtime_versions'],
        'runtime_backend': context.get('runtime_backend_by_dataset', {}).get(
            dataset, {'configured_policy': model_cfg.get('backend')}
        ),
        'pretrained_weight': checkpoint,
        'pretrained_weight_loaded': init_mode == 'pretrained',
        'encoder': {
            'implementation_repo': OFFICIAL_REPO,
            'implementation_revision': OFFICIAL_REVISION,
            'implementation_scope': 'Models/SSSM.py, Models/SGConv.py, LICENSE',
            'checkpoint_url': OFFICIAL_WEIGHT_URL,
            'checkpoint_revision': OFFICIAL_WEIGHT_REVISION,
            'checkpoint_expected_sha256': OFFICIAL_WEIGHT_SHA256,
            'layers': 8,
            'in_residual_skip_out_channels': [200, 200, 200, 200],
            's4_lmax': 570,
            's4_d_state': 64,
            'bidirectional': True,
            'layer_norm': True,
        },
        'input': input_cfg,
        'task_head': {
            'layers': [len(CHANNELS[dataset]) * 4 * 200, 800, 200, 2],
            'activations': ['ReLU', 'ReLU'],
            'dropout': model_cfg['dropout'],
            'trainable': True,
        },
        'optimization': {
            'optimizer': 'AdamW',
            'lr': (model_cfg['lr']
                   if model_cfg['scheduler'] != 'cosine_per_group_to_10pct'
                   else None),
            'lr_status': (
                'used as the shared initial learning rate'
                if model_cfg['scheduler'] != 'cosine_per_group_to_10pct'
                else 'unused; encoder_lr and head_lr are the active group rates'
            ),
            'encoder_lr': model_cfg.get('encoder_lr'),
            'head_lr': model_cfg.get('head_lr'),
            'weight_decay': model_cfg['weight_decay'],
            'dropout': model_cfg['dropout'],
            'scale_divisor': model_cfg['scale_divisor'],
            'batch_size': model_cfg['batch_size'],
            'epochs': configured_epochs,
            'epochs_this_action': int(actual_epochs or configured_epochs),
            'checkpoint_selection': 'final epoch; no validation or test selection',
            'scheduler': model_cfg['scheduler'],
            'scheduler_step_policy': (
                'once per training batch'
                if model_cfg['scheduler'] in (
                    'cosine_per_batch', 'cosine_per_group_to_10pct'
                ) else 'disabled'
            ),
            'scheduler_t_max': (
                'epochs * batches_per_epoch'
                if model_cfg['scheduler'] in (
                    'cosine_per_batch', 'cosine_per_group_to_10pct'
                ) else None
            ),
            'scheduler_eta_min': (
                model_cfg['min_lr']
                if model_cfg['scheduler'] == 'cosine_per_batch' else None
            ),
            'scheduler_group_min_lr_ratio': (
                model_cfg['group_min_lr_ratio']
                if model_cfg['scheduler'] == 'cosine_per_group_to_10pct' else None
            ),
            'scheduler_group_floor_lrs': (
                {
                    'encoder': model_cfg['encoder_lr'] * model_cfg['group_min_lr_ratio'],
                    'head': model_cfg['head_lr'] * model_cfg['group_min_lr_ratio'],
                } if model_cfg['scheduler'] == 'cosine_per_group_to_10pct' else None
            ),
            'loss': 'CrossEntropyLoss',
            'label_smoothing': model_cfg['label_smoothing'],
            'gradient_clipping_max_norm': model_cfg['max_grad_norm'],
            'gradient_clip_error_if_nonfinite': True,
            'all_encoder_parameters_trainable_except_official_mask_encoding': True,
            'official_fixed_parameters': ['backbone.patch_embedding.mask_encoding'],
            'knowledge_distillation': False,
            'mutual_information_loss': False,
            'input_mask_augmentation': False,
            'optimizer_parameter_groups': (
                [
                    {'name': 'encoder', 'initial_lr': model_cfg['encoder_lr'],
                     'minimum_lr_ratio': model_cfg['group_min_lr_ratio'],
                     'floor_lr': model_cfg['encoder_lr'] * model_cfg['group_min_lr_ratio']},
                    {'name': 'head', 'initial_lr': model_cfg['head_lr'],
                     'minimum_lr_ratio': model_cfg['group_min_lr_ratio'],
                     'floor_lr': model_cfg['head_lr'] * model_cfg['group_min_lr_ratio']},
                ] if model_cfg['scheduler'] == 'cosine_per_group_to_10pct'
                else [{'name': 'all_parameters', 'initial_lr': model_cfg['lr']}]
            ),
        },
        'smoke_run': bool(smoke),
        'formal_result': not bool(smoke),
    }


def _make_model(dataset: str, init_mode: str, seed: int, device: torch.device,
                model_cfg: dict, checkpoint_path: Path):
    _set_seed(seed)
    dropout = float(model_cfg['dropout'])
    backbone = build_backbone(dropout=dropout)
    if init_mode == 'pretrained':
        load_report = load_codebrain_backbone(backbone, checkpoint_path)
    elif init_mode == 'random':
        load_report = {
            'initialization': 'random',
            'source_repo': OFFICIAL_REPO,
            'source_revision': OFFICIAL_REVISION,
            'checkpoint_loaded': False,
            'reason': 'Same official encoder architecture, independently seeded initialization.',
        }
    else:
        raise ValueError(f'Unknown initialization mode {init_mode!r}')
    model = CodeBrainClassifier(
        n_channels=len(CHANNELS[dataset]),
        num_classes=2,
        dropout=dropout,
        backbone=backbone,
    ).to(device)
    all_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters()
                           if parameter.requires_grad)
    fixed_parameters = [
        {'name': name, 'numel': int(parameter.numel())}
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    allowed_fixed_names = {'backbone.patch_embedding.mask_encoding'}
    actual_fixed_names = {item['name'] for item in fixed_parameters}
    unexpected_fixed_names = actual_fixed_names - allowed_fixed_names
    missing_official_fixed = allowed_fixed_names - actual_fixed_names
    if unexpected_fixed_names or missing_official_fixed:
        raise RuntimeError(
            'Unexpected CodeBrain frozen-parameter set: '
            f'actual={sorted(actual_fixed_names)}, '
            f'unexpected={sorted(unexpected_fixed_names)}, '
            f'missing_official={sorted(missing_official_fixed)}'
        )
    load_report['total_parameter_count'] = int(all_params)
    load_report['trainable_parameter_count'] = int(trainable_params)
    load_report['fixed_parameter_count'] = int(all_params - trainable_params)
    load_report['fixed_parameters'] = fixed_parameters
    load_report['all_other_encoder_and_head_parameters_trainable'] = True
    load_report['task_head_parameter_count'] = int(
        sum(parameter.numel() for parameter in model.feature_mlp.parameters())
        + sum(parameter.numel() for parameter in model.classifier.parameters())
    )
    return model, load_report


def _tensor_data(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)),
                         torch.from_numpy(np.asarray(y, dtype=np.int64)))


def _infer(model: nn.Module, X: np.ndarray, y: np.ndarray,
           device: torch.device, batch_size: int):
    loader = DataLoader(_tensor_data(X, y), batch_size=int(batch_size),
                        shuffle=False, num_workers=0)
    hidden_all, logits_all, y_all = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            hidden, logits = model(xb)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise AssertionError(f'Expected [B,2] logits; got {tuple(logits.shape)}')
            if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
                raise FloatingPointError('Non-finite CodeBrain inference output')
            hidden_all.append(hidden.float().cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
            y_all.append(yb.numpy())
    return (
        np.concatenate(hidden_all, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(y_all, axis=0),
    )


def _score(y_true: np.ndarray, logits: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    pred = logits.argmax(axis=1).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'kappa': float(cohen_kappa_score(y_true, pred)),
        'predicted_class_counts': _counts(pred),
        'true_class_counts': _counts(y_true),
        'single_class_prediction_collapse': bool(len(np.unique(pred)) < 2),
        'predicted_labels': pred,
    }


def _run_dir(context: dict, init_mode: str, dataset: str, subject: int, seed: int,
             run_kind: str = 'train') -> Path:
    return (context['run_root'] / run_kind / init_mode / dataset / f'S{subject + 1}' /
            str(seed))


def _prepare(dataset: str, subject: int, seed: int,
             input_adapter: CodeBrainInputAdapter):
    split = _subject_split(dataset, subject, seed)
    Xp_tr = input_adapter.transform(split['X_tr'])
    Xp_te = input_adapter.transform(split['X_te'])
    split['Xp_tr'] = Xp_tr
    split['Xp_te'] = Xp_te
    return split


def _preflight_one(context: dict, dataset: str, subject: int, seed: int,
                   init_mode: str, device: torch.device) -> dict:
    model_cfg = context['model_configs'][dataset]
    split = _prepare(
        dataset, subject, seed,
        CodeBrainInputAdapter(scale_divisor=model_cfg['scale_divisor']),
    )
    run_dir = _run_dir(context, init_mode, dataset, subject, seed, 'preflight')
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(context, dataset, subject, seed, split,
                         split['Xp_tr'], split['Xp_te'])
    resolved = _resolved_config(context, dataset, init_mode, seed, subject,
                                actual_epochs=model_cfg['epochs'])
    resolved['input']['train_signal_stats'] = _training_signal_stats(split['X_tr'])
    resolved['data_fingerprint'] = {
        key: manifest[key] for key in (
            'raw_train_X_sha256', 'raw_test_X_sha256', 'train_uid_sha256',
            'test_uid_sha256', 'train_y_sha256', 'test_y_sha256', 'session',
            'session_environment',
        )
    }
    resolved['formal_result'] = False
    resolved['phase'] = 'preflight'
    resolved['config_sha256'] = _digest(resolved)
    _write_json(run_dir / 'split_manifest.json', manifest)
    _write_json(run_dir / 'config_resolved.json', resolved)
    model, load_report = _make_model(
        dataset, init_mode, seed, device, model_cfg, context['checkpoint_path']
    )
    _write_json(run_dir / 'weight_load_report.json', load_report)
    print(
        '[weights] '
        f'loaded={load_report.get("loaded_tensor_count", 0)} '
        f'missing={load_report.get("missing_keys", [])} '
        f'unexpected={load_report.get("unexpected_keys", [])} '
        f'sha256={load_report.get("checkpoint_sha256", "random-init")}',
        flush=True,
    )
    checks = {}
    model.eval()
    with torch.no_grad():
        for split_name, Xp, y in (('train', split['Xp_tr'], split['y_tr']),
                                  ('test', split['Xp_te'], split['y_te'])):
            checks[split_name] = {}
            for batch_size in sorted(set((1, min(2, len(Xp))))):
                xb = torch.as_tensor(
                    Xp[:batch_size], dtype=torch.float32, device=device
                )
                hidden, logits = model(xb)
                expected = (batch_size, 2)
                if tuple(logits.shape) != expected:
                    raise AssertionError(
                        f'{dataset} {split_name} B={batch_size}: '
                        f'logits {logits.shape} != {expected}'
                    )
                if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
                    raise FloatingPointError(
                        f'{dataset} {split_name} B={batch_size}: non-finite output'
                    )
                checks[split_name][f'batch_{batch_size}'] = {
                    'input_shape': list(xb.shape),
                    'feature_shape': list(hidden.shape),
                    'logits_shape': list(logits.shape),
                    'finite_features': True,
                    'finite_logits': True,
                }
            checks[split_name]['label_ids_in_full_split'] = sorted(np.unique(y).tolist())
            checks[split_name]['uid_order_check'] = True
    result = {
        'status': 'passed',
        'dataset': dataset,
        'subject': subject,
        'seed': seed,
        'initialization': init_mode,
        'device': str(device),
        'train': checks['train'],
        'test': checks['test'],
        'split_policy': SPLIT_POLICY,
        'weight_load_report': load_report,
        'formal_result': False,
    }
    _write_json(run_dir / 'preflight.json', result)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'[preflight:ok] {dataset} S{subject + 1} seed={seed} device={device}', flush=True)
    return result


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                 criterion: nn.Module, device: torch.device,
                 max_grad_norm: float,
                 max_batches: int | None = None) -> dict:
    model.train()
    loss_sum = 0.0
    count = 0
    correct = 0
    grad_norm_sum = 0.0
    lr_sums = [0.0 for _ in optimizer.param_groups]
    for batch_index, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        _hidden, logits = model(xb)
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(max_grad_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_count = int(len(yb))
        loss_sum += float(loss.detach().item()) * batch_count
        count += batch_count
        correct += int((logits.detach().argmax(dim=1) == yb).sum().item())
        grad_norm_sum += float(grad_norm.detach().item())
        for index, group in enumerate(optimizer.param_groups):
            lr_sums[index] += float(group['lr'])
    if count == 0:
        raise RuntimeError('Training epoch did not process a batch')
    batch_count = max(1, int(len(loader) if max_batches is None else min(len(loader), max_batches)))
    group_names = [
        str(group.get('group_name', f'group_{index}'))
        for index, group in enumerate(optimizer.param_groups)
    ]
    mean_lrs = {
        group_names[index]: lr_sums[index] / batch_count
        for index in range(len(group_names))
    }
    return {
        'train_loss': loss_sum / count,
        'train_accuracy': correct / count,
        'train_examples_seen': count,
        'train_batches_seen': min(len(loader), int(max_batches))
        if max_batches is not None else len(loader),
        'mean_preclip_grad_norm': grad_norm_sum / max(1, int(len(loader) if max_batches is None else min(len(loader), max_batches))),
        # Keep the historic scalar field for existing single-group consumers.
        'mean_learning_rate_after_step': mean_lrs[group_names[0]],
        'mean_learning_rate_after_step_by_group': mean_lrs,
    }


def _build_finetune_optimizer_scheduler(
    model: nn.Module, model_cfg: dict, epochs: int, batches_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Build the existing single-LR optimizer or an opt-in encoder/head pair."""
    if model_cfg['scheduler'] == 'cosine_per_group_to_10pct':
        encoder_params = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith('backbone.') and parameter.requires_grad
        ]
        head_params = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith('backbone.') and parameter.requires_grad
        ]
        if not encoder_params or not head_params:
            raise RuntimeError('Grouped LR needs trainable backbone and task-head parameters')
        groups = [
            {'params': encoder_params, 'lr': float(model_cfg['encoder_lr']),
             'group_name': 'encoder'},
            {'params': head_params, 'lr': float(model_cfg['head_lr']),
             'group_name': 'head'},
        ]
        optimizer = torch.optim.AdamW(
            groups, weight_decay=float(model_cfg['weight_decay']), eps=1e-8,
        )
        total_steps = max(1, int(epochs) * int(batches_per_epoch))
        floor_ratio = float(model_cfg['group_min_lr_ratio'])

        def cosine_to_ratio(step: int) -> float:
            progress = min(1.0, max(0.0, float(step) / total_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[cosine_to_ratio, cosine_to_ratio]
        )
        return optimizer, scheduler

    # Preserve the established one-group path exactly for existing YAMLs.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_cfg['lr']),
        weight_decay=float(model_cfg['weight_decay']), eps=1e-8,
    )
    if model_cfg['scheduler'] == 'cosine_per_batch':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs) * int(batches_per_epoch)),
            eta_min=float(model_cfg['min_lr']),
        )
    elif model_cfg['scheduler'] == 'none':
        scheduler = None
    else:
        raise AssertionError(f"Unhandled scheduler {model_cfg['scheduler']!r}")
    return optimizer, scheduler


def _optimizer_group_lrs(optimizer: torch.optim.Optimizer) -> list[dict]:
    return [
        {
            'name': str(group.get('group_name', f'group_{index}')),
            'lr': float(group['lr']),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def _save_final_checkpoint(path: Path, model: nn.Module, cfg: dict,
                           load_report: dict) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({
        'model_state_dict': state,
        'config': cfg,
        'weight_load_report': load_report,
    }, require_external_output(path))


def _hub_artifact_matches(path: Path, y: np.ndarray, uid: np.ndarray) -> bool:
    try:
        with np.load(resolve_local_file(path), allow_pickle=False) as stored:
            return (
                {'logits', 'feats', 'y', 'sample_uid', 'split_policy'} <= set(stored.files)
                and np.array_equal(stored['y'], y)
                and np.array_equal(stored['sample_uid'], uid)
                and str(stored['split_policy'].item()) == SPLIT_POLICY
                and stored['logits'].shape[0] == len(y)
                and stored['feats'].shape[0] == len(y)
            )
    except Exception:
        return False


def _prediction_matches(path: Path, y: np.ndarray, uid: np.ndarray) -> bool:
    try:
        with np.load(resolve_local_file(path), allow_pickle=False) as stored:
            return (
                {'logits', 'y_true', 'y_pred', 'sample_uid'} <= set(stored.files)
                and np.array_equal(stored['y_true'], y)
                and np.array_equal(stored['sample_uid'], uid)
                and stored['logits'].shape[0] == len(y)
                and stored['y_pred'].shape == y.shape
            )
    except Exception:
        return False


def _run_one(context: dict, dataset: str, subject: int, seed: int, init_mode: str,
             device: torch.device, *,
             smoke: bool = False, smoke_batches: int = 2,
             epoch_override: int | None = None):
    model_cfg = context['model_configs'][dataset]
    split = _prepare(
        dataset, subject, seed,
        CodeBrainInputAdapter(scale_divisor=model_cfg['scale_divisor']),
    )
    formal = not smoke
    run_dir = _run_dir(context, init_mode, dataset, subject, seed,
                       'smoke' if smoke else 'train')
    run_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_name = ARTIFACT_MODEL[init_mode]
    checkpoint_path = run_dir / 'checkpoint_final.pt'
    metrics_path = run_dir / 'metrics.json'
    artifact_test_path = Path(artifacts.artifact_path(
        dataset, model_artifact_name, subject, seed, 'test',
        root=context['artifact_root']))
    artifact_train_path = Path(artifacts.artifact_path(
        dataset, model_artifact_name, subject, seed, 'train',
        root=context['artifact_root']))
    configured_epochs = int(model_cfg['epochs'])
    epochs = int(epoch_override if smoke and epoch_override is not None
                 else configured_epochs)
    batch_size = int(model_cfg['batch_size'])
    cfg = _resolved_config(
        context, dataset, init_mode, seed, subject, smoke=smoke,
        actual_epochs=epochs,
    )
    cfg['input']['train_signal_stats'] = _training_signal_stats(split['X_tr'])
    manifest = _manifest(context, dataset, subject, seed, split,
                         split['Xp_tr'], split['Xp_te'])
    cfg['data_fingerprint'] = {
        'raw_train_X_sha256': manifest['raw_train_X_sha256'],
        'raw_test_X_sha256': manifest['raw_test_X_sha256'],
        'train_uid_sha256': manifest['train_uid_sha256'],
        'test_uid_sha256': manifest['test_uid_sha256'],
        'train_y_sha256': manifest['train_y_sha256'],
        'test_y_sha256': manifest['test_y_sha256'],
        'session': manifest['session'],
        'session_environment': manifest['session_environment'],
    }
    cfg['config_sha256'] = _digest(cfg)
    history_path = run_dir / 'history.jsonl'
    cfg_path = run_dir / 'config_resolved.json'
    if cfg_path.is_file():
        previous_cfg = json.loads(resolve_local_file(cfg_path).read_text())
        if previous_cfg.get('run_id') != context['run_id']:
            raise RuntimeError(f'Run-id mismatch inside output directory: {cfg_path}')
        if previous_cfg.get('recipe_sha256') != _run_recipe(context, dataset)['recipe_sha256']:
            raise RuntimeError(
                f'Recipe hash conflict inside run-id {context["run_id"]!r}: '
                f'{run_dir}; choose a new --run-id'
            )
        if previous_cfg.get('data_fingerprint') != cfg.get('data_fingerprint'):
            raise RuntimeError(
                f'Input/split fingerprint changed inside run-id {context["run_id"]!r}: '
                f'{run_dir}; choose a new --run-id'
            )
    if formal and metrics_path.is_file() and checkpoint_path.is_file():
        prior = json.loads(resolve_local_file(metrics_path).read_text())
        cfg_on_disk = None
        if cfg_path.is_file():
            cfg_on_disk = json.loads(resolve_local_file(cfg_path).read_text())
        required_files = (
            cfg_path,
            run_dir / 'split_manifest.json',
            run_dir / 'weight_load_report.json',
            history_path,
            run_dir / 'test_predictions.npz',
            run_dir / 'train_predictions.npz',
            artifact_test_path,
            artifact_train_path,
        )
        history_complete = False
        if history_path.is_file():
            with history_path.open() as stream:
                history_rows = [json.loads(line) for line in stream if line.strip()]
            history_complete = (
                len(history_rows) == epochs
                and history_rows[-1].get('epoch') == epochs
            ) if history_rows else False
        same_run_complete = (
            prior.get('run_id') == context['run_id']
            and prior.get('recipe_sha256') == _run_recipe(context, dataset)['recipe_sha256']
            and prior.get('config_sha256') == cfg['config_sha256']
            and cfg_on_disk is not None
            and cfg_on_disk.get('run_id') == context['run_id']
            and cfg_on_disk.get('recipe_sha256') == _run_recipe(context, dataset)['recipe_sha256']
            and cfg_on_disk.get('config_sha256') == cfg['config_sha256']
            and prior.get('formal_result') is True
            and int(prior.get('epochs_completed', -1)) == epochs
            and all(path.is_file() and path.stat().st_size > 0 for path in required_files)
            and history_complete
            and _prediction_matches(run_dir / 'test_predictions.npz', split['y_te'], split['uid_te'])
            and _prediction_matches(run_dir / 'train_predictions.npz', split['y_tr'], split['uid_tr'])
            and _hub_artifact_matches(artifact_test_path, split['y_te'], split['uid_te'])
            and _hub_artifact_matches(artifact_train_path, split['y_tr'], split['uid_tr'])
        )
        if (
            same_run_complete
        ):
            print(f'[skip] {dataset} S{subject + 1} seed={seed} init={init_mode}', flush=True)
            return prior

    # Any non-skipped run begins from scratch. This prevents an interrupted
    # previous attempt from leaving duplicate epoch rows in a new history.
    if history_path.exists():
        history_path.unlink()
    _write_json(run_dir / 'config_resolved.json', cfg)
    _write_json(run_dir / 'split_manifest.json', manifest)

    _set_seed(seed)
    model, load_report = _make_model(
        dataset, init_mode, seed, device, model_cfg, context['checkpoint_path']
    )
    _write_json(run_dir / 'weight_load_report.json', load_report)
    print(
        '[weights] '
        f'loaded={load_report.get("loaded_tensor_count", 0)} '
        f'missing={load_report.get("missing_keys", [])} '
        f'unexpected={load_report.get("unexpected_keys", [])} '
        f'sha256={load_report.get("checkpoint_sha256", "random-init")}',
        flush=True,
    )
    # The historical one-LR path remains the default. Grouped encoder/head
    # rates are opt-in through the resolved model config for D/E candidates.
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        _tensor_data(split['Xp_tr'], split['y_tr']),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    optimizer, scheduler = _build_finetune_optimizer_scheduler(
        model, model_cfg, epochs, len(loader)
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(model_cfg['label_smoothing'])
    )

    epoch_rows = []
    started = time.time()
    for epoch in range(epochs):
        row = {'epoch': int(epoch + 1)}
        row['optimizer_group_lrs_start'] = _optimizer_group_lrs(optimizer)
        row.update(_train_epoch(
            model, loader, optimizer, scheduler, criterion, device,
            max_grad_norm=float(model_cfg['max_grad_norm']),
            max_batches=smoke_batches if smoke else None,
        ))
        row['optimizer_group_lrs_end_after_scheduler'] = _optimizer_group_lrs(optimizer)
        row['elapsed_seconds'] = float(time.time() - started)
        epoch_rows.append(row)
        with history_path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print(
            f'[epoch] {dataset} S{subject + 1} seed={seed} init={init_mode} '
            f'{epoch + 1}/{epochs} loss={row["train_loss"]:.5f} '
            f'acc={row["train_accuracy"]:.4f}', flush=True,
        )

    if smoke:
        result = {
            'status': 'smoke_passed',
            'dataset': dataset,
            'subject': subject,
            'seed': seed,
            'initialization': init_mode,
            'epochs_completed': epochs,
            'batches_per_epoch': int(epoch_rows[-1]['train_batches_seen']),
            'last_epoch': epoch_rows[-1],
            'elapsed_seconds': float(time.time() - started),
            'formal_result': False,
        }
        _write_json(run_dir / 'smoke_result.json', result)
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'[smoke:ok] {dataset} S{subject + 1} seed={seed}', flush=True)
        return result

    # The final epoch is fixed before training. Test data is not consulted until
    # the one final inference below.
    _save_final_checkpoint(checkpoint_path, model, cfg, load_report)
    train_feats, train_logits, train_y = _infer(
        model, split['Xp_tr'], split['y_tr'], device, batch_size
    )
    test_feats, test_logits, test_y = _infer(
        model, split['Xp_te'], split['y_te'], device, batch_size
    )
    test_score = _score(test_y, test_logits)
    predictions_path = run_dir / 'test_predictions.npz'
    np.savez_compressed(
        require_external_output(predictions_path),
        logits=test_logits.astype(np.float32),
        y_true=test_y,
        y_pred=test_score['predicted_labels'],
        sample_uid=split['uid_te'],
    )
    train_predictions_path = run_dir / 'train_predictions.npz'
    np.savez_compressed(
        require_external_output(train_predictions_path),
        logits=train_logits.astype(np.float32),
        y_true=train_y,
        y_pred=train_logits.argmax(axis=1).astype(np.int64),
        sample_uid=split['uid_tr'],
    )
    # Hub-compatible exports include both train and test split artifacts.
    artifacts.save(
        dataset, model_artifact_name, subject, seed, 'train',
        logits=train_logits, feats=train_feats, y=train_y,
        root=context['artifact_root'], sample_uid=split['uid_tr'],
        split_policy=SPLIT_POLICY,
    )
    artifacts.save(
        dataset, model_artifact_name, subject, seed, 'test',
        logits=test_logits, feats=test_feats, y=test_y,
        root=context['artifact_root'], sample_uid=split['uid_te'],
        split_policy=SPLIT_POLICY,
    )
    test_score.pop('predicted_labels')
    metrics = {
        'status': 'complete',
        'formal_result': True,
        'dataset': dataset,
        'subject': f'S{subject + 1}',
        'subject_index_zero_based': int(subject),
        'seed': int(seed),
        'protocol': 'fewshot',
        'initialization': init_mode,
        'run_id': context['run_id'],
        'artifact_model_name': model_artifact_name,
        'split_policy': SPLIT_POLICY,
        'train_count': int(len(train_y)),
        'test_count': int(len(test_y)),
        'accuracy': test_score['accuracy'],
        'balanced_accuracy': test_score['balanced_accuracy'],
        'kappa': test_score['kappa'],
        'predicted_class_counts': test_score['predicted_class_counts'],
        'true_class_counts': test_score['true_class_counts'],
        'single_class_prediction_collapse': test_score['single_class_prediction_collapse'],
        'train_accuracy_final_epoch': float(
            (train_logits.argmax(axis=1) == train_y).mean()
        ),
        'epochs_completed': epochs,
        'elapsed_seconds': float(time.time() - started),
        'config_sha256': cfg['config_sha256'],
        'recipe_sha256': _run_recipe(context, dataset)['recipe_sha256'],
        'checkpoint_path': str(checkpoint_path.resolve()),
        'history_path': str(history_path.resolve()),
        'prediction_path': str(predictions_path.resolve()),
        'train_prediction_path': str(train_predictions_path.resolve()),
        'standard_artifact_train': artifacts.artifact_path(
            dataset, model_artifact_name, subject, seed, 'train',
            root=context['artifact_root']),
        'standard_artifact_test': artifacts.artifact_path(
            dataset, model_artifact_name, subject, seed, 'test',
            root=context['artifact_root']),
        'weight_load_report': load_report,
        'test_evaluated_once_after_final_checkpoint': True,
    }
    _write_json(metrics_path, metrics)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(
        f'[ok] {dataset} S{subject + 1} seed={seed} init={init_mode} '
        f'acc={metrics["accuracy"]:.4f} bac={metrics["balanced_accuracy"]:.4f} '
        f'kappa={metrics["kappa"]:.4f} pred={metrics["predicted_class_counts"]}',
        flush=True,
    )
    return metrics


def _action_subjects(action: str, dataset: str, provided: list[int] | None) -> list[int]:
    if provided is not None:
        return list(provided)
    if action in ('preflight', 'smoke'):
        return [0]
    return list(range(int(config.load_dataset_config(dataset)['num_subjects'])))


def _run_action(args) -> None:
    device = _device(args.gpu) if args.action in ('preflight', 'smoke', 'train') else torch.device('cpu')
    context = _build_run_context(args.run_id, device)
    datasets = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    unknown_datasets = sorted(set(datasets) - set(DATASETS))
    if unknown_datasets:
        raise ValueError(f'Unknown dataset(s): {unknown_datasets}')
    if args.action in ('preflight', 'smoke', 'check-splits'):
        seeds = args.seeds or [666]
    elif args.action == 'train':
        seeds = args.seeds or [666]
    else:
        seeds = args.seeds or [666, 667, 668]
    if args.action == 'validate-config':
        validation = {
            'run_id': context['run_id'],
            'run_root': str(context['run_root']),
            'artifact_root': str(context['artifact_root']),
            'run_signature_sha256': context['manifest']['run_signature_sha256'],
            'model_config_yaml_sha256': context['model_yaml_sha256'],
            'dataset_config_yaml_sha256': context['dataset_yaml_hashes'],
            'runtime_versions': context['runtime_versions'],
            'source_hashes': context['source_hashes'],
            'pretrained_weight': context['manifest']['pretrained_weight'],
            'resolved_model_config_by_dataset': context['model_configs'],
            'runtime_backend_policy_by_dataset': context['manifest'].get(
                'runtime_backend_policy_by_dataset', {}
            ),
            'runtime_backend_state_by_dataset': context['manifest'].get(
                'runtime_backend_state_by_dataset', {}
            ),
            'gradient_clip_error_if_nonfinite': True,
            'dataset_split_by_dataset': {
                dataset: {
                    'val_split': context['dataset_configs'][dataset]['val_split'],
                    'session': SESSIONS[dataset],
                }
                for dataset in DATASETS
            },
            'gpu_training_started': False,
        }
        print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
        return

    context['runtime_backend_by_dataset'] = {}
    for dataset in datasets:
        context['runtime_backend_by_dataset'][dataset] = _apply_runtime_backend(
            context['model_configs'][dataset], device
        )
    _persist_run_context(context)
    print(
        f'[codebrain] action={args.action} init={args.init} device={device} '
        f'datasets={datasets} seeds={seeds} run_id={context["run_id"]}', flush=True,
    )
    if args.action == 'aggregate':
        _aggregate(context, datasets=datasets, seeds=seeds, init_mode=args.init)
        return
    if args.action == 'train' and args.epochs is not None:
        raise ValueError(
            '--epochs cannot override the fixed formal budget; '
            'use --action smoke for short runs'
        )

    for seed in seeds:
        for dataset in datasets:
            context['runtime_backend_by_dataset'][dataset] = _apply_runtime_backend(
                context['model_configs'][dataset], device
            )
            subjects = _action_subjects(args.action, dataset, args.subjects)
            n_subjects = int(config.load_dataset_config(dataset)['num_subjects'])
            if any(subject < 0 or subject >= n_subjects for subject in subjects):
                raise ValueError(f'Invalid subject key for {dataset}: {subjects}')
            for subject in subjects:
                if args.action == 'check-splits':
                    split = _subject_split(dataset, subject, seed)
                    model_cfg = context['model_configs'][dataset]
                    x_adapter = CodeBrainInputAdapter(
                        scale_divisor=model_cfg['scale_divisor']
                    )
                    Xp_tr = x_adapter.transform(split['X_tr'])
                    Xp_te = x_adapter.transform(split['X_te'])
                    manifest = _manifest(context, dataset, subject, seed, split, Xp_tr, Xp_te)
                    path = _run_dir(
                        context, args.init, dataset, subject, seed, 'split_checks'
                    )
                    resolved = _resolved_config(
                        context, dataset, args.init, seed, subject,
                        actual_epochs=model_cfg['epochs'],
                    )
                    resolved['input']['train_signal_stats'] = _training_signal_stats(split['X_tr'])
                    resolved['data_fingerprint'] = {
                        key: manifest[key] for key in (
                            'raw_train_X_sha256', 'raw_test_X_sha256',
                            'train_uid_sha256', 'test_uid_sha256',
                            'train_y_sha256', 'test_y_sha256', 'session',
                            'session_environment',
                        )
                    }
                    resolved['formal_result'] = False
                    resolved['phase'] = 'check-splits'
                    resolved['config_sha256'] = _digest(resolved)
                    _write_json(path / 'split_manifest.json', manifest)
                    _write_json(path / 'config_resolved.json', resolved)
                    print(
                        f'[split:ok] {dataset} S{subject + 1} seed={seed} '
                        f'n={len(split["y_tr"])}/{len(split["y_te"])}', flush=True,
                    )
                elif args.action == 'preflight':
                    _preflight_one(context, dataset, subject, seed, args.init, device)
                elif args.action == 'smoke':
                    # The shape/finite checks touch one batch from each split;
                    # the short update run itself only sees the train split.
                    _preflight_one(context, dataset, subject, seed, args.init, device)
                    _run_one(
                        context, dataset, subject, seed, args.init, device,
                        smoke=True, smoke_batches=args.smoke_batches,
                        epoch_override=args.epochs or 1,
                    )
                elif args.action == 'train':
                    if args.init == 'random' and seed != 666:
                        raise ValueError('Random-init comparison is configured for seed 666 only')
                    _run_one(context, dataset, subject, seed, args.init, device,
                             epoch_override=args.epochs)
                else:
                    raise ValueError(f'Unknown action {args.action!r}')


def _metric_values(y: np.ndarray, logits: np.ndarray) -> dict:
    result = _score(y, logits)
    result.pop('predicted_labels')
    return result


def _bootstrap_ci(values: np.ndarray, seed: int = 20260927,
                  samples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _aggregate(context: dict, datasets: list[str], seeds: list[int],
               init_mode: str) -> None:
    model_name = ARTIFACT_MODEL[init_mode]
    cell_rows = []
    subject_metric = {}
    for dataset in datasets:
        n_subjects = int(config.load_dataset_config(dataset)['num_subjects'])
        for subject in range(n_subjects):
            for seed in seeds:
                run_dir = _run_dir(context, init_mode, dataset, subject, seed, 'train')
                metrics_path = run_dir / 'metrics.json'
                if not metrics_path.is_file():
                    raise FileNotFoundError(f'Missing CodeBrain result: {metrics_path}')
                run_metrics = json.loads(resolve_local_file(metrics_path).read_text())
                if not run_metrics.get('formal_result'):
                    raise ValueError(f'Not a formal run: {metrics_path}')
                target = artifacts.load(
                    dataset, model_name, subject, seed, 'test',
                    root=context['artifact_root'],
                )
                for baseline in REFERENCE_MODELS:
                    ref = artifacts.load(dataset, baseline, subject, seed, 'test')
                    if not np.array_equal(target['y'], ref['y']):
                        raise ValueError(f'y mismatch in aggregate: {dataset} S{subject + 1} {seed} {baseline}')
                    if not np.array_equal(target['sample_uid'], ref['sample_uid']):
                        raise ValueError(f'UID mismatch in aggregate: {dataset} S{subject + 1} {seed} {baseline}')
                    if target['split_policy'] != ref['split_policy']:
                        raise ValueError(f'policy mismatch in aggregate: {dataset} S{subject + 1} {seed} {baseline}')
                models = {model_name: target}
                for baseline in REFERENCE_MODELS:
                    models[baseline] = artifacts.load(dataset, baseline, subject, seed, 'test')
                for candidate, artifact in models.items():
                    score = _metric_values(artifact['y'], artifact['logits'])
                    row = {
                        'dataset': dataset,
                        'subject': f'S{subject + 1}',
                        'subject_index_zero_based': subject,
                        'seed': seed,
                        'model': candidate,
                        'accuracy': score['accuracy'],
                        'balanced_accuracy': score['balanced_accuracy'],
                        'kappa': score['kappa'],
                        'predicted_class_counts': json.dumps(score['predicted_class_counts'], sort_keys=True),
                        'single_class_prediction_collapse': score['single_class_prediction_collapse'],
                    }
                    cell_rows.append(row)
                    subject_metric[(dataset, subject, seed, candidate)] = score

    aggregate_root = context['run_root'] / 'aggregate'
    aggregate_root.mkdir(parents=True, exist_ok=True)
    seed_tag = 'seed' + '-'.join(str(seed) for seed in seeds)
    first_round = list(seeds) == [666]
    cell_csv = aggregate_root / f'aggregate_cells_{init_mode}_{seed_tag}.csv'
    with cell_csv.open('w', newline='') as stream:
        fields = list(cell_rows[0]) if cell_rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cell_rows)

    metrics_names = ('accuracy', 'balanced_accuracy', 'kappa')
    summary_rows = []
    comparison_rows = []
    for scope, scope_datasets in [('overall', datasets), *((ds, [ds]) for ds in datasets)]:
        for candidate in (model_name, *REFERENCE_MODELS):
            for metric in metrics_names:
                vals = [
                    subject_metric[(dataset, subject, seed, candidate)][metric]
                    for dataset in scope_datasets
                    for subject in range(int(config.load_dataset_config(dataset)['num_subjects']))
                    for seed in seeds
                ]
                summary_rows.append({
                    'scope': scope,
                    'model': candidate,
                    'metric': metric,
                    'n_cells': len(vals),
                    'mean': float(np.mean(vals)) if vals else float('nan'),
                    'std_cell': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })
        for baseline in REFERENCE_MODELS:
            for metric in metrics_names:
                per_subject_deltas = []
                for dataset in scope_datasets:
                    n_subjects = int(config.load_dataset_config(dataset)['num_subjects'])
                    for subject in range(n_subjects):
                        ds = [
                            subject_metric[(dataset, subject, seed, model_name)][metric]
                            - subject_metric[(dataset, subject, seed, baseline)][metric]
                            for seed in seeds
                        ]
                        per_subject_deltas.append(float(np.mean(ds)))
                ci_lo, ci_hi = _bootstrap_ci(
                    np.asarray(per_subject_deltas),
                    seed=20260927 + len(scope) + len(baseline) + len(metric),
                )
                delta_arr = np.asarray(per_subject_deltas, dtype=np.float64)
                comparison_rows.append({
                    'scope': scope,
                    'metric': metric,
                    'codebrain_init': init_mode,
                    'baseline': baseline,
                    'n_subjects': len(per_subject_deltas),
                    'mean_paired_delta_codebrain_minus_baseline': float(delta_arr.mean()),
                    'paired_subject_bootstrap_95ci_low': ci_lo,
                    'paired_subject_bootstrap_95ci_high': ci_hi,
                    'subject_wins': int((delta_arr > 1e-12).sum()),
                    'subject_ties': int((np.abs(delta_arr) <= 1e-12).sum()),
                    'subject_losses': int((delta_arr < -1e-12).sum()),
                })
    summary_csv = aggregate_root / f'aggregate_summary_{init_mode}_{seed_tag}.csv'
    comparison_csv = aggregate_root / f'aggregate_paired_{init_mode}_{seed_tag}.csv'
    for path, rows in ((summary_csv, summary_rows), (comparison_csv, comparison_rows)):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    _write_json(aggregate_root / f'aggregate_{init_mode}_{seed_tag}.json', {
        'run_id': context['run_id'],
        'run_signature_sha256': context['manifest']['run_signature_sha256'],
        'artifact_root': str(context['artifact_root'].resolve()),
        'initialization': init_mode,
        'datasets': datasets,
        'seeds': seeds,
        'first_round_seed666_only': first_round,
        'cell_results': str(cell_csv.resolve()),
        'summary': str(summary_csv.resolve()),
        'paired_comparisons': str(comparison_csv.resolve()),
        'confidence_interval': 'subject-cluster bootstrap, 10,000 resamples, percentile 95%',
        'collapse_cells': [
            row for row in cell_rows if row['single_class_prediction_collapse']
        ],
    })
    print(
        f'[aggregate:ok] wrote {cell_csv}, {summary_csv}, {comparison_csv}',
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=(
        'validate-config', 'preflight', 'smoke', 'check-splits', 'train', 'aggregate',
    ), default='preflight')
    parser.add_argument('--run-id', required=True,
                        help='unique output namespace under results/codebrain/<run-id>')
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='zero-based subject indices; preflight/smoke default to subject 0')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--init', choices=('pretrained', 'random'), default='pretrained')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override epoch budget; intended for short smoke runs')
    parser.add_argument('--smoke-batches', type=int, default=2)
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.smoke_batches <= 0:
        parser.error('--smoke-batches must be positive')
    return args


def main(argv=None):
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))
    _run_action(parse_args(argv))


if __name__ == '__main__':
    main()
