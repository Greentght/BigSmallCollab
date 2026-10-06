"""2x2 causal pilot: pretraining features versus delayed task KD.

The runner deliberately lives outside the regular distillation dispatcher.  It
uses the same subject-wise few-shot split and IFNet training schedule for four
conditions:

``BASE_CE``
    CE for all 100 epochs.
``PREALIGN_ONLY``
    CE + a projected cosine feature loss to the frozen, *unfine-tuned*
    MIRepNet for epochs 1--10, then CE only.
``DELAYED_KD``
    CE for epochs 1--10, then CE + cached fine-tuned MIRepNet KD for epochs
    11--100.
``PREALIGN_THEN_KD``
    The first two schedules combined, but never simultaneously.

The pretraining checkpoint is used only for features.  The fine-tuned teacher
artifact is used only for train logits.  No test artifact is read for either
teacher.  All four conditions share the same split, IFNet initial state,
projection initial state, and precomputed 100-epoch batch schedule.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from collab import artifacts
from collab.seed import set_seed
from data import split as split_utils
from models import get_adapter


CONDITIONS = (
    'BASE_CE',
    'PREALIGN_ONLY',
    'DELAYED_KD',
    'PREALIGN_THEN_KD',
)
_DEFAULT_DATASETS = ('BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
_ALLOWED_DATASET_SCOPES = (_DEFAULT_DATASETS, ('BNCI2014001-4',))
DATASETS = _DEFAULT_DATASETS


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', type=int, default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--force', action='store_true')
    p.add_argument('--preflight-only', action='store_true',
                   help='validate the pretraining checkpoint and all train artifacts without training')
    return p.parse_args(argv)


def _load_cfg(path):
    global DATASETS
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError('experiment config must be a mapping')
    datasets = tuple(cfg.get('datasets') or ())
    if datasets not in _ALLOWED_DATASET_SCOPES:
        raise ValueError(f'config datasets must be the original four datasets or the isolated BNCI2014001-4 supplement, got {list(datasets)}')
    DATASETS = datasets
    if cfg.get('protocol') != 'fewshot':
        raise ValueError('this pilot only supports protocol=fewshot')
    if int(cfg.get('seed')) != 666:
        raise ValueError('this pilot is fixed to seed 666')
    if tuple(cfg.get('conditions') or ()) != CONDITIONS:
        raise ValueError(f'conditions must be exactly {list(CONDITIONS)}')
    if cfg.get('pre_teacher') != 'mirepnet' or cfg.get('ft_teacher') != 'mirepnet':
        raise ValueError('both teacher states must be MIRepNet')
    if cfg.get('student') != 'ifnet':
        raise ValueError('this first pilot is fixed to IFNet')
    if int(cfg.get('epochs')) != 100 or int(cfg.get('prealign_epochs')) != 10:
        raise ValueError('pilot requires epochs=100 and prealign_epochs=10')
    if float(cfg.get('lambda_general')) != 1.0:
        raise ValueError('lambda_general is pre-registered as 1.0')
    if float(cfg.get('temperature_kd')) != 2.0 or float(cfg.get('lam_kd')) != 0.5:
        raise ValueError('KD is fixed to T=2.0 and lam_kd=0.5')
    return cfg


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _hash_bytes(*parts):
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode('utf-8')
        h.update(part)
    return h.hexdigest()


def _hash_array(value):
    a = np.asarray(value)
    return _hash_bytes(str(a.dtype), str(tuple(a.shape)), np.ascontiguousarray(a).tobytes())


def _hash_uid_split(uid_tr, uid_te):
    return _hash_bytes(
        b'train', np.asarray(uid_tr, dtype=np.int64).tobytes(),
        b'test', np.asarray(uid_te, dtype=np.int64).tobytes())


def _hash_state(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(str(key).encode('utf-8'))
        value = state[key]
        if torch.is_tensor(value):
            a = value.detach().cpu().numpy()
            h.update(str(a.dtype).encode('utf-8'))
            h.update(str(tuple(a.shape)).encode('utf-8'))
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(repr(value).encode('utf-8'))
    return h.hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False, default=str) + '\n')


def _atomic_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _git_value(*args):
    try:
        return subprocess.check_output(
            ['git', *args], cwd=ROOT, text=True,
            stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f'error: {exc}'


def _device(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def _artifact_path(dataset, model, subject, seed, split, root):
    return Path(artifacts.artifact_path(dataset, model, subject, seed, split, root))


def _load_ft_logits(dataset, subject, seed, y_ref, uid_ref, root):
    """Load only fine-tuned teacher logits, labels and UIDs from train NPZ.

    The ``feats`` member is intentionally never accessed: T_ft contributes
    task logits only.  The function rejects test paths by construction and
    validates explicit UID alignment instead of relying on row order.
    """
    path = _artifact_path(dataset, 'mirepnet', subject, seed, 'train', root)
    if path.name.endswith('_test.npz') or path.name.endswith('_test.npz'):
        raise ValueError(f'refusing a test artifact: {path}')
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {'logits', 'y', 'sample_uid'}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f'{path}: missing required fields {sorted(missing)}')
        logits = np.asarray(payload['logits'], dtype=np.float32)
        labels = np.asarray(payload['y'], dtype=np.int64)
        uids = np.asarray(payload['sample_uid'], dtype=np.int64)
        policy = str(payload['split_policy'].item()) if 'split_policy' in payload.files else None
    if policy != split_utils.FEWSHOT_SPLIT_POLICY:
        raise ValueError(f'{path}: split_policy={policy!r}, expected fewshot')
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) != len(uids):
        raise ValueError(f'{path}: logits/y/sample_uid first dimensions disagree')
    if uids.ndim != 2 or uids.shape[1] != 2:
        raise ValueError(f'{path}: sample_uid must have shape (N, 2)')
    if len({tuple(x) for x in uids.tolist()}) != len(uids):
        raise ValueError(f'{path}: duplicate sample_uid')
    uid_ref = np.asarray(uid_ref, dtype=np.int64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    if set(map(tuple, uids.tolist())) != set(map(tuple, uid_ref.tolist())):
        raise ValueError(f'{path}: teacher/train UID sets differ')
    lookup = {tuple(row): i for i, row in enumerate(uids.tolist())}
    order = np.asarray([lookup[tuple(row)] for row in uid_ref.tolist()], dtype=np.int64)
    aligned_uids = uids[order]
    aligned_labels = labels[order]
    if not np.array_equal(aligned_uids, uid_ref):
        raise ValueError(f'{path}: UID reorder failed')
    if not np.array_equal(aligned_labels, y_ref):
        raise ValueError(f'{path}: labels disagree after UID alignment')
    if not np.isfinite(logits).all():
        raise ValueError(f'{path}: teacher logits contain NaN/Inf')
    return {
        'logits': logits[order],
        'y': aligned_labels,
        'sample_uid': aligned_uids,
        'path': str(path.resolve()),
        'sha256': _sha256_file(path),
        'uid_alignment_hash': _hash_array(aligned_uids),
        'reordered': bool(not np.array_equal(uids, uid_ref)),
    }


def _build_mirepnet(dataset, device, num_classes):
    cfg = config.load_model_config('mirepnet', dataset, protocol='fewshot')
    cfg.update(dataset_name=dataset, in_channels=45, samples=1000)
    adapter = get_adapter('mirepnet', device=str(device), **cfg)
    model = adapter.build(num_classes)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return adapter, model, cfg


@torch.no_grad()
def _pretrain_features(adapter, model, X_raw, batch_size, device):
    Xp = adapter.preprocess(X_raw)
    parts = []
    for start in range(0, len(Xp), batch_size):
        feat, _ = adapter.forward(model, Xp[start:start + batch_size].to(device))
        parts.append(feat.detach().cpu())
    out = torch.cat(parts, dim=0).numpy().astype(np.float32)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError(f'pretrained MIRepNet feature output invalid: {out.shape}')
    return out


def _make_student_adapter(dataset, device, channels):
    cfg = config.load_model_config('ifnet', dataset, protocol='fewshot')
    # The pilot fixes one schedule for all four conditions and all datasets.
    cfg.update(
        dataset_name=dataset,
        in_channels=int(channels),
        samples=1000,
        epochs=100,
        lr=0.001,
        weight_decay=0.01,
        batch_size=16,
        optimizer='adamw',
    )
    return get_adapter('ifnet', device=str(device), **cfg), cfg


def _capture_initial_states(adapter, Xp, num_classes, device, pre_dim, seed):
    set_seed(seed)
    model = adapter.build(num_classes)
    student_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        feat_probe, _ = adapter.forward(model, Xp[:min(2, len(Xp))].to(device))
    if feat_probe.ndim != 2:
        raise ValueError(f'IFNet feature output must be 2-D, got {tuple(feat_probe.shape)}')
    student_dim = int(feat_probe.shape[1])
    projection = nn.Linear(student_dim, pre_dim).to(device)
    projection_state = {k: v.detach().cpu().clone() for k, v in projection.state_dict().items()}
    del model, projection
    return student_state, projection_state, student_dim


def _make_schedule(n, batch_size, epochs, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    schedule = []
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=generator).numpy().astype(np.int64)
        batches = [perm[i:i + batch_size] for i in range(0, n, batch_size)]
        schedule.append(batches)
    return schedule


def _schedule_hash(schedule, uid_tr):
    h = hashlib.sha256()
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    for epoch, batches in enumerate(schedule, start=1):
        h.update(np.asarray([epoch], dtype=np.int64).tobytes())
        for batch in batches:
            h.update(uid_tr[np.asarray(batch, dtype=np.int64)].tobytes())
    return h.hexdigest()


@torch.no_grad()
def _feature_cosine(adapter, model, projection, Xp, pre_features, batch_size, device):
    model.eval()
    projection.eval()
    parts = []
    for start in range(0, len(Xp), batch_size):
        feat, _ = adapter.forward(model, Xp[start:start + batch_size].to(device))
        parts.append(projection(feat).detach().cpu())
    projected = torch.cat(parts, dim=0)
    target = torch.as_tensor(pre_features, dtype=torch.float32)
    return float(F.cosine_similarity(projected, target, dim=1, eps=1e-12).mean().item())


@torch.no_grad()
def _predict(adapter, model, Xp, batch_size, device):
    model.eval()
    logits_parts = []
    feature_parts = []
    for start in range(0, len(Xp), batch_size):
        feat, logits = adapter.forward(model, Xp[start:start + batch_size].to(device))
        feature_parts.append(feat.detach().cpu())
        logits_parts.append(logits.detach().cpu())
    return torch.cat(feature_parts, dim=0).numpy(), torch.cat(logits_parts, dim=0).numpy()


def _metrics(y, logits):
    pred = np.asarray(logits).argmax(axis=1)
    return {
        'accuracy': float((pred == np.asarray(y)).mean() * 100.0),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred) * 100.0),
        'kappa': float(cohen_kappa_score(y, pred)),
        'pred': pred,
    }


def _train_condition(condition, adapter, Xp_tr, y_tr, Xp_te, y_te,
                     pre_features, ft_logits, schedule, student_state,
                     projection_state, student_dim, cfg, device, seed):
    epochs = int(cfg['epochs'])
    warmup = int(cfg['prealign_epochs'])
    batch_size = int(cfg['batch_size'])
    num_classes = int(np.max(y_tr)) + 1
    set_seed(seed)
    model = adapter.build(num_classes)
    model.load_state_dict(student_state, strict=True)
    projection = nn.Linear(student_dim, pre_features.shape[1]).to(device)
    projection.load_state_dict(projection_state, strict=True)
    # The projection is instantiated in every condition.  It is included in
    # every optimizer so optimizer/module topology and initialization are
    # controlled; in the CE/KD-only conditions it receives no gradient.
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    y_tensor = torch.as_tensor(y_tr, dtype=torch.long)
    pre_tensor = torch.as_tensor(pre_features, dtype=torch.float32)
    ft_tensor = torch.as_tensor(ft_logits, dtype=torch.float32)
    history = []
    feature_cos = {'epoch_0': _feature_cosine(
        adapter, model, projection, Xp_tr, pre_features, batch_size, device)}
    model.train()
    projection.train()
    for epoch in range(1, epochs + 1):
        model.train()
        projection.train()
        ce_sum = align_sum = kd_sum = total_sum = 0.0
        n_batches = 0
        n_samples = 0
        align_active = condition in ('PREALIGN_ONLY', 'PREALIGN_THEN_KD') and epoch <= warmup
        kd_active = condition in ('DELAYED_KD', 'PREALIGN_THEN_KD') and epoch > warmup
        for batch in schedule[epoch - 1]:
            idx = np.asarray(batch, dtype=np.int64)
            xb = Xp_tr[idx].to(device)
            yb = y_tensor[idx].to(device)
            fb = pre_tensor[idx].to(device)
            lb = ft_tensor[idx].to(device)
            # Every condition executes one complete IFNet forward before any
            # optional teacher term is selected.
            feat_s, logits = adapter.forward(model, xb)
            ce = F.cross_entropy(logits, yb)
            loss = ce
            align = logits.sum() * 0.0
            kd = logits.sum() * 0.0
            if align_active:
                align = 1.0 - F.cosine_similarity(
                    F.normalize(projection(feat_s), dim=1),
                    F.normalize(fb.detach(), dim=1), dim=1, eps=1e-12).mean()
                loss = loss + float(cfg['lambda_general']) * align
            if kd_active:
                T = float(cfg['temperature_kd'])
                pt = F.softmax(lb.detach() / T, dim=1)
                kd = F.kl_div(
                    F.log_softmax(logits / T, dim=1), pt,
                    reduction='batchmean')
                loss = loss + float(cfg['lam_kd']) * (T ** 2) * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(idx)
            ce_sum += float(ce.detach().item()) * n
            align_sum += float(align.detach().item()) * n if align_active else 0.0
            kd_sum += float(kd.detach().item()) * n if kd_active else 0.0
            total_sum += float(loss.detach().item())
            n_samples += n
            n_batches += 1
        scheduler.step()
        entry = {
            'epoch': epoch,
            'phase': 'prealign' if epoch <= warmup else 'delayed_kd',
            'ce_loss': ce_sum / n_samples,
            'alignment_loss': align_sum / n_samples if align_active else None,
            'kd_loss': kd_sum / n_samples if kd_active else None,
            'total_loss': total_sum / n_batches,
            'lr': float(optimizer.param_groups[0]['lr']),
            'n_samples': n_samples,
        }
        history.append(entry)
        if epoch == warmup:
            feature_cos[f'epoch_{epoch}'] = _feature_cosine(
                adapter, model, projection, Xp_tr, pre_features, batch_size, device)
    feature_cos[f'epoch_{epochs}'] = _feature_cosine(
        adapter, model, projection, Xp_tr, pre_features, batch_size, device)
    train_features, train_logits = _predict(adapter, model, Xp_tr, batch_size, device)
    test_features, test_logits = _predict(adapter, model, Xp_te, batch_size, device)
    train_metrics = _metrics(y_tr, train_logits)
    test_metrics = _metrics(y_te, test_logits)
    return {
        'model': model,
        'projection': projection,
        'history': history,
        'feature_cosine': feature_cos,
        'train_logits': train_logits,
        'test_logits': test_logits,
        'train_features': train_features,
        'test_features': test_features,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }


def _run_key(dataset, subject, seed, condition):
    return f'{dataset}__S{int(subject) + 1}__seed{int(seed)}__{condition}'


def _paths(out_dir, dataset, subject, seed, condition):
    key = _run_key(dataset, subject, seed, condition)
    return {
        'history': out_dir / 'training_history' / f'{key}.json',
        'checkpoint': out_dir / 'checkpoints' / f'{key}.pt',
        'prediction': out_dir / 'predictions' / f'{key}.npz',
    }


def _row_complete(row, paths):
    if row.get('status') != 'complete':
        return False
    try:
        for name in ('accuracy', 'balanced_accuracy', 'kappa'):
            if not np.isfinite(float(row[name])):
                return False
        if int(row.get('epochs_completed', 0)) != 100:
            return False
    except (TypeError, ValueError, KeyError):
        return False
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _load_existing(out_dir):
    rows = _read_rows(out_dir / 'results_per_run.csv')
    return {(r.get('dataset'), r.get('subject'), r.get('seed'), r.get('condition')): r
            for r in rows}


def _dataset_subjects(dataset):
    return list(range(int(config.load_dataset_config(dataset)['num_subjects'])))


def _prepare_unit(dataset, subject, seed, cfg, device):
    dcfg = config.load_dataset_config(dataset)
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset, subject, val_split=float(cfg['val_split']), seed=seed, return_uid=True)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    uid_te = np.asarray(uid_te, dtype=np.int64)
    if len({tuple(x) for x in uid_tr.tolist()}) != len(uid_tr):
        raise ValueError(f'{dataset} S{subject + 1}: duplicate train UID')
    if len({tuple(x) for x in uid_te.tolist()}) != len(uid_te):
        raise ValueError(f'{dataset} S{subject + 1}: duplicate test UID')
    ft = _load_ft_logits(dataset, subject, seed, y_tr, uid_tr, cfg['artifact_root'])
    pre_adapter, pre_model, pre_cfg = _build_mirepnet(dataset, device, int(dcfg['num_classes']))
    pre_features = _pretrain_features(pre_adapter, pre_model, X_tr,
                                      int(cfg['batch_size']), device)
    pretrain_path = Path(pre_cfg.get('pretrain') or config.weight_path('mirepnet'))
    pretrain_info = {
        'path': str(pretrain_path.resolve()),
        'sha256': _sha256_file(pretrain_path),
        'feature_dim': int(pre_features.shape[1]),
        'feature_layer': 'MIRepNetAdapter.forward pooled feature',
        'checkpoint_kind': 'pretrained foundation state dict; classification head is not loaded',
    }
    del pre_model, pre_adapter
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    student_adapter, student_cfg = _make_student_adapter(dataset, device, X_tr.shape[1])
    Xp_tr = student_adapter.preprocess(X_tr)
    Xp_te = student_adapter.preprocess(X_te)
    student_state, projection_state, student_dim = _capture_initial_states(
        student_adapter, Xp_tr, int(dcfg['num_classes']), device,
        int(pre_features.shape[1]), seed)
    schedule = _make_schedule(len(y_tr), int(cfg['batch_size']), int(cfg['epochs']), seed)
    return {
        'dataset_cfg': dcfg,
        'X_tr': X_tr, 'y_tr': np.asarray(y_tr, dtype=np.int64),
        'X_te': X_te, 'y_te': np.asarray(y_te, dtype=np.int64),
        'uid_tr': uid_tr, 'uid_te': uid_te,
        'ft_logits': ft['logits'], 'ft_info': ft,
        'pre_features': pre_features,
        'pretrain_info': pretrain_info,
        'student_adapter': student_adapter, 'student_cfg': student_cfg,
        'Xp_tr': Xp_tr, 'Xp_te': Xp_te,
        'student_state': student_state, 'projection_state': projection_state,
        'student_dim': student_dim,
        'schedule': schedule,
        'split_uid_hash': _hash_uid_split(uid_tr, uid_te),
        'initial_state_hash': _hash_state(student_state),
        'projection_initial_state_hash': _hash_state(projection_state),
        'batch_order_hash': _schedule_hash(schedule, uid_tr),
    }


def _run_unit(dataset, subject, seed, condition, cfg, device, unit):
    start = time.time()
    out_dir = Path(cfg['output_dir'])
    paths = _paths(out_dir, dataset, subject, seed, condition)
    result = _train_condition(
        condition, unit['student_adapter'], unit['Xp_tr'], unit['y_tr'],
        unit['Xp_te'], unit['y_te'], unit['pre_features'], unit['ft_logits'],
        unit['schedule'], unit['student_state'], unit['projection_state'],
        unit['student_dim'], cfg, device, seed)
    train_metrics = result['train_metrics']
    test_metrics = result['test_metrics']
    key = _run_key(dataset, subject, seed, condition)
    history_payload = {
        'run_key': key,
        'dataset': dataset,
        'subject': int(subject),
        'subject_label': f'S{int(subject) + 1}',
        'seed': int(seed),
        'condition': condition,
        'feature_cosine': result['feature_cosine'],
        'epochs': result['history'],
        'schedule': {
            'epochs': int(cfg['epochs']),
            'prealign_epochs': int(cfg['prealign_epochs']),
            'phase_1': 'CE + projected cosine feature alignment' if condition in ('PREALIGN_ONLY', 'PREALIGN_THEN_KD') else 'CE',
            'phase_2': 'CE + fine-tuned teacher KD' if condition in ('DELAYED_KD', 'PREALIGN_THEN_KD') else 'CE',
        },
    }
    _atomic_json(paths['history'], history_payload)
    checkpoint_payload = {
        'complete': True,
        'epochs_completed': int(cfg['epochs']),
        'condition': condition,
        'dataset': dataset,
        'subject': int(subject),
        'seed': int(seed),
        'student_state_dict': result['model'].state_dict(),
        'projection_state_dict': result['projection'].state_dict(),
        'initial_state_hash': unit['initial_state_hash'],
        'projection_initial_state_hash': unit['projection_initial_state_hash'],
        'split_uid_hash': unit['split_uid_hash'],
        'batch_order_hash': unit['batch_order_hash'],
    }
    paths['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = paths['checkpoint'].with_name(f'.{paths["checkpoint"].name}.tmp-{os.getpid()}')
    torch.save(checkpoint_payload, tmp_ckpt)
    os.replace(tmp_ckpt, paths['checkpoint'])
    paths['prediction'].parent.mkdir(parents=True, exist_ok=True)
    tmp_pred = paths['prediction'].with_name(f'.{paths["prediction"].name}.tmp-{os.getpid()}')
    np.savez(
        tmp_pred,
        train_sample_uid=unit['uid_tr'], test_sample_uid=unit['uid_te'],
        y_train=unit['y_tr'], y_test=unit['y_te'],
        train_pred=train_metrics['pred'], test_pred=test_metrics['pred'],
        train_logits=result['train_logits'].astype(np.float32),
        test_logits=result['test_logits'].astype(np.float32),
    )
    # np.savez appends .npz when passed a string without the suffix; tmp_pred
    # already has no semantic suffix only when the generated name ends in .tmp.
    generated = Path(str(tmp_pred) + '.npz') if not tmp_pred.name.endswith('.npz') else tmp_pred
    os.replace(generated, paths['prediction'])
    runtime = time.time() - start
    row = {
        'dataset': dataset,
        'subject': int(subject) + 1,
        'subject_index': int(subject),
        'session': {'BNCI2014001': 'sessionT', 'BNCI2014001-4': 'sessionT',
                    'BNCI2014004': 'session3',
                    'BNCI2015001': 'session_A (loader default)',
                    'AlexMI': 'canonical subject block (loader default)'}[dataset],
        'protocol': 'fewshot',
        'seed': int(seed),
        'condition': condition,
        'result_source': 'new',
        'train_count': len(unit['y_tr']),
        'test_count': len(unit['y_te']),
        'accuracy': test_metrics['accuracy'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'kappa': test_metrics['kappa'],
        'train_accuracy': train_metrics['accuracy'],
        'feature_cosine_epoch_0': result['feature_cosine']['epoch_0'],
        'feature_cosine_epoch_10': result['feature_cosine']['epoch_10'],
        'feature_cosine_epoch_100': result['feature_cosine']['epoch_100'],
        'predicted_class_counts': json.dumps(
            np.bincount(test_metrics['pred'], minlength=int(unit['dataset_cfg']['num_classes'])).tolist()),
        'collapse_flag': bool(len(np.unique(test_metrics['pred'])) < int(unit['dataset_cfg']['num_classes'])),
        'runtime_seconds': runtime,
        'split_uid_hash': unit['split_uid_hash'],
        'initial_state_hash': unit['initial_state_hash'],
        'projection_initial_state_hash': unit['projection_initial_state_hash'],
        'batch_order_hash': unit['batch_order_hash'],
        'pretrain_checkpoint_sha256': unit['pretrain_info']['sha256'],
        'ft_teacher_artifact_sha256': unit['ft_info']['sha256'],
        'ft_teacher_uid_alignment_hash': unit['ft_info']['uid_alignment_hash'],
        'epochs_completed': int(cfg['epochs']),
        'status': 'complete',
        'failure_reason': '',
        'checkpoint_path': str(paths['checkpoint'].resolve()),
        'history_path': str(paths['history'].resolve()),
        'prediction_path': str(paths['prediction'].resolve()),
    }
    return row


def _bootstrap_ci(values, seed=666, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        means[i] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _comparison_rows(rows):
    pairs = [
        ('PREALIGN_ONLY', 'BASE_CE', 'general_vs_base'),
        ('DELAYED_KD', 'BASE_CE', 'delayed_kd_vs_base'),
        ('PREALIGN_THEN_KD', 'DELAYED_KD', 'joint_vs_delayed_kd'),
    ]
    metrics = ('accuracy', 'balanced_accuracy', 'kappa')
    out = []
    for dataset in DATASETS + ('ALL_DATASETS',):
        scope = [r for r in rows if dataset == 'ALL_DATASETS' or r['dataset'] == dataset]
        by_subject = {}
        for r in scope:
            by_subject.setdefault((r['dataset'], int(r['subject_index'])), {})[r['condition']] = r
        for a, b, label in pairs:
            for metric in metrics:
                vals = []
                for methods in by_subject.values():
                    if a in methods and b in methods:
                        vals.append(float(methods[a][metric]) - float(methods[b][metric]))
                ci_lo, ci_hi = _bootstrap_ci(vals)
                if metric == 'balanced_accuracy':
                    wins = sum(v > 1e-12 for v in vals)
                    ties = sum(abs(v) <= 1e-12 for v in vals)
                    losses = sum(v < -1e-12 for v in vals)
                else:
                    wins = ties = losses = ''
                out.append({
                    'dataset': dataset,
                    'comparison': label,
                    'better_condition': a,
                    'reference_condition': b,
                    'metric': metric,
                    'n_subjects': len(vals),
                    'mean_delta': float(np.mean(vals)) if vals else np.nan,
                    'bootstrap_ci_low': ci_lo,
                    'bootstrap_ci_high': ci_hi,
                    'win': wins,
                    'tie': ties,
                    'loss': losses,
                })
        # 2x2 interaction: (C3-C2) - (C1-C0)
        vals = []
        for methods in by_subject.values():
            needed = {'PREALIGN_THEN_KD', 'DELAYED_KD', 'PREALIGN_ONLY', 'BASE_CE'}
            if needed <= methods.keys():
                vals.append((float(methods['PREALIGN_THEN_KD']['balanced_accuracy'])
                             - float(methods['DELAYED_KD']['balanced_accuracy']))
                            - (float(methods['PREALIGN_ONLY']['balanced_accuracy'])
                               - float(methods['BASE_CE']['balanced_accuracy'])))
        ci_lo, ci_hi = _bootstrap_ci(vals)
        out.append({
            'dataset': dataset,
            'comparison': 'interaction_2x2',
            'better_condition': 'PREALIGN_THEN_KD_minus_DELAYED_KD',
            'reference_condition': 'PREALIGN_ONLY_minus_BASE_CE',
            'metric': 'balanced_accuracy',
            'n_subjects': len(vals),
            'mean_delta': float(np.mean(vals)) if vals else np.nan,
            'bootstrap_ci_low': ci_lo,
            'bootstrap_ci_high': ci_hi,
            'win': '', 'tie': '', 'loss': '',
        })
    return out


def _write_summaries(out_dir, rows, cfg, provenance):
    fields = list(rows[0].keys()) if rows else []
    _atomic_csv(out_dir / 'results_per_run.csv', rows, fields)
    _atomic_csv(out_dir.parent / 'prealign_delayed_kd_pilot.csv', rows, fields)
    subject_rows = []
    for row in rows:
        subject_rows.append(dict(row))
    _atomic_csv(out_dir / 'results_per_subject.csv', subject_rows, fields)
    ds_rows = []
    for dataset in DATASETS:
        subset = [r for r in rows if r['dataset'] == dataset]
        for condition in CONDITIONS:
            part = [r for r in subset if r['condition'] == condition]
            ds_rows.append({
                'dataset': dataset,
                'condition': condition,
                'n_subjects': len(part),
                'accuracy_mean': float(np.mean([r['accuracy'] for r in part])) if part else np.nan,
                'accuracy_sd': float(np.std([r['accuracy'] for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'balanced_accuracy_mean': float(np.mean([r['balanced_accuracy'] for r in part])) if part else np.nan,
                'balanced_accuracy_sd': float(np.std([r['balanced_accuracy'] for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'kappa_mean': float(np.mean([r['kappa'] for r in part])) if part else np.nan,
                'kappa_sd': float(np.std([r['kappa'] for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'feature_cosine_epoch_0_mean': float(np.mean([r['feature_cosine_epoch_0'] for r in part])) if part else np.nan,
                'feature_cosine_epoch_10_mean': float(np.mean([r['feature_cosine_epoch_10'] for r in part])) if part else np.nan,
                'feature_cosine_epoch_100_mean': float(np.mean([r['feature_cosine_epoch_100'] for r in part])) if part else np.nan,
                'collapse_count': sum(bool(r['collapse_flag']) for r in part),
            })
    _atomic_csv(out_dir / 'results_per_dataset.csv', ds_rows)
    comparisons = _comparison_rows(rows)
    _atomic_csv(out_dir / 'paired_comparisons.csv', comparisons)
    report = _render_report(rows, ds_rows, comparisons, cfg, provenance)
    _atomic_text(out_dir / 'report.md', report)


def _render_report(rows, ds_rows, comparisons, cfg, provenance):
    lines = [
        '# Pretrained Representation Alignment + Delayed KD Pilot', '',
        'This is a seed-666, subject-wise few-shot 2×2 experiment.', '',
        '## Design', '',
        '- `BASE_CE`: CE for epochs 1–100.',
        '- `PREALIGN_ONLY`: CE + projected cosine alignment to unfine-tuned MIRepNet for epochs 1–10, then CE.',
        '- `DELAYED_KD`: CE for epochs 1–10, then CE + Vanilla KD from fine-tuned MIRepNet for epochs 11–100.',
        '- `PREALIGN_THEN_KD`: pretraining-feature alignment for epochs 1–10, then delayed KD for epochs 11–100.',
        '- All conditions share the same split, IFNet initialization, projection initialization, batch schedule, optimizer and cosine scheduler.',
        '', 'The pretraining MIRepNet contributes pooled features only; the fine-tuned MIRepNet contributes cached train logits only. No teacher is updated and no test teacher artifact is read.', '',
        '## Overall test accuracy (%)', '',
        '| condition | mean |', '|---|---:|',
    ]
    for condition in CONDITIONS:
        part = [r['accuracy'] for r in rows if r['condition'] == condition]
        lines.append(f'| {condition} | {np.mean(part):.2f} |')
    lines += ['', '## Per-dataset test accuracy (%)', '', '| dataset | BASE_CE | PREALIGN_ONLY | DELAYED_KD | PREALIGN_THEN_KD |', '|---|---:|---:|---:|---:|']
    for dataset in DATASETS:
        vals = []
        for condition in CONDITIONS:
            part = [r['accuracy'] for r in rows if r['dataset'] == dataset and r['condition'] == condition]
            vals.append(f'{np.mean(part):.2f}' if part else 'NA')
        lines.append(f'| {dataset} | ' + ' | '.join(vals) + ' |')
    lines += ['', '## Interaction quantities', '',
              'The primary paired analysis uses subject-level test Balanced Accuracy:',
              '`general = PREALIGN_ONLY − BASE_CE`, `task = DELAYED_KD − BASE_CE`, `joint = PREALIGN_THEN_KD − DELAYED_KD`, and `interaction = joint − general`.', '']
    for dataset in DATASETS + ('ALL_DATASETS',):
        lines.append(f'### {dataset}')
        for r in comparisons:
            if r['dataset'] == dataset and r['metric'] == 'balanced_accuracy':
                lines.append(f"- {r['comparison']}: mean Δ={r['mean_delta']:.3f} pp, 95% bootstrap CI [{r['bootstrap_ci_low']:.3f}, {r['bootstrap_ci_high']:.3f}], n={r['n_subjects']}")
        lines.append('')
    lines += [
        '## Provenance and limitations', '',
        f"- Git commit: `{provenance.get('git_commit')}`.",
        f"- Pretraining checkpoint: `{provenance.get('pretrain_checkpoint')}` (SHA256 recorded in provenance and every run).",
        '- The checkpoint is the repository pretraining state dict and has no downstream classification head; only its pooled encoder feature is used.',
        '- Feature alignment is label-free, but it is computed on the same subject training samples as the student.',
        '- Fine-tuned teacher logits come from the read-only MIRepNet train artifacts and are UID-aligned before training.',
        '- Test data is used only for final student evaluation; no test signal selects weights, epochs or conditions.',
        '- `PREALIGN_ONLY` and `PREALIGN_THEN_KD` use a trainable linear projection from IFNet features to the 256-D MIRepNet space; the projection is instantiated in all four conditions for control.',
    ]
    return '\n'.join(lines) + '\n'


def _preflight(cfg, device):
    results = []
    for dataset in DATASETS:
        for subject in _dataset_subjects(dataset):
            X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
                dataset, subject, val_split=float(cfg['val_split']), seed=666, return_uid=True)
            ft = _load_ft_logits(dataset, subject, 666, y_tr, uid_tr, cfg['artifact_root'])
            dcfg = config.load_dataset_config(dataset)
            adapter, model, pcfg = _build_mirepnet(dataset, device, int(dcfg['num_classes']))
            feats = _pretrain_features(adapter, model, X_tr, int(cfg['batch_size']), device)
            path = Path(pcfg.get('pretrain') or config.weight_path('mirepnet'))
            result = {
                'dataset': dataset, 'subject': int(subject) + 1,
                'train_count': len(y_tr), 'test_count': len(y_te),
                'feature_shape': list(feats.shape),
                'feature_finite': bool(np.isfinite(feats).all()),
                'ft_logits_shape': list(ft['logits'].shape),
                'pretrain_checkpoint': str(path.resolve()),
                'pretrain_checkpoint_sha256': _sha256_file(path),
                'ft_teacher_artifact': ft['path'],
                'ft_teacher_artifact_sha256': ft['sha256'],
                'status': 'pass',
            }
            results.append(result)
            del model, adapter
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            expected_units = sum(len(_dataset_subjects(ds)) for ds in DATASETS)
            print(f'[preflight {len(results)}/{expected_units}] {dataset} S{subject + 1} feature={tuple(feats.shape)}', flush=True)
    out_dir = Path(cfg['output_dir'])
    _atomic_json(out_dir / 'preflight.json', {'status': 'pass', 'units': results})
    print(f'[preflight-complete] {len(results)}/{sum(len(_dataset_subjects(ds)) for ds in DATASETS)} units', flush=True)


def main(argv=None):
    args = _parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = _load_cfg(cfg_path)
    cfg['artifact_root'] = str((ROOT / cfg['artifact_root']).resolve()) if not os.path.isabs(cfg['artifact_root']) else cfg['artifact_root']
    cfg['output_dir'] = str((ROOT / cfg['output_dir']).resolve()) if not os.path.isabs(cfg['output_dir']) else cfg['output_dir']
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        device = _device(args.gpu)
        _preflight(cfg, device)
        return
    if any(out_dir.iterdir()) and not (args.resume or args.force):
        raise RuntimeError(f'output directory is non-empty; use --resume or --force: {out_dir}')
    device = _device(args.gpu)
    seed = int(cfg['seed'])
    existing = _load_existing(out_dir) if args.resume else {}
    rows = []
    skipped = 0
    provenance = {
        'git_commit': _git_value('rev-parse', 'HEAD'),
        'git_status_before': _git_value('status', '--short'),
        'config_path': str(cfg_path),
        'config_sha256': _sha256_file(cfg_path),
        'artifact_root': cfg['artifact_root'],
        'output_dir': str(out_dir),
        'device': str(device),
        'gpu_argument': args.gpu,
        'command': ' '.join(sys.argv),
        'datasets': list(DATASETS),
        'subject_count': sum(len(_dataset_subjects(ds)) for ds in DATASETS),
        'seed': seed,
        'conditions': list(CONDITIONS),
        'pretrain_checkpoint': str(Path(config.weight_path('mirepnet')).resolve()),
        'pretrain_checkpoint_sha256': _sha256_file(config.weight_path('mirepnet')),
        'teacher_test_artifacts_read': False,
        'training_runs_expected': sum(len(_dataset_subjects(ds)) for ds in DATASETS) * len(CONDITIONS),
    }
    _atomic_text(out_dir / 'config_resolved.yaml', yaml.safe_dump(
        cfg, sort_keys=False, allow_unicode=True))
    _atomic_json(out_dir / 'execution_provenance.json', provenance)
    all_subjects = [(ds, s) for ds in DATASETS for s in _dataset_subjects(ds)]
    expected_runs = len(all_subjects) * len(CONDITIONS)
    for unit_index, (dataset, subject) in enumerate(all_subjects, start=1):
        print(f'[unit {unit_index}/{len(all_subjects)}] preparing {dataset} S{subject + 1}', flush=True)
        unit = _prepare_unit(dataset, subject, seed, cfg, device)
        for condition in CONDITIONS:
            key = (dataset, str(subject + 1), str(seed), condition)
            prior = existing.get(key)
            paths = _paths(out_dir, dataset, subject, seed, condition)
            if args.resume and prior is not None and _row_complete(prior, paths):
                rows.append(prior)
                skipped += 1
                print(f'[skip complete {len(rows)}/{expected_runs}] {dataset} S{subject + 1} {condition}', flush=True)
                continue
            try:
                row = _run_unit(dataset, subject, seed, condition, cfg, device, unit)
                rows.append(row)
                _atomic_csv(out_dir / 'results_per_run.csv', rows, list(row.keys()))
                print(f'[progress {len(rows)}/{expected_runs}] {dataset} S{subject + 1} {condition} acc={row["accuracy"]:.2f}', flush=True)
            except Exception as exc:  # keep a visible failure and stop, never hide an invalid run
                failure = {
                    'dataset': dataset, 'subject': subject + 1,
                    'subject_index': subject, 'seed': seed,
                    'condition': condition, 'status': 'failed',
                    'failure_reason': repr(exc),
                }
                _atomic_json(out_dir / f'failure_{_run_key(dataset, subject, seed, condition)}.json', failure)
                raise
        del unit
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    _write_summaries(out_dir, rows, cfg, provenance)
    provenance['completed_runs'] = len(rows)
    provenance['skipped_complete_runs'] = skipped
    provenance['git_status_after'] = _git_value('status', '--short')
    _atomic_json(out_dir / 'execution_provenance.json', provenance)
    print(f'[complete] {len(rows)}/{expected_runs} runs; output={out_dir}', flush=True)


if __name__ == '__main__':
    main()
