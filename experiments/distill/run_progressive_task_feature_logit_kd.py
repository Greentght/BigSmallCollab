"""Progressive pretrained-to-finetuned feature/logit KD pilot.

Two new conditions are trained from a common IFNet/projection initialization:

``TASK_FEAT_LOGIT_KD``
    CE for epochs 1--10; CE + fine-tuned-teacher feature alignment + Vanilla
    KD for epochs 11--100.
``PREALIGN_THEN_TASK_FEAT_LOGIT_KD``
    CE + unfine-tuned-teacher feature alignment for epochs 1--10; the same
    fine-tuned-teacher feature alignment + Vanilla KD for epochs 11--100.

The pretraining teacher is loaded from the immutable foundation checkpoint and
is used only for pooled features.  The fine-tuned teacher contributes aligned
train features and logits from one read-only train artifact.  No test teacher
artifact, MI, mask, prototype, or sample filtering is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
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
from experiments.distill import run_prealign_delayed_kd as previous
from models import get_adapter
from experiments.storage import external_path, require_external_output, resolve_local_file


_DEFAULT_DATASETS = ('BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
_ALLOWED_DATASET_SCOPES = (_DEFAULT_DATASETS, ('BNCI2014001-4',))
DATASETS = _DEFAULT_DATASETS
NEW_CONDITIONS = (
    'TASK_FEAT_LOGIT_KD',
    'PREALIGN_THEN_TASK_FEAT_LOGIT_KD',
)
OLD_CONDITIONS = (
    'BASE_CE',
    'PREALIGN_ONLY',
    'DELAYED_KD',
    'PREALIGN_THEN_KD',
)
ALL_CONDITIONS = OLD_CONDITIONS + NEW_CONDITIONS
OLD_ROOT = Path('/data1/llx/BigSmallcollab/results') / 'distill' / 'prealign_delayed_kd_pilot'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--force', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='isolated two-epoch, one-subject smoke run in /tmp')
    return parser.parse_args(argv)


def load_config(path):
    global DATASETS
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    datasets = tuple(cfg.get('datasets') or ())
    if datasets not in _ALLOWED_DATASET_SCOPES:
        raise ValueError('config datasets must be the original four or the isolated BNCI2014001-4 supplement')
    DATASETS = datasets
    if cfg.get('protocol') != 'fewshot' or int(cfg.get('seed')) != 666:
        raise ValueError('pilot requires protocol=fewshot and seed=666')
    if cfg.get('pre_teacher') != 'mirepnet' or cfg.get('ft_teacher') != 'mirepnet':
        raise ValueError('both teacher states must be MIRepNet')
    if cfg.get('student') != 'ifnet':
        raise ValueError('student must be IFNet')
    if tuple(cfg.get('conditions') or ()) != NEW_CONDITIONS:
        raise ValueError(f'conditions must be exactly {list(NEW_CONDITIONS)}')
    if int(cfg.get('epochs')) != 100 or int(cfg.get('prealign_epochs')) != 10:
        raise ValueError('formal config requires epochs=100 and prealign_epochs=10')
    if float(cfg.get('lambda_pre')) != 1.0 or float(cfg.get('lambda_ft')) != 1.0:
        raise ValueError('lambda_pre and lambda_ft are fixed at 1.0')
    if float(cfg.get('temperature_kd')) != 2.0 or float(cfg.get('lam_kd')) != 0.5:
        raise ValueError('KD requires temperature=2.0 and lam_kd=0.5')
    return cfg


def _hash_bytes(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode('utf-8')
        digest.update(part)
    return digest.hexdigest()


def _hash_array(value):
    array = np.asarray(value)
    return _hash_bytes(str(array.dtype), str(tuple(array.shape)),
                       np.ascontiguousarray(array).tobytes())


def _hash_state(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode('utf-8'))
        value = state[key]
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
            digest.update(str(array.dtype).encode('utf-8'))
            digest.update(str(tuple(array.shape)).encode('utf-8'))
            digest.update(np.ascontiguousarray(array).tobytes())
        else:
            digest.update(repr(value).encode('utf-8'))
    return digest.hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    path = require_external_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False, default=str) + '\n')


def _atomic_csv(path, rows, fieldnames=None):
    path = require_external_output(path)
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _device(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def _run_key(dataset, subject, seed, condition):
    return f'{dataset}__S{int(subject) + 1}__seed{int(seed)}__{condition}'


def _paths(out_dir, dataset, subject, seed, condition):
    key = _run_key(dataset, subject, seed, condition)
    return {
        'history': out_dir / 'training_history' / f'{key}.json',
        'checkpoint': out_dir / 'checkpoints' / f'{key}.pt',
        'prediction': out_dir / 'predictions' / f'{key}.npz',
    }


def _artifact_path(dataset, subject, seed, root):
    return Path(artifacts.artifact_path(
        dataset, 'mirepnet', subject, seed, 'train', root))


def align_teacher_arrays(logits, feats, labels, uids, y_ref, uid_ref):
    """Explicitly reorder one train artifact to canonical student UID order."""
    logits = np.asarray(logits); feats = np.asarray(feats)
    labels = np.asarray(labels, dtype=np.int64)
    uids = np.asarray(uids, dtype=np.int64)
    uid_ref = np.asarray(uid_ref, dtype=np.int64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    if uids.ndim != 2 or uids.shape[1] != 2 or uid_ref.ndim != 2 or uid_ref.shape[1] != 2:
        raise ValueError('sample_uid must have shape (N,2)')
    if len({tuple(x) for x in uids.tolist()}) != len(uids):
        raise ValueError('teacher sample_uid contains duplicates')
    if len({tuple(x) for x in uid_ref.tolist()}) != len(uid_ref):
        raise ValueError('reference sample_uid contains duplicates')
    if set(map(tuple, uids.tolist())) != set(map(tuple, uid_ref.tolist())):
        raise ValueError('teacher/reference UID sets differ')
    lookup = {tuple(row): i for i, row in enumerate(uids.tolist())}
    order = np.asarray([lookup[tuple(row)] for row in uid_ref.tolist()], dtype=np.int64)
    aligned_uids = uids[order]
    aligned_labels = labels[order]
    if not np.array_equal(aligned_uids, uid_ref):
        raise ValueError('UID reorder failed')
    if not np.array_equal(aligned_labels, y_ref):
        raise ValueError('labels disagree after UID alignment')
    return logits[order], feats[order], aligned_labels, aligned_uids, bool(not np.array_equal(uids, uid_ref))


def _load_ft_teacher(dataset, subject, seed, y_ref, uid_ref, root):
    """Load aligned fine-tuned pooled features and logits from train NPZ only."""
    path = _artifact_path(dataset, subject, seed, root)
    if path.name.endswith('_test.npz'):
        raise ValueError(f'refusing test artifact {path}')
    with np.load(resolve_local_file(path), allow_pickle=False) as payload:
        required = {'logits', 'feats', 'y', 'sample_uid', 'split_policy'}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f'{path}: missing {sorted(missing)}')
        logits = np.asarray(payload['logits'], dtype=np.float32)
        feats = np.asarray(payload['feats'], dtype=np.float32)
        labels = np.asarray(payload['y'], dtype=np.int64)
        uids = np.asarray(payload['sample_uid'], dtype=np.int64)
        policy = str(payload['split_policy'].item())
    if policy != split_utils.FEWSHOT_SPLIT_POLICY:
        raise ValueError(f'{path}: unexpected split_policy={policy!r}')
    if logits.ndim != 2 or feats.ndim != 2 or feats.shape[1] != 256:
        raise ValueError(f'{path}: expected logits 2-D and pooled feats (N,256), got {logits.shape}/{feats.shape}')
    if len(logits) != len(feats) or len(feats) != len(labels) or len(labels) != len(uids):
        raise ValueError(f'{path}: first dimensions are not aligned')
    if uids.ndim != 2 or uids.shape[1] != 2:
        raise ValueError(f'{path}: sample_uid must have shape (N,2)')
    if len({tuple(x) for x in uids.tolist()}) != len(uids):
        raise ValueError(f'{path}: duplicate sample_uid')
    logits, feats, aligned_y, aligned_uid, reordered = align_teacher_arrays(
        logits, feats, labels, uids, y_ref, uid_ref)
    if not np.isfinite(logits).all() or not np.isfinite(feats).all():
        raise ValueError(f'{path}: teacher features/logits contain NaN/Inf')
    alignment_hash = _hash_bytes(
        np.ascontiguousarray(aligned_uid, dtype=np.int64).tobytes(),
        np.ascontiguousarray(aligned_y, dtype=np.int64).tobytes())
    return {
        'logits': logits, 'feats': feats, 'y': aligned_y,
        'sample_uid': aligned_uid, 'path': str(path.resolve()),
        'sha256': _hash_file(path), 'reordered': reordered,
        'uid_alignment_hash': alignment_hash,
        'feature_hash': _hash_array(feats),
        'logits_hash': _hash_array(logits),
    }


def feature_alignment_loss(student_features, teacher_features, projection,
                            eps=1e-8):
    """Mean normalized cosine loss used by both feature alignment stages."""
    projected = projection(student_features)
    student_norm = F.normalize(projected, dim=1, eps=eps)
    teacher_norm = F.normalize(teacher_features.detach(), dim=1, eps=eps)
    return 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=1,
                                     eps=eps).mean()


def vanilla_kd_loss(student_logits, teacher_logits, temperature=2.0):
    """KL(T_ft || student) at the registered temperature, teacher detached."""
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    teacher_prob, reduction='batchmean')


def phase_activity(condition, epoch, prealign_epochs=10):
    """Return (pre_alignment_active, ft_alignment_active, kd_active)."""
    first = int(epoch) <= int(prealign_epochs)
    pre = condition == 'PREALIGN_THEN_TASK_FEAT_LOGIT_KD' and first
    second = not first
    return pre, second, second


def make_projection_pair(student_dim, teacher_dim, device='cpu'):
    """Construct independent single Linear projections."""
    return (nn.Linear(student_dim, teacher_dim).to(device),
            nn.Linear(student_dim, teacher_dim).to(device))


def _capture_states(adapter, Xp, num_classes, device, teacher_dim, seed):
    set_seed(seed)
    model = adapter.build(num_classes)
    student_state = {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        probe, _ = adapter.forward(model, Xp[:min(2, len(Xp))].to(device))
    if probe.ndim != 2:
        raise ValueError(f'IFNet feature must be 2-D, got {tuple(probe.shape)}')
    p_pre, p_ft = make_projection_pair(int(probe.shape[1]), teacher_dim, device)
    pre_state = {k: v.detach().cpu().clone() for k, v in p_pre.state_dict().items()}
    ft_state = {k: v.detach().cpu().clone() for k, v in p_ft.state_dict().items()}
    return (student_state, pre_state, ft_state, int(probe.shape[1]))


def _make_schedule(n, batch_size, epochs, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    schedule = []
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=generator).numpy().astype(np.int64)
        schedule.append([order[i:i + batch_size]
                         for i in range(0, n, batch_size)])
    return schedule


def _schedule_hash(schedule, uid_tr):
    digest = hashlib.sha256()
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    for epoch, batches in enumerate(schedule, start=1):
        digest.update(np.asarray([epoch], dtype=np.int64).tobytes())
        for batch in batches:
            digest.update(uid_tr[np.asarray(batch, dtype=np.int64)].tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _transition_metrics(adapter, model, p_pre, p_ft, Xp, pre_feats, ft_feats,
                        ft_logits, y, batch_size, device, temperature=2.0,
                        eps=1e-8):
    model.eval(); p_pre.eval(); p_ft.eval()
    feat_parts, logit_parts = [], []
    for start in range(0, len(Xp), batch_size):
        feat, logits = adapter.forward(model, Xp[start:start + batch_size].to(device))
        feat_parts.append(feat.detach().cpu())
        logit_parts.append(logits.detach().cpu())
    student_feat = torch.cat(feat_parts, dim=0)
    student_logits = torch.cat(logit_parts, dim=0)
    projection_device = next(p_pre.parameters()).device
    student_feat_for_projection = student_feat.to(projection_device)
    pre_target = torch.as_tensor(pre_feats, dtype=torch.float32,
                                 device=projection_device)
    ft_target = torch.as_tensor(ft_feats, dtype=torch.float32,
                                device=projection_device)
    pre_cos = F.cosine_similarity(
        F.normalize(p_pre(student_feat_for_projection), dim=1, eps=eps),
        F.normalize(pre_target, dim=1, eps=eps), dim=1, eps=eps).mean().item()
    ft_cos = F.cosine_similarity(
        F.normalize(p_ft(student_feat_for_projection), dim=1, eps=eps),
        F.normalize(ft_target, dim=1, eps=eps), dim=1, eps=eps).mean().item()
    ft_logit = torch.as_tensor(ft_logits, dtype=torch.float32)
    kl = vanilla_kd_loss(student_logits, ft_logit, temperature).item()
    pred = student_logits.argmax(dim=1).numpy()
    train_acc = float((pred == np.asarray(y)).mean() * 100.0)
    return {
        'pre_cosine': float(pre_cos), 'ft_cosine': float(ft_cos),
        'ft_logit_kl': float(kl), 'train_accuracy': train_acc,
    }


@torch.no_grad()
def _predict(adapter, model, Xp, batch_size, device):
    model.eval()
    features, logits = [], []
    for start in range(0, len(Xp), batch_size):
        feat, lg = adapter.forward(model, Xp[start:start + batch_size].to(device))
        features.append(feat.detach().cpu())
        logits.append(lg.detach().cpu())
    return torch.cat(features, 0).numpy(), torch.cat(logits, 0).numpy()


def _metrics(y, logits):
    pred = np.asarray(logits).argmax(axis=1)
    return {
        'accuracy': float((pred == np.asarray(y)).mean() * 100.0),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred) * 100.0),
        'kappa': float(cohen_kappa_score(y, pred)),
        'pred': pred,
    }


def _prepare_unit(dataset, subject, seed, cfg, device):
    dcfg = config.load_dataset_config(dataset)
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset, subject, val_split=float(cfg['val_split']), seed=seed,
        return_uid=True)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    uid_te = np.asarray(uid_te, dtype=np.int64)
    ft = _load_ft_teacher(dataset, subject, seed, y_tr, uid_tr,
                          cfg['artifact_root'])

    pre_adapter, pre_model, pre_cfg = previous._build_mirepnet(
        dataset, device, int(dcfg['num_classes']))
    pre_feats = previous._pretrain_features(
        pre_adapter, pre_model, X_tr, int(cfg['batch_size']), device)
    if pre_feats.shape != ft['feats'].shape or pre_feats.shape[1] != 256:
        raise ValueError(f'{dataset} S{subject + 1}: pre/ft feature shape mismatch {pre_feats.shape}/{ft["feats"].shape}')
    pre_path = Path(pre_cfg.get('pretrain') or config.weight_path('mirepnet'))
    pre_info = {
        'path': str(pre_path.resolve()), 'sha256': _hash_file(pre_path),
        'feature_hash': _hash_array(pre_feats), 'feature_dim': 256,
        'feature_layer': 'MIRepNetAdapter.forward pooled feature',
    }
    del pre_model, pre_adapter
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    student_adapter, student_cfg = previous._make_student_adapter(
        dataset, device, X_tr.shape[1])
    Xp_tr = student_adapter.preprocess(X_tr)
    Xp_te = student_adapter.preprocess(X_te)
    student_state, p_pre_state, p_ft_state, student_dim = _capture_states(
        student_adapter, Xp_tr, int(dcfg['num_classes']), device, 256, seed)
    schedule = _make_schedule(len(y_tr), int(cfg['batch_size']),
                              int(cfg['epochs']), seed)
    preprocessing_hash = _hash_bytes(
        json.dumps(student_cfg, sort_keys=True, default=str),
        _hash_array(Xp_tr.cpu().numpy()), _hash_array(Xp_te.cpu().numpy()))
    return {
        'dataset_cfg': dcfg, 'X_tr': X_tr, 'y_tr': y_tr,
        'X_te': X_te, 'y_te': y_te, 'uid_tr': uid_tr, 'uid_te': uid_te,
        'pre_feats': pre_feats, 'pre_info': pre_info, 'ft': ft,
        'student_adapter': student_adapter, 'student_cfg': student_cfg,
        'Xp_tr': Xp_tr, 'Xp_te': Xp_te,
        'student_state': student_state, 'p_pre_state': p_pre_state,
        'p_ft_state': p_ft_state, 'student_dim': student_dim,
        'schedule': schedule,
        'split_uid_hash': previous._hash_uid_split(uid_tr, uid_te),
        'test_uid_hash': _hash_array(uid_te),
        'initial_student_state_hash': _hash_state(student_state),
        'initial_p_pre_state_hash': _hash_state(p_pre_state),
        'initial_p_ft_state_hash': _hash_state(p_ft_state),
        'batch_order_hash': _schedule_hash(schedule, uid_tr),
        'preprocessing_hash': preprocessing_hash,
        'teacher_uid_alignment_hash': ft['uid_alignment_hash'],
    }


def _train_condition(condition, unit, cfg, device, seed):
    start = time.time()
    epochs = int(cfg['epochs']); warmup = int(cfg['prealign_epochs'])
    bs = int(cfg['batch_size'])
    n_classes = int(unit['dataset_cfg']['num_classes'])
    adapter = unit['student_adapter']
    set_seed(seed)
    model = adapter.build(n_classes)
    model.load_state_dict(unit['student_state'], strict=True)
    p_pre, p_ft = make_projection_pair(unit['student_dim'], 256, device)
    p_pre.load_state_dict(unit['p_pre_state'], strict=True)
    p_ft.load_state_dict(unit['p_ft_state'], strict=True)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(p_pre.parameters()) + list(p_ft.parameters()),
        lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    Xp_tr = unit['Xp_tr']; Xp_te = unit['Xp_te']
    y_tensor = torch.as_tensor(unit['y_tr'], dtype=torch.long)
    pre_tensor = torch.as_tensor(unit['pre_feats'], dtype=torch.float32)
    ft_features = torch.as_tensor(unit['ft']['feats'], dtype=torch.float32)
    ft_logits = torch.as_tensor(unit['ft']['logits'], dtype=torch.float32)
    epoch_metrics = []
    transition = []
    transition.append({
        'epoch': 0,
        **_transition_metrics(adapter, model, p_pre, p_ft, Xp_tr,
                               unit['pre_feats'], unit['ft']['feats'],
                               unit['ft']['logits'], unit['y_tr'], bs, device,
                               float(cfg['temperature_kd']), float(cfg['feature_eps']))})
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); p_pre.train(); p_ft.train()
        pre_active, ft_active, kd_active = phase_activity(condition, epoch, warmup)
        sums = {'ce': 0.0, 'pre': 0.0, 'ft': 0.0, 'kd': 0.0, 'total': 0.0}
        correct = 0; seen = 0; n_batches = 0
        for batch in unit['schedule'][epoch - 1]:
            idx = np.asarray(batch, dtype=np.int64)
            xb = Xp_tr[idx].to(device); yb = y_tensor[idx].to(device)
            pre_b = pre_tensor[idx].to(device); ft_b = ft_features[idx].to(device)
            ft_l = ft_logits[idx].to(device)
            # One complete IFNet forward is shared by CE, features and logits.
            student_feat, student_logits = adapter.forward(model, xb)
            ce = F.cross_entropy(student_logits, yb)
            zero = student_logits.sum() * 0.0
            pre_loss = zero; ft_loss = zero; kd_loss = zero
            if pre_active:
                pre_loss = feature_alignment_loss(student_feat, pre_b, p_pre,
                                                  float(cfg['feature_eps']))
            if ft_active:
                ft_loss = feature_alignment_loss(student_feat, ft_b, p_ft,
                                                 float(cfg['feature_eps']))
            if kd_active:
                kd_loss = vanilla_kd_loss(
                    student_logits, ft_l, float(cfg['temperature_kd']))
            loss = (ce + float(cfg['lambda_pre']) * pre_loss
                    + float(cfg['lambda_ft']) * ft_loss
                    + float(cfg['lam_kd']) * float(cfg['temperature_kd']) ** 2 * kd_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            count = len(idx); seen += count; n_batches += 1
            correct += int((student_logits.detach().argmax(1) == yb).sum().item())
            sums['ce'] += float(ce.detach().item()) * count
            sums['pre'] += float(pre_loss.detach().item()) * count
            sums['ft'] += float(ft_loss.detach().item()) * count
            sums['kd'] += float(kd_loss.detach().item()) * count
            sums['total'] += float(loss.detach().item())
        scheduler.step()
        row = {
            'epoch': epoch,
            'phase': 'prealign' if epoch <= warmup else 'task_feature_logit_kd',
            'ce_loss': sums['ce'] / seen,
            'pre_feature_loss': sums['pre'] / seen if pre_active else 'NA',
            'ft_feature_loss': sums['ft'] / seen if ft_active else 'NA',
            'kd_loss': sums['kd'] / seen if kd_active else 'NA',
            'scaled_pre_feature_contribution': sums['pre'] / seen if pre_active else 'NA',
            'scaled_ft_feature_contribution': sums['ft'] / seen if ft_active else 'NA',
            'scaled_kd_contribution': (float(cfg['lam_kd']) * float(cfg['temperature_kd']) ** 2 * sums['kd'] / seen
                                       if kd_active else 'NA'),
            'total_loss': sums['total'] / n_batches,
            'train_accuracy': correct / seen * 100.0,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'n_samples': seen,
        }
        epoch_metrics.append(row); history.append(row)
        if epoch == warmup or epoch == epochs:
            transition.append({
                'epoch': epoch,
                **_transition_metrics(adapter, model, p_pre, p_ft, Xp_tr,
                                       unit['pre_feats'], unit['ft']['feats'],
                                       unit['ft']['logits'], unit['y_tr'], bs, device,
                                       float(cfg['temperature_kd']), float(cfg['feature_eps']))})
    train_feat, train_logits = _predict(adapter, model, Xp_tr, bs, device)
    test_feat, test_logits = _predict(adapter, model, Xp_te, bs, device)
    train_m = _metrics(unit['y_tr'], train_logits)
    test_m = _metrics(unit['y_te'], test_logits)
    return {
        'model': model, 'p_pre': p_pre, 'p_ft': p_ft,
        'history': history, 'epoch_metrics': epoch_metrics,
        'transition': transition,
        'train_logits': train_logits, 'test_logits': test_logits,
        'train_m': train_m, 'test_m': test_m,
        'runtime': time.time() - start,
    }


def _save_run(out_dir, dataset, subject, seed, condition, unit, result, cfg):
    out_dir = require_external_output(out_dir)
    paths = _paths(out_dir, dataset, subject, seed, condition)
    key = _run_key(dataset, subject, seed, condition)
    _atomic_json(paths['history'], {
        'run_key': key, 'condition': condition, 'dataset': dataset,
        'subject': int(subject), 'seed': int(seed),
        'epochs': result['history'], 'feature_transition': result['transition'],
    })
    checkpoint = {
        'complete': True, 'epochs_completed': int(cfg['epochs']),
        'condition': condition, 'student_state_dict': result['model'].state_dict(),
        'p_pre_state_dict': result['p_pre'].state_dict(),
        'p_ft_state_dict': result['p_ft'].state_dict(),
        'split_uid_hash': unit['split_uid_hash'],
        'test_uid_hash': unit['test_uid_hash'],
        'initial_student_state_hash': unit['initial_student_state_hash'],
        'initial_p_pre_state_hash': unit['initial_p_pre_state_hash'],
        'initial_p_ft_state_hash': unit['initial_p_ft_state_hash'],
        'batch_order_hash': unit['batch_order_hash'],
        'preprocessing_hash': unit['preprocessing_hash'],
    }
    paths['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths['checkpoint'].with_name(f'.{paths["checkpoint"].name}.tmp-{os.getpid()}')
    torch.save(checkpoint, require_external_output(tmp)); os.replace(tmp, paths['checkpoint'])
    paths['prediction'].parent.mkdir(parents=True, exist_ok=True)
    tmp_pred = paths['prediction'].with_name(f'.{paths["prediction"].name}.tmp-{os.getpid()}')
    np.savez(require_external_output(tmp_pred), train_sample_uid=unit['uid_tr'], test_sample_uid=unit['uid_te'],
             y_train=unit['y_tr'], y_test=unit['y_te'],
             train_pred=result['train_m']['pred'], test_pred=result['test_m']['pred'],
             train_logits=result['train_logits'].astype(np.float32),
             test_logits=result['test_logits'].astype(np.float32))
    generated = Path(str(tmp_pred) + '.npz') if not tmp_pred.name.endswith('.npz') else tmp_pred
    os.replace(generated, paths['prediction'])
    test_m = result['test_m']; train_m = result['train_m']
    row = {
        'dataset': dataset, 'subject': int(subject) + 1, 'subject_index': int(subject),
        'session': {'BNCI2014001': 'sessionT', 'BNCI2014001-4': 'sessionT',
                    'BNCI2014004': 'session3',
                    'BNCI2015001': 'session_A (loader default)',
                    'AlexMI': 'canonical subject block (loader default)'}[dataset],
        'protocol': 'fewshot', 'seed': int(seed), 'condition': condition,
        'result_source': 'new', 'train_count': len(unit['y_tr']),
        'test_count': len(unit['y_te']), 'accuracy': test_m['accuracy'],
        'balanced_accuracy': test_m['balanced_accuracy'], 'kappa': test_m['kappa'],
        'train_accuracy': train_m['accuracy'],
        'final_pre_cosine': result['transition'][-1]['pre_cosine'],
        'final_ft_cosine': result['transition'][-1]['ft_cosine'],
        'final_ft_logit_kl': result['transition'][-1]['ft_logit_kl'],
        'predicted_class_counts': json.dumps(
            np.bincount(test_m['pred'], minlength=int(unit['dataset_cfg']['num_classes'])).tolist()),
        'collapse_flag': bool(len(np.unique(test_m['pred'])) < int(unit['dataset_cfg']['num_classes'])),
        'runtime_seconds': result['runtime'],
        'split_uid_hash': unit['split_uid_hash'], 'test_uid_hash': unit['test_uid_hash'],
        'initial_student_state_hash': unit['initial_student_state_hash'],
        'initial_p_pre_state_hash': unit['initial_p_pre_state_hash'],
        'initial_p_ft_state_hash': unit['initial_p_ft_state_hash'],
        'batch_order_hash': unit['batch_order_hash'],
        'preprocessing_hash': unit['preprocessing_hash'],
        't_pre_checkpoint_sha256': unit['pre_info']['sha256'],
        't_pre_feature_hash': unit['pre_info']['feature_hash'],
        't_ft_artifact_sha256': unit['ft']['sha256'],
        't_ft_feature_hash': unit['ft']['feature_hash'],
        't_ft_logits_hash': unit['ft']['logits_hash'],
        'teacher_uid_alignment_hash': unit['teacher_uid_alignment_hash'],
        'epochs_completed': int(cfg['epochs']), 'status': 'complete',
        'failure_reason': '', 'checkpoint_path': str(paths['checkpoint'].resolve()),
        'history_path': str(paths['history'].resolve()),
        'prediction_path': str(paths['prediction'].resolve()),
    }
    return row


def _read_rows(path):
    path = external_path(path)
    if not Path(path).exists():
        return []
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _old_controls():
    rows = _read_rows(OLD_ROOT / 'results_per_run.csv')
    return {(r.get('dataset'), r.get('subject'), r.get('seed'), r.get('condition')): r for r in rows}, rows


def _validate_old_controls():
    old_map, rows = _old_controls()
    validation = []
    expected = sum(len(previous._dataset_subjects(ds)) for ds in DATASETS) * len(OLD_CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f'old pilot has {len(rows)} rows, expected {expected}')
    for row in rows:
        key = (row.get('dataset'), row.get('subject'), row.get('seed'), row.get('condition'))
        reason = []
        if row.get('condition') not in OLD_CONDITIONS:
            reason.append('unexpected_condition')
        if row.get('status') != 'complete' or int(row.get('epochs_completed', 0)) != 100:
            reason.append('incomplete_status_or_epochs')
        for field in ('accuracy', 'balanced_accuracy', 'kappa', 'split_uid_hash',
                      'initial_state_hash', 'projection_initial_state_hash',
                      'batch_order_hash'):
            if not row.get(field):
                reason.append(f'missing_{field}')
        for field in ('checkpoint_path', 'history_path', 'prediction_path'):
            if not Path(row.get(field, '')).is_file():
                reason.append(f'missing_{field}')
        if not reason:
            try:
                checkpoint = torch.load(resolve_local_file(row['checkpoint_path']), map_location='cpu')
                if not checkpoint.get('complete') or int(checkpoint.get('epochs_completed', 0)) != 100:
                    reason.append('checkpoint_not_complete_100_epochs')
            except Exception as exc:
                reason.append(f'checkpoint_unreadable:{type(exc).__name__}')
            try:
                with open(resolve_local_file(row['history_path'])) as handle:
                    history = json.load(handle)
                if len(history.get('epochs', [])) != 100:
                    reason.append('history_not_exactly_100_epochs')
            except Exception as exc:
                reason.append(f'history_unreadable:{type(exc).__name__}')
            try:
                with np.load(resolve_local_file(row['prediction_path']), allow_pickle=False) as prediction:
                    for name in ('train_sample_uid', 'test_sample_uid', 'train_pred', 'test_pred'):
                        if name not in prediction.files:
                            reason.append(f'prediction_missing_{name}')
                    if len(prediction['train_sample_uid']) != int(row['train_count']):
                        reason.append('prediction_train_count_mismatch')
                    if len(prediction['test_sample_uid']) != int(row['test_count']):
                        reason.append('prediction_test_count_mismatch')
                    if not np.isfinite(prediction['train_pred']).all() or not np.isfinite(prediction['test_pred']).all():
                        reason.append('prediction_nonfinite')
            except Exception as exc:
                reason.append(f'prediction_unreadable:{type(exc).__name__}')
            try:
                artifact_path = _artifact_path(
                    row['dataset'], int(row['subject_index']), int(row['seed']),
                    str(Path('/data1/llx/BigSmallcollab/results') / 'artifacts'))
                if row.get('ft_teacher_artifact_sha256') != _hash_file(artifact_path):
                    reason.append('teacher_artifact_sha256_mismatch')
            except Exception as exc:
                reason.append(f'teacher_artifact_unreadable:{type(exc).__name__}')
        validation.append({
            'dataset': row.get('dataset'), 'subject': row.get('subject'),
            'subject_index': row.get('subject_index'), 'seed': row.get('seed'),
            'condition': row.get('condition'),
            'status': 'pass' if not reason else 'fail',
            'reason': ';'.join(reason) if reason else 'legacy_prealign_runner_fields_valid',
            'preprocessing_hash_status': 'not_recorded_in_legacy_control',
            'test_uid_hash_status': 'not_recorded_in_legacy_control',
        })
    if any(r['status'] != 'pass' for r in validation):
        raise RuntimeError('old controls failed validation; refusing to train new runs')
    return old_map, validation


def _validate_unit_controls(old_map, dataset, subject, seed, unit):
    issues = []
    for condition in OLD_CONDITIONS:
        row = old_map.get((dataset, str(subject + 1), str(seed), condition))
        if row is None:
            issues.append(f'missing_{condition}')
            continue
        checks = {
            'split_uid_hash': unit['split_uid_hash'],
            'initial_state_hash': unit['initial_student_state_hash'],
            'projection_initial_state_hash': unit['initial_p_pre_state_hash'],
            'batch_order_hash': unit['batch_order_hash'],
            'pretrain_checkpoint_sha256': unit['pre_info']['sha256'],
            'ft_teacher_artifact_sha256': unit['ft']['sha256'],
        }
        for field, expected in checks.items():
            if row.get(field) != expected:
                issues.append(f'{condition}_{field}_mismatch')
    if issues:
        raise RuntimeError(f'{dataset} S{subject + 1}: old/new control hash mismatch: {issues}')


def _row_complete(row, out_dir):
    if row.get('status') != 'complete' or int(row.get('epochs_completed', 0)) != 100:
        return False
    for field in ('accuracy', 'balanced_accuracy', 'kappa'):
        try:
            if not np.isfinite(float(row[field])):
                return False
        except (ValueError, TypeError, KeyError):
            return False
    for field in ('checkpoint_path', 'history_path', 'prediction_path'):
        if not Path(row.get(field, '')).is_file():
            return False
    return True


def _bootstrap(values, seed=666, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _comparison_rows(combined):
    comparisons = [
        ('TASK_FEAT_LOGIT_KD', 'DELAYED_KD', 'task_feature_vs_delayed_kd'),
        ('PREALIGN_THEN_TASK_FEAT_LOGIT_KD', 'TASK_FEAT_LOGIT_KD', 'progressive_vs_task_feature'),
        ('PREALIGN_THEN_TASK_FEAT_LOGIT_KD', 'PREALIGN_THEN_KD', 'task_feature_repairs_old_prealign'),
        ('TASK_FEAT_LOGIT_KD', 'BASE_CE', 'task_feature_vs_base'),
        ('PREALIGN_THEN_TASK_FEAT_LOGIT_KD', 'BASE_CE', 'progressive_vs_base'),
    ]
    out = []
    for dataset in DATASETS + ('ALL_DATASETS',):
        scope = [r for r in combined if dataset == 'ALL_DATASETS' or r['dataset'] == dataset]
        by_subject = {}
        for row in scope:
            by_subject.setdefault((row['dataset'], str(row['subject_index'])), {})[row['condition']] = row
        for a, b, label in comparisons:
            for metric in ('accuracy', 'balanced_accuracy', 'kappa'):
                values = []
                for methods in by_subject.values():
                    if a in methods and b in methods:
                        values.append(float(methods[a][metric]) - float(methods[b][metric]))
                lo, hi = _bootstrap(values)
                wins = sum(v > 1e-12 for v in values)
                ties = sum(abs(v) <= 1e-12 for v in values)
                losses = sum(v < -1e-12 for v in values)
                out.append({
                    'dataset': dataset, 'comparison': label,
                    'better_condition': a, 'reference_condition': b,
                    'metric': metric, 'n_subjects': len(values),
                    'mean_delta': float(np.mean(values)) if values else np.nan,
                    'bootstrap_seed': 666, 'bootstrap_draws': 10000,
                    'bootstrap_ci_low': lo, 'bootstrap_ci_high': hi,
                    'win': wins, 'tie': ties, 'loss': losses,
                })
        # Interaction: (progressive-task) - (old prealign-delayed).
        values = []
        for methods in by_subject.values():
            needed = {'PREALIGN_THEN_TASK_FEAT_LOGIT_KD', 'TASK_FEAT_LOGIT_KD',
                      'PREALIGN_THEN_KD', 'DELAYED_KD'}
            if needed <= methods.keys():
                values.append((float(methods['PREALIGN_THEN_TASK_FEAT_LOGIT_KD']['balanced_accuracy'])
                               - float(methods['TASK_FEAT_LOGIT_KD']['balanced_accuracy']))
                              - (float(methods['PREALIGN_THEN_KD']['balanced_accuracy'])
                               - float(methods['DELAYED_KD']['balanced_accuracy'])))
        lo, hi = _bootstrap(values)
        out.append({
            'dataset': dataset, 'comparison': 'progressive_interaction_2x2',
            'better_condition': 'progressive_minus_task_feature',
            'reference_condition': 'old_prealign_minus_delayed_kd',
            'metric': 'balanced_accuracy', 'n_subjects': len(values),
            'mean_delta': float(np.mean(values)) if values else np.nan,
            'bootstrap_seed': 666, 'bootstrap_draws': 10000,
            'bootstrap_ci_low': lo, 'bootstrap_ci_high': hi,
            'win': '', 'tie': '', 'loss': '',
        })
    return out


def _render_report(new_rows, combined, comparisons, cfg, provenance,
                   transition_rows):
    lines = [
        '# Progressive Pretrained-to-Finetuned Feature–Logit Distillation Pilot', '',
        'Seed 666, subject-wise few-shot (30% train / 70% test), 100 epochs.', '',
        '## Conditions', '',
        '- `TASK_FEAT_LOGIT_KD`: CE epochs 1–10; CE + fine-tuned MIRepNet feature alignment + Vanilla KD epochs 11–100.',
        '- `PREALIGN_THEN_TASK_FEAT_LOGIT_KD`: CE + unfine-tuned MIRepNet feature alignment epochs 1–10; the same fine-tuned feature + logit losses epochs 11–100.',
        '- `P_pre` and `P_ft` are independent single Linear projections to 256 dimensions.',
        '- Four old controls are reused only after hash/status validation.', '',
        '## Overall test metrics', '',
        '| condition | Accuracy | Balanced Accuracy | Kappa |', '|---|---:|---:|---:|',
    ]
    for condition in ALL_CONDITIONS:
        part = [r for r in combined if r['condition'] == condition]
        lines.append(f"| {condition} | {np.mean([float(r['accuracy']) for r in part]):.2f} | {np.mean([float(r['balanced_accuracy']) for r in part]):.2f} | {np.mean([float(r['kappa']) for r in part]):.4f} |")
    lines += ['', '## New-condition per-dataset Accuracy (%)', '',
              '| dataset | TASK_FEAT_LOGIT_KD | PREALIGN_THEN_TASK_FEAT_LOGIT_KD |', '|---|---:|---:|']
    for dataset in DATASETS:
        vals = []
        for condition in NEW_CONDITIONS:
            part = [r for r in new_rows if r['dataset'] == dataset and r['condition'] == condition]
            vals.append(np.mean([float(r['accuracy']) for r in part]))
        lines.append(f'| {dataset} | {vals[0]:.2f} | {vals[1]:.2f} |')
    lines += ['', '## Core paired Balanced Accuracy comparisons', '',
              '| comparison | mean Δ (pp) | 95% bootstrap CI | n |', '|---|---:|---:|---:|']
    for row in comparisons:
        if row['dataset'] == 'ALL_DATASETS' and row['metric'] == 'balanced_accuracy':
            lines.append(f"| {row['comparison']} | {float(row['mean_delta']):.3f} | [{float(row['bootstrap_ci_low']):.3f}, {float(row['bootstrap_ci_high']):.3f}] | {row['n_subjects']} |")
    lines += ['', '## Transition diagnostics', '',
              'Transition rows report full-train, eval-mode cosine/teacher-logit KL only; they are not test-time metrics.', '',
              '| condition | pre cosine e0/e10/e100 | ft cosine e0/e10/e100 | ft logit KL e0/e10/e100 |', '|---|---|---|---|']
    for condition in NEW_CONDITIONS:
        def series(field):
            out = []
            for epoch in (0, 10, 100):
                values = [float(r[field]) for r in transition_rows
                          if r['condition'] == condition and int(r['epoch']) == epoch]
                out.append(f'{np.mean(values):.3f}' if values else 'NA')
            return '/'.join(out)
        lines.append(f'| {condition} | {series("pre_cosine")} | {series("ft_cosine")} | {series("ft_logit_kl")} |')
    lines += ['', '## Provenance and limitations', '',
              f"- Git commit: `{provenance.get('git_commit')}`.",
              f"- T_pre checkpoint: `{provenance.get('t_pre_checkpoint')}`; all pre features are frozen/detached pooled 256-D features.",
              '- T_ft feature and logits are read from the same MIRepNet train artifact and UID-aligned before training.',
              '- No teacher test artifact, MI, prototype, mask, sample deletion, or additional seed was used.',
              '- The T_pre checkpoint is identified by the repository pretraining path/state structure; no downstream subject head is loaded.',
              '- Feature cosine and KL transition diagnostics use train samples and must not be interpreted as test generalization.',
              '- Legacy controls passed available hash/status/artifact checks; their separate preprocessing/test-UID hash fields were not recorded by the legacy runner.',
    ]
    return '\n'.join(lines) + '\n'


def _write_outputs(out_dir, cfg, provenance, new_rows, old_rows, validation,
                   teacher_manifest, epoch_rows, transition_rows):
    new_fields = list(new_rows[0].keys()) if new_rows else []
    _atomic_csv(out_dir / 'results_per_run.csv', new_rows, new_fields)
    _atomic_csv(out_dir.parent / 'progressive_task_feature_logit_kd_pilot.csv', new_rows, new_fields)
    _atomic_csv(out_dir / 'run_manifest.csv', [
        {'run_key': _run_key(r['dataset'], int(r['subject_index']), int(r['seed']), r['condition']),
         'dataset': r['dataset'], 'subject': r['subject'], 'subject_index': r['subject_index'],
         'seed': r['seed'], 'condition': r['condition'], 'status': r['status'],
         'epochs_completed': r['epochs_completed']}
        for r in new_rows])
    _atomic_csv(out_dir / 'reused_controls.csv', [
        {**r, 'result_source': 'reused'} for r in old_rows])
    _atomic_csv(out_dir / 'control_validation.csv', validation)
    _atomic_csv(out_dir / 'teacher_artifact_manifest.csv', teacher_manifest)
    _atomic_csv(out_dir / 'epoch_metrics.csv', epoch_rows)
    _atomic_csv(out_dir / 'feature_transition_metrics.csv', transition_rows)
    # Historical rows came from the old runner and may carry its own
    # ``result_source`` value.  In this pilot they are controls, so normalize
    # the provenance explicitly rather than allowing an old value to leak into
    # the six-condition combined table.
    reused_rows = [{**r, 'result_source': 'reused'} for r in old_rows]
    combined = [*reused_rows, *new_rows]
    _atomic_csv(out_dir / 'combined_results_per_run.csv', combined)
    subject_rows = [dict(r) for r in combined]
    _atomic_csv(out_dir / 'results_per_subject.csv', subject_rows)
    dataset_rows = []
    for dataset in DATASETS:
        for condition in ALL_CONDITIONS:
            part = [r for r in combined if r['dataset'] == dataset and r['condition'] == condition]
            dataset_rows.append({
                'dataset': dataset, 'condition': condition, 'n_subjects': len(part),
                'accuracy_mean': float(np.mean([float(r['accuracy']) for r in part])) if part else np.nan,
                'accuracy_sd': float(np.std([float(r['accuracy']) for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'balanced_accuracy_mean': float(np.mean([float(r['balanced_accuracy']) for r in part])) if part else np.nan,
                'balanced_accuracy_sd': float(np.std([float(r['balanced_accuracy']) for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'kappa_mean': float(np.mean([float(r['kappa']) for r in part])) if part else np.nan,
                'kappa_sd': float(np.std([float(r['kappa']) for r in part], ddof=1)) if len(part) > 1 else np.nan,
                'collapse_count': sum(str(r.get('collapse_flag')).lower() == 'true' for r in part),
            })
    _atomic_csv(out_dir / 'results_per_dataset.csv', dataset_rows)
    comparisons = _comparison_rows(combined)
    _atomic_csv(out_dir / 'paired_comparisons.csv', comparisons)
    _atomic_text(out_dir / 'config_resolved.yaml', yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    _atomic_json(out_dir / 'execution_provenance.json', provenance)
    _atomic_text(out_dir / 'report.md', _render_report(
        new_rows, combined, comparisons, cfg, provenance, transition_rows))


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    global OLD_ROOT
    if cfg.get('old_root'):
        OLD_ROOT = Path(cfg['old_root'])
        if not OLD_ROOT.is_absolute(): OLD_ROOT = ROOT / OLD_ROOT
    smoke = bool(args.smoke)
    if smoke:
        cfg = dict(cfg)
        cfg['epochs'] = 2; cfg['prealign_epochs'] = 1
        cfg['output_dir'] = '/tmp/progressive_task_feature_logit_kd_smoke'
    cfg['artifact_root'] = str(external_path(cfg['artifact_root']))
    cfg['output_dir'] = str(require_external_output(cfg['output_dir']))
    out_dir = require_external_output(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not (args.resume or args.force or smoke):
        raise RuntimeError(f'output directory is non-empty; use --resume/--force: {out_dir}')
    device = _device(args.gpu)
    old_map = {}; old_rows = []; validation = []
    if not smoke:
        old_map, validation = _validate_old_controls()
        old_rows = list(old_map.values())
    seed = int(cfg['seed'])
    provenance = {
        'git_commit': previous._git_value('rev-parse', 'HEAD'),
        'git_status_before': previous._git_value('status', '--short'),
        'config_path': str(cfg_path), 'config_sha256': _hash_file(cfg_path),
        'output_dir': str(out_dir), 'artifact_root': cfg['artifact_root'],
        'device': str(device), 'gpu_argument': args.gpu,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set'),
        'logical_device': str(device),
        'launch_command': ' '.join(sys.argv),
        'datasets': list(DATASETS), 'subject_count': sum(len(previous._dataset_subjects(ds)) for ds in DATASETS), 'seed': seed,
        'conditions_new': list(NEW_CONDITIONS), 'conditions_reused': list(OLD_CONDITIONS),
        't_pre_checkpoint': str(Path(config.weight_path('mirepnet')).resolve()),
        't_pre_checkpoint_sha256': _hash_file(config.weight_path('mirepnet')),
        'teacher_test_artifacts_read': False, 'formal_runs_expected': sum(len(previous._dataset_subjects(ds)) for ds in DATASETS) * len(NEW_CONDITIONS),
        'smoke': smoke,
    }
    previous._atomic_text(out_dir / 'git_status_before.txt', provenance['git_status_before'] + '\n')
    _atomic_json(out_dir / 'execution_provenance.json', provenance)
    _atomic_text(out_dir / 'config_resolved.yaml', yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    existing = {}
    if args.resume and (out_dir / 'results_per_run.csv').is_file():
        existing = {(r.get('dataset'), r.get('subject'), r.get('seed'), r.get('condition')): r
                    for r in _read_rows(out_dir / 'results_per_run.csv')}
    new_rows = []
    epoch_rows = []; transition_rows = []; teacher_manifest = []
    if smoke:
        units = [('BNCI2014001', 0)]
    else:
        units = [(ds, subject) for ds in DATASETS
                 for subject in previous._dataset_subjects(ds)]
    expected = len(units) * len(NEW_CONDITIONS)
    for unit_idx, (dataset, subject) in enumerate(units, start=1):
        print(f'[unit {unit_idx}/{len(units)}] preparing {dataset} S{subject + 1}', flush=True)
        unit = _prepare_unit(dataset, subject, seed, cfg, device)
        if not smoke:
            _validate_unit_controls(old_map, dataset, subject, seed, unit)
        teacher_manifest.append({
            'dataset': dataset, 'subject': subject + 1, 'subject_index': subject,
            'seed': seed, 'train_count': len(unit['y_tr']),
            't_pre_checkpoint_sha256': unit['pre_info']['sha256'],
            't_pre_feature_hash': unit['pre_info']['feature_hash'],
            't_pre_feature_shape': json.dumps(list(unit['pre_feats'].shape)),
            't_ft_artifact_path': unit['ft']['path'],
            't_ft_artifact_sha256': unit['ft']['sha256'],
            't_ft_feature_hash': unit['ft']['feature_hash'],
            't_ft_logits_hash': unit['ft']['logits_hash'],
            'teacher_uid_alignment_hash': unit['teacher_uid_alignment_hash'],
            'alignment_status': 'pass', 'test_artifact_read': False,
        })
        for condition in NEW_CONDITIONS:
            key = (dataset, str(subject + 1), str(seed), condition)
            prior = existing.get(key)
            if args.resume and prior is not None and not smoke and _row_complete(prior, out_dir):
                row = prior; new_rows.append(row)
                print(f'[skip complete {len(new_rows)}/{expected}] {dataset} S{subject + 1} {condition}', flush=True)
                continue
            result = _train_condition(condition, unit, cfg, device, seed)
            row = _save_run(out_dir, dataset, subject, seed, condition, unit, result, cfg)
            new_rows.append(row)
            for e in result['epoch_metrics']:
                epoch_rows.append({'dataset': dataset, 'subject': subject + 1,
                                   'subject_index': subject, 'seed': seed,
                                   'condition': condition, **e})
            for t in result['transition']:
                transition_rows.append({'dataset': dataset, 'subject': subject + 1,
                                        'subject_index': subject, 'seed': seed,
                                        'condition': condition, **t})
            _atomic_csv(out_dir / 'results_per_run.csv', new_rows, list(row.keys()))
            print(f'[progress {len(new_rows)}/{expected}] {dataset} S{subject + 1} {condition} acc={row["accuracy"]:.2f}', flush=True)
        del unit
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    # On resume, recover epoch/transition rows from histories for all rows.
    if args.resume and not smoke and len(epoch_rows) != len(new_rows) * 100:
        epoch_rows = []; transition_rows = []
        for row in new_rows:
            with open(resolve_local_file(row['history_path'])) as handle:
                hist = json.load(handle)
            for e in hist['epochs']:
                epoch_rows.append({'dataset': row['dataset'], 'subject': row['subject'],
                                   'subject_index': row['subject_index'], 'seed': row['seed'],
                                   'condition': row['condition'], **e})
            for t in hist['feature_transition']:
                transition_rows.append({'dataset': row['dataset'], 'subject': row['subject'],
                                        'subject_index': row['subject_index'], 'seed': row['seed'],
                                        'condition': row['condition'], **t})
    _write_outputs(out_dir, cfg, provenance, new_rows, old_rows, validation,
                   teacher_manifest, epoch_rows, transition_rows)
    provenance['completed_new_runs'] = len(new_rows)
    provenance['combined_runs'] = len(new_rows) + len(old_rows)
    provenance['git_status_after'] = previous._git_value('status', '--short')
    _atomic_json(out_dir / 'execution_provenance.json', provenance)
    print(f'[complete] {len(new_rows)}/{expected} new runs; combined={len(new_rows) + len(old_rows)}; output={out_dir}', flush=True)


if __name__ == '__main__':
    main()
