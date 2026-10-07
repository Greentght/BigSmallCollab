"""Six-pair task feature + logit distillation pilot (seed 666).

This runner is intentionally independent from the earlier MI/prototype and
pretrained-to-finetuned pilots.  It consumes only read-only *train* artifacts
for the fine-tuned teachers and trains the three small students from a common
initial state per dataset/subject.  Epochs 1--10 are CE-only warm-up; epochs
11--100 add the fine-tuned teacher's vanilla logit KD and feature alignment.
"""

from __future__ import annotations

import argparse
import csv
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
from models import get_adapter
from experiments.storage import external_path, require_external_output, resolve_local_file


_DEFAULT_DATASETS = ('BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'AlexMI')
_ALLOWED_DATASET_SCOPES = (_DEFAULT_DATASETS, ('BNCI2014001-4',))
_ALL_SUBJECT_COUNTS = {
    'BNCI2014001': 9,
    'BNCI2014001-4': 9,
    'BNCI2014004': 9,
    'BNCI2015001': 12,
    'AlexMI': 8,
}
DATASETS = _DEFAULT_DATASETS
SUBJECT_COUNTS = {name: _ALL_SUBJECT_COUNTS[name] for name in DATASETS}
TEACHERS = ('mirepnet', 'cbramod')
STUDENTS = ('ifnet', 'eegnet', 'adfcnn')
CONDITIONS = ('BASE_CE', 'DELAYED_LOGIT_KD', 'TASK_FEAT_LOGIT_KD')
TEACHER_CONDITIONS = ('DELAYED_LOGIT_KD', 'TASK_FEAT_LOGIT_KD')
SEED = 666
EPOCHS = 100
WARMUP_EPOCHS = 10
TEMPERATURE_KD = 2.0
LAM_KD = 0.5
LAM_FEATURE = 1.0
FEATURE_EPS = 1e-8
OUTPUT_ROOT = Path('/data1/llx/BigSmallcollab/results') / 'distill' / 'task_feature_logit_kd_six_pairs_seed666'
MAIN_CSV = Path('/data1/llx/BigSmallcollab/results') / 'distill' / 'task_feature_logit_kd_six_pairs_seed666.csv'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args(argv)


def _hash_bytes(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode('utf-8')
        digest.update(part)
    return digest.hexdigest()


def hash_array(value):
    array = np.asarray(value)
    return _hash_bytes(str(array.dtype), str(tuple(array.shape)),
                       np.ascontiguousarray(array).tobytes())


def hash_state(state):
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


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_uid_split(uid_tr, uid_te):
    return _hash_bytes(b'train', np.asarray(uid_tr, dtype=np.int64).tobytes(),
                       b'test', np.asarray(uid_te, dtype=np.int64).tobytes())


def combined_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def atomic_text(path, text):
    path = require_external_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True,
                                 ensure_ascii=False, default=str) + '\n')


def atomic_csv(path, rows, fieldnames=None):
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


def device_for(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def session_for(dataset):
    return {
        'BNCI2014001': 'sessionT',
        'BNCI2014001-4': 'sessionT',
        'BNCI2014004': 'session3',
        'BNCI2015001': 'session_A (loader default)',
        'AlexMI': 'canonical subject block (loader default)',
    }[dataset]


def validate_config(path):
    global DATASETS, SUBJECT_COUNTS
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    datasets = tuple(cfg.get('datasets') or ())
    if datasets not in _ALLOWED_DATASET_SCOPES:
        raise ValueError('datasets must be the original four or the isolated BNCI2014001-4 supplement')
    DATASETS = datasets
    SUBJECT_COUNTS = {name: _ALL_SUBJECT_COUNTS[name] for name in DATASETS}
    if tuple(cfg.get('teachers') or ()) != TEACHERS:
        raise ValueError(f'teachers must be exactly {list(TEACHERS)}')
    if tuple(cfg.get('students') or ()) != STUDENTS:
        raise ValueError(f'students must be exactly {list(STUDENTS)}')
    if tuple(cfg.get('conditions') or ()) != CONDITIONS:
        raise ValueError(f'conditions must be exactly {list(CONDITIONS)}')
    if cfg.get('protocol') != 'fewshot' or int(cfg.get('seed')) != SEED:
        raise ValueError('pilot requires protocol=fewshot and seed=666')
    for key, expected in (
        ('epochs', EPOCHS), ('warmup_epochs', WARMUP_EPOCHS),
        ('temperature_kd', TEMPERATURE_KD), ('lam_kd', LAM_KD),
        ('lam_feature', LAM_FEATURE), ('feature_eps', FEATURE_EPS),
    ):
        if abs(float(cfg.get(key)) - expected) > 1e-12:
            raise ValueError(f'{key} must be {expected}, got {cfg.get(key)}')
    if str(cfg.get('scheduler')) != 'CosineAnnealingLR':
        raise ValueError('scheduler must be CosineAnnealingLR')
    if cfg.get('drop_last') is not False:
        raise ValueError('drop_last must be false')
    return cfg


def artifact_path(dataset, teacher, subject, root):
    root = external_path(root)
    if not root.is_absolute():
        root = ROOT / root
    path = root / dataset / teacher / f'{int(subject)}_{SEED}_train.npz'
    if path.name.endswith('_test.npz') or not path.name.endswith('_train.npz'):
        raise ValueError(f'refusing non-train teacher artifact: {path}')
    return path


def align_teacher_artifact(path, y_ref, uid_ref):
    """Read and explicitly reorder one train artifact by canonical UID."""
    if path.name.endswith('_test.npz'):
        raise ValueError(f'refusing teacher test artifact: {path}')
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
    y_ref = np.asarray(y_ref, dtype=np.int64)
    uid_ref = np.asarray(uid_ref, dtype=np.int64)
    if policy != 'fewshot_stratified_random':
        raise ValueError(f'{path}: unexpected split_policy={policy!r}')
    if logits.ndim != 2 or feats.ndim != 2:
        raise ValueError(f'{path}: logits/feats must be 2-D')
    if len(logits) != len(feats) or len(feats) != len(labels) or len(labels) != len(uids):
        raise ValueError(f'{path}: first dimensions disagree')
    if uids.ndim != 2 or uids.shape[1] != 2 or uid_ref.ndim != 2 or uid_ref.shape[1] != 2:
        raise ValueError(f'{path}: sample_uid must have shape (N,2)')
    if len({tuple(x) for x in uids.tolist()}) != len(uids):
        raise ValueError(f'{path}: duplicate teacher UID')
    if len({tuple(x) for x in uid_ref.tolist()}) != len(uid_ref):
        raise ValueError('current train UID is duplicated')
    if set(map(tuple, uids.tolist())) != set(map(tuple, uid_ref.tolist())):
        raise ValueError(f'{path}: UID sets differ')
    lookup = {tuple(row): i for i, row in enumerate(uids.tolist())}
    order = np.asarray([lookup[tuple(row)] for row in uid_ref.tolist()], dtype=np.int64)
    aligned_uids = uids[order]
    aligned_y = labels[order]
    if not np.array_equal(aligned_uids, uid_ref):
        raise ValueError(f'{path}: UID reorder failed')
    if not np.array_equal(aligned_y, y_ref):
        raise ValueError(f'{path}: labels disagree after UID alignment')
    if not np.isfinite(logits).all() or not np.isfinite(feats).all():
        raise ValueError(f'{path}: teacher logits/features contain NaN/Inf')
    alignment_hash = _hash_bytes(
        np.ascontiguousarray(aligned_uids, dtype='<i8').tobytes(),
        np.ascontiguousarray(aligned_y, dtype='<i8').tobytes())
    return {
        'logits': logits[order], 'feats': feats[order], 'y': aligned_y,
        'sample_uid': aligned_uids, 'path': str(path.resolve()),
        'artifact_sha256': hash_file(path),
        'feature_hash': hash_array(feats[order]),
        'logits_hash': hash_array(logits[order]),
        'alignment_hash': alignment_hash,
        'reordered': bool(not np.array_equal(uids, uid_ref)),
        'split_policy': policy,
    }


def student_config(dataset, student, X_tr):
    cfg = config.load_model_config(student, dataset, 'fewshot')
    # Epoch count is fixed by this pilot; all other optimizer/preprocessing
    # values remain the Student's canonical few-shot configuration.
    cfg.update(dataset_name=dataset, in_channels=int(X_tr.shape[1]),
               samples=int(X_tr.shape[2]), epochs=EPOCHS)
    return cfg


def make_schedule(n, batch_size, epochs, seed=SEED):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    schedule = []
    for _ in range(int(epochs)):
        order = torch.randperm(int(n), generator=generator).numpy().astype(np.int64)
        schedule.append([order[i:i + int(batch_size)]
                         for i in range(0, len(order), int(batch_size))])
    return schedule


def schedule_hashes(schedule, uid_tr):
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    hashes = []
    for batches in schedule:
        digest = hashlib.sha256()
        for batch in batches:
            digest.update(uid_tr[np.asarray(batch, dtype=np.int64)].tobytes())
        hashes.append(digest.hexdigest())
    return hashes, combined_hash(hashes)


def capture_student_state(adapter, Xp, num_classes, student_cfg, teacher_dim=None):
    set_seed(SEED)
    model = adapter.build(int(num_classes))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        feat, _ = adapter.forward(model, Xp[:min(2, len(Xp))])
    if feat.ndim != 2:
        raise ValueError(f'student feature must be 2-D, got {tuple(feat.shape)}')
    student_dim = int(feat.shape[1])
    projection_state = None
    if teacher_dim is not None:
        # Projection state is captured independently of model state and then
        # loaded into both teacher-related conditions for this pair.
        set_seed(SEED)
        projection = nn.Linear(student_dim, int(teacher_dim))
        projection_state = {k: v.detach().cpu().clone()
                            for k, v in projection.state_dict().items()}
        del projection
    del model
    return state, student_dim, projection_state


def prepare_student_unit(dataset, subject, student, cfg):
    dcfg = config.load_dataset_config(dataset)
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset, subject, val_split=float(cfg['val_split']), seed=SEED,
        return_uid=True)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    uid_te = np.asarray(uid_te, dtype=np.int64)
    if uid_tr.ndim != 2 or uid_tr.shape[1] != 2:
        raise ValueError('train UID must have shape (N,2)')
    adapter = get_adapter(student, device='cpu',
                          **student_config(dataset, student, X_tr))
    scfg = dict(adapter.cfg)
    Xp_tr = adapter.preprocess(X_tr)
    Xp_te = adapter.preprocess(X_te)
    student_state, student_dim, _ = capture_student_state(
        adapter, Xp_tr, int(dcfg['num_classes']), scfg)
    schedule = make_schedule(len(y_tr), int(scfg.get('batch_size', 32)), EPOCHS)
    per_epoch_hash, batch_hash = schedule_hashes(schedule, uid_tr)
    preprocessing_hash = _hash_bytes(
        json.dumps(scfg, sort_keys=True, default=str),
        hash_array(Xp_tr.numpy()), hash_array(Xp_te.numpy()))
    return {
        'dataset': dataset, 'subject': int(subject), 'student': student,
        'dataset_cfg': dcfg, 'student_cfg': scfg, 'adapter': adapter,
        'X_tr': X_tr, 'y_tr': y_tr, 'X_te': X_te, 'y_te': y_te,
        'uid_tr': uid_tr, 'uid_te': uid_te, 'Xp_tr': Xp_tr, 'Xp_te': Xp_te,
        'student_state': student_state, 'student_dim': student_dim,
        'schedule': schedule, 'batch_order_hashes': per_epoch_hash,
        'batch_order_hash': batch_hash,
        'split_uid_hash': hash_uid_split(uid_tr, uid_te),
        'train_uid_hash': hash_array(uid_tr), 'test_uid_hash': hash_array(uid_te),
        'initial_state_hash': hash_state(student_state),
        'preprocessing_hash': preprocessing_hash,
    }


def prepare_teacher_unit(unit, teacher, cfg):
    teacher_roots = cfg.get('teacher_artifact_roots', {})
    root = teacher_roots.get(teacher, cfg['artifact_root'])
    path = artifact_path(unit['dataset'], teacher, unit['subject'], root)
    teacher_data = align_teacher_artifact(path, unit['y_tr'], unit['uid_tr'])
    if teacher_data['logits'].shape[1] != int(unit['dataset_cfg']['num_classes']):
        raise ValueError(f'{path}: teacher class dimension mismatch')
    teacher_dim = int(teacher_data['feats'].shape[1])
    expected_dim = cfg.get('cbramod_teacher_feature_dim') if teacher == 'cbramod' else None
    if expected_dim is not None and teacher_dim != int(expected_dim):
        raise ValueError(
            f'{path}: corrected {teacher} feature tap requires {int(expected_dim)} '
            f'dimensions, got {teacher_dim}; refusing to align the legacy flattened artifact')
    # A separate projection is initialized once per teacher/student/unit and
    # then shared by the two conditions in this pair.
    _, _, projection_state = capture_student_state(
        unit['adapter'], unit['Xp_tr'], int(unit['dataset_cfg']['num_classes']),
        unit['student_cfg'], teacher_dim=teacher_dim)
    teacher_data['teacher'] = teacher
    teacher_data['teacher_dim'] = teacher_dim
    teacher_data['feature_head'] = (
        str(cfg.get('cbramod_teacher_feature_head', 'artifact_native'))
        if teacher == 'cbramod' else 'mirepnet_pooled_256')
    teacher_data['projection_state'] = projection_state
    teacher_data['projection_initial_state_hash'] = hash_state(projection_state)
    return teacher_data


def feature_alignment_loss(student_feat, teacher_feat, projection):
    projected = projection(student_feat)
    projected = F.normalize(projected, dim=1, eps=FEATURE_EPS)
    target = F.normalize(teacher_feat.detach(), dim=1, eps=FEATURE_EPS)
    return 1.0 - F.cosine_similarity(projected, target, dim=1,
                                     eps=FEATURE_EPS).mean()


def vanilla_kd_loss(student_logits, teacher_logits):
    teacher_prob = F.softmax(teacher_logits.detach() / TEMPERATURE_KD, dim=1)
    return F.kl_div(F.log_softmax(student_logits / TEMPERATURE_KD, dim=1),
                    teacher_prob, reduction='batchmean')


def optimizer_for(model, projection, cfg):
    params = list(model.parameters())
    if projection is not None:
        params += list(projection.parameters())
    name = str(cfg.get('optimizer', 'adamw')).lower()
    lr = float(cfg.get('lr', 1e-3))
    wd = float(cfg.get('weight_decay', 1e-4))
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=float(cfg.get('momentum', 0.9)))
    raise ValueError(f'unsupported Student optimizer {name!r}')


@torch.no_grad()
def inference(adapter, model, Xp, batch_size, device):
    model.eval()
    feats, logits = [], []
    for start in range(0, len(Xp), int(batch_size)):
        feat, lg = adapter.forward(model, Xp[start:start + int(batch_size)].to(device))
        feats.append(feat.detach().cpu())
        logits.append(lg.detach().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


@torch.no_grad()
def transition_metrics(unit, teacher_data, model, projection, device):
    feat, logits = inference(unit['adapter_device'], model, unit['Xp_tr'],
                             unit['student_cfg'].get('batch_size', 32), device)
    out = {'train_accuracy': float((logits.argmax(1) == unit['y_tr']).mean() * 100.0)}
    if projection is None or teacher_data is None:
        out.update({'feature_cosine': 'NA', 'logit_kl': 'NA'})
        return out
    # Keep the trainable projection on its existing device.  Moving the
    # module to CPU in-place here would make the subsequent GPU phase fail
    # after transition evaluation.  Move only the evaluation tensors.
    projection_device = next(projection.parameters()).device
    p = projection
    s = torch.as_tensor(feat, dtype=torch.float32, device=projection_device)
    t = torch.as_tensor(teacher_data['feats'], dtype=torch.float32,
                        device=projection_device)
    out['feature_cosine'] = float(F.cosine_similarity(
        F.normalize(p(s), dim=1, eps=FEATURE_EPS),
        F.normalize(t, dim=1, eps=FEATURE_EPS), dim=1,
        eps=FEATURE_EPS).mean().item())
    out['logit_kl'] = float(vanilla_kd_loss(
        torch.as_tensor(logits, dtype=torch.float32),
        torch.as_tensor(teacher_data['logits'], dtype=torch.float32)).item())
    return out


def run_condition(unit, teacher_data, condition, device):
    started = time.time()
    set_seed(SEED)
    adapter = get_adapter(unit['student'], device=str(device), **unit['student_cfg'])
    model = adapter.build(int(unit['dataset_cfg']['num_classes']))
    model.load_state_dict(unit['student_state'], strict=True)
    projection = None
    if teacher_data is not None:
        projection = nn.Linear(unit['student_dim'], teacher_data['teacher_dim']).to(device)
        projection.load_state_dict(teacher_data['projection_state'], strict=True)
    # Reset random streams after construction so all three conditions have the
    # same dropout/RNG starting point; batch schedule is precomputed explicitly.
    set_seed(SEED)
    optimizer = optimizer_for(model, projection, unit['student_cfg'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    y_all = torch.as_tensor(unit['y_tr'], dtype=torch.long)
    teacher_feat = (torch.as_tensor(teacher_data['feats'], dtype=torch.float32)
                    if teacher_data is not None else None)
    teacher_logits = (torch.as_tensor(teacher_data['logits'], dtype=torch.float32)
                      if teacher_data is not None else None)
    history = []
    transitions = []
    transitions.append({'epoch': 0, **transition_metrics(
        {**unit, 'adapter_device': adapter}, teacher_data, model, projection, device)})
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if projection is not None:
            projection.train()
        sums = {'ce': 0.0, 'feature': 0.0, 'kd': 0.0, 'total': 0.0}
        seen = 0
        batches = 0
        correct = 0
        stage2 = epoch > WARMUP_EPOCHS
        for batch in unit['schedule'][epoch - 1]:
            idx = np.asarray(batch, dtype=np.int64)
            xb = unit['Xp_tr'][idx].to(device)
            yb = y_all[idx].to(device)
            student_feat, student_logits = adapter.forward(model, xb)
            ce = F.cross_entropy(student_logits, yb)
            feature_loss = student_logits.sum() * 0.0
            kd_loss = student_logits.sum() * 0.0
            if stage2 and teacher_data is not None:
                tf = teacher_feat[idx].to(device)
                tl = teacher_logits[idx].to(device)
                if condition == "TASK_FEAT_LOGIT_KD":
                    feature_loss = feature_alignment_loss(student_feat, tf, projection)
                kd_loss = vanilla_kd_loss(student_logits, tl)
            total = (ce + LAM_KD * TEMPERATURE_KD ** 2 * kd_loss
                     + LAM_FEATURE * feature_loss)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            count = len(idx)
            seen += count
            batches += 1
            correct += int((student_logits.detach().argmax(1) == yb).sum().item())
            sums['ce'] += float(ce.detach()) * count
            sums['feature'] += float(feature_loss.detach()) * count
            sums['kd'] += float(kd_loss.detach()) * count
            sums['total'] += float(total.detach())
        if seen != len(unit['y_tr']):
            raise RuntimeError(f'{unit["dataset"]} S{unit["subject"] + 1}: incomplete batch schedule')
        scheduler.step()
        row = {
            'epoch': epoch,
            'phase': 'ce_warmup' if not stage2 else 'task_feature_logit_kd',
            'ce_loss': sums['ce'] / seen,
            'feature_loss': (sums['feature'] / seen) if stage2 and condition == 'TASK_FEAT_LOGIT_KD' else 'NA',
            'kd_loss': (sums['kd'] / seen) if stage2 and teacher_data is not None else 'NA',
            'scaled_feature_contribution': (LAM_FEATURE * sums['feature'] / seen)
            if stage2 and condition == 'TASK_FEAT_LOGIT_KD' else 'NA',
            'scaled_kd_contribution': (LAM_KD * TEMPERATURE_KD ** 2 * sums['kd'] / seen)
            if stage2 and teacher_data is not None else 'NA',
            'total_loss': sums['total'] / batches,
            'train_accuracy': correct / seen * 100.0,
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'n_samples': seen,
        }
        history.append(row)
        if epoch in (WARMUP_EPOCHS, EPOCHS):
            transitions.append({'epoch': epoch, **transition_metrics(
                {**unit, 'adapter_device': adapter}, teacher_data, model,
                projection, device)})
    train_feat, train_logits = inference(adapter, model, unit['Xp_tr'],
                                         unit['student_cfg'].get('batch_size', 32), device)
    test_feat, test_logits = inference(adapter, model, unit['Xp_te'],
                                       unit['student_cfg'].get('batch_size', 32), device)
    train_pred = train_logits.argmax(1)
    test_pred = test_logits.argmax(1)
    return {
        'model': model, 'projection': projection, 'history': history,
        'transitions': transitions, 'train_feat': train_feat,
        'train_logits': train_logits, 'test_feat': test_feat,
        'test_logits': test_logits, 'train_pred': train_pred,
        'test_pred': test_pred, 'runtime_seconds': time.time() - started,
    }


def metric_row(unit, teacher_data, condition, result, paths):
    ytr, yte = unit['y_tr'], unit['y_te']
    pred = result['test_pred']
    train_pred = result['train_pred']
    nc = int(unit['dataset_cfg']['num_classes'])
    accuracy = float((pred == yte).mean() * 100.0)
    balanced = float(balanced_accuracy_score(yte, pred) * 100.0)
    kappa = float(cohen_kappa_score(yte, pred))
    t0, t10, t100 = result['transitions']
    if teacher_data is None:
        teacher_feature_layer = 'NA'
    elif teacher_data['teacher'] == 'cbramod' and int(teacher_data['teacher_dim']) == 200:
        if teacher_data.get('feature_head') == 'original_mlp_200':
            teacher_feature_layer = (
                'CBraMod original MLP penultimate 200D feature immediately '
                'before final Linear(200,num_classes)')
        else:
            teacher_feature_layer = (
                'CBraMod mean_pool_200 feature immediately before final '
                'Linear(200,num_classes)')
    elif teacher_data['teacher'] == 'mirepnet' and int(teacher_data['teacher_dim']) == 256:
        teacher_feature_layer = 'MIRepNetAdapter pooled feature (256D)'
    else:
        teacher_feature_layer = 'train artifact feats (native teacher tap)'
    return {
        'run_key': run_key(unit, teacher_data, condition),
        'dataset': unit['dataset'], 'subject': unit['subject'] + 1,
        'subject_index': unit['subject'], 'session': session_for(unit['dataset']),
        'protocol': 'fewshot', 'seed': SEED,
        'teacher': teacher_data['teacher'] if teacher_data is not None else 'shared',
        'student': unit['student'], 'condition': condition,
        'result_source': 'new', 'train_count': len(ytr), 'test_count': len(yte),
        'epochs': EPOCHS, 'warmup_epochs': WARMUP_EPOCHS,
        'distillation_start_epoch': WARMUP_EPOCHS + 1,
        'optimizer': str(unit['student_cfg'].get('optimizer', 'adamw')).lower(),
        'lr': float(unit['student_cfg'].get('lr', 1e-3)),
        'weight_decay': float(unit['student_cfg'].get('weight_decay', 1e-4)),
        'batch_size': int(unit['student_cfg'].get('batch_size', 32)),
        'scheduler': 'CosineAnnealingLR', 'temperature_kd': TEMPERATURE_KD,
        'lam_kd': LAM_KD, 'lam_feature': LAM_FEATURE,
        'student_feature_dim': unit['student_dim'],
        'teacher_feature_dim': teacher_data['teacher_dim'] if teacher_data is not None else 'NA',
        'student_feature_layer': 'adapter.forward final classifier-before-feature',
        'teacher_feature_layer': teacher_feature_layer,
        'accuracy': accuracy, 'balanced_accuracy': balanced, 'kappa': kappa,
        'final_train_accuracy': float((train_pred == ytr).mean() * 100.0),
        'predicted_class_counts': json.dumps(np.bincount(pred, minlength=nc).tolist()),
        'collapse_flag': bool(len(np.unique(pred)) < 2),
        'transition_epoch0_feature_cosine': t0['feature_cosine'],
        'transition_epoch10_feature_cosine': t10['feature_cosine'],
        'transition_epoch100_feature_cosine': t100['feature_cosine'],
        'transition_epoch0_logit_kl': t0['logit_kl'],
        'transition_epoch10_logit_kl': t10['logit_kl'],
        'transition_epoch100_logit_kl': t100['logit_kl'],
        'split_uid_hash': unit['split_uid_hash'], 'train_uid_hash': unit['train_uid_hash'],
        'test_uid_hash': unit['test_uid_hash'], 'initial_state_hash': unit['initial_state_hash'],
        'projection_initial_state_hash': teacher_data['projection_initial_state_hash'] if teacher_data is not None else 'NA',
        'batch_order_hash': unit['batch_order_hash'],
        'batch_order_hashes': json.dumps(unit['batch_order_hashes']),
        'preprocessing_hash': unit['preprocessing_hash'],
        'teacher_artifact_path': teacher_data['path'] if teacher_data is not None else '',
        'teacher_artifact_sha256': teacher_data['artifact_sha256'] if teacher_data is not None else '',
        'teacher_feature_hash': teacher_data['feature_hash'] if teacher_data is not None else '',
        'teacher_logits_hash': teacher_data['logits_hash'] if teacher_data is not None else '',
        'teacher_uid_alignment_hash': teacher_data['alignment_hash'] if teacher_data is not None else '',
        'runtime_seconds': result['runtime_seconds'], 'epochs_completed': EPOCHS,
        'status': 'complete', 'failure_reason': '',
        'checkpoint_path': str(paths['checkpoint'].resolve()),
        'history_path': str(paths['history'].resolve()),
        'prediction_path': str(paths['prediction'].resolve()),
    }


def run_key(unit, teacher_data, condition):
    teacher = teacher_data['teacher'] if teacher_data is not None else 'shared'
    return (f"{unit['dataset']}__S{unit['subject'] + 1}__{unit['student']}__"
            f"{teacher}__seed{SEED}__{condition}")


def save_run(output_root, unit, teacher_data, condition, result):
    output_root = require_external_output(output_root)
    key = run_key(unit, teacher_data, condition)
    paths = {
        'history': output_root / 'training_history' / f'{key}.json',
        'checkpoint': output_root / 'checkpoints' / f'{key}.pt',
        'prediction': output_root / 'predictions' / f'{key}.npz',
    }
    atomic_json(paths['history'], {
        'run_key': key, 'dataset': unit['dataset'], 'subject': unit['subject'],
        'student': unit['student'], 'teacher': teacher_data['teacher'] if teacher_data else 'shared',
        'condition': condition, 'epochs': result['history'],
        'feature_transition': result['transitions'],
    })
    checkpoint = {
        'complete': True, 'run_key': key, 'epochs_completed': EPOCHS,
        'condition': condition, 'state_dict': {k: v.detach().cpu() for k, v in result['model'].state_dict().items()},
        'projection_state_dict': ({k: v.detach().cpu() for k, v in result['projection'].state_dict().items()}
                                  if result['projection'] is not None else None),
        'split_uid_hash': unit['split_uid_hash'], 'train_uid_hash': unit['train_uid_hash'],
        'test_uid_hash': unit['test_uid_hash'], 'initial_state_hash': unit['initial_state_hash'],
        'projection_initial_state_hash': teacher_data['projection_initial_state_hash'] if teacher_data else 'NA',
        'batch_order_hash': unit['batch_order_hash'],
        'batch_order_hashes': unit['batch_order_hashes'],
        'preprocessing_hash': unit['preprocessing_hash'],
        'teacher_artifact_sha256': teacher_data['artifact_sha256'] if teacher_data else '',
    }
    paths['checkpoint'].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths['checkpoint'].with_name(f'.{paths["checkpoint"].name}.tmp-{os.getpid()}')
    torch.save(checkpoint, require_external_output(tmp))
    os.replace(tmp, paths['checkpoint'])
    paths['prediction'].parent.mkdir(parents=True, exist_ok=True)
    tmp_pred = paths['prediction'].with_name(f'.{paths["prediction"].name}.tmp-{os.getpid()}')
    np.savez(require_external_output(tmp_pred), train_uid=unit['uid_tr'], test_uid=unit['uid_te'],
             train_y=unit['y_tr'], test_y=unit['y_te'],
             train_logits=result['train_logits'].astype(np.float32),
             test_logits=result['test_logits'].astype(np.float32),
             train_pred=result['train_pred'], test_pred=result['test_pred'])
    generated = Path(str(tmp_pred) + '.npz') if not tmp_pred.name.endswith('.npz') else tmp_pred
    os.replace(generated, paths['prediction'])
    return metric_row(unit, teacher_data, condition, result, paths)


def read_rows(path):
    path = external_path(path)
    if not Path(path).exists():
        return []
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def row_complete(row):
    if row.get('status') != 'complete' or int(row.get('epochs_completed', 0)) != EPOCHS:
        return False
    for field in ('accuracy', 'balanced_accuracy', 'kappa'):
        try:
            if not np.isfinite(float(row[field])):
                return False
        except (TypeError, ValueError, KeyError):
            return False
    for field in ('checkpoint_path', 'history_path', 'prediction_path'):
        if not Path(row.get(field, '')).is_file():
            return False
    try:
        payload = json.loads(resolve_local_file(row['history_path']).read_text())
        if len(payload.get('epochs', [])) != EPOCHS:
            return False
        checkpoint = torch.load(resolve_local_file(row['checkpoint_path']), map_location='cpu')
        if not checkpoint.get('complete') or int(checkpoint.get('epochs_completed', 0)) != EPOCHS:
            return False
        with np.load(resolve_local_file(row['prediction_path']), allow_pickle=False) as pred:
            if len(pred['test_uid']) != int(row['test_count']):
                return False
    except Exception:
        return False
    return True


def bootstrap(values, seed=SEED, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        samples[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def comparison_rows(rows):
    pairs = (
        ('DELAYED_LOGIT_KD', 'BASE_CE', 'delayed_logit_kd_vs_base'),
        ('TASK_FEAT_LOGIT_KD', 'DELAYED_LOGIT_KD', 'task_feature_vs_delayed_logit_kd'),
        ('TASK_FEAT_LOGIT_KD', 'BASE_CE', 'task_feature_vs_base'),
    )
    out = []
    # Base rows are keyed by student, while teacher rows are keyed by pair.
    for teacher in TEACHERS:
        for student in STUDENTS:
            for dataset in (*DATASETS, 'ALL_DATASETS'):
                scope = [r for r in rows if r['student'] == student and
                         (dataset == 'ALL_DATASETS' or r['dataset'] == dataset) and
                         (r['teacher'] == teacher or r['teacher'] == 'shared')]
                grouped = {}
                for row in scope:
                    # Historical reused rows are read from CSV as strings,
                    # while newly trained rows carry integer subject_index.
                    # Normalize the paired key so comparisons against the
                    # shared Base row cannot silently become empty.
                    grouped.setdefault((row['dataset'], int(row['subject_index'])), {})[
                        (row['teacher'], row['condition'])] = row
                for a, b, label in pairs:
                    values = []
                    for methods in grouped.values():
                        ar = methods.get((teacher, a))
                        # The delayed-logit comparator is the same Teacher pair;
                        # only the Base comparator uses the shared Base row.
                        br_teacher = teacher if b != 'BASE_CE' else 'shared'
                        br = methods.get((br_teacher, b))
                        if ar is not None and br is not None:
                            values.append(float(ar['balanced_accuracy']) - float(br['balanced_accuracy']))
                    lo, hi = bootstrap(values)
                    out.append({
                        'teacher': teacher, 'student': student, 'dataset': dataset,
                        'comparison': label, 'metric': 'balanced_accuracy',
                        'n_subjects': len(values),
                        'mean_delta': float(np.mean(values)) if values else np.nan,
                        'bootstrap_seed': SEED, 'bootstrap_draws': 10000,
                        'bootstrap_ci_low': lo, 'bootstrap_ci_high': hi,
                        'win': sum(x > 1e-12 for x in values),
                        'tie': sum(abs(x) <= 1e-12 for x in values),
                        'loss': sum(x < -1e-12 for x in values),
                    })
    return out


def write_outputs(output_root, cfg, provenance, rows, epoch_rows,
                  transition_rows, teacher_manifest, controls):
    rows = sorted(rows, key=lambda r: r['run_key'])
    reused_rows = [r for r in rows if str(r.get('result_source', 'new')) == 'reused']
    # Reused controls are carried into a corrected pilot so that the
    # provenance distinguishes historical rows from newly trained rows.
    # They are never retrained or altered by this runner.
    control_validation = []
    for row in reused_rows:
        valid = row_complete(row)
        control_validation.append({
            'run_key': row.get('run_key', ''),
            'dataset': row.get('dataset', ''),
            'subject': row.get('subject', ''),
            'subject_index': row.get('subject_index', ''),
            'student': row.get('student', ''),
            'teacher': row.get('teacher', ''),
            'condition': row.get('condition', ''),
            'status': 'valid_reused' if valid else 'invalid_reused',
            'reason': ('prior complete result with finite metrics and valid '
                       'checkpoint/history/prediction' if valid else
                       'prior row failed completeness validation'),
        })
    atomic_csv(output_root / 'results_per_run.csv', rows)
    atomic_csv(MAIN_CSV, rows)
    atomic_csv(output_root / 'run_manifest.csv', [
        {'run_key': r['run_key'], 'dataset': r['dataset'], 'subject': r['subject'],
         'subject_index': r['subject_index'], 'student': r['student'],
         'teacher': r['teacher'], 'condition': r['condition'],
         'status': r['status'], 'epochs_completed': r['epochs_completed']}
        for r in rows])
    atomic_csv(output_root / 'teacher_artifact_manifest.csv', teacher_manifest)
    atomic_csv(output_root / 'control_validation.csv', control_validation)
    atomic_csv(output_root / 'reused_controls.csv', reused_rows)
    atomic_csv(output_root / 'epoch_metrics.csv', epoch_rows)
    atomic_csv(output_root / 'feature_transition_metrics.csv', transition_rows)
    atomic_csv(output_root / 'combined_results_per_run.csv', rows)
    atomic_csv(output_root / 'results_per_subject.csv', rows)
    dataset_rows = []
    for teacher in ('shared', *TEACHERS):
        for student in STUDENTS:
            for condition in CONDITIONS:
                part = [r for r in rows if r['teacher'] == teacher and
                        r['student'] == student and r['condition'] == condition]
                for dataset in DATASETS:
                    vals = [r for r in part if r['dataset'] == dataset]
                    dataset_rows.append({
                        'teacher': teacher, 'student': student, 'condition': condition,
                        'dataset': dataset, 'n_subjects': len(vals),
                        'accuracy_mean': float(np.mean([float(r['accuracy']) for r in vals])) if vals else np.nan,
                        'accuracy_sd': float(np.std([float(r['accuracy']) for r in vals], ddof=1)) if len(vals) > 1 else np.nan,
                        'balanced_accuracy_mean': float(np.mean([float(r['balanced_accuracy']) for r in vals])) if vals else np.nan,
                        'balanced_accuracy_sd': float(np.std([float(r['balanced_accuracy']) for r in vals], ddof=1)) if len(vals) > 1 else np.nan,
                        'kappa_mean': float(np.mean([float(r['kappa']) for r in vals])) if vals else np.nan,
                        'kappa_sd': float(np.std([float(r['kappa']) for r in vals], ddof=1)) if len(vals) > 1 else np.nan,
                        'collapse_count': sum(str(r.get('collapse_flag')).lower() == 'true' for r in vals),
                    })
    atomic_csv(output_root / 'results_per_dataset.csv', dataset_rows)
    comparisons = comparison_rows(rows)
    atomic_csv(output_root / 'paired_comparisons.csv', comparisons)
    atomic_text(output_root / 'config_resolved.yaml', yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    atomic_json(output_root / 'execution_provenance.json', provenance)
    atomic_text(output_root / 'report.md', render_report(rows, dataset_rows, comparisons, provenance))


def render_report(rows, dataset_rows, comparisons, provenance):
    lines = [
        '# Task Feature + Logit KD Six-Pair Pilot', '',
        'Seed 666; subject-wise few-shot; 30% train / 70% test; 100 epochs.', '',
        'Epochs 1–10 are pure Student CE warm-up. Epochs 11–100 use the same fine-tuned Teacher logits KD and feature alignment.', '',
        'No pretrained Teacher alignment, MI, prototype, mask, test Teacher output, LOSO, or extra seed was used.', '',
        '## Overall test metrics (Accuracy / Balanced Accuracy / Kappa)', '',
        '| teacher | student | condition | Accuracy | BA | Kappa |', '|---|---|---|---:|---:|---:|',
    ]
    for teacher in ('shared', *TEACHERS):
        for student in STUDENTS:
            for condition in CONDITIONS:
                part = [r for r in rows if r['teacher'] == teacher and r['student'] == student and r['condition'] == condition]
                if not part:
                    continue
                lines.append(f"| {teacher} | {student} | {condition} | {np.mean([float(r['accuracy']) for r in part]):.2f} | {np.mean([float(r['balanced_accuracy']) for r in part]):.2f} | {np.mean([float(r['kappa']) for r in part]):.3f} |")
    lines += ['', '## Core paired BA comparisons', '',
              '| teacher | student | dataset | comparison | mean Δ (pp) | 95% CI | W/T/L |', '|---|---|---|---|---:|---:|---:|']
    for r in comparisons:
        if r['dataset'] == 'ALL_DATASETS':
            lines.append(f"| {r['teacher']} | {r['student']} | ALL | {r['comparison']} | {float(r['mean_delta']):.3f} | [{float(r['bootstrap_ci_low']):.3f}, {float(r['bootstrap_ci_high']):.3f}] | {r['win']}/{r['tie']}/{r['loss']} |")
    lines += ["", "## Per-dataset test metrics", "",
              "| teacher | student | condition | dataset | n | Accuracy | BA | Kappa | collapse |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in dataset_rows:
        lines.append(f"| {r['teacher']} | {r['student']} | {r['condition']} | {r['dataset']} | {r['n_subjects']} | {float(r['accuracy_mean']):.2f} | {float(r['balanced_accuracy_mean']):.2f} | {float(r['kappa_mean']):.3f} | {r['collapse_count']} |")
    lines += ['', '## Transition diagnostics', '',
              'Values are full-train eval-mode diagnostics, not test metrics.', '',
              '| teacher | student | condition | feature cosine e0/e10/e100 | logit KL e0/e10/e100 |', '|---|---|---|---|---|']
    for teacher in TEACHERS:
        for student in STUDENTS:
            for condition in TEACHER_CONDITIONS:
                part = [r for r in rows if r['teacher'] == teacher and r['student'] == student and r['condition'] == condition]
                if not part:
                    continue
                def mean_field(field):
                    vals = [float(r[field]) for r in part if r[field] not in ('NA', '')]
                    return f'{np.mean(vals):.3f}' if vals else 'NA'
                lines.append(f"| {teacher} | {student} | {condition} | {mean_field('transition_epoch0_feature_cosine')}/{mean_field('transition_epoch10_feature_cosine')}/{mean_field('transition_epoch100_feature_cosine')} | {mean_field('transition_epoch0_logit_kl')}/{mean_field('transition_epoch10_logit_kl')}/{mean_field('transition_epoch100_logit_kl')} |")
    lines += ['', '## Provenance and limitations', '',
              f"- Git commit: `{provenance.get('git_commit')}`.",
              f"- Physical GPU: `{provenance.get('physical_gpu')}`; logical device: `{provenance.get('logical_device')}`.",
              '- Teacher features/logits came only from fine-tuned train artifacts and were explicitly UID-aligned.',
              '- Student Teacher-related conditions have different parameter trajectories after the shared initialization because feature loss is the only intended objective difference.',
              '- Feature alignment is a train-set transfer diagnostic; it is not evidence of test-time feature matching.',
              f"- The combined table contains {provenance.get('completed_runs', len(rows))} total rows: "
              f"{provenance.get('reused_runs', 0)} reused controls and "
              f"{provenance.get('completed_new_runs', 0)} newly trained rows. Base is shared once per Student/unit; "
              'each Teacher contributes delayed-logit and task-feature-logit rows.',
    ]
    return '\n'.join(lines) + '\n'


def git_value(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f'error: {exc}'


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = validate_config(cfg_path)
    cfg = dict(cfg)
    cfg['artifact_root'] = str(external_path(cfg['artifact_root']))
    cfg['output_dir'] = str(require_external_output(cfg['output_dir']))
    output_root = require_external_output(cfg['output_dir'])
    if output_root.exists() and any(output_root.iterdir()) and not (args.resume or args.force):
        raise RuntimeError(f'output directory is non-empty; use --resume or --force: {output_root}')
    if args.force:
        resolved = output_root.resolve()
        allowed = (Path('/data1/llx/BigSmallcollab/results') / 'distill' / 'task_feature_logit_kd_six_pairs_seed666').resolve()
        if resolved != allowed:
            raise ValueError('--force is restricted to the dedicated six-pair output directory')
    device = device_for(args.gpu)
    prior_provenance = {}
    prior_path = output_root / "execution_provenance.json"
    if args.resume and prior_path.is_file():
        try:
            prior_provenance = json.loads(resolve_local_file(prior_path).read_text())
        except Exception:
            prior_provenance = {}
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not_set")
    if physical_gpu in ("", "not_set"):
        physical_gpu = prior_provenance.get("physical_gpu", physical_gpu)
    launch_command = " ".join(sys.argv)
    if args.resume and prior_provenance.get("launch_command") and device.type == "cpu":
        launch_command = prior_provenance["launch_command"]
    logical_device = str(device)
    if device.type == "cpu" and str(prior_provenance.get("logical_device", "")).startswith("cuda"):
        logical_device = prior_provenance["logical_device"]
    expected_total_runs = sum(SUBJECT_COUNTS.values()) * (len(STUDENTS) +
                           len(TEACHERS) * len(STUDENTS) * len(TEACHER_CONDITIONS))
    provenance = {
        'git_commit': git_value('rev-parse', 'HEAD'),
        'git_status_before': git_value('status', '--short'),
        'config_path': str(cfg_path), 'config_sha256': hash_file(cfg_path),
        'artifact_root': cfg['artifact_root'], 'output_dir': str(output_root),
        'datasets': list(DATASETS), 'teachers': list(TEACHERS), 'students': list(STUDENTS),
        'conditions': list(CONDITIONS), 'seed': SEED, 'subject_units': sum(SUBJECT_COUNTS.values()),
        'expected_new_runs': expected_total_runs, 'teacher_test_artifacts_read': False,
        'gpu_argument': args.gpu, 'physical_gpu': physical_gpu,
        'logical_device': logical_device, 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'launch_command': launch_command,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    # Keep each configured pilot isolated: the top-level CSV follows the
    # configured output directory instead of a hard-coded prior pilot path.
    global MAIN_CSV
    MAIN_CSV = output_root.parent / f'{output_root.name}.csv'
    atomic_text(output_root / 'git_status_before.txt', provenance['git_status_before'] + '\n')
    atomic_json(output_root / 'execution_provenance.json', provenance)
    atomic_text(output_root / 'config_resolved.yaml', yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    existing = {}
    if args.resume:
        existing = {r.get('run_key'): r for r in read_rows(output_root / 'results_per_run.csv')}
        # Normalize provenance text for rows carried from an earlier pass.
        # This changes no metrics, checkpoints, predictions, or hashes.
        cbramod_head = str(cfg.get('cbramod_teacher_feature_head', 'artifact_native'))
        for row in existing.values():
            if row.get('teacher') == 'cbramod' and str(row.get('teacher_feature_dim')) in {'200', '200.0'}:
                if cbramod_head == 'original_mlp_200':
                    row['teacher_feature_layer'] = (
                        'CBraMod original MLP penultimate 200D feature immediately '
                        'before final Linear(200,num_classes)')
                else:
                    row['teacher_feature_layer'] = (
                        'CBraMod mean_pool_200 feature immediately before final '
                        'Linear(200,num_classes)')
            elif row.get('teacher') == 'mirepnet' and str(row.get('teacher_feature_dim')) in {'256', '256.0'}:
                row['teacher_feature_layer'] = 'MIRepNetAdapter pooled feature (256D)'
    all_rows, epoch_rows, transition_rows, teacher_manifest, controls = [], [], [], [], []
    valid_existing_reused = sum(
        1 for row in existing.values()
        if str(row.get('result_source', 'new')) == 'reused' and row_complete(row)
    )
    provenance['expected_total_runs'] = expected_total_runs
    provenance['expected_reused_runs'] = valid_existing_reused
    provenance['expected_new_runs'] = expected_total_runs - valid_existing_reused
    # Shared Base is trained once per Student/unit.  Teacher-related conditions
    # are then run independently for each Teacher.
    total_units = sum(SUBJECT_COUNTS.values())
    progress = 0
    for dataset in DATASETS:
        for subject in range(SUBJECT_COUNTS[dataset]):
            for student in STUDENTS:
                unit = prepare_student_unit(dataset, subject, student, cfg)
                teacher_units = {}
                for teacher in TEACHERS:
                    teacher_data = prepare_teacher_unit(unit, teacher, cfg)
                    teacher_units[teacher] = teacher_data
                    teacher_manifest.append({
                        'dataset': dataset, 'subject': subject + 1,
                        'subject_index': subject, 'student': student,
                        'teacher': teacher, 'seed': SEED,
                        'train_count': len(unit['y_tr']),
                        'teacher_artifact_path': teacher_data['path'],
                        'teacher_artifact_sha256': teacher_data['artifact_sha256'],
                        'teacher_feature_dim': teacher_data['teacher_dim'],
                        'teacher_feature_layer': (
                            'CBraMod original MLP penultimate 200D feature immediately '
                            'before final Linear(200,num_classes)' if teacher == 'cbramod' else
                            'MIRepNetAdapter pooled feature (256D)'),
                        'teacher_feature_hash': teacher_data['feature_hash'],
                        'teacher_logits_hash': teacher_data['logits_hash'],
                        'teacher_uid_alignment_hash': teacher_data['alignment_hash'],
                        'alignment_status': 'pass', 'test_artifact_read': False,
                    })
                specs = [('shared', None, 'BASE_CE')]
                specs += [(teacher, teacher_units[teacher], condition)
                          for teacher in TEACHERS for condition in TEACHER_CONDITIONS]
                for teacher_name, teacher_data, condition in specs:
                    key = run_key(unit, teacher_data, condition)
                    prior = existing.get(key)
                    if args.resume and prior is not None and row_complete(prior):
                        all_rows.append(prior)
                        progress += 1
                        print(f'[skip {progress}/{expected_total_runs}] {key}', flush=True)
                        # Histories are recovered below for deterministic summaries.
                        continue
                    result = run_condition(unit, teacher_data, condition, device)
                    paths = {
                        'checkpoint': output_root / 'checkpoints' / f'{key}.pt',
                        'history': output_root / 'training_history' / f'{key}.json',
                        'prediction': output_root / 'predictions' / f'{key}.npz',
                    }
                    row = save_run(output_root, unit, teacher_data, condition, result)
                    all_rows.append(row)
                    for epoch_row in result['history']:
                        epoch_rows.append({
                            'dataset': dataset, 'subject': subject + 1,
                            'subject_index': subject, 'student': student,
                            'teacher': teacher_data['teacher'] if teacher_data else 'shared',
                            'condition': condition, **epoch_row,
                        })
                    for transition in result['transitions']:
                        transition_rows.append({
                            'dataset': dataset, 'subject': subject + 1,
                            'subject_index': subject, 'student': student,
                            'teacher': teacher_data['teacher'] if teacher_data else 'shared',
                            'condition': condition, **transition,
                        })
                    progress += 1
                    print(f'[progress {progress}/{expected_total_runs}] {key} acc={row["accuracy"]:.2f}', flush=True)
                    atomic_csv(output_root / 'results_per_run.csv', sorted(all_rows, key=lambda r: r['run_key']))
                del teacher_units, unit
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    if args.resume and len(epoch_rows) != len(all_rows) * EPOCHS:
        epoch_rows = []; transition_rows = []
        for row in all_rows:
            payload = json.loads(resolve_local_file(row['history_path']).read_text())
            for epoch_row in payload.get('epochs', []):
                epoch_rows.append({
                    'dataset': row['dataset'], 'subject': row['subject'],
                    'subject_index': row['subject_index'], 'student': row['student'],
                    'teacher': row['teacher'], 'condition': row['condition'], **epoch_row,
                })
            for transition in payload.get('feature_transition', []):
                transition_rows.append({
                    'dataset': row['dataset'], 'subject': row['subject'],
                    'subject_index': row['subject_index'], 'student': row['student'],
                    'teacher': row['teacher'], 'condition': row['condition'], **transition,
                })
    reused_runs = sum(str(row.get('result_source', 'new')) == 'reused' for row in all_rows)
    provenance['completed_runs'] = len(all_rows)
    provenance['reused_runs'] = reused_runs
    provenance['completed_new_runs'] = len(all_rows) - reused_runs
    provenance['git_status_after'] = git_value('status', '--short')
    atomic_json(output_root / 'execution_provenance.json', provenance)
    write_outputs(output_root, cfg, provenance, all_rows, epoch_rows,
                  transition_rows, teacher_manifest, controls)
    provenance['completed_runs'] = len(all_rows)
    provenance['reused_runs'] = reused_runs
    provenance['completed_new_runs'] = len(all_rows) - reused_runs
    provenance['git_status_after'] = git_value('status', '--short')
    atomic_json(output_root / 'execution_provenance.json', provenance)
    print(f'[complete] {len(all_rows)}/{expected_total_runs} total rows; {len(all_rows) - reused_runs} new, '
          f'{reused_runs} reused; output={output_root}', flush=True)


if __name__ == '__main__':
    main()
