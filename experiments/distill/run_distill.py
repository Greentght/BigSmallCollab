"""Unified big-to-small distillation runner.

Runs CE baseline and selected distillation methods from teacher artifacts. For
LOSO, target caches are strictly validated and reused when valid. Missing,
partial, or mismatched derived caches are repaired under fold/model-input locks;
the teacher checkpoint is reused when its provenance is valid, or prepared from
the matching seed/fold when absent. The main runner still uses the student
environment, usually ``mirepnet`` for the small CNNs.

The direct CLI defaults to ``KD_all``. Supported methods (``--methods all``
selects the full legacy matrix):
  Base      : student CE only
  KD_all    : CE + logits KD on all train/support samples
  KD_masked : CE + logits KD only where teacher is correct on train/support
  MMD       : CE + feature-distribution MMD
  KD_MMD    : CE + logits KD + feature-distribution MMD
  CE_MI     : CE + class-joint probability mutual-information pilot

All teacher artifacts must contain sample_uid and split_policy metadata.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from collab import artifacts
from collab.distill import distill_student, probability_mi_loss
from collab.seed import set_seed as _set_seed
import config
import data
from data import split as split_utils
from data import loso_source
from eval import metrics
from models import BIG_MODELS, SMALL_MODELS, get_adapter
from sklearn.metrics import balanced_accuracy_score
from experiments.storage import (LOSO_CACHE_ROOT, REPORTS_ROOT, external_path,
                                 require_external_output, require_report_output,
                                 resolve_local_file)


DEFAULT_METHODS = ('Base', 'KD_all', 'KD_masked', 'MMD', 'KD_MMD')
METHOD_LABELS = {
    'Base': 'Base',
    'KD_all': 'Vanilla KD',
    'KD_masked': 'KD masked',
    'MMD': 'MMD',
    'KD_MMD': 'KD + MMD',
    'CE_MI': 'CE+MI',
}
METHOD_ALIASES = {
    'all': DEFAULT_METHODS,
    'base': ('Base',),
    'ce': ('Base',),
    'kd': ('KD_all',),
    'kd_all': ('KD_all',),
    'kd-masked': ('KD_masked',),
    'kd_masked': ('KD_masked',),
    'mmd': ('MMD',),
    'kd_mmd': ('KD_MMD',),
    'kd-mmd': ('KD_MMD',),
    # The probability-MI pilot is a fixed three-condition bundle.  Keep the
    # internal method names stable for config-driven dispatch.
    'mi': ('Base', 'KD_all', 'CE_MI'),
    'ce_mi': ('CE_MI',),
    'ce-mi': ('CE_MI',),
}
BASE_FIELDS = [
    'subject', 'seed', 'teacher_seed', 'method', 'acc', 'kappa', 'n_test', 'lam_mi',
    'source_manifest_sha256', 'student_input_manifest_sha256',
    'student_input_policy_sha256',
    'teacher_input_manifest_sha256', 'teacher_target_manifest_sha256',
    'teacher_input_policy_sha256', 'teacher_checkpoint_sha256',
    'teacher_artifact_path', 'teacher_artifact_sha256',
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data_):
        for stream in self.streams:
            stream.write(data_)
        return len(data_)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def normalize_cli_argv(argv=None):
    """Drop standalone whitespace-only shell tokens before argparse parses."""
    tokens = sys.argv[1:] if argv is None else argv
    return [token for token in tokens
            if not (isinstance(token, str) and token.isspace())]


def parse_args(argv=None):
    p = argparse.ArgumentParser(epilog='Legacy direct matrix flags are deprecated; use --config configs/experiments/<name>.yaml')
    p.add_argument('--dataset', default=None,
                   help='single dataset; kept for backward-compatible commands')
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--teacher', default=None,
                   help='single teacher; kept for backward-compatible commands')
    p.add_argument('--teachers', nargs='+', default=None)
    p.add_argument('--student', default=None,
                   help='single student; kept for backward-compatible commands')
    p.add_argument('--students', nargs='+', default=None)
    p.add_argument('--teacher_artifact', default=None,
                   help='override cached teacher artifact dir; only valid with one teacher')
    p.add_argument('--protocol', choices=['fewshot', 'within', 'loso'], default='fewshot')
    p.add_argument('--subjects', '--keys', dest='keys', type=int, nargs='+', default=None,
                   help='zero-based subjects/folds; default all')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--teacher-seed', '--teacher_seed', dest='teacher_seed',
                   type=int, default=None,
                   help='LOSO teacher-target seed; defaults to each student seed')
    p.add_argument('--train_percentage', type=float, default=None,
                   help='fewshot train fraction; overrides dataset val_split')
    p.add_argument('--val_split', type=float, default=None,
                   help='fewshot test fraction; default from dataset config')
    p.add_argument('--methods', nargs='+', default=['kd'],
                   help='all, Base, KD_all, KD_masked, MMD, KD_MMD, mi/CE_MI')
    p.add_argument('--lam_kd', type=float, default=0.5)
    p.add_argument('--lam_mi', type=float, default=0.1,
                   help='pilot probability-MI weight (CE_MI only)')
    p.add_argument('--lam_mmd', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--mmd_sigmas', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument('--no_mmd_normalize', dest='mmd_normalize', action='store_false',
                   default=True)
    p.add_argument('--mmd_class_conditional', action='store_true')
    p.add_argument('--epochs', type=int, default=None,
                   help='override student config epochs')
    p.add_argument('--lr', type=float, default=None,
                   help='override student config lr')
    p.add_argument('--weight_decay', type=float, default=None,
                   help='override student config weight_decay')
    p.add_argument('--batch_size', type=int, default=None,
                   help='override student config batch_size')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--artifact_root', default=None)
    p.add_argument('--out_csv', default=None)
    p.add_argument('--log_file', default=None)
    p.add_argument('--no-auto-teacher-cache', '--no_auto_teacher_cache',
                   dest='no_auto_teacher_cache', action='store_true',
                   help='read LOSO teacher targets only; do not generate missing caches')
    p.add_argument('--fail_fast', action='store_true')
    return p.parse_args(normalize_cli_argv(argv))


def _resolve_list(plural, singular, default):
    if plural is not None:
        return list(plural)
    if singular is not None:
        return [singular]
    return list(default)


def _validate_teacher_seed_protocol(protocol, teacher_seed):
    if teacher_seed is not None and int(teacher_seed) < 0:
        raise ValueError('--teacher-seed must be a non-negative integer')
    if teacher_seed is not None and protocol != 'loso':
        raise ValueError(
            '--teacher-seed is only supported with protocol=loso; '
            'fewshot teacher and student split seeds are coupled')


def _resolve_teacher_seed(student_seed, teacher_seed=None):
    """Resolve the target seed without changing the student's training seed."""
    return int(student_seed if teacher_seed is None else teacher_seed)


def _validate_names(teachers, students):
    bad_teachers = [t for t in teachers if t not in BIG_MODELS]
    bad_students = [s for s in students if s not in SMALL_MODELS]
    if bad_teachers:
        raise ValueError(f'unknown teacher(s) {bad_teachers}; expected {BIG_MODELS}')
    if bad_students:
        raise ValueError(f'unknown student(s) {bad_students}; expected {SMALL_MODELS}')


def _parse_methods(values):
    methods = []
    for raw in values:
        key = raw.strip()
        alias = METHOD_ALIASES.get(key.lower())
        if alias is None:
            valid = sorted(set(DEFAULT_METHODS) | set(METHOD_ALIASES))
            raise ValueError(f'unknown method {raw!r}; expected one of {valid}')
        for method in alias:
            if method not in methods:
                methods.append(method)
    return methods


def _teacher_artifact_name(teacher, protocol, override=None):
    if override is not None:
        return override
    return teacher if protocol == 'fewshot' else f'{teacher}_loso'


def _expected_split_policy(protocol):
    if protocol == 'fewshot':
        return split_utils.FEWSHOT_SPLIT_POLICY
    if protocol == 'loso':
        return split_utils.LOSO_SPLIT_POLICY
    raise ValueError(f'unsupported protocol {protocol!r}')


def _load_or_prepare_loso_dataset(dataset, student, cache_root,
                                  no_auto_teacher_cache=False):
    """Load LOSO student inputs, repairing registered caches unless read-only."""
    if no_auto_teacher_cache:
        return loso_source.load_dataset(dataset, student, cache_root=cache_root)
    from experiments.distill.loso_target_cache import load_or_prepare_loso_dataset

    return load_or_prepare_loso_dataset(
        dataset, student, cache_root=cache_root, auto_prepare=True)


def _load_or_ensure_loso_teacher_targets(
        dataset, teacher, seed, fold, student_data, cache_root, gpu=None,
        no_auto_teacher_cache=False):
    """Strictly load LOSO targets, repairing derived cache failures if enabled."""
    if no_auto_teacher_cache:
        return loso_source.load_teacher_targets(
            dataset, teacher, seed, fold, student_data=student_data,
            cache_root=cache_root)
    from experiments.distill.loso_target_cache import ensure_teacher_targets

    return ensure_teacher_targets(
        dataset, teacher, seed, fold, student_data,
        cache_root=cache_root, gpu=gpu)


def _set_thread_defaults():
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '4')))


def _sha256_file(path, chunk_size=1024 * 1024):
    path = resolve_local_file(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('utf-8'))
    digest.update(str(tuple(array.shape)).encode('utf-8'))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _sha256_uid_split(uid_tr, uid_te):
    digest = hashlib.sha256()
    for name, value in (('train', uid_tr), ('test', uid_te)):
        digest.update(name.encode('utf-8'))
        digest.update(np.asarray(value, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _sha256_state_dict(state):
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


def _combined_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _artifact_info(path):
    stat = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size': int(stat.st_size),
        'mtime': float(stat.st_mtime),
        'sha256': _sha256_file(path),
    }


def _finite_metric(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _final_train_mi_and_agreement(teacher_logits, student_logits, student_preds):
    teacher_logits = np.asarray(teacher_logits)
    student_logits = np.asarray(student_logits)
    teacher_prob = F.softmax(torch.as_tensor(teacher_logits, dtype=torch.float32), dim=1)
    student_prob = F.softmax(torch.as_tensor(student_logits, dtype=torch.float32), dim=1)
    full_mi = float((-probability_mi_loss(teacher_prob, student_prob)).item())
    teacher_pred = teacher_logits.argmax(axis=1)
    student_pred = np.asarray(student_preds)
    return {
        'full_train_mi': full_mi,
        'teacher_student_prediction_agreement': float(
            (teacher_pred == student_pred).mean() * 100.0),
        'mean_abs_probability_diff': float(
            np.abs(teacher_prob.numpy() - student_prob.numpy()).mean()),
    }


def _session_default(dataset):
    return {
        'BNCI2014001': 'sessionT',
        'BNCI2014004': 'session3',
        'BNCI2015001': 'session_A (loader default)',
        'AlexMI': 'canonical subject block (loader default)',
    }.get(dataset, 'loader default')


def _device(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f'cuda:{gpu}'
    return 'cpu'


def _split_cell(dataset, protocol, key, seed, val_split, train_percentage):
    if protocol == 'fewshot':
        if train_percentage is not None:
            if not 0.0 < train_percentage < 1.0:
                raise ValueError('--train_percentage must be in (0, 1)')
            val_split = 1.0 - float(train_percentage)
        return data.subject_split(
            dataset, key, val_split=val_split, seed=seed, return_uid=True) + (None,)
    X_tr, y_tr, subj_tr, X_te, y_te, uid_tr, uid_te = data.loso_split(
        dataset, key, return_uid=True)
    return X_tr, y_tr, X_te, y_te, uid_tr, uid_te, subj_tr


def _check_teacher(tch, y_tr, uid_tr, expected_policy, context):
    uid_ref = np.asarray(uid_tr, dtype=np.int64)
    if uid_ref.ndim != 2 or uid_ref.shape[1] != 2:
        raise ValueError(f'{context}: current train sample_uid must have shape (N, 2)')
    if len({tuple(row) for row in uid_ref.tolist()}) != len(uid_ref):
        raise ValueError(f'{context}: current train sample_uid is not unique')
    if 'sample_uid' not in tch:
        raise ValueError(f'{context}: teacher artifact missing sample_uid; regenerate artifacts')
    uid_teacher = np.asarray(tch['sample_uid'], dtype=np.int64)
    if uid_teacher.ndim != 2 or uid_teacher.shape[1] != 2:
        raise ValueError(f'{context}: teacher sample_uid must have shape (N, 2)')
    if len({tuple(row) for row in uid_teacher.tolist()}) != len(uid_teacher):
        raise ValueError(f'{context}: teacher sample_uid is not unique')
    if set(map(tuple, uid_teacher.tolist())) != set(map(tuple, uid_ref.tolist())):
        raise ValueError(f'{context}: teacher/train UID sets differ; refusing an inner join')
    order = np.asarray(
        [dict((tuple(row), index) for index, row in enumerate(uid_teacher.tolist()))[
            tuple(row)] for row in uid_ref.tolist()],
        dtype=np.int64)
    reordered = not np.array_equal(uid_teacher, uid_ref)
    for name in ('logits', 'feats', 'y'):
        if name not in tch:
            raise ValueError(f'{context}: teacher artifact missing {name}')
        array = np.asarray(tch[name])
        if array.ndim == 0 or len(array) != len(uid_teacher):
            raise ValueError(f'{context}: teacher {name} first dimension is misaligned')
        if name != 'y' and not np.isfinite(array).all():
            raise ValueError(f'{context}: teacher {name} contains NaN/Inf')
        tch[name] = array[order]
    tch['sample_uid'] = uid_teacher[order]
    labels = np.asarray(tch['y'], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(y_tr, dtype=np.int64)):
        raise ValueError(f'{context}: teacher labels disagree after UID alignment')
    policy = tch.get('split_policy')
    if policy is None:
        raise ValueError(f'{context}: teacher artifact missing split_policy; regenerate artifacts')
    if policy != expected_policy:
        raise ValueError(
            f'{context}: split_policy={policy!r}, expected {expected_policy!r}')
    return {
        'status': 'pass',
        'uid_set_match': True,
        'reordered': bool(reordered),
        'labels_match': True,
    }


def _load_teacher(dataset, artifact_name, key, seed, y_tr, uid_tr,
                  expected_policy, root):
    tch = artifacts.load(dataset, artifact_name, key, seed, 'train', root=root)
    alignment = _check_teacher(
        tch, y_tr, uid_tr, expected_policy,
        f'{dataset} {artifact_name} key={key} seed={seed}')
    path = artifacts.artifact_path(dataset, artifact_name, key, seed, 'train', root)
    tch['_uid_alignment'] = alignment
    tch['_artifact_info'] = _artifact_info(path)
    return tch


def _teacher_acc(tch, y_tr):
    logits = tch.get('logits')
    if logits is None:
        return ''
    return float((np.asarray(logits).argmax(1) == y_tr).mean() * 100.0)


def _student_runtime_cfg(student, dataset, protocol, args, X_tr):
    cfg = config.load_model_config(student, dataset, protocol)
    for key in ('epochs', 'lr', 'weight_decay', 'batch_size'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2], dataset_name=dataset)
    return cfg


def _method_kwargs(method, args, tch, y_tr):
    base = dict(
        temperature=args.temperature,
        lam_kd=0.0,
        lam_mi=0.0,
        lam_feat=0.0,
        lam_mmd=0.0,
        probability_mi=False,
        teacher_correct_only=False,
        mmd_sigmas=tuple(args.mmd_sigmas),
        mmd_normalize=args.mmd_normalize,
        mmd_class_conditional=args.mmd_class_conditional,
    )
    weight_mode = 'none'
    if method == 'Base':
        return base, weight_mode
    if method == 'KD_all':
        base['lam_kd'] = args.lam_kd
        weight_mode = 'all'
    elif method == 'KD_masked':
        base['lam_kd'] = args.lam_kd
        base['sample_weight'] = (tch['logits'].argmax(1) == y_tr).astype(np.float32)
        weight_mode = 'teacher_correct'
    elif method == 'MMD':
        base['lam_mmd'] = args.lam_mmd
        weight_mode = 'all'
    elif method == 'KD_MMD':
        base['lam_kd'] = args.lam_kd
        base['lam_mmd'] = args.lam_mmd
        weight_mode = 'all'
    elif method == 'CE_MI':
        base['lam_mi'] = args.lam_mi
        base['probability_mi'] = True
        weight_mode = 'none'
    else:
        raise ValueError(f'unknown method {method!r}')
    return base, weight_mode


def _run_method(args, dataset, protocol, teacher, student, teacher_artifact,
                   key, seed, method, X_tr, y_tr, X_te, y_te, subj_tr, tch,
                   nc, device, initial_state_dict=None, uid_tr=None,
                   uid_te=None, return_details=False):
    scfg = _student_runtime_cfg(student, dataset, protocol, args, X_tr)
    adapter = get_adapter(student, device=device, **scfg)
    kw, weight_mode = _method_kwargs(method, args, tch, y_tr)
    kw.update(
        epochs=scfg.get('epochs', 50),
        lr=scfg.get('lr', 1e-3),
        weight_decay=scfg.get('weight_decay', 0.01),
        batch_size=scfg.get('batch_size', 16),
        seed=seed,
        subject_ids=subj_tr,
        balanced_batch=(protocol == 'loso' and subj_tr is not None),
    )
    # Only feature-alignment methods receive cached teacher features.  Base,
    # vanilla KD and CE+MI are logits/label paths and must not touch feats.
    teacher_feats = tch['feats'] if method in {'MMD', 'KD_MMD'} else None
    teacher_logits = tch['logits'] if method != 'Base' else None
    if initial_state_dict is not None:
        kw['initial_state_dict'] = initial_state_dict
    if uid_tr is not None:
        kw['sample_uid'] = uid_tr
    if return_details:
        preds, details = distill_student(
            adapter, nc, X_tr, y_tr, teacher_feats, teacher_logits, X_te,
            return_training_details=True, **kw)
        mi_history = details['full_train_mi_history']
    elif method == 'CE_MI':
        preds, mi_history = distill_student(
            adapter, nc, X_tr, y_tr, teacher_feats, teacher_logits, X_te,
            return_mi_history=True, **kw)
        details = None
    else:
        preds = distill_student(
            adapter, nc, X_tr, y_tr, teacher_feats, teacher_logits, X_te, **kw)
        mi_history = []
        details = None
    m = metrics.evaluate(y_te, preds)
    test_balanced_accuracy = float(balanced_accuracy_score(y_te, preds) * 100.0)
    train_logits = details['train_logits'] if details is not None else None
    train_preds = details['train_preds'] if details is not None else None
    final_diagnostics = {}
    if train_logits is not None and tch.get('logits') is not None:
        final_diagnostics = _final_train_mi_and_agreement(
            tch['logits'], train_logits, train_preds)
    history = details['training_history'] if details is not None else []
    last_history = history[-1] if history else {}
    prediction_counts = np.bincount(
        np.asarray(preds, dtype=np.int64), minlength=int(nc)).tolist()
    metric_values = [m['acc'], m['kappa'], test_balanced_accuracy]
    metrics_finite = all(_finite_metric(value) for value in metric_values)
    batch_order_hashes = details['batch_order_hashes'] if details is not None else []
    initial_hash = (_sha256_state_dict(initial_state_dict)
                    if initial_state_dict is not None else '')
    train_uid_hash = _sha256_array(uid_tr) if uid_tr is not None else ''
    test_uid_hash = _sha256_array(uid_te) if uid_te is not None else ''
    row = {
        'dataset': dataset,
        'subject': int(key) + 1,
        'key': int(key),
        'session': _session_default(dataset),
        'protocol': protocol,
        'teacher': teacher,
        'student': student,
        'teacher_artifact': teacher_artifact,
        'seed': int(seed),
        'method': method,
        'condition_label': METHOD_LABELS.get(method, method),
        'acc': m['acc'],
        'test_accuracy': m['acc'],
        'test_balanced_accuracy': round(test_balanced_accuracy, 4),
        'kappa': m['kappa'],
        'test_kappa': m['kappa'],
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te)),
        'train_count': int(len(y_tr)),
        'test_count': int(len(y_te)),
        'num_classes': int(nc),
        'teacher_train_acc_pct': (round(_teacher_acc(tch, y_tr), 2)
                                  if tch.get('logits') is not None else ''),
        'teacher_train_uid_alignment': json.dumps(
            tch.get('_uid_alignment', {}), sort_keys=True),
        'teacher_artifact_path': tch.get('_artifact_info', {}).get('path', ''),
        'teacher_artifact_sha256': tch.get('_artifact_info', {}).get('sha256', ''),
        'teacher_target_manifest_path': tch.get('_manifest_path', ''),
        'teacher_target_manifest_sha256': tch.get('_manifest_sha256', ''),
        'teacher_checkpoint_sha256': tch.get('_manifest', {}).get(
            'teacher_checkpoint_sha256', ''),
        'source_manifest_sha256': tch.get('_source_manifest_sha256', ''),
        'student_input_manifest_sha256': tch.get('_student_input_manifest_sha256', ''),
        'student_input_policy_sha256': tch.get('_student_input_policy_sha256', ''),
        'teacher_input_manifest_sha256': tch.get('_input_manifest_sha256', ''),
        'teacher_input_policy_sha256': tch.get('_input_policy_sha256', ''),
        'train_uid_hash': train_uid_hash,
        'test_uid_hash': test_uid_hash,
        'split_uid_hash': _sha256_uid_split(uid_tr, uid_te)
        if uid_tr is not None and uid_te is not None else '',
        'initial_state_hash': initial_hash,
        'batch_order_hash': _combined_hash(batch_order_hashes)
        if batch_order_hashes else '',
        'batch_order_hashes': json.dumps(batch_order_hashes),
        'lam_kd': kw['lam_kd'],
        'lam_mi': kw['lam_mi'],
        'lam_mmd': kw['lam_mmd'],
        'temperature': args.temperature,
        'weight_mode': weight_mode,
        'student_epochs': kw['epochs'],
        'student_lr': kw['lr'],
        'student_weight_decay': kw['weight_decay'],
        'student_batch_size': kw['batch_size'],
        'mmd_sigmas': ' '.join(str(x) for x in args.mmd_sigmas),
        'mmd_normalize': bool(args.mmd_normalize),
        'mmd_class_conditional': bool(args.mmd_class_conditional),
        'optimizer': str(scfg.get('optimizer', 'adamw')).lower(),
        'scheduler': 'CosineAnnealingLR',
        'preprocessing': json.dumps({
            'use_filter_bank': bool(scfg.get('use_filter_bank', False)),
            'in_channels': int(X_tr.shape[1]),
            'samples': int(X_tr.shape[2]),
        }, sort_keys=True),
        'final_train_loss': last_history.get('total_loss', ''),
        'final_train_accuracy': last_history.get('train_accuracy', ''),
        'final_ce_loss': last_history.get('ce_loss', ''),
        'final_mi_loss': last_history.get('mi_loss', ''),
        'full_train_mi': final_diagnostics.get('full_train_mi', ''),
        'full_train_mi_last': (mi_history[-1] if mi_history else
                               final_diagnostics.get('full_train_mi', '')),
        'full_train_mi_history': ' '.join(str(value) for value in mi_history),
        'teacher_student_prediction_agreement': final_diagnostics.get(
            'teacher_student_prediction_agreement', ''),
        'mean_abs_probability_diff': final_diagnostics.get(
            'mean_abs_probability_diff', ''),
        'predicted_class_counts': json.dumps(prediction_counts),
        'collapse_flag': bool(len(np.unique(preds)) < 2),
        'failure_status': 'complete' if metrics_finite else 'invalid_metrics',
        'failure_reason': '' if metrics_finite else 'non-finite evaluation metric',
    }
    if return_details:
        return row, details
    return row


def _capture_initial_state(args, dataset, student, X_tr, nc, device, seed):
    """Build one deterministic student initialization for a fold.

    The returned CPU tensors are loaded into every Base/KD_all/CE_MI run for
    that fold, so condition differences cannot be attributed to initialization
    drift.  This helper never writes a checkpoint.
    """
    _set_seed(seed)
    scfg = _student_runtime_cfg(student, dataset, 'fewshot', args, X_tr)
    adapter = get_adapter(student, device=device, **scfg)
    model = adapter.build(int(nc))
    state = {key: value.detach().cpu().clone()
             for key, value in model.state_dict().items()}
    del model, adapter
    if device != 'cpu':
        torch.cuda.empty_cache()
    return state


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in BASE_FIELDS})


def _output_run_id():
    return f'{_log_timestamp()}_{os.getpid()}'


def _is_wideband_loso(dataset, protocol):
    return protocol == 'loso' and dataset in {
        'BNCI2014001', 'BNCI2014001-4', 'BNCI2014004', 'BNCI2015001',
    }


def _default_out_csv(args, datasets, teachers, students, protocol, run_id=None):
    if args.out_csv:
        return str(require_report_output(args.out_csv))
    out_dir = (Path(REPORTS_ROOT) / 'loso' / 'distill'
               if protocol == 'loso' else Path(REPORTS_ROOT) / 'distill')
    run_id = run_id or _output_run_id()
    if len(datasets) == len(teachers) == len(students) == 1:
        source_suffix = '_wide_fre_band' if _is_wideband_loso(datasets[0], protocol) else ''
        name = (f'{datasets[0]}_{protocol}_distill_{teachers[0]}_to_{students[0]}'
                f'{source_suffix}_{run_id}.csv')
    else:
        name = f'{protocol}_distill_matrix_{run_id}.csv'
    return str(require_report_output(out_dir / name))


def _pair_out_csv(out_csv, dataset, teacher, student, protocol,
                  explicit_out, multi_combo):
    if not multi_combo:
        return str(out_csv)
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    stem, ext = os.path.splitext(os.path.basename(out_csv))
    ext = ext or '.csv'
    if explicit_out:
        filename = f'{stem}_{dataset}_{protocol}_{teacher}_to_{student}{ext}'
    else:
        source_suffix = '_wide_fre_band' if _is_wideband_loso(dataset, protocol) else ''
        filename = (f'{stem}_{dataset}_{protocol}_{teacher}_to_{student}'
                    f'{source_suffix}{ext}')
    return str(require_report_output(Path(out_dir) / filename))


def _write_outputs(out_csv, rows, protocol, explicit_out, multi_combo):
    """Write one six-column result CSV per dataset/teacher/student cell."""
    combos = {}
    for row in rows:
        combo = (row['dataset'], row['teacher'], row['student'])
        combos.setdefault(combo, []).append(row)
    if not multi_combo:
        _write_csv(out_csv, rows)
        return [(out_csv, len(rows))]

    paths = []
    for dataset, teacher, student in sorted(combos):
        path = _pair_out_csv(
            out_csv, dataset, teacher, student, protocol,
            explicit_out=explicit_out, multi_combo=True)
        _write_csv(path, combos[(dataset, teacher, student)])
        paths.append((path, len(combos[(dataset, teacher, student)])))
    return paths


def _print_summary(rows):
    if not rows:
        print('[summary] no rows produced', flush=True)
        return
    print('[summary]', flush=True)
    for method in (*DEFAULT_METHODS, 'CE_MI'):
        vals = [float(r['acc']) for r in rows if r['method'] == method]
        kappas = [float(r['kappa']) for r in rows if r['method'] == method]
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        k = np.asarray(kappas, dtype=np.float64)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        print(
            f'  {METHOD_LABELS.get(method, method)} [{method}]: '
            f'n={len(a)} acc={a.mean():.2f} +/- {std:.2f} '
            f'kappa={k.mean():.4f}',
            flush=True,
        )


def _log_timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def _open_default_log(args):
    """Create a unique persistent LOSO log without replacing older runs."""
    timestamp = _log_timestamp()
    pid = os.getpid()
    if args.out_csv:
        output = Path(args.out_csv).expanduser().resolve(strict=False)
        parent = output.parent
        stem = output.stem or 'distill'
        basename = f'{stem}_{timestamp}_{pid}'
    else:
        parent = Path(REPORTS_ROOT) / 'loso' / 'distill' / 'logs'
        basename = f'{timestamp}_{pid}'
    parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        suffix = '' if index == 0 else f'_{index}'
        path = parent / f'{basename}{suffix}.log'
        try:
            return path, path.open('x', buffering=1)
        except FileExistsError:
            index += 1


def _log_seed_policy(args):
    if args.seeds is None:
        student = ('dataset configuration seeds (LOSO: loso.seeds, then seeds; '
                   'other protocols: seeds)')
    else:
        student = f'explicit --seeds {list(args.seeds)}'
    teacher = (f'explicit --teacher-seed {int(args.teacher_seed)}'
               if args.teacher_seed is not None else 'same-as-student-seed')
    return f'student={student}; teacher={teacher}'


def _run(args):
    _set_thread_defaults()
    protocol = data.canonical_protocol(args.protocol)
    args.protocol = protocol
    explicit_teacher_seed = getattr(args, 'teacher_seed', None)
    _validate_teacher_seed_protocol(protocol, explicit_teacher_seed)
    args._wide_source_default = protocol == 'loso'
    if args.artifact_root is None:
        args.artifact_root = str(LOSO_CACHE_ROOT if protocol == 'loso'
                                 else artifacts.ARTIFACT_ROOT)
    methods = _parse_methods(args.methods)
    if 'CE_MI' in methods:
        if protocol != 'fewshot':
            raise ValueError('CE_MI is subject-wise fewshot only; non-fewshot protocols are forbidden')
        if args.keys is not None:
            raise ValueError('CE_MI requires all subjects; do not pass --subjects/--keys')
    datasets = _resolve_list(args.datasets, args.dataset, ['BNCI2014004'])
    teachers = _resolve_list(args.teachers, args.teacher, ['mirepnet'])
    students = _resolve_list(args.students, args.student, ['ifnet'])
    _validate_names(teachers, students)
    if args.teacher_artifact is not None and len(teachers) != 1:
        raise ValueError('--teacher_artifact is only valid with one teacher')
    if protocol == 'loso' and args.teacher_artifact is not None:
        raise ValueError('--teacher_artifact overrides are not supported for canonical wide-source LOSO')
    if args.val_split is not None and args.train_percentage is not None:
        raise ValueError('pass only one of --val_split or --train_percentage')

    device = _device(args.gpu)
    expected_policy = _expected_split_policy(protocol)
    output_run_id = _output_run_id()
    out_csv = _default_out_csv(
        args, datasets, teachers, students, protocol, run_id=output_run_id)
    if args.out_csv is None and protocol == 'loso':
        default_outputs = (
            [out_csv] if len(datasets) == len(teachers) == len(students) == 1
            else [_pair_out_csv(
                out_csv, dataset, teacher, student, protocol,
                explicit_out=False, multi_combo=True)
                  for dataset in datasets for teacher in teachers for student in students]
        )
        existing = [path for path in default_outputs if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                f'refusing to overwrite existing LOSO output(s): {existing}; '
                'this run was assigned unique automatic output paths')
    print(
        f'[distill] datasets={datasets} teachers={teachers} students={students} '
        f'protocol={protocol} methods={methods} device={device} '
        f'teacher_seed_policy={explicit_teacher_seed if explicit_teacher_seed is not None else "same-as-student"}',
        flush=True,
    )
    print(
        f'[artifacts] root={args.artifact_root} require_uid=True '
        f'require_split_policy=True expected_policy={expected_policy}',
        flush=True,
    )

    rows, errors = [], []
    for dataset in datasets:
        dcfg = config.load_dataset_config(dataset)
        keys = args.keys if args.keys is not None else list(range(dcfg['num_subjects']))
        seeds = (args.seeds if args.seeds is not None else
                 dcfg.get('loso', {}).get('seeds', dcfg['seeds'])
                 if protocol == 'loso' else dcfg['seeds'])
        print(
            f'[selection] dataset={dataset} student_seeds={list(seeds)} '
            f'teacher_seed_policy={explicit_teacher_seed if explicit_teacher_seed is not None else "same-as-student"}',
            flush=True,
        )
        nc = int(dcfg['num_classes'])
        val_split = args.val_split if args.val_split is not None else dcfg['val_split']
        for teacher in teachers:
            teacher_artifact = _teacher_artifact_name(
                teacher, protocol, override=args.teacher_artifact)
            for student in students:
                student_data = (_load_or_prepare_loso_dataset(
                    dataset, student, cache_root=args.artifact_root,
                    no_auto_teacher_cache=getattr(args, 'no_auto_teacher_cache', False))
                    if protocol == 'loso' else None)
                for seed in seeds:
                    resolved_teacher_seed = _resolve_teacher_seed(
                        seed, explicit_teacher_seed)
                    for key in keys:
                        try:
                            if student_data is not None:
                                train_mask = student_data.subjects != int(key)
                                test_mask = ~train_mask
                                X_tr, y_tr = student_data.x[train_mask], student_data.y[train_mask]
                                X_te, y_te = student_data.x[test_mask], student_data.y[test_mask]
                                uid_tr = student_data.sample_uid[train_mask]
                                uid_te = student_data.sample_uid[test_mask]
                                subj_tr = student_data.subjects[train_mask]
                                if not len(y_tr) or not len(y_te):
                                    raise ValueError(f'{dataset}: fold {key} has an empty train or test split')
                                tch = {
                                    '_source_manifest_sha256': student_data.source_manifest_sha256,
                                    '_student_input_manifest_sha256': (
                                        student_data.input_manifest_sha256 or ''),
                                    '_student_input_policy_sha256': student_data.identity.get(
                                        'input_policy_sha256', ''),
                                }
                            else:
                                X_tr, y_tr, X_te, y_te, uid_tr, uid_te, subj_tr = _split_cell(
                                    dataset, protocol, key, seed, val_split, args.train_percentage)
                                tch = None
                            initial_state_dict = None
                            if 'CE_MI' in methods:
                                # Keep the three MI conditions on one exact
                                # initialization for this subject/seed cell.
                                initial_state_dict = _capture_initial_state(
                                    args, dataset, student, X_tr, nc, device, seed)
                            for method in methods:
                                if student_data is not None:
                                    if method == 'Base':
                                        method_tch = dict(tch)
                                    else:
                                        method_tch = _load_or_ensure_loso_teacher_targets(
                                            dataset, teacher, resolved_teacher_seed, key,
                                            student_data=student_data,
                                            cache_root=args.artifact_root,
                                            gpu=args.gpu,
                                            no_auto_teacher_cache=getattr(
                                                args, 'no_auto_teacher_cache', False))
                                        alignment = _check_teacher(
                                            method_tch, y_tr, uid_tr, expected_policy,
                                            f'{dataset} {teacher} key={key} '
                                            f'student_seed={seed} teacher_seed={resolved_teacher_seed}')
                                        method_tch['_uid_alignment'] = alignment
                                        method_tch['_artifact_info'] = _artifact_info(
                                            method_tch['_artifact_path'])
                                        method_tch['_student_input_manifest_sha256'] = (
                                            student_data.input_manifest_sha256 or '')
                                        method_tch['_student_input_policy_sha256'] = (
                                            student_data.identity.get('input_policy_sha256', ''))
                                elif tch is None:
                                    method_tch = _load_teacher(
                                        dataset, teacher_artifact, key, seed, y_tr, uid_tr,
                                        expected_policy, args.artifact_root)
                                else:
                                    method_tch = tch
                                method_kwargs = {}
                                if initial_state_dict is not None:
                                    method_kwargs['initial_state_dict'] = initial_state_dict
                                row = _run_method(
                                    args, dataset, protocol, teacher, student,
                                    teacher_artifact, key, seed, method,
                                    X_tr, y_tr, X_te, y_te, subj_tr, method_tch,
                                    nc, device,
                                    **method_kwargs)
                                row['teacher_seed'] = (
                                    '' if method == 'Base' else resolved_teacher_seed)
                                if student_data is not None:
                                    row.update(
                                        source_manifest_sha256=student_data.source_manifest_sha256,
                                        student_input_manifest_sha256=(
                                            student_data.input_manifest_sha256 or ''),
                                        student_input_policy_sha256=student_data.identity.get(
                                            'input_policy_sha256', ''),
                                        input_manifest_sha256=(
                                            student_data.input_manifest_sha256 or ''))
                                rows.append(row)
                                print(
                                    f'[{dataset}] {teacher}->{student} key={key} '
                                    f'student_seed={seed} teacher_seed='
                                    f'{row["teacher_seed"] if method != "Base" else ""} '
                                    f'{method} acc={row["acc"]} '
                                    f'kappa={row["kappa"]}',
                                    flush=True,
                                )
                                if device != 'cpu':
                                    torch.cuda.empty_cache()
                        except Exception as e:  # noqa: BLE001 - keep filling matrix
                            msg = (f'{dataset} {teacher_artifact}->{student} '
                                   f'key={key} seed={seed}: {e}')
                            errors.append(msg)
                            print(f'[ERR] {msg}', flush=True)
                            if args.fail_fast:
                                raise

    if rows:
        multi_combo = not (len(datasets) == len(teachers) == len(students) == 1)
        written = _write_outputs(
            out_csv, rows, protocol, explicit_out=args.out_csv is not None,
            multi_combo=multi_combo)
        if len(written) == 1:
            print(f'\nWrote {written[0][0]} rows={written[0][1]}', flush=True)
        else:
            for path, count in written:
                print(f'Wrote {path} rows={count}', flush=True)
    else:
        print('\nNo rows produced.', flush=True)
    _print_summary(rows)
    if errors:
        print(f'[errors] {len(errors)} cells failed', flush=True)
    print('Done.', flush=True)
    return 1 if errors and not rows else 0


def main(argv=None):
    raw_argv = normalize_cli_argv(argv)
    args = parse_args(raw_argv)
    if args.out_csv:
        args.out_csv = str(require_report_output(args.out_csv))
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_f = log_path.open('w', buffering=1)
    elif args.protocol == 'loso':
        log_path, log_f = _open_default_log(args)
        args.log_file = str(log_path)
    else:
        return _run(args)

    with log_f:
        tee_out = _Tee(sys.stdout, log_f)
        tee_err = _Tee(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f'[log] writing stdout/stderr to {log_path}', flush=True)
            print(f'[run] argv={json.dumps(raw_argv, ensure_ascii=False)}', flush=True)
            print(f'[seed-policy] {_log_seed_policy(args)}', flush=True)
            try:
                return _run(args)
            except BaseException:
                print('[exception] direct distillation run failed; traceback follows',
                      file=log_f, flush=True)
                traceback.print_exc(file=log_f)
                log_f.flush()
                raise


if __name__ == "__main__":
    if any(arg == "--config" or arg.startswith("--config=")
           for arg in sys.argv[1:]):
        from experiments.distill.config_runner import main as config_main
        raise SystemExit(config_main())
    raise SystemExit(main())
