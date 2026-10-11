"""Matched CE/KD LOSO run with diagnostic-only per-sample c trajectories.

The c score is measured after training and never enters either student's loss.
Teacher fitting and the shared CE warm-up reuse the seven-source protocol and
the frozen, provenance-checked Teacher artifacts from the sample-utility pilot.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)

from experiments.distill import run_sample_utility_adaptive_rl_loso as trainer
from experiments.distill import sample_utility_protocol as protocol
from experiments.distill.c_metric_diagnostics import (
    WINDOWS, c_components, late_signed_probability_gain,
    spearman_or_none, strict_median_groups,
)
from experiments.finetune import run_loso_five_datasets as canonical
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
                                 require_external_output, resolve_local_file)
from models import get_adapter


CONFIG_PATH = Path('configs/experiments/sample_utility_c_metric_loso_seed666.yaml')
PROTOCOL_ID = 'sample_utility_c_metric_loso_seed666_v2'
RESULT_ROOT = require_external_output(RESULTS_ROOT / 'distill' / PROTOCOL_ID)
EXPERIMENT_CACHE_ROOT = require_external_output(DATA_CACHE_ROOT / PROTOCOL_ID)
EXPERIMENT_WEIGHT_ROOT = require_external_output(WEIGHTS_ROOT / PROTOCOL_ID)
CONDITIONS = ('BASE_CE', 'DELAYED_KD_ALL')
PILOT_RESULT_ROOT = require_external_output(
    RESULTS_ROOT / 'distill' / 'sample_utility_adaptive_rl_loso_pilot_v1')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG_PATH))
    parser.add_argument('--stage', choices=('preflight', 'train', 'analyze',
                                             'evaluate', 'all'), default='preflight')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='zero-based target subject indices; default is all nine')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='one-fold 12-epoch state-transition smoke; never evaluates target')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total epochs; values below 100 are allowed only with --smoke')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-root', default=None,
                        help='external result root; smoke may use a unique /tmp directory')
    return parser.parse_args()


def device_for(gpu):
    if gpu is None:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requested but CUDA is unavailable')
    if not 0 <= int(gpu) < torch.cuda.device_count():
        raise ValueError(f'invalid GPU index {gpu}')
    torch.cuda.set_device(int(gpu))
    return torch.device(f'cuda:{int(gpu)}')


def load_experiment_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(resolve_local_file(path).read_text())
    required = {
        'protocol_id': PROTOCOL_ID, 'dataset': 'BNCI2014004', 'session': 'session_3',
        'seed': 666, 'target_folds': 9,
        'split': 'target_t_feedback_next_subject_train_remaining_7',
        'teacher': 'mirepnet', 'student': 'ifnet', 'teacher_frozen': True,
        'student_conditions': {'CE': 'BASE_CE', 'KD': 'DELAYED_KD_ALL'},
        'epochs': 100, 'warmup_epochs': 10, 'batch_size': 16,
        'lambda_kd': 0.5, 'temperature_kd': 2.0,
        'kd_type': 'logits', 'kd_all_samples': True,
        'target_evaluation_epoch': 100,
    }
    for key, value in required.items():
        if raw.get(key) != value:
            raise ValueError(f'c metric config {key} must be {value!r}, got {raw.get(key)!r}')
    if raw.get('c_metric', {}).get('use_for_training') is not False:
        raise ValueError('c metric must remain diagnostic-only')
    if raw['c_metric'].get('windows') != {'c_1_100': [1, 100],
                                           'c_1_30': [1, 30],
                                           'c_11_100': [11, 100]}:
        raise ValueError('c metric windows differ from the preregistered definitions')
    if raw.get('primary_predictive_check') != \
            'spearman_c_1_30_vs_late_benefit_on_feedback':
        raise ValueError('primary predictive check changed')
    return raw


def _external_roots(raw, args):
    if args.output_root:
        out = require_external_output(Path(args.output_root))
    elif args.smoke:
        out = require_external_output(Path('/tmp') / f'{PROTOCOL_ID}_smoke_{os.getpid()}')
    else:
        if require_external_output(raw['output_root']) != RESULT_ROOT:
            raise ValueError('configured c metric output_root differs from experiments.storage')
        if require_external_output(raw['cache_root']) != EXPERIMENT_CACHE_ROOT:
            raise ValueError('configured c metric cache_root differs from experiments.storage')
        if require_external_output(raw['weights_root']) != EXPERIMENT_WEIGHT_ROOT:
            raise ValueError('configured c metric weights_root differs from experiments.storage')
        out = RESULT_ROOT
    if args.smoke:
        cache = require_external_output(out / 'cache')
        weights = require_external_output(out / 'weights')
    else:
        cache = EXPERIMENT_CACHE_ROOT
        weights = EXPERIMENT_WEIGHT_ROOT
    return out, cache, weights


def _make_fold_paths(old_paths, out_root, cache_root, weight_root):
    key = old_paths['key']
    paths = dict(old_paths)
    paths['results'] = require_external_output(out_root / 'folds' / key)
    paths['cache'] = require_external_output(cache_root / 'folds' / key)
    paths['weights'] = require_external_output(weight_root / 'folds' / key)
    for key_name in ('results', 'cache', 'weights'):
        paths[key_name].mkdir(parents=True, exist_ok=True)
    return paths


def _student_probabilities(adapter, model, x, device, batch_size=64):
    """Eval probabilities on a copy, preserving the live model and RNG state."""
    x = np.asarray(x, dtype=np.float32)
    before_rng = trainer.rng_state(device)
    before_model = trainer.state_hash(model.state_dict())
    # IFNet LinearWithConstraint projects fc.weight.data inside forward().
    evaluation_model = copy.deepcopy(model)
    parts = []
    try:
        evaluation_model.eval()
        with trainer.preserve_torch_rng(device), torch.inference_mode():
            for start in range(0, len(x), int(batch_size)):
                xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32,
                                     device=device)
                _, logits = adapter.forward(evaluation_model, xb)
                parts.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    finally:
        trainer.restore_rng(before_rng, device)
        del evaluation_model
    if trainer.state_hash(model.state_dict()) != before_model:
        raise RuntimeError('eval prediction changed the live Student model state')
    after_rng = trainer.rng_state(device)
    if before_rng['python'] != after_rng['python']:
        raise RuntimeError('eval prediction changed Python RNG state')
    if (before_rng['numpy'][0] != after_rng['numpy'][0]
            or not np.array_equal(before_rng['numpy'][1], after_rng['numpy'][1])
            or before_rng['numpy'][2:] != after_rng['numpy'][2:]):
        raise RuntimeError('eval prediction changed NumPy RNG state')
    if not torch.equal(before_rng['torch_cpu'], after_rng['torch_cpu']):
        raise RuntimeError('eval prediction changed the Student CPU RNG stream')
    if device.type == 'cuda' and not torch.equal(before_rng['torch_cuda'], after_rng['torch_cuda']):
        raise RuntimeError('eval prediction changed the Student CUDA RNG stream')
    output = np.concatenate(parts).astype(np.float32) if parts else \
        np.empty((0, 2), dtype=np.float32)
    if output.shape != (len(x), 2) or not np.isfinite(output).all():
        raise RuntimeError('eval prediction has invalid shape or non-finite values')
    return output


def _capture_student_epoch(paths, epoch, adapter, model, train_data, split, device,
                           *, condition=None):
    train_prob = _student_probabilities(
        adapter, model, train_data['arrays']['train_x'], device)
    feedback_prob = _student_probabilities(
        adapter, model, train_data['arrays']['feedback_x'], device)
    if condition is None:
        destination = paths['results'] / 'shared_warmup' / f'epoch_{epoch:03d}.npz'
    else:
        destination = (paths['results'] / condition / 'prediction_dynamics'
                       / f'epoch_{epoch:03d}.npz')
    protocol.atomic_npz(
        destination, epoch=np.asarray(epoch, dtype=np.int32),
        train_uids=np.asarray(split['train']['uids'], dtype=np.int64),
        train_y=np.asarray(split['train']['y'], dtype=np.int64),
        train_subjects=np.asarray(split['train']['subjects'], dtype=np.int64),
        train_probabilities=train_prob,
        feedback_uids=np.asarray(split['feedback']['uids'], dtype=np.int64),
        feedback_y=np.asarray(split['feedback']['y'], dtype=np.int64),
        feedback_subjects=np.asarray(split['feedback']['subjects'], dtype=np.int64),
        feedback_probabilities=feedback_prob,
    )


def _nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) \
            and left.shape == right.shape and left.dtype == right.dtype \
            and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) \
            and left.keys() == right.keys() \
            and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)) \
            and len(left) == len(right) \
            and all(_nested_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _teacher_probabilities(old_paths, split, teacher_cache, teacher_bundle,
                           resolved, device, output_dir):
    provenance = teacher_bundle[1]
    expected_train = sorted(np.unique(split['train']['subjects']).astype(int).tolist())
    expected_feedback = int(np.unique(split['feedback']['subjects'])[0]) + 1
    expected_target = int(np.unique(split['test']['subjects'])[0]) + 1
    checks = {
        'train_uid_sha256': protocol.hash_uids(split['train']['uids']),
        'train_subject_ids': expected_train,
        'feedback_subject_id': expected_feedback,
        'target_subject_id': expected_target,
    }
    for key, expected in checks.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f'Teacher provenance mismatch for {key}: '
                               f'{provenance.get(key)!r} != {expected!r}')
    adapter = get_adapter('mirepnet', device=device, **resolved['teacher_config'])
    model = adapter.build(2)
    model.load_state_dict(teacher_bundle[2]['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    state_hash_before = trainer.state_hash(model.state_dict())
    feedback_raw = canonical._prepare_mirepnet(
        split['feedback']['x'], split['feedback']['subjects'], adapter)
    feedback_x = adapter.preprocess(feedback_raw).cpu().numpy().astype(np.float32)
    feedback_prob = _student_probabilities(adapter, model, feedback_x, device)
    teacher_train_prob = F.softmax(
        torch.as_tensor(teacher_cache['logits'], dtype=torch.float32), dim=1).numpy()
    state_hash_after = trainer.state_hash(model.state_dict())
    if state_hash_before != state_hash_after:
        raise RuntimeError('frozen Teacher parameters or BatchNorm buffers changed')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('frozen Teacher unexpectedly accumulated gradients')
    output = output_dir / 'teacher_predictions.npz'
    protocol.atomic_npz(
        output,
        train_uids=np.asarray(split['train']['uids'], dtype=np.int64),
        train_y=np.asarray(split['train']['y'], dtype=np.int64),
        train_subjects=np.asarray(split['train']['subjects'], dtype=np.int64),
        train_probabilities=np.asarray(teacher_train_prob, dtype=np.float32),
        feedback_uids=np.asarray(split['feedback']['uids'], dtype=np.int64),
        feedback_y=np.asarray(split['feedback']['y'], dtype=np.int64),
        feedback_subjects=np.asarray(split['feedback']['subjects'], dtype=np.int64),
        feedback_probabilities=feedback_prob,
        teacher_checkpoint_sha256=np.asarray(
            protocol.sha256_file(old_paths['weights'] / 'teacher_final.pt')),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    return output


def _read_capture(path, split, expected_epoch):
    if not path.is_file():
        raise FileNotFoundError(f'missing per-epoch probability artifact: {path}')
    with np.load(resolve_local_file(path), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if int(data['epoch']) != int(expected_epoch):
        raise RuntimeError(f'epoch mismatch in {path}')
    for part in ('train', 'feedback'):
        for key, expected in ((f'{part}_uids', split[part]['uids']),
                              (f'{part}_y', split[part]['y']),
                              (f'{part}_subjects', split[part]['subjects'])):
            if not np.array_equal(data[key], expected):
                raise RuntimeError(f'{key} order/values mismatch in {path}')
        if not np.isfinite(data[f'{part}_probabilities']).all():
            raise RuntimeError(f'non-finite probabilities in {path}')
    return data


def _aggregate_dynamics(paths, split):
    warmup = [_read_capture(paths['results'] / 'shared_warmup'
                            / f'epoch_{epoch:03d}.npz', split, epoch)
              for epoch in range(0, 11)]
    condition_data = {}
    for condition in CONDITIONS:
        condition_data[condition] = [
            _read_capture(paths['results'] / condition / 'prediction_dynamics'
                          / f'epoch_{epoch:03d}.npz', split, epoch)
            for epoch in range(11, 101)]
    teacher_path = paths['results'] / 'teacher_predictions.npz'
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    with np.load(resolve_local_file(teacher_path), allow_pickle=False) as archive:
        teacher = {key: np.asarray(archive[key]) for key in archive.files}
    for part in ('train', 'feedback'):
        if not np.array_equal(teacher[f'{part}_uids'], split[part]['uids']):
            raise RuntimeError(f'Teacher {part} UIDs are misaligned')
        if not np.array_equal(teacher[f'{part}_y'], split[part]['y']):
            raise RuntimeError(f'Teacher {part} labels are misaligned')
    payload = {
        'epochs': np.arange(0, 101, dtype=np.int16),
        'train_uids': np.asarray(split['train']['uids'], dtype=np.int64),
        'train_y': np.asarray(split['train']['y'], dtype=np.int64),
        'train_subjects': np.asarray(split['train']['subjects'], dtype=np.int64) + 1,
        'feedback_uids': np.asarray(split['feedback']['uids'], dtype=np.int64),
        'feedback_y': np.asarray(split['feedback']['y'], dtype=np.int64),
        'feedback_subjects': np.asarray(split['feedback']['subjects'], dtype=np.int64) + 1,
        'teacher_train_probabilities': teacher['train_probabilities'],
        'teacher_feedback_probabilities': teacher['feedback_probabilities'],
    }
    for part in ('train', 'feedback'):
        key = f'{part}_probabilities'
        for condition in CONDITIONS:
            values = [row[key] for row in warmup]
            values.extend(row[key] for row in condition_data[condition])
            array = np.stack(values).astype(np.float32)
            if array.shape != (101, len(split[part]['y']), 2):
                raise RuntimeError(f'{condition} {part} dynamics have an unexpected shape')
            payload[f'{part}_prob_{"ce" if condition == "BASE_CE" else "kd"}'] = array
        if not np.array_equal(payload[f'{part}_prob_ce'][:11],
                              payload[f'{part}_prob_kd'][:11]):
            raise RuntimeError(f'{part} CE/KD predictions differ during shared warm-up')
    output = paths['results'] / 'prediction_dynamics.npz'
    protocol.atomic_npz(output, **payload)
    return payload


def _metric_row(y, probabilities, *, fold, role, condition, subject_id):
    prediction = np.asarray(probabilities).argmax(axis=1)
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(prediction, minlength=2)
    return {
        'target_fold': int(fold) + 1, 'data_role': role,
        'subject_id': int(subject_id) if isinstance(subject_id, (int, np.integer)) else str(subject_id),
        'condition': condition,
        'n_samples': int(len(y)),
        'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
        'kappa': float(cohen_kappa_score(y, prediction)),
        'prediction_count_class_0': int(counts[0]),
        'prediction_count_class_1': int(counts[1]),
        'collapse': bool(np.count_nonzero(counts) == 1),
        'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def _c_score_rows(fold, payload):
    rows = []
    for role in ('train', 'feedback'):
        y = payload[f'{role}_y']
        probabilities_ce = payload[f'{role}_prob_ce'][1:]
        probabilities_kd = payload[f'{role}_prob_kd'][1:]
        scores = c_components(probabilities_ce, probabilities_kd, y)
        gain = late_signed_probability_gain(probabilities_ce, probabilities_kd, y)
        for index in range(len(y)):
            uid = payload[f'{role}_uids'][index].tolist()
            row = {
                'target_fold': int(fold) + 1, 'data_role': role,
                'subject_id': int(payload[f'{role}_subjects'][index]),
                'sample_uid': json.dumps(uid, separators=(',', ':')),
                'uid_component_0': int(uid[0]), 'uid_component_1': int(uid[1]),
                'label': int(y[index]), 'late_gain_81_100': float(gain[index]),
            }
            for name, values in scores.items():
                suffix = name.removeprefix('c_')
                for component in ('D', 'B', 'c', 'c_normalized'):
                    column = (f'c_{suffix}' if component == 'c' else
                              f'{component}_{suffix}')
                    row[column] = float(values[component][index])
            rows.append(row)
    return rows


def _summary_stat_rows(fold, payload):
    rows = []
    for role in ('train', 'feedback'):
        y = payload[f'{role}_y']
        subjects = payload[f'{role}_subjects']
        p_ce, p_kd = payload[f'{role}_prob_ce'][1:], payload[f'{role}_prob_kd'][1:]
        scores = c_components(p_ce, p_kd, y)
        for window, components in scores.items():
            for component in ('D', 'B', 'c', 'c_normalized'):
                values = components[component]
                groups = [('all', np.ones(len(y), dtype=bool))]
                groups.extend((f'class_{label}', y == label) for label in np.unique(y))
                groups.extend((f'subject_{subject}', subjects == subject)
                              for subject in np.unique(subjects))
                for group_name, mask in groups:
                    selected = values[mask]
                    if not len(selected):
                        continue
                    q25, median, q75 = np.quantile(selected, [0.25, 0.5, 0.75])
                    rows.append({
                        'target_fold': int(fold) + 1, 'data_role': role,
                        'group': group_name, 'window': window,
                        'component': component, 'n_samples': int(len(selected)),
                        'mean': float(np.mean(selected)), 'std': float(np.std(selected)),
                        'min': float(np.min(selected)), 'q05': float(np.quantile(selected, .05)),
                        'q25': float(q25), 'median': float(median), 'q75': float(q75),
                        'q95': float(np.quantile(selected, .95)), 'max': float(np.max(selected)),
                        'iqr': float(q75 - q25),
                        'zero_ratio': float(np.mean(selected == 0.0)),
                    })
    return rows


def _benefit_rows(fold, payload):
    rows = []
    for role in ('train', 'feedback'):
        y = payload[f'{role}_y']
        p_ce_all, p_kd_all = payload[f'{role}_prob_ce'], payload[f'{role}_prob_kd']
        teacher = payload[f'teacher_{role}_probabilities']
        c = c_components(p_ce_all[1:], p_kd_all[1:], y)['c_1_30']['c']
        late_gain = late_signed_probability_gain(p_ce_all[1:], p_kd_all[1:], y)
        ce_pred, kd_pred, t_pred = p_ce_all[-1].argmax(1), p_kd_all[-1].argmax(1), teacher.argmax(1)
        ce_correct, kd_correct, t_correct = ce_pred == y, kd_pred == y, t_pred == y
        for label in np.unique(y):
            class_mask = y == label
            groups = strict_median_groups(c[class_mask])
            masks = {'low': groups['low'], 'high': groups['high'], 'ties': groups['ties'],
                     'all': np.ones(int(class_mask.sum()), dtype=bool)}
            class_indices = np.flatnonzero(class_mask)
            for group_name, local_mask in masks.items():
                ids = class_indices[local_mask]
                if not len(ids):
                    continue
                correction = (~ce_correct[ids]) & kd_correct[ids]
                harm = ce_correct[ids] & (~kd_correct[ids])
                rows.append({
                    'target_fold': int(fold) + 1, 'data_role': role,
                    'class': int(label), 'c_window_for_grouping': 'c_1_30',
                    'group': group_name, 'threshold': groups['threshold'],
                    'n_samples': int(len(ids)), 'mean_c_1_30': float(np.mean(c[ids])),
                    'mean_late_gain': float(np.mean(late_gain[ids])),
                    'correction_count_ce_wrong_kd_right': int(correction.sum()),
                    'correction_rate': float(np.mean(correction)),
                    'harm_count_ce_right_kd_wrong': int(harm.sum()),
                    'harm_rate': float(np.mean(harm)),
                    'net_correction_rate': float(np.mean(correction) - np.mean(harm)),
                    'teacher_correct_ce_wrong_rate': float(np.mean(t_correct[ids] & ~ce_correct[ids])),
                    'teacher_wrong_ce_correct_rate': float(np.mean(~t_correct[ids] & ce_correct[ids])),
                    'teacher_correct_kd_wrong_rate': float(np.mean(t_correct[ids] & ~kd_correct[ids])),
                })
    return rows


def _complementarity_rows(fold, payload):
    rows = []
    for role in ('train', 'feedback'):
        y = payload[f'{role}_y']
        p_ce = payload[f'{role}_prob_ce'][-1]
        p_kd = payload[f'{role}_prob_kd'][-1]
        p_teacher = payload[f'teacher_{role}_probabilities']
        ce_correct = p_ce.argmax(1) == y
        kd_correct = p_kd.argmax(1) == y
        teacher_correct = p_teacher.argmax(1) == y
        c = c_components(payload[f'{role}_prob_ce'][1:],
                         payload[f'{role}_prob_kd'][1:], y)['c_1_30']['c']
        for teacher_ok in (True, False):
            for ce_ok in (True, False):
                mask = (teacher_correct == teacher_ok) & (ce_correct == ce_ok)
                if not mask.any():
                    continue
                rows.append({
                    'target_fold': int(fold) + 1, 'data_role': role,
                    'teacher_correct': teacher_ok, 'ce_correct': ce_ok,
                    'n_samples': int(mask.sum()),
                    'teacher_correct_ce_wrong': bool(teacher_ok and not ce_ok),
                    'mean_c_1_30': float(np.mean(c[mask])),
                    'median_c_1_30': float(np.median(c[mask])),
                    'kd_correct_rate': float(np.mean(kd_correct[mask])),
                    'ce_wrong_to_kd_right_rate': float(np.mean(~ce_correct[mask] & kd_correct[mask])),
                    'ce_right_to_kd_wrong_rate': float(np.mean(ce_correct[mask] & ~kd_correct[mask])),
                })
    return rows


def _fold_diagnostic_rows(fold, payload):
    rows = []
    for role in ('train', 'feedback'):
        y = payload[f'{role}_y']
        p_ce, p_kd = payload[f'{role}_prob_ce'][1:], payload[f'{role}_prob_kd'][1:]
        components = c_components(p_ce, p_kd, y)
        gain = late_signed_probability_gain(p_ce, p_kd, y)
        row = {'target_fold': int(fold) + 1, 'data_role': role,
               'subject_id': int(payload[f'{role}_subjects'][0]),
               'n_samples': int(len(y)), 'late_gain_mean': float(np.mean(gain)),
               'late_gain_std': float(np.std(gain))}
        for key in ('c_1_30', 'D_1_30', 'B_1_30', 'c_1_100', 'D_1_100', 'B_1_100',
                    'c_11_100', 'D_11_100', 'B_11_100'):
            if key.startswith('c_'):
                values = components[key]['c']
            elif key.startswith('D_'):
                values = components[f'c_{key[2:]}']['D']
            else:
                values = components[f'c_{key[2:]}']['B']
            rho, pvalue = spearman_or_none(values, gain)
            row[f'spearman_{key}_late_gain'] = rho
            row[f'pvalue_{key}_late_gain'] = pvalue
        class_rhos = []
        for label in np.unique(y):
            mask = y == label
            rho, _ = spearman_or_none(components['c_1_30']['c'][mask], gain[mask])
            if rho is not None:
                class_rhos.append(rho)
        row['macro_class_spearman_c_1_30_late_gain'] = \
            float(np.mean(class_rhos)) if class_rhos else None
        rows.append(row)
    return rows


def _plot_diagnostics(out_root, c_rows, payloads):
    feedback = [row for row in c_rows if row['data_role'] == 'feedback']
    values = np.asarray([row['c_1_30'] for row in feedback], dtype=np.float64)
    labels = np.asarray([row['label'] for row in feedback], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(np.unique(labels)):
        chosen = values[labels == label]
        ax.hist(chosen, bins=35, alpha=.55, density=True, label=f'class {label}')
    ax.set(title='Feedback c score distribution (epochs 1–30)',
           xlabel='c_1_30', ylabel='density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / 'c_distribution_feedback.png', dpi=160)
    plt.close(fig)

    curves = {}
    for group in ('low', 'high'):
        epoch_values = []
        for fold, payload in payloads.items():
            y = payload['feedback_y']
            c = c_components(payload['feedback_prob_ce'][1:],
                             payload['feedback_prob_kd'][1:], y)['c_1_30']['c']
            for label in np.unique(y):
                mask = y == label
                groups = strict_median_groups(c[mask])
                selected = np.flatnonzero(mask)[groups[group]]
                if len(selected):
                    selected_y = y[selected][None, :, None]
                    correct_ce = np.take_along_axis(
                        payload['feedback_prob_ce'][:, selected, :],
                        selected_y, axis=2).squeeze(axis=2)
                    correct_kd = np.take_along_axis(
                        payload['feedback_prob_kd'][:, selected, :],
                        selected_y, axis=2).squeeze(axis=2)
                    epoch_values.append((correct_ce.mean(axis=1), correct_kd.mean(axis=1)))
        if epoch_values:
            curves[group] = (np.mean([pair[0] for pair in epoch_values], axis=0),
                             np.mean([pair[1] for pair in epoch_values], axis=0))
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(0, 101)
    for group, color in (('low', '#377eb8'), ('high', '#e41a1c')):
        if group in curves:
            ax.plot(epochs, curves[group][0], color=color, linestyle='--',
                    label=f'{group} c: CE')
            ax.plot(epochs, curves[group][1], color=color,
                    label=f'{group} c: KD')
    ax.axvline(10, color='black', linewidth=1, alpha=.5)
    ax.set(title='Feedback true-class probability by early-c group',
           xlabel='epoch (epoch 0 shown separately)', ylabel='true-class probability',
           xlim=(0, 100), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / 'c_trajectory_feedback.png', dpi=160)
    plt.close(fig)


def analyze_source(out_root, folds):
    all_c, all_summary, all_benefit, all_diag, all_comp = [], [], [], [], []
    payloads = {}
    for fold, paths, split in folds:
        payload = _aggregate_dynamics(paths, split)
        payloads[fold + 1] = payload
        all_c.extend(_c_score_rows(fold, payload))
        all_summary.extend(_summary_stat_rows(fold, payload))
        all_benefit.extend(_benefit_rows(fold, payload))
        all_diag.extend(_fold_diagnostic_rows(fold, payload))
        all_comp.extend(_complementarity_rows(fold, payload))
    protocol.atomic_csv(out_root / 'c_scores.csv', all_c)
    protocol.atomic_csv(out_root / 'c_distribution_summary.csv', all_summary)
    protocol.atomic_csv(out_root / 'benefit_by_c_bin.csv', all_benefit)
    protocol.atomic_csv(out_root / 'diagnostics_by_fold.csv', all_diag)
    protocol.atomic_csv(out_root / 'teacher_complementarity.csv', all_comp)

    source_metrics = []
    for fold, payload in payloads.items():
        for role in ('train', 'feedback'):
            sid = (int(payload[f'{role}_subjects'][0]) if role == 'feedback'
                   else 'all_7_train_sources')
            source_metrics.append(_metric_row(
                payload[f'{role}_y'], payload[f'teacher_{role}_probabilities'],
                fold=fold - 1, role=role, condition='MIRepNet_TEACHER', subject_id=sid))
            for condition, pkey in (('BASE_CE', f'{role}_prob_ce'),
                                    ('DELAYED_KD_ALL', f'{role}_prob_kd')):
                source_metrics.append(_metric_row(
                    payload[f'{role}_y'], payload[pkey][-1], fold=fold - 1,
                    role=role, condition=condition, subject_id=sid))
    protocol.atomic_csv(out_root / 'source_metrics.csv', source_metrics)
    _plot_diagnostics(out_root, all_c, payloads)
    return {'c_rows': all_c, 'diagnostics': all_diag,
            'benefit_rows': all_benefit, 'source_metrics': source_metrics,
            'payloads': payloads}


def _fold_info(args, base_resolved, spec, snapshot, source, fold,
               out_root, cache_root, weight_root):
    feedback, split, split_manifest, old_paths, train_data, teacher_cache, \
        schedule, schedule_sha = trainer._prepare_fold(
            argparse.Namespace(smoke_only=False, smoke_id=None),
            base_resolved, spec, snapshot, source, args.device, fold)
    teacher_bundle = protocol.validate_teacher(
        old_paths['weights'], split['train']['uids'], split['train']['y'])
    if teacher_bundle is None:
        raise FileNotFoundError(f'no seven-source Teacher artifacts for {old_paths["key"]}')
    paths = _make_fold_paths(old_paths, out_root, cache_root, weight_root)
    train_indices = sorted(np.unique(split['train']['subjects']).astype(int).tolist())
    ids = {
        'target_subject_id': int(np.unique(split['test']['subjects'])[0]) + 1,
        'feedback_subject_id': int(np.unique(split['feedback']['subjects'])[0]) + 1,
        'train_subject_indices_zero_based': train_indices,
        'train_subject_ids': [index + 1 for index in train_indices],
    }
    if (ids['target_subject_id'] != fold + 1
            or ids['feedback_subject_id'] != feedback + 1
            or len(ids['train_subject_ids']) != 7
            or ids['target_subject_id'] in ids['train_subject_ids']
            or ids['feedback_subject_id'] in ids['train_subject_ids']):
        raise RuntimeError(f'7+1+1 fold membership is invalid: {ids}')
    if teacher_bundle[1].get('train_subject_ids') != train_indices:
        raise RuntimeError('Teacher provenance train subject indices do not match D_train')
    if (teacher_bundle[1].get('feedback_subject_id') != ids['feedback_subject_id']
            or teacher_bundle[1].get('target_subject_id') != ids['target_subject_id']):
        raise RuntimeError('Teacher provenance has a mismatched feedback or target subject')
    c_split_manifest = dict(split_manifest)
    c_split_manifest['subject_id_mapping'] = {
        'source_subject_array_is_zero_based': True,
        'target_subject_index_zero_based': int(fold),
        'target_subject_id_one_based': ids['target_subject_id'],
        'feedback_subject_index_zero_based': int(feedback),
        'feedback_subject_id_one_based': ids['feedback_subject_id'],
        'train_subject_indices_zero_based': train_indices,
        'train_subject_ids_one_based': ids['train_subject_ids'],
        'legacy_train_subject_ids_field_in_split_manifest_is_zero_based': True,
    }
    protocol.atomic_json(paths['results'] / 'split_manifest.json', c_split_manifest)
    preflight = {
        'protocol_id': PROTOCOL_ID, 'target_fold': fold + 1,
        'feedback_subject_id': ids['feedback_subject_id'],
        'train_subject_ids': ids['train_subject_ids'],
        'train_subject_indices_zero_based': train_indices,
        'split_sha256': split_manifest['split_sha256'],
        'split_uid_sha256': split_manifest['split_uid_sha256'],
        'teacher_provenance': teacher_bundle[1],
        'teacher_checkpoint': str(old_paths['weights'] / 'teacher_final.pt'),
        'teacher_checkpoint_sha256': protocol.sha256_file(
            old_paths['weights'] / 'teacher_final.pt'),
        'existing_controls_reused': False,
        'existing_controls_note': 'final-only predictions lack per-epoch class probabilities; both tracks are rerun',
        'student_preprocessing': 'reused UID/label-validated cached IFNet inputs',
        'batch_schedule_sha256': schedule_sha,
    }
    existing_audit = {}
    for condition in CONDITIONS:
        path = old_paths['results'] / condition / 'manifest.json'
        if path.is_file():
            manifest = json.loads(path.read_text())
            existing_audit[condition] = {
                'status': manifest.get('status'),
                'final_epoch': manifest.get('final_epoch'),
                'warmup_state_sha256': manifest.get('warmup_state_sha256'),
                'schedule_sha256': manifest.get('schedule_sha256'),
                'train_uid_sha256': manifest.get('train_uid_sha256'),
            }
    if set(existing_audit) == set(CONDITIONS):
        a, b = (existing_audit[name] for name in CONDITIONS)
        preflight['historical_control_pair_match'] = all(
            a.get(key) == b.get(key) for key in
            ('warmup_state_sha256', 'schedule_sha256', 'train_uid_sha256'))
        preflight['historical_controls'] = existing_audit
    protocol.atomic_json(paths['results'] / 'preflight.json', preflight)
    return {
        'fold': fold, 'feedback': feedback, 'split': split,
        'split_manifest': split_manifest, 'old_paths': old_paths,
        'paths': paths, 'train_data': train_data,
        'teacher_cache': teacher_cache, 'teacher_bundle': teacher_bundle,
        'schedule': schedule, 'schedule_sha': schedule_sha,
    }


def _train_fold(info, base_resolved, device, *, total_epochs, resume):
    fold, split, paths, train_data = (info['fold'], info['split'],
                                      info['paths'], info['train_data'])
    teacher_path = _teacher_probabilities(
        info['old_paths'], split, info['teacher_cache'], info['teacher_bundle'],
        base_resolved, device, paths['results'])
    print(f'[teacher-ready] target=S{fold+1:02d} '
          f'feedback=S{info["feedback"]+1:02d} path={teacher_path}', flush=True)
    lock = trainer._lock_fold(paths['results'].parent)
    try:
        warmup_capture = lambda epoch, adapter, model: _capture_student_epoch(
            paths, epoch, adapter, model, train_data, split, device)
        warmup = trainer.create_warmup(
            paths, split, train_data['arrays']['train_x'], split['train']['uids'],
            base_resolved, device, resume=resume, epoch_callback=warmup_capture)
        old_warmup_path = info['old_paths']['weights'] / 'warmup_epoch10.pt'
        warmup_audit = {'historical_warmup_checkpoint': str(old_warmup_path),
                        'current_warmup_state_sha256': warmup['warmup_state_sha256'],
                        'historical_warmup_present': old_warmup_path.is_file()}
        if old_warmup_path.is_file():
            historical = torch.load(resolve_local_file(old_warmup_path), map_location='cpu')
            warmup_audit.update({
                'historical_warmup_state_sha256': historical['warmup_state_sha256'],
                'model_state_equal': historical['warmup_state_sha256']
                                     == warmup['warmup_state_sha256'],
                'optimizer_state_equal': _nested_equal(historical['optimizer'],
                                                       warmup['optimizer']),
                'scheduler_state_equal': _nested_equal(historical['scheduler'],
                                                       warmup['scheduler']),
                'rng_state_equal': _nested_equal(historical['rng'], warmup['rng']),
                'schedule_sha256_equal': historical['schedule_sha256']
                                         == warmup['schedule_sha256'],
                'train_uid_sha256_equal': historical['train_uid_sha256']
                                          == warmup['train_uid_sha256'],
            })
            exact = all(warmup_audit[key] for key in
                        ('model_state_equal', 'optimizer_state_equal',
                         'scheduler_state_equal', 'rng_state_equal',
                         'schedule_sha256_equal', 'train_uid_sha256_equal'))
            warmup_audit['complete_state_equal'] = exact
            if device.type == 'cpu' and not exact:
                raise RuntimeError('regenerated CPU warm-up differs from the recorded '
                                   f'complete state: {warmup_audit}')
        protocol.atomic_json(paths['results'] / 'warmup_replay_audit.json', warmup_audit)
        warmup['train_schedule'] = protocol.epoch_schedule(
            len(split['train']['y']), trainer.BATCH_SIZE, trainer.TOTAL_EPOCHS, 666)
        warmup['fold'] = fold
        reference = trainer._export_reference_features(
            paths, warmup, train_data['arrays']['train_x'], split['train']['y'],
            split['train']['uids'], device)
        if total_epochs <= trainer.WARMUP_EPOCHS:
            raise ValueError('total epochs must exceed the 10-epoch warm-up')
        for condition in CONDITIONS:
            cell = paths['results'] / condition
            cell.mkdir(parents=True, exist_ok=True)
            resume_path = cell / 'checkpoint_resume.pt'
            if resume_path.is_file() and not resume:
                raise RuntimeError(f'partial {condition} run exists; pass --resume: {resume_path}')
            if resume_path.is_file():
                saved = torch.load(resolve_local_file(resume_path), map_location='cpu')
                completed = int(saved['completed_epoch'])
                missing = [epoch for epoch in range(11, completed + 1)
                           if not (cell / 'prediction_dynamics'
                                   / f'epoch_{epoch:03d}.npz').is_file()]
                if missing:
                    raise RuntimeError(f'{condition} resume state lacks prediction files '
                                       f'for completed epochs {missing}')
            # Existing completed runs are returned by train_condition before it
            # tries to load a resume state. Otherwise resume only an actual state.
            capture = lambda epoch, adapter, model, cond=condition: _capture_student_epoch(
                paths, epoch, adapter, model, train_data, split, device, condition=cond)
            trainer.train_condition(
                condition, paths, fold, info['feedback'], split, train_data,
                info['teacher_cache'], reference, warmup, base_resolved, device,
                resume=resume_path.is_file(),
                continuation_epochs=total_epochs - trainer.WARMUP_EPOCHS,
                epoch_callback=capture)
            manifest_path = cell / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('final_epoch') != total_epochs or manifest.get('status') != 'complete':
                raise RuntimeError(f'{condition} did not finish at epoch {total_epochs}')
        print(f'[fold-complete] target=S{fold+1:02d} total_epochs={total_epochs}', flush=True)
    finally:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _test_paths(base_resolved, spec, snapshot, source, fold, out_root,
                cache_root, weight_root):
    feedback, split = protocol.split_fold(*source[:6], fold)
    split_manifest = protocol.split_manifest(fold, feedback, split,
                                             source[8], base_resolved)
    old_paths = protocol.fold_paths(fold, feedback, split_manifest['split_sha256'])
    paths = _make_fold_paths(old_paths, out_root, cache_root, weight_root)
    return feedback, split, paths


def evaluate_targets(base_resolved, spec, snapshot, source, folds,
                     out_root, cache_root, weight_root, device):
    if sorted(folds) != list(range(9)):
        raise ValueError('target evaluation requires all nine completed folds')
    rows = []
    for fold in folds:
        feedback, split, paths = _test_paths(base_resolved, spec, snapshot, source,
                                             fold, out_root, cache_root, weight_root)
        for condition in CONDITIONS:
            manifest_path = paths['results'] / condition / 'manifest.json'
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') != 'complete' or manifest.get('final_epoch') != 100:
                raise RuntimeError(f'target evaluation is restricted to final epoch 100: {manifest_path}')
        adapter = trainer._student_adapter(device, base_resolved)
        xb = adapter.preprocess(split['test']['x']).cpu().numpy().astype(np.float32)
        fold_predictions = {}
        for condition in CONDITIONS:
            model = trainer._load_final_model(paths, condition, base_resolved, device)
            probabilities = _student_probabilities(adapter, model, xb, device)
            fold_predictions[condition] = probabilities
            rows.append({
                'dataset': 'BNCI2014004', 'session': 'session_3',
                'target_subject': fold + 1, 'feedback_subject': feedback + 1,
                'seed': 666, 'condition': condition, 'epoch': 100,
                'n_test': len(split['test']['y']),
                **{key: value for key, value in _metric_row(
                    split['test']['y'], probabilities, fold=fold,
                    role='test', condition=condition, subject_id=fold + 1).items()
                   if key not in ('target_fold', 'data_role', 'subject_id', 'condition', 'n_samples')},
                'test_uid_sha256': protocol.hash_uids(split['test']['uids']),
            })
            del model
        protocol.atomic_npz(
            paths['results'] / 'target_predictions.npz',
            target_uids=np.asarray(split['test']['uids'], dtype=np.int64),
            target_y=np.asarray(split['test']['y'], dtype=np.int64),
            ce_probabilities=fold_predictions['BASE_CE'],
            kd_probabilities=fold_predictions['DELAYED_KD_ALL'],
            target_subject_id=np.asarray(fold + 1, dtype=np.int16),
        )
        print(f'[target-evaluated] S{fold+1:02d} '
              f'CE_BA={rows[-2]["balanced_accuracy"]:.4f} '
              f'KD_BA={rows[-1]["balanced_accuracy"]:.4f}', flush=True)
    protocol.atomic_csv(out_root / 'results.csv', rows)
    return rows


def _bootstrap_mean_delta(deltas, draws=10000, seed=666):
    deltas = np.asarray(deltas, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(deltas), size=(draws, len(deltas)))
    return np.quantile(deltas[indices].mean(axis=1), [0.025, 0.975]).tolist()


def build_report(out_root, analysis, target_rows=None, resolved_c=None):
    diag = pd.DataFrame(analysis['diagnostics'])
    feedback_diag = diag[diag['data_role'] == 'feedback']
    feedback_scores = [row for row in analysis['c_rows'] if row['data_role'] == 'feedback']
    c_values = np.asarray([row['c_1_30'] for row in feedback_scores], dtype=np.float64)
    d_values = np.asarray([row['D_1_30'] for row in feedback_scores], dtype=np.float64)
    b_values = np.asarray([row['B_1_30'] for row in feedback_scores], dtype=np.float64)
    rho_by_fold = feedback_diag['spearman_c_1_30_late_gain'].dropna().astype(float)
    mean_rho = float(rho_by_fold.mean()) if len(rho_by_fold) else float('nan')
    feedback_benefit = pd.DataFrame([
        row for row in analysis['benefit_rows']
        if row['data_role'] == 'feedback' and row['group'] in ('low', 'high')])
    lines = [
        '# EEG c 指标 LOSO 诊断', '',
        f'- 正式轨迹：18 条（9 折 × CE/KD 两条件）；seed 666；BNCI2014004 session_3。',
        '- 每折按 7 个源训练被试、1 个完整源反馈被试、1 个最终目标被试划分；Teacher 只在七个训练源被试上微调并冻结。',
        '- 两个 IFNet 从同一 epoch10 完整 warm-up 状态分叉；CE 与 KD 使用相同训练 UID/minibatch schedule、优化器、scheduler 和每步 Student 随机流。',
        f'- 固定 KD 参数：λ={resolved_c["resolved_kd_settings"]["lambda_kd"]}，'
        f'τ={resolved_c["resolved_kd_settings"]["temperature"]}，'
        f'batch size={resolved_c["resolved_kd_settings"]["batch_size"]}；'
        '其他 Student 参数取同一历史 7+1 配置。',
        '- 旧 CE/KD 对照只有最终概率，不能生成 c 轨迹；本次重新训练 18 条 Student 轨迹。每折七源 Teacher 通过 UID 和被试来源核验后复用，不重新微调。',
        '- c 只在训练后诊断，不参与 CE/KD 损失；epoch0 单独保存，c 窗口只使用 epoch1–100。',
        '- 目标被试只在所有 Student 完成 epoch100 后评测一次；不用于 c、阈值或配置选择。',
        '', '## c 分布与预测性', '',
        f'- 源反馈集共 {len(feedback_scores)} 条 fold/UID 记录；反馈 c_1_30 中位数 {np.median(c_values):.6g}，IQR {np.quantile(c_values,.75)-np.quantile(c_values,.25):.6g}，标准差 {np.std(c_values):.6g}，零值比例 {np.mean(c_values == 0):.3%}。',
        f'- 同窗口 D 中位数 {np.median(d_values):.6g}，B 中位数 {np.median(b_values):.6g}。',
        f'- 每折反馈集 Spearman(c_1_30, G_81_100) 的可计算折数 {len(rho_by_fold)}/9，跨折均值相关 {mean_rho:.4f}。每折值见 `diagnostics_by_fold.csv`。',
        '- 主要预测性检查固定为反馈集 c_1_30 与 epoch81–100 真实类别概率有符号增益的 Spearman 相关。c、D、B 的比较同时列出；相关性是样本级探索诊断，折内样本不是独立实验重复。',
        '- 高低 c 分组在每折×真实类别内部以严格中位数两侧定义；等于中位数的样本列为 ties，不强行分组。',
        '', '## KD 与 CE 的源反馈关联', '',
    ]
    if len(feedback_benefit):
        grouped = feedback_benefit.groupby('group')[[
            'n_samples', 'mean_late_gain', 'correction_rate', 'harm_rate', 'net_correction_rate']].mean()
        for name in ('low', 'high'):
            if name in grouped.index:
                row = grouped.loc[name]
                lines.append(f'- {name} c 组：平均 late probability gain {row.mean_late_gain:.5f}；'
                             f'CE 错→KD 对比例 {row.correction_rate:.3%}；'
                             f'CE 对→KD 错比例 {row.harm_rate:.3%}；'
                             f'平均净纠错率 {row.net_correction_rate:.3%}。')
    lines += [
        '', '## 最终目标被试评测', '',
    ]
    if target_rows:
        frame = pd.DataFrame(target_rows)
        ce = frame[frame.condition == 'BASE_CE'].sort_values('target_subject')
        kd = frame[frame.condition == 'DELAYED_KD_ALL'].sort_values('target_subject')
        delta = (kd['balanced_accuracy'].to_numpy()
                 - ce['balanced_accuracy'].to_numpy()) * 100.0
        ci = _bootstrap_mean_delta(delta)
        wins = int(np.sum(delta > 0)); ties = int(np.sum(delta == 0)); losses = int(np.sum(delta < 0))
        lines += [
            '| target subject | CE BA | KD BA | KD − CE (pp) |',
            '|---:|---:|---:|---:|',
        ]
        for (_, a), (_, b), difference in zip(ce.iterrows(), kd.iterrows(), delta):
            lines.append(f'| {int(a.target_subject)} | {a.balanced_accuracy:.4f} | '
                         f'{b.balanced_accuracy:.4f} | {difference:+.2f} |')
        lines += [
            '', f'- BA 配对差均值 {np.mean(delta):+.2f} pp；Win/Tie/Loss {wins}/{ties}/{losses}；'
            f'按被试 bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp。',
            f'- Accuracy 均值：CE {ce.accuracy.mean():.4f}，KD {kd.accuracy.mean():.4f}；'
            f'Kappa 均值：CE {ce.kappa.mean():.4f}，KD {kd.kappa.mean():.4f}。',
        ]
    else:
        lines.append('目标评测尚未执行。')
    lines += [
        '', '## 产物与解释边界', '',
        '- `prediction_dynamics.npz`：每折逐 UID、epoch0–100 的 CE/KD 完整类别概率；epoch0–10 共享 warm-up 概率相同。Teacher train/feedback 概率也包含在同一文件。',
        '- `c_scores.csv`、`c_distribution_summary.csv`、`benefit_by_c_bin.csv`、`diagnostics_by_fold.csv` 和 `teacher_complementarity.csv` 保存逐样本与逐折诊断。',
        '- 图像 `c_distribution_feedback.png` 与 `c_trajectory_feedback.png` 展示反馈被试 c 分布及早期 c 高低组的概率轨迹。',
        '- 9 折共享部分源数据、单 seed 仅为探索性诊断；被试 bootstrap 只描述有限的不确定性，不代表独立跨数据集或跨 seed 证据。',
        '- c 有离散度只说明它区分了当前两条训练轨迹；与反馈收益相关才提示方向性，但仍不能证明按 c 加权训练会提升目标泛化。需要另做加权训练对照。',
        '',
    ]
    report_path = out_root / 'report.md'
    require_external_output(report_path).parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


def _source_object(base_resolved):
    spec = yaml.safe_load(resolve_local_file(canonical.SPEC_PATH).read_text())
    snapshot = json.loads(resolve_local_file(canonical.SNAPSHOT_PATH).read_text())
    source = protocol.load_data(spec, snapshot, base_resolved['source_hashes'])
    return spec, snapshot, source


def _load_compatible_base_config(config_path, device):
    """Load the exact recorded 7+1 protocol after checking current source inputs.

    The repository's current canonical model-config helpers add default-valued
    fields absent from the original resolved pilot. That changes its digest and
    fold directory despite identical actual required settings. We validate the
    stored digest and all current source/model settings, then retain the original
    resolved object so its per-fold Teacher and UID manifests resolve exactly.
    """
    current, spec, snapshot = protocol.load_resolved_config(config_path, device)
    archived_path = require_external_output(PILOT_RESULT_ROOT / 'resolved.yaml')
    if not archived_path.is_file():
        raise FileNotFoundError(f'recorded seven-source resolved config is missing: {archived_path}')
    archived = yaml.safe_load(resolve_local_file(archived_path).read_text())
    digest_input = dict(archived)
    recorded_hash = digest_input.pop('resolved_config_sha256', None)
    digest_input.pop('runtime_device', None)
    if not recorded_hash or protocol.hash_json(digest_input) != recorded_hash:
        raise RuntimeError('stored seven-source resolved config hash is invalid')
    for key in ('dataset', 'seed', 'epochs', 'warmup_epochs', 'split',
                'canonical_spec_sha256', 'canonical_source_snapshot_sha256',
                'trial_manifest_sha256', 'source_hashes', 'pretrained_checkpoint'):
        if current.get(key) != archived.get(key):
            raise RuntimeError(f'current source/config field differs from recorded pilot: {key}')
    for section in ('student_config', 'teacher_config'):
        for key, value in archived[section].items():
            if current[section].get(key) != value:
                raise RuntimeError(f'{section}.{key} changed since the recorded pilot')
    for key in ('student', 'teacher', 'loss', 'execution', 'conditions',
                'optimizer_defaults'):
        if current.get(key) != archived.get(key):
            # Some resolved dictionaries contain tuples in memory and lists in
            # YAML; JSON normalizes both to the same sequence representation.
            if json.dumps(current.get(key), sort_keys=True, default=str) != \
                    json.dumps(archived.get(key), sort_keys=True, default=str):
                raise RuntimeError(f'recorded protocol setting differs: {key}')
    return archived, spec, snapshot, current


def main():
    args = parse_args()
    raw = load_experiment_config(args.config)
    if args.smoke:
        if args.epochs != 12:
            raise ValueError('the fixed smoke runs 12 total epochs (10 warm-up + 2 continuation)')
        if args.stage not in ('preflight', 'train', 'all'):
            raise ValueError('smoke mode supports only preflight/train and never evaluates targets')
        folds_requested = [0] if args.folds is None else list(args.folds)
        if len(folds_requested) != 1:
            raise ValueError('smoke mode accepts exactly one fold')
        stage = 'train' if args.stage == 'all' else args.stage
    else:
        if args.epochs != 100:
            raise ValueError('--epochs below 100 requires --smoke')
        folds_requested = list(range(9)) if args.folds is None else list(args.folds)
        stage = args.stage
    if len(set(folds_requested)) != len(folds_requested) or any(
            fold < 0 or fold > 8 for fold in folds_requested):
        raise ValueError('--folds must contain unique indices in [0,8]')
    if stage in ('evaluate', 'all', 'analyze') and not args.smoke \
            and sorted(folds_requested) != list(range(9)):
        raise ValueError(f'{stage} requires all nine folds')
    args.device = device_for(args.gpu)
    out_root, cache_root, weight_root = _external_roots(raw, args)
    out_root.mkdir(parents=True, exist_ok=True)

    base_path = raw['base_training_config']
    base_resolved, _base_spec, _base_snapshot, current_base = \
        _load_compatible_base_config(base_path, args.device)
    c_identity = {
        'config': raw, 'base_resolved_config_sha256': base_resolved['resolved_config_sha256'],
        'current_resolved_config_sha256': current_base['resolved_config_sha256'],
        'resolved_kd_settings': {
            'lambda_kd': base_resolved['loss']['lam_kd'],
            'temperature': base_resolved['loss']['temperature'],
            'batch_size': base_resolved['student_config']['batch_size'],
            'optimizer': base_resolved['student_config'],
            'optimizer_defaults': base_resolved['optimizer_defaults'],
        },
    }
    c_config_sha = protocol.hash_json(c_identity)
    resolved_c = dict(c_identity, c_metric_resolved_config_sha256=c_config_sha,
                      runtime_device=str(args.device))
    resolved_path = out_root / 'resolved.yaml'
    resolved_yaml = yaml.safe_dump(resolved_c, sort_keys=True, allow_unicode=True)
    if resolved_path.is_file():
        previous = yaml.safe_load(resolved_path.read_text())
        if previous.get('c_metric_resolved_config_sha256') != c_config_sha:
            raise RuntimeError(f'result root contains another resolved config: {resolved_path}')
    else:
        resolved_path.write_text(resolved_yaml, encoding='utf-8')
    protocol.atomic_json(out_root / 'execution.json', {
        'protocol_id': PROTOCOL_ID, 'seed': 666, 'stage': stage,
        'requested_folds': [int(fold + 1) for fold in folds_requested],
        'device': str(args.device), 'smoke': bool(args.smoke),
        'epochs': int(args.epochs), 'c_metric_config_sha256': c_config_sha,
        'base_resolved_config_sha256': base_resolved['resolved_config_sha256'],
        'teacher_policy': 'reuse only per-fold seven-source frozen MIRepNet checkpoints',
        'target_policy': 'read only after all epoch-100 training tracks are complete',
        'git': trainer.git_snapshot(),
        'environment': trainer.environment_snapshot(args.device),
    })
    spec, snapshot, source = _source_object(base_resolved)
    fold_infos = []
    for fold in folds_requested:
        info = _fold_info(args, base_resolved, spec, snapshot, source, fold,
                          out_root, cache_root, weight_root)
        fold_infos.append(info)
        print(f'[preflight-ok] target=S{fold+1:02d} '
              f'feedback=S{info["feedback"]+1:02d} '
              f'train={len(info["split"]["train"]["y"])} '
              f'feedback_n={len(info["split"]["feedback"]["y"])} '
              f'test={len(info["split"]["test"]["y"])}', flush=True)
    if stage == 'preflight':
        return

    if stage in ('train', 'all'):
        for info in fold_infos:
            _train_fold(info, base_resolved, args.device,
                        total_epochs=args.epochs, resume=args.resume)
        if args.smoke:
            print('[smoke-complete] 12 epochs; target inference was not run', flush=True)
            return

    if stage in ('analyze', 'all'):
        analysis = analyze_source(out_root, fold_infos)
    elif stage == 'evaluate':
        analysis = None
    else:
        analysis = None

    target_rows = None
    if stage in ('evaluate', 'all'):
        target_rows = evaluate_targets(base_resolved, spec, snapshot, source,
                                       list(range(9)), out_root, cache_root,
                                       weight_root, args.device)
    if stage in ('analyze', 'all'):
        if analysis is None:
            analysis = analyze_source(out_root, fold_infos)
        report = build_report(out_root, analysis, target_rows, resolved_c)
        print(f'[report-complete] {report}', flush=True)


if __name__ == '__main__':
    main()
