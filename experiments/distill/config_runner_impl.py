"""Config-driven distillation backend used by ``run_distill.py``."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from experiments import config_loader
from experiments.distill import run_distill as legacy


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/distill/run_distill.py",
        description="Run a distillation experiment from configs/experiments/*.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--resume", action="store_true")
    write_mode.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _device(gpu):
    import torch
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def _namespace(spec, artifact_root):
    params = dict(spec["params"])
    training = dict(spec["training"])
    return SimpleNamespace(
        dataset=spec["dataset"], protocol=spec["protocol"],
        teacher=spec["teacher"], student=spec["student"],
        teacher_artifact=None, keys=spec.get("subjects"), seeds=spec.get("seeds"),
        train_percentage=spec.get("train_percentage"), val_split=spec.get("val_split"),
        methods=spec["methods"],
        lam_kd=float(params.get("lam_kd", 0.5)),
        lam_mi=float(params.get("lam_mi", 0.1)),
        lam_mmd=float(params.get("lam_mmd", 0.5)),
        temperature=float(params.get("temperature", 2.0)),
        mmd_sigmas=list(params.get("mmd_sigmas", [0.5, 1.0, 2.0, 4.0])),
        mmd_normalize=bool(params.get("mmd_normalize", True)),
        mmd_class_conditional=bool(params.get("mmd_class_conditional", False)),
        epochs=int(training["epochs"]), lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        artifact_root=str(artifact_root),
    )


def _row_key(row):
    values = (
        row.get("run_id", ""), row.get("dataset", ""),
        row.get("teacher", ""), row.get("student", ""),
        row.get("key", row.get("subject", "")), row.get("seed", ""),
        row.get("method", ""),
    )
    return tuple(str(value) for value in values)


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path, value):
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True,
                                         ensure_ascii=False, default=str) + '\n')


def _atomic_write_csv(path, rows, fieldnames=None):
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


def _git_value(*args):
    try:
        return subprocess.check_output(
            ['git', *args], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f'error: {exc}'


def _config_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _run_key(dataset, subject, seed, method):
    return f'{dataset}__S{int(subject) + 1}__seed{int(seed)}__{method}'


def _artifact_paths(run_root):
    return {
        'history': Path(run_root) / 'training_history',
        'checkpoint': Path(run_root) / 'checkpoints',
        'prediction': Path(run_root) / 'predictions',
    }


def _artifact_file_map(run_root, row):
    key = _run_key(row['dataset'], row['key'], row['seed'], row['method'])
    roots = _artifact_paths(run_root)
    return {
        'history': roots['history'] / f'{key}.json',
        'checkpoint': roots['checkpoint'] / f'{key}.pt',
        'prediction': roots['prediction'] / f'{key}.npz',
    }


def _row_complete(row, run_root):
    if row.get('failure_status') != 'complete':
        return False
    for value in (row.get('acc'), row.get('test_balanced_accuracy'), row.get('kappa')):
        try:
            if not np.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    files = _artifact_file_map(run_root, row)
    if not all(path.is_file() and path.stat().st_size > 0 for path in files.values()):
        return False
    try:
        checkpoint = torch.load(files['checkpoint'], map_location='cpu')
        if not isinstance(checkpoint, dict) or not checkpoint.get('complete', False):
            return False
        history = json.loads(files['history'].read_text())
        if not history or len(history) != int(row.get('student_epochs', 0)):
            return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _write_rows(path, rows):
    priority = [
        "config_name", "run_id", "grid_index", "dataset", "protocol",
        "teacher", "student", "teacher_artifact", "subject", "key", "session", "seed",
        "method", "acc", "kappa", "n_train", "n_test", "num_classes",
        "condition_label", "test_accuracy", "test_balanced_accuracy", "test_kappa",
        "teacher_train_acc_pct", "lam_kd", "lam_mi", "lam_mmd", "temperature",
        "split_uid_hash", "initial_state_hash", "batch_order_hash",
        "train_uid_hash", "test_uid_hash", "teacher_train_uid_alignment",
        "weight_mode", "student_epochs", "student_lr",
        "student_weight_decay", "student_batch_size", "optimizer", "scheduler",
        "final_train_loss", "final_train_accuracy", "final_ce_loss", "final_mi_loss",
        "full_train_mi", "full_train_mi_last", "teacher_student_prediction_agreement",
        "mean_abs_probability_diff", "predicted_class_counts", "collapse_flag",
        "runtime_seconds", "failure_status", "failure_reason",
    ]
    fields = list(priority)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    _atomic_write_csv(path, rows, fields)


def _build_manifest(specs):
    manifest = []
    for spec in specs:
        dcfg = legacy.config.load_dataset_config(spec['dataset'])
        keys = spec.get('subjects')
        if keys is None:
            keys = list(range(int(dcfg['num_subjects'])))
        seeds = spec.get('seeds')
        if seeds is None:
            seeds = list(dcfg['seeds'])
        for key in keys:
            for seed in seeds:
                for method in spec['methods']:
                    manifest.append({
                        'run_key': _run_key(spec['dataset'], key, seed, method),
                        'config_name': spec['config_name'],
                        'run_id': spec['run_id'],
                        'dataset': spec['dataset'],
                        'subject': int(key) + 1,
                        'key': int(key),
                        'seed': int(seed),
                        'protocol': spec['protocol'],
                        'teacher': spec['teacher'],
                        'student': spec['student'],
                        'condition': method,
                        'status': 'pending',
                        'failure_reason': '',
                    })
    return manifest


def _update_manifest(manifest, row, status, reason=''):
    run_key = _run_key(row['dataset'], row['key'], row['seed'], row['method'])
    for item in manifest:
        if item['run_key'] == run_key:
            item['status'] = status
            item['failure_reason'] = str(reason or '')
            return


def _upsert_rows(rows, incoming):
    keyed = {_row_key(row): row for row in rows}
    for row in incoming:
        keyed[_row_key(row)] = row
    return list(keyed.values())


def _write_manifest(path, manifest):
    fields = [
        'run_key', 'config_name', 'run_id', 'dataset', 'subject', 'key', 'seed',
        'protocol', 'teacher', 'student', 'condition', 'status', 'failure_reason',
    ]
    _atomic_write_csv(path, manifest, fields)


def _save_run_artifacts(run_root, row, details, checkpoint_payload):
    files = _artifact_file_map(run_root, row)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    history = details.get('training_history', [])
    _atomic_write_json(files['history'], history)
    checkpoint_tmp = files['checkpoint'].with_name(
        f'.{files["checkpoint"].name}.tmp-{os.getpid()}')
    torch.save(checkpoint_payload, checkpoint_tmp)
    os.replace(checkpoint_tmp, files['checkpoint'])
    train_logits = np.asarray(details['train_logits'])
    test_logits = np.asarray(details['test_logits'])
    npz_tmp = files['prediction'].with_name(
        f'.{files["prediction"].name}.tmp-{os.getpid()}')
    np.savez(
        npz_tmp,
        train_logits=train_logits,
        test_logits=test_logits,
        train_preds=np.asarray(details['train_preds'], dtype=np.int64),
        test_preds=np.asarray(details['test_preds'], dtype=np.int64),
    )
    # np.savez appends .npz when given a path without that suffix.
    generated = Path(f'{npz_tmp}.npz') if not npz_tmp.name.endswith('.npz') else npz_tmp
    os.replace(generated, files['prediction'])


def _load_existing_complete_rows(output, run_root):
    rows = _read_rows(output)
    complete = []
    for row in rows:
        if _row_complete(row, run_root):
            complete.append(row)
    return complete


def _run_spec(spec, device, existing_keys, fail_fast, run_root=None,
              manifest=None, total_runs=None, persist=None):
    dcfg = legacy.config.load_dataset_config(spec["dataset"])
    keys = spec.get("subjects")
    if keys is None:
        keys = list(range(dcfg["num_subjects"]))
    seeds = spec.get("seeds")
    if seeds is None:
        seeds = list(dcfg["seeds"])
    val_split = spec.get("val_split")
    if val_split is None:
        val_split = dcfg["val_split"]
    args = _namespace(spec, config_loader.resolve_path(spec["artifact_root"]))
    expected_policy = legacy._expected_split_policy(spec["protocol"])
    teacher_artifact = legacy._teacher_artifact_name(
        spec["teacher"], spec["protocol"])
    rows, errors = [], []
    for seed in seeds:
        for key in keys:
            prefix = (spec["run_id"], spec["dataset"], spec["teacher"],
                      spec["student"], str(key), str(seed))
            if all(prefix + (method,) in existing_keys for method in spec["methods"]):
                print(f"[resume] skip dataset={spec['dataset']} key={key} seed={seed} run_id={spec['run_id']}", flush=True)
                continue
            try:
                X_tr, y_tr, X_te, y_te, uid_tr, uid_te, subj_tr = legacy._split_cell(
                    spec["dataset"], spec["protocol"], key, seed,
                    val_split, spec.get("train_percentage"))
                tch = legacy._load_teacher(
                    spec["dataset"], teacher_artifact, key, seed, y_tr, uid_tr,
                    expected_policy, args.artifact_root)
                # The MI pilot compares three conditions on the exact same
                # fold initialization.  Capture one deterministic state once
                # and load it into Base/KD_all/CE_MI; no checkpoint is written.
                initial_state_dict = None
                if spec["method"] == "mi":
                    initial_state_dict = legacy._capture_initial_state(
                        args, spec["dataset"], spec["student"], X_tr,
                        int(dcfg["num_classes"]), device, seed)
                for method in spec["methods"]:
                    row_key = prefix + (method,)
                    if row_key in existing_keys:
                        continue
                    started = time.time()
                    details = None
                    try:
                        method_kwargs = {}
                        if initial_state_dict is not None:
                            method_kwargs["initial_state_dict"] = initial_state_dict
                        row, details = legacy._run_method(
                            args, spec["dataset"], spec["protocol"], spec["teacher"],
                            spec["student"], teacher_artifact, key, seed, method,
                            X_tr, y_tr, X_te, y_te, subj_tr, tch,
                            int(dcfg["num_classes"]), device,
                            uid_tr=uid_tr, uid_te=uid_te, return_details=True,
                            **method_kwargs)
                        row.update(config_name=spec["config_name"],
                                   run_id=spec["run_id"], grid_index=spec["grid_index"],
                                   runtime_seconds=round(time.time() - started, 3))
                        if run_root is not None:
                            model = details.pop('model')
                            state_dict = {
                                name: (value.detach().cpu().clone()
                                       if torch.is_tensor(value) else value)
                                for name, value in model.state_dict().items()
                            }
                            checkpoint_payload = {
                                'complete': True,
                                'run_key': _run_key(spec['dataset'], key, seed, method),
                                'dataset': spec['dataset'],
                                'subject': int(key) + 1,
                                'seed': int(seed),
                                'condition': method,
                                'epochs': int(row['student_epochs']),
                                'initial_state_hash': row['initial_state_hash'],
                                'split_uid_hash': row['split_uid_hash'],
                                'state_dict': state_dict,
                            }
                            _save_run_artifacts(
                                run_root, row, details, checkpoint_payload)
                            files = _artifact_file_map(run_root, row)
                            row.update(
                                checkpoint_path=str(files['checkpoint']),
                                history_path=str(files['history']),
                                prediction_path=str(files['prediction']),
                            )
                            del model
                        row['failure_status'] = (
                            'complete' if row.get('failure_status') == 'complete'
                            else row.get('failure_status', 'invalid_metrics'))
                        rows.append(row)
                        existing_keys.add(row_key)
                        if manifest is not None:
                            _update_manifest(manifest, row, row['failure_status'],
                                             row.get('failure_reason', ''))
                        if persist is not None:
                            persist([row])
                        completed = len(existing_keys)
                        progress = (f'{completed}/{total_runs}'
                                    if total_runs is not None else str(completed))
                        print(
                            f"[progress {progress}] [{spec['dataset']}] "
                            f"subject={int(key)+1} seed={seed} condition={method} "
                            f"acc={row['acc']} bal_acc={row['test_balanced_accuracy']} "
                            f"kappa={row['kappa']}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        reason = str(exc)
                        failure = {
                            'config_name': spec['config_name'],
                            'run_id': spec['run_id'],
                            'grid_index': spec['grid_index'],
                            'dataset': spec['dataset'],
                            'subject': int(key) + 1,
                            'key': int(key),
                            'session': legacy._session_default(spec['dataset']),
                            'seed': int(seed),
                            'protocol': spec['protocol'],
                            'teacher': spec['teacher'],
                            'student': spec['student'],
                            'method': method,
                            'condition_label': legacy.METHOD_LABELS.get(method, method),
                            'failure_status': 'failed',
                            'failure_reason': reason,
                            'runtime_seconds': round(time.time() - started, 3),
                        }
                        rows.append(failure)
                        if manifest is not None:
                            _update_manifest(manifest, failure, 'failed', reason)
                        if persist is not None:
                            persist([failure])
                        message = (f"{spec['dataset']} subject={int(key)+1} seed={seed} "
                                   f"condition={method}: {reason}")
                        errors.append(message)
                        print(f"[ERR] {message}", flush=True)
                        if fail_fast:
                            raise
            except Exception as exc:  # noqa: BLE001
                message = (f"{spec['dataset']} {spec['teacher']}->{spec['student']} "
                           f"key={key} seed={seed} run_id={spec['run_id']}: {exc}")
                errors.append(message)
                print(f"[ERR] {message}", flush=True)
                for method in spec['methods']:
                    row = {
                        'config_name': spec['config_name'],
                        'run_id': spec['run_id'],
                        'grid_index': spec['grid_index'],
                        'dataset': spec['dataset'],
                        'subject': int(key) + 1,
                        'key': int(key),
                        'session': legacy._session_default(spec['dataset']),
                        'seed': int(seed),
                        'protocol': spec['protocol'],
                        'teacher': spec['teacher'],
                        'student': spec['student'],
                        'method': method,
                        'condition_label': legacy.METHOD_LABELS.get(method, method),
                        'failure_status': 'failed',
                        'failure_reason': str(exc),
                    }
                    if manifest is not None:
                        _update_manifest(manifest, row, 'failed', str(exc))
                    rows.append(row)
                    if persist is not None:
                        persist([row])
                if fail_fast:
                    raise
    return rows, errors


def _float_or_nan(row, key):
    try:
        value = float(row.get(key, ''))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _bootstrap_ci(values, seed=0, n_boot=4000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))


def _wilcoxon(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0.0):
        return np.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(values, zero_method='wilcox', alternative='two-sided').pvalue)
    except Exception:  # noqa: BLE001
        return np.nan


def _holm_adjust(pairs):
    ordered = sorted(pairs, key=lambda item: item[1] if np.isfinite(item[1]) else 1.0)
    adjusted = {}
    m = len(ordered)
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        if not np.isfinite(value):
            adjusted[key] = np.nan
            continue
        adjusted[key] = min(1.0, max(running, (m - index) * value))
        running = adjusted[key]
    return adjusted


def _write_summaries(run_root, rows):
    complete = [row for row in rows if row.get('failure_status') == 'complete']
    methods = ['Base', 'KD_all', 'CE_MI']
    subject_rows = []
    for dataset in sorted({row.get('dataset') for row in complete}):
        for subject in sorted({row.get('subject') for row in complete
                               if row.get('dataset') == dataset}, key=int):
            for method in methods:
                selected = [row for row in complete
                            if row.get('dataset') == dataset
                            and row.get('subject') == subject
                            and row.get('method') == method]
                values = {
                    key: np.asarray([_float_or_nan(row, key) for row in selected], dtype=float)
                    for key in ('test_accuracy', 'test_balanced_accuracy', 'test_kappa')
                }
                subject_rows.append({
                    'dataset': dataset,
                    'subject': int(subject),
                    'method': method,
                    'seed_count': len(selected),
                    'failed_seed_count': 3 - len(selected),
                    'accuracy_mean': float(np.nanmean(values['test_accuracy']))
                    if np.isfinite(values['test_accuracy']).any() else np.nan,
                    'balanced_accuracy_mean': float(np.nanmean(values['test_balanced_accuracy']))
                    if np.isfinite(values['test_balanced_accuracy']).any() else np.nan,
                    'kappa_mean': float(np.nanmean(values['test_kappa']))
                    if np.isfinite(values['test_kappa']).any() else np.nan,
                })
    dataset_rows = []
    contrast_rows = []
    raw_p = []
    for dataset in sorted({row['dataset'] for row in subject_rows}):
        for method in methods:
            selected = [row for row in subject_rows
                        if row['dataset'] == dataset and row['method'] == method]
            bal = np.asarray([row['balanced_accuracy_mean'] for row in selected], dtype=float)
            bal = bal[np.isfinite(bal)]
            dataset_rows.append({
                'dataset': dataset,
                'method': method,
                'subject_count': len(bal),
                'balanced_accuracy_mean': float(bal.mean()) if len(bal) else np.nan,
                'balanced_accuracy_sd': float(bal.std(ddof=1)) if len(bal) > 1 else np.nan,
                'accuracy_mean': float(np.nanmean([
                    row['accuracy_mean'] for row in selected])) if selected else np.nan,
                'kappa_mean': float(np.nanmean([
                    row['kappa_mean'] for row in selected])) if selected else np.nan,
            })
        for left, right in (('CE_MI', 'Base'), ('CE_MI', 'KD_all'), ('KD_all', 'Base')):
            left_map = {row['subject']: row['balanced_accuracy_mean']
                        for row in subject_rows
                        if row['dataset'] == dataset and row['method'] == left}
            right_map = {row['subject']: row['balanced_accuracy_mean']
                         for row in subject_rows
                         if row['dataset'] == dataset and row['method'] == right}
            deltas = np.asarray([
                left_map[s] - right_map[s]
                for s in sorted(set(left_map) & set(right_map))
                if np.isfinite(left_map[s]) and np.isfinite(right_map[s])
            ], dtype=float)
            ci_low, ci_high = _bootstrap_ci(deltas, seed=666 + len(contrast_rows))
            wins = int((deltas > 0).sum())
            ties = int((deltas == 0).sum())
            losses = int((deltas < 0).sum())
            p_value = _wilcoxon(deltas)
            comparison = f'{left}-{right}'
            raw_p.append((f'{dataset}:{comparison}', p_value))
            contrast_rows.append({
                'dataset': dataset,
                'comparison': comparison,
                'n_subjects': len(deltas),
                'mean_delta_balanced_accuracy': float(deltas.mean()) if len(deltas) else np.nan,
                'median_delta_balanced_accuracy': float(np.median(deltas)) if len(deltas) else np.nan,
                'wins': wins,
                'ties': ties,
                'losses': losses,
                'bootstrap_ci95_low': ci_low,
                'bootstrap_ci95_high': ci_high,
                'wilcoxon_p': p_value,
            })
    holm = _holm_adjust(raw_p)
    for row in contrast_rows:
        row['holm_p'] = holm.get(
            f"{row['dataset']}:{row['comparison']}", np.nan)
    dataset_rows.extend(contrast_rows)
    if dataset_rows:
        _atomic_write_csv(Path(run_root) / 'results_per_dataset.csv', dataset_rows)
    _atomic_write_csv(Path(run_root) / 'results_per_subject.csv', subject_rows)
    _atomic_write_csv(Path(run_root) / 'results_per_run.csv', rows)

    macro = []
    for method in methods:
        values = [row['balanced_accuracy_mean'] for row in subject_rows
                  if row['method'] == method and np.isfinite(row['balanced_accuracy_mean'])]
        by_dataset = []
        for dataset in sorted({row['dataset'] for row in subject_rows}):
            ds = [row['balanced_accuracy_mean'] for row in subject_rows
                  if row['dataset'] == dataset and row['method'] == method
                  and np.isfinite(row['balanced_accuracy_mean'])]
            if ds:
                by_dataset.append(float(np.mean(ds)))
        macro.append({
            'scope': 'dataset_macro',
            'method': method,
            'dataset_count': len(by_dataset),
            'balanced_accuracy_mean': float(np.mean(by_dataset)) if by_dataset else np.nan,
            'subject_count': len(values),
        })
    _atomic_write_csv(Path(run_root) / 'results_per_dataset_macro.csv', macro)
    return subject_rows, dataset_rows, macro, contrast_rows


def _write_report(run_root, loaded, rows, subject_rows, dataset_rows, macro,
                  contrast_rows, total_runs, errors):
    complete = [row for row in rows if row.get('failure_status') == 'complete']
    failed = [row for row in rows if row.get('failure_status') != 'complete']
    macro_map = {row['method']: row['balanced_accuracy_mean'] for row in macro}
    ce_base = macro_map.get('CE_MI', np.nan) - macro_map.get('Base', np.nan)
    ce_kd = macro_map.get('CE_MI', np.nan) - macro_map.get('KD_all', np.nan)
    dataset_values = {}
    for row in subject_rows:
        dataset_values.setdefault(row['dataset'], {})[row['method']] = (
            row['balanced_accuracy_mean'])
    ce_base_by_dataset = [
        values['CE_MI'] - values['Base'] for values in dataset_values.values()
        if all(method in values for method in ('Base', 'CE_MI'))]
    ce_kd_by_dataset = [
        values['CE_MI'] - values['KD_all'] for values in dataset_values.values()
        if all(method in values for method in ('KD_all', 'CE_MI'))]
    direction_stable = (
        bool(ce_base_by_dataset) and all(value >= 0 for value in ce_base_by_dataset)
        and bool(ce_kd_by_dataset) and all(value >= 0 for value in ce_kd_by_dataset))
    diag_means = {}
    for method in ('Base', 'KD_all', 'CE_MI'):
        selected = [row for row in complete if row.get('method') == method]
        diag_means[method] = {
            field: float(np.mean([float(row[field]) for row in selected]))
            if selected else np.nan
            for field in ('full_train_mi', 'teacher_student_prediction_agreement',
                          'mean_abs_probability_diff')
        }
    mi_delta = diag_means['CE_MI']['full_train_mi'] - diag_means['Base']['full_train_mi']
    agreement_delta = (diag_means['CE_MI']['teacher_student_prediction_agreement']
                       - diag_means['Base']['teacher_student_prediction_agreement'])
    # Use the requested conservative decision rule: a small positive macro
    # delta with mixed dataset directions and non-significant paired tests is
    # not A/B evidence.  C is reserved for a diagnostic MI/agreement increase
    # without any test improvement; otherwise the formal result is D.
    if (direction_stable and np.isfinite(ce_base) and ce_base > 0
            and np.isfinite(ce_kd) and ce_kd >= 0):
        conclusion = 'A'
    elif (direction_stable and np.isfinite(ce_base) and ce_base > 0
          and np.isfinite(ce_kd) and ce_kd < 0):
        conclusion = 'B'
    elif (np.isfinite(mi_delta) and mi_delta > 0
          and np.isfinite(agreement_delta) and agreement_delta >= 0
          and np.isfinite(ce_base) and ce_base <= 0):
        conclusion = 'C'
    else:
        conclusion = 'D'

    def _fmt(value, digits=3):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 'NA'
        return 'NA' if not np.isfinite(value) else f'{value:.{digits}f}'

    grouped = {}
    for row in complete:
        grouped.setdefault((row.get('dataset'), row.get('subject'),
                            row.get('seed')), []).append(row)
    hash_fields = ('split_uid_hash', 'initial_state_hash', 'batch_order_hash')
    hash_failures = sum(
        any(len({item.get(field, '') for item in group}) != 1
            for field in hash_fields)
        for group in grouped.values())
    alignment_ok = all(
        '"labels_match": true' in row.get('teacher_train_uid_alignment', '')
        and '"uid_set_match": true' in row.get('teacher_train_uid_alignment', '')
        for row in complete)
    finite_ok = all(
        legacy._finite_metric(row.get(field))
        for row in complete
        for field in ('test_accuracy', 'test_balanced_accuracy', 'test_kappa',
                      'full_train_mi', 'teacher_student_prediction_agreement'))
    collapsed = sum(str(row.get('collapse_flag')).lower() == 'true'
                    for row in complete)
    first = complete[0] if complete else {}
    kd_first = next((row for row in complete if row.get('method') == 'KD_all'), first)
    ce_first = next((row for row in complete if row.get('method') == 'CE_MI'), first)
    lines = [
        '# CE+MI formal distillation report', '',
        f'- conclusion: **{conclusion}**',
        f'- configured runs: {total_runs}',
        f'- complete condition runs: {len(complete)}',
        f'- failed/incomplete condition runs: {len(failed)}',
        '- protocol: subject-wise few-shot, 30% train / 70% test within one loader session',
        '- datasets: BNCI2014001, BNCI2014004, BNCI2015001, AlexMI',
        '- teacher/student: MIRepNet → IFNet',
        '- no LOSO, K-fold, QC filtering, subject exclusion, or fifth dataset was run.',
        '- teacher artifacts were read only from train paths; no test artifact was read.',
        f'- CE_MI was fixed at lam_mi={ce_first.get("lam_mi", "NA")}, '
        'eps=1e-8, T=1; no test-based tuning was used.',
        f'- integrity: finite_metrics={finite_ok}, collapsed_runs={collapsed}, '
        f'UID/hash failures={hash_failures}, teacher_alignment_pass={alignment_ok}.',
        f'- runtime parameters: epochs={first.get("student_epochs", "NA")}, '
        f'optimizer={first.get("optimizer", "NA")}, lr={first.get("student_lr", "NA")}, '
        f'weight_decay={first.get("student_weight_decay", "NA")}, '
        f'batch_size={first.get("student_batch_size", "NA")}, '
        f'scheduler={first.get("scheduler", "NA")}, final-epoch model used.',
        '- KD_all: existing Vanilla KD, all train samples, '
        f'lam_kd={kd_first.get("lam_kd", "NA")}, T={kd_first.get("temperature", "NA")}; '
        'loss is CE + lam_kd*T^2*KL(log_softmax(student/T), softmax(teacher/T)).',
        f'- CE_MI: CE + {ce_first.get("lam_mi", "NA")}*(-class-joint MI), '
        'eps=1e-8, raw-logit softmax T=1; '
        'no KL, teacher feature, projection, InfoNCE, MMD, prototype or BiKD.',
        '', '## Dataset macro balanced accuracy', '',
        '| method | mean | subjects | datasets |', '|---|---:|---:|---:|',
    ]
    for row in macro:
        lines.append(f"| {row['method']} | {row['balanced_accuracy_mean']:.3f} | "
                     f"{row['subject_count']} | {row['dataset_count']} |")
    lines.extend(['', '## Per-dataset test metrics (subject means)', '',
                  '| dataset | method | accuracy mean | balanced accuracy mean ± SD | kappa mean | n subjects |',
                  '|---|---|---:|---:|---:|---:|'])
    for dataset in sorted(dataset_values):
        for method in ('Base', 'KD_all', 'CE_MI'):
            selected = [row for row in subject_rows
                        if row['dataset'] == dataset and row['method'] == method]
            ba = np.asarray([row['balanced_accuracy_mean'] for row in selected], dtype=float)
            lines.append(
                f'| {dataset} | {method} | '
                f'{_fmt(np.nanmean([row["accuracy_mean"] for row in selected]))} | '
                f'{_fmt(np.nanmean(ba))} ± {_fmt(np.nanstd(ba, ddof=1) if len(ba) > 1 else np.nan)} | '
                f'{_fmt(np.nanmean([row["kappa_mean"] for row in selected]))} | {len(selected)} |')
    lines.extend(['', '## Paired subject contrasts', '',
                  '| dataset | comparison | n | mean Δ BA | win/tie/loss | CI95 | Wilcoxon | Holm |',
                  '|---|---|---:|---:|---|---|---:|---:|'])
    for row in contrast_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['n_subjects']} | "
            f"{row['mean_delta_balanced_accuracy']:.3f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"[{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}] | "
            f"{row['wilcoxon_p']:.4g} | {row['holm_p']:.4g} |")
    lines.extend(['', '## Full-train MI and teacher/student diagnostics', '',
                  '| method | full-train MI mean | prediction agreement mean (%) | mean abs probability diff |',
                  '|---|---:|---:|---:|'])
    for method in ('Base', 'KD_all', 'CE_MI'):
        values = diag_means[method]
        lines.append(
            f'| {method} | {_fmt(values["full_train_mi"])} | '
            f'{_fmt(values["teacher_student_prediction_agreement"])} | '
            f'{_fmt(values["mean_abs_probability_diff"])} |')
    lines.extend([
        f'- CE_MI minus Base: full-train MI {_fmt(mi_delta)}, '
        f'prediction agreement {_fmt(agreement_delta)} percentage points, '
        f'macro test BA {_fmt(ce_base)} percentage points.',
        '', '## Interpretation', '',
        f'- Conservative decision is **{conclusion}**: dataset directions are '
        f'{"consistent" if direction_stable else "mixed"}, and the formal paired '
        'comparisons do not establish a stable cross-dataset CE_MI advantage.',
        '- Full-train MI is a diagnostic only. A small MI change or output '
        'agreement change is not evidence of test-time generalization.',
        '- Test data were used only for final evaluation after each complete '
        'training run; never for lam_mi, epoch, checkpoint, threshold, seed or '
        'subject selection.',
        '- The three conditions share the same session-local 30/70 split, '
        'initial_state_dict, shuffle/batch order and formal IFNet settings.',
        '- Results are subject-wise few-shot experiment units, not folds; no '
        'LOSO/K-fold aggregation or excluding-S1 result was produced.',
        '',
                  'Final full-train MI and teacher/student agreement are diagnostics only; '
                  'test metrics determine generalization. If CE+MI raises MI/agreement '
                  'without raising test balanced accuracy, it makes the student more '
                  'similar to the teacher without evidence of classification benefit.', '',
                  'Training histories and final checkpoints are stored per run under '
                  '`training_history/` and `checkpoints/`; predictions are under '
                  '`predictions/`.'])
    _atomic_write_text(Path(run_root) / 'report.md', '\n'.join(lines) + '\n')
    return conclusion


def run(args, loaded):
    config_loader.require_experiment_type(loaded, "distill")
    legacy._set_thread_defaults()
    specs = loaded["runs"]
    if not specs:
        raise ValueError("resolved distillation config contains no runs")

    # This config is intentionally a single, fixed formal matrix.  Keep the
    # assertions here next to execution so an accidental config expansion can
    # never silently turn into a different experiment.
    expected_datasets = [
        "BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI",
    ]
    config = loaded.get("config", {})
    if config.get("datasets") != expected_datasets:
        raise ValueError(
            "formal CE+MI scope must contain exactly the four configured "
            f"datasets in order: {expected_datasets}")
    if config.get("protocols") != ["fewshot"]:
        raise ValueError("formal CE+MI scope must use protocol=fewshot only")
    if config.get("subjects") is not None or config.get("seeds") is not None:
        raise ValueError("formal CE+MI scope must expand all subjects and dataset seeds")
    if any("BNCI2014001-4" in str(spec.get("dataset")) for spec in specs):
        raise ValueError("BNCI2014001-4 is excluded from the formal CE+MI scope")
    if any((spec.get("teacher"), spec.get("student")) != ("mirepnet", "ifnet")
           for spec in specs):
        raise ValueError("formal CE+MI scope must use the unique mirepnet -> ifnet pair")
    if any(spec.get("protocol") != "fewshot" for spec in specs):
        raise ValueError("formal CE+MI scope contains a non-fewshot run")
    if any(spec.get("methods") != ["Base", "KD_all", "CE_MI"] for spec in specs):
        raise ValueError("each expanded unit must contain exactly Base, KD_all, CE_MI")
    lam_mi_values = {
        float(spec.get("params", {}).get("lam_mi", 0.1)) for spec in specs
    }
    if len(lam_mi_values) != 1 or not np.isfinite(next(iter(lam_mi_values))):
        raise ValueError("all formal CE+MI specs must share one finite lam_mi")
    configured_lam_mi = next(iter(lam_mi_values))

    manifest = _build_manifest(specs)
    expected_subjects = {
        "BNCI2014001": 9,
        "BNCI2014004": 9,
        "BNCI2015001": 12,
        "AlexMI": 8,
    }
    manifest_datasets = sorted({item["dataset"] for item in manifest})
    manifest_units = {
        (item["dataset"], item["key"], item["seed"])
        for item in manifest
    }
    manifest_subjects = {
        dataset: len({item["key"] for item in manifest if item["dataset"] == dataset})
        for dataset in expected_datasets
    }
    if manifest_datasets != sorted(expected_datasets):
        raise ValueError(f"unexpected expanded datasets: {manifest_datasets}")
    if manifest_subjects != expected_subjects:
        raise ValueError(
            f"unexpected subject expansion: {manifest_subjects}; "
            f"expected {expected_subjects}")
    if len(manifest_units) != 114 or len(manifest) != 342:
        raise ValueError(
            f"unexpected formal expansion: {len(manifest_units)} units, "
            f"{len(manifest)} condition trainings; expected 114 and 342")
    if {item["seed"] for item in manifest} != {666, 667, 668}:
        raise ValueError("formal CE+MI scope must use seeds 666, 667 and 668")

    output = config_loader.resolve_path(specs[0]["output"])
    run_root = output.with_suffix("")
    resolved = config_loader.resolved_path(output, specs[0]["config_name"])
    config_resolved_path = run_root / "config_resolved.yaml"
    manifest_path = run_root / "run_manifest.csv"
    provenance_path = run_root / "execution_provenance.json"
    run_results_path = run_root / "results_per_run.csv"
    # ``main`` opens execution.log before entering ``run``.  That empty/log-only
    # directory is not a formal result and is safe to initialize; any actual
    # CSV, manifest, provenance, checkpoint, history or prediction is still a
    # hard overwrite refusal unless --resume/--force is explicit.
    run_root_contents = (
        [path for path in run_root.iterdir() if path.name != "execution.log"]
        if run_root.is_dir() else []
    )
    targets = (output, resolved)
    if (any(path.exists() for path in targets) or run_root_contents) \
            and not (args.force or args.resume):
        existing = next((path for path in targets if path.exists()),
                        run_root_contents[0])
        raise FileExistsError(
            f"refusing to overwrite formal output target {existing}; "
            "use --resume or --force")
    run_root.mkdir(parents=True, exist_ok=True)

    git_before = {
        "commit": _git_value("rev-parse", "HEAD"),
        "status_short": _git_value("status", "--short"),
    }
    source_path = config_loader.resolve_path(loaded["source_path"])
    formal_scope = {
        "datasets": expected_datasets,
        "protocol": "fewshot",
        "train_fraction": 0.30,
        "test_fraction": 0.70,
        "teacher": "mirepnet",
        "student": "ifnet",
        "seeds": [666, 667, 668],
        "subjects_by_dataset": expected_subjects,
        "experiment_units": len(manifest_units),
        "condition_trainings": len(manifest),
        "conditions": ["Base", "KD_all", "CE_MI"],
        "ce_mi": {"lam_mi": configured_lam_mi, "eps": 1e-8, "temperature": 1.0},
        "no_loso": True,
        "no_kfold": True,
    }
    resolved_payload = config_loader.resolved_payload(loaded)
    resolved_payload["formal_scope"] = formal_scope
    resolved_text = yaml.safe_dump(
        resolved_payload, sort_keys=False, allow_unicode=True)
    _atomic_write_text(config_resolved_path, resolved_text)
    # Preserve the loader's conventional sibling resolved config as well, but
    # write it atomically and with the same formal-scope annotation.
    _atomic_write_text(resolved, resolved_text)

    provenance = {
        "status": "running",
        "source_config": str(source_path),
        "config_sha256": _config_sha256(source_path),
        "formal_scope": formal_scope,
        "session_provenance": {
            dataset: legacy._session_default(dataset) for dataset in expected_datasets
        },
        "git": {"before": git_before},
        "gpu": {
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
            "requested_logical_gpu": args.gpu,
            "device": None,
        },
        "teacher_artifacts": [],
        "test_artifacts_read": False,
        "test_split_used_during_training": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_write_json(provenance_path, provenance)
    _write_manifest(manifest_path, manifest)

    device = _device(args.gpu)
    provenance["gpu"]["device"] = device
    _atomic_write_json(provenance_path, provenance)

    prior_path = output if output.exists() else run_results_path
    prior_all = _read_rows(prior_path) if args.resume else []
    prior_all = _upsert_rows([], prior_all)
    prior_complete = (
        _load_existing_complete_rows(prior_path, run_root)
        if args.resume and prior_path.exists() else []
    )
    existing_keys = {_row_key(row) for row in prior_complete}
    for row in prior_complete:
        _update_manifest(manifest, row, "complete", "")
    rows = list(prior_all)
    errors = []

    def persist(incoming):
        nonlocal rows
        rows = _upsert_rows(rows, incoming)
        _write_rows(output, rows)
        _write_rows(run_results_path, rows)
        _write_manifest(manifest_path, manifest)

    # The static expansion check above is deliberately printed once before
    # the first model is built, making the formal scope auditable in the log.
    print(
        f"[static] datasets={expected_datasets} subjects={sum(expected_subjects.values())} "
        f"units={len(manifest_units)} condition_trainings={len(manifest)}",
        flush=True,
    )
    print(
        f"[config] {loaded['source_path']} type=distill specs={len(specs)} "
        f"device={device} resume={args.resume}", flush=True)
    print(f"[resolved] {config_resolved_path}", flush=True)
    for spec in specs:
        new_rows, spec_errors = _run_spec(
            spec, device, existing_keys,
            bool(args.fail_fast or spec.get("fail_fast", False)),
            run_root=run_root, manifest=manifest, total_runs=len(manifest),
            persist=persist)
        persist(new_rows)
        errors.extend(spec_errors)
    _write_rows(output, rows)
    _write_rows(run_results_path, rows)
    _write_manifest(manifest_path, manifest)
    subject_rows, dataset_rows, macro, contrast_rows = _write_summaries(
        run_root, rows)
    conclusion = _write_report(
        run_root, loaded, rows, subject_rows, dataset_rows, macro,
        contrast_rows, len(manifest), errors)

    artifact_map = {}
    for row in rows:
        path = row.get("teacher_artifact_path")
        sha = row.get("teacher_artifact_sha256")
        if path and sha:
            artifact_map[path] = {
                "path": path,
                "sha256": sha,
                "size": os.path.getsize(path) if os.path.isfile(path) else None,
                "mtime": os.path.getmtime(path) if os.path.isfile(path) else None,
            }
    git_after = {
        "commit": _git_value("rev-parse", "HEAD"),
        "status_short": _git_value("status", "--short"),
    }
    provenance.update({
        "status": "complete" if not errors else "completed_with_failures",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"before": git_before, "after": git_after},
        "teacher_artifacts": sorted(artifact_map.values(), key=lambda item: item["path"]),
        "completed_condition_runs": sum(item["status"] == "complete" for item in manifest),
        "failed_condition_runs": sum(item["status"] == "failed" for item in manifest),
        "conclusion": conclusion,
    })
    _atomic_write_json(provenance_path, provenance)
    print(f"[write] {output} rows={len(rows)}", flush=True)
    print(f"[write] formal directory={run_root}", flush=True)
    if errors:
        print(f"[errors] {len(errors)} cells failed", flush=True)
        for error in errors:
            print(f"  {error}", flush=True)
    return 1 if errors else 0


def main(argv=None):
    args = parse_args(argv)
    loaded = config_loader.load_experiment(args.config)
    config_loader.require_experiment_type(loaded, "distill")
    output = config_loader.resolve_path(loaded["runs"][0]["output"])
    # Formal MI logs live with the run manifest and checkpoints, rather than
    # in the generic distillation log directory.
    log_path = output.with_suffix("") / "execution.log"
    run_root_contents = (
        [path for path in log_path.parent.iterdir() if path.name != log_path.name]
        if log_path.parent.is_dir() else []
    )
    if log_path.exists() and run_root_contents and not (args.force or args.resume):
        raise FileExistsError(f"refusing to overwrite {log_path}; use --resume or --force")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with log_path.open(mode, buffering=1) as log_handle:
        tee_out = _Tee(sys.stdout, log_handle)
        tee_err = _Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[log] writing stdout/stderr to {log_path}", flush=True)
            return run(args, loaded)
