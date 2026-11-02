#!/usr/bin/env python3
"""Read-only MIRepNet teacher EA ablation over existing LOSO artifacts.

The original per-subject-EA logits are loaded from existing predictions. The
other two input conditions are evaluated with completed T_CE/T_MCC checkpoints.
No model is trained and no student input or checkpoint is changed.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.linalg import fractional_matrix_power

from collab import mirepnet_t2s as core
from collab import t2s_training as training
from data import loso_source
from data.preproc import bandpass
from experiments.distill import mirepnet_t2s_data as source
from experiments.storage import (REPORTS_ROOT, RESULTS_ROOT, resolve_local_file,
                                 require_report_output)
from models import get_adapter


EXPERIMENT = "mirepnet_t2s_allsets_loso_v1"
DATASETS = ("BNCI2014004", "BNCI2014001", "BNCI2014001-4",
            "BNCI2015001", "AlexMI")
N_SUBJECTS = dict(zip(DATASETS, (9, 9, 9, 12, 8)))
SEEDS = (666, 667, 668)
TEACHERS = ("T_CE", "T_MCC")
ALIGNMENTS = ("subject_EA_existing", "source_mean", "no_EA")
STUDENTS = ("ifnet", "eegnet", "adfcnn")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261102
REPORT_DIR = ROOT / "results/diagnostics/mirepnet_teacher_ea_alignment_20261102"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(resolve_local_file(path)).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path = require_report_output(path)
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    labels = np.unique(y)
    return float(np.mean([np.mean(pred[y == label] == label) for label in labels]))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan, math.nan
    mean = float(x.mean())
    indices = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    draws = x[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, float(low), float(high)


def cell_root(dataset: str, seed: int, fold: int) -> Path:
    return (RESULTS_ROOT / "distill" / EXPERIMENT / dataset /
            f"seed_{seed}" / f"subject_{fold + 1:02d}")


def records_path(dataset: str) -> Path:
    return ROOT / "results/distill/mirepnet_t2s_allsets_v1" / f"{dataset}_records.json"


def load_record_index() -> dict[tuple, dict]:
    index = {}
    for dataset in DATASETS:
        path = records_path(dataset)
        if not path.is_file():
            raise FileNotFoundError(f"existing allsets report is missing: {path}")
        for row in read_json(path):
            key = (dataset, str(row["student"]).lower(), row["method"],
                   int(row["seed"]), int(row["subject"]))
            if key in index:
                raise ValueError(f"duplicate existing evaluation record: {key}")
            index[key] = row
    return index


def load_prediction(row: dict, y: np.ndarray, uids: np.ndarray,
                    context: str) -> dict:
    path = Path(resolve_local_file(row["prediction_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{context}: existing prediction is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        item = {name: np.asarray(saved[name]) for name in saved.files}
    for name in ("trial_uid", "y", "logits"):
        if name not in item:
            raise ValueError(f"{context}: {path} lacks {name}")
    if (not np.array_equal(item["trial_uid"].astype(str), np.asarray(uids).astype(str))
            or not np.array_equal(item["y"], y)
            or item["logits"].shape != (len(y), len(np.unique(y)))):
        raise ValueError(f"{context}: existing prediction UID/label/shape mismatch")
    measured = balanced_accuracy(y, item["logits"].argmax(axis=1))
    if not np.isclose(measured, float(row["balanced_accuracy"]), atol=1e-10):
        raise ValueError(f"{context}: stored BA does not match its logits")
    return {"logits": np.asarray(item["logits"], dtype=np.float32),
            "ba": measured, "path": str(path)}


def source_mean_covariance(x: np.ndarray, subject_ids: np.ndarray,
                           target_subject: int,
                           covariance_cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Equal-weight average of source-subject trial-mean covariances only."""
    subjects = np.asarray(subject_ids)
    source_subjects = [int(s) for s in np.unique(subjects) if int(s) != int(target_subject)]
    if not source_subjects:
        raise ValueError("source-mean EA requires at least one source subject")
    subject_covariances = []
    for subject in source_subjects:
        if covariance_cache is not None and subject in covariance_cache:
            subject_covariances.append(covariance_cache[subject])
            continue
        # This indexing deliberately excludes the held-out target before any
        # covariance is computed.
        trials = np.asarray(x[subjects == subject], dtype=np.float64)
        if trials.ndim != 3 or not len(trials):
            raise ValueError(f"empty/invalid source trials for subject {subject}")
        trial_covariances = np.stack([np.cov(trial) for trial in trials], axis=0)
        subject_covariance = trial_covariances.mean(axis=0)
        if covariance_cache is not None:
            covariance_cache[subject] = subject_covariance
        subject_covariances.append(subject_covariance)
    result = np.mean(np.stack(subject_covariances, axis=0), axis=0)
    return np.asarray((result + result.T) * 0.5, dtype=np.float64)


def source_mean_invariance_check(x: np.ndarray, subject_ids: np.ndarray,
                                 target_subject: int, dataset: str) -> str:
    """Replace target EEG and prove R_src is bitwise unchanged."""
    before = source_mean_covariance(x, subject_ids, target_subject)
    altered = np.array(x, copy=True)
    target = np.asarray(subject_ids) == int(target_subject)
    rng = np.random.default_rng(20261102 + int(target_subject))
    altered[target] = rng.standard_normal(altered[target].shape).astype(altered.dtype)
    after = source_mean_covariance(altered, subject_ids, target_subject)
    np.testing.assert_array_equal(before, after)
    return array_sha(before)


def load_dataset_bundle(dataset: str) -> dict:
    raw = loso_source.load_dataset(dataset, None)
    teacher_data, _ = source.load_inputs(dataset, "ifnet", auto_prepare=False)
    raw_uid = np.asarray(raw.trial_uid).astype(str)
    teacher_uid = np.asarray(teacher_data.trial_uid).astype(str)
    if len(set(raw_uid.tolist())) != len(raw_uid) or set(raw_uid) != set(teacher_uid):
        raise ValueError(f"{dataset}: raw and teacher input UID sets differ")
    raw_row = {uid: index for index, uid in enumerate(raw_uid.tolist())}
    order = np.asarray([raw_row[uid] for uid in teacher_uid.tolist()], dtype=np.int64)
    x = np.asarray(raw.x[order], dtype=np.float32)
    subjects = np.asarray(raw.subjects[order], dtype=np.int64)
    y = np.asarray(raw.y[order], dtype=np.int64)
    if (not np.array_equal(subjects, np.asarray(teacher_data.subjects))
            or not np.array_equal(y, np.asarray(teacher_data.y))):
        raise ValueError(f"{dataset}: raw and teacher labels/subjects differ after UID alignment")
    filtered = bandpass(x.astype(np.float64), 250, 8.0, 30.0).astype(np.float32)
    adapter = get_adapter("mirepnet", device="cpu", dataset_name=dataset,
                          skip_preprocess=True)
    return {"raw": raw, "teacher_data": teacher_data, "filtered": filtered,
            "subjects": subjects, "y": y, "uids": teacher_uid,
            "adapter": adapter}


def pad_channels(adapter, x: np.ndarray, dataset: str) -> np.ndarray:
    result = adapter._pad(np.asarray(x, dtype=np.float32), adapter._template,
                          adapter._src_channels[dataset])
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 3 or tuple(result.shape[1:]) != (45, 1000):
        raise ValueError(f"{dataset}: MIRepNet input shape after channel padding is {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{dataset}: non-finite MIRepNet input")
    return result


def inverse_sqrt_covariance(r_source: np.ndarray) -> np.ndarray:
    root = fractional_matrix_power(r_source, -0.5)
    if np.iscomplexobj(root):
        maximum_imaginary = float(np.max(np.abs(root.imag)))
        if maximum_imaginary > 1e-8:
            raise FloatingPointError(
                f"source covariance inverse square root is complex ({maximum_imaginary})")
        root = root.real
    root = np.asarray(root, dtype=np.float64)
    if not np.isfinite(root).all():
        raise FloatingPointError("source covariance inverse square root is non-finite")
    return root


def whiten_from_source(x_target: np.ndarray, r_source: np.ndarray) -> np.ndarray:
    root = inverse_sqrt_covariance(r_source)
    transformed = np.einsum("cd,nct->ndt", root,
                            np.asarray(x_target, dtype=np.float64), optimize=True)
    return np.asarray(transformed, dtype=np.float32)


def load_teacher(record: dict, dataset: str, seed: int, fold: int,
                 teacher_name: str, device: torch.device,
                 baseline: tuple[object, torch.nn.Module]) -> tuple[object, torch.nn.Module, str]:
    adapter, t0 = baseline
    result_dir = cell_root(dataset, seed, fold)
    stage_manifest_path = result_dir / teacher_name / "complete.json"
    manifest = read_json(stage_manifest_path)
    if manifest.get("status") != "complete" or int(manifest.get("final_epoch", -1)) != 10:
        raise ValueError(f"{dataset}/{seed}/{fold + 1}/{teacher_name}: incomplete/nonterminal stage")
    checkpoint = Path(resolve_local_file(manifest["checkpoint"]))
    if not checkpoint.is_file() or sha_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError(f"{dataset}/{seed}/{fold + 1}/{teacher_name}: checkpoint hash mismatch")
    model = copy.deepcopy(t0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if training.state_hash(model) != manifest.get("model_state_sha256"):
        raise ValueError(f"{dataset}/{seed}/{fold + 1}/{teacher_name}: model state hash mismatch")
    model.to(device).eval()
    return adapter, model, manifest["checkpoint_sha256"]


def infer_logits(adapter, model: torch.nn.Module, x: np.ndarray,
                 device: torch.device, batch_size: int = 64) -> np.ndarray:
    before = training.state_hash(model)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size],
                                 dtype=torch.float32, device=device)
            _, logits = core.forward_native(adapter, model, xb)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("MIRepNet returned non-finite logits")
            chunks.append(logits.detach().float().cpu().numpy())
    after = training.state_hash(model)
    if before != after:
        raise RuntimeError("read-only inference changed the adapted teacher checkpoint state")
    return np.concatenate(chunks, axis=0)


def preflight(record_index: dict) -> int:
    missing = []
    expected_records = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            for fold in range(N_SUBJECTS[dataset]):
                expected_records += 1
                cell = cell_root(dataset, seed, fold)
                completion = cell / "complete.json"
                evaluation = cell / "evaluation.json"
                if not completion.is_file() or not evaluation.is_file():
                    missing.append(f"{dataset}/seed{seed}/S{fold + 1}: completion/evaluation")
                    continue
                done = read_json(completion)
                ev = read_json(evaluation)
                if done.get("status") != "complete" or ev.get("status") != "complete":
                    missing.append(f"{dataset}/seed{seed}/S{fold + 1}: status")
                    continue
                if done.get("fingerprint") != ev.get("fingerprint"):
                    missing.append(f"{dataset}/seed{seed}/S{fold + 1}: fingerprint")
                if not done.get("target_evaluated") is False:
                    missing.append(f"{dataset}/seed{seed}/S{fold + 1}: unexpected training target flag")
                records = {(str(r["student"]).lower(), r["method"]): r
                           for r in ev.get("records", [])}
                for teacher in TEACHERS:
                    row = record_index.get((dataset, "mirepnet", teacher, seed, fold + 1))
                    if row is None:
                        missing.append(f"{dataset}/seed{seed}/S{fold + 1}: {teacher} registry")
                    elif not Path(resolve_local_file(row["prediction_path"])).is_file():
                        missing.append(f"{dataset}/seed{seed}/S{fold + 1}: {teacher} logits")
                    stage = cell / teacher / "complete.json"
                    if not stage.is_file():
                        missing.append(f"{dataset}/seed{seed}/S{fold + 1}: {teacher} manifest")
                    else:
                        manifest = read_json(stage)
                        checkpoint = Path(resolve_local_file(manifest.get("checkpoint", "")))
                        if (not checkpoint.is_file()
                                or sha_file(checkpoint) != manifest.get("checkpoint_sha256")):
                            missing.append(f"{dataset}/seed{seed}/S{fold + 1}: {teacher} checkpoint")
                for student in STUDENTS:
                    row = record_index.get((dataset, student, "none", seed, fold + 1))
                    if row is None or not Path(resolve_local_file(
                            row.get("prediction_path", ""))).is_file():
                        missing.append(f"{dataset}/seed{seed}/S{fold + 1}: {student} CE logits")
                for key in ((("mirepnet", "T_CE"), ("mirepnet", "T_MCC"))
                            + tuple((student, "none") for student in STUDENTS)):
                    if key not in records:
                        missing.append(f"{dataset}/seed{seed}/S{fold + 1}: evaluation {key}")
    if missing:
        raise FileNotFoundError(
            f"Step A preflight found {len(missing)} missing/inconsistent artifacts; "
            f"no inference was run. First items: {missing[:20]}")
    return expected_records


def fold_and_gap_summaries(fold_rows: list[dict],
                           gap_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fold_summary = []
    keys = sorted({(r["dataset"], r["teacher"], r["alignment"]) for r in fold_rows})
    for dataset, teacher, alignment in keys:
        selected = [r for r in fold_rows if (r["dataset"], r["teacher"], r["alignment"])
                    == (dataset, teacher, alignment)]
        by_subject = {}
        for row in selected:
            by_subject.setdefault(int(row["subject"]), []).append(float(row["BA_pct"]))
        values = np.asarray([np.mean(by_subject[s]) for s in sorted(by_subject)])
        mean, low, high = bootstrap_mean(values, rng)
        orig_by_subject = {}
        for row in fold_rows:
            if (row["dataset"], row["teacher"], row["alignment"]) == (
                    dataset, teacher, "subject_EA_existing"):
                orig_by_subject.setdefault(int(row["subject"]), []).append(float(row["BA_pct"]))
        if alignment == "subject_EA_existing":
            diff = np.zeros_like(values)
        else:
            diffs = []
            for subject in sorted(by_subject):
                current = [float(r["BA_pct"]) for r in selected
                           if int(r["subject"]) == subject]
                original = orig_by_subject[subject]
                if len(current) != len(original):
                    raise RuntimeError("seed counts differ in paired teacher comparison")
                diffs.append(float(np.mean(np.asarray(current) - np.asarray(original))))
            diff = np.asarray(diffs)
        dmean, dlow, dhigh = bootstrap_mean(diff, rng)
        fold_summary.append({
            "dataset": dataset, "teacher": teacher, "alignment": alignment,
            "n_subject": len(values), "n_seed": len(SEEDS),
            "mean_BA_pct": mean, "bootstrap95_low_pct": low,
            "bootstrap95_high_pct": high,
            "paired_diff_vs_subject_EA_pp": dmean,
            "paired_diff95_low_pp": dlow, "paired_diff95_high_pp": dhigh,
        })

    gap_summary = []
    gap_keys = sorted({(r["dataset"], r["teacher"], r["alignment"], r["student"])
                       for r in gap_rows})
    for dataset, teacher, alignment, student in gap_keys:
        selected = [r for r in gap_rows if (r["dataset"], r["teacher"],
                    r["alignment"], r["student"]) ==
                    (dataset, teacher, alignment, student)]
        by_subject = {}
        for row in selected:
            by_subject.setdefault(int(row["subject"]), []).append(float(row["teacher_minus_CE_BA_pp"]))
        values = np.asarray([np.mean(by_subject[s]) for s in sorted(by_subject)])
        mean, low, high = bootstrap_mean(values, rng)
        gap_summary.append({
            "dataset": dataset, "teacher": teacher, "alignment": alignment,
            "student": student, "n_subject": len(values), "n_seed": len(SEEDS),
            "mean_teacher_minus_CE_BA_pp": mean,
            "bootstrap95_low_pp": low, "bootstrap95_high_pp": high,
        })
    return fold_summary, gap_summary


def run(device: torch.device) -> dict:
    record_index = load_record_index()
    total_cells = preflight(record_index)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running", "experiment": EXPERIMENT,
        "datasets": list(DATASETS), "seeds": list(SEEDS),
        "expected_loso_cells": total_cells, "completed_cells": 0,
        "teachers": list(TEACHERS), "students_for_gap": list(STUDENTS),
        "alignment_definitions": {
            "subject_EA_existing": "existing saved target logits reused; not recomputed",
            "source_mean": "8-30 Hz; equal mean of source-subject mean trial covariances; inverse square-root; then 45-channel pad",
            "no_EA": "8-30 Hz; no covariance whitening; then 45-channel pad",
        },
        "student_EA": False, "training_started": False,
        "source_mean_invariance_tests": {},
    }
    write_json(REPORT_DIR / "status.json", status)
    fold_rows, gap_rows = [], []
    invariance_by_dataset = {}
    for dataset in DATASETS:
        print(f"[data] {dataset}: loading raw and verified teacher inputs", flush=True)
        bundle = load_dataset_bundle(dataset)
        filtered = bundle["filtered"]
        subjects = bundle["subjects"]
        y_all = bundle["y"]
        uids_all = bundle["uids"]
        adapter = bundle["adapter"]
        target_subject_ids = sorted(np.unique(subjects).tolist())
        subject_covariance_cache = {}
        rsrc_cache = {}
        rsrc_sha = {}
        whitener_cache = {}
        for fold in range(N_SUBJECTS[dataset]):
            if fold not in target_subject_ids:
                raise ValueError(f"{dataset}: expected LOSO target subject {fold} is absent")
            if fold not in rsrc_cache:
                cov = source_mean_covariance(
                    filtered, subjects, fold, subject_covariance_cache)
                rsrc_cache[fold] = cov
                rsrc_sha[fold] = array_sha(cov)
                whitener_cache[fold] = inverse_sqrt_covariance(cov)
        # Direct source-only replacement test on one held-out subject per dataset.
        test_fold = 0
        invariance_hash = source_mean_invariance_check(
            filtered, subjects, test_fold, dataset)
        if invariance_hash != rsrc_sha[test_fold]:
            raise RuntimeError(f"{dataset}: stored R_src differs from source-only unit test")
        invariance_by_dataset[dataset] = {
            "target_subject_tested_zero_based": test_fold,
            "target_signal_replacement_changes_R_src": False,
            "R_src_sha256": invariance_hash,
        }
        status["source_mean_invariance_tests"] = invariance_by_dataset
        write_json(REPORT_DIR / "status.json", status)

        for seed in SEEDS:
            for fold in range(N_SUBJECTS[dataset]):
                teacher_data = bundle["teacher_data"]
                target_indices = np.flatnonzero(subjects == fold)
                if not len(target_indices):
                    raise ValueError(f"{dataset}/S{fold + 1}: target has no trials")
                target_y = y_all[target_indices]
                target_uids = uids_all[target_indices]
                if not np.array_equal(target_y, teacher_data.y[target_indices]):
                    raise RuntimeError(f"{dataset}/S{fold + 1}: target labels misaligned")
                source_cov = rsrc_cache[fold]
                source_whitener = whitener_cache[fold]
                target_raw = filtered[target_indices]
                # Apply only the source-derived whitener, then pad.
                source_mean_native = np.einsum(
                    "cd,nct->ndt", source_whitener,
                    target_raw.astype(np.float64), optimize=True).astype(np.float32)
                aligned_inputs = {
                    "source_mean": pad_channels(adapter, source_mean_native, dataset),
                    "no_EA": pad_channels(adapter, target_raw, dataset),
                }
                cell = cell_root(dataset, seed, fold)
                evaluation = read_json(cell / "evaluation.json")
                eval_records = {(str(row["student"]).lower(), row["method"]): row
                                for row in evaluation.get("records", [])}
                ce_predictions = {}
                for student in STUDENTS:
                    ce_row = record_index[(dataset, student, "none", seed, fold + 1)]
                    ce_predictions[student] = load_prediction(
                        ce_row, target_y, target_uids,
                        f"{dataset}/seed{seed}/S{fold + 1}/{student}/CE")
                t_record = source.teacher_record(
                    dataset, seed, fold, teacher_data)
                baseline = source.build_teacher(t_record, teacher_data, device)
                for teacher_name in TEACHERS:
                    teacher_record_row = record_index[
                        (dataset, "mirepnet", teacher_name, seed, fold + 1)]
                    original = load_prediction(
                        teacher_record_row, target_y, target_uids,
                        f"{dataset}/seed{seed}/S{fold + 1}/{teacher_name}/existing EA")
                    existing_eval_row = eval_records.get(("mirepnet", teacher_name))
                    if existing_eval_row is None or not np.isclose(
                            original["ba"], float(existing_eval_row["balanced_accuracy"]),
                            atol=1e-10):
                        raise ValueError(f"{dataset}/seed{seed}/S{fold + 1}/{teacher_name}: "
                                         "evaluation registry differs from reused logits")
                    fold_rows.append({
                        "dataset": dataset, "seed": seed, "subject": fold + 1,
                        "teacher": teacher_name, "alignment": "subject_EA_existing",
                        "n_trials": len(target_y), "BA_pct": 100.0 * original["ba"],
                        "source_mean_cov_sha256": "",
                        "prediction_source": str(original["path"]),
                        "checkpoint_sha256": record_index[
                            (dataset, "mirepnet", teacher_name, seed, fold + 1)
                        ].get("checkpoint_sha256", ""),
                    })
                    for student in STUDENTS:
                        ce_ba = balanced_accuracy(target_y,
                                                  ce_predictions[student]["logits"].argmax(1))
                        gap_rows.append({
                            "dataset": dataset, "seed": seed, "subject": fold + 1,
                            "teacher": teacher_name, "alignment": "subject_EA_existing",
                            "student": student, "teacher_BA_pct": 100 * original["ba"],
                            "CE_student_BA_pct": 100 * ce_ba,
                            "teacher_minus_CE_BA_pp": 100 * (original["ba"] - ce_ba),
                        })

                    t_adapter, model, checkpoint_sha = load_teacher(
                        t_record, dataset, seed, fold, teacher_name, device, baseline)
                    for alignment in ("source_mean", "no_EA"):
                        logits = infer_logits(t_adapter, model, aligned_inputs[alignment], device)
                        if logits.shape != (len(target_y), len(np.unique(y_all))):
                            raise ValueError(f"{dataset}/seed{seed}/S{fold + 1}/{teacher_name}/"
                                             f"{alignment}: logits shape {logits.shape}")
                        ba = balanced_accuracy(target_y, logits.argmax(1))
                        fold_rows.append({
                            "dataset": dataset, "seed": seed, "subject": fold + 1,
                            "teacher": teacher_name, "alignment": alignment,
                            "n_trials": len(target_y), "BA_pct": 100.0 * ba,
                            "source_mean_cov_sha256": rsrc_sha[fold] if alignment == "source_mean" else "",
                            "prediction_source": "new inference with existing frozen checkpoint",
                            "checkpoint_sha256": checkpoint_sha,
                        })
                        for student in STUDENTS:
                            ce_ba = balanced_accuracy(
                                target_y, ce_predictions[student]["logits"].argmax(1))
                            gap_rows.append({
                                "dataset": dataset, "seed": seed, "subject": fold + 1,
                                "teacher": teacher_name, "alignment": alignment,
                                "student": student, "teacher_BA_pct": 100 * ba,
                                "CE_student_BA_pct": 100 * ce_ba,
                                "teacher_minus_CE_BA_pp": 100 * (ba - ce_ba),
                            })
                    del model, t_adapter
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                del baseline
                status["completed_cells"] += 1
                status["last_completed_cell"] = {
                    "dataset": dataset, "seed": seed, "subject": fold + 1}
                write_csv(REPORT_DIR / "fold_teacher_metrics.partial.csv", fold_rows)
                write_csv(REPORT_DIR / "fold_teacher_student_gaps.partial.csv", gap_rows)
                write_json(REPORT_DIR / "status.json", status)
                print(f"[complete] {dataset} seed={seed} subject={fold + 1}/"
                      f"{N_SUBJECTS[dataset]} cells={status['completed_cells']}/{total_cells}",
                      flush=True)
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_summary, gap_summary = fold_and_gap_summaries(fold_rows, gap_rows)
    write_csv(REPORT_DIR / "fold_teacher_metrics.csv", fold_rows)
    write_csv(REPORT_DIR / "fold_teacher_student_gaps.csv", gap_rows)
    write_csv(REPORT_DIR / "dataset_teacher_summary.csv", fold_summary)
    write_csv(REPORT_DIR / "dataset_gap_summary.csv", gap_summary)
    (REPORT_DIR / "fold_teacher_metrics.partial.csv").unlink(missing_ok=True)
    (REPORT_DIR / "fold_teacher_student_gaps.partial.csv").unlink(missing_ok=True)
    status.update({
        "status": "complete", "completed_cells": total_cells,
        "n_subjects": sum(N_SUBJECTS.values()), "n_seed": len(SEEDS),
        "n_teacher_fold_alignment_rows": len(fold_rows),
        "n_teacher_student_gap_rows": len(gap_rows),
        "bootstrap": {"unit": "LOSO subject after seed averaging",
                      "replicates": BOOTSTRAPS, "seed": BOOTSTRAP_SEED},
        "report_files": [
            "fold_teacher_metrics.csv", "fold_teacher_student_gaps.csv",
            "dataset_teacher_summary.csv", "dataset_gap_summary.csv",
        ],
    })
    write_json(REPORT_DIR / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1",
                        help="inference device; use cpu to avoid shared GPU contention")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    status = run(device)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
