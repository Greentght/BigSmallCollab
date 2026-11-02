#!/usr/bin/env python3
"""Read-only CPU feature extraction and H4/H5 diagnostics for MIRepNet T2S.

Uses only the completed allsets checkpoints, fixed projections, and DiffKD priors.
The feature dumps are retained under the external project cache; reports are
written to the checkout results directory. This script never trains a model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from collab import mirepnet_t2s as core
from collab import t2s_training as training
from collab.mcc_diffkd import EpsilonPredictor, ddpm_schedule, ddim_denoise
from experiments.distill import mirepnet_t2s_data as source
from experiments.storage import (DATA_CACHE_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
    REPORTS_ROOT, require_external_output, require_report_output, resolve_local_file)
from models import get_adapter


EXPERIMENT = "mirepnet_t2s_allsets_loso_v1"
ARTIFACT_ROOT = require_external_output(
    DATA_CACHE_ROOT / "diagnostics/mirepnet_t2s_h4_h5_no_student_ea_v1")
REPORT_ROOT = require_report_output(
    ROOT / "results/diagnostics/mirepnet_t2s_h4_h5_no_student_ea_20261102/status.json").parent
RESOLVED = json.loads((ROOT / "results/distill/mirepnet_t2s_allsets_v1/resolved.json").read_text())
DATASETS = ("BNCI2014004", "BNCI2014001", "BNCI2014001-4", "BNCI2015001", "AlexMI")
SUBJECTS = dict(zip(DATASETS, (9, 9, 9, 12, 8)))
STUDENTS = ("ifnet", "eegnet", "adfcnn")
SEEDS = (666, 667, 668)
METHODS = ("none", "CE_T2S_FEATURE", "MCC_T2S_FEATURE", "CE_T2S_DIFFKD", "MCC_T2S_DIFFKD")
TRANSFER = {
    "CE_T2S_FEATURE": ("Feature", "T_CE"),
    "MCC_T2S_FEATURE": ("Feature", "T_MCC"),
    "CE_T2S_DIFFKD": ("DiffKD", "T_CE"),
    "MCC_T2S_DIFFKD": ("DiffKD", "T_MCC"),
}
TEACHERS = ("T0", "T_CE", "T_MCC")
BOOTSTRAPS = 10_000
BOOT_SEED = 20261102
PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
PROBE_PROTOCOL = "StandardScaler + L2 logistic regression (liblinear, one-vs-rest, max_iter=2000, tol=1e-4); C selected by source-only GroupKFold"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value: dict) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_external_json(path: Path, value: dict) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
    ids = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    lo, hi = np.quantile(x[ids].mean(axis=1), [0.025, 0.975])
    return mean, float(lo), float(hi)


def cka_linear(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("linear CKA expects paired [N,D] feature matrices")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    return float(np.square(cross).sum() / denominator) if denominator > 0 else math.nan


def paired_cosine(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    labels = np.asarray(labels)
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    cosine = np.sum(x * y, axis=1) / np.maximum(x_norm * y_norm, 1e-12)
    return float(np.mean([cosine[labels == c].mean() for c in np.unique(labels)]))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def stage_paths(dataset: str, seed: int, fold: int) -> tuple[Path, Path, Path]:
    suffix = Path(EXPERIMENT) / dataset / f"seed_{seed}" / f"subject_{fold + 1:02d}"
    return (RESULTS_ROOT / "distill" / suffix,
            WEIGHTS_ROOT / suffix,
            DATA_CACHE_ROOT / suffix)


def load_complete(result_dir: Path, tag: str) -> dict:
    path = Path(resolve_local_file(result_dir / tag / "complete.json"))
    if not path.is_file():
        raise FileNotFoundError(f"missing completed stage manifest (no retrain): {path}")
    record = json.loads(path.read_text())
    if record.get("status") != "complete":
        raise ValueError(f"stage is not complete: {path}")
    checkpoint = Path(resolve_local_file(record["checkpoint"]))
    if not checkpoint.is_file() or sha_file(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint is missing or changed: {checkpoint}")
    return record


def load_model_state(model, complete: dict) -> None:
    checkpoint = Path(resolve_local_file(complete["checkpoint"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"unexpected checkpoint payload: {checkpoint}")
    model.load_state_dict(payload["model"], strict=True)
    actual = training.state_hash(model)
    if actual != complete.get("model_state_sha256"):
        raise ValueError(f"loaded state hash mismatch: {checkpoint}")
    model.eval()


def extract_arrays(adapter, model, x: torch.Tensor, context: str) -> dict:
    arrays = training.extract(adapter, model, x, batch_size=128, device="cpu",
                              forward_fn=core.forward_native)
    feature = np.asarray(arrays["features"], dtype=np.float32)
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    if len(feature) != len(x) or len(logits) != len(x):
        raise ValueError(f"{context}: extracted row count differs from input")
    if not np.isfinite(feature).all() or not np.isfinite(logits).all():
        raise FloatingPointError(f"{context}: non-finite feature/logit output")
    return {"features": feature, "logits": logits}


def validate_saved_logits(dataset: str, model: str, method: str, seed: int,
                          subject: int, target_indices: np.ndarray,
                          logits: np.ndarray, record_index: dict) -> np.ndarray:
    key = (dataset, model.lower(), method, int(seed), int(subject))
    row = record_index.get(key)
    if row is None:
        raise ValueError(f"existing target-logit registry lacks {key}")
    with np.load(Path(resolve_local_file(row["prediction_path"])), allow_pickle=False) as saved:
        old = np.asarray(saved["logits"], dtype=np.float32)
    new = np.asarray(logits[target_indices], dtype=np.float32)
    if old.shape != new.shape:
        raise RuntimeError(f"stored and extracted logits have different shapes: {key}")
    # Existing predictions were produced on GPU; the required no-GPU feature
    # pass runs on CPU, so compare the actual decisions and retain the exact
    # archived target logits for head metrics. Record numeric drift separately.
    if not np.array_equal(old.argmax(axis=1), new.argmax(axis=1)):
        mismatch = int(np.count_nonzero(old.argmax(axis=1) != new.argmax(axis=1)))
        max_diff = float(np.max(np.abs(old - new)))
        raise RuntimeError(f"CPU extraction changes archived target predictions: {key}; "
                           f"mismatches={mismatch}/{len(old)}, max_diff={max_diff}")
    return old


def load_record_index() -> dict:
    index = {}
    for dataset in DATASETS:
        path = ROOT / "results/distill/mirepnet_t2s_allsets_v1" / f"{dataset}_records.json"
        for row in json.loads(path.read_text()):
            key = (dataset, row["student"].lower(), row["method"],
                   int(row["seed"]), int(row["subject"]))
            if key in index:
                raise ValueError(f"duplicate record key: {key}")
            index[key] = row
    return index


def fold_feature_path(dataset: str, seed: int, fold: int) -> Path:
    return Path(require_external_output(
        ARTIFACT_ROOT / dataset / f"seed_{seed}" / f"subject_{fold + 1:02d}" / "features.npz"))


def build_fold_features(dataset: str, seed: int, fold: int, record_index: dict,
                        resolved_identity: str) -> dict:
    output = fold_feature_path(dataset, seed, fold)
    meta_path = output.with_suffix(".json")
    # Read all canonical inputs; auto_prepare=False forbids cache creation.
    loaded = {student: source.load_inputs(dataset, student, auto_prepare=False)
              for student in STUDENTS}
    teacher_data = loaded["ifnet"][0]
    for student in STUDENTS:
        td, sd = loaded[student]
        teacher_data, student_data = source.ensure_uid_alignment(td, sd)
        if not np.array_equal(teacher_data.trial_uid, loaded["ifnet"][1].trial_uid):
            raise ValueError(f"{dataset}: subject UID order differs for {student}")
    student_data_by_name = {student: pair[1] for student, pair in loaded.items()}
    n_subjects = SUBJECTS[dataset]
    fold = int(fold)
    train_idx = np.flatnonzero(teacher_data.subjects != fold)
    test_idx = np.flatnonzero(teacher_data.subjects == fold)
    if not len(train_idx) or not len(test_idx):
        raise ValueError(f"empty source/test split: {dataset}/{seed}/{fold}")
    if set(teacher_data.trial_uid[train_idx]) & set(teacher_data.trial_uid[test_idx]):
        raise ValueError("source and target UID sets overlap")

    input_hashes = {
        "source_manifest_sha256": teacher_data.source_manifest_sha256,
        "teacher_input_manifest_sha256": teacher_data.input_manifest_sha256,
        "student_input_manifest_sha256": {
            student: student_data_by_name[student].input_manifest_sha256
            for student in STUDENTS},
        "train_uid_sha256": json_hash(teacher_data.trial_uid[train_idx].astype(str).tolist()),
        "test_uid_sha256": json_hash(teacher_data.trial_uid[test_idx].astype(str).tolist()),
    }
    result_dir, _, cache_dir = stage_paths(dataset, seed, fold)
    stage_sha = {}
    for name in ("T_CE", "T_MCC"):
        complete = load_complete(result_dir, name)
        stage_sha[name] = complete["checkpoint_sha256"]
    for student in STUDENTS:
        for method in METHODS:
            complete = load_complete(result_dir, f"{student}/{method}")
            stage_sha[f"{student}/{method}"] = complete["checkpoint_sha256"]
        for teacher_name in ("T_CE", "T_MCC"):
            prior = load_complete(result_dir, f"{student}/prior_{teacher_name}")
            stage_sha[f"{student}/prior_{teacher_name}"] = prior["checkpoint_sha256"]
        ppath = Path(resolve_local_file(cache_dir / student / "P.npz"))
        if not ppath.is_file():
            raise FileNotFoundError(f"fixed projection is missing; no refit: {ppath}")
        stage_sha[f"{student}/P"] = sha_file(ppath)
    fingerprint = json_hash({"resolved": resolved_identity, "dataset": dataset,
                             "seed": seed, "fold": fold, "inputs": input_hashes,
                             "checkpoints": stage_sha,
                             "student_EA": False,
                             "diagnostic": "read-only native-feature dump"})
    if output.exists() or meta_path.exists():
        if not (output.is_file() and meta_path.is_file()):
            raise RuntimeError(f"partial feature cache exists: {output}")
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint or meta.get("file_sha256") != sha_file(output):
            raise RuntimeError(f"existing feature cache identity/hash differs: {output}")
        with np.load(output, allow_pickle=False) as saved:
            return {name: np.asarray(saved[name]) for name in saved.files}

    arrays: dict[str, np.ndarray] = {
        "trial_uid": np.asarray(teacher_data.trial_uid).astype(str),
        "subjects": np.asarray(teacher_data.subjects, dtype=np.int64),
        "y": np.asarray(teacher_data.y, dtype=np.int64),
        "train_indices": train_idx.astype(np.int64),
        "test_indices": test_idx.astype(np.int64),
    }
    n_classes = len(teacher_data.class_mapping)
    # All three MIRepNet teachers use the same verified fold-specific input.
    teacher_record = source.teacher_record(dataset, seed, fold, teacher_data)
    t_adapter, t0 = source.build_teacher(teacher_record, teacher_data, "cpu")
    all_teacher_models = {"T0": t0}
    for name in ("T_CE", "T_MCC"):
        model = __import__("copy").deepcopy(t0)
        load_model_state(model, load_complete(result_dir, name))
        all_teacher_models[name] = model
    teacher_raw = {}
    for name, model in all_teacher_models.items():
        extracted = extract_arrays(t_adapter, model,
            torch.as_tensor(np.array(teacher_data.x, dtype=np.float32, copy=True)),
            f"{dataset}/{seed}/{fold}/teacher/{name}")
        if extracted["features"].shape[1] != 256:
            raise ValueError(f"MIRepNet feature dimension changed: {extracted['features'].shape}")
        archived_target_logits = validate_saved_logits(dataset, "MIRepNet", name,
            seed, fold + 1, test_idx, extracted["logits"], record_index)
        extracted["logits"][test_idx] = archived_target_logits
        arrays[f"teacher_feature_{name}"] = extracted["features"]
        arrays[f"teacher_logits_{name}"] = extracted["logits"]
        teacher_raw[name] = extracted["features"]

    student_raw: dict[tuple[str, str], np.ndarray] = {}
    student_logits: dict[tuple[str, str], np.ndarray] = {}
    student_native_ln: dict[tuple[str, str], np.ndarray] = {}
    for student in STUDENTS:
        data = student_data_by_name[student]
        cfg = dict(RESOLVED["students"][dataset][student])
        adapter = get_adapter(student, device="cpu", **cfg)
        model_inputs = adapter.preprocess(np.array(data.x, dtype=np.float32, copy=True))
        if not torch.is_tensor(model_inputs):
            model_inputs = torch.as_tensor(model_inputs, dtype=torch.float32)
        else:
            model_inputs = model_inputs.detach().cpu().to(torch.float32)
        if len(model_inputs) != len(data.y):
            raise ValueError(f"{dataset}/{student}: preprocessed row count changed")
        for method in METHODS:
            complete = load_complete(result_dir, f"{student}/{method}")
            model = adapter.build(n_classes)
            load_model_state(model, complete)
            extracted = extract_arrays(adapter, model, model_inputs,
                f"{dataset}/{seed}/{fold}/{student}/{method}")
            archived_target_logits = validate_saved_logits(dataset, student, method,
                seed, fold + 1, test_idx, extracted["logits"], record_index)
            extracted["logits"][test_idx] = archived_target_logits
            key = (student, method)
            student_raw[key] = extracted["features"]
            student_logits[key] = extracted["logits"]
            with torch.no_grad():
                normalized = torch.nn.functional.layer_norm(
                    torch.as_tensor(extracted["features"], dtype=torch.float32),
                    (extracted["features"].shape[1],), weight=None, bias=None, eps=1e-5)
            student_native_ln[key] = normalized.numpy()
            arrays[f"student_feature_{student}_{method}"] = extracted["features"]
            arrays[f"student_logits_{student}_{method}"] = extracted["logits"]

        ppath = Path(resolve_local_file(cache_dir / student / "P.npz"))
        with np.load(ppath, allow_pickle=False) as saved:
            projection = np.asarray(saved["P_float64"], dtype=np.float64)
        core.projection_properties(projection)
        arrays[f"projection_{student}"] = projection.astype(np.float32)
        projected_teacher: dict[str, np.ndarray] = {}
        for teacher_name in ("T_CE", "T_MCC"):
            _, normalized_teacher = core.project_teacher_features(
                teacher_raw[teacher_name], projection, device="cpu", batch_size=128)
            projected_teacher[teacher_name] = normalized_teacher
            arrays[f"projected_teacher_{student}_{teacher_name}"] = normalized_teacher

        # One independent, deterministic diagnostic noise tensor is reused for
        # both teacher variants and never enters any training RNG stream.
        noise_seed = core.derive_seed("diagnostic_noise", seed, fold, dataset, student)
        noise_generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        dimensions = int(student_raw[(student, "none")].shape[1])
        all_noise = torch.randn((len(data.y), dimensions), generator=noise_generator,
                                dtype=torch.float32)
        _, alpha, sigma = ddpm_schedule(1000, 1e-4, 0.02, torch.device("cpu"))
        for teacher_name, method in (("T_CE", "CE_T2S_DIFFKD"),
                                     ("T_MCC", "MCC_T2S_DIFFKD")):
            prior_complete = load_complete(result_dir, f"{student}/prior_{teacher_name}")
            predictor = EpsilonPredictor(dimensions, 128, 32)
            load_model_state(predictor, prior_complete)
            predicted = []
            source_features = torch.as_tensor(student_raw[(student, method)], dtype=torch.float32)
            with torch.no_grad():
                for start in range(0, len(source_features), 128):
                    stop = min(start + 128, len(source_features))
                    predicted.append(ddim_denoise(source_features[start:stop],
                        all_noise[start:stop], predictor, alpha, sigma))
            denoised = torch.cat(predicted, dim=0).cpu().numpy().astype(np.float32)
            arrays[f"student_denoised_{student}_{teacher_name}"] = denoised

    arrays["labels_class_mapping_json"] = np.asarray(
        json.dumps(teacher_data.class_mapping, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, output)
    atomic_external_json(meta_path, {
        "status": "complete", "fingerprint": fingerprint,
        "file_sha256": sha_file(output), "dataset": dataset, "seed": seed,
        "target_subject_0based": fold, "n_source_subjects": n_subjects - 1,
        "n_source_trials": int(len(train_idx)), "n_target_trials": int(len(test_idx)),
        "train_uid_sha256": input_hashes["train_uid_sha256"],
        "test_uid_sha256": input_hashes["test_uid_sha256"],
        "student_EA": False,
        "EA_setting": "no student EA; transductive teacher EA",
        "target_labels_used_for_fitting": False,
        "feature_dimension": {student: int(student_raw[(student, "none")].shape[1])
                              for student in STUDENTS},
        "teachers": stage_sha,
    })
    return arrays


def diagnostic_metrics(student_native: np.ndarray, student_match: np.ndarray,
                       teacher_raw: np.ndarray, teacher_projected: np.ndarray,
                       labels: np.ndarray) -> dict:
    return {
        "match_mse": float(np.mean((student_match - teacher_projected) ** 2)),
        "cka_match_vs_projected": cka_linear(student_match, teacher_projected),
        "cka_native_vs_raw_teacher": cka_linear(student_native, teacher_raw),
        "cka_match_vs_raw_teacher": cka_linear(student_match, teacher_raw),
        "cosine_match_vs_projected": paired_cosine(student_match, teacher_projected, labels),
    }


def logged_feature_mse(dataset: str, seed: int, fold: int, student: str, method: str) -> float:
    result_dir, _, _ = stage_paths(dataset, seed, fold)
    path = Path(resolve_local_file(result_dir / student / method / "history.csv"))
    if not path.is_file():
        raise FileNotFoundError(f"missing student history: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(rows[-1].get("epoch", -1)) != 100:
        raise ValueError(f"student history is not a complete 100 epoch run: {path}")
    return float(rows[-1].get("feature_mse", 0.0))


def analyze_h4(cache_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    per_fold: list[dict] = []
    for cache_row in cache_rows:
        dataset, seed, fold = cache_row["dataset"], cache_row["seed"], cache_row["fold"]
        with np.load(cache_row["path"], allow_pickle=False) as saved:
            a = {name: np.asarray(saved[name]) for name in saved.files}
        train_idx, test_idx = a["train_indices"], a["test_indices"]
        for student in STUDENTS:
            for method, (family, teacher_name) in TRANSFER.items():
                baseline_native = a[f"student_feature_{student}_none"]
                kd_native = a[f"student_feature_{student}_{method}"]
                baseline_match = a[f"student_feature_{student}_none"]
                # Baseline receives the same fixed no-affine LayerNorm as its
                # diagnostic matching branch; the feature-trained student uses
                # that same branch, while DiffKD adds the frozen DDIM map.
                baseline_match = torch.nn.functional.layer_norm(
                    torch.as_tensor(baseline_match, dtype=torch.float32),
                    (baseline_match.shape[1],), weight=None, bias=None, eps=1e-5).numpy()
                kd_match = (a[f"student_denoised_{student}_{teacher_name}"]
                            if family == "DiffKD" else torch.nn.functional.layer_norm(
                                torch.as_tensor(kd_native, dtype=torch.float32),
                                (kd_native.shape[1],), weight=None, bias=None,
                                eps=1e-5).numpy())
                teacher_raw = a[f"teacher_feature_{teacher_name}"]
                teacher_projected = a[f"projected_teacher_{student}_{teacher_name}"]
                for split_name, indices in (("train", train_idx), ("target", test_idx)):
                    labels = a["y"][indices]
                    base = diagnostic_metrics(baseline_native[indices], baseline_match[indices],
                        teacher_raw[indices], teacher_projected[indices], labels)
                    kd = diagnostic_metrics(kd_native[indices], kd_match[indices],
                        teacher_raw[indices], teacher_projected[indices], labels)
                    row = {"dataset": dataset, "student": student, "method": method,
                           "family": family, "teacher": teacher_name,
                           "seed": seed, "target_subject": fold + 1,
                           "split": split_name, "n_samples": len(indices),
                           "n_seed": 1, "student_EA": False,
                           "EA_setting": "no student EA; transductive teacher EA",
                           "matching_branch": ("student LayerNorm then fixed-noise 5-step DDIM"
                               if family == "DiffKD" else "student no-affine LayerNorm"),
                           "diagnostic_DiffKD_noise": "fixed seeded independent noise; not training RNG"}
                    for key, value in kd.items():
                        row[f"KD_{key}"] = value
                        row[f"CE_{key}"] = base[key]
                        row[f"delta_KD_minus_CE_{key}"] = value - base[key]
                    row["train_logged_feature_mse_epoch100"] = (
                        logged_feature_mse(dataset, seed, fold, student, method)
                        if split_name == "train" else "")
                    per_fold.append(row)

    # Average the three seeds within each target subject before resampling subjects.
    group_keys = ("dataset", "student", "method", "family", "teacher", "split")
    by_cell: dict[tuple, dict[int, list[dict]]] = {}
    for row in per_fold:
        key = tuple(row[k] for k in group_keys)
        by_cell.setdefault(key, {}).setdefault(int(row["target_subject"]), []).append(row)
    fold_mean_rows = []
    numeric = [k for k in per_fold[0] if k.startswith(("KD_", "CE_", "delta_"))]
    for key, subjects in by_cell.items():
        for subject, rows in subjects.items():
            result = dict(zip(group_keys, key))
            result.update(target_subject=subject, n_seed=len(rows), student_EA=False,
                          EA_setting="no student EA; transductive teacher EA")
            for col in numeric:
                result[col] = float(np.mean([float(row[col]) for row in rows]))
            fold_mean_rows.append(result)
    rng = np.random.default_rng(BOOT_SEED)
    summaries = []
    for key, subjects in by_cell.items():
        rows = [r for r in fold_mean_rows if tuple(r[k] for k in group_keys) == key]
        summary = dict(zip(group_keys, key))
        used_seeds = {int(r["seed"]) for r in per_fold
                      if tuple(r[k] for k in group_keys) == key}
        summary.update(n_subject=len(rows), n_seed=len(used_seeds),
                       bootstrap_unit="LOSO subject after seed averaging",
                       bootstrap_replicates=BOOTSTRAPS,
                       EA_setting="no student EA; transductive teacher EA")
        for col in numeric:
            vals = np.asarray([r[col] for r in rows], dtype=np.float64)
            mean, low, high = bootstrap_mean(vals, rng)
            summary[f"{col}_mean"] = mean
            summary[f"{col}_95ci_low"] = low
            summary[f"{col}_95ci_high"] = high
        summaries.append(summary)

    # Correlate target alignment change with the exact H1 BA gain, subject by subject.
    h1_path = ROOT / "results/diagnostics/mirepnet_t2s_h1_h6_no_student_ea_20261102/H1_fold_points.csv"
    h1 = list(csv.DictReader(h1_path.open(encoding="utf-8-sig")))
    h1_by = {(r["dataset"], r["student"], r["transfer"], int(r["subject"])):
             float(r["KD_gain_pp"]) for r in h1}
    correlations = []
    for key, subjects in by_cell.items():
        dataset, student, method, family, teacher, split = key
        if split != "target":
            continue
        delta_metric = "delta_KD_minus_CE_cka_match_vs_projected"
        aligned = [r for r in fold_mean_rows if tuple(r[k] for k in group_keys) == key]
        x = np.asarray([float(r[delta_metric]) for r in aligned])
        y = np.asarray([h1_by[(dataset, student, method, int(r["target_subject"]))]
                        for r in aligned])
        from scipy.stats import pearsonr, spearmanr
        pr = float(pearsonr(x, y).statistic) if len(x) > 2 and np.std(x) and np.std(y) else math.nan
        sr = float(spearmanr(x, y).statistic) if len(x) > 2 and np.std(x) and np.std(y) else math.nan
        correlations.append({"dataset": dataset, "student": student, "method": method,
            "family": family, "teacher": teacher, "n_subject": len(x), "n_seed": len(SEEDS),
            "target_delta_CKA_vs_BA_gain_pearson_r": pr,
            "target_delta_CKA_vs_BA_gain_spearman_rho": sr,
            "alignment_delta": delta_metric,
            "EA_setting": "no student EA; transductive teacher EA"})
    return per_fold, summaries, correlations


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
              x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("linear probe needs at least two source subjects for GroupKFold")
    splitter = GroupKFold(n_splits=n_splits)
    scores = []
    for c in PROBE_C_GRID:
        fold_scores = []
        for train_part, valid_part in splitter.split(x_train, y_train, groups):
            if len(np.unique(y_train[train_part])) != len(np.unique(y_train)):
                raise ValueError("source-only probe CV fold is missing a class")
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=float(c), penalty="l2", solver="liblinear", multi_class="ovr",
                max_iter=2000, tol=1e-4, random_state=0))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train[train_part], y_train[train_part])
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError(f"linear probe did not converge: C={c}, CV fold")
            fold_scores.append(balanced_accuracy(y_train[valid_part],
                model.predict(x_train[valid_part])))
        scores.append((float(np.mean(fold_scores)), float(c)))
    # Deterministic tie-break: select the stronger L2 penalty (smaller C).
    best_score = max(score for score, _ in scores)
    best_c = min(c for score, c in scores if np.isclose(score, best_score, atol=1e-12))
    final = make_pipeline(StandardScaler(), LogisticRegression(
        C=best_c, penalty="l2", solver="liblinear", multi_class="ovr",
        max_iter=2000, tol=1e-4, random_state=0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final.fit(x_train, y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"final linear probe did not converge: C={best_c}")
    return balanced_accuracy(y_test, final.predict(x_test)), float(best_c)


def analyze_h5(cache_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    per_fold = []
    for cache_row in cache_rows:
        dataset, seed, fold = cache_row["dataset"], cache_row["seed"], cache_row["fold"]
        with np.load(cache_row["path"], allow_pickle=False) as saved:
            a = {name: np.asarray(saved[name]) for name in saved.files}
        train_idx, test_idx = a["train_indices"], a["test_indices"]
        y, subjects = a["y"], a["subjects"]
        source_y, target_y = y[train_idx], y[test_idx]
        source_groups = subjects[train_idx]
        teacher_probes = {}
        for teacher in TEACHERS:
            features = a[f"teacher_feature_{teacher}"]
            probe_ba, c = fit_probe(features[train_idx], source_y, source_groups,
                                    features[test_idx], target_y)
            head_ba = balanced_accuracy(target_y,
                a[f"teacher_logits_{teacher}"][test_idx].argmax(axis=1))
            teacher_probes[teacher] = (probe_ba, c)
            per_fold.append({"dataset": dataset, "student": "MIRepNet",
                "representation": teacher, "teacher": teacher, "method": "teacher_feature_probe",
                "seed": seed, "target_subject": fold + 1, "probe_BA_pct": 100 * probe_ba,
                "selected_C": c, "head_BA_pct": 100 * head_ba,
                "probe_minus_head_pp": 100 * (probe_ba - head_ba),
                "probe_inner_cv": "GroupKFold by source subject; source-only",
                "probe_protocol": PROBE_PROTOCOL,
                "target_labels_used_for_probe_fit_or_C_selection": False,
                "n_source_subjects": len(np.unique(source_groups)), "n_seed": 1,
                "student_EA": False,
                "EA_setting": "no student EA; transductive teacher EA"})
        for student in STUDENTS:
            for method in METHODS:
                features = a[f"student_feature_{student}_{method}"]
                probe_ba, c = fit_probe(features[train_idx], source_y, source_groups,
                                        features[test_idx], target_y)
                head_ba = balanced_accuracy(target_y,
                    a[f"student_logits_{student}_{method}"][test_idx].argmax(axis=1))
                per_fold.append({"dataset": dataset, "student": student,
                    "representation": method, "teacher": "",
                    "method": "CE_head" if method == "none" else method,
                    "seed": seed, "target_subject": fold + 1,
                    "probe_BA_pct": 100 * probe_ba, "selected_C": c,
                    "head_BA_pct": 100 * head_ba,
                    "probe_minus_head_pp": 100 * (probe_ba - head_ba),
                    "probe_inner_cv": "GroupKFold by source subject; source-only",
                    "probe_protocol": PROBE_PROTOCOL,
                    "target_labels_used_for_probe_fit_or_C_selection": False,
                    "n_source_subjects": len(np.unique(source_groups)), "n_seed": 1,
                    "student_EA": False,
                    "EA_setting": "no student EA; transductive teacher EA"})

    # Subject-level comparison means are built only after averaging the three seeds.
    value_fields = ("probe_BA_pct", "head_BA_pct", "probe_minus_head_pp")
    by_rep: dict[tuple, dict[int, list[dict]]] = {}
    for row in per_fold:
        key = (row["dataset"], row["student"], row["representation"], row["teacher"], row["method"])
        by_rep.setdefault(key, {}).setdefault(int(row["target_subject"]), []).append(row)
    subject_means = []
    for key, subjects in by_rep.items():
        for subject, rows in subjects.items():
            result = dict(zip(("dataset", "student", "representation", "teacher", "method"), key))
            result.update(target_subject=subject, n_seed=len(rows), student_EA=False,
                          EA_setting="no student EA; transductive teacher EA")
            for field in value_fields:
                result[field] = float(np.mean([float(r[field]) for r in rows]))
            subject_means.append(result)
    rng = np.random.default_rng(BOOT_SEED)
    summaries = []
    for key, subjects in by_rep.items():
        rows = [r for r in subject_means
                if tuple(r[k] for k in ("dataset", "student", "representation", "teacher", "method")) == key]
        summary = dict(zip(("dataset", "student", "representation", "teacher", "method"), key))
        used_seeds = {int(r["seed"]) for r in per_fold
                      if (r["dataset"], r["student"], r["representation"],
                          r["teacher"], r["method"]) == key}
        summary.update(n_subject=len(rows), n_seed=len(used_seeds),
            bootstrap_unit="LOSO subject after seed averaging", bootstrap_replicates=BOOTSTRAPS,
            probe_inner_cv="GroupKFold by source subject; source-only",
            probe_protocol=PROBE_PROTOCOL,
            EA_setting="no student EA; transductive teacher EA")
        for field in value_fields:
            mean, low, high = bootstrap_mean(np.asarray([float(r[field]) for r in rows]), rng)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_95ci_low"] = low
            summary[f"{field}_95ci_high"] = high
        summaries.append(summary)

    # Teacher-probe minus student-CE probe and KD-probe minus student-CE probe.
    by_lookup = {(r["dataset"], r["student"], r["representation"], r["target_subject"]): r
                 for r in subject_means}
    comparisons = []
    available_datasets = {row["dataset"] for row in cache_rows}
    for dataset in DATASETS:
        if dataset not in available_datasets:
            continue
        for student in STUDENTS:
            available_subjects = sorted(subject for subject in range(1, SUBJECTS[dataset] + 1)
                if all((dataset, model, representation, subject) in by_lookup
                    for model, representation in (("MIRepNet", "T0"),
                        ("MIRepNet", "T_CE"), ("MIRepNet", "T_MCC"),
                        (student, "none"), *( (student, method) for method in METHODS if method != "none"))))
            n_subject = len(available_subjects)
            if n_subject == 0:
                continue
            for teacher in TEACHERS:
                values = []
                for subject in available_subjects:
                    tr = by_lookup[(dataset, "MIRepNet", teacher, subject)]
                    ce = by_lookup[(dataset, student, "none", subject)]
                    values.append(float(tr["probe_BA_pct"]) - float(ce["probe_BA_pct"]))
                mean, low, high = bootstrap_mean(np.asarray(values), rng)
                comparisons.append({"dataset": dataset, "student": student,
                    "comparison": f"teacher_{teacher}_probe_minus_student_CE_probe",
                    "n_subject": n_subject,
                    "n_seed": len({int(r["seed"]) for r in per_fold if r["dataset"] == dataset}),
                    "mean_delta_pp": mean, "bootstrap95_low_pp": low,
                    "bootstrap95_high_pp": high,
                    "EA_setting": "no student EA; transductive teacher EA"})
            for method in METHODS:
                if method == "none":
                    continue
                values = []
                for subject in available_subjects:
                    kd = by_lookup[(dataset, student, method, subject)]
                    ce = by_lookup[(dataset, student, "none", subject)]
                    values.append(float(kd["probe_BA_pct"]) - float(ce["probe_BA_pct"]))
                mean, low, high = bootstrap_mean(np.asarray(values), rng)
                comparisons.append({"dataset": dataset, "student": student,
                    "comparison": f"{method}_probe_minus_student_CE_probe",
                    "n_subject": n_subject,
                    "n_seed": len({int(r["seed"]) for r in per_fold if r["dataset"] == dataset}),
                    "mean_delta_pp": mean, "bootstrap95_low_pp": low,
                    "bootstrap95_high_pp": high,
                    "EA_setting": "no student EA; transductive teacher EA"})
    return per_fold, summaries, comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "extract", "analyze"), default="all")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="optional debugging subset; omit for the requested complete scope")
    args = parser.parse_args()
    if RESOLVED.get("implementation_sha256") != __import__(
            "experiments.distill.run_mirepnet_t2s_loso", fromlist=["implementation"]).implementation():
        raise RuntimeError("current implementation hash differs from the completed experiment")
    resolved_identity = str(RESOLVED["identity"])
    record_index = load_record_index()
    tasks = [(dataset, seed, fold) for dataset in DATASETS for seed in SEEDS
             for fold in range(SUBJECTS[dataset])]
    if args.max_folds is not None:
        if args.max_folds < 1:
            raise ValueError("--max-folds must be positive")
        tasks = tasks[:args.max_folds]
    feature_rows = []
    status_path = REPORT_ROOT / "status.json"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for index, (dataset, seed, fold) in enumerate(tasks, 1):
        output = fold_feature_path(dataset, seed, fold)
        try:
            if args.phase in ("all", "extract"):
                data = build_fold_features(dataset, seed, fold, record_index, resolved_identity)
                del data
            else:
                meta_path = output.with_suffix(".json")
                if not output.is_file() or not meta_path.is_file():
                    raise FileNotFoundError(f"feature dump missing; run extract phase first: {output}")
                meta = json.loads(meta_path.read_text())
                if meta.get("status") != "complete" or meta.get("file_sha256") != sha_file(output):
                    raise RuntimeError(f"feature dump hash/status mismatch: {output}")
        except Exception as exc:
            atomic_json(status_path, {
                "status": "stopped_error", "phase": args.phase,
                "completed_seed_folds": index - 1, "total_seed_folds": len(tasks),
                "failed_task": {"dataset": dataset, "seed": seed,
                                "target_subject": fold + 1},
                "error_type": type(exc).__name__, "error": str(exc),
                "EA_setting": "no student EA; transductive teacher EA",
                "student_EA": False,
                "elapsed_seconds": time.monotonic() - start,
            })
            raise
        feature_rows.append({"dataset": dataset, "seed": seed, "fold": fold,
                             "path": output, "status": "complete"})
        atomic_json(status_path, {"status": "running", "phase": args.phase,
            "completed_seed_folds": index, "total_seed_folds": len(tasks),
            "current": {"dataset": dataset, "seed": seed, "target_subject": fold + 1},
            "student_EA": False, "EA_setting": "no student EA; transductive teacher EA",
            "elapsed_seconds": time.monotonic() - start})
        print(f"[feature-dump {index}/{len(tasks)}] {dataset} seed={seed} target={fold + 1}",
              flush=True)

    if args.phase == "extract":
        atomic_json(status_path, {"status": "complete", "phase": "extract",
            "completed_seed_folds": len(tasks), "total_seed_folds": len(tasks),
            "student_EA": False,
            "EA_setting": "no student EA; transductive teacher EA",
            "external_feature_root": str(ARTIFACT_ROOT),
            "elapsed_seconds": time.monotonic() - start})
        return

    # Analyze all available fold caches, not only the optional debugging subset.
    if args.max_folds is not None:
        cache_rows = feature_rows
    else:
        cache_rows = [{"dataset": ds, "seed": seed, "fold": fold,
                       "path": fold_feature_path(ds, seed, fold)}
                      for ds in DATASETS for seed in SEEDS for fold in range(SUBJECTS[ds])]
    h4_fold, h4_summary, h4_corr = analyze_h4(cache_rows)
    h5_fold, h5_summary, h5_comparison = analyze_h5(cache_rows)
    write_csv(REPORT_ROOT / "H4_fold_metrics_seed_level.csv", h4_fold)
    write_csv(REPORT_ROOT / "H4_summary_subject_bootstrap.csv", h4_summary)
    write_csv(REPORT_ROOT / "H4_target_alignment_gain_correlations.csv", h4_corr)
    write_csv(REPORT_ROOT / "H5_probe_seed_fold_metrics.csv", h5_fold)
    write_csv(REPORT_ROOT / "H5_probe_summary_subject_bootstrap.csv", h5_summary)
    write_csv(REPORT_ROOT / "H5_paired_comparisons.csv", h5_comparison)
    atomic_json(status_path, {"status": "complete", "phase": args.phase,
        "completed_seed_folds": len(cache_rows), "total_seed_folds": len(cache_rows),
        "datasets": list(DATASETS), "students": list(STUDENTS), "seeds": list(SEEDS),
        "target_subject_folds": sum(SUBJECTS.values()),
        "seed_fold_units": sum(SUBJECTS.values()) * len(SEEDS),
        "feature_caches": len(cache_rows), "student_EA": False,
        "EA_setting": "no student EA; transductive teacher EA",
        "validation_subject": "not present in standard LOSO",
        "probe_C_grid": PROBE_C_GRID, "probe_C_selection": "source-only GroupKFold by subject",
        "probe_protocol": PROBE_PROTOCOL,
        "bootstrap": {"unit": "LOSO subject after seed averaging",
                      "replicates": BOOTSTRAPS, "seed": BOOT_SEED},
        "elapsed_seconds": time.monotonic() - start,
        "external_feature_root": str(ARTIFACT_ROOT)})
    print(json.dumps(json.loads(status_path.read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
