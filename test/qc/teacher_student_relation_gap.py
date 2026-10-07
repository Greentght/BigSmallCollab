"""Seed-specific MIRepNet/IFNet relation-gap diagnostic.

This module is deliberately independent from the distillation and fusion
experiments.  It consumes the already exported ``train`` artifacts, performs
sample-level checks and relation analysis, and optionally retrains IFNet after
UID-based masks have been applied *before* the DataLoader is built.

The important protocol detail is that every seed owns its split, ranking and
deletion mask.  No sample UID is aggregated across seeds and no common 60-trial
pool is constructed.

Examples
--------
Analysis (the default output is under /data1/llx/BigSmallCollab_results/qc_artifacts/relation_gap/...):

    python test/qc/teacher_student_relation_gap.py analyze

Formal IFNet deletion runs (20 random masks per seed):

    python test/qc/teacher_student_relation_gap.py retrain

For a quick CPU pilot, use ``--epochs 2 --n-random-masks 5``.  The formal
configuration is read from the existing model/dataset YAML files unless an
explicit smoke-only epoch override is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.storage import external_path, require_external_output, resolve_local_file

DEFAULT_DATASET = "BNCI2015001"
DEFAULT_SUBJECT = 0  # S1 in the repository's zero-based convention.
DEFAULT_SPLIT = "train"
DEFAULT_TEACHER = "mirepnet"
DEFAULT_STUDENT = "ifnet"
DEFAULT_SEEDS = (666, 667, 668)
DEFAULT_BAD_UID = (0, 1)
DEFAULT_N_RANDOM_MASKS = 20
DEFAULT_RANDOM_SEED = 20260907
DEFAULT_VAL_SPLIT = 0.7
N_EXPECTED_TRIALS = 60


def uid_key(uid: Sequence[int] | np.ndarray) -> tuple[int, int]:
    """Convert a UID row to the canonical immutable key."""
    a = np.asarray(uid, dtype=np.int64).reshape(-1)
    if len(a) != 2:
        raise ValueError(f"sample_uid rows must have length 2, got {a}")
    return int(a[0]), int(a[1])


def uid_text(uid: Sequence[int] | tuple[int, int]) -> str:
    a, b = uid_key(uid)
    return f"({a}, {b})"


def jsonable(value: Any) -> Any:
    """Make numpy/scalar values safe for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    path = external_path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def default_output_root(dataset: str, subject: int, protocol: str = "fewshot") -> Path:
    return (Path('/data1/llx/BigSmallCollab_results/qc_artifacts') / "relation_gap" / "teacher_student_relation_gap"
            / dataset / f"S{int(subject) + 1}" / protocol)


def artifact_file(root: Path, dataset: str, model: str, subject: int,
                  seed: int, split: str) -> Path:
    return external_path(root) / dataset / model / f"{subject}_{seed}_{split}.npz"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    path = resolve_local_file(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: Sequence[str]) -> str:
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                           check=False)
    except OSError as exc:
        return f"<unavailable: {exc}>"
    text = (p.stdout or p.stderr or "").strip()
    return text


def git_provenance() -> dict[str, Any]:
    return {
        "commit_sha": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "status_short": run_text(["git", "status", "--short"]).splitlines(),
    }


def load_npz_artifact(path: Path, expected_n: int = N_EXPECTED_TRIALS) -> dict[str, Any]:
    path = external_path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    with np.load(resolve_local_file(path), allow_pickle=False) as z:
        required = {"logits", "feats", "y", "sample_uid"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        out = {key: np.asarray(z[key]) for key in required}
        if "split_policy" in z.files:
            out["split_policy"] = str(z["split_policy"].item())
    uid = np.asarray(out["sample_uid"], dtype=np.int64)
    if uid.ndim != 2 or uid.shape[1] != 2:
        raise ValueError(f"{path}: sample_uid has shape {uid.shape}, expected (N,2)")
    n = len(uid)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{path}: expected exactly {expected_n} trials, got {n}")
    if len(set(map(uid_key, uid))) != n:
        raise ValueError(f"{path}: sample_uid is not unique")
    for key in ("logits", "feats", "y"):
        if len(out[key]) != n:
            raise ValueError(f"{path}: {key} length {len(out[key])} != {n}")
        if not np.all(np.isfinite(out[key])):
            raise ValueError(f"{path}: {key} contains non-finite values")
    if out["feats"].ndim != 2:
        raise ValueError(f"{path}: feats must be (N,D), got {out['feats'].shape}")
    if out["logits"].ndim != 2:
        raise ValueError(f"{path}: logits must be (N,C), got {out['logits'].shape}")
    return out


def align_by_uid(reference_uid: np.ndarray, other_uid: np.ndarray,
                 other_arrays: Mapping[str, np.ndarray] | None = None
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align arrays to reference UID order, rejecting duplicate/set mismatches."""
    ref = [uid_key(x) for x in np.asarray(reference_uid)]
    oth = [uid_key(x) for x in np.asarray(other_uid)]
    if len(set(ref)) != len(ref):
        raise ValueError("reference sample_uid contains duplicates")
    if len(set(oth)) != len(oth):
        raise ValueError("other sample_uid contains duplicates")
    missing = sorted(set(ref) - set(oth))
    extra = sorted(set(oth) - set(ref))
    if missing or extra:
        raise ValueError(f"UID set mismatch: missing={missing}, extra={extra}")
    positions = {key: i for i, key in enumerate(oth)}
    order = np.asarray([positions[key] for key in ref], dtype=np.int64)
    aligned = {}
    for name, array in (other_arrays or {}).items():
        aligned[name] = np.asarray(array)[order]
    return np.asarray(reference_uid, dtype=np.int64), aligned


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x)
    p /= np.sum(p, axis=1, keepdims=True)
    if not np.all(np.isfinite(p)):
        raise ValueError("softmax produced non-finite probabilities")
    return p.astype(np.float32)


def relation_matrix(features: np.ndarray, name: str = "features") -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized features and the complete cosine relation matrix."""
    h = np.asarray(features, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(h, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        bad = np.where((norms <= 0.0) | ~np.isfinite(norms))[0].tolist()
        raise ValueError(f"{name} has zero/non-finite L2 norm at rows {bad}")
    normalized = h / norms[:, None]
    relation = normalized @ normalized.T
    relation = (relation + relation.T) / 2.0
    if relation.shape[0] != relation.shape[1]:
        raise ValueError(f"{name} relation matrix is not square: {relation.shape}")
    if not np.all(np.isfinite(relation)):
        raise ValueError(f"{name} relation matrix contains non-finite values")
    symmetry_error = float(np.max(np.abs(relation - relation.T)))
    diag_error = float(np.max(np.abs(np.diag(relation) - 1.0)))
    if symmetry_error > 1e-7:
        raise ValueError(f"{name} relation matrix is not symmetric: {symmetry_error}")
    if diag_error > 1e-5:
        raise ValueError(f"{name} relation diagonal is not ~1: {diag_error}")
    return normalized.astype(np.float32), relation.astype(np.float32)


def pearson_distance(x: np.ndarray, y: np.ndarray,
                     min_count: int = 3) -> tuple[float, str]:
    """Compute 1-Pearson with an explicit status for degenerate rows."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"Pearson inputs have different lengths: {len(x)} vs {len(y)}")
    if len(x) < min_count:
        return np.nan, "insufficient_n"
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan, "nonfinite"
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return np.nan, "zero_variance"
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return np.nan, "nan_correlation"
    return float(1.0 - np.clip(corr, -1.0, 1.0)), "ok"


def relation_mae(x: np.ndarray, y: np.ndarray,
                 min_count: int = 1) -> tuple[float, str]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"MAE inputs have different lengths: {len(x)} vs {len(y)}")
    if len(x) < min_count:
        return np.nan, "insufficient_n"
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan, "nonfinite"
    return float(np.mean(np.abs(x - y))), "ok"


def prediction_group(teacher_correct: bool, student_correct: bool) -> str:
    if teacher_correct and student_correct:
        return "T_correct_S_correct"
    if teacher_correct and not student_correct:
        return "T_correct_S_wrong"
    if not teacher_correct and student_correct:
        return "T_wrong_S_correct"
    return "T_wrong_S_wrong"


def compute_sample_rows(seed: int, teacher: Mapping[str, Any],
                        student: Mapping[str, Any], dataset: str,
                        subject: int, split: str,
                        bad_uid: tuple[int, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align two artifacts and calculate all per-trial diagnostics for one seed."""
    ref_uid = np.asarray(teacher["sample_uid"], dtype=np.int64)
    uid, aligned = align_by_uid(
        ref_uid, student["sample_uid"],
        {"logits": student["logits"], "feats": student["feats"],
         "y": student["y"]})
    y_t = np.asarray(teacher["y"], dtype=np.int64)
    y_s = np.asarray(aligned["y"], dtype=np.int64)
    if not np.array_equal(y_t, y_s):
        diff = np.where(y_t != y_s)[0].tolist()
        raise ValueError(f"label mismatch after UID alignment for seed {seed}: rows {diff}")
    if len(uid) != N_EXPECTED_TRIALS:
        raise ValueError(f"seed {seed}: expected {N_EXPECTED_TRIALS} aligned rows, got {len(uid)}")

    teacher_features, r_t = relation_matrix(teacher["feats"], "teacher features")
    student_features, r_s = relation_matrix(aligned["feats"], "student features")
    if r_t.shape != (N_EXPECTED_TRIALS, N_EXPECTED_TRIALS):
        raise ValueError(f"teacher relation shape is {r_t.shape}, expected (60,60)")
    if r_s.shape != (N_EXPECTED_TRIALS, N_EXPECTED_TRIALS):
        raise ValueError(f"student relation shape is {r_s.shape}, expected (60,60)")

    p_t = softmax_np(teacher["logits"])
    p_s = softmax_np(aligned["logits"])
    pred_t = np.argmax(p_t, axis=1).astype(np.int64)
    pred_s = np.argmax(p_s, axis=1).astype(np.int64)
    correct_t = pred_t == y_t
    correct_s = pred_s == y_t
    rel_diff = np.abs(r_t - r_s)
    rows: list[dict[str, Any]] = []
    for i in range(len(uid)):
        same = (y_t == y_t[i])
        same[i] = False
        inter = ~same
        all_mask = np.ones(len(uid), dtype=bool)
        all_mask[i] = False
        d_all, st_all = pearson_distance(r_t[i, all_mask], r_s[i, all_mask])
        d_intra, st_intra = pearson_distance(r_t[i, same], r_s[i, same])
        d_inter, st_inter = pearson_distance(r_t[i, inter], r_s[i, inter])
        mae_all, mae_st_all = relation_mae(r_t[i, all_mask], r_s[i, all_mask])
        mae_intra, mae_st_intra = relation_mae(r_t[i, same], r_s[i, same])
        mae_inter, mae_st_inter = relation_mae(r_t[i, inter], r_s[i, inter])
        key = uid_key(uid[i])
        row = {
            "dataset": dataset,
            "subject": f"S{int(subject) + 1}",
            "subject_index": int(subject),
            "split": split,
            "seed": int(seed),
            "sample_uid": uid_text(key),
            "uid_subject": key[0],
            "uid_trial": key[1],
            "label": int(y_t[i]),
            "teacher_pred": int(pred_t[i]),
            "student_pred": int(pred_s[i]),
            "teacher_p_true": float(p_t[i, y_t[i]]),
            "student_p_true": float(p_s[i, y_t[i]]),
            "teacher_correct": bool(correct_t[i]),
            "student_correct": bool(correct_s[i]),
            "prediction_group": prediction_group(bool(correct_t[i]), bool(correct_s[i])),
            "d_rel_all": d_all,
            "d_rel_intra": d_intra,
            "d_rel_inter": d_inter,
            "relation_mae": mae_all,
            "relation_mae_intra": mae_intra,
            "relation_mae_inter": mae_inter,
            "d_rel_all_status": st_all,
            "d_rel_intra_status": st_intra,
            "d_rel_inter_status": st_inter,
            "relation_mae_status": mae_st_all,
            "relation_mae_intra_status": mae_st_intra,
            "relation_mae_inter_status": mae_st_inter,
            "is_known_bad": key == bad_uid,
            "_uid_tuple": key,
            "_row_index": i,
        }
        rows.append(row)
    bad_diagnostics = [r for r in rows if r["is_known_bad"]]
    if not bad_diagnostics:
        bad_status = "not_in_train_split"
    else:
        bad_status = "in_train_split"

    if not all(np.isfinite(r["d_rel_all"]) for r in rows):
        bad = [(r["sample_uid"], r["d_rel_all_status"]) for r in rows
               if not np.isfinite(r["d_rel_all"])]
        raise ValueError(f"seed {seed}: d_rel_all cannot be ranked due to {bad}")
    order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]["d_rel_all"]),
                                                      rows[i]["_uid_tuple"]))
    for rank, idx in enumerate(order, start=1):
        rows[idx]["rank_in_seed"] = rank
        rows[idx]["rank_percentile"] = ((len(rows) - rank) / (len(rows) - 1)
                                         if len(rows) > 1 else 1.0)
    summary = {
        "seed": int(seed),
        "n": len(rows),
        "uid_order": [list(uid_key(x)) for x in uid],
        "teacher_feature_dim": int(teacher_features.shape[1]),
        "student_feature_dim": int(student_features.shape[1]),
        "teacher_relation_shape": list(r_t.shape),
        "student_relation_shape": list(r_s.shape),
        "relation_symmetry_error_teacher": float(np.max(np.abs(r_t - r_t.T))),
        "relation_symmetry_error_student": float(np.max(np.abs(r_s - r_s.T))),
        "relation_diag_error_teacher": float(np.max(np.abs(np.diag(r_t) - 1.0))),
        "relation_diag_error_student": float(np.max(np.abs(np.diag(r_s) - 1.0))),
        "known_bad_status": bad_status,
        "known_bad_rows": bad_diagnostics,
    }
    summary["_arrays"] = {
        "uid": uid,
        "y": y_t,
        "teacher_features": teacher_features,
        "student_features": student_features,
        "teacher_logits": np.asarray(teacher["logits"], dtype=np.float32),
        "student_logits": np.asarray(aligned["logits"], dtype=np.float32),
        "teacher_probs": p_t,
        "student_probs": p_s,
        "teacher_pred": pred_t,
        "student_pred": pred_s,
        "teacher_correct": correct_t,
        "student_correct": correct_s,
        "relation_teacher": r_t,
        "relation_student": r_s,
        "relation_abs_diff": rel_diff.astype(np.float32),
        "d_rel_all": np.asarray([r["d_rel_all"] for r in rows], dtype=np.float32),
        "d_rel_intra": np.asarray([r["d_rel_intra"] for r in rows], dtype=np.float32),
        "d_rel_inter": np.asarray([r["d_rel_inter"] for r in rows], dtype=np.float32),
        "relation_mae": np.asarray([r["relation_mae"] for r in rows], dtype=np.float32),
    }
    return rows, summary


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (bool, np.bool_)):
                    value = int(value)
                elif isinstance(value, (list, tuple, dict)):
                    value = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
                elif isinstance(value, (np.floating, float)) and not np.isfinite(value):
                    value = ""
                elif isinstance(value, np.generic):
                    value = value.item()
                clean[key] = value
            writer.writerow(clean)


PER_SAMPLE_FIELDS = [
    "dataset", "subject", "subject_index", "split", "seed", "sample_uid",
    "uid_subject", "uid_trial", "label", "teacher_pred", "student_pred",
    "teacher_p_true", "student_p_true", "teacher_correct", "student_correct",
    "prediction_group", "d_rel_all", "d_rel_intra", "d_rel_inter",
    "relation_mae", "relation_mae_intra", "relation_mae_inter",
    "d_rel_all_status", "d_rel_intra_status", "d_rel_inter_status",
    "relation_mae_status", "relation_mae_intra_status", "relation_mae_inter_status",
    "rank_in_seed", "rank_percentile", "is_known_bad",
]


def descriptive_stats(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {"n": int(len(x)), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "q1": float(np.quantile(x, .25)),
            "q3": float(np.quantile(x, .75))}


def group_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["prediction_group"]))].append(row)
    result = []
    for (seed, group), rr in sorted(grouped.items()):
        d = descriptive_stats([r["d_rel_all"] for r in rr])
        ranks = descriptive_stats([r["rank_percentile"] for r in rr])
        gaps = [float(r["teacher_p_true"]) - float(r["student_p_true"]) for r in rr]
        abs_gaps = [abs(x) for x in gaps]
        gap_stats = descriptive_stats(gaps)
        abs_gap_stats = descriptive_stats(abs_gaps)
        result.append({
            "dataset": rr[0]["dataset"], "subject": rr[0]["subject"],
            "split": rr[0]["split"], "seed": seed,
            "prediction_group": group, "n": d["n"],
            "d_rel_all_mean": d["mean"], "d_rel_all_median": d["median"],
            "d_rel_all_q1": d["q1"], "d_rel_all_q3": d["q3"],
            "rank_percentile_mean": ranks["mean"],
            "rank_percentile_median": ranks["median"],
            "teacher_student_p_true_gap_mean": gap_stats["mean"],
            "teacher_student_p_true_gap_median": gap_stats["median"],
            "abs_p_true_gap_mean": abs_gap_stats["mean"],
            "abs_p_true_gap_median": abs_gap_stats["median"],
        })
    return result


AGG_FIELDS = [
    "dataset", "subject", "split", "seed", "prediction_group", "n",
    "d_rel_all_mean", "d_rel_all_median", "d_rel_all_q1", "d_rel_all_q3",
    "rank_percentile_mean", "rank_percentile_median",
    "teacher_student_p_true_gap_mean", "teacher_student_p_true_gap_median",
    "abs_p_true_gap_mean", "abs_p_true_gap_median",
]


def class_counts(labels: Sequence[int]) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(Counter(map(int, labels)).items())}


def choose_random_mask(uid_keys: Sequence[tuple[int, int]], labels: np.ndarray,
                       excluded: set[tuple[int, int]], target_labels: Sequence[int],
                       rng: np.random.Generator) -> list[tuple[int, int]]:
    available: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, label in zip(uid_keys, labels):
        if key not in excluded:
            available[int(label)].append(key)
    need = Counter(map(int, target_labels))
    chosen: list[tuple[int, int]] = []
    for label, count in sorted(need.items()):
        pool = available.get(label, [])
        if len(pool) < count:
            raise ValueError(f"not enough candidates for class {label}: need {count}, got {len(pool)}")
        indices = rng.choice(len(pool), size=count, replace=False)
        chosen.extend(pool[int(i)] for i in np.asarray(indices).reshape(-1))
    return sorted(chosen)


def make_seed_selection(rows: Sequence[Mapping[str, Any]], seed: int,
                        bad_uid: tuple[int, int], n_random_masks: int,
                        random_seed: int) -> dict[str, Any]:
    rr = [r for r in rows if int(r["seed"]) == int(seed)]
    if len(rr) != N_EXPECTED_TRIALS:
        raise ValueError(f"seed {seed}: selection received {len(rr)} rows")
    rr = sorted(rr, key=lambda r: (-float(r["d_rel_all"]), r["_uid_tuple"]))
    top1 = rr[:1]
    top3 = rr[:3]
    uid_keys = [r["_uid_tuple"] for r in rr]
    by_uid = {r["_uid_tuple"]: r for r in rr}
    top3_set = set(uid_keys[:3])
    selections: dict[str, Any] = {}

    def add_selection(name: str, selected: Sequence[Mapping[str, Any] | tuple[int, int]],
                      reason: str, mask_id: str | None = None) -> None:
        keys = [x["_uid_tuple"] if isinstance(x, Mapping) else uid_key(x)
                for x in selected]
        labels = [int(by_uid[k]["label"]) for k in keys]
        selections[name] = {
            "condition": name,
            "mask_id": mask_id,
            "removed_uids": [list(k) for k in keys],
            "removed_uid_text": [uid_text(k) for k in keys],
            "removed_labels": labels,
            "removed_count": len(keys),
            "class_counts": class_counts(labels),
            "reason": reason,
            "is_known_bad": bad_uid in set(keys),
        }

    add_selection("rel_top1", top1, "largest d_rel_all within this seed")
    add_selection("rel_top3", top3, "three largest d_rel_all within this seed")
    if bad_uid in by_uid:
        add_selection("known_bad_only", [by_uid[bad_uid]],
                      "explicit known_bad UID, only because it is in this seed train split")
    else:
        selections["known_bad_only"] = {
            "condition": "known_bad_only", "mask_id": None,
            "removed_uids": [], "removed_uid_text": [], "removed_labels": [],
            "removed_count": 0, "class_counts": {}, "reason": "not_in_train_split",
            "is_known_bad": False,
        }

    for k in (1, 3):
        target_labels = [int(r["label"]) for r in rr[:k]]
        seen: set[tuple[tuple[int, int], ...]] = set()
        rng = np.random.default_rng(int(random_seed) + int(seed) * 1009 + k * 100003)
        for mask_index in range(1, n_random_masks + 1):
            for _attempt in range(10000):
                chosen = choose_random_mask(uid_keys, np.asarray([r["label"] for r in rr]),
                                            top3_set, target_labels, rng)
                signature = tuple(chosen)
                if signature not in seen:
                    seen.add(signature)
                    break
            else:
                raise RuntimeError(f"could not find a unique random{k} mask for seed {seed}")
            name = f"random{k}_{mask_index:02d}"
            add_selection(name, chosen,
                          f"class-matched random{k}; candidates exclude this seed rel_top3",
                          mask_id=f"{k}_{mask_index:02d}")
    return {
        "seed": int(seed),
        "n_train": len(rr),
        "top1": selections["rel_top1"],
        "top3": selections["rel_top3"],
        "known_bad_only": selections["known_bad_only"],
        "random1": {k: v for k, v in selections.items() if k.startswith("random1_")},
        "random3": {k: v for k, v in selections.items() if k.startswith("random3_")},
        "all_conditions": selections,
    }


def save_relation_npz(out_dir: Path, seed: int, arrays: Mapping[str, np.ndarray]) -> Path:
    out_dir = require_external_output(out_dir)
    path = out_dir / f"relation_matrices_seed{seed}.npz"
    payload = dict(arrays)
    if "uid" in payload:
        payload["sample_uid"] = payload["uid"]
    np.savez_compressed(require_external_output(path), **payload)
    return path


def provenance_for_analysis(dataset: str, subject: int, seeds: Sequence[int],
                            teacher: str, student: str, artifact_root: Path,
                            split: str, out_dir: Path) -> dict[str, Any]:
    import config

    entries = []
    for seed in seeds:
        for model in (teacher, student):
            path = artifact_file(artifact_root, dataset, model, subject, seed, split)
            entries.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
                "model": model, "seed": int(seed), "split": split,
            })
    log_dir = ROOT / "logs" / "artifact_export_uid"
    source_logs = sorted(str(p) for p in log_dir.glob(f"*/{dataset}_fewshot_*.log"))
    pretrained = {}
    for model in (teacher, student):
        try:
            pretrained[model] = str(config.weight_path(model))
        except Exception as exc:  # small models have no pretrained path
            pretrained[model] = f"not_applicable_or_unavailable: {exc}"
    return {
        "experiment": "teacher_student_relation_gap",
        "protocol": "seed_specific_fewshot",
        "dataset": dataset, "subject_index": int(subject),
        "subject_label": f"S{int(subject) + 1}", "split": split,
        "seeds": [int(s) for s in seeds],
        "teacher": teacher, "student": student,
        "artifact_root": str(artifact_root), "artifacts": entries,
        "source_export_logs": source_logs,
        "pretrained_weight_paths": pretrained,
        "fine_tuned_checkpoint_note": (
            "No persisted fine-tuned model checkpoint was found in the repository; "
            "the analysis reuses the exported eval/no_grad train artifacts."
        ),
        "dataset_config_path": str(ROOT / "configs" / "datasets" / f"{dataset}.yaml"),
        "teacher_config_path": str(ROOT / "configs" / "models" / f"{teacher}.yaml"),
        "student_config_path": str(ROOT / "configs" / "models" / f"{student}.yaml"),
        "split_source": str(ROOT / "data" / "split.py"),
        "dataset_loader_source": str(ROOT / "data" / "eeg_dataset.py"),
        "artifact_source": str(ROOT / "collab" / "artifacts.py"),
        "current_worktree": git_provenance(),
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"]),
        },
        "output_dir": str(out_dir),
    }


def save_resolved_config(path: Path, args: argparse.Namespace) -> None:
    path = require_external_output(path)
    import config
    import yaml

    dataset_cfg = config.load_dataset_config(args.dataset)
    teacher_cfg = config.load_model_config(args.teacher, args.dataset, "fewshot")
    student_cfg = config.load_model_config(args.student, args.dataset, "fewshot")
    resolved = {
        "dataset": args.dataset, "subject_index": int(args.subject),
        "subject_label": f"S{int(args.subject) + 1}", "split": args.split,
        "seeds": [int(s) for s in args.seeds], "teacher": args.teacher,
        "student": args.student, "known_bad_uid": list(args.bad_uid),
        "artifact_root": str(args.artifact_root), "out_dir": str(args.out_dir),
        "n_expected_trials": N_EXPECTED_TRIALS,
        "seed_specific_protocol": True,
        "cross_seed_uid_aggregation": False,
        "data_session": os.environ.get("MI2015001_SESSION", "session_A"),
        "dataset_config": dataset_cfg,
        "teacher_runtime_config": teacher_cfg,
        "student_runtime_config": student_cfg,
        "split": {
            "function": "data.split.subject_split",
            "val_split": float(args.val_split),
            "train_fraction": 1.0 - float(args.val_split),
            "test_fraction": float(args.val_split),
            "policy": "fewshot_stratified_random",
            "random_state": "seed-specific seed value",
        },
        "random_masks": {
            "n_random_masks": int(args.n_random_masks),
            "random_seed": int(args.random_seed),
            "candidates_exclude_seed_specific_rel_top3": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(jsonable(resolved), f, sort_keys=False, allow_unicode=True)


def generate_figures(out_dir: Path, all_rows: Sequence[Mapping[str, Any]],
                     seed_summaries: Mapping[int, Mapping[str, Any]],
                     bad_uid: tuple[int, int]) -> list[str]:
    out_dir = require_external_output(out_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    for seed, summary in sorted(seed_summaries.items()):
        arrays = summary["_arrays"]
        uid_list = [uid_key(x) for x in arrays["uid"]]
        bad_idx = uid_list.index(bad_uid) if bad_uid in uid_list else None
        for title, matrix, filename, cmap in (
            ("Teacher cosine relation", arrays["relation_teacher"],
             f"seed{seed}_teacher_relation.png", "viridis"),
            ("Student cosine relation", arrays["relation_student"],
             f"seed{seed}_student_relation.png", "viridis"),
            ("Absolute relation difference", arrays["relation_abs_diff"],
             f"seed{seed}_absolute_relation_difference.png", "magma"),
        ):
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(matrix, aspect="equal", interpolation="nearest", cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            ax.set_title(f"{title} — seed {seed}")
            ax.set_xlabel("trial index in seed-specific UID order")
            ax.set_ylabel("trial index in seed-specific UID order")
            if bad_idx is not None:
                ax.axhline(bad_idx, color="red", linewidth=1.5)
                ax.axvline(bad_idx, color="red", linewidth=1.5,
                           label=f"known bad {uid_text(bad_uid)}")
                ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            path = fig_dir / filename
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))

    groups = ["T_correct_S_correct", "T_correct_S_wrong",
              "T_wrong_S_correct", "T_wrong_S_wrong"]
    values = [[float(r["d_rel_all"]) for r in all_rows
               if r["prediction_group"] == group] for group in groups]
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(1, len(groups) + 1)
    box_values = [v if v else [np.nan] for v in values]
    ax.boxplot(box_values, positions=positions, widths=.55, showfliers=False)
    rng = np.random.default_rng(8128)
    for x, group, vals in zip(positions, groups, values):
        rr = [r for r in all_rows if r["prediction_group"] == group]
        jitter = rng.uniform(-.13, .13, size=len(rr))
        ax.scatter(np.full(len(rr), x) + jitter, vals, s=24, alpha=.65,
                   color="#4c78a8")
        for j, row in enumerate(rr):
            if row["is_known_bad"]:
                ax.scatter([x + jitter[j]], [vals[j]], s=90, color="red",
                           marker="*", zorder=5, label="known bad" if x == 1 else None)
    ax.set_xticks(positions, [g.replace("_", "\n") for g in groups])
    ax.set_ylabel("d_rel_all = 1 − Pearson(relation rows)")
    ax.set_title("Relation difference by prediction combination (all seed-specific rows)")
    if any(r["is_known_bad"] for r in all_rows):
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "prediction_groups_d_rel_all_box_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    # Descriptive rank stability only: this plot never creates a cross-seed mask.
    by_uid: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_uid[row["_uid_tuple"]].append(row)
    fig, ax = plt.subplots(figsize=(10, 7))
    seed_order = sorted(seed_summaries)
    x_by_seed = {seed: i for i, seed in enumerate(seed_order)}
    for key, rr in sorted(by_uid.items()):
        rr = sorted(rr, key=lambda r: int(r["seed"]))
        xs = [x_by_seed[int(r["seed"])] for r in rr]
        ys = [float(r["rank_percentile"]) for r in rr]
        color = "red" if key == bad_uid else "#999999"
        alpha = 1.0 if key == bad_uid else .22
        lw = 2.5 if key == bad_uid else .7
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw,
                marker="o" if key == bad_uid else None)
    ax.set_xticks(range(len(seed_order)), [str(s) for s in seed_order])
    ax.set_ylim(-.03, 1.03)
    ax.set_xlabel("seed (each seed has its own split and ranking)")
    ax.set_ylabel("within-seed rank percentile; 1 = largest d_rel_all")
    ax.set_title("Descriptive rank stability; no cross-seed selection")
    if bad_uid in by_uid:
        ax.annotate(f"known bad {uid_text(bad_uid)}", xy=(0, 1),
                    xytext=(.05, .92), textcoords="axes fraction", color="red")
    fig.tight_layout()
    path = fig_dir / "rank_percentile_stability_descriptive.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    fig, axes = plt.subplots(1, len(seed_order), figsize=(5 * len(seed_order), 4),
                             squeeze=False)
    for j, seed in enumerate(seed_order):
        rr = [r for r in all_rows if int(r["seed"]) == seed]
        ax = axes[0, j]
        ax.scatter([r["teacher_p_true"] for r in rr],
                   [r["student_p_true"] for r in rr], alpha=.7, s=30)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        bad = [r for r in rr if r["is_known_bad"]]
        if bad:
            ax.scatter([bad[0]["teacher_p_true"]], [bad[0]["student_p_true"]],
                       color="red", marker="*", s=130, label="known bad")
            ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("MIRepNet p(true)"); ax.set_ylabel("IFNet p(true)")
        ax.set_title(f"seed {seed}")
    fig.suptitle("Teacher/student true-class probability")
    fig.tight_layout()
    path = fig_dir / "teacher_student_true_probability_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))
    return created


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty; use --force only for this independent output: {args.out_dir}")
    (args.out_dir / "figures").mkdir(exist_ok=True)
    save_resolved_config(args.out_dir / "config_resolved.yaml", args)
    provenance = provenance_for_analysis(
        args.dataset, args.subject, args.seeds, args.teacher, args.student,
        args.artifact_root, args.split, args.out_dir)
    write_json(args.out_dir / "checkpoint_provenance.json", provenance)

    all_rows: list[dict[str, Any]] = []
    seed_summaries: dict[int, dict[str, Any]] = {}
    selections: dict[str, Any] = {}
    for seed in args.seeds:
        teacher_path = artifact_file(args.artifact_root, args.dataset, args.teacher,
                                     args.subject, seed, args.split)
        student_path = artifact_file(args.artifact_root, args.dataset, args.student,
                                     args.subject, seed, args.split)
        teacher = load_npz_artifact(teacher_path)
        student = load_npz_artifact(student_path)
        rows, summary = compute_sample_rows(
            seed, teacher, student, args.dataset, args.subject, args.split,
            tuple(args.bad_uid))
        arrays = summary.pop("_arrays")
        save_relation_npz(args.out_dir, seed, arrays)
        seed_summaries[int(seed)] = {**summary, "_arrays": arrays}
        all_rows.extend(rows)
        selections[str(seed)] = make_seed_selection(
            rows, seed, tuple(args.bad_uid), args.n_random_masks, args.random_seed)

    write_csv(args.out_dir / "per_sample_metrics.csv", serialize_rows(all_rows),
              PER_SAMPLE_FIELDS)
    aggregate_rows = group_summaries(all_rows)
    write_csv(args.out_dir / "per_sample_aggregate.csv", aggregate_rows, AGG_FIELDS)
    selection_manifest = {
        "dataset": args.dataset,
        "subject": f"S{int(args.subject) + 1}",
        "subject_index": int(args.subject),
        "split": args.split,
        "known_bad_uid": list(args.bad_uid),
        "seed_specific": True,
        "cross_seed_uid_aggregation": False,
        "note": "Each seed uses its own original 60-trial split, ranking and masks.",
        "seeds": selections,
    }
    write_json(args.out_dir / "selection_manifest.json", selection_manifest)
    figure_paths = generate_figures(args.out_dir, all_rows, seed_summaries,
                                    tuple(args.bad_uid))

    bad_by_seed = {}
    for seed in args.seeds:
        bad = [r for r in all_rows if int(r["seed"]) == int(seed)
               and r["is_known_bad"]]
        if not bad:
            bad_by_seed[str(seed)] = {"status": "not_in_train_split"}
        else:
            r = bad[0]
            bad_by_seed[str(seed)] = {
                "status": "in_train_split", "d_rel_all": r["d_rel_all"],
                "d_rel_intra": r["d_rel_intra"], "d_rel_inter": r["d_rel_inter"],
                "rank_in_seed": r["rank_in_seed"],
                "rank_percentile": r["rank_percentile"],
                "label": r["label"], "teacher_pred": r["teacher_pred"],
                "student_pred": r["student_pred"],
                "teacher_p_true": r["teacher_p_true"],
                "student_p_true": r["student_p_true"],
                "prediction_group": r["prediction_group"],
            }
    summary = {
        "status": "analysis_complete",
        "protocol": "seed-specific; no cross-seed UID aggregation",
        "dataset": args.dataset, "subject": f"S{int(args.subject) + 1}",
        "split": args.split, "seeds": [int(s) for s in args.seeds],
        "n_rows": len(all_rows), "n_trials_per_seed": N_EXPECTED_TRIALS,
        "known_bad": {"uid": list(args.bad_uid), "by_seed": bad_by_seed},
        "group_summaries": aggregate_rows,
        "seed_summaries": {str(k): {kk: vv for kk, vv in v.items()
                                     if kk != "_arrays"}
                           for k, v in seed_summaries.items()},
        "selection_manifest": selection_manifest,
        "figures": figure_paths,
        "retraining": {"status": "not_run", "run_command":
                        "python test/qc/teacher_student_relation_gap.py retrain"},
    }
    write_json(args.out_dir / "summary.json", summary)
    write_analysis_report(args.out_dir, summary, all_rows)
    print_analysis_console(summary, selections)
    return summary


def print_analysis_console(summary: Mapping[str, Any], selections: Mapping[str, Any]) -> None:
    print("[analysis] seed-specific relation analysis complete")
    print(f"[analysis] rows={summary['n_rows']} output={summary.get('output_dir', '')}")
    for seed, info in summary["known_bad"]["by_seed"].items():
        if info["status"] == "not_in_train_split":
            print(f"[known_bad] seed={seed} not_in_train_split")
        else:
            print(f"[known_bad] seed={seed} rank={info['rank_in_seed']}/60 "
                  f"d_rel_all={info['d_rel_all']:.6f} "
                  f"group={info['prediction_group']}")
    for seed, sel in selections.items():
        print(f"[selection] seed={seed} top1={sel['top1']['removed_uid_text']} "
              f"top3={sel['top3']['removed_uid_text']}")


def write_analysis_report(out_dir: Path, summary: Mapping[str, Any],
                          all_rows: Sequence[Mapping[str, Any]]) -> None:
    out_dir = require_external_output(out_dir)
    lines = [
        "# MIRepNet–IFNet relation-gap diagnostic",
        "",
        "## Protocol",
        "",
        "This is a seed-specific analysis. Each seed keeps its existing "
        "`subject_split`, its own 60-trial training artifact, ranking and deletion "
        "mask. No cross-seed UID aggregation is used.",
        "",
        f"- Dataset/subject: `{summary['dataset']}` / `{summary['subject']}`",
        f"- Split: `{summary['split']}`; 60 train trials per seed",
        f"- Seeds: `{summary['seeds']}`",
        "- Relation score: `d_rel_all = 1 - Pearson` after removing the diagonal",
        "- `d_rel_intra` and `d_rel_inter` are reported separately; MAE is sensitivity analysis only.",
        "",
        "## Known bad UID `(0, 1)`",
        "",
        "| seed | status | rank | rank percentile | d_rel_all | teacher pred/p(true) | student pred/p(true) | group |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for seed, info in summary["known_bad"]["by_seed"].items():
        if info["status"] == "not_in_train_split":
            lines.append(f"| {seed} | not_in_train_split | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {seed} | in_train_split | {info['rank_in_seed']}/60 | "
                f"{info['rank_percentile']:.4f} | {info['d_rel_all']:.6f} | "
                f"{info['teacher_pred']} / {info['teacher_p_true']:.4f} | "
                f"{info['student_pred']} / {info['student_p_true']:.4f} | "
                f"{info['prediction_group']} |")
    lines += [
        "",
        "The known UID is not inserted into `rel_top1` unless its within-seed "
        "relation score actually ranks first. `known_bad_only` is only emitted "
        "for seeds where the UID is present.",
        "",
        "## Prediction-group descriptive statistics",
        "",
        "See `per_sample_aggregate.csv` for seed-by-seed counts, quartiles and "
        "teacher/student true-class probability gaps. Because the analysis has "
        "only 60 trials per seed, these are descriptive rather than significance tests.",
        "",
        "## Selection",
        "",
        "The immutable UID masks are in `selection_manifest.json`. `rel_top1` and "
        "`rel_top3` are selected independently inside each seed; random masks are "
        "class-matched to the corresponding relation mask and exclude that seed's top3.",
        "",
        "## Retraining status",
        "",
        "Deletion retraining has not been run yet. Run the independent retraining "
        "command after reviewing the artifacts; it reuses the current split and "
        "filters the training tensor before DataLoader construction.",
        "",
        "## Files",
        "",
        "- `per_sample_metrics.csv`: one row per seed-specific trial.",
        "- `relation_matrices_seed*.npz`: features, predictions and 60×60 relations.",
        "- `selection_manifest.json`: fixed per-seed UID masks.",
        "- `figures/`: diagnostic plots; the known UID is marked in red.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compare_uid_labels(expected_uid: np.ndarray, expected_y: np.ndarray,
                       actual_uid: np.ndarray, actual_y: np.ndarray,
                       context: str) -> None:
    ref = {uid_key(u): int(y) for u, y in zip(expected_uid, expected_y)}
    got = {uid_key(u): int(y) for u, y in zip(actual_uid, actual_y)}
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    wrong = sorted(k for k in set(ref) & set(got) if ref[k] != got[k])
    if missing or extra or wrong:
        raise ValueError(f"{context}: UID/label mismatch missing={missing} extra={extra} labels={wrong}")


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "accuracy_pct": float(np.mean(pred == y_true) * 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "balanced_accuracy_pct": float(balanced_accuracy_score(y_true, pred) * 100.0),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
    }


def eval_preprocessed(adapter: Any, model: Any, x_preprocessed: Any,
                      batch_size: int) -> np.ndarray:
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_preprocessed), batch_size):
            _, logits = adapter.forward(model,
                                        x_preprocessed[start:start + batch_size].to(adapter.device))
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_ifnet_mask(adapter: Any, x_train: Any, y_train: np.ndarray,
                     x_test: Any, y_test: np.ndarray, keep: np.ndarray,
                     seed: int, epochs: int, batch_size: int,
                     lr: float, weight_decay: float,
                     num_classes: int) -> dict[str, Any]:
    """Train exactly the current generic adapter loop on a pre-filtered tensor."""
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    model = adapter.build(num_classes)
    x_kept = x_train[keep]
    y_kept = torch.as_tensor(np.asarray(y_train)[keep], dtype=torch.long)
    loader = DataLoader(TensorDataset(x_kept, y_kept), batch_size=batch_size,
                        shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(adapter.device)
            yb = yb.to(adapter.device)
            _, logits = adapter.forward(model, xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
            n_seen += len(yb)
        scheduler.step()
        losses.append(total_loss / max(1, n_seen))
    logits_test = eval_preprocessed(adapter, model, x_test, batch_size)
    pred = np.argmax(logits_test, axis=1).astype(np.int64)
    metrics = metric_row(np.asarray(y_test, dtype=np.int64), pred)
    best_epoch = int(np.argmin(losses) + 1) if losses else None
    result = {
        **metrics,
        "best_epoch_by_train_loss": best_epoch,
        "final_train_loss": losses[-1] if losses else None,
        "best_train_loss": min(losses) if losses else None,
        "train_loss_std": float(np.std(losses)) if losses else None,
        "n_train_after_mask": int(np.sum(keep)),
        "epochs": int(epochs),
    }
    del model
    if adapter.device.type != "cpu":
        torch.cuda.empty_cache()
    return result


def baseline_from_artifact(path: Path) -> dict[str, Any]:
    d = load_npz_artifact(path, expected_n=None)
    y = np.asarray(d["y"], dtype=np.int64)
    pred = np.argmax(d["logits"], axis=1).astype(np.int64)
    return {
        **metric_row(y, pred),
        "best_epoch_by_train_loss": None,
        "final_train_loss": None,
        "best_train_loss": None,
        "train_loss_std": None,
        "n_train_after_mask": N_EXPECTED_TRIALS,
        "epochs": None,
        "baseline_source": str(path),
    }


def load_raw_seed_split(dataset: str, subject: int, seed: int, val_split: float,
                        train_artifact: Mapping[str, Any],
                        test_artifact: Mapping[str, Any]):
    import data
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset, subject, val_split=val_split, seed=seed, return_uid=True)
    compare_uid_labels(train_artifact["sample_uid"], train_artifact["y"],
                       uid_tr, y_tr, f"seed {seed} raw train split")
    compare_uid_labels(test_artifact["sample_uid"], test_artifact["y"],
                       uid_te, y_te, f"seed {seed} raw test split")
    if len(uid_tr) != N_EXPECTED_TRIALS:
        raise ValueError(f"seed {seed}: raw train split has {len(uid_tr)} trials, expected 60")
    return X_tr, np.asarray(y_tr, dtype=np.int64), X_te, np.asarray(y_te, dtype=np.int64), uid_tr, uid_te


def retrain_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {int(r["seed"]): float(r["accuracy_pct"]) for r in results
            if r["condition"] == "full_data"}
    paired = []
    for r in results:
        if r["condition"] == "full_data":
            continue
        x = dict(r)
        x["delta_accuracy_pct_vs_full"] = float(r["accuracy_pct"]) - full[int(r["seed"])]
        x["delta_balanced_accuracy_pct_vs_full"] = (
            float(r["balanced_accuracy_pct"]) - next(float(q["balanced_accuracy_pct"])
            for q in results if q["condition"] == "full_data" and int(q["seed"]) == int(r["seed"])))
        paired.append(x)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in paired:
        by_condition[str(r["condition"])].append(r)
    condition_stats = {}
    for condition, rr in sorted(by_condition.items()):
        deltas = np.asarray([r["delta_accuracy_pct_vs_full"] for r in rr], dtype=float)
        condition_stats[condition] = {
            "n": int(len(deltas)),
            "delta_accuracy_pct_mean": float(np.mean(deltas)),
            "delta_accuracy_pct_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "delta_accuracy_pct_values": deltas.tolist(),
            "accuracy_pct_values": [float(r["accuracy_pct"]) for r in rr],
        }
    random_stats = {}
    for k in (1, 3):
        condition = f"random{k}"
        rr = by_condition.get(condition, [])
        target = by_condition.get(f"rel_top{k}", [])
        target_by_seed = {int(r["seed"]): float(r["delta_accuracy_pct_vs_full"]) for r in target}
        values = np.asarray([float(r["delta_accuracy_pct_vs_full"]) for r in rr], dtype=float)
        per_seed = {}
        for seed in sorted(set(int(r["seed"]) for r in rr)):
            vals = np.asarray([float(r["delta_accuracy_pct_vs_full"]) for r in rr
                               if int(r["seed"]) == seed], dtype=float)
            t = target_by_seed.get(seed)
            pct = None if t is None else float(100.0 * (np.sum(vals < t) + .5 * np.sum(vals == t)) / len(vals))
            per_seed[str(seed)] = {
                "n": int(len(vals)),
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "q025": float(np.quantile(vals, .025)), "q975": float(np.quantile(vals, .975)),
                "rel_top_percentile_among_random": pct,
            }
        random_stats[condition] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) else None),
            "q025": float(np.quantile(values, .025)) if len(values) else None,
            "q975": float(np.quantile(values, .975)) if len(values) else None,
            "per_seed": per_seed,
        }
    return {
        "full_data_accuracy_pct_by_seed": {str(k): v for k, v in sorted(full.items())},
        "paired_results": paired,
        "condition_stats": condition_stats,
        "random_control_stats": random_stats,
    }


RETRAIN_FIELDS = [
    "dataset", "subject", "split", "seed", "condition", "mask_id",
    "removed_uids", "removed_count", "removed_class_counts", "source",
    "accuracy", "accuracy_pct", "balanced_accuracy", "balanced_accuracy_pct",
    "cohen_kappa", "best_epoch_by_train_loss", "final_train_loss",
    "best_train_loss", "train_loss_std", "n_train_after_mask", "epochs",
    "delta_accuracy_pct_vs_full", "delta_balanced_accuracy_pct_vs_full",
    "device", "runtime_sec",
]


def run_retraining(args: argparse.Namespace) -> dict[str, Any]:
    import config
    import torch
    from models import get_adapter

    manifest = read_json(args.out_dir / "selection_manifest.json")
    if not manifest.get("seed_specific") or manifest.get("cross_seed_uid_aggregation"):
        raise ValueError("selection manifest is not the required seed-specific manifest")
    if tuple(manifest["known_bad_uid"]) != tuple(args.bad_uid):
        raise ValueError("CLI known_bad_uid does not match selection manifest")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    if args.device != "cpu":
        print("[warning] only user-owned GPU processes may be used on the shared server; "
              "verify before launching a GPU run", flush=True)

    dcfg = config.load_dataset_config(args.dataset)
    cfg = config.load_model_config(args.student, args.dataset, "fewshot")
    epochs = int(args.epochs if args.epochs is not None else cfg["epochs"])
    if args.epochs is not None:
        print(f"[smoke/pilot] overriding formal epochs={cfg['epochs']} with {epochs}", flush=True)
    batch_size = int(cfg.get("batch_size", 16))
    lr = float(cfg.get("lr", .001))
    weight_decay = float(cfg.get("weight_decay", .01))
    num_classes = int(dcfg["num_classes"])
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))

    results: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_manifest = manifest["seeds"][str(seed)]
        train_artifact = load_npz_artifact(
            artifact_file(args.artifact_root, args.dataset, args.student,
                          args.subject, seed, "train"))
        test_artifact = load_npz_artifact(
            artifact_file(args.artifact_root, args.dataset, args.student,
                          args.subject, seed, "test"), expected_n=None)
        X_tr, y_tr, X_te, y_te, uid_tr, uid_te = load_raw_seed_split(
            args.dataset, args.subject, seed, args.val_split,
            train_artifact, test_artifact)
        adapter_cfg = dict(cfg)
        adapter_cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                           dataset_name=args.dataset)
        adapter = get_adapter(args.student, device=args.device, **adapter_cfg)
        print(f"[retrain] seed={seed} preprocessing train/test", flush=True)
        x_train = adapter.preprocess(X_tr)
        x_test = adapter.preprocess(X_te)
        uid_keys = [uid_key(x) for x in uid_tr]
        uid_to_index = {k: i for i, k in enumerate(uid_keys)}
        by_name = {**seed_manifest["all_conditions"]}
        # The official full_data baseline is the existing exported artifact. It is
        # reused only after split/config provenance checks above.
        baseline_path = artifact_file(args.artifact_root, args.dataset, args.student,
                                      args.subject, seed, "test")
        baseline = baseline_from_artifact(baseline_path)
        base_row = {
            "dataset": args.dataset, "subject": f"S{args.subject + 1}",
            "split": "train", "seed": int(seed), "condition": "full_data",
            "mask_id": "", "removed_uids": "[]", "removed_count": 0,
            "removed_class_counts": "{}", "source": "existing_eval_artifact",
            **baseline, "device": "artifact", "runtime_sec": 0.0,
        }
        results.append(base_row)

        condition_names = ["known_bad_only", "rel_top1", "rel_top3"]
        condition_names += sorted(by_name[k] and k for k in by_name
                                  if k.startswith("random1_") or k.startswith("random3_"))
        for condition in condition_names:
            sel = by_name[condition]
            if condition == "known_bad_only" and not sel["removed_uids"]:
                print(f"[retrain] seed={seed} known_bad_only skipped: not_in_train_split", flush=True)
                continue
            removed = [uid_key(x) for x in sel["removed_uids"]]
            missing = sorted(set(removed) - set(uid_to_index))
            if missing:
                raise ValueError(f"seed {seed} {condition}: removed UIDs not in train split: {missing}")
            keep = np.ones(len(uid_tr), dtype=bool)
            for key in removed:
                keep[uid_to_index[key]] = False
            start = time.time()
            trained = train_ifnet_mask(
                adapter, x_train, y_tr, x_test, y_te, keep, seed, epochs,
                batch_size, lr, weight_decay, num_classes)
            elapsed = time.time() - start
            cond = "random1" if condition.startswith("random1_") else (
                "random3" if condition.startswith("random3_") else condition)
            row = {
                "dataset": args.dataset, "subject": f"S{args.subject + 1}",
                "split": "train", "seed": int(seed), "condition": cond,
                "mask_id": sel.get("mask_id") or condition,
                "removed_uids": sel["removed_uid_text"],
                "removed_count": sel["removed_count"],
                "removed_class_counts": sel["class_counts"],
                "source": "fresh_retrain_seed_specific_mask",
                **trained, "device": str(args.device), "runtime_sec": elapsed,
            }
            results.append(row)
            if cond.startswith("random"):
                random_rows.append(row)
            print(f"[retrain] seed={seed} {condition} acc={row['accuracy_pct']:.2f}% "
                  f"runtime={elapsed:.1f}s", flush=True)

    # Pair each result with its seed's full_data only after all conditions have run.
    full_by_seed = {int(r["seed"]): r for r in results if r["condition"] == "full_data"}
    for row in results:
        base = full_by_seed[int(row["seed"])]
        row["delta_accuracy_pct_vs_full"] = float(row["accuracy_pct"]) - float(base["accuracy_pct"])
        row["delta_balanced_accuracy_pct_vs_full"] = float(row["balanced_accuracy_pct"]) - float(base["balanced_accuracy_pct"])
    write_csv(args.out_dir / "retrain_results.csv", results, RETRAIN_FIELDS)
    write_csv(args.out_dir / "random_control_results.csv", random_rows, RETRAIN_FIELDS)
    rs = retrain_summary(results)
    rs.update({
        "status": "retraining_complete",
        "formal_epochs_from_config": int(cfg["epochs"]),
        "epochs_used": epochs,
        "epochs_overridden": args.epochs is not None,
        "training_protocol": {
            "optimizer": "AdamW (models/base.py default when optimizer key is absent)",
            "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
            "scheduler": "CosineAnnealingLR(T_max=epochs)",
            "loss": "CrossEntropyLoss", "augmentation": False,
            "sampler": "torch.utils.data.DataLoader shuffle=True; no BalancedBatchSampler in existing fewshot path",
            "mask_filter_stage": "preprocessed training tensor filtered before DataLoader/TensorDataset",
            "test_split_unchanged": True,
        },
    })
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
    else:
        summary = {}
    summary["retraining"] = rs
    summary["status"] = "complete" if args.epochs is None else "pilot_complete"
    write_json(summary_path, summary)
    write_final_report(args.out_dir, summary)
    print(f"[retrain] complete: {len(results)} rows; output={args.out_dir}", flush=True)
    return rs


def write_final_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    out_dir = require_external_output(out_dir)
    analysis = summary.get("known_bad", {})
    retr = summary.get("retraining", {})
    lines = [
        "# MIRepNet–IFNet relation-gap diagnostic",
        "",
        "## Protocol and provenance",
        "",
        "The experiment uses each existing seed's original `subject_split` and "
        "60-trial train artifact independently. It does not aggregate sample UIDs "
        "across seeds. See `config_resolved.yaml` and `checkpoint_provenance.json`.",
        "",
        "## Known bad UID `(0, 1)`",
        "",
        "| seed | status | rank | d_rel_all | prediction group |",
        "|---:|---|---:|---:|---|",
    ]
    for seed, info in sorted(analysis.get("by_seed", {}).items(), key=lambda x: int(x[0])):
        if info.get("status") == "not_in_train_split":
            lines.append(f"| {seed} | not_in_train_split | — | — | — |")
        else:
            lines.append(f"| {seed} | in_train_split | {info['rank_in_seed']}/60 | "
                         f"{info['d_rel_all']:.6f} | {info['prediction_group']} |")
    lines += [
        "",
        "## Deletion results",
        "",
        "`retrain_results.csv` reports every seed and condition, plus paired changes "
        "against that same seed's `full_data` baseline. `random_control_results.csv` "
        "contains all class-matched random masks.",
        "",
        "| condition | n | mean Δ accuracy (percentage points) | SD |",
        "|---|---:|---:|---:|",
    ]
    for condition, info in sorted(retr.get("condition_stats", {}).items()):
        lines.append(f"| {condition} | {info['n']} | {info['delta_accuracy_pct_mean']:.4f} | "
                     f"{info['delta_accuracy_pct_std']:.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The evidence must be read from the paired seed-specific deltas and the "
        "random-control percentiles. A positive deletion delta alone is not enough "
        "to claim a stable relation-gap effect; it must be compared with the matched "
        "random masks. The report does not use final test performance to choose masks.",
        "",
        "## Required question-by-question audit",
        "",
        "1. Known-bad rank: see the table above and `per_sample_metrics.csv`.",
        "2. `T_correct_S_wrong`: check the prediction-group column; no forced relabeling is used.",
        "3. Group-level relation differences: see `per_sample_aggregate.csv`.",
        "4–7. Deletion versus full and matched random controls: see the paired deltas and percentiles above.",
        "8. The conclusion should distinguish a robust gap-screening signal from a single signal-quality outlier.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_common_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--subject", type=int, default=DEFAULT_SUBJECT,
                   help="zero-based repository subject; 0 means S1")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--student", default=DEFAULT_STUDENT)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--bad-uid", type=int, nargs=2, default=list(DEFAULT_BAD_UID),
                   metavar=("SUBJECT_ID", "TRIAL_ID"))
    p.add_argument("--artifact-root", type=Path,
                   default=Path('/data1/llx/BigSmallCollab_results') / "artifacts")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--n-random-masks", type=int, default=DEFAULT_N_RANDOM_MASKS)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--force", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="analyze existing 60-trial artifacts")
    add_common_arguments(analyze)
    retrain = sub.add_parser("retrain", help="retrain IFNet with seed-specific UID masks")
    add_common_arguments(retrain)
    retrain.add_argument("--device", default="cpu")
    retrain.add_argument("--epochs", type=int, default=None,
                         help="smoke/pilot override; omit for formal config epochs")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = default_output_root(args.dataset, args.subject)
    args.out_dir = require_external_output(args.out_dir)
    args.artifact_root = external_path(args.artifact_root).resolve()
    if args.n_random_masks < 1:
        raise ValueError("--n-random-masks must be >= 1")
    if not (0.0 < args.val_split < 1.0):
        raise ValueError("--val-split must be in (0,1)")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "retrain":
        run_retraining(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
