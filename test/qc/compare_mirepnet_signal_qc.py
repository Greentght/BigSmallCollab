#!/usr/bin/env python3
"""Independent MIRepNet feature-QC versus Signal-QC diagnostic.

The script lives under test/qc and does not modify the formal experiment
runners. MIRepNet is an offline, in-sample feature source only; EEGNet uses CE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from collab import artifacts
from collab.seed import set_seed
from models import get_adapter

DATASET = "BNCI2015001"
SUBJECT = 0
SUBJECT_NAME = "S1"
PROTOCOL = "fewshot"
SESSION = "session_A"
VAL_SPLIT = 0.7
SEEDS = (666, 667, 668)
KNOWN_BAD = (0, 1)
TEACHER = "mirepnet"
STUDENT = "eegnet"
NUM_CLASSES = 2
MIREPNET_K = 5
MIREPNET_THRESHOLD = 3.5
SIGNAL_THRESHOLD = 3.5
ROBUST_CONSTANT = 0.6745
ROBUST_EPSILON = 1e-12
ZERO_DIFF_EPS_MULTIPLIER = 10.0
RANDOM_MASKS = 20
METHOD_VERSION = "mirepnet_signal_qc_v1"
DEFAULT_OUTPUT = (
    ROOT / "test" / "qc" / "artifacts" / "qc_v1"
    / "mirepnet_vs_signal_qc" / DATASET / SUBJECT_NAME / PROTOCOL
)
SIGNAL_FEATURES = (
    "nonfinite_fraction",
    "max_abs_amplitude",
    "max_channel_peak_to_peak",
    "max_channel_rms",
    "max_channel_first_difference_rms",
    "min_channel_std",
    "max_channel_zero_difference_fraction",
)
SIGNAL_HIGH_FEATURES = (
    "max_abs_amplitude",
    "max_channel_peak_to_peak",
    "max_channel_rms",
    "max_channel_first_difference_rms",
    "max_channel_zero_difference_fraction",
)
SIGNAL_LOW_FEATURE = "min_channel_std"


class QCScoreError(ValueError):
    """A fixed QC rule could not be evaluated safely."""


def uid_tuple(uid: Sequence[int] | np.ndarray) -> tuple[int, int]:
    a = np.asarray(uid).reshape(-1)
    if len(a) != 2:
        raise ValueError(f"UID must have two integers, got {uid!r}")
    return int(a[0]), int(a[1])


def uid_json(uid: Sequence[int] | np.ndarray) -> list[int]:
    return list(uid_tuple(uid))


def unique_uid_set(uids: np.ndarray, context: str) -> set[tuple[int, int]]:
    arr = np.asarray(uids)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{context}: expected UID shape (N,2), got {arr.shape}")
    keys = [uid_tuple(x) for x in arr]
    if len(keys) != len(set(keys)):
        dup = [k for k, n in Counter(keys).items() if n > 1]
        raise ValueError(f"{context}: duplicate UIDs {dup}")
    return set(keys)


def uid_hash(uids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(uids, dtype=np.int64).tobytes()).hexdigest()


def align_by_uid(
    reference_uid: np.ndarray,
    candidate_uid: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    context: str,
) -> dict[str, np.ndarray]:
    ref = np.asarray(reference_uid, dtype=np.int64)
    cand = np.asarray(candidate_uid, dtype=np.int64)
    ref_set = unique_uid_set(ref, f"{context} reference")
    cand_set = unique_uid_set(cand, f"{context} candidate")
    if ref_set != cand_set:
        raise ValueError(
            f"{context}: UID set mismatch; missing={sorted(ref_set - cand_set)}, "
            f"extra={sorted(cand_set - ref_set)}"
        )
    positions = {uid_tuple(x): i for i, x in enumerate(cand)}
    order = np.asarray([positions[uid_tuple(x)] for x in ref], dtype=np.int64)
    out = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if arr.shape[0] != len(cand):
            raise ValueError(
                f"{context}: {name} first dimension {arr.shape[0]} != "
                f"candidate UID count {len(cand)}"
            )
        out[name] = arr[order]
    return out


def _finite_or_raise(values: np.ndarray, context: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise QCScoreError(
            f"{context}: {int((~np.isfinite(values)).sum())} non-finite values"
        )
    return values


def compute_mirepnet_knn_scores(features: np.ndarray, k: int = MIREPNET_K) -> np.ndarray:
    """Leave-self-out mean k-NN cosine distance.

    Only features and k are accepted: no labels, test data, or known-bad UID.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise QCScoreError(f"MIRepNet features must be 2-D, got {x.shape}")
    if x.shape[0] <= int(k):
        raise QCScoreError(f"need n > k, got n={x.shape[0]}, k={k}")
    _finite_or_raise(x, "MIRepNet features")
    norms = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise QCScoreError("MIRepNet features contain zero or non-finite L2 norm")
    z = x / norms[:, None]
    similarity = z @ z.T
    _finite_or_raise(similarity, "MIRepNet cosine similarity")
    distance = 1.0 - similarity
    np.fill_diagonal(distance, np.inf)
    nearest = np.argpartition(distance, kth=int(k) - 1, axis=1)[:, :int(k)]
    score = np.mean(np.take_along_axis(distance, nearest, axis=1), axis=1)
    return _finite_or_raise(score.astype(np.float64), "MIRepNet 5-NN score")


def mirepnet_robust_z(
    scores: np.ndarray, epsilon: float = ROBUST_EPSILON
) -> tuple[np.ndarray, float, float]:
    x = _finite_or_raise(np.asarray(scores, dtype=np.float64), "MIRepNet scores")
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    if mad == 0.0:
        raise QCScoreError(
            "MIRepNet score MAD is zero; fixed protocol requires stopping this seed"
        )
    z = ROBUST_CONSTANT * (x - median) / (mad + float(epsilon))
    return _finite_or_raise(z, "MIRepNet robust z"), median, mad


def _safe_trial_features(trial: np.ndarray, zero_tol: float) -> dict[str, float]:
    x = np.asarray(trial)
    if x.ndim != 2:
        raise QCScoreError(f"trial must be C x T, got {x.shape}")
    finite = np.isfinite(x)
    finite_values = x[finite]
    result = {
        "nonfinite_fraction": float((~finite).mean()),
        "max_abs_amplitude": (
            float(np.max(np.abs(finite_values))) if finite_values.size else float("inf")
        ),
    }
    p2p, rms, diff_rms, stds, zero_fracs = [], [], [], [], []
    for channel in x:
        valid = np.isfinite(channel)
        vals = channel[valid]
        if vals.size:
            p2p.append(float(np.max(vals) - np.min(vals)))
            rms.append(float(np.sqrt(np.mean(np.square(vals, dtype=np.float64)))))
            stds.append(float(np.std(vals, dtype=np.float64)))
        else:
            p2p.append(float("inf"))
            rms.append(float("inf"))
            stds.append(float("inf"))
        if len(channel) >= 2:
            diff = np.diff(channel)
            pair_valid = np.isfinite(channel[:-1]) & np.isfinite(channel[1:])
            d = diff[pair_valid]
            if d.size:
                diff_rms.append(float(np.sqrt(np.mean(np.square(d, dtype=np.float64)))))
                zero_fracs.append(float(np.mean(np.abs(d) <= zero_tol)))
            else:
                diff_rms.append(float("inf"))
                zero_fracs.append(1.0)
        else:
            diff_rms.append(float("inf"))
            zero_fracs.append(1.0)
    result.update({
        "max_channel_peak_to_peak": float(np.max(p2p)),
        "max_channel_rms": float(np.max(rms)),
        "max_channel_first_difference_rms": float(np.max(diff_rms)),
        "min_channel_std": float(np.min(stds)),
        "max_channel_zero_difference_fraction": float(np.max(zero_fracs)),
    })
    return result


def compute_signal_features(
    signal: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute the seven fixed signal features, with no labels or UIDs."""
    x = np.asarray(signal)
    if x.ndim != 3:
        raise QCScoreError(f"Signal-QC input must be N x C x T, got {x.shape}")
    if x.shape[0] == 0 or not np.issubdtype(x.dtype, np.floating):
        raise QCScoreError("Signal-QC requires a non-empty floating array")
    finite_values = x[np.isfinite(x)]
    data_range = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
    dtype_eps = float(np.finfo(x.dtype).eps)
    zero_tol = ZERO_DIFF_EPS_MULTIPLIER * dtype_eps * max(1.0, data_range)
    rows = [_safe_trial_features(t, zero_tol) for t in x]
    features = {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in SIGNAL_FEATURES
    }
    return features, {
        "signal_dtype": str(x.dtype),
        "shape": list(x.shape),
        "finite_abs_data_range": data_range,
        "zero_difference_tolerance": zero_tol,
        "zero_difference_tolerance_rule": (
            f"{ZERO_DIFF_EPS_MULTIPLIER:g} * finfo(dtype).eps * "
            "max(1.0, max_abs_finite_over_seed_train)"
        ),
    }


def _directional_robust_z(
    values: np.ndarray, direction: str, epsilon: float = ROBUST_EPSILON
) -> tuple[np.ndarray, float, float]:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(len(x), np.inf), float("nan"), float("nan")
    median = float(np.median(x[finite]))
    mad = float(np.median(np.abs(x[finite] - median)))
    delta = x - median if direction == "high" else median - x
    if direction not in ("high", "low"):
        raise ValueError(f"unknown robust-z direction {direction}")
    z = np.empty(len(x), dtype=np.float64)
    if mad == 0.0:
        z[:] = 0.0
        z[np.isfinite(delta) & (delta > 0)] = np.inf
        z[~finite] = np.inf
    else:
        z[:] = ROBUST_CONSTANT * delta / (mad + float(epsilon))
        z[~finite] = np.inf
    return z, median, mad


def compute_signal_scores(
    signal: np.ndarray, threshold: float = SIGNAL_THRESHOLD
) -> dict[str, Any]:
    """Compute robust Signal-QC scores and deterministic trigger reasons."""
    features, metadata = compute_signal_features(signal)
    z_by_feature, medians, mads = {}, {}, {}
    for name in SIGNAL_HIGH_FEATURES:
        z, med, mad = _directional_robust_z(features[name], "high")
        z_by_feature[name], medians[name], mads[name] = z, med, mad
    z, med, mad = _directional_robust_z(features[SIGNAL_LOW_FEATURE], "low")
    z_by_feature[SIGNAL_LOW_FEATURE], medians[SIGNAL_LOW_FEATURE], mads[SIGNAL_LOW_FEATURE] = z, med, mad
    score = np.max(
        np.vstack([z_by_feature[name] for name in SIGNAL_HIGH_FEATURES] + [z]),
        axis=0,
    )
    if np.any(np.isnan(score)):
        raise QCScoreError("Signal-QC score contains NaN values")
    flags = (
        (features["nonfinite_fraction"] > 0.0)
        | (score > float(threshold))
    )
    triggers, reasons = [], []
    for i in range(len(score)):
        names = [
            name for name in SIGNAL_HIGH_FEATURES + (SIGNAL_LOW_FEATURE,)
            if z_by_feature[name][i] > float(threshold)
        ]
        if features["nonfinite_fraction"][i] > 0.0:
            names = ["nonfinite_fraction"] + names
        triggers.append(";".join(names))
        parts = []
        if features["nonfinite_fraction"][i] > 0:
            parts.append("nonfinite_fraction>0")
        if names and any(x != "nonfinite_fraction" for x in names):
            parts.append("robust_feature_z>3.5")
        reasons.append(";".join(parts))
    return {
        "features": features,
        "metadata": metadata,
        "z_by_feature": z_by_feature,
        "medians": medians,
        "mads": mads,
        "score": score,
        "flag": flags.astype(bool),
        "trigger_features": triggers,
        "reasons": reasons,
    }


def rank_descending(scores: np.ndarray, uids: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if np.any(np.isnan(x)):
        raise QCScoreError("ranking scores contain NaN")
    if len(x) != len(uids):
        raise ValueError("score/UID length mismatch")
    order = sorted(
        range(len(x)),
        key=lambda i: (-float(x[i]), uid_tuple(uids[i])[0], uid_tuple(uids[i])[1]),
    )
    ranks = np.empty(len(x), dtype=np.int64)
    ranks[np.asarray(order)] = np.arange(1, len(x) + 1)
    return ranks


def _class_counts(
    labels: np.ndarray, removed: Sequence[tuple[int, int]], uids: np.ndarray
) -> dict[str, int]:
    positions = {uid_tuple(x): i for i, x in enumerate(uids)}
    return dict(sorted(Counter(int(labels[positions[u]]) for u in removed).items()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n")


def _write_yaml_once(path: Path, payload: Mapping[str, Any]) -> None:
    # The output directory is protected by --resume.  Rewriting this metadata
    # on an explicit resume keeps score-only then formal runs provenance-correct.
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True))


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(args), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _runtime_config() -> dict[str, Any]:
    cfg = dict(config.load_model_config(STUDENT, DATASET, PROTOCOL))
    cfg.update(in_channels=13, samples=1000, dataset_name=DATASET)
    return cfg


def _load_seed_pack(seed: int, artifact_root: Path) -> dict[str, Any]:
    if os.environ.get("MI2015001_SESSION", SESSION) != SESSION:
        raise RuntimeError("MI2015001_SESSION is not session_A")
    X_train, y_train, X_test, y_test, uid_train, uid_test = data.subject_split(
        DATASET, SUBJECT, val_split=VAL_SPLIT, seed=seed, return_uid=True
    )
    X_train, X_test = np.asarray(X_train, np.float32), np.asarray(X_test, np.float32)
    y_train, y_test = np.asarray(y_train, np.int64), np.asarray(y_test, np.int64)
    uid_train, uid_test = np.asarray(uid_train, np.int64), np.asarray(uid_test, np.int64)
    if X_train.shape != (60, 13, 1000) or X_test.shape != (140, 13, 1000):
        raise AssertionError(f"seed {seed}: wrong train/test shapes")
    train_set = unique_uid_set(uid_train, f"seed {seed} train")
    test_set = unique_uid_set(uid_test, f"seed {seed} test")
    if len(train_set) != 60 or len(test_set) != 140 or train_set & test_set:
        raise AssertionError(f"seed {seed}: invalid UID split")
    if KNOWN_BAD not in train_set:
        raise AssertionError(f"seed {seed}: known bad not in training split")

    train_path = Path(artifacts.artifact_path(
        DATASET, TEACHER, SUBJECT, seed, "train", root=str(artifact_root)
    ))
    test_path = Path(artifacts.artifact_path(
        DATASET, TEACHER, SUBJECT, seed, "test", root=str(artifact_root)
    ))
    train_art = artifacts.load(DATASET, TEACHER, SUBJECT, seed, "train", root=str(artifact_root))
    test_art = artifacts.load(DATASET, TEACHER, SUBJECT, seed, "test", root=str(artifact_root))
    for name, art in (("train", train_art), ("test", test_art)):
        if "sample_uid" not in art:
            raise ValueError(f"seed {seed}: teacher {name} has no sample_uid")
        unique_uid_set(art["sample_uid"], f"seed {seed} teacher {name}")
        if art.get("split_policy") != "fewshot_stratified_random":
            raise ValueError(f"seed {seed}: unexpected {name} split policy")
    train_aligned = align_by_uid(
        uid_train, train_art["sample_uid"],
        {"feats": train_art["feats"], "logits": train_art["logits"], "y": train_art["y"]},
        f"seed {seed} teacher train",
    )
    test_aligned = align_by_uid(uid_test, test_art["sample_uid"], {"y": test_art["y"]},
                                 f"seed {seed} teacher test")
    if not np.array_equal(train_aligned["y"], y_train) or not np.array_equal(test_aligned["y"], y_test):
        raise ValueError(f"seed {seed}: artifact labels do not match split")
    if train_aligned["feats"].shape != (60, 256):
        raise ValueError(f"seed {seed}: expected feats shape (60,256), got {train_aligned['feats'].shape}")

    # Labels, UIDs and known-bad metadata are deliberately not passed to either
    # score function; they are attached only after both score arrays exist.
    mirep_score = compute_mirepnet_knn_scores(train_aligned["feats"], MIREPNET_K)
    mirep_z, mirep_median, mirep_mad = mirepnet_robust_z(mirep_score)
    signal = compute_signal_scores(X_train)
    mirep_flag = mirep_z > MIREPNET_THRESHOLD
    mirep_rank = rank_descending(mirep_score, uid_train)
    signal_rank = rank_descending(signal["score"], uid_train)

    rows = []
    for i, uid in enumerate(uid_train):
        row = {
            "seed": seed, "uid_session": int(uid[0]), "uid_trial": int(uid[1]),
            "label": int(y_train[i]), "is_known_bad": uid_tuple(uid) == KNOWN_BAD,
            "mirepnet_score": float(mirep_score[i]),
            "mirepnet_robust_z": float(mirep_z[i]),
            "mirepnet_rank": int(mirep_rank[i]), "mirepnet_flag": bool(mirep_flag[i]),
            "signal_score": float(signal["score"][i]), "signal_rank": int(signal_rank[i]),
            "signal_flag": bool(signal["flag"][i]),
            "signal_trigger_features": signal["trigger_features"][i],
        }
        row.update({name: float(signal["features"][name][i]) for name in SIGNAL_FEATURES})
        rows.append(row)
    return {
        "seed": seed, "X_train": X_train, "y_train": y_train, "uid_train": uid_train,
        "X_test": X_test, "y_test": y_test, "uid_test": uid_test,
        "features": np.asarray(train_aligned["feats"], np.float32),
        "mirepnet_scores": mirep_score, "mirepnet_z": mirep_z,
        "mirepnet_median": mirep_median, "mirepnet_mad": mirep_mad,
        "mirepnet_rank": mirep_rank, "mirepnet_flag": mirep_flag,
        "signal": signal, "signal_rank": signal_rank, "rows": rows,
        "artifact_train_path": train_path, "artifact_test_path": test_path,
        "artifact_train_sha256": _sha256(train_path), "artifact_test_sha256": _sha256(test_path),
        "artifact_fields": {
            "train": {k: list(np.asarray(v).shape) for k, v in train_art.items() if k != "split_policy"},
            "test": {k: list(np.asarray(v).shape) for k, v in test_art.items() if k != "split_policy"},
        },
    }


def _manifest(
    pack: Mapping[str, Any], method: str, score: np.ndarray, robust_z: np.ndarray,
    rank: np.ndarray, flag: np.ndarray, threshold: float, reasons: Sequence[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    uid = np.asarray(pack["uid_train"])
    entries = []
    for i, value in enumerate(uid):
        entries.append({
            "sample_uid": uid_json(value), "score": float(score[i]),
            "robust_z": float(robust_z[i]), "rank": int(rank[i]),
            "flag": bool(flag[i]), "reason": str(reasons[i]),
        })
    return {
        "method_version": METHOD_VERSION, "method": method, "dataset": DATASET,
        "subject": SUBJECT_NAME, "subject_index": SUBJECT, "protocol": PROTOCOL,
        "session": SESSION, "seed": int(pack["seed"]),
        "input_train_uids": [uid_json(x) for x in uid],
        "input_train_uid_hash": uid_hash(uid),
        "threshold": float(threshold),
        "flagged_uids": [uid_json(uid[i]) for i in range(len(uid)) if flag[i]],
        "removed_count": int(np.sum(flag)), "scores": entries,
        "provenance": {
            "artifact_train": str(pack["artifact_train_path"]),
            "artifact_train_sha256": pack["artifact_train_sha256"],
            "feature_field": "feats",
            "feature_shape": list(np.asarray(pack["features"]).shape),
            "feature_definition": "mean transformer output before clshead",
            "in_sample_pilot": True, **_jsonable(extra),
        },
    }


def _write_qc_outputs(output_dir: Path, packs: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for pack in packs for row in pack["rows"]]
    fields = [
        "seed", "uid_session", "uid_trial", "label", "is_known_bad",
        "mirepnet_score", "mirepnet_robust_z", "mirepnet_rank", "mirepnet_flag",
        "signal_score", "signal_rank", "signal_flag", "signal_trigger_features",
        *SIGNAL_FEATURES,
    ]
    _write_csv(output_dir / "per_sample_qc_scores.csv", all_rows, fields)
    comparison = {}
    for pack in packs:
        seed = int(pack["seed"])
        signal = pack["signal"]
        mirep_reason = [
            "mirepnet_robust_z>3.5" if bool(x) else ""
            for x in pack["mirepnet_flag"]
        ]
        _write_json(
            output_dir / f"mirepnet_qc_manifest_seed{seed}.json",
            _manifest(
                pack, "mirepnet_feature_5nn_cosine", pack["mirepnet_scores"],
                pack["mirepnet_z"], pack["mirepnet_rank"], pack["mirepnet_flag"],
                MIREPNET_THRESHOLD, mirep_reason, {
                    "k": MIREPNET_K, "median_score": pack["mirepnet_median"],
                    "mad": pack["mirepnet_mad"], "robust_constant": ROBUST_CONSTANT,
                    "epsilon": ROBUST_EPSILON, "flag_rule": "robust_z > 3.5",
                },
            ),
        )
        _write_json(
            output_dir / f"signal_qc_manifest_seed{seed}.json",
            _manifest(
                pack, "signal_robust_features", signal["score"],
                signal["score"], pack["signal_rank"], signal["flag"],
                SIGNAL_THRESHOLD, signal["reasons"], {
                    "features": SIGNAL_FEATURES,
                    "high_direction_features": SIGNAL_HIGH_FEATURES,
                    "low_direction_feature": SIGNAL_LOW_FEATURE,
                    "feature_medians": signal["medians"],
                    "feature_mads": signal["mads"],
                    **signal["metadata"],
                    "flag_rule": "nonfinite_fraction > 0 OR signal_score > 3.5",
                },
            ),
        )
        uid = np.asarray(pack["uid_train"])
        mirep_uids = {uid_tuple(uid[i]) for i in range(60) if pack["mirepnet_flag"][i]}
        signal_uids = {uid_tuple(uid[i]) for i in range(60) if signal["flag"][i]}
        inter, union = mirep_uids & signal_uids, mirep_uids | signal_uids
        bad_i = next(i for i, u in enumerate(uid) if uid_tuple(u) == KNOWN_BAD)
        comparison[str(seed)] = {
            "known_bad": list(KNOWN_BAD),
            "mirepnet_flagged_uids": [list(x) for x in sorted(mirep_uids)],
            "signal_flagged_uids": [list(x) for x in sorted(signal_uids)],
            "mirepnet_removed_count": len(mirep_uids),
            "signal_removed_count": len(signal_uids),
            "intersection": [list(x) for x in sorted(inter)],
            "mirepnet_only": [list(x) for x in sorted(mirep_uids - signal_uids)],
            "signal_only": [list(x) for x in sorted(signal_uids - mirep_uids)],
            "jaccard": float(len(inter) / len(union)) if union else 1.0,
            "mirepnet_class_counts": _class_counts(pack["y_train"], sorted(mirep_uids), uid),
            "signal_class_counts": _class_counts(pack["y_train"], sorted(signal_uids), uid),
            "known_bad_mirepnet": {
                "score": float(pack["mirepnet_scores"][bad_i]),
                "robust_z": float(pack["mirepnet_z"][bad_i]),
                "rank": int(pack["mirepnet_rank"][bad_i]),
                "flag": bool(KNOWN_BAD in mirep_uids),
            },
            "known_bad_signal": {
                "score": float(signal["score"][bad_i]),
                "rank": int(pack["signal_rank"][bad_i]),
                "flag": bool(KNOWN_BAD in signal_uids),
                "trigger_features": signal["trigger_features"][bad_i],
            },
        }
    _write_json(output_dir / "mask_comparison.json", comparison)
    return comparison


def _hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in lines) + "\n")


def _train_one(
    *, condition: str, seed: int, X_train: np.ndarray, y_train: np.ndarray,
    uid_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    uid_test: np.ndarray, runtime_cfg: Mapping[str, Any], output_dir: Path,
    device: str, epochs: int, removed_uids: Sequence[tuple[int, int]],
    reused_from: str = "", random_group: str = "", random_mask_id: str = "",
) -> dict[str, Any]:
    if len(X_train) != len(y_train) or len(y_train) != len(uid_train):
        raise ValueError(f"{condition}: training arrays/UID mismatch")
    if len(X_test) != len(y_test) or len(y_test) != len(uid_test):
        raise ValueError(f"{condition}: test arrays/UID mismatch")
    if set(map(tuple, uid_train)) & set(map(tuple, uid_test)):
        raise AssertionError(f"{condition}: train/test UID overlap")
    set_seed(seed)
    started = time.perf_counter()
    adapter = get_adapter(STUDENT, device=device, **dict(runtime_cfg))
    model = adapter.build(NUM_CLASSES)
    initial_hash = _hash_state_dict(model.state_dict())

    # The UID mask is applied before this first TensorDataset/Sampler/DataLoader.
    Xp = adapter.preprocess(np.asarray(X_train, np.float32))
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(Xp.cpu(), y_tensor)
    loader = DataLoader(dataset, batch_size=int(runtime_cfg["batch_size"]),
                        shuffle=True, num_workers=0)
    optimizer_name = str(runtime_cfg.get(
        "optimizer", runtime_cfg.get("optimizer_type", "adamw"))).lower()
    lr = float(runtime_cfg["lr"])
    weight_decay = float(runtime_cfg["weight_decay"])
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                              momentum=float(runtime_cfg.get("momentum", 0.9)))
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history, log_lines = [], [
        f"condition={condition}", f"seed={seed}",
        f"train_count={len(y_train)} test_count={len(y_test)}",
        f"removed_uids={json.dumps([list(x) for x in removed_uids])}",
        "loss=CrossEntropyLoss", f"epochs={epochs} lr={lr} weight_decay={weight_decay}",
        f"batch_size={runtime_cfg['batch_size']} optimizer={optimizer_name}",
        f"scheduler=CosineAnnealingLR(T_max={epochs})",
        "mask_applied_before=TensorDataset,Sampler,DataLoader",
        f"initial_state_hash={initial_hash}",
    ]
    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        correct = 0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(adapter.device), yb.to(adapter.device)
            _, logits = adapter.forward(model, xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = int(yb.shape[0])
            loss_sum += float(loss.detach().cpu()) * n
            correct += int((logits.detach().argmax(1) == yb).sum().cpu())
            seen += n
        scheduler.step()
        item = {
            "condition": condition, "seed": seed, "epoch": epoch,
            "train_loss": float(loss_sum / seen),
            "train_accuracy": float(100.0 * correct / seen),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(item)
        log_lines.append(
            f"epoch={epoch:03d} train_loss={item['train_loss']:.8f} "
            f"train_accuracy={item['train_accuracy']:.4f} "
            f"lr={item['learning_rate']:.10g}"
        )
    model.eval()
    with torch.no_grad():
        _, test_logits = adapter.infer(model, X_test)
    pred = np.asarray(test_logits).argmax(1).astype(np.int64)
    accuracy = float(100.0 * np.mean(pred == y_test))
    balanced = float(100.0 * balanced_accuracy_score(y_test, pred))
    kappa = float(cohen_kappa_score(y_test, pred))
    counts = np.bincount(pred, minlength=NUM_CLASSES)
    final = history[-1]
    elapsed = time.perf_counter() - started
    condition_dir = output_dir / "conditions" / condition / f"seed{seed}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "condition": condition, "seed": seed, "model": STUDENT,
        "model_state_dict": model.state_dict(),
        "runtime_config": _jsonable(dict(runtime_cfg)),
        "initial_state_hash": initial_hash,
        "removed_uids": [list(x) for x in removed_uids],
        "epochs": epochs, "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
    }, condition_dir / "checkpoint.pt")
    _write_csv(condition_dir / "train_history.csv", history,
               ["condition", "seed", "epoch", "train_loss",
                "train_accuracy", "learning_rate"])
    np.savez_compressed(
        condition_dir / "test_predictions.npz",
        sample_uid=np.asarray(uid_test, np.int64), y=np.asarray(y_test, np.int64),
        logits=np.asarray(test_logits, np.float32), pred=pred,
    )
    log_lines += [
        f"test_accuracy={accuracy:.8f}", f"balanced_accuracy={balanced:.8f}",
        f"kappa={kappa:.8f}", f"predicted_class_counts={counts.tolist()}",
        f"elapsed_seconds={elapsed:.6f}",
    ]
    _write_log(condition_dir / "train.log", log_lines)
    return {
        "condition": condition, "seed": seed, "train_count": len(uid_train),
        "test_count": len(uid_test), "removed_count": len(removed_uids),
        "removed_uids": json.dumps([list(x) for x in removed_uids]),
        "contains_known_bad": KNOWN_BAD in set(map(tuple, removed_uids)),
        "accuracy": accuracy, "balanced_accuracy": balanced, "kappa": kappa,
        "final_train_loss": float(final["train_loss"]),
        "final_train_accuracy": float(final["train_accuracy"]),
        "predicted_class_0_count": int(counts[0]),
        "predicted_class_1_count": int(counts[1]),
        "predicted_unique_class_count": int(np.unique(pred).size),
        "collapsed": bool(np.unique(pred).size == 1 or balanced <= 55.0),
        "initial_state_hash": initial_hash, "train_uid_hash": uid_hash(uid_train),
        "test_uid_hash": uid_hash(uid_test), "reused_from": reused_from,
        "random_group": random_group, "random_mask_id": random_mask_id,
        "checkpoint": str(condition_dir / "checkpoint.pt"),
        "elapsed_seconds": elapsed,
    }


def _copy_reused_result(
    base: Mapping[str, Any], condition: str, removed: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    row = dict(base)
    row.update({
        "condition": condition, "removed_count": len(removed),
        "removed_uids": json.dumps([list(x) for x in removed]),
        "contains_known_bad": KNOWN_BAD in set(removed),
        "reused_from": str(base["condition"]),
    })
    return row


def _random_masks_for_composition(
    uids: np.ndarray, labels: np.ndarray, target: Sequence[tuple[int, int]],
    seed: int, n_masks: int,
) -> list[list[tuple[int, int]]]:
    positions = {uid_tuple(x): i for i, x in enumerate(uids)}
    target_counts = Counter(int(labels[positions[u]]) for u in target)
    pools = {}
    for i, x in enumerate(uids):
        pools.setdefault(int(labels[i]), []).append(uid_tuple(x))
    rng = np.random.default_rng(seed * 1009 + len(target) * 9176)
    unique: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(10000):
        if len(unique) >= n_masks:
            break
        selected = []
        for cls, count in sorted(target_counts.items()):
            pool = np.asarray(sorted(pools[cls]), dtype=np.int64)
            if count:
                chosen = rng.choice(len(pool), size=count, replace=False)
                selected.extend(uid_tuple(x) for x in pool[np.sort(chosen)])
        unique.add(tuple(sorted(selected)))
    if len(unique) != n_masks:
        raise RuntimeError(f"could not generate {n_masks} unique masks")
    return [list(x) for x in sorted(unique)]


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for condition in sorted({str(r["condition"]) for r in rows}):
        group = [r for r in rows if str(r["condition"]) == condition]
        arrays = {
            name: np.asarray([float(r[name]) for r in group])
            for name in ("accuracy", "balanced_accuracy", "kappa")
        }
        out[condition] = {
            "n": len(group),
            **{f"{name}_mean": float(v.mean()) for name, v in arrays.items()},
            **{f"{name}_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0
               for name, v in arrays.items()},
            "collapsed_count": int(sum(bool(r["collapsed"]) for r in group)),
        }
    return out


def _random_summary(
    results: list[Mapping[str, Any]], random_rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    out = {}
    for auto in results:
        if auto["condition"] not in ("mirepnet_feature_qc", "signal_qc"):
            continue
        method = auto["condition"]
        candidates = [r for r in random_rows
                      if r["seed"] == auto["seed"] and r["qc_method"] == method]
        key = f"{method}_seed{auto['seed']}"
        if not candidates:
            out[key] = {"n": 0}
            continue
        values = np.asarray([float(r["accuracy"]) for r in candidates])
        sd = float(values.std(ddof=1))
        mean = float(values.mean())
        out[key] = {
            "n": len(values),
            "accuracy_percentile": float(100 * np.mean(values <= float(auto["accuracy"]))),
            "random_accuracy_mean": mean, "random_accuracy_sd": sd,
            "automatic_minus_random_mean": float(auto["accuracy"] - mean),
            "random_accuracy_ci95_normal": [
                mean - 1.96 * sd / np.sqrt(len(values)),
                mean + 1.96 * sd / np.sqrt(len(values)),
            ],
            "random_contains_known_bad_n": int(sum(bool(r["contains_known_bad"]) for r in candidates)),
            "contains_bad_mean": (
                float(np.mean([r["accuracy"] for r in candidates if r["contains_known_bad"]]))
                if any(r["contains_known_bad"] for r in candidates) else None
            ),
            "excludes_bad_mean": (
                float(np.mean([r["accuracy"] for r in candidates if not r["contains_known_bad"]]))
                if any(not r["contains_known_bad"] for r in candidates) else None
            ),
        }
    return out


def _write_figures(
    output_dir: Path, packs: list[Mapping[str, Any]], results: list[Mapping[str, Any]]
) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        seed = int(pack["seed"])
        i_bad = next(i for i, u in enumerate(pack["uid_train"])
                     if uid_tuple(u) == KNOWN_BAD)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(pack["mirepnet_scores"], pack["signal"]["score"],
                   s=24, alpha=.75, label="training trials")
        ax.scatter([pack["mirepnet_scores"][i_bad]], [pack["signal"]["score"][i_bad]],
                   color="red", marker="*", s=100, label="known bad (0,1)")
        ax.set_xlabel("MIRepNet 5-NN mean cosine distance")
        ax.set_ylabel("Signal-QC robust score")
        ax.set_title(f"QC scores, seed {seed}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"scores_scatter_seed{seed}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(len(packs), 1, figsize=(10, 3.2 * len(packs)), squeeze=False)
    for row, pack in enumerate(packs):
        ax = axes[row, 0]
        for values, label, color in (
            (pack["mirepnet_scores"], "MIRepNet", "tab:blue"),
            (pack["signal"]["score"], "Signal-QC", "tab:orange"),
        ):
            order = np.argsort(-np.asarray(values))
            ax.plot(np.arange(1, len(order) + 1), np.asarray(values)[order],
                    marker=".", linewidth=1, label=label, color=color)
            for rank, i in enumerate(order, 1):
                if uid_tuple(pack["uid_train"][i]) == KNOWN_BAD:
                    ax.scatter([rank], [values[i]], color="red", marker="*", s=100)
        ax.set_title(f"seed {pack['seed']} (red star = known bad)")
        ax.set_xlabel("descending rank")
        ax.set_ylabel("score")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_rankings.png", dpi=160)
    plt.close(fig)
    if results:
        conditions = ["base_full", "oracle_bad_only", "mirepnet_feature_qc", "signal_qc"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(conditions))
        width = .24
        for j, seed in enumerate(SEEDS):
            vals = [next(r for r in results if r["condition"] == c and r["seed"] == seed)["accuracy"]
                    for c in conditions]
            ax.bar(x + (j - 1) * width, vals, width, label=f"seed {seed}")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("EEGNet downstream accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "downstream_accuracy.png", dpi=160)
        plt.close(fig)
    return None


def _write_report(
    output_dir: Path, comparisons: Mapping[str, Any], results: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    def fmt(x: Any, n: int = 2) -> str:
        return "NA" if x is None else f"{float(x):.{n}f}"
    lines = [
        "# MIRepNet 特征离群 QC vs Signal-QC", "",
        "本报告是 BNCI2015001 / S1 / few-shot 的单被试诊断，不是通用 EEG-QC 结论。",
        "",
        "## 固定协议", "",
        "- session_A，subject index 0，val_split=0.7，seed 666/667/668；每个 seed 为 60 train / 140 test。",
        "- known-bad UID (0,1) 只用于打分完成后的识别、mask 和结果评价，没有进入 QC 分数或阈值拟合。",
        "- MIRepNet artifact 是见过这些训练 trial 的微调表征，因此本实验标记为 in-sample pilot，不是 frozen/OOF 结果。",
        "- EEGNet 只使用 CrossEntropyLoss；无 KD、MMD、feature alignment、teacher logits 或 teacher 预测。",
        "",
        "## QC 定义", "",
        "- MIRepNet 使用 artifact 字段 feats，(60,256)，来源是 mlm_mask 的 mean transformer output，位于 clshead 之前。float64、逐 trial L2 normalize、排除自身、5-NN cosine distance 均值，robust z > 3.5 标记。",
        "- Signal-QC 使用 data.subject_split 输出的 float32 (N,13,1000)：session 选择、512→250 Hz 重采样、1000 点截断之后，EEGNet identity preprocess 之前。使用七个固定信号质量特征；非有限值或 robust score > 3.5 标记。",
        "- zero-difference tolerance = 10 × finfo(dtype).eps × max(1, seed-train finite max_abs)，epsilon=1e-12。",
        "",
        "## QC 样本选择", "",
        "| seed | method | (0,1) rank | score / z | 是否标记 | 删除数量 | 删除 UID |",
        "| ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for seed in SEEDS:
        c = comparisons[str(seed)]
        lines.append(
            f"| {seed} | MIRepNet-QC | {c['known_bad_mirepnet']['rank']} | "
            f"{fmt(c['known_bad_mirepnet']['score'], 6)} / {fmt(c['known_bad_mirepnet']['robust_z'], 3)} | "
            f"{c['known_bad_mirepnet']['flag']} | {c['mirepnet_removed_count']} | "
            f"{c['mirepnet_flagged_uids']} |"
        )
        lines.append(
            f"| {seed} | Signal-QC | {c['known_bad_signal']['rank']} | "
            f"{fmt(c['known_bad_signal']['score'], 6)} / — | {c['known_bad_signal']['flag']} | "
            f"{c['signal_removed_count']} | {c['signal_flagged_uids']} |"
        )
    lines += [
        "", "## EEGNet 正式结果", "",
        "| condition | Accuracy mean±SD | Balanced Accuracy mean±SD | Kappa mean±SD | 相对 full 变化 | 稳定避免坍塌 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    agg = summary["condition_aggregate"]
    base_acc = agg["base_full"]["accuracy_mean"]
    for condition in ("base_full", "oracle_bad_only", "mirepnet_feature_qc", "signal_qc"):
        a = agg[condition]
        lines.append(
            f"| {condition} | {fmt(a['accuracy_mean'])}±{fmt(a['accuracy_sd'])} | "
            f"{fmt(a['balanced_accuracy_mean'])}±{fmt(a['balanced_accuracy_sd'])} | "
            f"{fmt(a['kappa_mean'], 4)}±{fmt(a['kappa_sd'], 4)} | "
            f"{fmt(a['accuracy_mean'] - base_acc)} pp | {a['collapsed_count'] == 0} |"
        )
    lines += [
        "", "### 每个 seed 的 paired delta（相对 base_full）", "",
        "| seed | oracle_bad_only | mirepnet_feature_qc | signal_qc |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        base = next(r for r in results if r["condition"] == "base_full" and r["seed"] == seed)
        deltas = []
        for condition in ("oracle_bad_only", "mirepnet_feature_qc", "signal_qc"):
            row = next(r for r in results if r["condition"] == condition and r["seed"] == seed)
            deltas.append(fmt(row["accuracy"] - base["accuracy"]))
        lines.append(f"| {seed} | {deltas[0]} pp | {deltas[1]} pp | {deltas[2]} pp |")
    lines += [
        "", "## 随机删除", "",
        "- 每个 seed、每种自动 QC mask 按实际删除数量和类别组成生成 20 个唯一随机 mask；相同组合复用同一组，不重复训练。",
        "- 随机 mask 不排除 (0,1)，每个 mask 是否包含它记录在 random_control_results.csv。",
    ]
    for key, value in summary["random_percentiles"].items():
        if value.get("n"):
            ci = value["random_accuracy_ci95_normal"]
            lines.append(
                f"- {key}: n={value['n']}，自动 QC accuracy 随机经验百分位 "
                f"{fmt(value['accuracy_percentile'])}%，随机均值 {fmt(value['random_accuracy_mean'])}%，"
                f" 95% 正态区间 [{fmt(ci[0])}, {fmt(ci[1])}]。"
            )
    lines += [
        "", "## 结论边界", "",
        "- 两种 QC 都识别并恢复时，只能说明该明显异常可被两种训练前 QC 发现，不能说明 MIRepNet 优于 Signal-QC。",
        "- 只有 MIRepNet-QC 超过等数量、等类别随机删除且 Signal-QC 没有同样效果时，才有本单案例的初步额外价值证据。",
        "- 当前只有一个被试和一个明显异常 trial，不报告有意义的整体 precision/recall，也不据此选择最终方法。",
        "- 正式研究需要冻结预训练 MIRepNet 或 K-fold OOF 表征；Autoreject/Riemannian Potato 尚未验证。",
        "- 测试准确率分辨率为 1/140，约 0.714 个百分点。",
        "",
        "## 复现命令",
        "",
        "score-only: conda run -n mirepnet python test/qc/compare_mirepnet_signal_qc.py --score-only --device cpu",
        "smoke: conda run -n mirepnet python test/qc/compare_mirepnet_signal_qc.py --smoke --epochs 2 --device cpu --output-dir /tmp/mirepnet_signal_qc_smoke",
        "formal on an available isolated GPU: CUDA_VISIBLE_DEVICES=2 conda run -n mirepnet python test/qc/compare_mirepnet_signal_qc.py --formal --device cuda:0 --resume",
        "",
        f"输出目录：{output_dir}",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "results" / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None,
                        help="only allowed with smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-random-masks", type=int, default=RANDOM_MASKS)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    if args.n_random_masks <= 0:
        raise ValueError("n-random-masks must be positive")
    output_dir = (args.output_dir if args.output_dir.is_absolute()
                  else ROOT / args.output_dir).resolve()
    artifact_root = (args.artifact_root if args.artifact_root.is_absolute()
                     else ROOT / args.artifact_root).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty output; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.epochs is not None and not args.smoke:
        raise ValueError("--epochs is only allowed with --smoke")
    epochs = int(args.epochs) if args.smoke and args.epochs is not None else (
        2 if args.smoke else int(_runtime_config()["epochs"])
    )
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    commit = _git(["git", "rev-parse", "HEAD"])
    status = _git(["git", "status", "--short"])
    packs = []
    qc_started = time.perf_counter()
    for seed in SEEDS:
        print(f"[qc] loading/scoring seed {seed}", flush=True)
        packs.append(_load_seed_pack(seed, artifact_root))
    qc_seconds = time.perf_counter() - qc_started
    comparisons = _write_qc_outputs(output_dir, packs)

    runtime_cfg = _runtime_config()
    artifact_provenance = {
        str(p["seed"]): {
            "train_path": str(p["artifact_train_path"]),
            "train_sha256": p["artifact_train_sha256"],
            "test_path": str(p["artifact_test_path"]),
            "test_sha256": p["artifact_test_sha256"],
            "fields_and_shapes": p["artifact_fields"],
            "split_uid_hash": uid_hash(p["uid_train"]),
            "split_policy": "fewshot_stratified_random",
        } for p in packs
    }
    config_payload = {
        "experiment": "mirepnet_vs_signal_qc", "method_version": METHOD_VERSION,
        "dataset": DATASET, "session": SESSION, "subject": SUBJECT_NAME,
        "subject_index": SUBJECT, "protocol": PROTOCOL, "val_split": VAL_SPLIT,
        "seeds": list(SEEDS), "known_bad_uid_evaluation_only": list(KNOWN_BAD),
        "teacher": TEACHER, "student": STUDENT,
        "mirepnet_feature": {
            "artifact_field": "feats", "shape": "(N,256)",
            "definition": "mean transformer output before clshead",
            "dtype_for_score": "float64", "normalization": "per-trial L2",
            "k": MIREPNET_K, "distance": "cosine",
            "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON,
            "threshold": MIREPNET_THRESHOLD, "in_sample_pilot": True,
        },
        "signal_qc": {
            "input_stage": "after session selection, 512->250 Hz resampling, 1000-point truncation; before augmentation, training normalization and EEGNet identity adapter",
            "features": list(SIGNAL_FEATURES),
            "high_direction_features": list(SIGNAL_HIGH_FEATURES),
            "low_direction_feature": SIGNAL_LOW_FEATURE,
            "threshold": SIGNAL_THRESHOLD, "robust_constant": ROBUST_CONSTANT,
            "epsilon": ROBUST_EPSILON, "zero_difference_tolerance_multiplier": ZERO_DIFF_EPS_MULTIPLIER,
        },
        "training": {
            "loss": "CrossEntropyLoss", "formal_epochs": int(runtime_cfg["epochs"]),
            "epochs_this_run": epochs, "lr": runtime_cfg["lr"],
            "weight_decay": runtime_cfg["weight_decay"],
            "batch_size": runtime_cfg["batch_size"],
            "optimizer": runtime_cfg.get("optimizer", runtime_cfg.get("optimizer_type", "adamw")),
            "scheduler": "CosineAnnealingLR", "scheduler_t_max": epochs,
            "model_selection": "last epoch", "device": args.device,
            "random_masks_per_group": int(args.n_random_masks),
        },
        "model_config_path": str(ROOT / "configs" / "models" / "eegnet.yaml"),
        "dataset_config_path": str(ROOT / "configs" / "datasets" / f"{DATASET}.yaml"),
        "artifact_root": str(artifact_root), "artifact_provenance": artifact_provenance,
        "source_data_untouched": True,
    }
    _write_yaml_once(output_dir / "config_resolved.yaml", config_payload)
    _write_json(output_dir / "config_resolved.json", config_payload)
    _write_json(output_dir / "provenance.json", {
        "git_commit": commit, "git_status_short": status,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "conda_environment_hint": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": _nvidia_smi(), "artifact_root": str(artifact_root),
        "artifact_provenance": artifact_provenance,
        "signal_stage": config_payload["signal_qc"]["input_stage"],
        "qc_seconds": qc_seconds, "source_data_untouched": True,
    })
    if args.score_only:
        print(f"[qc] score-only complete: {output_dir}", flush=True)
        return 0

    results, random_rows = [], []
    train_started = time.perf_counter()
    for pack in packs:
        seed = int(pack["seed"])
        uid = np.asarray(pack["uid_train"], np.int64)
        all_uids = set(map(tuple, uid))
        masks = {
            "base_full": all_uids,
            "oracle_bad_only": all_uids - {KNOWN_BAD},
            "mirepnet_feature_qc": all_uids - {
                uid_tuple(uid[i]) for i in range(60) if pack["mirepnet_flag"][i]
            },
            "signal_qc": all_uids - {
                uid_tuple(uid[i]) for i in range(60) if pack["signal"]["flag"][i]
            },
        }
        base_row = None
        for condition in ("base_full", "oracle_bad_only",
                          "mirepnet_feature_qc", "signal_qc"):
            keep = np.asarray([uid_tuple(x) in masks[condition] for x in uid])
            removed = [uid_tuple(x) for x in uid[~keep]]
            if condition == "base_full" and (removed or len(uid) != 60 or KNOWN_BAD not in masks[condition]):
                raise AssertionError("base_full mask invalid")
            if condition == "oracle_bad_only" and removed != [KNOWN_BAD]:
                raise AssertionError("oracle_bad_only mask invalid")
            if condition.startswith("mirepnet"):
                expected = {uid_tuple(uid[i]) for i in range(60) if pack["mirepnet_flag"][i]}
                if set(removed) != expected:
                    raise AssertionError("MIRepNet mask differs from manifest")
            if condition == "signal_qc":
                expected = {uid_tuple(uid[i]) for i in range(60) if pack["signal"]["flag"][i]}
                if set(removed) != expected:
                    raise AssertionError("Signal-QC mask differs from manifest")
            if condition == "base_full":
                row = _train_one(
                    condition=condition, seed=seed,
                    X_train=pack["X_train"][keep], y_train=pack["y_train"][keep],
                    uid_train=uid[keep], X_test=pack["X_test"], y_test=pack["y_test"],
                    uid_test=pack["uid_test"], runtime_cfg=runtime_cfg,
                    output_dir=output_dir, device=args.device, epochs=epochs,
                    removed_uids=removed)
                base_row = row
            elif not removed:
                row = _copy_reused_result(base_row, condition, removed)
            else:
                row = _train_one(
                    condition=condition, seed=seed,
                    X_train=pack["X_train"][keep], y_train=pack["y_train"][keep],
                    uid_train=uid[keep], X_test=pack["X_test"], y_test=pack["y_test"],
                    uid_test=pack["uid_test"], runtime_cfg=runtime_cfg,
                    output_dir=output_dir, device=args.device, epochs=epochs,
                    removed_uids=removed)
            row["qc_method"] = condition if condition in ("mirepnet_feature_qc", "signal_qc") else ""
            results.append(row)

    if not args.smoke:
        groups = {}
        for pack in packs:
            seed = int(pack["seed"])
            uid = np.asarray(pack["uid_train"], np.int64)
            labels = np.asarray(pack["y_train"], np.int64)
            for method in ("mirepnet_feature_qc", "signal_qc"):
                removed = [
                    uid_tuple(uid[i]) for i in range(60)
                    if (pack["mirepnet_flag"][i] if method.startswith("mirepnet")
                        else pack["signal"]["flag"][i])
                ]
                if not removed:
                    continue
                composition = tuple(sorted(_class_counts(labels, removed, uid).items()))
                key = (seed, len(removed), composition)
                if key not in groups:
                    groups[key] = {
                        "seed": seed, "count": len(removed),
                        "class_counts": dict(composition),
                        "masks": _random_masks_for_composition(
                            uid, labels, removed, seed, args.n_random_masks),
                        "group_id": f"seed{seed}_n{len(removed)}_" + "_".join(
                            f"c{c}x{n}" for c, n in composition),
                        "methods": [],
                    }
                groups[key]["methods"].append(method)
        _write_json(output_dir / "random_masks_manifest.json", list(groups.values()))
        for group in groups.values():
            seed = int(group["seed"])
            pack = next(p for p in packs if int(p["seed"]) == seed)
            uid = np.asarray(pack["uid_train"], np.int64)
            for index, removed in enumerate(group["masks"]):
                removed = [uid_tuple(x) for x in removed]
                remove_set = set(removed)
                keep = np.asarray([uid_tuple(x) not in remove_set for x in uid])
                mask_id = f"mask{index:02d}"
                trained = _train_one(
                    condition=f"random_{group['group_id']}_{mask_id}", seed=seed,
                    X_train=pack["X_train"][keep], y_train=pack["y_train"][keep],
                    uid_train=uid[keep], X_test=pack["X_test"], y_test=pack["y_test"],
                    uid_test=pack["uid_test"], runtime_cfg=runtime_cfg,
                    output_dir=output_dir, device=args.device, epochs=epochs,
                    removed_uids=removed, random_group=group["group_id"],
                    random_mask_id=mask_id)
                for method in sorted(set(group["methods"])):
                    row = dict(trained)
                    row.update({
                        "condition": f"random_control_{method}",
                        "qc_method": method,
                        "random_group": group["group_id"],
                        "random_mask_id": mask_id,
                    })
                    random_rows.append(row)

    training_seconds = time.perf_counter() - train_started
    fields = [
        "condition", "seed", "train_count", "test_count", "removed_count",
        "removed_uids", "contains_known_bad", "accuracy", "balanced_accuracy", "kappa",
        "final_train_loss", "final_train_accuracy", "predicted_class_0_count",
        "predicted_class_1_count", "predicted_unique_class_count", "collapsed",
        "reused_from", "initial_state_hash", "train_uid_hash", "test_uid_hash",
        "qc_method", "checkpoint", "elapsed_seconds",
    ]
    _write_csv(output_dir / "results_per_seed.csv", results, fields)
    _write_csv(output_dir / "random_control_results.csv", random_rows,
               fields + ["random_group", "random_mask_id"])
    summary = {
        "experiment": "mirepnet_vs_signal_qc", "mode": "smoke" if args.smoke else "formal",
        "epochs": epochs, "qc_seconds": qc_seconds,
        "training_seconds": training_seconds, "condition_aggregate": _aggregate(results),
        "paired_delta_vs_base": {
            str(seed): {
                c: float(next(r["accuracy"] for r in results
                              if r["seed"] == seed and r["condition"] == c)
                         - next(r["accuracy"] for r in results
                                if r["seed"] == seed and r["condition"] == "base_full"))
                for c in ("oracle_bad_only", "mirepnet_feature_qc", "signal_qc")
            } for seed in SEEDS
        },
        "random_percentiles": _random_summary(results, random_rows),
        "random_rows": len(random_rows), "mask_comparison": comparisons,
        "test_accuracy_resolution_percentage_points": 100.0 / 140.0,
        "in_sample_pilot": True, "source_data_untouched": True,
    }
    _write_json(output_dir / "summary.json", summary)
    figure_error = _write_figures(output_dir, packs, results)
    if figure_error:
        summary["figure_error"] = figure_error
        _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir, comparisons, results, summary)
    print(f"[done] output={output_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
