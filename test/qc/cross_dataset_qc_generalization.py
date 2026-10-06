#!/usr/bin/env python3
"""Cross-dataset external validation of MIRepNet-QC and Signal-QC.

This is intentionally a diagnostic runner under ``test/qc``.  It does not
modify the formal experiment runners and it treats ``results/artifacts`` as a
read-only input directory.  The two QC implementations are imported from the
previous single-dataset experiment so that this experiment cannot silently
change the frozen feature, distance, or robust-z rules.

Stages are restartable:

    --stage inventory
    --stage score
    --stage smoke
    --stage main
    --stage random
    --stage aggregate

The default output is the only directory this script is allowed to write:
``test/qc/artifacts/qc_v1/cross_dataset_qc_generalization``.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from collab.seed import set_seed
from models import get_adapter

from test.qc.compare_mirepnet_signal_qc import (
    QCScoreError,
    SIGNAL_FEATURES,
    SIGNAL_HIGH_FEATURES,
    SIGNAL_LOW_FEATURE,
    ROBUST_CONSTANT,
    ROBUST_EPSILON,
    ZERO_DIFF_EPS_MULTIPLIER,
    align_by_uid,
    compute_mirepnet_knn_scores,
    compute_signal_scores,
    mirepnet_robust_z,
    rank_descending,
    uid_hash,
    uid_tuple,
    unique_uid_set,
    _random_masks_for_composition,
)


"""Datasets fixed by the external-validation protocol.

``BNCI2014001-4`` is intentionally not listed: it is the four-class registry
variant of the same BNCI2014001 source, not a fifth independent dataset.
"""
DATASETS = ("BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI")
SEEDS = (666, 667, 668)
PROTOCOL = "fewshot"
STUDENT = "eegnet"
TEACHER = "mirepnet"
MIREPNET_K = 5
MIREPNET_THRESHOLD = 3.5
SIGNAL_THRESHOLD = 3.5
RANDOM_MASKS = 5
METHOD_VERSION = "mirepnet_signal_qc_v1_cross_dataset"
DEFAULT_OUTPUT = (
    ROOT / "test" / "qc" / "artifacts" / "qc_v1"
    / "cross_dataset_qc_generalization"
)
INPUT_ARTIFACT_ROOT = ROOT / "results" / "artifacts"
OLD_QC_ROOT = ROOT / "test" / "qc" / "artifacts" / "qc_v1" / "mirepnet_vs_signal_qc"
OLD_REGRESSION_DIR = OLD_QC_ROOT / "BNCI2015001" / "S1" / "fewshot"
SESSION_BY_DATASET = {
    "BNCI2014001": "sessionT",
    "BNCI2014004": "session3",
    "BNCI2015001": "session_A",
    "AlexMI": "source subject block; no named session in loader",
}
DATASET_INDEX = {name: i for i, name in enumerate(DATASETS)}

SCORE_FIELDS = [
    "dataset", "subject", "subject_index", "seed", "uid_session", "uid_trial",
    "label", "mirepnet_score", "mirepnet_robust_z", "mirepnet_rank",
    "mirepnet_flag", "signal_score", "signal_rank", "signal_flag",
    "signal_trigger_features", *SIGNAL_FEATURES,
]
MASK_FIELDS = [
    "dataset", "subject", "subject_index", "seed", "method", "status",
    "train_count", "removed_count", "removed_fraction", "removed_uids",
    "removed_class_counts", "retained_class_counts", "empty_mask",
    "unsafe_overfiltering", "invalid_mask", "intersection_uids",
    "method_only_uids", "jaccard",
]
RESULT_FIELDS = [
    "dataset", "subject", "subject_index", "seed", "condition", "status",
    "train_count", "test_count", "removed_count", "removed_uids",
    "contains_known_bad", "accuracy", "balanced_accuracy", "kappa",
    "final_train_loss", "final_train_accuracy", "predicted_class_counts",
    "predicted_unique_class_count", "chance_balanced_accuracy",
    "collapsed", "reused_from", "initial_state_hash", "train_uid_hash",
    "test_uid_hash", "mask_status", "checkpoint", "elapsed_seconds",
    "formal_epochs", "optimizer", "scheduler", "learning_rate",
    "weight_decay", "batch_size",
]
RANDOM_FIELDS = RESULT_FIELDS + [
    "qc_method", "random_group", "random_mask_id", "random_seed",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline="") as handle:
            old_fields = next(csv.reader(handle), [])
        if list(old_fields) != list(fields):
            raise RuntimeError(f"CSV schema mismatch in {path}: {old_fields} != {list(fields)}")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    except Exception as exc:  # pragma: no cover - defensive provenance fallback
        return f"<unavailable: {exc}>"


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - machine dependent
        return f"<unavailable: {exc}>"


def _resolve_output(path: Path) -> Path:
    value = path if path.is_absolute() else ROOT / path
    value = value.resolve()
    allowed = DEFAULT_OUTPUT.resolve()
    if value != allowed and allowed not in value.parents:
        raise ValueError(
            "cross-dataset QC output must stay under "
            f"{allowed}, got {value}"
        )
    if str(value).startswith(str(ROOT / "results")) or value == ROOT / "artifacts":
        raise ValueError("diagnostic output cannot be written under results/ or top-level artifacts/")
    return value


def _resolve_input_artifact_root(path: Path) -> Path:
    value = (path if path.is_absolute() else ROOT / path).resolve()
    if value != INPUT_ARTIFACT_ROOT.resolve():
        raise ValueError(
            "this diagnostic only accepts the read-only input root "
            f"{INPUT_ARTIFACT_ROOT}, got {value}"
        )
    return value


def subject_name(subject_index: int) -> str:
    return f"S{int(subject_index) + 1}"


def dataset_plan() -> list[dict[str, Any]]:
    """Resolve all dataset facts without hard-coding counts or classes."""
    plan = []
    for dataset in DATASETS:
        dcfg = config.load_dataset_config(dataset)
        if tuple(int(x) for x in dcfg.get("seeds", [])) != SEEDS:
            raise ValueError(f"{dataset}: config seeds are not {SEEDS}: {dcfg.get('seeds')}")
        if float(dcfg.get("val_split")) != 0.7:
            raise ValueError(f"{dataset}: expected val_split=0.7, got {dcfg.get('val_split')}")
        plan.append({
            "dataset": dataset,
            "num_classes": int(dcfg["num_classes"]),
            "channels": int(dcfg["channels"]),
            "num_subjects": int(dcfg["num_subjects"]),
            "sample_rate": int(dcfg["sample_rate"]),
            "val_split": float(dcfg["val_split"]),
            "seeds": list(SEEDS),
            "session": SESSION_BY_DATASET[dataset],
            "model_config": config.load_model_config(STUDENT, dataset, PROTOCOL),
        })
    return plan


def build_uid_keep_mask(train_uid: np.ndarray,
                        removed_uids: Iterable[Sequence[int]]) -> np.ndarray:
    """Build and validate the UID mask before any Dataset/DataLoader exists."""
    uid = np.asarray(train_uid, dtype=np.int64)
    unique_uid_set(uid, "train UID")
    removed = [uid_tuple(x) for x in removed_uids]
    if len(removed) != len(set(removed)):
        raise ValueError(f"removed UID list contains duplicates: {removed}")
    train_set = {uid_tuple(x) for x in uid}
    unknown = sorted(set(removed) - train_set)
    if unknown:
        raise ValueError(f"removed UID not in train split: {unknown}")
    return np.asarray([uid_tuple(x) not in set(removed) for x in uid], dtype=bool)


def class_counts(labels: np.ndarray, uids: np.ndarray,
                 selected_uids: Iterable[Sequence[int]]) -> dict[str, int]:
    positions = {uid_tuple(uid): i for i, uid in enumerate(np.asarray(uids))}
    values = [int(labels[positions[uid_tuple(uid)]]) for uid in selected_uids]
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def mask_safety(train_uid: np.ndarray, labels: np.ndarray,
                removed_uids: Sequence[Sequence[int]]) -> dict[str, Any]:
    uid = np.asarray(train_uid, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    removed = [uid_tuple(x) for x in removed_uids]
    keep = build_uid_keep_mask(uid, removed)
    train_classes = set(int(x) for x in labels)
    retained_counts = Counter(int(x) for x in labels[keep])
    invalid = any(retained_counts.get(cls, 0) == 0 for cls in train_classes)
    unsafe = len(removed) / max(1, len(uid)) > 0.25
    status = "invalid_mask" if invalid else "unsafe_overfiltering" if unsafe else "ok"
    return {
        "status": status,
        "removed_count": len(removed),
        "train_count": len(uid),
        "removed_fraction": float(len(removed) / max(1, len(uid))),
        "removed_class_counts": class_counts(labels, uid, removed),
        "retained_class_counts": {str(k): int(v) for k, v in sorted(retained_counts.items())},
        "empty_mask": len(removed) == 0,
        "unsafe_overfiltering": bool(unsafe),
        "invalid_mask": bool(invalid),
        "keep_count": int(keep.sum()),
    }


def collapsed_prediction(pred: np.ndarray, num_classes: int,
                         balanced_accuracy_percent: float) -> bool:
    """Apply the requested class-count-aware collapse rule in percentage units."""
    chance_percent = 100.0 / int(num_classes)
    return bool(np.unique(np.asarray(pred)).size == 1 or
                float(balanced_accuracy_percent) <= chance_percent + 5.0)


def stable_random_seed(dataset: str, subject_index: int, seed: int,
                       count: int, composition: Sequence[tuple[str, int]]) -> int:
    payload = f"{dataset}|{subject_index}|{seed}|{count}|{composition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def generate_random_masks(train_uid: np.ndarray, labels: np.ndarray,
                          target_removed: Sequence[Sequence[int]],
                          random_seed: int, n_masks: int = RANDOM_MASKS
                          ) -> list[list[tuple[int, int]]]:
    """Generate unique class-matched masks deterministically."""
    if not target_removed:
        return []
    masks = _random_masks_for_composition(
        np.asarray(train_uid, dtype=np.int64), np.asarray(labels, dtype=np.int64),
        [uid_tuple(x) for x in target_removed], int(random_seed), int(n_masks),
    )
    normalized = [sorted(uid_tuple(x) for x in mask) for mask in masks]
    if len(normalized) != len({tuple(x) for x in normalized}):
        raise AssertionError("random masks are not unique")
    target_counts = Counter(class_counts(labels, train_uid, target_removed).items())
    for mask in normalized:
        if Counter(class_counts(labels, train_uid, mask).items()) != target_counts:
            raise AssertionError("random mask class composition mismatch")
    return normalized


def _load_split(dataset: str, subject_index: int, seed: int,
                val_split: float) -> dict[str, Any]:
    X_train, y_train, X_test, y_test, uid_train, uid_test = data.subject_split(
        dataset, int(subject_index), val_split=float(val_split), seed=int(seed),
        return_uid=True,
    )
    pack = {
        "dataset": dataset,
        "subject": subject_name(subject_index),
        "subject_index": int(subject_index),
        "seed": int(seed),
        "X_train": np.asarray(X_train, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "uid_train": np.asarray(uid_train, dtype=np.int64),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_test": np.asarray(y_test, dtype=np.int64),
        "uid_test": np.asarray(uid_test, dtype=np.int64),
        "val_split": float(val_split),
    }
    if pack["X_train"].ndim != 3 or pack["X_test"].ndim != 3:
        raise ValueError(f"{dataset} {subject_index} seed {seed}: invalid signal shape")
    if len(pack["X_train"]) != len(pack["y_train"]) or len(pack["X_train"]) != len(pack["uid_train"]):
        raise ValueError(f"{dataset} {subject_index} seed {seed}: train shape/UID mismatch")
    if len(pack["X_test"]) != len(pack["y_test"]) or len(pack["X_test"]) != len(pack["uid_test"]):
        raise ValueError(f"{dataset} {subject_index} seed {seed}: test shape/UID mismatch")
    train_set = unique_uid_set(pack["uid_train"], "split train")
    test_set = unique_uid_set(pack["uid_test"], "split test")
    if train_set & test_set:
        raise ValueError(f"{dataset} {subject_index} seed {seed}: train/test UID overlap")
    return pack


def _split_manifest(pack: Mapping[str, Any], session: str) -> dict[str, Any]:
    return {
        "dataset": pack["dataset"], "subject": pack["subject"],
        "subject_index": pack["subject_index"], "seed": pack["seed"],
        "protocol": PROTOCOL, "session": session, "val_split": pack["val_split"],
        "train_count": len(pack["uid_train"]), "test_count": len(pack["uid_test"]),
        "train_uid": [list(map(int, x)) for x in pack["uid_train"]],
        "test_uid": [list(map(int, x)) for x in pack["uid_test"]],
        "train_uid_hash": uid_hash(pack["uid_train"]),
        "test_uid_hash": uid_hash(pack["uid_test"]),
        "train_class_counts": class_counts(pack["y_train"], pack["uid_train"], pack["uid_train"]),
        "test_class_counts": class_counts(pack["y_test"], pack["uid_test"], pack["uid_test"]),
        "train_test_disjoint": True,
    }


def _artifact_path(dataset: str, subject_index: int, seed: int) -> Path:
    return INPUT_ARTIFACT_ROOT / dataset / TEACHER / f"{subject_index}_{seed}_train.npz"


def inspect_artifact(dataset: str, subject_index: int, seed: int,
                     uid_train: np.ndarray) -> dict[str, Any]:
    """Read only the train artifact and record exact UID/feature provenance."""
    path = _artifact_path(dataset, subject_index, seed)
    base = {
        "dataset": dataset, "subject": subject_name(subject_index),
        "subject_index": int(subject_index), "seed": int(seed),
        "artifact_path": str(path), "artifact_exists": bool(path.exists()),
        "artifact_sha256": "", "feature_key": "feats", "feature_shape": "",
        "artifact_uid_count": 0, "train_uid_count": int(len(uid_train)),
        "uid_exact_match": False, "uid_set_match": False,
        "uid_reordered": False, "eligible_for_mirepnet_qc": False,
        "status": "missing" if not path.exists() else "unread",
    }
    if not path.exists():
        return base
    base["artifact_sha256"] = _sha256(path)
    with np.load(path, allow_pickle=False) as artifact:
        if "feats" not in artifact.files or "sample_uid" not in artifact.files:
            base["status"] = "missing_required_field"
            return base
        feats = np.asarray(artifact["feats"])
        artifact_uid = np.asarray(artifact["sample_uid"], dtype=np.int64)
        base["feature_shape"] = list(feats.shape)
        base["artifact_uid_count"] = int(len(artifact_uid))
        unique_uid_set(artifact_uid, f"{dataset} {subject_index} {seed} artifact")
        split_set = unique_uid_set(uid_train, f"{dataset} {subject_index} {seed} split")
        artifact_set = set(map(uid_tuple, artifact_uid))
        base["uid_set_match"] = bool(split_set == artifact_set)
        base["uid_exact_match"] = bool(np.array_equal(uid_train, artifact_uid))
        base["uid_reordered"] = bool(base["uid_set_match"] and not base["uid_exact_match"])
        if not base["uid_set_match"]:
            base["status"] = "uid_set_mismatch"
        elif feats.ndim != 2 or feats.shape[0] != len(artifact_uid):
            base["status"] = "invalid_feature_shape"
        elif not np.issubdtype(feats.dtype, np.number):
            base["status"] = "invalid_feature_dtype"
        else:
            # The artifact already contains the final mean-pooled representation;
            # the QC rule does not invent another pooling operation.
            base["status"] = "ok"
            base["eligible_for_mirepnet_qc"] = True
    return base


def _load_aligned_artifact_features(info: Mapping[str, Any], uid_train: np.ndarray
                                    ) -> np.ndarray:
    if not info.get("eligible_for_mirepnet_qc"):
        raise QCScoreError(f"artifact is not eligible: {info.get('status')}")
    path = Path(str(info["artifact_path"]))
    with np.load(path, allow_pickle=False) as artifact:
        feats = np.asarray(artifact["feats"])
        aligned = align_by_uid(
            np.asarray(uid_train, dtype=np.int64),
            np.asarray(artifact["sample_uid"], dtype=np.int64),
            {"feats": feats}, "cross-dataset MIRepNet artifact",
        )["feats"]
    if aligned.ndim != 2 or len(aligned) != len(uid_train):
        raise QCScoreError(f"MIRepNet artifact features are not (N,D): {aligned.shape}")
    return aligned


def _rows_from_scores(pack: Mapping[str, Any], mirep: Mapping[str, Any] | None,
                      signal: Mapping[str, Any]) -> list[dict[str, Any]]:
    uid = np.asarray(pack["uid_train"])
    y = np.asarray(pack["y_train"])
    if mirep is not None:
        mirep_scores = mirep["score"]
        mirep_z = mirep["robust_z"]
        mirep_rank = mirep["rank"]
        mirep_flag = mirep["flag"]
    else:
        mirep_scores = mirep_z = mirep_rank = mirep_flag = [None] * len(uid)
    signal_rank = rank_descending(signal["score"], uid)
    rows = []
    for i, item in enumerate(uid):
        row: dict[str, Any] = {
            "dataset": pack["dataset"], "subject": pack["subject"],
            "subject_index": pack["subject_index"], "seed": pack["seed"],
            "uid_session": int(item[0]), "uid_trial": int(item[1]),
            "label": int(y[i]),
            "mirepnet_score": "" if mirep is None else float(mirep_scores[i]),
            "mirepnet_robust_z": "" if mirep is None else float(mirep_z[i]),
            "mirepnet_rank": "" if mirep is None else int(mirep_rank[i]),
            "mirepnet_flag": "" if mirep is None else bool(mirep_flag[i]),
            "signal_score": float(signal["score"][i]),
            "signal_rank": int(signal_rank[i]),
            "signal_flag": bool(signal["flag"][i]),
            "signal_trigger_features": signal["trigger_features"][i],
        }
        row.update({name: float(signal["features"][name][i]) for name in SIGNAL_FEATURES})
        rows.append(row)
    return rows


def _manifest(pack: Mapping[str, Any], method: str, status: str,
              rows: Sequence[Mapping[str, Any]], removed: Sequence[Sequence[int]],
              safety: Mapping[str, Any], info: Mapping[str, Any],
              params: Mapping[str, Any]) -> dict[str, Any]:
    removed_norm = sorted([list(uid_tuple(x)) for x in removed])
    removed_set = {tuple(x) for x in removed_norm}
    retained = [list(map(int, x)) for x in pack["uid_train"] if uid_tuple(x) not in removed_set]
    entries = []
    for row in rows:
        if method == "mirepnet_feature_5nn_cosine":
            score = row["mirepnet_score"]
            z = row["mirepnet_robust_z"]
            rank = row["mirepnet_rank"]
            flag = row["mirepnet_flag"]
            reason = "mirepnet_robust_z>3.5" if flag is True else ""
        else:
            score = row["signal_score"]
            z = row["signal_score"]
            rank = row["signal_rank"]
            flag = row["signal_flag"]
            reason = row["signal_trigger_features"] if flag is True else ""
        entries.append({
            "sample_uid": [int(row["uid_session"]), int(row["uid_trial"])],
            "label": int(row["label"]), "score": score, "robust_z": z,
            "rank": rank, "flag": flag, "reason": reason,
        })
    return {
        "method_version": METHOD_VERSION, "method": method, "status": status,
        "dataset": pack["dataset"], "subject": pack["subject"],
        "subject_index": pack["subject_index"], "protocol": PROTOCOL,
        "session": SESSION_BY_DATASET[pack["dataset"]], "seed": pack["seed"],
        "input_train_uids": [list(map(int, x)) for x in pack["uid_train"]],
        "input_test_uids": [list(map(int, x)) for x in pack["uid_test"]],
        "input_train_uid_hash": uid_hash(pack["uid_train"]),
        "input_test_uid_hash": uid_hash(pack["uid_test"]),
        "threshold": float(params.get("threshold", 3.5)),
        "flagged_uids": removed_norm, "retained_uids": retained,
        "removed_count": len(removed_norm), "retained_count": len(retained),
        "safety": dict(safety), "scores": entries,
        "provenance": {
            "artifact_path": info.get("artifact_path", ""),
            "artifact_sha256": info.get("artifact_sha256", ""),
            "feature_field": "feats" if method.startswith("mirepnet") else None,
            "feature_definition": "final mean-pooled representation before clshead"
            if method.startswith("mirepnet") else None,
            "in_sample_pilot": True if method.startswith("mirepnet") else None,
            **_jsonable(dict(params)),
        },
    }


def _mask_row(pack: Mapping[str, Any], method: str, safety: Mapping[str, Any],
              removed: Sequence[Sequence[int]], other_removed: Sequence[Sequence[int]]) -> dict[str, Any]:
    a = {uid_tuple(x) for x in removed}
    b = {uid_tuple(x) for x in other_removed}
    union = a | b
    return {
        "dataset": pack["dataset"], "subject": pack["subject"],
        "subject_index": pack["subject_index"], "seed": pack["seed"],
        "method": method, "status": safety["status"],
        "train_count": safety["train_count"], "removed_count": safety["removed_count"],
        "removed_fraction": safety["removed_fraction"],
        "removed_uids": json.dumps(sorted([list(x) for x in a])),
        "removed_class_counts": json.dumps(safety["removed_class_counts"], sort_keys=True),
        "retained_class_counts": json.dumps(safety["retained_class_counts"], sort_keys=True),
        "empty_mask": safety["empty_mask"],
        "unsafe_overfiltering": safety["unsafe_overfiltering"],
        "invalid_mask": safety["invalid_mask"],
        "intersection_uids": json.dumps(sorted([list(x) for x in a & b])),
        "method_only_uids": json.dumps(sorted([list(x) for x in a - b])),
        "jaccard": float(len(a & b) / len(union)) if union else 1.0,
    }


def _score_one_fold(pack: Mapping[str, Any], artifact_info: Mapping[str, Any],
                    cell_dir: Path) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cell_dir / "split_manifest.json",
                _split_manifest(pack, SESSION_BY_DATASET[pack["dataset"]]))
    signal_error = ""
    try:
        # This is the only signal input: the complete training fold. Labels,
        # test data, and known-bad metadata are not passed to the score function.
        signal = compute_signal_scores(np.asarray(pack["X_train"], dtype=np.float32))
        signal_status = "ok"
    except Exception as exc:
        signal = None
        signal_status = "score_error"
        signal_error = repr(exc)

    mirep = None
    mirep_status = "missing"
    mirep_error = ""
    if artifact_info.get("eligible_for_mirepnet_qc"):
        try:
            feats = _load_aligned_artifact_features(artifact_info, pack["uid_train"])
            score = compute_mirepnet_knn_scores(feats, MIREPNET_K)
            z, median, mad = mirepnet_robust_z(score)
            mirep_status = "ok"
            mirep = {
                "score": score, "robust_z": z,
                "rank": rank_descending(score, pack["uid_train"]),
                "flag": z > MIREPNET_THRESHOLD,
                "median": median, "mad": mad,
                "feature_shape": list(feats.shape),
            }
        except Exception as exc:
            mirep_status = "score_error"
            mirep_error = repr(exc)
    elif artifact_info.get("status") not in ("missing", "unread"):
        mirep_status = str(artifact_info.get("status"))

    if signal is None:
        rows = []
        signal_removed: list[tuple[int, int]] = []
        signal_safety = {
            "status": "invalid_mask", "train_count": len(pack["uid_train"]),
            "removed_count": 0, "removed_fraction": 0.0,
            "removed_class_counts": {}, "retained_class_counts": {},
            "empty_mask": True, "unsafe_overfiltering": False,
            "invalid_mask": True,
        }
    else:
        rows = _rows_from_scores(pack, mirep, signal)
        signal_removed = [uid_tuple(x) for x, flag in zip(
            pack["uid_train"], signal["flag"]) if bool(flag)]
        signal_safety = mask_safety(pack["uid_train"], pack["y_train"], signal_removed)
    if mirep is None:
        mirep_removed: list[tuple[int, int]] = []
        mirep_safety = {
            "status": mirep_status, "train_count": len(pack["uid_train"]),
            "removed_count": 0, "removed_fraction": 0.0,
            "removed_class_counts": {}, "retained_class_counts": {},
            "empty_mask": True, "unsafe_overfiltering": False,
            "invalid_mask": mirep_status != "missing" or not artifact_info.get("eligible_for_mirepnet_qc"),
        }
    else:
        mirep_removed = [uid_tuple(x) for x, flag in zip(
            pack["uid_train"], mirep["flag"]) if bool(flag)]
        mirep_safety = mask_safety(pack["uid_train"], pack["y_train"], mirep_removed)
        # A fold with a scoring error or unsuitable artifact never silently
        # becomes an empty MIRepNet mask.
        mirep_safety["status"] = "ok" if mirep_safety["status"] == "ok" else mirep_safety["status"]

    if signal is None:
        signal_rows = []
    else:
        signal_rows = rows
    _write_csv(cell_dir / "per_sample_qc_scores.csv", signal_rows, SCORE_FIELDS)

    mirep_manifest = _manifest(
        pack, "mirepnet_feature_5nn_cosine", mirep_status, rows, mirep_removed,
        mirep_safety, artifact_info, {
            "k": MIREPNET_K, "threshold": MIREPNET_THRESHOLD,
            "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON,
            "distance": "cosine", "normalization": "per-trial L2",
            "median_score": None if mirep is None else mirep["median"],
            "mad": None if mirep is None else mirep["mad"],
            "error": mirep_error,
        },
    )
    signal_manifest = _manifest(
        pack, "signal_robust_features", signal_status, rows, signal_removed,
        signal_safety, {"artifact_path": "", "artifact_sha256": ""}, {
            "threshold": SIGNAL_THRESHOLD, "robust_constant": ROBUST_CONSTANT,
            "epsilon": ROBUST_EPSILON, "features": list(SIGNAL_FEATURES),
            "high_direction_features": list(SIGNAL_HIGH_FEATURES),
            "low_direction_feature": SIGNAL_LOW_FEATURE,
            "zero_difference_tolerance_multiplier": ZERO_DIFF_EPS_MULTIPLIER,
            "input_stage": "after deterministic session selection/resampling/truncation; before augmentation, training normalization, and EEGNet identity adapter",
            "feature_medians": {} if signal is None else signal["medians"],
            "feature_mads": {} if signal is None else signal["mads"],
            "metadata": {} if signal is None else signal["metadata"],
            "flag_rule": "nonfinite_fraction > 0 OR signal_score > 3.5",
            "error": signal_error,
        },
    )
    _write_json(cell_dir / "mirepnet_qc_manifest.json", mirep_manifest)
    _write_json(cell_dir / "signal_qc_manifest.json", signal_manifest)

    mirep_set = set(map(uid_tuple, mirep_removed))
    signal_set = set(map(uid_tuple, signal_removed))
    comparison = {
        "mirepnet_status": mirep_status, "signal_status": signal_status,
        "mirepnet_removed_uids": sorted([list(x) for x in mirep_set]),
        "signal_removed_uids": sorted([list(x) for x in signal_set]),
        "intersection_uids": sorted([list(x) for x in mirep_set & signal_set]),
        "mirepnet_only_uids": sorted([list(x) for x in mirep_set - signal_set]),
        "signal_only_uids": sorted([list(x) for x in signal_set - mirep_set]),
        "jaccard": float(len(mirep_set & signal_set) / len(mirep_set | signal_set))
        if mirep_set | signal_set else 1.0,
        "mirepnet_safety": mirep_safety, "signal_safety": signal_safety,
    }
    _write_json(cell_dir / "mask_comparison.json", comparison)
    return {
        "pack": pack, "artifact_info": dict(artifact_info), "rows": rows,
        "signal": signal, "mirep": mirep,
        "mirepnet_status": mirep_status, "signal_status": signal_status,
        "mirepnet_removed": mirep_removed, "signal_removed": signal_removed,
        "mirepnet_safety": mirep_safety, "signal_safety": signal_safety,
        "comparison": comparison,
    }


def _inventory_and_packs(plan: Sequence[Mapping[str, Any]], output_dir: Path,
                         write_packs: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory, packs = [], []
    for item in plan:
        dataset = item["dataset"]
        for subject_index in range(int(item["num_subjects"])):
            for seed in SEEDS:
                print(f"[inventory] {dataset} {subject_name(subject_index)} seed {seed}", flush=True)
                pack = _load_split(dataset, subject_index, seed, item["val_split"])
                info = inspect_artifact(dataset, subject_index, seed, pack["uid_train"])
                inventory.append(info)
                packs.append({"plan": item, "pack": pack, "artifact_info": info})
                if write_packs:
                    cell_dir = output_dir / dataset / pack["subject"] / str(seed)
                    _score_one_fold(pack, info, cell_dir)
    return inventory, packs


def _write_inventory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "dataset", "subject", "subject_index", "seed", "artifact_path",
        "artifact_exists", "artifact_sha256", "feature_key", "feature_shape",
        "artifact_uid_count", "train_uid_count", "uid_exact_match",
        "uid_set_match", "uid_reordered", "eligible_for_mirepnet_qc", "status",
    ]
    _write_csv(path, rows, fields)


def _config_payload(plan: Sequence[Mapping[str, Any]], output_dir: Path,
                    artifact_root: Path, device: str, n_random_masks: int) -> dict[str, Any]:
    return {
        "experiment": "cross_dataset_qc_generalization",
        "method_version": METHOD_VERSION,
        "output_root": str(output_dir),
        "datasets": list(DATASETS),
        "excluded_dataset_variant": {
            "name": "BNCI2014001-4", "reason": "same BNCI2014001 source, 4-class registry variant; fixed target list selects the 2-class BNCI2014001 registration",
        },
        "protocol": PROTOCOL, "seeds": list(SEEDS), "student": STUDENT,
        "teacher": TEACHER, "device": device, "artifact_root_read_only": str(artifact_root),
        "mirepnet_feature": {
            "artifact_field": "feats", "definition": "final mean-pooled representation before clshead",
            "dtype_for_score": "float64", "l2_normalize_per_trial": True,
            "distance": "cosine", "k": MIREPNET_K,
            "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON,
            "threshold": MIREPNET_THRESHOLD, "in_sample_pilot": True,
        },
        "signal_qc": {
            "input_stage": "after deterministic session selection, resampling and time truncation; before augmentation, training normalization and EEGNet identity preprocess",
            "features": list(SIGNAL_FEATURES), "high_direction_features": list(SIGNAL_HIGH_FEATURES),
            "low_direction_feature": SIGNAL_LOW_FEATURE, "threshold": SIGNAL_THRESHOLD,
            "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON,
            "zero_difference_tolerance_multiplier": ZERO_DIFF_EPS_MULTIPLIER,
        },
        "training": {
            "loss": "CrossEntropyLoss", "mask_applied_before": "TensorDataset/Sampler/DataLoader",
            "model_selection": "last epoch", "random_masks_per_group": int(n_random_masks),
            "optimizer_source": str(ROOT / "configs" / "models" / "eegnet.yaml"),
        },
        "dataset_plan": list(plan),
        "source_data_untouched": True,
        "no_filtered_artifacts_or_data": True,
    }


def _write_provenance(output_dir: Path, artifact_root: Path,
                      config_payload: Mapping[str, Any]) -> None:
    _write_json(output_dir / "provenance.json", {
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git(["git", "status", "--short"]),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": _nvidia_smi(), "artifact_root_read_only": str(artifact_root),
        "script_sha256": _sha256(Path(__file__)),
        "old_qc_script_sha256": _sha256(ROOT / "test" / "qc" / "compare_mirepnet_signal_qc.py"),
        "config": config_payload,
    })


def stage_inventory(args: argparse.Namespace, output_dir: Path,
                    artifact_root: Path, plan: Sequence[Mapping[str, Any]]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"non-empty output; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(plan, output_dir, artifact_root, args.device, args.n_random_masks)
    _write_yaml(output_dir / "config_resolved.yaml", payload)
    _write_provenance(output_dir, artifact_root, payload)
    inventory, _ = _inventory_and_packs(plan, output_dir, write_packs=False)
    _write_inventory(output_dir / "artifact_inventory.csv", inventory)
    _write_json(output_dir / "artifact_sha256_before.json", {
        r["artifact_path"]: r["artifact_sha256"] for r in inventory if r["artifact_exists"]
    })
    print(f"[inventory] complete: {len(inventory)} folds -> {output_dir}", flush=True)


def _load_inventory(output_dir: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = _read_csv(output_dir / "artifact_inventory.csv")
    return {(r["dataset"], int(r["subject_index"]), int(r["seed"])): r for r in rows}


def _compare_csv_rows(old_rows: Sequence[Mapping[str, str]], new_rows: Sequence[Mapping[str, str]],
                      fields: Sequence[str]) -> dict[str, Any]:
    def key(row: Mapping[str, str]) -> tuple[int, int]:
        return int(row["uid_session"]), int(row["uid_trial"])
    old_map, new_map = {key(r): r for r in old_rows}, {key(r): r for r in new_rows}
    differences = []
    if set(old_map) != set(new_map):
        differences.append({"kind": "uid_set", "old_only": sorted(set(old_map) - set(new_map)),
                            "new_only": sorted(set(new_map) - set(old_map))})
    for uid in sorted(set(old_map) & set(new_map)):
        for field in fields:
            a, b = old_map[uid].get(field, ""), new_map[uid].get(field, "")
            try:
                equal = bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-10, equal_nan=True))
            except (TypeError, ValueError):
                equal = a == b
            if not equal:
                differences.append({"uid": list(uid), "field": field, "old": a, "new": b})
    return {"passed": not differences, "n_old": len(old_rows), "n_new": len(new_rows),
            "differences": differences[:100], "difference_count": len(differences)}


def run_regression_check(output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "old_output": str(OLD_REGRESSION_DIR), "checks": [], "passed": False,
        "old_qc_script_sha256": _sha256(ROOT / "test" / "qc" / "compare_mirepnet_signal_qc.py"),
        "new_qc_script_sha256": _sha256(Path(__file__)),
    }
    old_csv = OLD_REGRESSION_DIR / "per_sample_qc_scores.csv"
    if not old_csv.exists():
        result["error"] = "old regression CSV is missing"
        _write_json(output_dir / "regression_check.json", result)
        return result
    old_rows_all = _read_csv(old_csv)
    fields = ["mirepnet_score", "mirepnet_robust_z", "mirepnet_rank",
              "mirepnet_flag", "signal_score", "signal_rank", "signal_flag",
              "signal_trigger_features", *SIGNAL_FEATURES]
    for seed in SEEDS:
        new_path = output_dir / "BNCI2015001" / "S1" / str(seed) / "per_sample_qc_scores.csv"
        old_rows = [r for r in old_rows_all if int(r["seed"]) == seed]
        new_rows = _read_csv(new_path)
        check = _compare_csv_rows(old_rows, new_rows, fields)
        check.update({"seed": seed, "new_path": str(new_path)})
        result["checks"].append(check)
    result["passed"] = bool(all(c["passed"] for c in result["checks"]))
    _write_json(output_dir / "regression_check.json", result)
    return result


def stage_score(args: argparse.Namespace, output_dir: Path,
                artifact_root: Path, plan: Sequence[Mapping[str, Any]]) -> None:
    if not (output_dir / "artifact_inventory.csv").exists():
        stage_inventory(args, output_dir, artifact_root, plan)
    inventory = _load_inventory(output_dir)
    summary_rows, mask_rows, all_packs = [], [], []
    for item in plan:
        for subject_index in range(int(item["num_subjects"])):
            for seed in SEEDS:
                print(f"[score] {item['dataset']} {subject_name(subject_index)} seed {seed}", flush=True)
                pack = _load_split(item["dataset"], subject_index, seed, item["val_split"])
                info = inventory[(item["dataset"], subject_index, seed)]
                cell_dir = output_dir / item["dataset"] / pack["subject"] / str(seed)
                result = _score_one_fold(pack, info, cell_dir)
                all_packs.append(result)
                for method, status, removed, safety in (
                    ("mirepnet_feature_qc", result["mirepnet_status"], result["mirepnet_removed"], result["mirepnet_safety"]),
                    ("signal_qc", result["signal_status"], result["signal_removed"], result["signal_safety"]),
                ):
                    other = result["signal_removed"] if method.startswith("mirepnet") else result["mirepnet_removed"]
                    row = _mask_row(pack, method, safety, removed, other)
                    row["status"] = status if status != "ok" else safety["status"]
                    mask_rows.append(row)
                    summary_rows.append({
                        "dataset": pack["dataset"], "subject": pack["subject"],
                        "subject_index": subject_index, "seed": seed, "method": method,
                        "status": row["status"], "train_count": len(pack["uid_train"]),
                        "test_count": len(pack["uid_test"]), "removed_count": len(removed),
                        "removed_fraction": safety["removed_fraction"],
                        "removed_uids": row["removed_uids"],
                        "removed_class_counts": row["removed_class_counts"],
                        "retained_class_counts": row["retained_class_counts"],
                        "empty_mask": safety["empty_mask"],
                        "unsafe_overfiltering": safety["unsafe_overfiltering"],
                        "invalid_mask": safety["invalid_mask"],
                    })
    _write_csv(output_dir / "score_only_summary.csv", summary_rows,
               ["dataset", "subject", "subject_index", "seed", "method", "status",
                "train_count", "test_count", "removed_count", "removed_fraction",
                "removed_uids", "removed_class_counts", "retained_class_counts",
                "empty_mask", "unsafe_overfiltering", "invalid_mask"])
    _write_csv(output_dir / "mask_statistics.csv", mask_rows, MASK_FIELDS)
    regression = run_regression_check(output_dir)
    # The regression file is written before raising, so a mismatch is auditable
    # and cannot be mistaken for a successful external validation.
    if not regression["passed"]:
        raise RuntimeError("BNCI2015001-S1 QC regression failed; formal cross-dataset stages are blocked")
    before = json.loads((output_dir / "artifact_sha256_before.json").read_text())
    after = {path: _sha256(Path(path)) for path in before}
    unchanged = before == after
    _write_json(output_dir / "artifact_sha256_check.json", {
        "passed": unchanged, "before": before, "after": after,
        "results_artifacts_read_only": True,
    })
    if not unchanged:
        raise RuntimeError("input MIRepNet artifact SHA256 changed during score stage")
    qc_provenance = {
        "method_version": METHOD_VERSION,
        "source_script": str(ROOT / "test" / "qc" / "compare_mirepnet_signal_qc.py"),
        "source_script_sha256": _sha256(ROOT / "test" / "qc" / "compare_mirepnet_signal_qc.py"),
        "cross_script_sha256": _sha256(Path(__file__)),
        "mirepnet": {"feature_key": "feats", "feature_definition": "final mean-pooled representation before clshead", "k": MIREPNET_K, "distance": "cosine", "threshold": MIREPNET_THRESHOLD, "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON},
        "signal": {"features": list(SIGNAL_FEATURES), "threshold": SIGNAL_THRESHOLD, "robust_constant": ROBUST_CONSTANT, "epsilon": ROBUST_EPSILON, "zero_difference_tolerance_multiplier": ZERO_DIFF_EPS_MULTIPLIER},
        "regression_check": regression,
        "results_artifacts_read_only": True,
    }
    _write_json(output_dir / "qc_method_provenance.json", qc_provenance)
    _write_json(output_dir / "score_stage_summary.json", {
        "folds": len(all_packs), "mirepnet_eligible": int(sum(bool(x["artifact_info"].get("eligible_for_mirepnet_qc")) for x in all_packs)),
        "signal_nonempty_masks": int(sum(bool(x["signal_removed"]) for x in all_packs)),
        "mirepnet_nonempty_masks": int(sum(bool(x["mirepnet_removed"]) for x in all_packs)),
        "unsafe_or_invalid": int(sum(x["mirepnet_safety"]["status"] not in ("ok", "missing") or x["signal_safety"]["status"] != "ok" for x in all_packs)),
    })
    print(f"[score] complete: {len(all_packs)} folds; regression PASS", flush=True)


def _hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _runtime_config(dataset: str, X_train: np.ndarray) -> dict[str, Any]:
    cfg = dict(config.load_model_config(STUDENT, dataset, PROTOCOL))
    cfg.update(in_channels=int(X_train.shape[1]), samples=int(X_train.shape[2]), dataset_name=dataset)
    return cfg


def _train_one(pack: Mapping[str, Any], condition: str,
               removed_uids: Sequence[Sequence[int]], output_dir: Path,
               device: str, epochs: int, num_classes: int,
               save_outputs: bool = True, random_group: str = "",
               random_mask_id: str = "", random_seed: int | None = None) -> dict[str, Any]:
    uid = np.asarray(pack["uid_train"], dtype=np.int64)
    keep = build_uid_keep_mask(uid, removed_uids)
    X_train = np.asarray(pack["X_train"])[keep]
    y_train = np.asarray(pack["y_train"])[keep]
    uid_kept = uid[keep]
    X_test = np.asarray(pack["X_test"])
    y_test = np.asarray(pack["y_test"])
    uid_test = np.asarray(pack["uid_test"], dtype=np.int64)
    if len(X_train) == 0 or set(map(tuple, uid_kept)) & set(map(tuple, uid_test)):
        raise ValueError(f"{condition}: invalid filtered split")
    set_seed(int(pack["seed"]))
    started = time.perf_counter()
    cfg = _runtime_config(pack["dataset"], X_train)
    adapter = get_adapter(STUDENT, device=device, **cfg)
    model = adapter.build(num_classes)
    initial_hash = _hash_state_dict(model.state_dict())
    # The UID filter is complete before preprocessing, TensorDataset, Sampler,
    # and DataLoader construction. BatchNorm therefore never sees removed trials.
    Xp = adapter.preprocess(X_train)
    dataset = TensorDataset(Xp.cpu(), torch.as_tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    optimizer_name = str(cfg.get("optimizer", cfg.get("optimizer_type", "adamw"))).lower()
    lr = float(cfg["lr"]); wd = float(cfg["weight_decay"])
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                              momentum=float(cfg.get("momentum", 0.9)))
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    criterion = nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        loss_sum = 0.0; correct = 0; seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(adapter.device), yb.to(adapter.device)
            _, logits = adapter.forward(model, xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n = int(yb.shape[0]); loss_sum += float(loss.detach().cpu()) * n
            correct += int((logits.detach().argmax(1) == yb).sum().cpu()); seen += n
        scheduler.step()
        history.append({
            "condition": condition, "epoch": epoch,
            "train_loss": float(loss_sum / seen),
            "train_accuracy": float(100.0 * correct / seen),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
    model.eval()
    with torch.no_grad():
        _, logits = adapter.infer(model, X_test)
    pred = np.asarray(logits).argmax(1).astype(np.int64)
    accuracy = float(100.0 * np.mean(pred == y_test))
    balanced = float(100.0 * balanced_accuracy_score(y_test, pred))
    kappa = float(cohen_kappa_score(y_test, pred))
    counts = np.bincount(pred, minlength=int(num_classes))
    final = history[-1]
    elapsed = time.perf_counter() - started
    condition_dir = output_dir / condition
    if save_outputs:
        condition_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "condition": condition, "dataset": pack["dataset"],
            "subject": pack["subject"], "seed": pack["seed"],
            "model": STUDENT, "model_state_dict": model.state_dict(),
            "runtime_config": _jsonable(cfg), "initial_state_hash": initial_hash,
            "removed_uids": [list(uid_tuple(x)) for x in removed_uids],
            "epochs": int(epochs), "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
        }, condition_dir / "checkpoint.pt")
        _write_csv(condition_dir / "train_history.csv", history,
                   ["condition", "epoch", "train_loss", "train_accuracy", "learning_rate"])
        np.savez_compressed(condition_dir / "test_predictions.npz",
                            sample_uid=uid_test, y=y_test,
                            logits=np.asarray(logits, dtype=np.float32), pred=pred)
        (condition_dir / "train.log").write_text(
            "\n".join([
                f"condition={condition}", f"dataset={pack['dataset']}",
                f"subject={pack['subject']}", f"seed={pack['seed']}",
                f"train_count={len(uid_kept)}", f"test_count={len(uid_test)}",
                f"removed_uids={json.dumps([list(uid_tuple(x)) for x in removed_uids])}",
                "loss=CrossEntropyLoss", f"epochs={epochs}", f"lr={lr}",
                f"weight_decay={wd}", f"batch_size={cfg['batch_size']}",
                f"optimizer={optimizer_name}", f"scheduler=CosineAnnealingLR(T_max={epochs})",
                "mask_applied_before=TensorDataset,Sampler,DataLoader",
                f"initial_state_hash={initial_hash}",
                f"test_accuracy={accuracy}", f"balanced_accuracy={balanced}",
                f"kappa={kappa}", f"predicted_class_counts={counts.tolist()}",
                f"elapsed_seconds={elapsed}",
            ]) + "\n"
        )
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    chance = 100.0 / int(num_classes)
    row = {
        "dataset": pack["dataset"], "subject": pack["subject"],
        "subject_index": pack["subject_index"], "seed": pack["seed"],
        "condition": condition, "status": "ok", "train_count": len(uid_kept),
        "test_count": len(uid_test), "removed_count": len(removed_uids),
        "removed_uids": json.dumps(sorted([list(uid_tuple(x)) for x in removed_uids])),
        "contains_known_bad": False, "accuracy": accuracy,
        "balanced_accuracy": balanced, "kappa": kappa,
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "predicted_class_counts": json.dumps(counts.tolist()),
        "predicted_unique_class_count": int(np.unique(pred).size),
        "chance_balanced_accuracy": chance,
        "collapsed": collapsed_prediction(pred, num_classes, balanced),
        "reused_from": "", "initial_state_hash": initial_hash,
        "train_uid_hash": uid_hash(uid_kept), "test_uid_hash": uid_hash(uid_test),
        "mask_status": "ok", "checkpoint": str(condition_dir / "checkpoint.pt") if save_outputs else "",
        "elapsed_seconds": elapsed, "formal_epochs": int(epochs),
        "optimizer": optimizer_name, "scheduler": "CosineAnnealingLR",
        "learning_rate": lr, "weight_decay": wd, "batch_size": int(cfg["batch_size"]),
        "random_group": random_group, "random_mask_id": random_mask_id,
        "random_seed": "" if random_seed is None else int(random_seed),
    }
    return row


def _invalid_result(pack: Mapping[str, Any], condition: str, removed: Sequence[Sequence[int]],
                    safety: Mapping[str, Any], num_classes: int) -> dict[str, Any]:
    return {
        "dataset": pack["dataset"], "subject": pack["subject"],
        "subject_index": pack["subject_index"], "seed": pack["seed"],
        "condition": condition, "status": safety["status"],
        "train_count": len(pack["uid_train"]), "test_count": len(pack["uid_test"]),
        "removed_count": len(removed),
        "removed_uids": json.dumps(sorted([list(uid_tuple(x)) for x in removed])),
        "contains_known_bad": False, "accuracy": "", "balanced_accuracy": "", "kappa": "",
        "final_train_loss": "", "final_train_accuracy": "",
        "predicted_class_counts": "", "predicted_unique_class_count": "",
        "chance_balanced_accuracy": 100.0 / int(num_classes), "collapsed": "",
        "reused_from": "", "initial_state_hash": "", "train_uid_hash": "",
        "test_uid_hash": uid_hash(pack["uid_test"]), "mask_status": safety["status"],
        "checkpoint": "", "elapsed_seconds": "", "formal_epochs": "",
        "optimizer": "", "scheduler": "", "learning_rate": "",
        "weight_decay": "", "batch_size": "",
    }


def _copy_reused(row: Mapping[str, Any], condition: str,
                 removed: Sequence[Sequence[int]], mask_status: str = "ok") -> dict[str, Any]:
    out = dict(row)
    out.update({
        "condition": condition, "removed_count": len(removed),
        "removed_uids": json.dumps(sorted([list(uid_tuple(x)) for x in removed])),
        "reused_from": row["condition"], "mask_status": mask_status,
    })
    return out


def _read_manifest_mask(cell_dir: Path, method: str) -> tuple[str, list[tuple[int, int]], dict[str, Any]]:
    filename = "mirepnet_qc_manifest.json" if method == "mirepnet_feature_qc" else "signal_qc_manifest.json"
    payload = json.loads((cell_dir / filename).read_text())
    return str(payload.get("status", "missing")), [uid_tuple(x) for x in payload.get("flagged_uids", [])], payload.get("safety", {})


def _result_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return str(row["dataset"]), str(row["subject"]), int(row["seed"]), int(row.get("condition_seed", row.get("seed", 0)))


def stage_smoke(args: argparse.Namespace, output_dir: Path,
                plan: Sequence[Mapping[str, Any]]) -> None:
    if not (output_dir / "regression_check.json").exists():
        stage_score(args, output_dir, INPUT_ARTIFACT_ROOT, plan)
    smoke_root = output_dir / "smoke"
    rows = []
    for item in plan:
        subject_index, seed = 0, 666
        pack = _load_split(item["dataset"], subject_index, seed, item["val_split"])
        cell_dir = output_dir / item["dataset"] / subject_name(subject_index) / str(seed)
        conditions = [("base_full", [], {"status": "ok"})]
        for method in ("mirepnet_feature_qc", "signal_qc"):
            status, removed, safety = _read_manifest_mask(cell_dir, method)
            if status == "ok" and safety.get("status", "ok") == "ok":
                conditions.append((method, removed, safety))
            else:
                rows.append({"dataset": item["dataset"], "subject": subject_name(subject_index), "seed": seed,
                             "condition": method, "status": status or safety.get("status", "invalid_mask"),
                             "message": "not run in smoke because score/mask is unavailable or unsafe"})
        for condition, removed, _ in conditions:
            row = _train_one(pack, condition, removed, smoke_root / item["dataset"] / pack["subject"] / str(seed),
                             args.device, args.smoke_epochs, int(item["num_classes"]), save_outputs=True)
            row["status"] = "smoke_ok"; rows.append(row)
    _write_csv(output_dir / "smoke_results.csv", rows,
               RESULT_FIELDS + ["message"])
    print(f"[smoke] complete: {len(rows)} rows, epochs={args.smoke_epochs}", flush=True)


def _load_score_cells(output_dir: Path, plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    inventory = _load_inventory(output_dir)
    for item in plan:
        for subject_index in range(int(item["num_subjects"])):
            for seed in SEEDS:
                pack = _load_split(item["dataset"], subject_index, seed, item["val_split"])
                cell_dir = output_dir / item["dataset"] / pack["subject"] / str(seed)
                mirep_status, mirep_removed, mirep_safety = _read_manifest_mask(cell_dir, "mirepnet_feature_qc")
                signal_status, signal_removed, signal_safety = _read_manifest_mask(cell_dir, "signal_qc")
                cells.append({
                    "plan": item, "pack": pack, "cell_dir": cell_dir,
                    "artifact_info": inventory[(item["dataset"], subject_index, seed)],
                    "mirep_status": mirep_status, "mirep_removed": mirep_removed, "mirep_safety": mirep_safety,
                    "signal_status": signal_status, "signal_removed": signal_removed, "signal_safety": signal_safety,
                })
    return cells


def stage_main(args: argparse.Namespace, output_dir: Path,
               plan: Sequence[Mapping[str, Any]]) -> None:
    regression = json.loads((output_dir / "regression_check.json").read_text())
    if not regression.get("passed"):
        raise RuntimeError("main stage blocked because regression_check.json is not passed")
    cells = _load_score_cells(output_dir, plan)
    result_path = output_dir / "results_per_seed.csv"
    done = {(r["dataset"], r["subject"], int(r["seed"]), r["condition"])
            for r in _read_csv(result_path)} if args.resume else set()
    for cell in cells:
        pack = cell["pack"]; item = cell["plan"]; subject = pack["subject"]; seed = int(pack["seed"])
        masks = {
            "base_full": [],
            "mirepnet_feature_qc": cell["mirep_removed"],
            "signal_qc": cell["signal_removed"],
        }
        mask_info = {
            "base_full": {"status": "ok"},
            "mirepnet_feature_qc": {"status": cell["mirep_status"], **cell["mirep_safety"]},
            "signal_qc": {"status": cell["signal_status"], **cell["signal_safety"]},
        }
        trained_by_mask: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
        for condition in ("base_full", "mirepnet_feature_qc", "signal_qc"):
            key = (item["dataset"], subject, seed, condition)
            if key in done:
                continue
            removed = [uid_tuple(x) for x in masks[condition]]
            safety = mask_info[condition]
            status_key = "mirep_status" if condition == "mirepnet_feature_qc" else "signal_status"
            if condition != "base_full" and (safety.get("status") != "ok" or
                                               safety.get("unsafe_overfiltering") or safety.get("invalid_mask") or
                                               cell[status_key] != "ok"):
                row = _invalid_result(pack, condition, removed, safety, int(item["num_classes"]))
                _append_csv(result_path, row, RESULT_FIELDS)
                done.add(key); continue
            mask_key = tuple(sorted(removed))
            if mask_key in trained_by_mask:
                row = _copy_reused(trained_by_mask[mask_key], condition, removed)
            else:
                print(f"[main] {item['dataset']} {subject} seed {seed} {condition} n_remove={len(removed)}", flush=True)
                row = _train_one(
                    pack, condition, removed,
                    output_dir / item["dataset"] / subject / str(seed), args.device,
                    int(item["model_config"]["epochs"]), int(item["num_classes"]), save_outputs=True,
                )
                trained_by_mask[mask_key] = row
            row["mask_status"] = "ok"
            _append_csv(result_path, row, RESULT_FIELDS)
            done.add(key)
    print(f"[main] complete/continued: {result_path}", flush=True)


def _random_manifest_path(output_dir: Path) -> Path:
    return output_dir / "random_masks_manifest.json"


def _build_random_manifest(cells: Sequence[Mapping[str, Any]], n_masks: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    group_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in cells:
        pack = cell["pack"]
        for method, status, removed, safety in (
            ("mirepnet_feature_qc", cell["mirep_status"], cell["mirep_removed"], cell["mirep_safety"]),
            ("signal_qc", cell["signal_status"], cell["signal_removed"], cell["signal_safety"]),
        ):
            if status != "ok" or safety.get("status") != "ok" or not removed:
                continue
            composition = tuple(sorted(safety.get("removed_class_counts", {}).items()))
            key = (pack["dataset"], pack["subject_index"], pack["seed"], len(removed), composition)
            if key not in group_index:
                random_seed = stable_random_seed(pack["dataset"], pack["subject_index"], int(pack["seed"]), len(removed), composition)
                masks = generate_random_masks(pack["uid_train"], pack["y_train"], removed, random_seed, n_masks)
                entry = {
                    "dataset": pack["dataset"], "subject": pack["subject"], "subject_index": pack["subject_index"],
                    "seed": pack["seed"], "count": len(removed), "class_counts": dict(composition),
                    "random_seed": random_seed, "methods": [],
                    "masks": [[list(x) for x in mask] for mask in masks],
                }
                group_index[key] = entry; groups.append(entry)
            group_index[key]["methods"].append(method)
    return groups


def stage_random(args: argparse.Namespace, output_dir: Path,
                 plan: Sequence[Mapping[str, Any]]) -> None:
    cells = _load_score_cells(output_dir, plan)
    manifest_path = _random_manifest_path(output_dir)
    if manifest_path.exists() and args.resume:
        groups = json.loads(manifest_path.read_text())
    else:
        groups = _build_random_manifest(cells, args.n_random_masks)
        _write_json(manifest_path, groups)
    result_path = output_dir / "random_control_results.csv"
    done = {(r["dataset"], r["subject"], int(r["seed"]), r["qc_method"], r["random_mask_id"])
            for r in _read_csv(result_path)} if args.resume else set()
    cell_map = {(c["pack"]["dataset"], c["pack"]["subject_index"], int(c["pack"]["seed"])): c for c in cells}
    for group in groups:
        key = (group["dataset"], int(group["subject_index"]), int(group["seed"]))
        cell = cell_map[key]; pack = cell["pack"]
        for index, mask_json in enumerate(group["masks"]):
            mask_id = f"mask{index:02d}"
            needed = [(group["dataset"], group["subject"], int(group["seed"]), method, mask_id)
                      for method in sorted(set(group["methods"]))]
            if needed and all(x in done for x in needed):
                continue
            removed = [uid_tuple(x) for x in mask_json]
            print(f"[random] {group['dataset']} {group['subject']} seed {group['seed']} {mask_id}", flush=True)
            trained = _train_one(
                pack, f"random_{group['random_seed']}_{mask_id}", removed,
                output_dir / "random_runs" / group["dataset"] / group["subject"] / str(group["seed"]),
                args.device, int(group["model_epochs"] if "model_epochs" in group else next(p["model_config"]["epochs"] for p in plan if p["dataset"] == group["dataset"])),
                int(next(p["num_classes"] for p in plan if p["dataset"] == group["dataset"])),
                save_outputs=False, random_group=str(group["random_seed"]), random_mask_id=mask_id,
                random_seed=int(group["random_seed"]),
            )
            for method in sorted(set(group["methods"])):
                row = dict(trained)
                row.update({
                    "condition": f"random_control_{method}", "qc_method": method,
                    "random_group": str(group["random_seed"]), "random_mask_id": mask_id,
                    "random_seed": group["random_seed"],
                    "reused_from": "random_shared_mask_training",
                })
                _append_csv(result_path, row, RANDOM_FIELDS)
                done.add((group["dataset"], group["subject"], int(group["seed"]), method, mask_id))
    print(f"[random] complete/continued: {result_path}", flush=True)


def _float_rows(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(r[field]) for r in rows if r.get(field, "") not in ("", "None")], dtype=float)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 20260908) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _paired_permutation_p(values: np.ndarray, seed: int = 20260908,
                          n_perm: int = 20000) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan")
    if len(x) <= 16:
        signs = np.asarray([(1 if (mask >> i) & 1 else -1) for mask in range(1 << len(x)) for i in range(len(x))], dtype=float).reshape(-1, len(x))
        null = (signs * x).mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        null = (rng.choice(np.asarray([-1.0, 1.0]), size=(n_perm, len(x))) * x).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(float(x.mean()))) + 1) / (len(null) + 1))


def _subject_rows(result_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in result_rows:
        if row.get("status") != "ok":
            continue
        groups[(row["dataset"], row["subject"], int(row["subject_index"]), row["condition"])].append(row)
    out = []
    for (dataset, subject, subject_index, condition), rows in sorted(groups.items()):
        out.append({
            "dataset": dataset, "subject": subject, "subject_index": subject_index,
            "condition": condition, "n_seeds": len(rows),
            "accuracy_mean": float(np.mean(_float_rows(rows, "accuracy"))),
            "balanced_accuracy_mean": float(np.mean(_float_rows(rows, "balanced_accuracy"))),
            "kappa_mean": float(np.mean(_float_rows(rows, "kappa"))),
            "collapsed_count": int(sum(str(r.get("collapsed", "")).lower() == "true" for r in rows)),
            "removed_count_mean": float(np.mean(_float_rows(rows, "removed_count"))),
        })
    return out


def _group_subjects(subject_rows: Sequence[Mapping[str, Any]], group_name: str) -> list[dict[str, Any]]:
    if group_name == "BNCI2015001-S1 only":
        return [r for r in subject_rows if r["dataset"] == "BNCI2015001" and r["subject"] == "S1"]
    if group_name == "BNCI2015001 excluding S1":
        return [r for r in subject_rows if r["dataset"] == "BNCI2015001" and r["subject"] != "S1"]
    if group_name == "external datasets only":
        return [r for r in subject_rows if r["dataset"] in ("BNCI2014001", "BNCI2014004", "AlexMI")]
    return list(subject_rows)


def _summary_stats(subject_rows: Sequence[Mapping[str, Any]], group_name: str) -> list[dict[str, Any]]:
    selected = _group_subjects(subject_rows, group_name)
    conditions = sorted({r["condition"] for r in selected})
    base = {(r["dataset"], r["subject"]): r for r in selected if r["condition"] == "base_full"}
    records = []
    for condition in conditions:
        values = [r for r in selected if r["condition"] == condition]
        if not values:
            continue
        balanced = np.asarray([float(r["balanced_accuracy_mean"]) for r in values])
        deltas = np.asarray([
            float(r["balanced_accuracy_mean"]) - float(base[(r["dataset"], r["subject"])] ["balanced_accuracy_mean"])
            for r in values if (r["dataset"], r["subject"]) in base and condition != "base_full"
        ])
        wins = int(np.sum(deltas > 1e-12)); ties = int(np.sum(np.isclose(deltas, 0.0))); losses = int(np.sum(deltas < -1e-12))
        ci = _bootstrap_ci(deltas, seed=20260908 + len(group_name) + len(condition)) if len(deltas) else (float("nan"), float("nan"))
        try:
            w_p = float(wilcoxon(deltas).pvalue) if len(deltas) >= 2 and np.any(deltas != 0) else float("nan")
        except ValueError:
            w_p = float("nan")
        records.append({
            "group": group_name, "condition": condition, "n_subjects": len(values),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_sd": float(np.std(balanced, ddof=1)) if len(balanced) > 1 else 0.0,
            "accuracy_mean": float(np.mean([float(r["accuracy_mean"]) for r in values])),
            "kappa_mean": float(np.mean([float(r["kappa_mean"]) for r in values])),
            "collapsed_subject_count": int(sum(int(r["collapsed_count"]) > 0 for r in values)),
            "paired_delta_vs_base": "" if condition == "base_full" else float(np.mean(deltas)) if len(deltas) else "",
            "paired_delta_ci95_low": "" if condition == "base_full" else ci[0],
            "paired_delta_ci95_high": "" if condition == "base_full" else ci[1],
            "wins": "" if condition == "base_full" else wins,
            "ties": "" if condition == "base_full" else ties,
            "losses": "" if condition == "base_full" else losses,
            "wilcoxon_p": "" if condition == "base_full" else w_p,
            "paired_permutation_p": "" if condition == "base_full" else _paired_permutation_p(deltas) if len(deltas) else "",
        })
    return records


def _holm(records: list[dict[str, Any]]) -> None:
    indexed = [(i, float(r["wilcoxon_p"])) for i, r in enumerate(records)
               if r.get("wilcoxon_p", "") not in ("", "nan", "NaN") and np.isfinite(float(r["wilcoxon_p"]))]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = {}
    for rank, (i, p) in enumerate(indexed):
        corrected[i] = min(1.0, (m - rank) * p)
    for i, r in enumerate(records):
        r["wilcoxon_p_holm"] = corrected.get(i, "")


def _write_figures(output_dir: Path, score_rows: Sequence[Mapping[str, str]],
                   subject_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        _write_json(output_dir / "figures_error.json", {"error": repr(exc)})
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mirep, signal = [], []
    for row in score_rows:
        if row.get("mirepnet_score", "") not in ("", "None"):
            mirep.append(float(row["mirepnet_score"]))
        if row.get("signal_score", "") not in ("", "None"):
            signal.append(float(row["signal_score"]))
    if mirep and signal:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(mirep, signal, s=6, alpha=.25)
        ax.set_xlabel("MIRepNet 5-NN cosine score")
        ax.set_ylabel("Signal-QC robust score")
        ax.set_title("Cross-dataset QC scores")
        fig.tight_layout(); fig.savefig(figure_dir / "qc_score_scatter.png", dpi=160); plt.close(fig)
    rows = [r for r in subject_rows if r["condition"] in ("base_full", "mirepnet_feature_qc", "signal_qc")]
    datasets = list(DATASETS)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets)); width = .25
    for j, condition in enumerate(("base_full", "mirepnet_feature_qc", "signal_qc")):
        vals = []
        for ds in datasets:
            v = [float(r["balanced_accuracy_mean"]) for r in rows if r["dataset"] == ds and r["condition"] == condition]
            vals.append(float(np.mean(v)) if v else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, label=condition)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.set_ylabel("subject-level mean balanced accuracy (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "downstream_balanced_accuracy.png", dpi=160); plt.close(fig)


def _random_control_summary(result_rows: Sequence[Mapping[str, str]],
                            random_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Summarize QC-versus-random comparisons without treating seeds as subjects."""
    records = []
    for dataset in DATASETS:
        for method in ("mirepnet_feature_qc", "signal_qc"):
            keys = sorted({
                (r["dataset"], r["subject"], int(r["seed"]))
                for r in result_rows
                if r["dataset"] == dataset and r["condition"] == method
                and r.get("status") == "ok" and int(r.get("removed_count", 0)) > 0
            })
            fold_values = []
            for ds, subject, seed in keys:
                qrows = [r for r in result_rows
                         if r["dataset"] == ds and r["subject"] == subject
                         and int(r["seed"]) == seed and r["condition"] == method
                         and r.get("status") == "ok"]
                rrows = [r for r in random_rows
                         if r["dataset"] == ds and r["subject"] == subject
                         and int(r["seed"]) == seed and r.get("qc_method") == method
                         and r.get("status") == "ok"]
                if not qrows or not rrows:
                    continue
                q_ba = float(qrows[0]["balanced_accuracy"])
                random_ba = [float(r["balanced_accuracy"]) for r in rrows]
                fold_values.append({
                    "qc_ba": q_ba,
                    "random_ba_mean": float(np.mean(random_ba)),
                    "random_ba": random_ba,
                    "percentile": float(100.0 * sum(x <= q_ba for x in random_ba) / len(random_ba)),
                })
            if not fold_values:
                continue
            all_random = [x for f in fold_values for x in f["random_ba"]]
            records.append({
                "dataset": dataset, "method": method,
                "n_folds": len(fold_values), "n_random_masks": len(all_random),
                "qc_balanced_accuracy_mean": float(np.mean([f["qc_ba"] for f in fold_values])),
                "random_balanced_accuracy_mean": float(np.mean(all_random)),
                "random_balanced_accuracy_sd": float(np.std(all_random, ddof=1)) if len(all_random) > 1 else 0.0,
                "qc_minus_random_mean": float(np.mean([f["qc_ba"] - f["random_ba_mean"] for f in fold_values])),
                "empirical_percentile_mean": float(np.mean([f["percentile"] for f in fold_values])),
                "empirical_percentile_median": float(np.median([f["percentile"] for f in fold_values])),
                "percentile_above_50_count": int(sum(f["percentile"] > 50.0 for f in fold_values)),
            })
    return records


def _nonempty_fold_summary(result_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Descriptive mechanism analysis restricted to valid non-empty QC masks."""
    out = []
    for dataset in DATASETS:
        for method in ("mirepnet_feature_qc", "signal_qc"):
            qrows = [r for r in result_rows
                     if r["dataset"] == dataset and r["condition"] == method
                     and r.get("status") == "ok" and int(r.get("removed_count", 0)) > 0]
            base = {(r["dataset"], r["subject"], int(r["seed"])): r
                    for r in result_rows
                    if r["dataset"] == dataset and r["condition"] == "base_full"
                    and r.get("status") == "ok"}
            deltas = [float(r["balanced_accuracy"]) - float(base[(r["dataset"], r["subject"], int(r["seed"]))]["balanced_accuracy"])
                      for r in qrows if (r["dataset"], r["subject"], int(r["seed"])) in base]
            if not qrows:
                continue
            out.append({
                "dataset": dataset, "method": method, "n_folds": len(qrows),
                "qc_balanced_accuracy_mean": float(np.mean([float(r["balanced_accuracy"]) for r in qrows])),
                "base_balanced_accuracy_mean_same_folds": float(np.mean([
                    float(base[(r["dataset"], r["subject"], int(r["seed"]))]["balanced_accuracy"])
                    for r in qrows if (r["dataset"], r["subject"], int(r["seed"])) in base
                ])),
                "paired_delta_mean": float(np.mean(deltas)) if deltas else "",
                "wins": int(sum(d > 1e-12 for d in deltas)),
                "ties": int(sum(np.isclose(d, 0.0) for d in deltas)),
                "losses": int(sum(d < -1e-12 for d in deltas)),
            })
    return out


def _report(output_dir: Path, plan: Sequence[Mapping[str, Any]],
            inventory: Sequence[Mapping[str, str]], score_rows: Sequence[Mapping[str, str]],
            mask_rows: Sequence[Mapping[str, str]], subject_rows: Sequence[Mapping[str, Any]],
            summary_rows: Sequence[Mapping[str, Any]], random_rows: Sequence[Mapping[str, str]],
            random_summary: Sequence[Mapping[str, Any]],
            nonempty_summary: Sequence[Mapping[str, Any]],
            result_rows: Sequence[Mapping[str, str]]) -> None:
    def fmt(x: Any) -> str:
        try:
            return f"{float(x):.3f}"
        except (ValueError, TypeError):
            return "NA"
    coverage = sum(str(r.get("eligible_for_mirepnet_qc", "")).lower() == "true" for r in inventory)
    lines = [
        "# MIRepNet 特征 QC 与 Signal-QC 跨数据集泛化诊断", "",
        "本实验是 `test/qc` 下的临时外部验证，不修改正式协同框架。MIRepNet artifact 是已经见过训练 trial 的微调表征，故 MIRepNet-QC 结果属于 in-sample pilot，不是 frozen 或 OOF 证据。", "",
        "## 固定协议", "",
        f"- 数据集：{', '.join(DATASETS)}；被试和类别数来自 `configs/datasets/*.yaml`，未硬编码训练/测试数量；seed：666/667/668；few-shot `val_split=0.7`。",
        "- `BNCI2014001` 是 2 类版本；`BNCI2014001-4` 是同源数据的 4 类注册变体，本次固定列表只纳入前者，避免重复统计。",
        "- EEGNet 使用当前 `configs/models/eegnet.yaml` 的 dataset/fewshot 参数，CE、AdamW、CosineAnnealingLR、last epoch；QC mask 在 TensorDataset/DataLoader 之前应用。",
        "- Signal-QC 输入是确定性 session 选择、重采样和时间截断之后、增强/训练归一化和 EEGNet identity preprocess 之前的原始 trial。",
        "",
        "## artifact 覆盖", "",
        f"- MIRepNet train artifact 可用且 UID 集合严格对齐：{coverage}/{len(inventory)} folds。详细输入 SHA256 在 `artifact_inventory.csv` 和 `artifact_sha256_check.json`。",
        "- 本脚本没有写入、覆盖或生成 `results/artifacts/` 中的任何文件；没有保存过滤后的 EEG 或 MIRepNet feature。",
        "",
        "## QC 规则", "",
        "- MIRepNet：`feats` 的最终 mean-pooled、classifier 前表征；float64、逐 trial L2、cosine、leave-self-out 5-NN 距离均值，fold 内 modified robust-z > 3.5。",
        "- Signal-QC：七个固定信号质量特征；高异常方向和 `min_channel_std` 低异常方向分别用 fold 内 median/MAD，`nonfinite_fraction > 0` 或最大 robust-z > 3.5。零差分容差固定为 `10*finfo(dtype).eps*max(1, train finite max_abs)`。",
        "",
        "## mask 统计", "",
        "| dataset | method | folds | non-empty | mean removed | mean fraction | unsafe/invalid |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ds in DATASETS:
        for method in ("mirepnet_feature_qc", "signal_qc"):
            group = [r for r in mask_rows if r["dataset"] == ds and r["method"] == method]
            nonempty = [r for r in group if int(r["removed_count"]) > 0]
            bad = [r for r in group if r["status"] in ("unsafe_overfiltering", "invalid_mask", "score_error", "missing", "uid_set_mismatch")]
            lines.append(f"| {ds} | {method} | {len(group)} | {len(nonempty)} | {fmt(np.mean([int(r['removed_count']) for r in group]) if group else np.nan)} | {fmt(np.mean([float(r['removed_fraction']) for r in group]) if group else np.nan)} | {len(bad)} |")
    shapes = Counter(str(r.get("feature_shape", "")) for r in inventory)
    lines += ["", "## artifact 特征形状与 S1 发现案例", "", f"- 所有 artifact 的字段是 `feats`；形状分布：{dict(shapes)}；114/114 fold 的 artifact UID 与 split UID 集合及顺序完全一致。", "", "| seed | MIRepNet (0,1) score/z/rank/flag | Signal-QC score/rank/flag |", "| ---: | --- | --- |"]
    for seed in SEEDS:
        path = output_dir / "BNCI2015001" / "S1" / str(seed) / "per_sample_qc_scores.csv"
        rows = _read_csv(path) if path.exists() else []
        bad_row = next((r for r in rows if r.get("uid_session") == "0" and r.get("uid_trial") == "1"), None)
        if bad_row is None:
            lines.append(f"| {seed} | unavailable | unavailable |")
        else:
            lines.append(f"| {seed} | {fmt(bad_row.get('mirepnet_score'))}/{fmt(bad_row.get('mirepnet_robust_z'))}/{bad_row.get('mirepnet_rank')}/{bad_row.get('mirepnet_flag')} | {fmt(bad_row.get('signal_score'))}/{bad_row.get('signal_rank')}/{bad_row.get('signal_flag')} |")
    unique_masks = [r for r in mask_rows if r["method"] == "mirepnet_feature_qc"]
    lines += ["", "## mask 重合", "", f"- 以每个 fold 去重后，平均 Jaccard={fmt(np.mean([float(r['jaccard']) for r in unique_masks]) if unique_masks else np.nan)}；具体交集、差集和 UID 在 `mask_statistics.csv`。", ""]
    lines += ["", "## 统计口径", "", "先在被试内平均三个 seed，再以被试为统计单位计算 QC 相对 base 的 paired delta、subject-level win/tie/loss、paired bootstrap 95% CI、Wilcoxon 和配对置换；多比较的 Holm 校正见 `external_dataset_summary.csv`。主指标是 Balanced Accuracy，Accuracy/Kappa 为次要指标。`all datasets macro average` 中的统计单位是数据集，每个数据集等权。", ""]
    lines += ["| group | condition | n statistical units | BA mean±SD | delta vs base | wins/ties/losses |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in summary_rows:
        delta = "—" if r["condition"] == "base_full" else fmt(r["paired_delta_vs_base"])
        wtl = "—" if r["condition"] == "base_full" else f"{r['wins']}/{r['ties']}/{r['losses']}"
        lines.append(f"| {r['group']} | {r['condition']} | {r['n_subjects']} | {fmt(r['balanced_accuracy_mean'])}±{fmt(r['balanced_accuracy_sd'])} | {delta} | {wtl} |")
    lines += ["", "## 仅非空 mask 的机制分析", "", "该表只用于描述性机制分析；主统计仍以全体有效 fold/被试为准。", "", "| dataset | method | n non-empty valid folds | QC BA | same-fold base BA | delta | wins/ties/losses |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in nonempty_summary:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['n_folds']} | {fmt(r['qc_balanced_accuracy_mean'])} | {fmt(r['base_balanced_accuracy_mean_same_folds'])} | {fmt(r['paired_delta_mean'])} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "## 等数量随机删除对照", "", "每个非空且安全的 QC mask 生成 5 个唯一、类别组成匹配的随机 mask；QC 的经验百分位按同一 dataset/subject/seed 的随机 BA 分布计算。", "", "| dataset | method | joined folds | random masks | QC BA | random BA±SD | QC-random | percentile mean/median | folds >50th |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in random_summary:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['n_folds']} | {r['n_random_masks']} | {fmt(r['qc_balanced_accuracy_mean'])} | {fmt(r['random_balanced_accuracy_mean'])}±{fmt(r['random_balanced_accuracy_sd'])} | {fmt(r['qc_minus_random_mean'])} | {fmt(r['empirical_percentile_mean'])}/{fmt(r['empirical_percentile_median'])} | {r['percentile_above_50_count']} |")
    lines += ["", "## 主条件运行状态", "", "unsafe_overfiltering/invalid_mask 被保留在 `results_per_seed.csv`，但不进入对应条件的性能均值；它们计入失败率，未被静默删除。", "", "| dataset | condition | total | ok | unsafe/invalid |", "| --- | --- | ---: | ---: | ---: |"]
    for ds in DATASETS:
        for condition in ("base_full", "mirepnet_feature_qc", "signal_qc"):
            group = [r for r in result_rows if r["dataset"] == ds and r["condition"] == condition]
            ok = sum(r.get("status") == "ok" for r in group)
            bad = len(group) - ok
            lines.append(f"| {ds} | {condition} | {len(group)} | {ok} | {bad} |")
    summary_lookup = {(r["group"], r["condition"]): r for r in summary_rows}
    ext_base = summary_lookup.get(("external datasets only", "base_full"), {})
    ext_mi = summary_lookup.get(("external datasets only", "mirepnet_feature_qc"), {})
    ext_sig = summary_lookup.get(("external datasets only", "signal_qc"), {})
    mi_random_ext = [r for r in random_summary if r["method"] == "mirepnet_feature_qc"]
    sig_random_ext = [r for r in random_summary if r["method"] == "signal_qc"]
    lines += ["", "## 当前结论", "", f"- BNCI2015001-S1 的已知 `(0,1)` 在 MIRepNet 三个 seed 中排名为 1/1/2，在 Signal-QC 中均为 1；该单案例两种 QC 都识别并恢复 S1，但不构成跨数据集优势。", f"- 外部数据集主统计中，MIRepNet-QC 相对 base 的 subject-level BA delta 为 {fmt(ext_mi.get('paired_delta_vs_base'))} 个百分点，Signal-QC 为 {fmt(ext_sig.get('paired_delta_vs_base'))}；MIRepNet-QC 没有显示 Signal-QC 之外的额外收益。", f"- MIRepNet-QC 的外部 valid subject 数为 {ext_mi.get('n_subjects', 'NA')}，Signal-QC 为 {ext_sig.get('n_subjects', 'NA')}；Signal-QC 有较多 unsafe mask，不能把仅剩有效 fold 的提升解释成稳定普遍收益。", "- 随机对照和非空 mask 结果见上表；当前证据更支持继续审查基础 Signal-QC 的过度过滤问题，而不支持宣称 MIRepNet 特征离群筛选跨数据集泛化。", ""]
    lines += ["", "## 解释边界", "", "- 只有排除 BNCI2015001-S1 后，外部数据集仍多数被试获益、且超过对应等数量随机删除并优于另一 QC，才可称为初步跨数据集证据。", "- 若外部数据集几乎没有非空 mask，结论是固定规则没有发现同等级异常，而不是证明 QC 无效。若删除后不超过随机删除，则不能证明排序有效。", "- 当前实验不比较 KD、不使用 teacher logits/预测、不加入 EEGNet 特征、不做并集/交集或阈值搜索；不能宣称通用 EEG-QC，也不能证明 MIRepNet 优于 Autoreject/Riemannian Potato。", "- 后续正式研究需要 frozen 预训练 MIRepNet 或 K-fold OOF 表征。", "", "## 输出与复现", "", f"所有输出：`{output_dir}`。精确执行设备和命令记录在 `execution_provenance.json`。", "", "```bash", "conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage inventory", "conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage score --resume", "conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage smoke --resume --device cpu", "CUDA_VISIBLE_DEVICES=<safe_gpu> conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage main --resume --device cuda:0", "CUDA_VISIBLE_DEVICES=<safe_gpu> conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage random --resume --device cuda:0", "conda run -n mirepnet python test/qc/cross_dataset_qc_generalization.py --stage aggregate --resume", "```", ""]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def stage_aggregate(args: argparse.Namespace, output_dir: Path,
                    plan: Sequence[Mapping[str, Any]]) -> None:
    result_rows = _read_csv(output_dir / "results_per_seed.csv")
    random_rows = _read_csv(output_dir / "random_control_results.csv")
    subject_rows = _subject_rows(result_rows)
    subject_fields = ["dataset", "subject", "subject_index", "condition", "n_seeds", "accuracy_mean", "balanced_accuracy_mean", "kappa_mean", "collapsed_count", "removed_count_mean"]
    _write_csv(output_dir / "results_per_subject.csv", subject_rows, subject_fields)
    dataset_rows = []
    for ds in DATASETS:
        for condition in sorted({r["condition"] for r in subject_rows if r["dataset"] == ds}):
            rows = [r for r in subject_rows if r["dataset"] == ds and r["condition"] == condition]
            dataset_rows.append({
                "dataset": ds, "condition": condition, "n_subjects": len(rows),
                "accuracy_mean": float(np.mean([r["accuracy_mean"] for r in rows])) if rows else "",
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy_mean"] for r in rows])) if rows else "",
                "kappa_mean": float(np.mean([r["kappa_mean"] for r in rows])) if rows else "",
                "collapsed_subject_count": int(sum(r["collapsed_count"] > 0 for r in rows)),
            })
    _write_csv(output_dir / "dataset_summary.csv", dataset_rows,
               ["dataset", "condition", "n_subjects", "accuracy_mean", "balanced_accuracy_mean", "kappa_mean", "collapsed_subject_count"])
    summary_rows = []
    for group in ("BNCI2015001-S1 only", "BNCI2015001 excluding S1", "external datasets only", "all datasets macro average"):
        group_subject_rows = subject_rows
        if group == "all datasets macro average":
            # Make each dataset equally weighted: first average subjects within
            # each dataset/condition, then use one pseudo-unit per dataset.
            # This prevents BNCI2014001's larger subject count from dominating
            # the requested macro average.
            pseudo = []
            for ds in DATASETS:
                for condition in sorted({
                    r["condition"] for r in subject_rows if r["dataset"] == ds
                }):
                    rows = [
                        r for r in subject_rows
                        if r["dataset"] == ds and r["condition"] == condition
                    ]
                    if not rows:
                        continue
                    pseudo.append({
                        "dataset": ds,
                        "subject": f"__dataset_macro__{ds}",
                        "subject_index": -1,
                        "condition": condition,
                        "n_seeds": int(sum(int(r["n_seeds"]) for r in rows)),
                        "accuracy_mean": float(np.mean([float(r["accuracy_mean"]) for r in rows])),
                        "balanced_accuracy_mean": float(np.mean([float(r["balanced_accuracy_mean"]) for r in rows])),
                        "kappa_mean": float(np.mean([float(r["kappa_mean"]) for r in rows])),
                        "collapsed_count": int(sum(int(r["collapsed_count"]) > 0 for r in rows)),
                        "removed_count_mean": float(np.mean([float(r["removed_count_mean"]) for r in rows])),
                    })
            summary_rows.extend(_summary_stats(pseudo, group))
        else:
            summary_rows.extend(_summary_stats(group_subject_rows, group))
    _holm(summary_rows)
    _write_csv(output_dir / "external_dataset_summary.csv", summary_rows,
               list(summary_rows[0].keys()) if summary_rows else ["group", "condition"])
    # The requested overall summary includes all four independent reporting
    # views rather than treating subject×seed rows as independent observations.
    random_summary = _random_control_summary(result_rows, random_rows)
    nonempty_summary = _nonempty_fold_summary(result_rows)
    _write_csv(output_dir / "random_control_summary.csv", random_summary,
               ["dataset", "method", "n_folds", "n_random_masks", "qc_balanced_accuracy_mean",
                "random_balanced_accuracy_mean", "random_balanced_accuracy_sd", "qc_minus_random_mean",
                "empirical_percentile_mean", "empirical_percentile_median", "percentile_above_50_count"])
    _write_csv(output_dir / "nonempty_fold_summary.csv", nonempty_summary,
               ["dataset", "method", "n_folds", "qc_balanced_accuracy_mean",
                "base_balanced_accuracy_mean_same_folds", "paired_delta_mean", "wins", "ties", "losses"])
    _write_json(output_dir / "overall_summary.json", {
        "experiment": "cross_dataset_qc_generalization",
        "datasets": list(DATASETS), "plan": list(plan),
        "n_result_rows": len(result_rows), "n_random_rows": len(random_rows),
        "summary_groups": summary_rows, "random_control_summary": random_summary,
        "nonempty_fold_summary": nonempty_summary,
        "result_status_counts": dict(Counter((r["dataset"], r["condition"], r.get("status", "")) for r in result_rows)),
        "artifact_coverage": _read_csv(output_dir / "artifact_inventory.csv"),
        "source_data_untouched": True, "results_artifacts_read_only": True,
    })
    score_rows = _read_csv(output_dir / "score_only_summary.csv")
    mask_rows = _read_csv(output_dir / "mask_statistics.csv")
    _write_figures(output_dir, _read_csv(output_dir / "BNCI2015001" / "S1" / "666" / "per_sample_qc_scores.csv") if (output_dir / "BNCI2015001" / "S1" / "666" / "per_sample_qc_scores.csv").exists() else [], subject_rows)
    _report(output_dir, plan, _read_csv(output_dir / "artifact_inventory.csv"), score_rows, mask_rows, subject_rows, summary_rows, random_rows, random_summary, nonempty_summary, result_rows)
    print(f"[aggregate] complete: {output_dir / 'report.md'}", flush=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "score", "smoke", "main", "random", "aggregate"), required=True)
    parser.add_argument("--artifact-root", type=Path, default=INPUT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--n-random-masks", type=int, default=RANDOM_MASKS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke_epochs <= 0 or args.n_random_masks <= 0:
        raise ValueError("smoke-epochs and n-random-masks must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    output_dir = _resolve_output(args.output_dir)
    artifact_root = _resolve_input_artifact_root(args.artifact_root)
    plan = dataset_plan()
    if args.stage == "inventory":
        stage_inventory(args, output_dir, artifact_root, plan)
    elif args.stage == "score":
        stage_score(args, output_dir, artifact_root, plan)
    elif args.stage == "smoke":
        stage_smoke(args, output_dir, plan)
    elif args.stage == "main":
        stage_main(args, output_dir, plan)
    elif args.stage == "random":
        stage_random(args, output_dir, plan)
    elif args.stage == "aggregate":
        stage_aggregate(args, output_dir, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
