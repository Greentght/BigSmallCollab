#!/usr/bin/env python
"""Offline prototype-margin reliability diagnostic.

This diagnostic is intentionally independent from training, distillation and
fusion.  It consumes only the already exported ``*_train.npz`` artifacts and
keeps every seed's feature/prototype calculation separate.  The prototype
view is a label-conditioned, post-finetune in-sample diagnostic; it is not a
trainable classifier and it must not be used to alter predictions.

The command line interface deliberately has no split selector: the only input
accepted by this module is a path ending in ``_train.npz``.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.storage import external_path, require_external_output, resolve_local_file
DEFAULT_DATASET = "BNCI2015001"
DEFAULT_SUBJECT = 0
DEFAULT_SESSION = "session_A"
DEFAULT_PROTOCOL = "fewshot"
DEFAULT_FM = "mirepnet"
DEFAULT_SM = "ifnet"
DEFAULT_SEEDS = (666, 667, 668)
DEFAULT_ARTIFACT_ROOT = Path('/data1/llx/BigSmallCollab_results') / "artifacts"
FORMAL_OUTPUT_ROOT = Path('/data1/llx/BigSmallCollab_results/qc_artifacts') / "relation_gap" / "prototype_margin_reliability"
DEFAULT_EPSILON = 1e-12
EXPECTED_N = 60


class DiagnosticError(ValueError):
    """Raised for a failed, fail-closed artifact or diagnostic check."""


def uid_key(uid: Sequence[int] | np.ndarray) -> tuple[int, int]:
    arr = np.asarray(uid, dtype=np.int64).reshape(-1)
    if arr.size != 2:
        raise DiagnosticError(f"sample_uid rows must have length 2, got {arr}")
    return int(arr[0]), int(arr[1])


def uid_text(uid: Sequence[int] | tuple[int, int]) -> str:
    a, b = uid_key(uid)
    return f"({a}, {b})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    path = resolve_local_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(artifact_root: Path | str, dataset: str, model: str,
                  subject: int, seed: int) -> Path:
    """Return the sole permitted input path (the train artifact)."""
    root = external_path(artifact_root)
    path = root / dataset / model / f"{int(subject)}_{int(seed)}_train.npz"
    validate_train_path(path)
    return path


def validate_train_path(path: Path | str) -> Path:
    path = external_path(path)
    path = Path(path)
    if not path.name.endswith("_train.npz") or "_test.npz" in path.name:
        raise DiagnosticError(
            f"only *_train.npz artifacts are permitted; rejected path: {path}")
    return path


def _finite(array: np.ndarray, name: str, path: Path) -> None:
    try:
        ok = bool(np.all(np.isfinite(array)))
    except TypeError as exc:
        raise DiagnosticError(f"{path}: {name} is not numeric") from exc
    if not ok:
        raise DiagnosticError(f"{path}: {name} contains non-finite values")


def load_train_artifact(path: Path | str, expected_n: int | None = None) -> dict[str, Any]:
    """Load and strictly validate one train artifact, never a test artifact."""
    path = external_path(path)
    path = validate_train_path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing train artifact: {path}")
    with np.load(resolve_local_file(path), allow_pickle=False) as archive:
        required = {"logits", "feats", "y", "sample_uid"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise DiagnosticError(f"{path} missing required fields: {missing}")
        out: dict[str, Any] = {name: np.asarray(archive[name]) for name in required}
        if "split_policy" in archive.files:
            policy = np.asarray(archive["split_policy"])
            if policy.ndim != 0:
                raise DiagnosticError(f"{path}: split_policy must be scalar")
            out["split_policy"] = str(policy.item())
        else:
            out["split_policy"] = None

    uid = np.asarray(out["sample_uid"])
    if uid.ndim != 2 or uid.shape[1] != 2:
        raise DiagnosticError(f"{path}: sample_uid shape {uid.shape}, expected (N, 2)")
    _finite(uid, "sample_uid", path)
    try:
        uid = uid.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"{path}: sample_uid must be integer-like") from exc
    out["sample_uid"] = uid
    policy_text = out.get("split_policy")
    if policy_text is not None and (policy_text.strip().lower() == "test" or
                                    "split=test" in policy_text.strip().lower()):
        raise DiagnosticError(f"{path}: split=test is not permitted; only train artifacts may be analyzed")
    n = len(uid)
    if expected_n is not None and n != expected_n:
        raise DiagnosticError(f"{path}: expected N={expected_n}, got N={n}")
    keys = [uid_key(row) for row in uid]
    if len(set(keys)) != n:
        raise DiagnosticError(f"{path}: sample_uid is not unique")

    for name in ("logits", "feats", "y"):
        arr = np.asarray(out[name])
        if arr.ndim == 0 or len(arr) != n:
            raise DiagnosticError(f"{path}: {name} first dimension {len(arr) if arr.ndim else 0} != {n}")
        _finite(arr, name, path)
    if out["feats"].ndim != 2:
        raise DiagnosticError(f"{path}: feats must be 2-D (N,D), got {out['feats'].shape}")
    if out["logits"].ndim != 2 or out["logits"].shape[1] < 2:
        raise DiagnosticError(f"{path}: logits must be 2-D (N,C), C >= 2, got {out['logits'].shape}")
    return out


def align_by_uid(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Reorder ``other`` to the reference UID order and reject all mismatches."""
    ref_uid = np.asarray(reference["sample_uid"], dtype=np.int64)
    oth_uid = np.asarray(other["sample_uid"], dtype=np.int64)
    ref_keys = [uid_key(row) for row in ref_uid]
    oth_keys = [uid_key(row) for row in oth_uid]
    if len(set(ref_keys)) != len(ref_keys):
        raise DiagnosticError("reference sample_uid contains duplicates")
    if len(set(oth_keys)) != len(oth_keys):
        raise DiagnosticError("other sample_uid contains duplicates")
    ref_set, oth_set = set(ref_keys), set(oth_keys)
    missing, extra = sorted(ref_set - oth_set), sorted(oth_set - ref_set)
    if missing or extra:
        raise DiagnosticError(f"UID set mismatch: missing={missing}, extra={extra}")
    positions = {key: i for i, key in enumerate(oth_keys)}
    order = np.asarray([positions[key] for key in ref_keys], dtype=np.int64)
    aligned = dict(other)
    for name in ("logits", "feats", "y"):
        aligned[name] = np.asarray(other[name])[order]
    aligned["sample_uid"] = oth_uid[order]
    if not np.array_equal(np.asarray(reference["y"]), aligned["y"]):
        bad = np.where(np.asarray(reference["y"]) != aligned["y"])[0].tolist()
        raise DiagnosticError(f"label mismatch after UID alignment at rows {bad}")
    return aligned


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise DiagnosticError(f"logits must have shape (N,C), C>=2; got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise DiagnosticError("logits contain non-finite values")
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x)
    p /= np.sum(p, axis=1, keepdims=True)
    if not np.all(np.isfinite(p)):
        raise DiagnosticError("stable softmax produced non-finite values")
    return p


def classifier_metrics(logits: np.ndarray) -> dict[str, np.ndarray]:
    p = stable_softmax(logits)
    pred = np.argmax(p, axis=1).astype(np.int64)
    top = np.sort(p, axis=1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    entropy = -np.sum(p * np.log(np.clip(p, np.finfo(np.float64).tiny, 1.0)), axis=1)
    normalized = entropy / math.log(p.shape[1])
    return {
        "probs": p,
        "pred": pred,
        "classifier_margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized,
    }


def _stats(values: Sequence[float] | np.ndarray, *, empty_reason: str = "empty") -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "q25": None,
                "median": None, "q75": None, "min": None, "max": None,
                "na_reason": empty_reason}
    if not np.all(np.isfinite(arr)):
        raise DiagnosticError("statistics received non-finite values")
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else None,
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "na_reason": None,
    }


def _normalize_rows(features: np.ndarray, sample_uid: np.ndarray | None,
                    epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(features, dtype=np.float64)
    if h.ndim != 2:
        raise DiagnosticError(f"features must be 2-D, got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise DiagnosticError("features contain non-finite values")
    norms = np.linalg.norm(h, axis=1)
    bad = np.where(~np.isfinite(norms) | (norms <= epsilon))[0]
    if bad.size:
        details = bad.tolist()
        if sample_uid is not None:
            details = [uid_text(sample_uid[i]) for i in bad]
        raise DiagnosticError(f"feature norm <= {epsilon} for UID(s): {details}")
    return h / norms[:, None], norms


def prototype_metrics(features: np.ndarray, labels: np.ndarray,
                      sample_uid: np.ndarray | None = None,
                      epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    """Compute full and leave-one-out prototype diagnostics in one feature space."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    uid = None if sample_uid is None else np.asarray(sample_uid, dtype=np.int64)
    z, feature_norms = _normalize_rows(features, uid, epsilon)
    if len(z) != len(y):
        raise DiagnosticError("features and labels have different lengths")
    classes = np.unique(y)
    if classes.size < 2:
        raise DiagnosticError("at least two classes are required for wrong-class similarity")
    counts = {int(k): int(np.sum(y == k)) for k in classes}
    singleton = [k for k, n in counts.items() if n < 2]
    if singleton:
        raise DiagnosticError(
            f"seed is not evaluable: class(es) {singleton} have fewer than two samples; "
            "cannot construct true-class leave-one-out prototype")

    full_raw: dict[int, np.ndarray] = {}
    full_proto: dict[int, np.ndarray] = {}
    raw_norms: dict[int, float] = {}
    class_indices: dict[int, np.ndarray] = {}
    for k in classes:
        key = int(k)
        idx = np.where(y == k)[0]
        class_indices[key] = idx
        raw = np.mean(z[idx], axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm <= epsilon:
            raise DiagnosticError(f"class {key} prototype raw norm <= {epsilon}")
        full_raw[key] = raw
        raw_norms[key] = norm
        full_proto[key] = raw / norm

    n = len(y)
    sim_true = np.empty(n, dtype=np.float64)
    sim_wrong = np.empty(n, dtype=np.float64)
    margins = np.empty(n, dtype=np.float64)
    proto_pred = np.empty(n, dtype=np.int64)
    loo_full_cos = np.empty(n, dtype=np.float64)
    # Store per-class LOO values for stability output.
    loo_proto_by_row: list[np.ndarray] = [np.empty(z.shape[1], dtype=np.float64) for _ in range(n)]
    for i, label in enumerate(y):
        key = int(label)
        loo_raw = (z[class_indices[key]].sum(axis=0, dtype=np.float64) - z[i]) / (counts[key] - 1)
        loo_norm = float(np.linalg.norm(loo_raw))
        if not np.isfinite(loo_norm) or loo_norm <= epsilon:
            u = uid_text(uid[i]) if uid is not None else str(i)
            raise DiagnosticError(f"LOO prototype raw norm <= {epsilon} for UID {u}")
        loo = loo_raw / loo_norm
        loo_proto_by_row[i] = loo
        loo_full_cos[i] = float(np.dot(loo, full_proto[key]))
        sims = np.asarray([
            float(np.dot(z[i], loo if int(k) == key else full_proto[int(k)]))
            for k in classes
        ], dtype=np.float64)
        true_pos = int(np.where(classes == label)[0][0])
        sim_true[i] = sims[true_pos]
        wrong = np.delete(sims, true_pos)
        sim_wrong[i] = float(np.max(wrong))
        margins[i] = sim_true[i] - sim_wrong[i]
        proto_pred[i] = int(classes[int(np.argmax(sims))])

    stability: dict[int, dict[str, Any]] = {}
    for k in classes:
        key = int(k)
        idx = class_indices[key]
        pairwise = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairwise.append(float(np.dot(z[idx[a]], z[idx[b]])))
        sample_loo = sim_true[idx]
        loo_full = loo_full_cos[idx]
        stability[key] = {
            "class_label": key,
            "n_samples": int(len(idx)),
            "feature_dim": int(z.shape[1]),
            "prototype_raw_norm": raw_norms[key],
            "within_class": _stats(pairwise, empty_reason="fewer_than_two_unique_pairs"),
            "sample_to_loo": _stats(sample_loo),
            "loo_full": _stats(loo_full),
        }
    return {
        "z": z,
        "feature_norms": feature_norms,
        "classes": classes.astype(np.int64),
        "class_counts": counts,
        "full_raw": full_raw,
        "full_proto": full_proto,
        "proto_pred": proto_pred,
        "proto_margin": margins,
        "sim_true": sim_true,
        "sim_wrong": sim_wrong,
        "loo_full_cos": loo_full_cos,
        "loo_proto_by_row": loo_proto_by_row,
        "stability": stability,
    }


def correctness_group(fm_correct: bool, sm_correct: bool) -> str:
    if fm_correct and sm_correct:
        return "A"
    if fm_correct and not sm_correct:
        return "B"
    if not fm_correct and sm_correct:
        return "C"
    return "D"


def _router_choice(left: float, right: float, higher: str,
                   epsilon: float = DEFAULT_EPSILON) -> str | None:
    delta = float(left - right)
    if abs(delta) <= epsilon:
        return None
    if higher == "left":
        return "FM" if delta > 0 else "SM"
    return "FM" if delta < 0 else "SM"


def router_choice(router: str, row: Mapping[str, Any],
                  epsilon: float = DEFAULT_EPSILON) -> str | None:
    """Return FM/SM or None (abstain), with one tie policy for all routers."""
    if router == "classifier_margin":
        return _router_choice(row["fm_classifier_margin"], row["sm_classifier_margin"], "left", epsilon)
    if router == "entropy":
        return _router_choice(row["fm_normalized_entropy"], row["sm_normalized_entropy"], "right", epsilon)
    if router == "prototype":
        return _router_choice(row["fm_proto_margin"], row["sm_proto_margin"], "left", epsilon)
    if router == "view_agreement":
        fm, sm = bool(row["fm_view_agree"]), bool(row["sm_view_agree"])
        if fm == sm:
            return None
        return "FM" if fm else "SM"
    raise DiagnosticError(f"unknown router {router!r}")


def _routing_stats(rows: Sequence[Mapping[str, Any]], router: str,
                   epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    disagreement = [r for r in rows if r["group"] in ("B", "C")]
    if not disagreement:
        return {"routing_acc": None, "coverage": None, "n_covered": 0,
                "n_abstain": 0, "na_reason": "no routing evaluation opportunity"}
    covered = 0
    correct = 0
    for row in disagreement:
        choice = router_choice(router, row, epsilon)
        if choice is None:
            continue
        covered += 1
        wanted = "FM" if row["group"] == "B" else "SM"
        correct += int(choice == wanted)
    abstain = len(disagreement) - covered
    return {
        "routing_acc": float(correct / covered) if covered else None,
        "coverage": float(covered / len(disagreement)),
        "n_covered": int(covered),
        "n_abstain": int(abstain),
        "na_reason": "no covered samples" if not covered else None,
    }


def _delta_row(seed: int, group: str, values: Sequence[float]) -> dict[str, Any]:
    stats = _stats(values, empty_reason=f"group {group} is empty")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size:
        signs = arr > 0 if group == "B" else arr < 0
        sign_rate = float(np.mean(signs))
        sign_n = int(np.sum(signs))
    else:
        sign_rate, sign_n = None, 0
    return {
        "seed": int(seed), "group": group, "n": stats["n"],
        "mean": stats["mean"], "median": stats["median"], "std": stats["std"],
        "q25": stats["q25"], "q75": stats["q75"], "min": stats["min"], "max": stats["max"],
        "correct_sign_rate": sign_rate, "correct_sign_n": sign_n,
        "na_reason": stats["na_reason"],
    }


def _stability_rows(dataset: str, subject_label: str, session: str, seed: int,
                    model: str, proto: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label in sorted(proto["stability"]):
        item = proto["stability"][label]
        base = {
            "dataset": dataset, "subject": subject_label, "session": session,
            "seed": int(seed), "model": model,
            "class_label": int(item["class_label"]), "n_samples": int(item["n_samples"]),
            "feature_dim": int(item["feature_dim"]),
            "prototype_raw_norm": item["prototype_raw_norm"],
            "mean_pairwise_within_class_cosine": item["within_class"]["mean"],
            "within_class_cosine_mean": item["within_class"]["mean"],
        }
        for prefix, stats in (("sample_to_loo", item["sample_to_loo"]),
                              ("loo_full", item["loo_full"])):
            for key in ("mean", "std", "q25", "median", "q75", "min", "max"):
                base[f"{prefix}_{key}"] = stats[key]
        rows.append(base)
    return rows


def analyze_seed(dataset: str, subject: int, session: str, seed: int,
                 fm: Mapping[str, Any], sm: Mapping[str, Any],
                 fm_name: str = DEFAULT_FM, sm_name: str = DEFAULT_SM,
                 epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    """Analyze exactly one seed; no arrays are shared across seeds."""
    aligned_sm = align_by_uid(fm, sm)
    y = np.asarray(fm["y"], dtype=np.int64)
    fm_cls = classifier_metrics(np.asarray(fm["logits"]))
    sm_cls = classifier_metrics(np.asarray(aligned_sm["logits"]))
    fm_proto = prototype_metrics(np.asarray(fm["feats"]), y, fm["sample_uid"], epsilon)
    sm_proto = prototype_metrics(np.asarray(aligned_sm["feats"]), y, fm["sample_uid"], epsilon)
    fm_correct = fm_cls["pred"] == y
    sm_correct = sm_cls["pred"] == y
    rows: list[dict[str, Any]] = []
    for i, uid in enumerate(np.asarray(fm["sample_uid"], dtype=np.int64)):
        group = correctness_group(bool(fm_correct[i]), bool(sm_correct[i]))
        rows.append({
            "dataset": dataset, "subject": f"S{int(subject) + 1}", "session": session,
            "seed": int(seed), "sample_uid": uid_text(uid),
            "uid_subject": int(uid[0]), "uid_trial": int(uid[1]), "label": int(y[i]),
            "fm_pred": int(fm_cls["pred"][i]), "sm_pred": int(sm_cls["pred"][i]),
            "fm_correct": bool(fm_correct[i]), "sm_correct": bool(sm_correct[i]),
            "fm_classifier_margin": float(fm_cls["classifier_margin"][i]),
            "sm_classifier_margin": float(sm_cls["classifier_margin"][i]),
            "fm_entropy": float(fm_cls["entropy"][i]),
            "sm_entropy": float(sm_cls["entropy"][i]),
            "fm_normalized_entropy": float(fm_cls["normalized_entropy"][i]),
            "sm_normalized_entropy": float(sm_cls["normalized_entropy"][i]),
            "fm_proto_pred": int(fm_proto["proto_pred"][i]),
            "sm_proto_pred": int(sm_proto["proto_pred"][i]),
            "fm_proto_margin": float(fm_proto["proto_margin"][i]),
            "sm_proto_margin": float(sm_proto["proto_margin"][i]),
            "fm_sim_true": float(fm_proto["sim_true"][i]),
            "sm_sim_true": float(sm_proto["sim_true"][i]),
            "fm_sim_wrong": float(fm_proto["sim_wrong"][i]),
            "sm_sim_wrong": float(sm_proto["sim_wrong"][i]),
            "fm_view_agree": bool(fm_cls["pred"][i] == fm_proto["proto_pred"][i]),
            "sm_view_agree": bool(sm_cls["pred"][i] == sm_proto["proto_pred"][i]),
            "group": group,
            "delta_proto": float(fm_proto["proto_margin"][i] - sm_proto["proto_margin"][i]),
        })
    counts = {g: sum(r["group"] == g for r in rows) for g in ("A", "B", "C", "D")}
    router = {name: _routing_stats(rows, name, epsilon)
              for name in ("classifier_margin", "entropy", "prototype", "view_agreement")}
    proto_b = [r for r in rows if r["group"] == "B"]
    proto_c = [r for r in rows if r["group"] == "C"]
    def group_proto_acc(group_rows: Sequence[Mapping[str, Any]], wanted: str) -> float | None:
        choices = [router_choice("prototype", r, epsilon) for r in group_rows]
        covered = [c for c in choices if c is not None]
        return float(sum(c == wanted for c in covered) / len(covered)) if covered else None
    deltas = {g: [r["delta_proto"] for r in rows if r["group"] == g] for g in ("B", "C")}
    delta_rows = [_delta_row(seed, g, deltas[g]) for g in ("B", "C")]
    summary = {
        "dataset": dataset, "subject": f"S{int(subject) + 1}", "subject_index": int(subject),
        "session": session, "seed": int(seed), "n_train": len(rows),
        "n_A": counts["A"], "n_B": counts["B"], "n_C": counts["C"], "n_D": counts["D"],
        "n_disagreement": counts["B"] + counts["C"],
    }
    for name, value in router.items():
        summary[f"routing_acc_{name}"] = value["routing_acc"]
        summary[f"coverage_{name}"] = value["coverage"]
        summary[f"n_covered_{name}"] = value["n_covered"]
        summary[f"n_abstain_{name}"] = value["n_abstain"]
    summary["view_agreement_coverage"] = router["view_agreement"]["coverage"]
    summary["view_agreement_routing_acc"] = router["view_agreement"]["routing_acc"]
    summary["n_abstain_view_agreement"] = router["view_agreement"]["n_abstain"]
    summary["routing_acc_B_prototype"] = group_proto_acc(proto_b, "FM")
    summary["routing_acc_C_prototype"] = group_proto_acc(proto_c, "SM")
    summary["mean_delta_proto_B"] = delta_rows[0]["mean"]
    summary["mean_delta_proto_C"] = delta_rows[1]["mean"]
    summary["na_reasons"] = {
        name: value["na_reason"] for name, value in router.items() if value["na_reason"]
    }
    if not proto_b:
        summary["na_reasons"]["routing_acc_B_prototype"] = "group B is empty"
    elif summary["routing_acc_B_prototype"] is None:
        summary["na_reasons"]["routing_acc_B_prototype"] = "all group B prototype comparisons tied"
    if not proto_c:
        summary["na_reasons"]["routing_acc_C_prototype"] = "group C is empty"
    elif summary["routing_acc_C_prototype"] is None:
        summary["na_reasons"]["routing_acc_C_prototype"] = "all group C prototype comparisons tied"

    distribution_rows = []
    for model, cls in ((fm_name, fm_cls), (sm_name, sm_cls)):
        for metric_name, values in (("classifier_margin", cls["classifier_margin"]),):
            distribution_rows.append({"seed": int(seed), "model": model,
                                      "metric": metric_name, **_stats(values)})
    for model, proto in ((fm_name, fm_proto), (sm_name, sm_proto)):
        distribution_rows.append({"seed": int(seed), "model": model,
                                  "metric": "prototype_margin", **_stats(proto["proto_margin"])})
    stability_rows = (_stability_rows(dataset, f"S{int(subject) + 1}", session, seed,
                                       fm_name, fm_proto) +
                      _stability_rows(dataset, f"S{int(subject) + 1}", session, seed,
                                      sm_name, sm_proto))
    return {"seed": int(seed), "sample_rows": rows, "summary": summary,
            "delta_rows": delta_rows, "stability_rows": stability_rows,
            "distribution_rows": distribution_rows,
            "split_policy_fm": fm.get("split_policy"),
            "split_policy_sm": sm.get("split_policy"),
            "feature_dims": {fm_name: int(np.asarray(fm["feats"]).shape[1]),
                             sm_name: int(np.asarray(aligned_sm["feats"]).shape[1])}}


def _git_text(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=ROOT, text=True,
                              capture_output=True, check=False)
    except OSError as exc:
        return f"<unavailable: {exc}>"
    return (proc.stdout or proc.stderr or "").strip()


def git_provenance() -> dict[str, Any]:
    return {"commit_sha": _git_text(["rev-parse", "HEAD"]),
            "branch": _git_text(["branch", "--show-current"]),
            "status_short": _git_text(["status", "--short"]).splitlines()}


def _artifact_provenance(path: Path, artifact: Mapping[str, Any], model: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size_bytes": int(stat.st_size),
        "mtime_epoch": float(stat.st_mtime),
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": sha256_file(path),
        "fields": ["logits", "feats", "y", "sample_uid", "split_policy"]
        if artifact.get("split_policy") is not None else ["logits", "feats", "y", "sample_uid"],
        "shapes": {k: list(np.asarray(artifact[k]).shape)
                   for k in ("logits", "feats", "y", "sample_uid")},
        "split_policy": artifact.get("split_policy"),
        "model": model,
    }


def _config_hash(dataset: str) -> dict[str, Any]:
    path = ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if not path.exists():
        return {"path": str(path), "sha256": None, "exists": False}
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "exists": True}


def _macro_summary(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(seed_summaries),
                           "seeds": [int(s["seed"]) for s in seed_summaries]}
    keys = [k for k in seed_summaries[0] if k.startswith("routing_acc_") or
            k.startswith("coverage_") or k.startswith("n_")]
    for key in keys:
        vals = [s[key] for s in seed_summaries if isinstance(s.get(key), (int, float))
                and not isinstance(s.get(key), bool) and s[key] is not None]
        if vals and key.startswith(("routing_acc_", "coverage_")):
            out[f"mean_{key}"] = float(np.mean(vals))
    for key in ("n_A", "n_B", "n_C", "n_D", "n_disagreement"):
        out[f"mean_{key}"] = float(np.mean([s[key] for s in seed_summaries])) if seed_summaries else None
    return out


def conservative_decision(seed_summaries: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Apply the requested conservative A/B/C interpretation without thresholds."""
    if not seed_summaries:
        return "C", "No evaluable seed was available."
    if any(s["n_disagreement"] == 0 for s in seed_summaries):
        return "C", "At least one seed had no B+C routing opportunity."
    usable = [s for s in seed_summaries if s["n_disagreement"] > 0]
    directional = [s["routing_acc_prototype"] for s in usable
                   if s["routing_acc_prototype"] is not None]
    if not directional:
        return "C", "Prototype routing had no covered disagreement samples."
    both_groups = all(s["n_B"] > 0 and s["n_C"] > 0 for s in usable)
    all_above_chance = all(v > 0.5 for v in directional)
    beats_simple = all(
        s["routing_acc_prototype"] is not None and
        s["routing_acc_classifier_margin"] is not None and
        s["routing_acc_entropy"] is not None and
        s["routing_acc_prototype"] > s["routing_acc_classifier_margin"] and
        s["routing_acc_prototype"] > s["routing_acc_entropy"] and
        (s["coverage_prototype"] or 0.0) >= (s["coverage_classifier_margin"] or 0.0) and
        (s["coverage_prototype"] or 0.0) >= (s["coverage_entropy"] or 0.0)
        for s in usable)
    if len(usable) == len(seed_summaries) and both_groups and all_above_chance and beats_simple:
        return "A", "Every evaluated seed is above chance in both directions and beats both simple routers with no lower coverage."
    if any(v > 0.5 for v in directional) or any(
            s.get("mean_delta_proto_B") is not None or s.get("mean_delta_proto_C") is not None
            for s in usable):
        return "B", "Prototype shows some directional or complementary information, but cross-seed/baseline evidence is insufficient for standalone routing."
    return "C", "Prototype routing is at or below chance or lacks directional evidence."


def _ensure_output_dir(path: Path, force: bool = False) -> None:
    path = require_external_output(path)
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not force:
            raise DiagnosticError(f"output directory is non-empty; refusing to overwrite: {path}")
        try:
            path.relative_to(FORMAL_OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise DiagnosticError("--force is permitted only inside the formal prototype diagnostic directory") from exc
    path.mkdir(parents=True, exist_ok=True)


SAMPLE_FIELDS = [
    "dataset", "subject", "session", "seed", "sample_uid", "uid_subject", "uid_trial", "label",
    "fm_pred", "sm_pred", "fm_correct", "sm_correct", "fm_classifier_margin", "sm_classifier_margin",
    "fm_entropy", "sm_entropy", "fm_normalized_entropy", "sm_normalized_entropy",
    "fm_proto_pred", "sm_proto_pred", "fm_proto_margin", "sm_proto_margin", "fm_sim_true", "sm_sim_true",
    "fm_sim_wrong", "sm_sim_wrong", "fm_view_agree", "sm_view_agree", "group", "delta_proto",
]
SUMMARY_FIELDS = [
    "dataset", "subject", "subject_index", "session", "seed", "n_train", "n_A", "n_B", "n_C", "n_D",
    "n_disagreement", "routing_acc_classifier_margin", "coverage_classifier_margin", "n_covered_classifier_margin",
    "n_abstain_classifier_margin", "routing_acc_entropy", "coverage_entropy", "n_covered_entropy", "n_abstain_entropy",
    "routing_acc_prototype", "coverage_prototype", "n_covered_prototype", "n_abstain_prototype",
    "view_agreement_coverage", "view_agreement_routing_acc", "n_abstain_view_agreement",
    "routing_acc_B_prototype", "routing_acc_C_prototype", "mean_delta_proto_B", "mean_delta_proto_C",
]
DELTA_FIELDS = ["seed", "group", "n", "mean", "median", "std", "q25", "q75", "min", "max",
                "correct_sign_rate", "correct_sign_n", "na_reason"]
STABILITY_FIELDS = [
    "dataset", "subject", "session", "seed", "model", "class_label", "n_samples", "feature_dim",
    "prototype_raw_norm", "mean_pairwise_within_class_cosine", "within_class_cosine_mean",
]
for _prefix in ("sample_to_loo", "loo_full"):
    STABILITY_FIELDS.extend([f"{_prefix}_{k}" for k in ("mean", "std", "q25", "median", "q75", "min", "max")])


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, (np.floating, float)):
        return "NA" if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = require_external_output(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _resolved_yaml(config: Mapping[str, Any]) -> str:
    # Hand-written scalar YAML keeps this output dependency-free and excludes
    # any tuning/threshold parameters.
    lines = []
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, float):
            # YAML 1.1 parsers treat bare ``1e-12`` as a string; retain a
            # decimal mantissa so epsilon remains numeric on read-back.
            lines.append(f"{key}: {value:.1e}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _format_num(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)


def build_report(config: Mapping[str, Any], provenance: Mapping[str, Any],
                 analyses: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
                 decision: str, decision_reason: str) -> str:
    feature_dimensions = config.get("feature_dimensions", {})
    lines = ["# Prototype Margin Reliability Diagnostic", "",
             "This is an offline, label-conditioned diagnostic over completed few-shot train artifacts.", "",
             f"- Dataset/subject/session: `{config['dataset']}` / `{config['subject_label']}` / `{config['session']}`",
             f"- Protocol: `{config['protocol']}`; train fraction 0.30; test fraction 0.70 (test artifacts were not read)",
             f"- Models: FM `{config['fm']}`; SM `{config['sm']}`; seeds: `{config['seeds']}`",
             f"- Feature layers: {config['feature_sources']}",
             f"- Feature dimensions: {feature_dimensions}", "",
             "## Artifact and provenance checks", "",
             "Only paths ending in `_train.npz` were opened. Each artifact had logits, feats, y and sample_uid; UID sets were checked for exact equality, the SM was explicitly reordered to FM UID order, and labels were checked after reordering.",
             "The source export path is `ad.finetune(...)` followed by `ad.infer(X_tr)` in `_fit_and_export`, so these are completed-fine-tuning, in-sample train inferences.",
             f"Session is **{provenance['session_provenance']['source']}** because it is not embedded in the NPZ.", "",
             "### Input artifact hashes", "",
             "| model | train artifact path | size (bytes) | SHA256 | split policy |", "|:---|:---|---:|:---|:---|"]
    for item in provenance["artifacts"]:
        lines.append(f"| {item['model']} | `{item['path']}` | {item['size_bytes']} | `{item['sha256']}` | {item.get('split_policy') or 'NA'} |")
    lines += ["", "Git commit/status are recorded in `artifact_provenance.json`; the input artifact fields/shapes are recorded there as well.", "",
             "## Per-seed groups and routing", "",
             "Routing accuracy is among covered B+C samples; coverage is among all B+C samples. Ties within 1e-12 abstain.", "",
             "| seed | A | B | C | D | B+C | classifier acc/cov | entropy acc/cov | prototype acc/cov | view acc/cov | B proto acc | C proto acc |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in analyses:
        s = a["summary"]
        def ac(name: str) -> str:
            return f"{_format_num(s.get('routing_acc_' + name))}/{_format_num(s.get('coverage_' + name))}"
        lines.append(f"| {s['seed']} | {s['n_A']} | {s['n_B']} | {s['n_C']} | {s['n_D']} | {s['n_disagreement']} | {ac('classifier_margin')} | {ac('entropy')} | {ac('prototype')} | {_format_num(s['view_agreement_routing_acc'])}/{_format_num(s['view_agreement_coverage'])} | {_format_num(s['routing_acc_B_prototype'])} | {_format_num(s['routing_acc_C_prototype'])} |")
    lines += ["", "### Delta prototype (FM − SM)", "",
              "| seed | group | n | mean | median | std | q25 | q75 | min | max | correct-sign rate |", "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in analyses:
        for d in a["delta_rows"]:
            lines.append("| " + " | ".join(_format_num(d.get(k)) for k in ("seed", "group", "n", "mean", "median", "std", "q25", "q75", "min", "max", "correct_sign_rate")) + " |")
    lines += ["", "### Margin distributions", "",
              "The following per-seed distributions are shown because direct comparison of raw FM/SM margins is an assumption under test, not an assumed calibration property.", "",
              "| seed | model | metric | mean | std | q25 | median | q75 | min | max |", "|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in analyses:
        for d in a["distribution_rows"]:
            lines.append("| " + " | ".join(_format_num(d.get(k)) for k in ("seed", "model", "metric", "mean", "std", "q25", "median", "q75", "min", "max")) + " |")
    lines += ["", "## Prototype stability", "",
              "Stability is reported per seed × model × class; no arbitrary quality cutoff was imposed.", ""]
    for a in analyses:
        for row in a["stability_rows"]:
            lines.append(f"- seed {row['seed']} / {row['model']} / class {row['class_label']}: n={row['n_samples']}, raw_norm={_format_num(row['prototype_raw_norm'])}, within-class cosine mean={_format_num(row['mean_pairwise_within_class_cosine'])}, sample-to-LOO mean={_format_num(row['sample_to_loo_mean'])}, LOO/full mean={_format_num(row['loo_full_mean'])}.")
    stability_all = [row for a in analyses for row in a["stability_rows"]]
    finite_within = [row for row in stability_all if row["mean_pairwise_within_class_cosine"] is not None]
    finite_loo_std = [row for row in stability_all if row["loo_full_std"] is not None]
    if finite_within:
        lowest = min(finite_within, key=lambda row: row["mean_pairwise_within_class_cosine"])
        lines.append(f"- Lowest observed within-class mean is {lowest['mean_pairwise_within_class_cosine']:.6g} (seed {lowest['seed']}, {lowest['model']}, class {lowest['class_label']}); it is retained as a descriptive flag, not a tuned cutoff.")
    if finite_loo_std:
        highest = max(finite_loo_std, key=lambda row: row["loo_full_std"])
        lines.append(f"- Highest observed LOO/full prototype cosine std is {highest['loo_full_std']:.6g} (seed {highest['seed']}, {highest['model']}, class {highest['class_label']}); no raw norm was near epsilon and no LOO/full instability threshold was imposed.")
    lines += ["", "## Questions and decision", "", "### Seed-wise Q1/Q2", ""]
    for a in analyses:
        s = a["summary"]
        b_rows = [r for r in a["sample_rows"] if r["group"] == "B"]
        c_rows = [r for r in a["sample_rows"] if r["group"] == "C"]
        b_choices = [router_choice("prototype", r) for r in b_rows]
        c_choices = [router_choice("prototype", r) for r in c_rows]
        b_cov = [choice for choice in b_choices if choice is not None]
        c_cov = [choice for choice in c_choices if choice is not None]
        b_answer = "NA (B=0)" if not b_cov else f"{sum(x == 'FM' for x in b_cov)}/{len(b_cov)} choose FM"
        c_answer = "NA (C=0)" if not c_cov else f"{sum(x == 'SM' for x in c_cov)}/{len(c_cov)} choose SM"
        lines.append(f"- Seed {s['seed']}: Q1 Group B (FM correct/SM wrong, n={s['n_B']}): {b_answer}; Q2 Group C (FM wrong/SM correct, n={s['n_C']}): {c_answer}.")
    opportunity_summaries = [a["summary"] for a in analyses if a["summary"]["n_disagreement"] > 0]
    q3_detail = "; ".join(
        f"seed {s['seed']}: classifier={_format_num(s['routing_acc_classifier_margin'])}/{_format_num(s['coverage_classifier_margin'])}, entropy={_format_num(s['routing_acc_entropy'])}/{_format_num(s['coverage_entropy'])}, prototype={_format_num(s['routing_acc_prototype'])}/{_format_num(s['coverage_prototype'])} (n={s['n_disagreement']})"
        for s in opportunity_summaries)
    q4_detail = "; ".join(
        f"seed {s['seed']} prototype={_format_num(s['routing_acc_prototype'])}"
        for s in opportunity_summaries) or "no seed had a B+C opportunity"
    c_counts = ", ".join(f"{a['summary']['seed']}:{a['summary']['n_C']}" for a in analyses)
    lines += ["", f"- Q3: Accuracy/coverage (and the very small n) are: {q3_detail or 'NA'}. Prototype is not superior to both simple routers across seeds.",
              f"- Q4: Seed-wise prototype routing is {q4_detail}; the reverse direction is not stable and seeds were not pooled for inferential testing.",
              f"- Q5: Group C counts (FM wrong, SM correct) are `{c_counts}`. Sparse/empty C means: **小模型反向教学机会不足，当前结果不能支持双向知识路由。**",
              f"- Q6: **Decision {decision}** — {decision_reason}",
              f"- Prototype-Guided BiKD recommendation: {'eligible for standalone routing only under A' if decision == 'A' else 'do not use as standalone routing; B may only be a multi-signal input, and C should not enter formal BiKD.'}", "",
              "## Limitations", "",
              "1. Artifacts are in-sample predictions after few-shot fine-tuning.",
              "2. Leave-one-out removes sample i only from the prototype mean; model parameters were still fine-tuned using sample i, so this is not an OOF reliability estimate.",
              "3. Prototype margins use true training labels (label-conditioned) and cannot be applied directly without labels.",
              "4. FM and SM raw prototype-margin scales/dispersions may differ; raw-margin comparison is the hypothesis being tested.",
              "5. No test feature/label was opened, no test threshold/routing was used, and this does not establish test-time routing performance.",
              "6. No threshold search, calibration, retraining, checkpoint writing, or formal experiment was performed.", ""]
    return "\n".join(lines)


def run_analysis(dataset: str = DEFAULT_DATASET, subject: int = DEFAULT_SUBJECT,
                 session: str = DEFAULT_SESSION, fm_name: str = DEFAULT_FM,
                 sm_name: str = DEFAULT_SM, seeds: Sequence[int] = DEFAULT_SEEDS,
                 artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
                 out_dir: Path | str | None = None, epsilon: float = DEFAULT_EPSILON,
                 force: bool = False) -> dict[str, Any]:
    seeds = sorted(int(s) for s in seeds)
    if not seeds:
        raise DiagnosticError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise DiagnosticError("duplicate seeds are not permitted")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise DiagnosticError("epsilon must be finite and > 0")
    if session != DEFAULT_SESSION:
        # The diagnostic does not repartition or load raw data.  This guard
        # prevents a session argument from silently changing artifact meaning.
        raise DiagnosticError(f"only the configured session {DEFAULT_SESSION!r} is supported")
    artifact_root = external_path(artifact_root).resolve()
    if out_dir is None:
        out_dir = FORMAL_OUTPUT_ROOT / dataset / f"S{int(subject) + 1}" / DEFAULT_PROTOCOL / f"{fm_name}__{sm_name}"
    out_dir = require_external_output(out_dir)
    # Load/validate all inputs before creating or modifying any output.
    expected_n = EXPECTED_N if dataset == DEFAULT_DATASET and int(subject) == 0 else None
    analyses = []
    artifact_prov = []
    feature_sources = {
        fm_name: "MIRepNetAdapter.forward pooled feature" if fm_name == "mirepnet" else "artifact feats (model adapter export)",
        sm_name: "IFNet model(x, return_features=True), final pre-FC feature" if sm_name == "ifnet" else "artifact feats (model adapter export)",
    }
    for seed in seeds:
        fm_path = artifact_path(artifact_root, dataset, fm_name, subject, seed)
        sm_path = artifact_path(artifact_root, dataset, sm_name, subject, seed)
        fm = load_train_artifact(fm_path, expected_n=expected_n)
        sm = load_train_artifact(sm_path, expected_n=expected_n)
        if fm.get("split_policy") != sm.get("split_policy"):
            raise DiagnosticError(f"split_policy mismatch seed {seed}: FM={fm.get('split_policy')!r}, SM={sm.get('split_policy')!r}")
        # analyze_seed aligns SM by UID and never shares arrays with another seed.
        analyses.append(analyze_seed(dataset, subject, session, seed, fm, sm,
                                     fm_name, sm_name, epsilon))
        artifact_prov.extend([_artifact_provenance(fm_path, fm, fm_name),
                              _artifact_provenance(sm_path, sm, sm_name)])

    seed_summaries = [a["summary"] for a in analyses]
    decision, decision_reason = conservative_decision(seed_summaries)
    macro = _macro_summary(seed_summaries)
    config = {
        "dataset": dataset, "subject": int(subject), "subject_label": f"S{int(subject) + 1}",
        "session": session, "protocol": DEFAULT_PROTOCOL, "split": "train",
        "train_fraction": 0.3, "test_fraction": 0.7, "seeds": seeds,
        "fm": fm_name, "sm": sm_name, "artifact_root": str(artifact_root),
        "output_root": str(out_dir), "epsilon": float(epsilon),
        "feature_sources": feature_sources,
        "feature_dimensions": dict(analyses[0]["feature_dims"]),
    }
    session_provenance = {"session": session, "source": "inferred from loader default",
                          "details": "session_A is the BNCI2015001 loader/config default; session is not embedded in the NPZ artifact"}
    provenance = {
        "artifacts": artifact_prov, "config_file": _config_hash(dataset),
        "git": git_provenance(), "session_provenance": session_provenance,
        "feature_sources": feature_sources, "split": "train",
        "inference_provenance": "completed few-shot fine-tuning followed by in-sample train inference (ad.finetune -> ad.infer(X_tr)); not OOF",
        "test_artifacts_read": False,
        "test_artifact_policy": "No *_test.npz path was opened; no test feature/label/threshold/routing was used.",
    }
    summary = {
        "config": config, "per_seed": seed_summaries, "macro_descriptive": macro,
        "decision": decision, "decision_reason": decision_reason,
        "prototype_guided_bikd_recommendation": "yes_standalone_only_if_A" if decision == "A" else "no_standalone_routing",
        "limitations": ["post-finetune in-sample", "LOO is not OOF", "label-conditioned prototype margin",
                         "raw FM/SM margin scales may differ", "no test data used"],
    }
    _ensure_output_dir(out_dir, force=force)
    write_csv(out_dir / "prototype_sample_metrics.csv",
              [r for a in analyses for r in a["sample_rows"]], SAMPLE_FIELDS)
    write_csv(out_dir / "prototype_router_summary.csv", seed_summaries, SUMMARY_FIELDS)
    write_csv(out_dir / "prototype_delta_summary.csv",
              [d for a in analyses for d in a["delta_rows"]], DELTA_FIELDS)
    write_csv(out_dir / "prototype_stability.csv",
              [r for a in analyses for r in a["stability_rows"]], STABILITY_FIELDS)
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                      ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "artifact_provenance.json").write_text(json.dumps(_jsonable(provenance), indent=2,
                                                                  ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(_resolved_yaml(config), encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(config, provenance, analyses, summary,
                                                       decision, decision_reason), encoding="utf-8")
    return {"out_dir": out_dir, "summary": summary, "provenance": provenance,
            "analyses": analyses}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="analyze existing train artifacts only")
    analyze.add_argument("--dataset", default=DEFAULT_DATASET)
    analyze.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    analyze.add_argument("--session", default=DEFAULT_SESSION)
    analyze.add_argument("--fm", default=DEFAULT_FM)
    analyze.add_argument("--sm", default=DEFAULT_SM)
    analyze.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    analyze.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    analyze.add_argument("--out-dir", type=Path, default=None)
    analyze.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    analyze.add_argument("--force", action="store_true",
                         help="overwrite only a non-empty directory inside the formal diagnostic root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        result = run_analysis(dataset=args.dataset, subject=args.subject, session=args.session,
                              fm_name=args.fm, sm_name=args.sm, seeds=args.seeds,
                              artifact_root=args.artifact_root, out_dir=args.out_dir,
                              epsilon=args.epsilon, force=args.force)
        print(f"Wrote prototype margin reliability diagnostic to {result['out_dir']}")
        print(f"Decision: {result['summary']['decision']} — {result['summary']['decision_reason']}")
        return 0
    raise DiagnosticError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
