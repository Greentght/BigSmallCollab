#!/usr/bin/env python3
"""Paired evaluation of pretrained and random-init CodeBrain artifacts.

The subject is the statistical unit. Each pair is admitted only when labels,
sample UIDs, and split policy match exactly. Reports are written to
``/data1/llx/BigSmallCollab_results/codebrain/pretrain_ablation_seed666`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.storage import (RESULTS_ROOT, external_path,
                                 require_external_output, resolve_local_file)

DEFAULT_ARTIFACT_ROOT = RESULTS_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "codebrain" / "pretrain_ablation_seed666"
DATASET_SUBJECTS = {
    "BNCI2014001": 9,
    "BNCI2014004": 9,
    "BNCI2015001": 12,
    "AlexMI": 8,
}
PRETRAINED = "codebrain"
RANDOM = "codebrain_random"
METRICS = ("accuracy_pct", "balanced_accuracy_pct", "kappa")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 666


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _policy_string(value: np.ndarray) -> str:
    if value.size != 1:
        raise ValueError(f"split_policy must be a scalar, got shape {value.shape}")
    raw = value.reshape(()).item() if value.ndim == 0 else value.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    result = str(raw).strip()
    if not result:
        raise ValueError("split_policy is empty")
    return result


def _metrics_from_logits(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (N, C), got {logits.shape}")
    if logits.shape[0] != y.shape[0]:
        raise ValueError(f"logits/y length mismatch: {logits.shape[0]} != {y.shape[0]}")
    if logits.shape[0] == 0:
        raise ValueError("artifact has no test samples")
    if logits.shape[1] < 1:
        raise ValueError("logits has no class columns")
    if not np.issubdtype(logits.dtype, np.number) or not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite numeric values")

    observed_classes = np.unique(y)
    n_classes = logits.shape[1]
    if len(observed_classes) == n_classes:
        class_labels = observed_classes
    elif (
        np.issubdtype(y.dtype, np.integer)
        and len(observed_classes)
        and int(observed_classes.min()) >= 0
        and int(observed_classes.max()) < n_classes
    ):
        # A test split can omit a class; use the conventional zero-based
        # mapping from output columns to labels in that case.
        class_labels = np.arange(n_classes, dtype=y.dtype)
    else:
        raise ValueError(
            f"cannot map {n_classes} logit columns to observed labels "
            f"{observed_classes.tolist()}"
        )

    predicted_columns = logits.argmax(axis=1)
    y_pred = class_labels[predicted_columns]
    accuracy = float(np.mean(y_pred == y) * 100.0)
    balanced_accuracy = float(
        np.mean([np.mean(y_pred[y == label] == label) for label in observed_classes]) * 100.0
    )

    label_to_index = {
        label.item() if isinstance(label, np.generic) else label: i
        for i, label in enumerate(class_labels)
    }
    y_idx = np.fromiter(
        (label_to_index[v.item() if isinstance(v, np.generic) else v] for v in y),
        dtype=np.int64,
        count=len(y),
    )
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_idx, predicted_columns.astype(np.int64, copy=False)), 1)
    expected_agreement = float(
        np.dot(confusion.sum(axis=1), confusion.sum(axis=0)) / (len(y) ** 2)
    )
    observed_agreement = float(np.trace(confusion) / len(y))
    denominator = 1.0 - expected_agreement
    kappa = (
        (observed_agreement - expected_agreement) / denominator
        if denominator != 0.0
        else float("nan")
    )

    counts = {
        str(label.item() if isinstance(label, np.generic) else label): int(
            np.count_nonzero(y_pred == label)
        )
        for label in class_labels
    }
    predicted_classes = np.unique(y_pred)
    collapsed = len(predicted_classes) == 1
    collapsed_to = (
        predicted_classes[0].item()
        if collapsed and isinstance(predicted_classes[0], np.generic)
        else (predicted_classes[0] if collapsed else None)
    )
    return {
        "n_samples": int(len(y)),
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "kappa": kappa,
        "predicted_class_count": int(len(predicted_classes)),
        "single_class_prediction_collapse": collapsed,
        "collapsed_to_class": collapsed_to,
        "predicted_class_counts": counts,
    }


@dataclass
class Artifact:
    dataset: str
    subject0: int
    seed: int
    model: str
    path: Path
    exists: bool = False
    errors: list[str] = field(default_factory=list)
    y: np.ndarray | None = None
    sample_uid: np.ndarray | None = None
    split_policy: str | None = None
    metrics: dict[str, Any] | None = None

    @property
    def valid_for_alignment(self) -> bool:
        return (
            self.exists
            and not self.errors
            and self.y is not None
            and self.sample_uid is not None
            and self.split_policy is not None
            and self.metrics is not None
        )


def _load_artifact(dataset: str, subject0: int, seed: int, model: str,
                   path: Path) -> Artifact:
    path = external_path(path)
    result = Artifact(dataset, subject0, seed, model, path)
    if not path.is_file():
        return result
    result.exists = True
    try:
        with np.load(resolve_local_file(path), allow_pickle=False) as data:
            fields = set(data.files)
            required = {"logits", "y", "sample_uid", "split_policy"}
            missing = sorted(required - fields)
            if missing:
                result.errors.append("missing required fields: " + ", ".join(missing))

            logits = np.asarray(data["logits"]) if "logits" in fields else None
            y = np.asarray(data["y"]) if "y" in fields else None
            uid = np.asarray(data["sample_uid"]) if "sample_uid" in fields else None

            if y is not None:
                if y.ndim != 1:
                    result.errors.append(f"y must be one-dimensional, got {y.shape}")
                elif y.size == 0:
                    result.errors.append("y has no test samples")
                else:
                    result.y = y

            if uid is not None:
                if uid.ndim != 2 or uid.shape[1] != 2:
                    result.errors.append(f"sample_uid must have shape (N, 2), got {uid.shape}")
                elif y is not None and y.ndim == 1 and uid.shape[0] != y.shape[0]:
                    result.errors.append(
                        f"sample_uid/y length mismatch: {uid.shape[0]} != {y.shape[0]}"
                    )
                else:
                    result.sample_uid = uid

            if "split_policy" in fields:
                try:
                    result.split_policy = _policy_string(np.asarray(data["split_policy"]))
                except (TypeError, ValueError, UnicodeDecodeError) as exc:
                    result.errors.append(f"invalid split_policy: {exc}")

            if logits is not None and y is not None and y.ndim == 1 and y.size:
                try:
                    result.metrics = _metrics_from_logits(logits, y)
                    if uid is not None and uid.shape[0] != y.shape[0]:
                        # Keep model metrics visible, but disqualify this run
                        # from paired aggregation.
                        pass
                except (ValueError, TypeError, IndexError, KeyError) as exc:
                    result.errors.append(f"invalid prediction arrays: {exc}")
    except Exception as exc:  # report corrupt or unreadable NPZ files too
        result.errors.append(f"cannot read NPZ: {type(exc).__name__}: {exc}")
    return result


def _pair_row(pretrained: Artifact, random: Artifact) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": pretrained.dataset,
        "subject0": pretrained.subject0,
        "seed": pretrained.seed,
        "status": "matched",
        "pretrained_exists": pretrained.exists,
        "random_exists": random.exists,
        "y_equal": None,
        "sample_uid_equal": None,
        "split_policy_equal": None,
        "reason": "",
    }
    for metric in METRICS:
        p_value = pretrained.metrics.get(metric) if pretrained.metrics else None
        r_value = random.metrics.get(metric) if random.metrics else None
        row[f"pretrained_{metric}"] = p_value
        row[f"random_{metric}"] = r_value
        row[f"delta_{metric}"] = (
            float(p_value - r_value)
            if p_value is not None and r_value is not None
            and math.isfinite(float(p_value)) and math.isfinite(float(r_value))
            else None
        )

    if not pretrained.exists or not random.exists:
        absent = []
        if not pretrained.exists:
            absent.append(PRETRAINED)
        if not random.exists:
            absent.append(RANDOM)
        row["status"] = "missing_artifact"
        row["reason"] = "missing: " + ", ".join(absent)
        return row
    if not pretrained.valid_for_alignment or not random.valid_for_alignment:
        problems = []
        for artifact in (pretrained, random):
            if not artifact.valid_for_alignment:
                detail = "; ".join(artifact.errors) or "required metadata or metrics unavailable"
                problems.append(f"{artifact.model}: {detail}")
        row["status"] = "invalid_artifact"
        row["reason"] = " | ".join(problems)
        return row

    assert pretrained.y is not None and random.y is not None
    assert pretrained.sample_uid is not None and random.sample_uid is not None
    y_equal = np.array_equal(pretrained.y, random.y)
    uid_equal = np.array_equal(pretrained.sample_uid, random.sample_uid)
    policy_equal = pretrained.split_policy == random.split_policy
    row.update(y_equal=y_equal, sample_uid_equal=uid_equal,
               split_policy_equal=policy_equal)
    mismatches = []
    if not y_equal:
        mismatches.append("y mismatch")
    if not uid_equal:
        mismatches.append("sample_uid mismatch")
    if not policy_equal:
        mismatches.append("split_policy mismatch")
    if mismatches:
        row["status"] = "mismatch"
        row["reason"] = "; ".join(mismatches)
        for metric in METRICS:
            row[f"delta_{metric}"] = None
    return row


def _bootstrap_ci(values: np.ndarray, label: str) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    seed = (BOOTSTRAP_SEED + zlib.crc32(label.encode("utf-8"))) % (2**32)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summary_row(label: str, group_rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    eligible = [row for row in group_rows if row["status"] == "matched"]
    result: dict[str, Any] = {
        "group": label,
        "expected_pairs": expected,
        "matched_pairs": len(eligible),
        "status": "complete" if len(eligible) == expected else "incomplete",
        "pretrained_collapse_runs": 0,
        "random_collapse_runs": 0,
    }
    # Collapse counts include all readable per-run metrics in the group,
    # including unpaired runs, and are diagnostic rather than pair estimates.
    result["pretrained_collapse_runs"] = sum(
        bool(row.get("pretrained_single_class_prediction_collapse")) for row in group_rows
    )
    result["random_collapse_runs"] = sum(
        bool(row.get("random_single_class_prediction_collapse")) for row in group_rows
    )

    for metric in METRICS:
        triplets = [
            (
                float(row[f"pretrained_{metric}"]),
                float(row[f"random_{metric}"]),
                float(row[f"delta_{metric}"]),
            )
            for row in eligible
            if all(
                row[key] is not None and math.isfinite(float(row[key]))
                for key in (
                    f"pretrained_{metric}",
                    f"random_{metric}",
                    f"delta_{metric}",
                )
            )
        ]
        p_values = [values[0] for values in triplets]
        r_values = [values[1] for values in triplets]
        deltas = [values[2] for values in triplets]
        # A matched pair normally has finite values for all metrics. Keep the
        # guards explicit so undefined kappa values are never serialized as NaN.
        n = len(triplets)
        if n:
            wins = sum(delta > 1e-12 for delta in deltas)
            losses = sum(delta < -1e-12 for delta in deltas)
            ties = n - wins - losses
            ci_low, ci_high = _bootstrap_ci(np.asarray(deltas), f"{label}:{metric}")
            result.update({
                f"n_{metric}": n,
                f"pretrained_mean_{metric}": float(np.mean(p_values)),
                f"random_mean_{metric}": float(np.mean(r_values)),
                f"mean_delta_{metric}": float(np.mean(deltas)),
                f"ci95_low_delta_{metric}": ci_low,
                f"ci95_high_delta_{metric}": ci_high,
                f"wins_{metric}": int(wins),
                f"ties_{metric}": int(ties),
                f"losses_{metric}": int(losses),
            })
        else:
            result.update({
                f"n_{metric}": 0,
                f"pretrained_mean_{metric}": None,
                f"random_mean_{metric}": None,
                f"mean_delta_{metric}": None,
                f"ci95_low_delta_{metric}": None,
                f"ci95_high_delta_{metric}": None,
                f"wins_{metric}": 0,
                f"ties_{metric}": 0,
                f"losses_{metric}": 0,
            })
    return result


def _run_rows(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        metrics = artifact.metrics or {}
        rows.append({
            "dataset": artifact.dataset,
            "subject0": artifact.subject0,
            "seed": artifact.seed,
            "model": artifact.model,
            "path": str(artifact.path),
            "exists": artifact.exists,
            "valid": artifact.valid_for_alignment,
            "n_samples": metrics.get("n_samples"),
            "accuracy_pct": metrics.get("accuracy_pct"),
            "balanced_accuracy_pct": metrics.get("balanced_accuracy_pct"),
            "kappa": metrics.get("kappa"),
            "predicted_class_count": metrics.get("predicted_class_count"),
            "single_class_prediction_collapse": metrics.get("single_class_prediction_collapse"),
            "collapsed_to_class": metrics.get("collapsed_to_class"),
            "predicted_class_counts_json": json.dumps(
                metrics.get("predicted_class_counts"), ensure_ascii=False, sort_keys=True
            ) if metrics else None,
            "split_policy": artifact.split_policy,
            "errors": "; ".join(artifact.errors),
        })
    return rows


def _csv_value(value: Any) -> Any:
    return _json_safe(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown(summary_rows: list[dict[str, Any]], seed: int, output_dir: Path) -> str:
    lines = [
        f"# CodeBrain pretraining ablation (seed {seed})",
        "",
        "Positive delta means pretrained CodeBrain scored higher than the same-architecture random-init control.",
        "Each mean is over paired subjects; confidence intervals resample subjects with replacement (10,000 draws, fixed base seed 666).",
        "Accuracy and balanced accuracy are percentages; Kappa is unitless.",
        "",
        "| Group | Metric | Status | Pairs | Pretrained mean | Random mean | Δ (95% bootstrap CI) | W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        for metric in METRICS:
            label = {
                "accuracy_pct": "Accuracy (%)",
                "balanced_accuracy_pct": "Balanced accuracy (%)",
                "kappa": "Kappa",
            }[metric]
            low = row[f"ci95_low_delta_{metric}"]
            high = row[f"ci95_high_delta_{metric}"]
            ci = f"[{_fmt(low)}, {_fmt(high)}]" if low is not None and high is not None else "—"
            wtl = f"{row[f'wins_{metric}']}/{row[f'ties_{metric}']}/{row[f'losses_{metric}']}"
            lines.append(
                f"| {row['group']} | {label} | {row['status']} | "
                f"{row['matched_pairs']}/{row['expected_pairs']} | "
                f"{_fmt(row[f'pretrained_mean_{metric}'])} | "
                f"{_fmt(row[f'random_mean_{metric}'])} | "
                f"{_fmt(row[f'mean_delta_{metric}'])} ({ci}) | {wtl} |"
            )
    lines.extend([
        "",
        "W/T/L counts paired subjects where pretrained minus random is positive, within 1e-12 of zero, or negative.",
        "Any group with missing, invalid, or misaligned pairs is marked `incomplete`; its reported means and intervals use only the listed matched pairs.",
        "Run-level prediction counts and single-class collapse flags are in `runs.csv`; alignment details are in `pairs.csv`.",
        f"",
        f"Output directory: `{output_dir}`",
        "",
    ])
    return "\n".join(lines)


def build_report(artifact_root: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    artifact_root = external_path(artifact_root).resolve()
    output_dir = require_external_output(output_dir)
    artifacts: list[Artifact] = []
    pairs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pair_by_dataset: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_SUBJECTS}

    for dataset, n_subjects in DATASET_SUBJECTS.items():
        for subject0 in range(n_subjects):
            p_path = artifact_root / dataset / PRETRAINED / f"{subject0}_{seed}_test.npz"
            r_path = artifact_root / dataset / RANDOM / f"{subject0}_{seed}_test.npz"
            p_artifact = _load_artifact(dataset, subject0, seed, PRETRAINED, p_path)
            r_artifact = _load_artifact(dataset, subject0, seed, RANDOM, r_path)
            artifacts.extend((p_artifact, r_artifact))
            pair = _pair_row(p_artifact, r_artifact)
            pairs.append(pair)
            pair_by_dataset[dataset].append(pair)

        group_run_rows = [
            row for row in _run_rows(artifacts)
            if row["dataset"] == dataset
        ]
        summary = _summary_row(dataset, pair_by_dataset[dataset], n_subjects)
        summary["pretrained_collapse_runs"] = sum(
            bool(row["single_class_prediction_collapse"])
            for row in group_run_rows if row["model"] == PRETRAINED
        )
        summary["random_collapse_runs"] = sum(
            bool(row["single_class_prediction_collapse"])
            for row in group_run_rows if row["model"] == RANDOM
        )
        summaries.append(summary)

    run_rows = _run_rows(artifacts)
    overall = _summary_row("OVERALL", pairs, sum(DATASET_SUBJECTS.values()))
    overall["pretrained_collapse_runs"] = sum(
        bool(row["single_class_prediction_collapse"])
        for row in run_rows if row["model"] == PRETRAINED
    )
    overall["random_collapse_runs"] = sum(
        bool(row["single_class_prediction_collapse"])
        for row in run_rows if row["model"] == RANDOM
    )
    summaries.append(overall)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "pairs.csv", pairs)
    _write_csv(output_dir / "summary.csv", summaries)
    markdown = _markdown(summaries, seed, output_dir)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    payload = {
        "seed": seed,
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "datasets": DATASET_SUBJECTS,
        "expected_subject_pairs": sum(DATASET_SUBJECTS.values()),
        "bootstrap": {
            "unit": "subject",
            "replicates": BOOTSTRAP_REPS,
            "base_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "summaries": summaries,
        "pairs": pairs,
        "runs": run_rows,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=666, help="artifact seed (default: 666)")
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT,
        help=f"artifact root (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    payload = build_report(args.artifact_root, args.output_dir, args.seed)
    overall = payload["summaries"][-1]
    print(
        f"Wrote ablation report to {args.output_dir} "
        f"({overall['matched_pairs']}/{overall['expected_pairs']} matched pairs; "
        f"{overall['status']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
