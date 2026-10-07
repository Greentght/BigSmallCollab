#!/usr/bin/env python3
"""Strict, sample-aligned comparison of CodeBrain and EEG baselines.

Artifacts are read from ``/data1/llx/BigSmallcollab/results/artifacts/<dataset>/<model>/`` by default.
Each expected file is ``<subject0>_<seed>_test.npz`` with ``logits``, ``y``,
``sample_uid`` and ``split_policy`` fields. Results are written to
``/data1/llx/BigSmallcollab/results/codebrain/comparison``. Model-specific roots can be supplied with
``--model-root MODEL=PATH``; each root must contain the usual
``<dataset>/<model>/<subject0>_<seed>_test.npz`` layout.

The subject is the statistical unit: seeds are averaged within each subject
before model means, paired differences, or confidence intervals are computed.
Incomplete groups are marked and have no aggregate estimate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
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
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "codebrain" / "comparison"
DATASET_SUBJECTS = {
    "BNCI2014001": 9,
    "BNCI2014004": 9,
    "BNCI2015001": 12,
    "AlexMI": 8,
}
DEFAULT_SEEDS = (666, 667, 668)
MODELS = ("mirepnet", "cbramod", "codebrain")
COMPARISONS = (("codebrain", "mirepnet"), ("codebrain", "cbramod"))
METRICS = ("accuracy_pct", "balanced_accuracy_pct", "kappa")
ARTIFACT_RE = re.compile(r"^(\d+)_(\d+)_test\.npz$")


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
    metric_row: dict[str, Any] | None = None

    @property
    def ready_for_alignment(self) -> bool:
        return (
            self.exists
            and not self.errors
            and self.y is not None
            and self.sample_uid is not None
            and self.split_policy is not None
        )


def _plain(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to JSON/CSV-safe values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
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


def _metrics_from_logits(logits: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], dict[str, int]]:
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
        # A test split may omit a class; for conventional zero-based labels the
        # logit column still names that class.
        class_labels = np.arange(n_classes, dtype=y.dtype)
    else:
        raise ValueError(
            f"cannot map {n_classes} logit columns to observed labels "
            f"{observed_classes.tolist()}"
        )

    predicted_columns = logits.argmax(axis=1)
    y_pred = class_labels[predicted_columns]
    accuracy = float(np.mean(y_pred == y) * 100.0)
    recalls = [float(np.mean(y_pred[y == label] == label)) for label in observed_classes]
    balanced_accuracy = float(np.mean(recalls) * 100.0)

    # Cohen's kappa from the confusion matrix, with the same undefined case as
    # common metric libraries: all mass in one row and one column.
    label_to_index = {label.item() if isinstance(label, np.generic) else label: i
                      for i, label in enumerate(class_labels)}
    y_idx = np.fromiter(
        (label_to_index[v.item() if isinstance(v, np.generic) else v] for v in y),
        dtype=np.int64,
        count=len(y),
    )
    pred_idx = predicted_columns.astype(np.int64, copy=False)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_idx, pred_idx), 1)
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

    predicted_values, predicted_counts = np.unique(y_pred, return_counts=True)
    counts = {
        str(label.item() if isinstance(label, np.generic) else label): int(count)
        for label, count in zip(predicted_values, predicted_counts)
    }
    collapsed_class = None
    if len(predicted_values) == 1:
        only = predicted_values[0]
        collapsed_class = only.item() if isinstance(only, np.generic) else only
    metrics = {
        "n_samples": int(len(y)),
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "kappa": kappa,
        "predicted_class_count": int(len(predicted_values)),
        "single_class_prediction_collapse": len(predicted_values) == 1,
        "collapsed_to_class": collapsed_class,
    }
    return metrics, counts


def _load_artifact(dataset: str, subject0: int, seed: int, model: str,
                   path: Path) -> Artifact:
    path = external_path(path)
    artifact = Artifact(dataset, subject0, seed, model, path)
    if not path.is_file():
        return artifact
    artifact.exists = True
    try:
        with np.load(resolve_local_file(path), allow_pickle=False) as data:
            fields = set(data.files)
            required = {"logits", "y", "sample_uid", "split_policy"}
            missing = sorted(required - fields)
            if missing:
                artifact.errors.append("missing required fields: " + ", ".join(missing))

            logits = np.asarray(data["logits"]) if "logits" in fields else None
            y = np.asarray(data["y"]) if "y" in fields else None
            uid = np.asarray(data["sample_uid"]) if "sample_uid" in fields else None

            if y is not None:
                if y.ndim != 1:
                    artifact.errors.append(f"y must be one-dimensional, got {y.shape}")
                elif y.size == 0:
                    artifact.errors.append("y has no test samples")
                else:
                    artifact.y = y

            if uid is not None:
                if uid.ndim != 2 or uid.shape[1] != 2:
                    artifact.errors.append(
                        f"sample_uid must have shape (N, 2), got {uid.shape}"
                    )
                elif y is not None and y.ndim == 1 and uid.shape[0] != y.shape[0]:
                    artifact.errors.append(
                        f"sample_uid/y length mismatch: {uid.shape[0]} != {y.shape[0]}"
                    )
                else:
                    artifact.sample_uid = uid

            if "split_policy" in fields:
                try:
                    artifact.split_policy = _policy_string(np.asarray(data["split_policy"]))
                except (TypeError, ValueError, UnicodeDecodeError) as exc:
                    artifact.errors.append(f"invalid split_policy: {exc}")

            if logits is not None and y is not None and y.ndim == 1 and y.size:
                try:
                    metrics, counts = _metrics_from_logits(logits, y)
                    artifact.metric_row = {
                        "dataset": dataset,
                        "subject0": subject0,
                        "seed": seed,
                        "model": model,
                        **metrics,
                        "_pred_counts": counts,
                    }
                except (ValueError, TypeError, IndexError) as exc:
                    artifact.errors.append(f"invalid prediction arrays: {exc}")
    except Exception as exc:  # corrupt or unreadable NPZ: surface it in the report
        artifact.errors.append(f"cannot read NPZ: {type(exc).__name__}: {exc}")
    return artifact


def _pair_alignment(a: Artifact, b: Artifact) -> dict[str, Any]:
    row = {
        "dataset": a.dataset,
        "subject0": a.subject0,
        "seed": a.seed,
        "model_a": a.model,
        "model_b": b.model,
        "status": "matched",
        "y_equal": None,
        "sample_uid_equal": None,
        "split_policy_equal": None,
        "reason": "",
    }
    if not a.exists or not b.exists:
        absent = [x.model for x in (a, b) if not x.exists]
        row["status"] = "missing_artifact"
        row["reason"] = "missing: " + ", ".join(absent)
        return row
    if not a.ready_for_alignment or not b.ready_for_alignment:
        row["status"] = "invalid_artifact"
        invalid = []
        if not a.ready_for_alignment:
            invalid.append(f"{a.model}: {'; '.join(a.errors) or 'metadata unavailable'}")
        if not b.ready_for_alignment:
            invalid.append(f"{b.model}: {'; '.join(b.errors) or 'metadata unavailable'}")
        row["reason"] = " | ".join(invalid)
        return row

    y_equal = np.array_equal(a.y, b.y)
    uid_equal = np.array_equal(a.sample_uid, b.sample_uid)
    policy_equal = a.split_policy == b.split_policy
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
    return row


def _bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float,
                  seed: int) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def _stable_ci_seed(base_seed: int, label: str) -> int:
    return (int(base_seed) + zlib.crc32(label.encode("utf-8"))) % (2**32)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _plain(value) for key, value in row.items()})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_plain,
                  allow_nan=False)
        handle.write("\n")


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def esc(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(esc(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _parse_model_roots(items: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            parser.error(f"--model-root must be MODEL=PATH, got {item!r}")
        model, raw_path = item.split("=", 1)
        if model not in MODELS:
            parser.error(f"unknown model in --model-root: {model!r}; choose from {MODELS}")
        if not raw_path:
            parser.error(f"empty path in --model-root {item!r}")
        roots[model] = external_path(raw_path).resolve()
    return roots


def _expected_keys(datasets: list[str], seeds: list[int]) -> list[tuple[str, int, int]]:
    return [
        (dataset, subject0, seed)
        for dataset in datasets
        for subject0 in range(DATASET_SUBJECTS[dataset])
        for seed in seeds
    ]


def _main_summary_rows(
    datasets: list[str], seeds: list[int], artifacts: dict[tuple[str, int, int, str], Artifact]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        expected_subjects = DATASET_SUBJECTS[dataset]
        for model in MODELS:
            subject_metrics: list[dict[str, float]] = []
            available_runs = 0
            for subject0 in range(expected_subjects):
                seed_metrics = []
                for seed in seeds:
                    artifact = artifacts[(dataset, subject0, seed, model)]
                    if artifact.metric_row is not None and not artifact.errors:
                        available_runs += 1
                        seed_metrics.append(artifact.metric_row)
                if len(seed_metrics) == len(seeds):
                    subject_metrics.append({
                        metric: float(np.mean([float(row[metric]) for row in seed_metrics]))
                        for metric in METRICS
                    })
            complete = len(subject_metrics) == expected_subjects
            row: dict[str, Any] = {
                "dataset": dataset,
                "model": model,
                "status": "complete" if complete else "incomplete",
                "n_subjects": len(subject_metrics),
                "expected_subjects": expected_subjects,
                "n_valid_runs": available_runs,
                "expected_runs": expected_subjects * len(seeds),
            }
            for metric in METRICS:
                row[f"mean_{metric}"] = (
                    _mean([x[metric] for x in subject_metrics]) if complete else None
                )
            dataset_rows.append(row)

    total_subjects = sum(DATASET_SUBJECTS[d] for d in datasets)
    for model in MODELS:
        subject_metrics = []
        available_runs = 0
        for dataset in datasets:
            for subject0 in range(DATASET_SUBJECTS[dataset]):
                seed_metrics = []
                for seed in seeds:
                    artifact = artifacts[(dataset, subject0, seed, model)]
                    if artifact.metric_row is not None and not artifact.errors:
                        available_runs += 1
                        seed_metrics.append(artifact.metric_row)
                if len(seed_metrics) == len(seeds):
                    subject_metrics.append({
                        metric: float(np.mean([float(row[metric]) for row in seed_metrics]))
                        for metric in METRICS
                    })
        complete = len(subject_metrics) == total_subjects
        row = {
            "model": model,
            "status": "complete" if complete else "incomplete",
            "n_subjects": len(subject_metrics),
            "expected_subjects": total_subjects,
            "n_valid_runs": available_runs,
            "expected_runs": total_subjects * len(seeds),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = (
                _mean([x[metric] for x in subject_metrics]) if complete else None
            )
        overall_rows.append(row)
    return dataset_rows, overall_rows


def _seed_summary_rows(
    datasets: list[str], seeds: list[int], artifacts: dict[tuple[str, int, int, str], Artifact]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        expected = DATASET_SUBJECTS[dataset]
        for seed in seeds:
            for model in MODELS:
                runs = []
                for subject0 in range(expected):
                    artifact = artifacts[(dataset, subject0, seed, model)]
                    runs.append(
                        artifact.metric_row if not artifact.errors else None
                    )
                available = [r for r in runs if r is not None]
                complete = len(available) == expected
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "status": "complete" if complete else "incomplete",
                    "n_subjects": len(available),
                    "expected_subjects": expected,
                }
                for metric in METRICS:
                    row[f"mean_{metric}"] = (
                        _mean([float(r[metric]) for r in available]) if complete else None
                    )
                rows.append(row)
    return rows


def _comparison_rows(
    datasets: list[str], seeds: list[int], artifacts: dict[tuple[str, int, int, str], Artifact],
    alignments: dict[tuple[str, int, int, str, str], dict[str, Any]],
    n_boot: int, ci_level: float, ci_seed: int,
) -> list[dict[str, Any]]:
    # Store only subjects whose every requested seed is present, valid, and aligned.
    subject_values: dict[tuple[str, str, str, int], dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        for model, baseline in COMPARISONS:
            for subject0 in range(DATASET_SUBJECTS[dataset]):
                rows_by_model: dict[str, list[dict[str, Any]]] = {model: [], baseline: []}
                subject_complete = True
                for seed in seeds:
                    a = artifacts[(dataset, subject0, seed, model)]
                    b = artifacts[(dataset, subject0, seed, baseline)]
                    alignment = alignments[(dataset, subject0, seed, model, baseline)]
                    if (
                        alignment["status"] != "matched"
                        or a.metric_row is None
                        or b.metric_row is None
                        or a.errors
                        or b.errors
                    ):
                        subject_complete = False
                        break
                    rows_by_model[model].append(a.metric_row)
                    rows_by_model[baseline].append(b.metric_row)
                if subject_complete and all(len(v) == len(seeds) for v in rows_by_model.values()):
                    subject_values[(dataset, model, baseline, subject0)] = {
                        current_model: {
                            metric: float(np.mean([float(row[metric]) for row in run_rows]))
                            for metric in METRICS
                        }
                        for current_model, run_rows in rows_by_model.items()
                    }

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[str]]] = [(dataset, [dataset]) for dataset in datasets]
    scopes.append(("OVERALL", datasets))
    for scope, scope_datasets in scopes:
        expected_subjects = sum(DATASET_SUBJECTS[d] for d in scope_datasets)
        for model, baseline in COMPARISONS:
            selected_subjects = [
                (dataset, subject0)
                for dataset in scope_datasets
                for subject0 in range(DATASET_SUBJECTS[dataset])
                if (dataset, model, baseline, subject0) in subject_values
            ]
            n_paired = len(selected_subjects)
            complete = n_paired == expected_subjects
            for metric in METRICS:
                deltas = np.asarray([
                    subject_values[(dataset, model, baseline, subject0)][model][metric]
                    - subject_values[(dataset, model, baseline, subject0)][baseline][metric]
                    for dataset, subject0 in selected_subjects
                ], dtype=float)
                row: dict[str, Any] = {
                    "scope": scope,
                    "model": model,
                    "baseline": baseline,
                    "metric": metric,
                    "status": "complete" if complete else "incomplete",
                    "n_paired_subjects": n_paired,
                    "expected_subjects": expected_subjects,
                    "mean_model": None,
                    "mean_baseline": None,
                    "mean_delta": None,
                    "ci_level_pct": ci_level * 100.0,
                    "ci_low": None,
                    "ci_high": None,
                    "wins": None,
                    "ties": None,
                    "losses": None,
                }
                if complete:
                    model_values = [
                        subject_values[(dataset, model, baseline, subject0)][model][metric]
                        for dataset, subject0 in selected_subjects
                    ]
                    baseline_values = [
                        subject_values[(dataset, model, baseline, subject0)][baseline][metric]
                        for dataset, subject0 in selected_subjects
                    ]
                    ci_low, ci_high = _bootstrap_ci(
                        deltas, n_boot, 1.0 - ci_level,
                        _stable_ci_seed(ci_seed, f"{scope}|{model}|{baseline}|{metric}"),
                    )
                    row.update({
                        "mean_model": _mean(model_values),
                        "mean_baseline": _mean(baseline_values),
                        "mean_delta": _mean(deltas.tolist()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "wins": int(np.sum(deltas > 1e-12)),
                        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                        "losses": int(np.sum(deltas < -1e-12)),
                    })
                rows.append(row)
    return rows


def _markdown_report(
    *, datasets: list[str], seeds: list[int], artifacts: dict[tuple[str, int, int, str], Artifact],
    missing_rows: list[dict[str, Any]], issue_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]], dataset_summary: list[dict[str, Any]],
    overall_summary: list[dict[str, Any]], seed_summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]], output_dir: Path,
) -> str:
    expected_runs = sum(DATASET_SUBJECTS[d] for d in datasets) * len(seeds) * len(MODELS)
    found_runs = sum(a.exists for a in artifacts.values())
    mismatch_count = sum(r["status"] == "mismatch" for r in alignment_rows)
    complete_comp = [r for r in comparisons if r["status"] == "complete"]
    lines = [
        "# CodeBrain baseline comparison",
        "",
        f"Datasets: {', '.join(datasets)}  ",
        f"Seeds: {', '.join(str(s) for s in seeds)}  ",
        f"Expected files: {expected_runs}; present files: {found_runs}; missing: {len(missing_rows)}; invalid: {len(issue_rows)}; alignment mismatches: {mismatch_count}.",
        "",
        "Accuracy and balanced accuracy are percentages; kappa is a coefficient. "
        "Each subject is one unit: its requested seeds are averaged before means and paired bootstrap confidence intervals.",
        "",
        "## Dataset means",
        "",
        _md_table(
            ["Dataset", "Model", "Status", "Subjects", "Accuracy %", "Balanced accuracy %", "Kappa"],
            [[r["dataset"], r["model"], r["status"],
              f"{r['n_subjects']}/{r['expected_subjects']}",
              _format(r["mean_accuracy_pct"]),
              _format(r["mean_balanced_accuracy_pct"]), _format(r["mean_kappa"])]
             for r in dataset_summary],
        ),
        "",
        "## Overall means",
        "",
        _md_table(
            ["Model", "Status", "Subjects", "Accuracy %", "Balanced accuracy %", "Kappa"],
            [[r["model"], r["status"], f"{r['n_subjects']}/{r['expected_subjects']}",
              _format(r["mean_accuracy_pct"]),
              _format(r["mean_balanced_accuracy_pct"]), _format(r["mean_kappa"])]
             for r in overall_summary],
        ),
        "",
        "## Paired comparisons",
        "",
    ]
    if complete_comp:
        lines.append(_md_table(
            ["Scope", "Model", "Baseline", "Metric", "Delta", "CI", "Wins/Ties/Losses", "N"],
            [[r["scope"], r["model"], r["baseline"], r["metric"],
              _format(r["mean_delta"]),
              f"[{_format(r['ci_low'])}, {_format(r['ci_high'])}]",
              f"{r['wins']}/{r['ties']}/{r['losses']}",
              f"{r['n_paired_subjects']}/{r['expected_subjects']}"]
             for r in complete_comp],
        ))
    else:
        lines.append("No complete paired comparison is available; incomplete comparisons have blank estimates in the CSV.")
    lines.extend([
        "",
        "## Per-seed means",
        "",
        _md_table(
            ["Dataset", "Seed", "Model", "Status", "Subjects", "Accuracy %", "Balanced accuracy %", "Kappa"],
            [[r["dataset"], r["seed"], r["model"], r["status"],
              f"{r['n_subjects']}/{r['expected_subjects']}",
              _format(r["mean_accuracy_pct"]), _format(r["mean_balanced_accuracy_pct"]),
              _format(r["mean_kappa"])]
             for r in seed_summary],
        ),
        "",
        "## Missing artifacts",
        "",
    ])
    if missing_rows:
        counts: dict[tuple[str, str], int] = {}
        for row in missing_rows:
            key = (row["dataset"], row["model"])
            counts[key] = counts.get(key, 0) + 1
        lines.append(_md_table(
            ["Dataset", "Model", "Missing files"],
            [[dataset, model, count] for (dataset, model), count in sorted(counts.items())],
        ))
        lines.append("")
        lines.append("The full expected key list and paths are in `missing_runs.csv`.")
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Files",
        "",
        f"Reports are in `{output_dir}`. `run_metrics.csv` includes per-run class prediction counts and single-class collapse flags; `alignment_checks.csv` records y, sample UID, and split policy checks.",
        "The permanent seed-666 run table and its dataset/overall summaries and paired comparisons are written to `seed666_first_round.csv`, `seed666_summary_by_dataset.csv`, `seed666_summary_overall.csv`, and `seed666_paired_comparisons.csv`.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT,
        help="root containing <dataset>/<model>/<subject0>_<seed>_test.npz",
    )
    parser.add_argument(
        "--model-root", action="append", default=[], metavar="MODEL=PATH",
        help="override artifact root for a model (repeatable; same dataset/model layout)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SUBJECTS),
        default=list(DATASET_SUBJECTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--ci-seed", type=int, default=0,
                        help="base seed for deterministic subject-level bootstrap intervals")
    args = parser.parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique seed values")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if not 0.0 < args.ci_level < 1.0:
        parser.error("--ci-level must be strictly between 0 and 1")

    datasets = list(args.datasets)
    seeds = list(args.seeds)
    model_roots = _parse_model_roots(args.model_root, parser)
    artifact_root = external_path(args.artifact_root).resolve()
    output_dir = require_external_output(args.output_dir)

    # Seed 666 remains available as a dedicated first-round table when a caller
    # selects another seed subset for the aggregate comparison.
    load_seeds = sorted(set(seeds) | {666})
    artifacts: dict[tuple[str, int, int, str], Artifact] = {}
    for dataset, subject0, seed in _expected_keys(datasets, load_seeds):
        for model in MODELS:
            root = model_roots.get(model, artifact_root)
            path = root / dataset / model / f"{subject0}_{seed}_test.npz"
            artifacts[(dataset, subject0, seed, model)] = _load_artifact(
                dataset, subject0, seed, model, path
            )

    missing_rows = []
    issue_rows = []
    run_rows = []
    for key, artifact in artifacts.items():
        if not artifact.exists:
            if key[2] in seeds:
                missing_rows.append({
                    "dataset": artifact.dataset,
                    "subject0": artifact.subject0,
                    "seed": artifact.seed,
                    "model": artifact.model,
                    "expected_path": str(artifact.path),
                })
            continue
        if artifact.errors:
            issue_rows.append({
                "dataset": artifact.dataset,
                "subject0": artifact.subject0,
                "seed": artifact.seed,
                "model": artifact.model,
                "path": str(artifact.path),
                "errors": " | ".join(artifact.errors),
            })
        if artifact.metric_row is not None:
            run_rows.append(artifact.metric_row)

    alignment_rows = []
    alignment_lookup: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for dataset, subject0, seed in _expected_keys(datasets, seeds):
        for model_a, model_b in itertools.combinations(MODELS, 2):
            a = artifacts[(dataset, subject0, seed, model_a)]
            b = artifacts[(dataset, subject0, seed, model_b)]
            check = _pair_alignment(a, b)
            alignment_rows.append(check)
            alignment_lookup[(dataset, subject0, seed, model_a, model_b)] = check
            alignment_lookup[(dataset, subject0, seed, model_b, model_a)] = check

    seed666_alignment_lookup: dict[
        tuple[str, int, int, str, str], dict[str, Any]
    ] = {}
    for dataset, subject0, seed in _expected_keys(datasets, [666]):
        for model_a, model_b in itertools.combinations(MODELS, 2):
            a = artifacts[(dataset, subject0, seed, model_a)]
            b = artifacts[(dataset, subject0, seed, model_b)]
            check = _pair_alignment(a, b)
            seed666_alignment_lookup[(dataset, subject0, seed, model_a, model_b)] = check
            seed666_alignment_lookup[(dataset, subject0, seed, model_b, model_a)] = check

    # Canonicalize class-count columns across all files, retaining zero counts.
    class_labels = sorted({
        label for row in run_rows for label in row["_pred_counts"]
    }, key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
    count_columns = [f"pred_count_class_{label}" for label in class_labels]
    clean_run_rows = []
    for raw in run_rows:
        row = {key: value for key, value in raw.items() if key != "_pred_counts"}
        row["artifact_status"] = "valid" if not artifacts[(
            raw["dataset"], raw["subject0"], raw["seed"], raw["model"]
        )].errors else "invalid_metadata_or_arrays"
        for label, column in zip(class_labels, count_columns):
            row[column] = raw["_pred_counts"].get(label, 0)
        clean_run_rows.append(row)

    dataset_summary, overall_summary = _main_summary_rows(datasets, seeds, artifacts)
    seed_summary = _seed_summary_rows(datasets, seeds, artifacts)
    comparisons = _comparison_rows(
        datasets, seeds, artifacts, alignment_lookup,
        args.bootstrap_samples, args.ci_level, args.ci_seed,
    )
    seed666_dataset_summary, seed666_overall_summary = _main_summary_rows(
        datasets, [666], artifacts
    )
    seed666_comparisons = _comparison_rows(
        datasets, [666], artifacts, seed666_alignment_lookup,
        args.bootstrap_samples, args.ci_level, args.ci_seed,
    )

    seed666_rows = []
    for dataset in datasets:
        for subject0 in range(DATASET_SUBJECTS[dataset]):
            for model in MODELS:
                artifact = artifacts[(dataset, subject0, 666, model)]
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "subject0": subject0,
                    "seed": 666,
                    "model": model,
                    "status": (
                        "missing" if not artifact.exists else
                        "valid" if artifact.metric_row is not None and not artifact.errors else "invalid"
                    ),
                    "path": str(artifact.path),
                    "errors": " | ".join(artifact.errors),
                }
                if artifact.metric_row is not None:
                    row.update({k: v for k, v in artifact.metric_row.items()
                                if k not in ("dataset", "subject0", "seed", "model", "_pred_counts")})
                    counts = artifact.metric_row["_pred_counts"]
                    for label, column in zip(class_labels, count_columns):
                        row[column] = counts.get(label, 0)
                seed666_rows.append(row)

    # Output CSV headers are explicit even if CodeBrain artifacts have not been
    # exported yet, so empty result files remain machine-readable.
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "run_metrics.csv", [
        "dataset", "subject0", "seed", "model", "artifact_status", "n_samples",
        "accuracy_pct", "balanced_accuracy_pct", "kappa", "predicted_class_count",
        "single_class_prediction_collapse", "collapsed_to_class", *count_columns,
    ], clean_run_rows)
    _write_csv(output_dir / "missing_runs.csv", [
        "dataset", "subject0", "seed", "model", "expected_path",
    ], missing_rows)
    _write_csv(output_dir / "artifact_issues.csv", [
        "dataset", "subject0", "seed", "model", "path", "errors",
    ], issue_rows)
    _write_csv(output_dir / "alignment_checks.csv", [
        "dataset", "subject0", "seed", "model_a", "model_b", "status",
        "y_equal", "sample_uid_equal", "split_policy_equal", "reason",
    ], alignment_rows)
    summary_fields = ["dataset", "model", "status", "n_subjects", "expected_subjects",
                      "n_valid_runs", "expected_runs", *[f"mean_{m}" for m in METRICS]]
    _write_csv(output_dir / "summary_by_dataset.csv", summary_fields, dataset_summary)
    overall_fields = ["model", "status", "n_subjects", "expected_subjects",
                      "n_valid_runs", "expected_runs", *[f"mean_{m}" for m in METRICS]]
    _write_csv(output_dir / "summary_overall.csv", overall_fields, overall_summary)
    _write_csv(output_dir / "per_seed_summary.csv", [
        "dataset", "seed", "model", "status", "n_subjects", "expected_subjects",
        *[f"mean_{m}" for m in METRICS],
    ], seed_summary)
    _write_csv(output_dir / "paired_comparisons.csv", [
        "scope", "model", "baseline", "metric", "status", "n_paired_subjects",
        "expected_subjects", "mean_model", "mean_baseline", "mean_delta",
        "ci_level_pct", "ci_low", "ci_high", "wins", "ties", "losses",
    ], comparisons)
    _write_csv(output_dir / "seed666_first_round.csv", [
        "dataset", "subject0", "seed", "model", "status", "path", "errors",
        "n_samples", "accuracy_pct", "balanced_accuracy_pct", "kappa",
        "predicted_class_count", "single_class_prediction_collapse", "collapsed_to_class",
        *count_columns,
    ], seed666_rows)
    _write_csv(output_dir / "seed666_summary_by_dataset.csv", summary_fields,
               seed666_dataset_summary)
    _write_csv(output_dir / "seed666_summary_overall.csv", overall_fields,
               seed666_overall_summary)
    _write_csv(output_dir / "seed666_paired_comparisons.csv", [
        "scope", "model", "baseline", "metric", "status", "n_paired_subjects",
        "expected_subjects", "mean_model", "mean_baseline", "mean_delta",
        "ci_level_pct", "ci_low", "ci_high", "wins", "ties", "losses",
    ], seed666_comparisons)

    report = {
        "config": {
            "datasets": datasets,
            "subject_counts": {d: DATASET_SUBJECTS[d] for d in datasets},
            "seeds": seeds,
            "first_round_seed": 666,
            "models": list(MODELS),
            "artifact_root": str(artifact_root),
            "model_roots": {k: str(v) for k, v in model_roots.items()},
            "output_dir": str(output_dir),
            "bootstrap_samples": args.bootstrap_samples,
            "ci_level": args.ci_level,
            "ci_seed": args.ci_seed,
            "statistical_unit": "subject0; average seeds within subject before comparison",
            "accuracy_units": "percent",
        },
        "coverage": {
            "expected_model_run_files": sum(DATASET_SUBJECTS[d] for d in datasets) * len(seeds) * len(MODELS),
            "present_model_run_files": sum(
                artifacts[(d, s0, seed, model)].exists
                for d, s0, seed in _expected_keys(datasets, seeds)
                for model in MODELS
            ),
            "missing_files": len(missing_rows),
            "invalid_artifacts": len(issue_rows),
            "alignment_mismatches": sum(r["status"] == "mismatch" for r in alignment_rows),
        },
        "dataset_summary": dataset_summary,
        "overall_summary": overall_summary,
        "per_seed_summary": seed_summary,
        "paired_comparisons": comparisons,
        "seed666_summary_by_dataset": seed666_dataset_summary,
        "seed666_summary_overall": seed666_overall_summary,
        "seed666_paired_comparisons": seed666_comparisons,
        "missing_runs": missing_rows,
        "artifact_issues": issue_rows,
        "alignment_checks": alignment_rows,
    }
    _write_json(output_dir / "comparison_report.json", report)
    markdown = _markdown_report(
        datasets=datasets, seeds=seeds, artifacts=artifacts,
        missing_rows=missing_rows, issue_rows=issue_rows,
        alignment_rows=alignment_rows, dataset_summary=dataset_summary,
        overall_summary=overall_summary, seed_summary=seed_summary,
        comparisons=comparisons, output_dir=output_dir,
    )
    (output_dir / "comparison_report.md").write_text(markdown, encoding="utf-8")

    print(f"Wrote CodeBrain comparison report to {output_dir}")
    print(
        f"Artifacts present: {report['coverage']['present_model_run_files']}/"
        f"{report['coverage']['expected_model_run_files']} | "
        f"missing: {len(missing_rows)} | invalid: {len(issue_rows)} | "
        f"alignment mismatches: {report['coverage']['alignment_mismatches']}"
    )
    complete = sum(row["status"] == "complete" for row in comparisons)
    print(f"Complete paired comparison rows: {complete}/{len(comparisons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
