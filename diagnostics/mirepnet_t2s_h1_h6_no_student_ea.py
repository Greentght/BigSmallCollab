#!/usr/bin/env python3
"""CPU-only H1/H6 diagnostics from existing MIRepNet T2S prediction artifacts.

No training code or artifacts are modified. Seed means are computed within
LOSO subject first; bootstrap resampling then treats subjects as the units.
All source records come from the existing *no student EA* allsets experiment.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/distill/mirepnet_t2s_allsets_v1"
OUT = ROOT / "results/diagnostics/mirepnet_t2s_h1_h6_no_student_ea_20261102"
DATASETS = {
    "BNCI2014004": 9,
    "BNCI2014001": 9,
    "BNCI2014001-4": 9,
    "BNCI2015001": 12,
    "AlexMI": 8,
}
STUDENTS = ("ifnet", "eegnet", "adfcnn")
SEEDS = (666, 667, 668)
TRANSFER = {
    "CE_T2S_FEATURE": ("Feature", "T_CE"),
    "MCC_T2S_FEATURE": ("Feature", "T_MCC"),
    "CE_T2S_DIFFKD": ("DiffKD", "T_CE"),
    "MCC_T2S_DIFFKD": ("DiffKD", "T_MCC"),
}
BOOTSTRAPS = 10_000
BOOT_SEED = 20261102


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    labels = np.unique(y)
    return float(np.mean([np.mean(pred[y == label] == label) for label in labels]))


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    probs = np.exp(values)
    return probs / probs.sum(axis=1, keepdims=True)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(x))
    n = len(x)
    ids = rng.integers(0, n, size=(BOOTSTRAPS, n))
    draws = x[ids].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def regression_bootstrap(x: np.ndarray, y: np.ndarray,
                         rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    def stable_slope(xb: np.ndarray, yb: np.ndarray) -> float:
        xc, yc = xb - xb.mean(), yb - yb.mean()
        denominator = float(np.dot(xc, xc))
        if denominator <= 1e-12:
            return math.nan
        return float(np.dot(xc, yc) / denominator)

    slope = stable_slope(x, y) if len(x) > 1 else math.nan
    n = len(x)
    if n < 3 or not np.isfinite(slope):
        return slope, math.nan, math.nan
    ids = rng.integers(0, n, size=(BOOTSTRAPS, n))
    draws = []
    for row in ids:
        xb, yb = x[row], y[row]
        value = stable_slope(xb, yb)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return slope, math.nan, math.nan
    low, high = np.quantile(draws, [0.025, 0.975])
    return slope, float(low), float(high)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except Exception:
        xr = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        yr = np.argsort(np.argsort(y, kind="stable"), kind="stable")
        return float(np.corrcoef(xr, yr)[0, 1])


def load_predictions() -> dict[tuple, dict]:
    predictions: dict[tuple, dict] = {}
    for dataset, n_subjects in DATASETS.items():
        path = RESULTS / f"{dataset}_records.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        for row in records:
            if row["dataset"] != dataset:
                raise ValueError(f"dataset identity mismatch in {path}")
            key = (dataset, row["student"].lower(), row["method"],
                   int(row["seed"]), int(row["subject"]))
            if key in predictions:
                raise ValueError(f"duplicate prediction record: {key}")
            pred_path = Path(row["prediction_path"])
            with np.load(pred_path, allow_pickle=False) as saved:
                item = {name: np.asarray(saved[name]) for name in saved.files}
            if set(("trial_uid", "y", "logits")) - item.keys():
                raise ValueError(f"prediction fields missing in {pred_path}")
            if len(item["y"]) != int(row["n_trials"]):
                raise ValueError(f"trial count mismatch in {pred_path}")
            if not np.isclose(balanced_accuracy(item["y"], item["logits"].argmax(1)),
                              float(row["balanced_accuracy"]), atol=1e-10):
                raise ValueError(f"BA mismatch in {pred_path}")
            predictions[key] = item
        expected = n_subjects * len(SEEDS) * (3 + len(STUDENTS) * 5)
        observed = sum(1 for k in predictions if k[0] == dataset)
        if observed != expected:
            raise ValueError(f"{dataset}: expected {expected} records, found {observed}")
    return predictions


def get(predictions: dict, dataset: str, model: str, method: str,
        seed: int, subject: int) -> dict:
    return predictions[(dataset, model.lower(), method, seed, subject)]


def validate_alignment(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ref = items[0]
    for other in items[1:]:
        if not np.array_equal(ref["trial_uid"], other["trial_uid"]):
            raise ValueError("target trial UID order differs across matched models")
        if not np.array_equal(ref["y"], other["y"]):
            raise ValueError("target labels differ across matched models")
        if ref["logits"].shape != other["logits"].shape:
            raise ValueError("matched logits have different shapes")
    return ref["y"], ref["trial_uid"]


def run_h1(predictions: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict]]:
    fold_rows: list[dict] = []
    gain_rows: list[dict] = []
    regression_rows: list[dict] = []
    for dataset, n_subjects in DATASETS.items():
        for student in STUDENTS:
            for method, (family, teacher_method) in TRANSFER.items():
                paired = []
                for subject in range(1, n_subjects + 1):
                    per_seed = []
                    for seed in SEEDS:
                        teacher = get(predictions, dataset, "mirepnet", teacher_method,
                                      seed, subject)
                        ce = get(predictions, dataset, student, "none", seed, subject)
                        kd = get(predictions, dataset, student, method, seed, subject)
                        validate_alignment([teacher, ce, kd])
                        teacher_ba = balanced_accuracy(teacher["y"], teacher["logits"].argmax(1))
                        ce_ba = balanced_accuracy(ce["y"], ce["logits"].argmax(1))
                        kd_ba = balanced_accuracy(kd["y"], kd["logits"].argmax(1))
                        per_seed.append((teacher_ba, ce_ba, kd_ba))
                    values = np.asarray(per_seed, dtype=np.float64).mean(axis=0)
                    row = {
                        "dataset": dataset, "student": student, "transfer": method,
                        "family": family, "teacher": teacher_method,
                        "subject": subject, "n_seed": len(SEEDS),
                        "teacher_BA_pct": values[0] * 100,
                        "CE_student_BA_pct": values[1] * 100,
                        "KD_student_BA_pct": values[2] * 100,
                        "teacher_gap_g_pp": (values[0] - values[1]) * 100,
                        "KD_gain_pp": (values[2] - values[1]) * 100,
                        "EA_setting": "no student EA; transductive teacher EA",
                    }
                    fold_rows.append(row)
                    paired.append(row)
                gains = np.asarray([row["KD_gain_pp"] for row in paired])
                m, lo, hi = bootstrap_mean(gains, rng)
                gain_rows.append({
                    "dataset": dataset, "student": student, "transfer": method,
                    "family": family, "teacher": teacher_method,
                    "n_subject": len(paired), "n_seed": len(SEEDS),
                    "mean_KD_gain_pp": m, "bootstrap95_low_pp": lo,
                    "bootstrap95_high_pp": hi,
                    "CI_excludes_zero": bool(lo > 0 or hi < 0),
                    "EA_setting": "no student EA; transductive teacher EA",
                })
                g = np.asarray([row["teacher_gap_g_pp"] for row in paired])
                slope, slope_lo, slope_hi = regression_bootstrap(g, gains, rng)
                k = max(1, int(math.ceil(len(paired) * 0.25)))
                order = np.argsort(g, kind="stable")
                top_bottom = float(gains[order[-k:]].mean() - gains[order[:k]].mean())
                regression_rows.append({
                    "dataset": dataset, "student": student, "transfer": method,
                    "family": family, "teacher": teacher_method,
                    "n_subject": len(paired), "n_seed": len(SEEDS),
                    "slope_gain_pp_per_gap_pp": slope,
                    "slope_bootstrap95_low": slope_lo,
                    "slope_bootstrap95_high": slope_hi,
                    "slope_CI_excludes_zero": bool(slope_lo > 0 or slope_hi < 0),
                    "spearman_rho": spearman(g, gains),
                    "top25_minus_bottom25_mean_gain_pp": top_bottom,
                    "quartile_subject_count": k,
                    "EA_setting": "no student EA; transductive teacher EA",
                })
    return fold_rows, gain_rows, regression_rows


def h6_seed_metrics(teacher: dict, student: dict, ce_student: dict) -> dict:
    y, _ = validate_alignment([teacher, student, ce_student])
    tprob = softmax(teacher["logits"])
    sprob = softmax(student["logits"])
    cprob = softmax(ce_student["logits"])
    tpred, spred, cpred = tprob.argmax(1), sprob.argmax(1), cprob.argmax(1)
    tok, sok, cok = tpred == y, spred == y, cpred == y
    corrections = ~cok & tok
    harmful = cok & ~tok
    ensemble = (tprob + sprob) / 2.0
    ce_ensemble = (tprob + cprob) / 2.0
    return {
        "both_correct_rate": float(np.mean(tok & sok)),
        "teacher_only_rate": float(np.mean(tok & ~sok)),
        "student_only_rate": float(np.mean(~tok & sok)),
        "both_wrong_rate": float(np.mean(~tok & ~sok)),
        "teacher_student_agreement_rate": float(np.mean(tpred == spred)),
        "correction_rate": (float(np.mean(sok[corrections])) if corrections.any() else math.nan),
        "correction_n": int(corrections.sum()),
        "harm_rate": (float(np.mean(~sok[harmful])) if harmful.any() else math.nan),
        "harm_n": int(harmful.sum()),
        "oracle_accuracy": float(np.mean(tok | sok)),
        "ensemble_accuracy": accuracy(y, ensemble.argmax(1)),
        "student_accuracy": accuracy(y, spred),
        "ensemble_gain_vs_student": accuracy(y, ensemble.argmax(1)) - accuracy(y, spred),
        "ce_both_wrong_rate": float(np.mean(~tok & ~cok)),
        "ce_ensemble_accuracy": accuracy(y, ce_ensemble.argmax(1)),
        "ce_student_accuracy": accuracy(y, cpred),
        "ce_ensemble_gain_vs_student": accuracy(y, ce_ensemble.argmax(1)) - accuracy(y, cpred),
        "ce_oracle_accuracy": float(np.mean(tok | cok)),
        "ce_agreement_rate": float(np.mean(tpred == cpred)),
    }


def run_h6(predictions: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    metrics = (
        "both_correct_rate", "teacher_only_rate", "student_only_rate", "both_wrong_rate",
        "teacher_student_agreement_rate", "correction_rate", "harm_rate",
        "oracle_accuracy", "ensemble_accuracy", "student_accuracy",
        "ensemble_gain_vs_student", "ce_both_wrong_rate", "ce_ensemble_accuracy",
        "ce_student_accuracy", "ce_ensemble_gain_vs_student", "ce_oracle_accuracy",
        "ce_agreement_rate",
    )
    for dataset, n_subjects in DATASETS.items():
        for student in STUDENTS:
            for method, (family, teacher_method) in TRANSFER.items():
                paired = []
                for subject in range(1, n_subjects + 1):
                    seed_values = []
                    seed_raw = []
                    for seed in SEEDS:
                        teacher = get(predictions, dataset, "mirepnet", teacher_method,
                                      seed, subject)
                        kd = get(predictions, dataset, student, method, seed, subject)
                        ce = get(predictions, dataset, student, "none", seed, subject)
                        values = h6_seed_metrics(teacher, kd, ce)
                        seed_values.append(values)
                        seed_raw.append((values["both_correct_rate"], values["teacher_only_rate"],
                                         values["student_only_rate"], values["both_wrong_rate"]))
                    row = {"dataset": dataset, "student": student, "transfer": method,
                           "family": family, "teacher": teacher_method,
                           "subject": subject, "n_seed": len(SEEDS),
                           "EA_setting": "no student EA; transductive teacher EA"}
                    for metric in metrics:
                        vals = np.asarray([entry[metric] for entry in seed_values], dtype=float)
                        row[metric] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
                    for idx, name in enumerate(("both_correct", "teacher_only", "student_only", "both_wrong")):
                        row[f"{name}_pct"] = 100 * float(np.mean([v[idx] for v in seed_raw]))
                    row["mean_correction_trial_count"] = float(np.mean([x["correction_n"] for x in seed_values]))
                    row["mean_harm_trial_count"] = float(np.mean([x["harm_n"] for x in seed_values]))
                    row["both_wrong_rate_change_pp_vs_CE"] = 100 * (row["both_wrong_rate"] - row["ce_both_wrong_rate"])
                    row["ensemble_gain_change_pp_vs_CE"] = 100 * (row["ensemble_gain_vs_student"] - row["ce_ensemble_gain_vs_student"])
                    row["ensemble_accuracy_change_pp_vs_CE"] = 100 * (row["ensemble_accuracy"] - row["ce_ensemble_accuracy"])
                    row["agreement_change_pp_vs_CE"] = 100 * (row["teacher_student_agreement_rate"] - row["ce_agreement_rate"])
                    paired.append(row)
                    fold_rows.append(row)

                summary = {"dataset": dataset, "student": student, "transfer": method,
                           "family": family, "teacher": teacher_method,
                           "n_subject": len(paired), "n_seed": len(SEEDS),
                           "EA_setting": "no student EA; transductive teacher EA"}
                for metric in metrics:
                    vals = np.asarray([row[metric] for row in paired], dtype=float)
                    m, lo, hi = bootstrap_mean(vals, rng)
                    summary[f"{metric}_mean"] = m
                    summary[f"{metric}_95ci_low"] = lo
                    summary[f"{metric}_95ci_high"] = hi
                for metric in ("both_wrong_rate_change_pp_vs_CE", "ensemble_gain_change_pp_vs_CE",
                               "ensemble_accuracy_change_pp_vs_CE", "agreement_change_pp_vs_CE"):
                    vals = np.asarray([row[metric] for row in paired], dtype=float)
                    m, lo, hi = bootstrap_mean(vals, rng)
                    summary[f"{metric}_mean"] = m
                    summary[f"{metric}_95ci_low"] = lo
                    summary[f"{metric}_95ci_high"] = hi
                for name in ("both_correct", "teacher_only", "student_only", "both_wrong"):
                    vals = np.asarray([row[f"{name}_pct"] for row in paired], dtype=float)
                    m, lo, hi = bootstrap_mean(vals, rng)
                    summary[f"{name}_pct_mean"] = m
                    summary[f"{name}_pct_95ci_low"] = lo
                    summary[f"{name}_pct_95ci_high"] = hi
                summary_rows.append(summary)
    return fold_rows, summary_rows


def write_plots(regression_folds: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ifnet": "#1f77b4", "eegnet": "#d62728", "adfcnn": "#2ca02c"}
    markers = {"CE_T2S_FEATURE": "o", "MCC_T2S_FEATURE": "^",
               "CE_T2S_DIFFKD": "o", "MCC_T2S_DIFFKD": "^"}
    for family in ("Feature", "DiffKD"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, dataset in zip(axes.ravel(), DATASETS):
            rows = [r for r in regression_folds
                    if r["dataset"] == dataset and r["family"] == family]
            for student in STUDENTS:
                for teacher in ("T_CE", "T_MCC"):
                    method = f"{teacher.replace('T_', '')}_T2S_{'FEATURE' if family == 'Feature' else 'DIFFKD'}"
                    values = [r for r in rows if r["student"] == student and r["transfer"] == method]
                    if not values:
                        continue
                    ax.scatter([r["teacher_gap_g_pp"] for r in values],
                               [r["KD_gain_pp"] for r in values],
                               color=colors[student], marker=markers[method], alpha=.75,
                               label=f"{student}/{teacher}")
            ax.axhline(0, color="black", linewidth=.7)
            ax.axvline(0, color="gray", linewidth=.7)
            ax.set_title(dataset)
            ax.set_xlabel("Teacher BA − CE student BA (pp)")
            ax.set_ylabel("KD student BA − CE student BA (pp)")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=4)
        fig.suptitle(f"H1: KD gain vs teacher gap — {family}; no student EA")
        fig.savefig(OUT / f"H1_{family}_no_student_EA.png", dpi=160)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    rng = np.random.default_rng(BOOT_SEED)
    h1_fold, h1_gain, h1_reg = run_h1(predictions, rng)
    h6_fold, h6_summary = run_h6(predictions, rng)
    write_csv(OUT / "H1_fold_points.csv", h1_fold)
    write_csv(OUT / "H1_gain_summary.csv", h1_gain)
    write_csv(OUT / "H1_regression_summary.csv", h1_reg)
    write_csv(OUT / "H6_fold_metrics.csv", h6_fold)
    write_csv(OUT / "H6_summary.csv", h6_summary)
    write_plots(h1_fold)
    status = {
        "status": "complete", "experiment": "MIREPNET_T2S_ALLSETS_LOSO_V1",
        "student_EA": False, "label": "no student EA; transductive teacher EA",
        "datasets": list(DATASETS), "students": list(STUDENTS), "seeds": list(SEEDS),
        "expected_subject_folds": sum(DATASETS.values()),
        "seed_fold_units": sum(DATASETS.values()) * len(SEEDS),
        "H1_transfer_conditions": list(TRANSFER), "H6_transfer_conditions": list(TRANSFER),
        "bootstrap": {"unit": "LOSO subject after seed averaging", "replicates": BOOTSTRAPS,
                      "seed": BOOT_SEED},
        "inputs": "existing prediction artifacts only; no training or model fitting",
    }
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
