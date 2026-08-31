"""M2 fixed-evaluation multi-support K-shot curve.

Protocol:
  * Fixed source_seed trains one LOSO source checkpoint per model/fold.
  * Fixed Q_fixed is selected before support sampling and never used for target adaptation.
  * Supports are sampled only from the remaining candidate pool.
  * For each support_seed, supports are nested per class: S5 subset S10 subset S20 subset S30.
  * adapt_seed is fixed; this measures K/support behavior under a fixed source checkpoint.

This replaces the old "query is the complement of the sampled support" protocol for
K curves, but it does not reopen FullFT, LastBlock, Hybrid, or KD.
"""
import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
M0_PATH = ROOT / "scripts" / "adapt" / "run_target_support_m0.py"
spec = importlib.util.spec_from_file_location("m0_impl", M0_PATH)
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

METHODS = ["MIRepNet-SourceOnly", "MIRepNet-Head", "IFNet-SourceOnly", "IFNet-FT"]
ADAPT_METHODS = ["MIRepNet-Head", "IFNet-FT"]
PAIRS = [
    ("MIRepNet-Head", "MIRepNet-SourceOnly", "Head_minus_MIRepNetSourceOnly"),
    ("IFNet-FT", "IFNet-SourceOnly", "IFNetFT_minus_IFNetSourceOnly"),
    ("IFNet-FT", "MIRepNet-Head", "IFNetFT_minus_Head"),
    ("IFNet-FT", "MIRepNet-SourceOnly", "IFNetFT_minus_MIRepNetSourceOnly"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--source_seed", type=int, default=666)
    p.add_argument("--support_seeds", type=int, nargs="+", default=[666, 667, 668, 669, 670])
    p.add_argument("--adapt_seed", type=int, default=666)
    p.add_argument("--query_seed", type=int, default=666)
    p.add_argument("--query_fraction", type=float, default=0.3)
    p.add_argument("--support_draw", type=int, default=0)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--torch_threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    p.add_argument("--base_epochs_mirepnet", type=int, default=None)
    p.add_argument("--base_epochs_ifnet", type=int, default=None)
    p.add_argument("--adapt_epochs", type=int, default=30)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_weight_decay", type=float, default=None)
    p.add_argument("--adapt_batch_size", type=int, default=None)
    p.add_argument("--lora_rank", type=int, default=4)  # kept for adapter compatibility; unused here
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_targets", default="qv")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--no_trial_preds", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_list_tag(values):
    values = [int(v) for v in values]
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return "-".join(str(v) for v in values)


def default_prefix(args):
    kvals = "-".join(str(int(k)) for k in args.k_values)
    return Path("results") / "metrics" / (
        f"{args.dataset}_target_support_m2_k{kvals}"
        f"_source{args.source_seed}_support{seed_list_tag(args.support_seeds)}"
        f"_adapt{args.adapt_seed}_q{args.query_seed}_fixedeval"
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def split_seed_for(support_seed, fold, support_draw):
    return int(support_seed + 1009 * fold + 100000 * support_draw)


def choose_fixed_query_and_candidate(y, query_fraction, query_seed):
    all_idx = list(range(len(y)))
    cand_idx, q_idx = m0.data.split_indices_with_val_ratio(
        all_idx, y, val_split=query_fraction, seed=query_seed
    )
    return np.array(sorted(cand_idx), dtype=np.int64), np.array(sorted(q_idx), dtype=np.int64)


def nested_supports(candidate_idx, y, num_classes, k_values, seed):
    max_k = max(int(k) for k in k_values)
    rng = np.random.default_rng(seed)
    by_k = {int(k): [] for k in k_values}
    for cls in range(num_classes):
        cls_idx = np.array([int(i) for i in candidate_idx if int(y[int(i)]) == cls], dtype=np.int64)
        if len(cls_idx) < max_k:
            raise ValueError(f"class {cls} has only {len(cls_idx)} candidate trials, needs K={max_k}")
        perm = rng.permutation(cls_idx)
        for k in k_values:
            by_k[int(k)].extend(perm[:int(k)].tolist())
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in by_k.items()}


def target_array(model_name, ad, X_target, idx, fold):
    idx = np.asarray(idx, dtype=np.int64)
    if model_name == m0.BIG_MODEL:
        return ad.ea_pad_per_subject(X_target[idx], np.full(len(idx), fold, dtype=np.int64))
    return X_target[idx]


def metric_record(dataset, fold, source_seed, support_seed, adapt_seed, query_seed, k, method,
                  model_name, target_update, params_meta, source_meta, support_idx, candidate_idx,
                  q_fixed_idx, split_seed, y_query, logits, source_eval_reused=False):
    met = m0.eval_logits(logits, y_query, int(logits.shape[1]))
    return {
        "dataset": dataset,
        "fold": int(fold),
        "target_subject": int(fold),
        "source_seed": int(source_seed),
        "support_seed": int(support_seed),
        "adapt_seed": np.nan if adapt_seed is None else int(adapt_seed),
        "query_seed": int(query_seed),
        "support_k_per_class": int(k),
        "method": method,
        "model": model_name,
        "target_update": target_update,
        "available_support_n": int(len(support_idx)) if support_idx is not None else int(k * logits.shape[1]),
        "target_train_support_n": 0 if method.endswith("SourceOnly") else int(len(support_idx)),
        "candidate_n": int(len(candidate_idx)),
        "q_fixed_n": int(len(q_fixed_idx)),
        "split_seed": int(split_seed),
        "acc": round(float(met["acc"]), 4),
        "bac": round(float(met["bac"]), 6),
        "kappa": round(float(met["kappa"]), 6),
        "macro_f1": round(float(met["macro_f1"]), 6),
        "target_trainable_params": int(params_meta["target_trainable_params"]),
        "total_params": int(params_meta["total_params"]),
        "target_trainable_frac": round(float(params_meta["target_trainable_frac"]), 8),
        "source_epochs": int(source_meta["source_epochs"]),
        "source_lr": float(source_meta["source_lr"]),
        "source_weight_decay": float(source_meta["source_weight_decay"]),
        "source_batch_size": int(source_meta["source_batch_size"]),
        "adapt_epochs": 0 if method.endswith("SourceOnly") else int(params_meta["adapt_epochs"]),
        "adapt_lr": np.nan if method.endswith("SourceOnly") else float(params_meta["adapt_lr"]),
        "adapt_weight_decay": np.nan if method.endswith("SourceOnly") else float(params_meta["adapt_weight_decay"]),
        "adapt_batch_size": 0 if method.endswith("SourceOnly") else int(params_meta["adapt_batch_size"]),
        "fixed_eval_policy": "q_fixed_stratified_before_support_sampling",
        "support_policy": "nested_stratified_support_within_candidate_pool",
        "session_split_policy": "random_kshot_within_loaded_downstream_session",
        "source_eval_reused": bool(source_eval_reused),
    }


def prediction_records(dataset, fold, source_seed, support_seed, adapt_seed, query_seed, k, method,
                       model_name, q_fixed_idx, y_query, logits):
    probs = m0.softmax_np(logits)
    pred = logits.argmax(1)
    rows = []
    for row_i, raw_i in enumerate(q_fixed_idx):
        rec = {
            "dataset": dataset,
            "fold": int(fold),
            "target_subject": int(fold),
            "source_seed": int(source_seed),
            "support_seed": int(support_seed),
            "adapt_seed": np.nan if adapt_seed is None else int(adapt_seed),
            "query_seed": int(query_seed),
            "support_k_per_class": int(k),
            "method": method,
            "model": model_name,
            "q_fixed_row": int(row_i),
            "target_trial_index": int(raw_i),
            "y": int(y_query[row_i]),
            "pred": int(pred[row_i]),
            "correct": bool(pred[row_i] == y_query[row_i]),
            "confidence": float(probs[row_i].max()),
        }
        for c in range(logits.shape[1]):
            rec[f"logit_c{c}"] = float(logits[row_i, c])
            rec[f"prob_c{c}"] = float(probs[row_i, c])
        rows.append(rec)
    return rows


def split_records(dataset, fold, y, candidate_idx, q_fixed_idx, supports_by_seed, split_seeds, args):
    rows = []
    for split, indices in (("q_fixed", q_fixed_idx), ("support_candidate", candidate_idx)):
        for i in indices:
            rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "target_subject": int(fold),
                "source_seed": int(args.source_seed),
                "support_seed": "all",
                "query_seed": int(args.query_seed),
                "support_k_per_class": "all",
                "split_seed": "fixed_query" if split == "q_fixed" else "candidate_pool",
                "split": split,
                "target_trial_index": int(i),
                "y": int(y[int(i)]),
            })
    for support_seed, by_k in supports_by_seed.items():
        for k, support_idx in by_k.items():
            for i in support_idx:
                rows.append({
                    "dataset": dataset,
                    "fold": int(fold),
                    "target_subject": int(fold),
                    "source_seed": int(args.source_seed),
                    "support_seed": int(support_seed),
                    "query_seed": int(args.query_seed),
                    "support_k_per_class": int(k),
                    "split_seed": int(split_seeds[int(support_seed)]),
                    "split": "support",
                    "target_trial_index": int(i),
                    "y": int(y[int(i)]),
                })
    return rows


def check_records(dataset, fold, y, candidate_idx, q_fixed_idx, supports_by_seed, args):
    rows = []
    qset = set(int(i) for i in q_fixed_idx)
    cset = set(int(i) for i in candidate_idx)
    rows.append({
        "dataset": dataset,
        "fold": int(fold),
        "target_subject": int(fold),
        "check": "q_fixed_candidate_disjoint",
        "support_seed": "all",
        "support_k_per_class": "all",
        "passes": bool(qset.isdisjoint(cset) and len(qset | cset) == len(y)),
        "detail": f"q_fixed_n={len(qset)} candidate_n={len(cset)} total={len(y)}",
    })
    q_counts = {int(cls): int((y[q_fixed_idx] == cls).sum()) for cls in sorted(np.unique(y))}
    rows.append({
        "dataset": dataset,
        "fold": int(fold),
        "target_subject": int(fold),
        "check": "q_fixed_class_counts",
        "support_seed": "all",
        "support_k_per_class": "all",
        "passes": True,
        "detail": str(q_counts),
    })
    for support_seed, by_k in supports_by_seed.items():
        previous = None
        previous_k = None
        for k in sorted(by_k):
            support_idx = by_k[k]
            sset = set(int(i) for i in support_idx)
            counts = {int(cls): int((y[support_idx] == cls).sum()) for cls in sorted(np.unique(y))}
            rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "target_subject": int(fold),
                "check": "support_counts_disjoint_candidate",
                "support_seed": int(support_seed),
                "support_k_per_class": int(k),
                "passes": bool(all(v == int(k) for v in counts.values()) and qset.isdisjoint(sset) and sset.issubset(cset)),
                "detail": str(counts),
            })
            if previous is not None:
                rows.append({
                    "dataset": dataset,
                    "fold": int(fold),
                    "target_subject": int(fold),
                    "check": "nested_support",
                    "support_seed": int(support_seed),
                    "support_k_per_class": int(k),
                    "passes": bool(previous.issubset(sset)),
                    "detail": f"S{previous_k} subset S{k}",
                })
            previous = sset
            previous_k = int(k)
    return rows


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = float(np.mean(x)) if n else np.nan
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1 and stats is not None:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        se = sd / np.sqrt(n)
        t_p = float(stats.ttest_1samp(x, 0.0).pvalue)
        try:
            w_p = float(stats.wilcoxon(x, zero_method="wilcox").pvalue)
        except ValueError:
            w_p = np.nan
        return mean, mean - tcrit * se, mean + tcrit * se, t_p, w_p
    return mean, np.nan, np.nan, np.nan, np.nan


def summarize(metrics_rows, checks_rows, out_prefix):
    metrics = pd.DataFrame(metrics_rows)
    checks = pd.DataFrame(checks_rows)
    if metrics.empty:
        return
    out_prefix = Path(out_prefix)

    method_summary = metrics.groupby(["support_k_per_class", "method"], as_index=False).agg(
        mean_acc=("acc", "mean"),
        std_acc=("acc", "std"),
        mean_kappa=("kappa", "mean"),
        n_rows=("acc", "size"),
        target_trainable_params=("target_trainable_params", "mean"),
    )
    method_summary.to_csv(f"{out_prefix}_method_summary_by_k.csv", index=False)

    support_axis_rows = []
    for (k, method), g in metrics.groupby(["support_k_per_class", "method"], sort=False):
        by_subject = g.groupby(["fold", "support_seed"], as_index=False)["acc"].mean()
        axis = by_subject.groupby("fold")["acc"].agg(axis_mean="mean", axis_std="std", axis_min="min", axis_max="max").reset_index()
        axis["axis_range"] = axis["axis_max"] - axis["axis_min"]
        support_axis_rows.append({
            "support_k_per_class": int(k),
            "quantity": method,
            "axis": "support_seed",
            "mean_of_subject_means": float(axis["axis_mean"].mean()),
            "mean_support_std": float(axis["axis_std"].fillna(0).mean()),
            "mean_support_range": float(axis["axis_range"].mean()),
            "max_support_range": float(axis["axis_range"].max()),
        })
    support_axis = pd.DataFrame(support_axis_rows)
    support_axis.to_csv(f"{out_prefix}_support_variance_by_k.csv", index=False)

    subject_mean = metrics.groupby(["support_k_per_class", "fold", "target_subject", "method"], as_index=False).agg(
        mean_acc=("acc", "mean"),
        support_std_acc=("acc", "std"),
        support_min_acc=("acc", "min"),
        support_max_acc=("acc", "max"),
    )
    subject_mean.to_csv(f"{out_prefix}_subject_mean_by_k.csv", index=False)

    paired_rows = []
    diff_rows = []
    for k, kg in subject_mean.groupby("support_k_per_class"):
        piv = kg.pivot(index="target_subject", columns="method", values="mean_acc")
        for a, b, label in PAIRS:
            diff = (piv[a] - piv[b]).dropna()
            mean, lo, hi, t_p, w_p = ci95(diff.values)
            paired_rows.append({
                "support_k_per_class": int(k),
                "comparison": label,
                "method_a": a,
                "method_b": b,
                "mean_gain_pp": round(mean, 4),
                "ci95_low_pp": round(lo, 4),
                "ci95_high_pp": round(hi, 4),
                "improved_subjects": int((diff > 0).sum()),
                "nonnegative_subjects": int((diff >= 0).sum()),
                "n_subjects": int(len(diff)),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
            for subject, value in diff.items():
                diff_rows.append({
                    "support_k_per_class": int(k),
                    "comparison": label,
                    "target_subject": int(subject),
                    "diff_pp": float(value),
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(f"{out_prefix}_paired_stats_by_k.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(f"{out_prefix}_subject_diffs_by_k.csv", index=False)

    raw_piv = metrics.pivot_table(index=["support_k_per_class", "fold", "target_subject", "support_seed"], columns="method", values="acc")
    benefit_rows = []
    for k, kg in raw_piv.reset_index().groupby("support_k_per_class"):
        for a, b, label in PAIRS:
            tmp = kg[["target_subject", "support_seed", a, b]].copy()
            tmp["diff_pp"] = tmp[a] - tmp[b]
            by_subj = tmp.groupby("target_subject")["diff_pp"].agg(
                positive_supports=lambda x: int((np.asarray(x) > 0).sum()),
                nonnegative_supports=lambda x: int((np.asarray(x) >= 0).sum()),
                negative_supports=lambda x: int((np.asarray(x) < 0).sum()),
                mean_diff_pp="mean",
                min_diff_pp="min",
                max_diff_pp="max",
            ).reset_index()
            by_subj["support_k_per_class"] = int(k)
            by_subj["comparison"] = label
            by_subj["n_support_seeds"] = tmp["support_seed"].nunique()
            by_subj["sign_pattern"] = np.where(
                by_subj["positive_supports"].eq(by_subj["n_support_seeds"]),
                "positive_all_supports",
                np.where(by_subj["negative_supports"].eq(by_subj["n_support_seeds"]), "negative_all_supports", "mixed_or_zero"),
            )
            benefit_rows.append(by_subj)
    benefit = pd.concat(benefit_rows, ignore_index=True)
    benefit.to_csv(f"{out_prefix}_benefit_counts_by_subject_k.csv", index=False)

    winner_rows = []
    for _, row in raw_piv.reset_index().iterrows():
        vals = {m: float(row[m]) for m in METHODS}
        best = max(vals.values())
        winners = [m for m, v in vals.items() if np.isclose(v, best)]
        winner_rows.append({
            "support_k_per_class": int(row["support_k_per_class"]),
            "fold": int(row["fold"]),
            "target_subject": int(row["target_subject"]),
            "support_seed": int(row["support_seed"]),
            "winners": ";".join(winners),
            "tie_size": int(len(winners)),
            **{f"acc_{m}": vals[m] for m in METHODS},
        })
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(f"{out_prefix}_cell_winners_by_k.csv", index=False)
    win_summary = []
    for k, g in winners.groupby("support_k_per_class"):
        for method in METHODS:
            win_summary.append({
                "support_k_per_class": int(k),
                "method": method,
                "top_count_ties_count_all": int(g["winners"].str.split(";").apply(lambda xs: method in xs).sum()),
                "unique_top_count": int(((g["winners"] == method) & (g["tie_size"] == 1)).sum()),
                "n_cells": int(len(g)),
            })
    pd.DataFrame(win_summary).to_csv(f"{out_prefix}_winner_summary_by_k.csv", index=False)

    decision_rows = []
    for k in sorted(metrics["support_k_per_class"].unique()):
        kg_pair = paired[paired["support_k_per_class"] == k].set_index("comparison")
        kg_benefit = benefit[benefit["support_k_per_class"] == k]
        ifnet_head = kg_pair.loc["IFNetFT_minus_Head"]
        subject12 = kg_benefit[(kg_benefit["comparison"] == "IFNetFT_minus_Head") & (kg_benefit["target_subject"].isin([1, 2]))]
        decision_rows.append({
            "support_k_per_class": int(k),
            "check": "ifnet_ft_vs_head_summary",
            "value": f"mean {ifnet_head.mean_gain_pp:.4f} pp; improved subjects {int(ifnet_head.improved_subjects)}/9; subject1/2 mean diffs "
                     + ";".join(f"s{int(r.target_subject)}={r.mean_diff_pp:.4f}" for r in subject12.itertuples()),
        })
    pd.DataFrame(decision_rows).to_csv(f"{out_prefix}_decision_notes_by_k.csv", index=False)

    protocol_summary = pd.DataFrame([
        {"check": "all_checks_pass", "passes": bool(checks["passes"].all()) if not checks.empty else False,
         "detail": f"n_checks={len(checks)}"},
        {"check": "fixed_eval_protocol", "passes": True,
         "detail": "Q_fixed is selected before support sampling; support candidates exclude Q_fixed"},
        {"check": "nested_support_protocol", "passes": bool(checks.loc[checks["check"].eq("nested_support"), "passes"].all()) if not checks.empty else False,
         "detail": "S_K is nested within larger K for each subject/support seed/class"},
    ])
    protocol_summary.to_csv(f"{out_prefix}_protocol_summary.csv", index=False)

    print("Method summary by K:")
    print(method_summary.round(4).to_string(index=False))
    print("\nSupport variance by K:")
    print(support_axis.round(4).to_string(index=False))
    print("\nPaired stats by K:")
    print(paired.round(4).to_string(index=False))
    print("\nWinner summary by K:")
    print(pd.DataFrame(win_summary).to_string(index=False))
    print("\nProtocol summary:")
    print(protocol_summary.to_string(index=False))


def run_fold(dataset, dcfg, fold, device, args):
    num_classes = int(dcfg["num_classes"])
    k_values = sorted(int(k) for k in args.k_values)

    print(f"[fold {fold}] train fixed MIRepNet source seed={args.source_seed}", flush=True)
    big_ad, big_base, X_target, y_target, big_source_meta = m0.build_source_model(
        m0.BIG_MODEL, dataset, fold, args.source_seed, num_classes, device, args
    )
    candidate_idx, q_fixed_idx = choose_fixed_query_and_candidate(
        y_target, args.query_fraction, args.query_seed + 1009 * int(fold)
    )
    y_query = y_target[q_fixed_idx]
    X_big_q = target_array(m0.BIG_MODEL, big_ad, X_target, q_fixed_idx, fold)
    big_cfg = big_ad.cfg

    print(f"[fold {fold}] train fixed IFNet source seed={args.source_seed}", flush=True)
    small_ad, small_base, X_target_small, y_target_small, small_source_meta = m0.build_source_model(
        m0.SMALL_MODEL, dataset, fold, args.source_seed, num_classes, device, args
    )
    if not np.array_equal(y_target, y_target_small):
        raise ValueError("target labels are not aligned between MIRepNet and IFNet loaders")
    X_small_q = target_array(m0.SMALL_MODEL, small_ad, X_target_small, q_fixed_idx, fold)
    small_cfg = small_ad.cfg

    supports_by_seed = {}
    split_seeds = {}
    for support_seed in args.support_seeds:
        support_seed = int(support_seed)
        split_seed = split_seed_for(support_seed, int(fold), int(args.support_draw))
        supports_by_seed[support_seed] = nested_supports(candidate_idx, y_target, num_classes, k_values, split_seed)
        split_seeds[support_seed] = split_seed

    metrics_rows = []
    pred_rows = []
    split_rows = split_records(dataset, fold, y_target, candidate_idx, q_fixed_idx, supports_by_seed, split_seeds, args)
    checks_rows = check_records(dataset, fold, y_target, candidate_idx, q_fixed_idx, supports_by_seed, args)

    big_source_logits = m0.infer_logits(big_ad, big_base, X_big_q)
    small_source_logits = m0.infer_logits(small_ad, small_base, X_small_q)
    big_source_params = {
        "target_trainable_params": 0,
        "total_params": m0.count_params(big_base)[1],
        "target_trainable_frac": 0.0,
        "adapt_epochs": 0,
        "adapt_lr": np.nan,
        "adapt_weight_decay": np.nan,
        "adapt_batch_size": 0,
    }
    small_source_params = {
        "target_trainable_params": 0,
        "total_params": m0.count_params(small_base)[1],
        "target_trainable_frac": 0.0,
        "adapt_epochs": 0,
        "adapt_lr": np.nan,
        "adapt_weight_decay": np.nan,
        "adapt_batch_size": 0,
    }
    print(
        f"fold={fold} Q_fixed={len(q_fixed_idx)} candidate={len(candidate_idx)} "
        f"MIRepNet-SourceOnly={m0.eval_logits(big_source_logits, y_query, num_classes)['acc']:.2f} "
        f"IFNet-SourceOnly={m0.eval_logits(small_source_logits, y_query, num_classes)['acc']:.2f}",
        flush=True,
    )

    for support_seed in args.support_seeds:
        support_seed = int(support_seed)
        split_seed = split_seeds[support_seed]
        for k in k_values:
            support_idx = supports_by_seed[support_seed][k]
            y_support = y_target[support_idx]
            X_big_sup = target_array(m0.BIG_MODEL, big_ad, X_target, support_idx, fold)
            X_small_sup = target_array(m0.SMALL_MODEL, small_ad, X_target_small, support_idx, fold)

            metrics_rows.append(metric_record(
                dataset, fold, args.source_seed, support_seed, None, args.query_seed, k,
                "MIRepNet-SourceOnly", m0.BIG_MODEL, "none", big_source_params, big_source_meta,
                support_idx, candidate_idx, q_fixed_idx, split_seed, y_query, big_source_logits,
                source_eval_reused=True,
            ))
            metrics_rows.append(metric_record(
                dataset, fold, args.source_seed, support_seed, None, args.query_seed, k,
                "IFNet-SourceOnly", m0.SMALL_MODEL, "none", small_source_params, small_source_meta,
                support_idx, candidate_idx, q_fixed_idx, split_seed, y_query, small_source_logits,
                source_eval_reused=True,
            ))
            if not args.no_trial_preds:
                pred_rows.extend(prediction_records(
                    dataset, fold, args.source_seed, support_seed, None, args.query_seed, k,
                    "MIRepNet-SourceOnly", m0.BIG_MODEL, q_fixed_idx, y_query, big_source_logits,
                ))
                pred_rows.extend(prediction_records(
                    dataset, fold, args.source_seed, support_seed, None, args.query_seed, k,
                    "IFNet-SourceOnly", m0.SMALL_MODEL, q_fixed_idx, y_query, small_source_logits,
                ))

            fit_seed = int(args.adapt_seed) + m0.METHOD_SEED_OFFSET["MIRepNet-Head"] + 1000 * int(fold)
            m0.set_seed(fit_seed)
            head_model = copy.deepcopy(big_base)
            params_meta = m0.prepare_mirepnet_target_method(head_model, "MIRepNet-Head", args)
            params_meta = m0.with_adapt_meta(params_meta, big_cfg, args)
            head_model = m0.finetune_ce(
                big_ad, head_model, X_big_sup, y_support, num_classes,
                epochs=args.adapt_epochs, lr=args.adapt_lr,
                weight_decay=params_meta["adapt_weight_decay"], batch_size=params_meta["adapt_batch_size"],
                seed=fit_seed, train_mode=params_meta["train_mode"],
            )
            head_logits = m0.infer_logits(big_ad, head_model, X_big_q)
            metrics_rows.append(metric_record(
                dataset, fold, args.source_seed, support_seed, args.adapt_seed, args.query_seed, k,
                "MIRepNet-Head", m0.BIG_MODEL, params_meta["target_update"], params_meta, big_source_meta,
                support_idx, candidate_idx, q_fixed_idx, split_seed, y_query, head_logits,
            ))
            if not args.no_trial_preds:
                pred_rows.extend(prediction_records(
                    dataset, fold, args.source_seed, support_seed, args.adapt_seed, args.query_seed, k,
                    "MIRepNet-Head", m0.BIG_MODEL, q_fixed_idx, y_query, head_logits,
                ))
            del head_model

            fit_seed = int(args.adapt_seed) + m0.METHOD_SEED_OFFSET["IFNet-FT"] + 1000 * int(fold)
            m0.set_seed(fit_seed)
            small_model = copy.deepcopy(small_base)
            params_meta = m0.prepare_ifnet_fullft(small_model)
            params_meta = m0.with_adapt_meta(params_meta, small_cfg, args)
            small_model = m0.finetune_ce(
                small_ad, small_model, X_small_sup, y_support, num_classes,
                epochs=args.adapt_epochs, lr=args.adapt_lr,
                weight_decay=params_meta["adapt_weight_decay"], batch_size=params_meta["adapt_batch_size"],
                seed=fit_seed, train_mode=params_meta["train_mode"],
            )
            small_logits = m0.infer_logits(small_ad, small_model, X_small_q)
            metrics_rows.append(metric_record(
                dataset, fold, args.source_seed, support_seed, args.adapt_seed, args.query_seed, k,
                "IFNet-FT", m0.SMALL_MODEL, params_meta["target_update"], params_meta, small_source_meta,
                support_idx, candidate_idx, q_fixed_idx, split_seed, y_query, small_logits,
            ))
            if not args.no_trial_preds:
                pred_rows.extend(prediction_records(
                    dataset, fold, args.source_seed, support_seed, args.adapt_seed, args.query_seed, k,
                    "IFNet-FT", m0.SMALL_MODEL, q_fixed_idx, y_query, small_logits,
                ))
            del small_model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            latest = pd.DataFrame(metrics_rows)
            cell = latest[(latest["support_seed"] == support_seed) & (latest["support_k_per_class"] == k)]
            msg = " ".join(f"{r.method}={r.acc:.2f}" for r in cell.itertuples())
            print(f"fold={fold} support={support_seed} K={k} | {msg}", flush=True)

    del big_base, small_base
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics_rows, pred_rows, split_rows, checks_rows


def main():
    args = parse_args()
    args.k_values = sorted(int(k) for k in args.k_values)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    dataset = args.dataset
    dcfg = m0.config.load_dataset_config(dataset)
    folds = args.folds if args.folds is not None else list(range(int(dcfg["num_subjects"])))
    device = m0.device_from_gpu(args.gpu)
    out_prefix = Path(args.out_prefix) if args.out_prefix else default_prefix(args)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(f"{out_prefix}_metrics.csv")
    pred_csv = Path(f"{out_prefix}_trial_preds.csv")
    split_csv = Path(f"{out_prefix}_fixed_query_support_split.csv")
    checks_csv = Path(f"{out_prefix}_checks.csv")
    outputs = [metrics_csv, split_csv, checks_csv]
    if not args.no_trial_preds:
        outputs.append(pred_csv)
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise FileExistsError(f"output exists for {out_prefix}; pass --overwrite")

    print(
        f"[cfg] dataset={dataset} K={args.k_values} source_seed={args.source_seed} "
        f"support_seeds={args.support_seeds} adapt_seed={args.adapt_seed} "
        f"query_seed={args.query_seed} query_fraction={args.query_fraction:g} folds={folds} "
        f"adapt_epochs={args.adapt_epochs} adapt_lr={args.adapt_lr:g} device={device}",
        flush=True,
    )
    print("[cfg] methods=MIRepNet-SourceOnly,MIRepNet-Head,IFNet-SourceOnly,IFNet-FT", flush=True)

    all_metrics, all_preds, all_splits, all_checks = [], [], [], []
    for fold in folds:
        mr, pr, sr, cr = run_fold(dataset, dcfg, int(fold), device, args)
        all_metrics.extend(mr)
        all_preds.extend(pr)
        all_splits.extend(sr)
        all_checks.extend(cr)
        write_csv(metrics_csv, all_metrics)
        if not args.no_trial_preds:
            write_csv(pred_csv, all_preds)
        write_csv(split_csv, all_splits)
        write_csv(checks_csv, all_checks)
        summarize(all_metrics, all_checks, out_prefix)
        print(f"[write] {metrics_csv}", flush=True)

    summarize(all_metrics, all_checks, out_prefix)
    print(f"\nWrote {metrics_csv}", flush=True)
    if not args.no_trial_preds:
        print(f"Wrote {pred_csv}", flush=True)
    print(f"Wrote {split_csv}", flush=True)
    print(f"Wrote {checks_csv}", flush=True)
    print(f"Wrote {out_prefix}_*.csv", flush=True)


if __name__ == "__main__":
    main()
