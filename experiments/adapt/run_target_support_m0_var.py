"""M0-VAR controlled target-support adaptation variance split.

Diagnostic protocol:
  * Train each source model once per LOSO fold with fixed source_seed.
  * Cross support_seed x adapt_seed for target adaptation.
  * Evaluate every adapted model on Q_common, the complement of the union of all
    support sets for that target subject.
  * Run only MIRepNet-Head, MIRepNet-LoRA, and IFNet-FT. SourceOnly is evaluated
    once per target subject on Q_common.

This is a variance diagnostic and does not replace the main random K-shot M0 table.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
M0_PATH = ROOT / "scripts" / "adapt" / "run_target_support_m0.py"
spec = importlib.util.spec_from_file_location("m0_impl", M0_PATH)
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

METHODS = ["MIRepNet-Head", "MIRepNet-LoRA", "IFNet-FT"]
PAIRS = [
    ("MIRepNet-LoRA", "MIRepNet-Head", "LoRA_minus_Head"),
    ("IFNet-FT", "MIRepNet-Head", "IFNet_minus_Head"),
    ("IFNet-FT", "MIRepNet-LoRA", "IFNet_minus_LoRA"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--source_seed", type=int, default=666)
    p.add_argument("--support_seeds", type=int, nargs="+", default=[666, 667, 668])
    p.add_argument("--adapt_seeds", type=int, nargs="+", default=[666, 667, 668])
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
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_targets", default="qv")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_list_tag(values):
    values = [int(v) for v in values]
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return "-".join(str(v) for v in values)


def default_prefix(args):
    tag = (
        f"{args.dataset}_target_support_m0var_k{args.k_per_class}"
        f"_source{args.source_seed}"
        f"_support{seed_list_tag(args.support_seeds)}"
        f"_adapt{seed_list_tag(args.adapt_seeds)}"
    )
    return Path("results") / "metrics" / tag


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def target_array(model_name, ad, X_target, idx, fold):
    idx = np.asarray(idx, dtype=np.int64)
    if model_name == m0.BIG_MODEL:
        return ad.ea_pad_per_subject(
            X_target[idx], np.full(len(idx), fold, dtype=np.int64)
        )
    return X_target[idx]


def split_seed_for(support_seed, fold, support_draw):
    return int(support_seed + 1009 * fold + 100000 * support_draw)


def support_and_common_query(y_target, num_classes, fold, args):
    support_by_seed = {}
    split_seed_by_seed = {}
    union = set()
    for support_seed in args.support_seeds:
        split_seed = split_seed_for(int(support_seed), int(fold), int(args.support_draw))
        support_idx, _ = m0.kshot_indices_per_class(y_target, num_classes, args.k_per_class, split_seed)
        support_by_seed[int(support_seed)] = support_idx
        split_seed_by_seed[int(support_seed)] = split_seed
        union.update(int(i) for i in support_idx)
    q_common = np.array([i for i in range(len(y_target)) if int(i) not in union], dtype=np.int64)
    return support_by_seed, split_seed_by_seed, np.array(sorted(union), dtype=np.int64), q_common


def add_metric(rows, dataset, fold, source_seed, support_seed, adapt_seed, method, model_name,
               target_update, params_meta, source_meta, support_idx, q_common, support_union,
               split_seed, y_query, logits):
    met = m0.eval_logits(logits, y_query, int(logits.shape[1]))
    rows.append({
        "dataset": dataset,
        "fold": int(fold),
        "target_subject": int(fold),
        "source_seed": int(source_seed),
        "support_seed": np.nan if support_seed is None else int(support_seed),
        "adapt_seed": np.nan if adapt_seed is None else int(adapt_seed),
        "method": method,
        "model": model_name,
        "target_update": target_update,
        "support_k_per_class": int(len(support_idx) // logits.shape[1]) if support_idx is not None else 0,
        "support_n": int(len(support_idx)) if support_idx is not None else 0,
        "support_union_n": int(len(support_union)),
        "q_common_n": int(len(q_common)),
        "support_draw": int(getattr(params_meta, "support_draw", 0)) if isinstance(params_meta, argparse.Namespace) else 0,
        "split_seed": np.nan if split_seed is None else int(split_seed),
        "acc": round(float(met["acc"]), 4),
        "bac": round(float(met["bac"]), 6),
        "kappa": round(float(met["kappa"]), 6),
        "macro_f1": round(float(met["macro_f1"]), 6),
        "target_trainable_params": int(params_meta["target_trainable_params"]),
        "total_params": int(params_meta["total_params"]),
        "target_trainable_frac": round(float(params_meta["target_trainable_frac"]), 8),
        "n_lora_layers": int(params_meta.get("n_lora_layers", 0)),
        "source_epochs": int(source_meta["source_epochs"]),
        "source_lr": float(source_meta["source_lr"]),
        "source_weight_decay": float(source_meta["source_weight_decay"]),
        "source_batch_size": int(source_meta["source_batch_size"]),
        "adapt_epochs": 0 if adapt_seed is None else int(params_meta["adapt_epochs"]),
        "adapt_lr": np.nan if adapt_seed is None else float(params_meta["adapt_lr"]),
        "adapt_weight_decay": np.nan if adapt_seed is None else float(params_meta["adapt_weight_decay"]),
        "adapt_batch_size": 0 if adapt_seed is None else int(params_meta["adapt_batch_size"]),
        "query_policy": "q_common_excludes_union_of_all_support_seeds",
        "session_split_policy": "random_kshot_within_loaded_downstream_session",
        "diagnostic_only": True,
    })


def prediction_rows(dataset, fold, source_seed, support_seed, adapt_seed, method, model_name,
                    q_common, y_query, logits):
    probs = m0.softmax_np(logits)
    pred = logits.argmax(1)
    rows = []
    for row_i, raw_i in enumerate(q_common):
        rec = {
            "dataset": dataset,
            "fold": int(fold),
            "target_subject": int(fold),
            "source_seed": int(source_seed),
            "support_seed": np.nan if support_seed is None else int(support_seed),
            "adapt_seed": np.nan if adapt_seed is None else int(adapt_seed),
            "method": method,
            "model": model_name,
            "q_common_row": int(row_i),
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


def support_rows(dataset, fold, y_target, support_by_seed, split_seed_by_seed, support_union, q_common, args):
    rows = []
    for support_seed, support_idx in support_by_seed.items():
        for i in support_idx:
            rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "target_subject": int(fold),
                "source_seed": int(args.source_seed),
                "support_seed": int(support_seed),
                "adapt_seed": "all",
                "support_k_per_class": int(args.k_per_class),
                "split_seed": int(split_seed_by_seed[support_seed]),
                "split": "support",
                "target_trial_index": int(i),
                "y": int(y_target[i]),
                "query_policy": "q_common_excludes_union_of_all_support_seeds",
            })
    for split, indices in (("support_union", support_union), ("q_common", q_common)):
        for i in indices:
            rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "target_subject": int(fold),
                "source_seed": int(args.source_seed),
                "support_seed": "all",
                "adapt_seed": "all",
                "support_k_per_class": int(args.k_per_class),
                "split_seed": "union",
                "split": split,
                "target_trial_index": int(i),
                "y": int(y_target[i]),
                "query_policy": "q_common_excludes_union_of_all_support_seeds",
            })
    return rows


def check_rows(dataset, fold, y_target, support_by_seed, support_union, q_common, args):
    rows = []
    qset = set(int(i) for i in q_common)
    union_set = set(int(i) for i in support_union)
    counts_common = {int(c): int((y_target[q_common] == c).sum()) for c in sorted(np.unique(y_target))}
    for support_seed, support_idx in support_by_seed.items():
        sset = set(int(i) for i in support_idx)
        counts = {int(c): int((y_target[support_idx] == c).sum()) for c in sorted(np.unique(y_target))}
        rows.append({
            "dataset": dataset,
            "fold": int(fold),
            "target_subject": int(fold),
            "source_seed": int(args.source_seed),
            "support_seed": int(support_seed),
            "support_counts": str(counts),
            "support_counts_ok": bool(all(v == args.k_per_class for v in counts.values())),
            "support_q_common_disjoint": bool(len(sset & qset) == 0),
            "support_in_union": bool(sset.issubset(union_set)),
            "q_common_n": int(len(q_common)),
            "q_common_counts": str(counts_common),
            "support_union_n": int(len(support_union)),
            "query_policy": "q_common_excludes_union_of_all_support_seeds",
        })
    rows.append({
        "dataset": dataset,
        "fold": int(fold),
        "target_subject": int(fold),
        "source_seed": int(args.source_seed),
        "support_seed": "all",
        "support_counts": "all_support_seeds_union",
        "support_counts_ok": True,
        "support_q_common_disjoint": bool(len(union_set & qset) == 0),
        "support_in_union": True,
        "q_common_n": int(len(q_common)),
        "q_common_counts": str(counts_common),
        "support_union_n": int(len(support_union)),
        "query_policy": "q_common_excludes_union_of_all_support_seeds",
    })
    return rows


def summarize(metrics_rows, out_prefix):
    metrics = pd.DataFrame(metrics_rows)
    if metrics.empty:
        return
    out_prefix = Path(out_prefix)

    source = metrics[metrics["method"] == "MIRepNet-SourceOnly"].copy()
    adapt = metrics[metrics["method"].isin(METHODS)].copy()

    method_summary_rows = []
    if not source.empty:
        method_summary_rows.append({
            "method": "MIRepNet-SourceOnly",
            "n_cells": int(len(source)),
            "mean_acc": float(source["acc"].mean()),
            "std_acc": float(source["acc"].std(ddof=1)) if len(source) > 1 else 0.0,
            "target_trainable_params": 0,
        })
    for method, g in adapt.groupby("method", sort=False):
        method_summary_rows.append({
            "method": method,
            "n_cells": int(len(g)),
            "mean_acc": float(g["acc"].mean()),
            "std_acc": float(g["acc"].std(ddof=1)) if len(g) > 1 else 0.0,
            "target_trainable_params": float(g["target_trainable_params"].mean()),
        })
    pd.DataFrame(method_summary_rows).to_csv(f"{out_prefix}_method_summary.csv", index=False)

    axis_rows = []
    for method, g in adapt.groupby("method", sort=False):
        support_means = g.groupby(["fold", "support_seed"], as_index=False)["acc"].mean()
        support_axis = support_means.groupby("fold")["acc"].agg(
            axis_mean="mean", axis_std="std", axis_min="min", axis_max="max"
        ).reset_index()
        support_axis["axis_range"] = support_axis["axis_max"] - support_axis["axis_min"]
        axis_rows.append({
            "quantity": method,
            "axis": "support_seed",
            "mean_of_axis_means": float(support_axis["axis_mean"].mean()),
            "mean_axis_std": float(support_axis["axis_std"].mean()),
            "mean_axis_range": float(support_axis["axis_range"].mean()),
            "max_axis_range": float(support_axis["axis_range"].max()),
        })
        adapt_means = g.groupby(["fold", "adapt_seed"], as_index=False)["acc"].mean()
        adapt_axis = adapt_means.groupby("fold")["acc"].agg(
            axis_mean="mean", axis_std="std", axis_min="min", axis_max="max"
        ).reset_index()
        adapt_axis["axis_range"] = adapt_axis["axis_max"] - adapt_axis["axis_min"]
        axis_rows.append({
            "quantity": method,
            "axis": "adapt_seed",
            "mean_of_axis_means": float(adapt_axis["axis_mean"].mean()),
            "mean_axis_std": float(adapt_axis["axis_std"].mean()),
            "mean_axis_range": float(adapt_axis["axis_range"].mean()),
            "max_axis_range": float(adapt_axis["axis_range"].max()),
        })

    piv = adapt.pivot_table(index=["fold", "target_subject", "support_seed", "adapt_seed"], columns="method", values="acc")
    pair_rows = []
    for a, b, label in PAIRS:
        diff = (piv[a] - piv[b]).rename("diff_pp").reset_index()
        diff["comparison"] = label
        pair_rows.append(diff)
        support_means = diff.groupby(["fold", "support_seed"], as_index=False)["diff_pp"].mean()
        support_axis = support_means.groupby("fold")["diff_pp"].agg(
            axis_mean="mean", axis_std="std", axis_min="min", axis_max="max"
        ).reset_index()
        support_axis["axis_range"] = support_axis["axis_max"] - support_axis["axis_min"]
        axis_rows.append({
            "quantity": label,
            "axis": "support_seed",
            "mean_of_axis_means": float(support_axis["axis_mean"].mean()),
            "mean_axis_std": float(support_axis["axis_std"].mean()),
            "mean_axis_range": float(support_axis["axis_range"].mean()),
            "max_axis_range": float(support_axis["axis_range"].max()),
        })
        adapt_means = diff.groupby(["fold", "adapt_seed"], as_index=False)["diff_pp"].mean()
        adapt_axis = adapt_means.groupby("fold")["diff_pp"].agg(
            axis_mean="mean", axis_std="std", axis_min="min", axis_max="max"
        ).reset_index()
        adapt_axis["axis_range"] = adapt_axis["axis_max"] - adapt_axis["axis_min"]
        axis_rows.append({
            "quantity": label,
            "axis": "adapt_seed",
            "mean_of_axis_means": float(adapt_axis["axis_mean"].mean()),
            "mean_axis_std": float(adapt_axis["axis_std"].mean()),
            "mean_axis_range": float(adapt_axis["axis_range"].mean()),
            "max_axis_range": float(adapt_axis["axis_range"].max()),
        })
    pair_diffs = pd.concat(pair_rows, ignore_index=True)
    pair_diffs.to_csv(f"{out_prefix}_pair_diffs.csv", index=False)
    pd.DataFrame(axis_rows).to_csv(f"{out_prefix}_axis_variance.csv", index=False)

    win_rows = []
    for idx, row in piv.reset_index().iterrows():
        vals = {method: float(row[method]) for method in METHODS}
        best = max(vals.values())
        winners = [m for m, v in vals.items() if np.isclose(v, best)]
        win_rows.append({
            "fold": int(row["fold"]),
            "target_subject": int(row["target_subject"]),
            "support_seed": int(row["support_seed"]),
            "adapt_seed": int(row["adapt_seed"]),
            "winners": ";".join(winners),
            "tie_size": int(len(winners)),
            **{f"acc_{m}": vals[m] for m in METHODS},
        })
    winners = pd.DataFrame(win_rows)
    winners.to_csv(f"{out_prefix}_cell_winners.csv", index=False)
    winner_summary = []
    for method in METHODS:
        winner_summary.append({
            "method": method,
            "top_count_ties_count_all": int(winners["winners"].str.split(";").apply(lambda xs: method in xs).sum()),
            "unique_top_count": int(((winners["winners"] == method) & (winners["tie_size"] == 1)).sum()),
            "n_cells": int(len(winners)),
        })
    pd.DataFrame(winner_summary).to_csv(f"{out_prefix}_winner_summary.csv", index=False)

    benefit = pair_diffs.groupby(["comparison", "fold", "target_subject"], as_index=False).agg(
        positive_cells=("diff_pp", lambda x: int((np.asarray(x) > 0).sum())),
        nonnegative_cells=("diff_pp", lambda x: int((np.asarray(x) >= 0).sum())),
        negative_cells=("diff_pp", lambda x: int((np.asarray(x) < 0).sum())),
        mean_diff_pp=("diff_pp", "mean"),
        min_diff_pp=("diff_pp", "min"),
        max_diff_pp=("diff_pp", "max"),
    )
    benefit["n_cells"] = len(set(zip(adapt["support_seed"], adapt["adapt_seed"])))
    benefit["sign_pattern"] = np.where(
        benefit["positive_cells"].eq(benefit["n_cells"]),
        "positive_all_cells",
        np.where(benefit["negative_cells"].eq(benefit["n_cells"]), "negative_all_cells", "mixed_or_zero"),
    )
    benefit.to_csv(f"{out_prefix}_benefit_counts_by_subject.csv", index=False)

    decision_rows = []
    lora_head = benefit[benefit["comparison"] == "LoRA_minus_Head"]
    ifnet_lora = benefit[benefit["comparison"] == "IFNet_minus_LoRA"]
    decision_rows.append({
        "check": "lora_minus_head_sign_stability",
        "positive_all_subjects": int((lora_head["sign_pattern"] == "positive_all_cells").sum()),
        "negative_all_subjects": int((lora_head["sign_pattern"] == "negative_all_cells").sum()),
        "mixed_subjects": int((lora_head["sign_pattern"] == "mixed_or_zero").sum()),
        "mean_diff_pp": float(pair_diffs[pair_diffs["comparison"] == "LoRA_minus_Head"]["diff_pp"].mean()),
    })
    decision_rows.append({
        "check": "ifnet_minus_lora_sign_stability",
        "positive_all_subjects": int((ifnet_lora["sign_pattern"] == "positive_all_cells").sum()),
        "negative_all_subjects": int((ifnet_lora["sign_pattern"] == "negative_all_cells").sum()),
        "mixed_subjects": int((ifnet_lora["sign_pattern"] == "mixed_or_zero").sum()),
        "mean_diff_pp": float(pair_diffs[pair_diffs["comparison"] == "IFNet_minus_LoRA"]["diff_pp"].mean()),
    })
    pd.DataFrame(decision_rows).to_csv(f"{out_prefix}_sign_stability.csv", index=False)

    print("Method summary:")
    print(pd.DataFrame(method_summary_rows).round(4).to_string(index=False))
    print("\nAxis variance:")
    print(pd.DataFrame(axis_rows).round(4).to_string(index=False))
    print("\nWinner summary:")
    print(pd.DataFrame(winner_summary).to_string(index=False))
    print("\nBenefit counts by comparison:")
    print(benefit.groupby("comparison")[["positive_cells", "negative_cells", "mean_diff_pp"]].mean().round(4).to_string())
    print("\nSign stability:")
    print(pd.DataFrame(decision_rows).round(4).to_string(index=False))


def run_fold(dataset, dcfg, fold, device, args):
    num_classes = int(dcfg["num_classes"])

    print(f"[fold {fold}] train fixed MIRepNet source seed={args.source_seed}", flush=True)
    big_ad, big_base, X_target, y_target, big_source_meta = m0.build_source_model(
        m0.BIG_MODEL, dataset, fold, args.source_seed, num_classes, device, args
    )
    support_by_seed, split_seed_by_seed, support_union, q_common = support_and_common_query(
        y_target, num_classes, fold, args
    )
    y_query = y_target[q_common]
    X_big_q = target_array(m0.BIG_MODEL, big_ad, X_target, q_common, fold)
    big_cfg = big_ad.cfg

    print(f"[fold {fold}] train fixed IFNet source seed={args.source_seed}", flush=True)
    small_ad, small_base, X_target_small, y_target_small, small_source_meta = m0.build_source_model(
        m0.SMALL_MODEL, dataset, fold, args.source_seed, num_classes, device, args
    )
    if not np.array_equal(y_target, y_target_small):
        raise ValueError("target labels are not aligned between MIRepNet and IFNet loaders")
    X_small_q = target_array(m0.SMALL_MODEL, small_ad, X_target_small, q_common, fold)
    small_cfg = small_ad.cfg

    metrics_rows = []
    pred_rows = []
    split_rows = support_rows(dataset, fold, y_target, support_by_seed, split_seed_by_seed, support_union, q_common, args)
    checks = check_rows(dataset, fold, y_target, support_by_seed, support_union, q_common, args)

    source_logits = m0.infer_logits(big_ad, big_base, X_big_q)
    source_params = {
        "target_trainable_params": 0,
        "total_params": m0.count_params(big_base)[1],
        "target_trainable_frac": 0.0,
        "n_lora_layers": 0,
        "adapt_epochs": 0,
        "adapt_lr": np.nan,
        "adapt_weight_decay": np.nan,
        "adapt_batch_size": 0,
    }
    add_metric(metrics_rows, dataset, fold, args.source_seed, None, None,
               "MIRepNet-SourceOnly", m0.BIG_MODEL, "none", source_params,
               big_source_meta, None, q_common, support_union, None, y_query, source_logits)
    pred_rows.extend(prediction_rows(dataset, fold, args.source_seed, None, None,
                                     "MIRepNet-SourceOnly", m0.BIG_MODEL, q_common, y_query, source_logits))
    print(
        f"fold={fold} source_seed={args.source_seed} MIRepNet-SourceOnly@Q_common | "
        f"q={len(q_common)} acc={metrics_rows[-1]['acc']:.2f}",
        flush=True,
    )

    for support_seed in args.support_seeds:
        support_seed = int(support_seed)
        support_idx = support_by_seed[support_seed]
        split_seed = split_seed_by_seed[support_seed]
        y_support = y_target[support_idx]
        X_big_sup = target_array(m0.BIG_MODEL, big_ad, X_target, support_idx, fold)
        X_small_sup = target_array(m0.SMALL_MODEL, small_ad, X_target_small, support_idx, fold)
        for adapt_seed in args.adapt_seeds:
            adapt_seed = int(adapt_seed)
            for method in METHODS:
                if method in {"MIRepNet-Head", "MIRepNet-LoRA"}:
                    fit_seed = adapt_seed + m0.METHOD_SEED_OFFSET[method] + 1000 * fold
                    m0.set_seed(fit_seed)
                    model = copy.deepcopy(big_base)
                    params_meta = m0.prepare_mirepnet_target_method(model, method, args)
                    params_meta = m0.with_adapt_meta(params_meta, big_cfg, args)
                    model = m0.finetune_ce(
                        big_ad,
                        model,
                        X_big_sup,
                        y_support,
                        num_classes,
                        epochs=args.adapt_epochs,
                        lr=args.adapt_lr,
                        weight_decay=params_meta["adapt_weight_decay"],
                        batch_size=params_meta["adapt_batch_size"],
                        seed=fit_seed,
                        train_mode=params_meta["train_mode"],
                    )
                    logits = m0.infer_logits(big_ad, model, X_big_q)
                    add_metric(metrics_rows, dataset, fold, args.source_seed, support_seed, adapt_seed,
                               method, m0.BIG_MODEL, params_meta["target_update"], params_meta,
                               big_source_meta, support_idx, q_common, support_union, split_seed,
                               y_query, logits)
                    pred_rows.extend(prediction_rows(dataset, fold, args.source_seed, support_seed, adapt_seed,
                                                     method, m0.BIG_MODEL, q_common, y_query, logits))
                    del model
                elif method == "IFNet-FT":
                    fit_seed = adapt_seed + m0.METHOD_SEED_OFFSET[method] + 1000 * fold
                    m0.set_seed(fit_seed)
                    model = copy.deepcopy(small_base)
                    params_meta = m0.prepare_ifnet_fullft(model)
                    params_meta = m0.with_adapt_meta(params_meta, small_cfg, args)
                    model = m0.finetune_ce(
                        small_ad,
                        model,
                        X_small_sup,
                        y_support,
                        num_classes,
                        epochs=args.adapt_epochs,
                        lr=args.adapt_lr,
                        weight_decay=params_meta["adapt_weight_decay"],
                        batch_size=params_meta["adapt_batch_size"],
                        seed=fit_seed,
                        train_mode=params_meta["train_mode"],
                    )
                    logits = m0.infer_logits(small_ad, model, X_small_q)
                    add_metric(metrics_rows, dataset, fold, args.source_seed, support_seed, adapt_seed,
                               method, m0.SMALL_MODEL, params_meta["target_update"], params_meta,
                               small_source_meta, support_idx, q_common, support_union, split_seed,
                               y_query, logits)
                    pred_rows.extend(prediction_rows(dataset, fold, args.source_seed, support_seed, adapt_seed,
                                                     method, m0.SMALL_MODEL, q_common, y_query, logits))
                    del model
                else:  # pragma: no cover
                    raise ValueError(method)
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            cell = pd.DataFrame(metrics_rows)
            latest = cell[(cell["support_seed"] == support_seed) & (cell["adapt_seed"] == adapt_seed)]
            msg = " ".join(f"{r.method}={r.acc:.2f}" for r in latest.itertuples())
            print(
                f"fold={fold} source={args.source_seed} support={support_seed} adapt={adapt_seed} | {msg}",
                flush=True,
            )

    del big_base, small_base
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics_rows, pred_rows, split_rows, checks


def main():
    args = parse_args()
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
    support_csv = Path(f"{out_prefix}_support_common_split.csv")
    checks_csv = Path(f"{out_prefix}_checks.csv")
    if any(p.exists() for p in [metrics_csv, pred_csv, support_csv, checks_csv]) and not args.overwrite:
        raise FileExistsError(f"output exists for {out_prefix}; pass --overwrite")

    print(
        f"[cfg] dataset={dataset} K/class={args.k_per_class} source_seed={args.source_seed} "
        f"support_seeds={args.support_seeds} adapt_seeds={args.adapt_seeds} folds={folds} "
        f"adapt_epochs={args.adapt_epochs} adapt_lr={args.adapt_lr:g} device={device}",
        flush=True,
    )
    print("[cfg] methods=MIRepNet-Head,MIRepNet-LoRA,IFNet-FT; SourceOnly once on Q_common", flush=True)

    all_metrics, all_preds, all_splits, all_checks = [], [], [], []
    for fold in folds:
        mr, pr, sr, cr = run_fold(dataset, dcfg, int(fold), device, args)
        all_metrics.extend(mr)
        all_preds.extend(pr)
        all_splits.extend(sr)
        all_checks.extend(cr)
        write_csv(metrics_csv, all_metrics)
        write_csv(pred_csv, all_preds)
        write_csv(support_csv, all_splits)
        write_csv(checks_csv, all_checks)
        summarize(all_metrics, out_prefix)
        print(f"[write] {metrics_csv}", flush=True)

    summarize(all_metrics, out_prefix)
    print(f"\nWrote {metrics_csv}", flush=True)
    print(f"Wrote {pred_csv}", flush=True)
    print(f"Wrote {support_csv}", flush=True)
    print(f"Wrote {checks_csv}", flush=True)
    print(f"Wrote {out_prefix}_*.csv", flush=True)


if __name__ == "__main__":
    main()
