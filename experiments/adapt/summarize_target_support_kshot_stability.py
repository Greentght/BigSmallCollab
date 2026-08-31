import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

METHODS = ["SourceOnly", "MIRepNet-Head", "MIRepNet-LoRA", "IFNet-FT"]
DISPLAY = {"SourceOnly": "MIRepNet-SourceOnly"}
PAIRS = [
    ("MIRepNet-Head", "SourceOnly", "Head_minus_SourceOnly"),
    ("MIRepNet-LoRA", "SourceOnly", "LoRA_minus_SourceOnly"),
    ("MIRepNet-LoRA", "MIRepNet-Head", "LoRA_minus_Head"),
    ("IFNet-FT", "MIRepNet-Head", "IFNetFT_minus_Head"),
    ("IFNet-FT", "MIRepNet-LoRA", "IFNetFT_minus_LoRA"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[666, 667, 668])
    p.add_argument("--metrics_csvs", nargs="+", required=True)
    p.add_argument("--support_csvs", nargs="+", default=[])
    p.add_argument("--out_prefix", default=None)
    return p.parse_args()


def read_csvs(paths):
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def display_method(method):
    return DISPLAY.get(method, method)


def ci95_and_tests(values):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = float(np.mean(x)) if n else np.nan
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and stats is not None:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        paired_t_p = float(stats.ttest_1samp(x, 0.0).pvalue)
        try:
            wilcoxon_p = float(stats.wilcoxon(x, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
    else:
        tcrit = np.nan
        paired_t_p = np.nan
        wilcoxon_p = np.nan
    return mean, mean - tcrit * se, mean + tcrit * se, paired_t_p, wilcoxon_p


def validate_metrics(df, dataset, k_per_class, seeds):
    df = df.copy()
    df = df[(df["dataset"] == dataset) & (df["support_k_per_class"] == k_per_class)]
    df = df[df["seed"].isin(seeds) & df["method"].isin(METHODS)]
    if df.empty:
        raise ValueError("no matching metric rows")
    dup = df.duplicated(["seed", "fold", "method"], keep=False)
    if dup.any():
        raise ValueError("duplicate seed/fold/method rows:\n" + df.loc[dup].to_string(index=False))
    expected = pd.MultiIndex.from_product([seeds, sorted(df["fold"].unique()), METHODS], names=["seed", "fold", "method"])
    have = pd.MultiIndex.from_frame(df[["seed", "fold", "method"]])
    missing = expected.difference(have)
    if len(missing):
        raise ValueError("missing seed/fold/method rows: " + str(list(missing)[:20]))
    return df


def protocol_checks(metrics, support, seeds, k_per_class):
    rows = []
    g = metrics.groupby(["seed", "fold"])
    rows.append({
        "check": "one_split_seed_per_seed_fold_across_methods",
        "passes": bool((g["split_seed"].nunique() == 1).all()),
        "detail": "same seed/fold rows share split_seed across methods",
    })
    rows.append({
        "check": "one_test_n_per_seed_fold_across_methods",
        "passes": bool((g["test_n"].nunique() == 1).all()),
        "detail": "same seed/fold rows share query size across methods",
    })
    expected_policy = "random_kshot_within_loaded_downstream_session"
    inferred_policy = expected_policy + "_legacy_inferred"
    if "session_split_policy" in metrics.columns:
        policy = metrics["session_split_policy"].astype(str)
        exact_n = int(policy.eq(expected_policy).sum())
        inferred_n = int(policy.eq(inferred_policy).sum())
        passes = bool(policy.isin([expected_policy, inferred_policy]).all())
        detail = (
            f"exact rows={exact_n}; legacy inferred rows={inferred_n}; "
            f"all rows reported as random K-shot, not cross-session adaptation"
        )
    else:
        passes = False
        detail = "session_split_policy column missing"
    rows.append({
        "check": "random_kshot_protocol_marked_or_legacy_inferred",
        "passes": passes,
        "detail": detail,
    })
    if support is not None and not support.empty:
        support = support[support["seed"].isin(seeds)].copy()
        sup = support[support["split"] == "support"]
        counts = sup.groupby(["seed", "fold", "y"]).size()
        rows.append({
            "check": "support_k_per_class_exact",
            "passes": bool((counts == k_per_class).all()),
            "detail": f"all support class counts equal {k_per_class}",
        })
        disjoint_ok = True
        for (_seed, _fold), gg in support.groupby(["seed", "fold"]):
            sidx = set(gg.loc[gg["split"] == "support", "target_trial_index"].astype(int))
            tidx = set(gg.loc[gg["split"] == "test", "target_trial_index"].astype(int))
            if sidx & tidx:
                disjoint_ok = False
                break
        rows.append({
            "check": "support_query_disjoint",
            "passes": bool(disjoint_ok),
            "detail": "support and query trial indices do not overlap per seed/fold",
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    metrics = validate_metrics(read_csvs(args.metrics_csvs), args.dataset, args.k_per_class, seeds)
    support = read_csvs(args.support_csvs) if args.support_csvs else None
    out_prefix = Path(args.out_prefix or f"results/metrics/{args.dataset}_target_support_m0_k{args.k_per_class}_seeds{seeds[0]}-{seeds[-1]}_core")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics.copy()
    metrics["display_method"] = metrics["method"].map(display_method)
    metrics["protocol_annotation"] = "random_kshot"
    if "session_split_policy" in metrics.columns:
        metrics["session_split_policy"] = metrics["session_split_policy"].fillna(
            "random_kshot_within_loaded_downstream_session_legacy_inferred"
        )
    if "support_query_policy" in metrics.columns:
        metrics["support_query_policy"] = metrics["support_query_policy"].fillna(
            "random_stratified_k_per_class_within_loader_downstream_session_legacy_inferred"
        )
    metrics.to_csv(f"{out_prefix}_fold_metrics.csv", index=False)

    seed_macro = (
        metrics.groupby(["seed", "method", "display_method"], as_index=False)
        .agg(seed_macro_acc=("acc", "mean"), seed_macro_kappa=("kappa", "mean"))
    )
    seed_macro.to_csv(f"{out_prefix}_seed_macro.csv", index=False)

    overall = (
        seed_macro.groupby(["method", "display_method"], as_index=False)
        .agg(
            mean_acc=("seed_macro_acc", "mean"),
            sd_seed_acc=("seed_macro_acc", "std"),
            mean_kappa=("seed_macro_kappa", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    param_meta = (
        metrics.groupby("method", as_index=False)
        .agg(
            target_trainable_params=("target_trainable_params", "mean"),
            target_trainable_frac=("target_trainable_frac", "mean"),
        )
    )
    overall = overall.merge(param_meta, on="method", how="left")
    order = {m: i for i, m in enumerate(METHODS)}
    overall["order"] = overall["method"].map(order)
    overall = overall.sort_values("order").drop(columns="order")
    overall.to_csv(f"{out_prefix}_method_summary.csv", index=False)

    subject_mean = (
        metrics.groupby(["fold", "target_subject", "method", "display_method"], as_index=False)
        .agg(subject_mean_acc=("acc", "mean"), subject_mean_kappa=("kappa", "mean"))
    )
    subject_mean.to_csv(f"{out_prefix}_subject_mean.csv", index=False)

    piv_subject = subject_mean.pivot(index="fold", columns="method", values="subject_mean_acc")
    piv_seed = seed_macro.pivot(index="seed", columns="method", values="seed_macro_acc")
    pair_rows = []
    diff_rows = []
    for a, b, label in PAIRS:
        diffs = piv_subject[a] - piv_subject[b]
        seed_diffs = piv_seed[a] - piv_seed[b]
        mean, lo, hi, paired_t_p, wilcoxon_p = ci95_and_tests(diffs.values)
        pair_rows.append({
            "comparison": label,
            "method_a": display_method(a),
            "method_b": display_method(b),
            "mean_gain_pp": round(mean, 4),
            "ci95_low_pp": round(lo, 4),
            "ci95_high_pp": round(hi, 4),
            "improved_subjects": int((diffs > 0).sum()),
            "nonnegative_subjects": int((diffs >= 0).sum()),
            "n_subjects": int(len(diffs)),
            "seed_wins": int((seed_diffs > 0).sum()),
            "n_seeds": int(len(seed_diffs)),
            "paired_t_p": paired_t_p,
            "wilcoxon_p": wilcoxon_p,
        })
        for fold, diff in diffs.items():
            diff_rows.append({"comparison": label, "fold": int(fold), "diff_pp": float(diff)})
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(f"{out_prefix}_paired_stats.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(f"{out_prefix}_subject_diffs.csv", index=False)

    checks = protocol_checks(metrics, support, seeds, args.k_per_class)
    checks.to_csv(f"{out_prefix}_protocol_checks.csv", index=False)

    ranked = seed_macro.copy()
    ranked["rank"] = ranked.groupby("seed")["seed_macro_acc"].rank(ascending=False, method="min")
    top_by_seed = ranked.loc[ranked.groupby("seed")["seed_macro_acc"].idxmax()].sort_values("seed")
    top_detail = ";".join(f"{int(r.seed)}:{display_method(r.method)}={r.seed_macro_acc:.4f}" for r in top_by_seed.itertuples())
    overall_top = overall.loc[overall["mean_acc"].idxmax(), "method"]
    pair_idx = paired.set_index("comparison")
    lora_head = pair_idx.loc["LoRA_minus_Head"]
    ifnet_head = pair_idx.loc["IFNetFT_minus_Head"]
    ifnet_lora = pair_idx.loc["IFNetFT_minus_LoRA"]
    decision = pd.DataFrame([
        {
            "rule": "delete_lora_from_future_k_if_not_above_head_and_seed_wins_lt2of3",
            "value": f"LoRA-Head mean {lora_head.mean_gain_pp:.4f} pp; seed wins {int(lora_head.seed_wins)}/3",
            "passes": bool((lora_head.mean_gain_pp <= 0.0) and (int(lora_head.seed_wins) < 2)),
            "interpretation": "delete LoRA from K expansion if true",
        },
        {
            "rule": "ifnet_ft_mainline_if_highest_and_subject_wins_vs_head_at_least6of9",
            "value": f"top={display_method(overall_top)}; IFNetFT-Head subject wins {int(ifnet_head.improved_subjects)}/9",
            "passes": bool((overall_top == "IFNet-FT") and (int(ifnet_head.improved_subjects) >= 6)),
            "interpretation": "use IFNet-FT as current K=10 mainline if true",
        },
        {
            "rule": "ifnet_ft_subject_wins_vs_lora_at_least6of9",
            "value": f"IFNetFT-LoRA subject wins {int(ifnet_lora.improved_subjects)}/9",
            "passes": bool(int(ifnet_lora.improved_subjects) >= 6),
            "interpretation": "auxiliary check against LoRA baseline",
        },
        {
            "rule": "top_method_same_across_seeds",
            "value": top_detail,
            "passes": bool(top_by_seed["method"].nunique() == 1),
            "interpretation": "if false, inspect seed/support sensitivity before expanding K",
        },
    ])
    decision.to_csv(f"{out_prefix}_decision.csv", index=False)

    print("Method summary:")
    print(overall.round(4).to_string(index=False))
    print("\nPaired stats:")
    print(paired.round(4).to_string(index=False))
    print("\nProtocol checks:")
    print(checks.to_string(index=False))
    print("\nDecision:")
    print(decision.to_string(index=False))
    print(f"\nWrote {out_prefix}_*.csv")


if __name__ == "__main__":
    main()
