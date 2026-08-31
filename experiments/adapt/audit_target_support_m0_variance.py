import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CORE_METHODS = ["SourceOnly", "MIRepNet-Head", "MIRepNet-LoRA", "IFNet-FT"]
DISPLAY = {"SourceOnly": "MIRepNet-SourceOnly"}
PAIRS = [
    ("MIRepNet-LoRA", "MIRepNet-Head", "LoRA_minus_Head"),
    ("IFNet-FT", "MIRepNet-Head", "IFNet_minus_Head"),
    ("IFNet-FT", "MIRepNet-LoRA", "IFNet_minus_LoRA"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[666, 667, 668])
    p.add_argument("--metrics_csvs", nargs="+", required=True)
    p.add_argument("--support_csvs", nargs="+", required=True)
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


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    out_prefix = Path(args.out_prefix or f"results/metrics/{args.dataset}_target_support_m0_k{args.k_per_class}_seeds{seeds[0]}-{seeds[-1]}_audit")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics = read_csvs(args.metrics_csvs)
    metrics = metrics[
        (metrics["dataset"] == args.dataset)
        & (metrics["support_k_per_class"] == args.k_per_class)
        & (metrics["seed"].isin(seeds))
        & (metrics["method"].isin(CORE_METHODS))
    ].copy()
    if metrics.empty:
        raise ValueError("no matching core M0 metrics")
    metrics["display_method"] = metrics["method"].map(display_method)

    expected = pd.MultiIndex.from_product([seeds, sorted(metrics["fold"].unique()), CORE_METHODS], names=["seed", "fold", "method"])
    have = pd.MultiIndex.from_frame(metrics[["seed", "fold", "method"]])
    missing = expected.difference(have)
    if len(missing):
        raise ValueError("missing seed/fold/method rows: " + str(list(missing)[:20]))

    seed_macro = metrics.groupby(["seed", "method", "display_method"], as_index=False).agg(
        seed_macro_acc=("acc", "mean"),
        seed_macro_kappa=("kappa", "mean"),
    )
    seed_macro["rank"] = seed_macro.groupby("seed")["seed_macro_acc"].rank(ascending=False, method="min").astype(int)
    seed_macro.to_csv(f"{out_prefix}_seed_macro_rank.csv", index=False)

    method_range = seed_macro.groupby(["method", "display_method"], as_index=False).agg(
        mean_acc=("seed_macro_acc", "mean"),
        min_seed_acc=("seed_macro_acc", "min"),
        max_seed_acc=("seed_macro_acc", "max"),
        range_seed_acc=("seed_macro_acc", lambda x: float(np.max(x) - np.min(x))),
        sd_seed_acc=("seed_macro_acc", "std"),
    )
    method_range.to_csv(f"{out_prefix}_method_seed_range.csv", index=False)

    subject_stats = metrics.groupby(["fold", "target_subject", "method", "display_method"], as_index=False).agg(
        mean_acc=("acc", "mean"),
        std_acc=("acc", "std"),
        min_acc=("acc", "min"),
        max_acc=("acc", "max"),
        range_acc=("acc", lambda x: float(np.max(x) - np.min(x))),
    )
    subject_stats.to_csv(f"{out_prefix}_subject_method_seed_stats.csv", index=False)

    piv = metrics.pivot_table(index=["fold", "target_subject", "seed"], columns="method", values="acc")
    diff_rows = []
    for a, b, label in PAIRS:
        diff = (piv[a] - piv[b]).rename(label).reset_index()
        for row in diff.itertuples(index=False):
            diff_rows.append({
                "comparison": label,
                "fold": int(row.fold),
                "target_subject": int(row.target_subject),
                "seed": int(row.seed),
                "diff_pp": float(getattr(row, label)),
            })
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(f"{out_prefix}_pair_diff_by_subject_seed.csv", index=False)

    for label in [p[2] for p in PAIRS]:
        mat = diffs[diffs["comparison"] == label].pivot(index="target_subject", columns="seed", values="diff_pp")
        mat.to_csv(f"{out_prefix}_{label}_matrix.csv")

    benefit = diffs.groupby(["comparison", "fold", "target_subject"], as_index=False).agg(
        positive_seed_count=("diff_pp", lambda x: int((np.asarray(x) > 0).sum())),
        nonnegative_seed_count=("diff_pp", lambda x: int((np.asarray(x) >= 0).sum())),
        mean_diff_pp=("diff_pp", "mean"),
        min_diff_pp=("diff_pp", "min"),
        max_diff_pp=("diff_pp", "max"),
    )
    benefit.to_csv(f"{out_prefix}_benefit_counts_by_subject.csv", index=False)

    ifnet_best_rows = []
    for (fold, target, seed), row in piv.reset_index().set_index(["fold", "target_subject", "seed"]).iterrows():
        best_big = max(float(row["MIRepNet-Head"]), float(row["MIRepNet-LoRA"]))
        ifnet = float(row["IFNet-FT"])
        ifnet_best_rows.append({
            "fold": int(fold),
            "target_subject": int(target),
            "seed": int(seed),
            "ifnet_minus_head": ifnet - float(row["MIRepNet-Head"]),
            "ifnet_minus_lora": ifnet - float(row["MIRepNet-LoRA"]),
            "ifnet_minus_best_big": ifnet - best_big,
        })
    ifnet_best = pd.DataFrame(ifnet_best_rows)
    ifnet_subject = ifnet_best.groupby(["fold", "target_subject"], as_index=False).agg(
        mean_ifnet_minus_best_big=("ifnet_minus_best_big", "mean"),
        min_ifnet_minus_best_big=("ifnet_minus_best_big", "min"),
        negative_seed_count=("ifnet_minus_best_big", lambda x: int((np.asarray(x) < 0).sum())),
    )
    ifnet_subject["pattern"] = np.where(
        ifnet_subject["negative_seed_count"].eq(len(seeds)),
        "persistent_negative_all_seeds",
        np.where(ifnet_subject["negative_seed_count"].gt(0), "seed_specific_negative", "nonnegative_all_seeds"),
    )
    ifnet_subject = ifnet_subject.sort_values(["mean_ifnet_minus_best_big", "min_ifnet_minus_best_big"])
    ifnet_best.to_csv(f"{out_prefix}_ifnet_minus_big_by_subject_seed.csv", index=False)
    ifnet_subject.to_csv(f"{out_prefix}_ifnet_negative_subjects.csv", index=False)

    support = read_csvs(args.support_csvs)
    support = support[
        (support["dataset"] == args.dataset)
        & (support["support_k_per_class"] == args.k_per_class)
        & (support["seed"].isin(seeds))
        & (support["split"] == "support")
    ].copy()
    overlap_rows = []
    for fold, g in support.groupby("fold"):
        by_seed = {int(seed): set(sg["target_trial_index"].astype(int)) for seed, sg in g.groupby("seed")}
        for s1, s2 in combinations(sorted(by_seed), 2):
            a, b = by_seed[s1], by_seed[s2]
            inter = len(a & b)
            union = len(a | b)
            overlap_rows.append({
                "fold": int(fold),
                "target_subject": int(fold),
                "seed_a": s1,
                "seed_b": s2,
                "intersection_n": inter,
                "union_n": union,
                "support_n": len(a),
                "overlap_frac_of_support": inter / len(a) if a else np.nan,
                "jaccard": inter / union if union else np.nan,
            })
            for cls in sorted(g["y"].unique()):
                ca = set(g[(g["seed"] == s1) & (g["y"] == cls)]["target_trial_index"].astype(int))
                cb = set(g[(g["seed"] == s2) & (g["y"] == cls)]["target_trial_index"].astype(int))
                ci, cu = len(ca & cb), len(ca | cb)
                overlap_rows.append({
                    "fold": int(fold),
                    "target_subject": int(fold),
                    "seed_a": s1,
                    "seed_b": s2,
                    "class": int(cls),
                    "intersection_n": ci,
                    "union_n": cu,
                    "support_n": len(ca),
                    "overlap_frac_of_support": ci / len(ca) if ca else np.nan,
                    "jaccard": ci / cu if cu else np.nan,
                })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(f"{out_prefix}_support_overlap.csv", index=False)
    overlap_summary = overlap[overlap.get("class").isna() if "class" in overlap else slice(None)].agg({
        "overlap_frac_of_support": ["mean", "std", "min", "max"],
        "jaccard": ["mean", "std", "min", "max"],
    })
    overlap_summary.to_csv(f"{out_prefix}_support_overlap_summary.csv")

    session_rows = []
    if "session" in support.columns:
        sess = support.groupby(["seed", "fold", "session", "y"], as_index=False).size()
        sess.to_csv(f"{out_prefix}_support_session_class_counts.csv", index=False)
        session_rows.append({"check": "support_session_class_counts", "available": True, "detail": "session column present"})
    else:
        session_rows.append({
            "check": "support_session_class_counts",
            "available": False,
            "detail": "support split has no session column; BNCI2014004 loader currently returns only fixed session_3 rows",
        })
    pd.DataFrame(session_rows).to_csv(f"{out_prefix}_session_note.csv", index=False)

    print("Seed macro/rank:")
    print(seed_macro.sort_values(["seed", "rank"]).round(4).to_string(index=False))
    print("\nMethod seed range:")
    print(method_range.round(4).to_string(index=False))
    print("\nPair benefit counts:")
    print(benefit.groupby("comparison")[["positive_seed_count", "mean_diff_pp", "min_diff_pp", "max_diff_pp"]].mean().round(4).to_string())
    print("\nWorst IFNet-minus-best-big subjects:")
    print(ifnet_subject.head(2).round(4).to_string(index=False))
    print("\nSupport overlap summary:")
    print(overlap_summary.round(4).to_string())
    print("\nSession note:")
    print(pd.DataFrame(session_rows).to_string(index=False))
    print(f"\nWrote {out_prefix}_*.csv")


if __name__ == "__main__":
    main()
