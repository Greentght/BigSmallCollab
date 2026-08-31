"""M1a zero-training scalar-fusion upper bound.

This script reads M0 per-trial logits and sweeps a single scalar alpha:

    z = (1 - alpha) * z_big + alpha * z_ifnet

Alpha is selected with the same target query labels being evaluated, so the
selected result is an oracle upper bound only. It must not be reported as a
deployable method.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from collab import artifacts


PAIR_SPECS = [
    ("Head+IFNetFT", "MIRepNet-Head", "IFNet-FT"),
    ("LoRA+IFNetFT", "MIRepNet-LoRA", "IFNet-FT"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--trial_preds_csv", default=None)
    p.add_argument("--support_split_csv", default=None)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--ifnet_source_artifact", default="ifnet_loso")
    p.add_argument("--out_grid_csv", default=None)
    p.add_argument("--out_best_csv", default=None)
    p.add_argument("--out_summary_csv", default=None)
    p.add_argument("--out_ifnet_source_csv", default=None)
    p.add_argument("--out_protocol_csv", default=None)
    return p.parse_args()


def default_prefix(dataset, k_per_class, seed):
    return Path("results") / "metrics" / f"{dataset}_target_support_m1a_oracle_alpha_k{k_per_class}_seed{seed}"


def m0_prefix(dataset, k_per_class, seed):
    return Path("results") / "metrics" / f"{dataset}_target_support_m0_k{k_per_class}_seed{seed}"


def alpha_grid(step):
    n = int(round(1.0 / step))
    vals = np.linspace(0.0, 1.0, n + 1)
    return np.round(vals, 10)


def logit_cols(df):
    cols = [c for c in df.columns if c.startswith("logit_c")]
    return sorted(cols, key=lambda x: int(x.replace("logit_c", "")))


def method_frame(df, method):
    out = df[df["method"].eq(method)].copy()
    if out.empty:
        raise ValueError(f"missing method in trial preds: {method}")
    return out.sort_values(["fold", "target_trial_index"]).reset_index(drop=True)


def metric_record(y, logits):
    pred = logits.argmax(1)
    labels = list(range(logits.shape[1]))
    return {
        "acc": float((pred == y).mean() * 100.0),
        "kappa": float(cohen_kappa_score(y, pred, labels=labels)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro")),
    }


def assert_aligned(a, b, name_a, name_b):
    cols = ["dataset", "seed", "fold", "target_trial_index", "y"]
    if not a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)):
        raise ValueError(f"{name_a} and {name_b} are not row-aligned")


def sweep_pair(df, pair_name, big_method, small_method, alphas):
    big = method_frame(df, big_method)
    small = method_frame(df, small_method)
    assert_aligned(big, small, big_method, small_method)

    cols = logit_cols(big)
    rows = []
    best_rows = []
    for fold, bfold in big.groupby("fold", sort=True):
        sfold = small[small["fold"].eq(fold)].sort_values("target_trial_index")
        assert_aligned(
            bfold.sort_values("target_trial_index").reset_index(drop=True),
            sfold.reset_index(drop=True),
            big_method,
            small_method,
        )
        b = bfold.sort_values("target_trial_index").reset_index(drop=True)
        s = sfold.reset_index(drop=True)
        y = b["y"].to_numpy(dtype=np.int64)
        z_big = b[cols].to_numpy(dtype=np.float64)
        z_small = s[cols].to_numpy(dtype=np.float64)

        fold_rows = []
        for alpha in alphas:
            logits = (1.0 - alpha) * z_big + alpha * z_small
            m = metric_record(y, logits)
            row = {
                "pair": pair_name,
                "big_method": big_method,
                "small_method": small_method,
                "fold": int(fold),
                "alpha": float(alpha),
                "uses_query_labels_for_alpha": True,
                "validity": "oracle_upper_bound",
                "n_query": int(len(y)),
                **m,
            }
            rows.append(row)
            fold_rows.append(row)

        best = max(
            fold_rows,
            key=lambda r: (r["acc"], -abs(r["alpha"] - 0.5), -r["alpha"]),
        )
        best_rows.append({**best, "selection": "best_alpha_per_target_query"})

    return rows, best_rows


def load_ifnet_source_query(dataset, seed, support_split_csv, artifact_name):
    split = pd.read_csv(support_split_csv)
    split = split[(split["dataset"].eq(dataset)) & (split["seed"].eq(seed)) & (split["split"].eq("test"))]
    rows = []
    for fold, g in split.groupby("fold", sort=True):
        d = artifacts.load(dataset, artifact_name, int(fold), int(seed), "test")
        logits = d["logits"]
        y_all = d["y"]
        idx = g.sort_values("target_trial_index")["target_trial_index"].to_numpy(dtype=np.int64)
        y = g.sort_values("target_trial_index")["y"].to_numpy(dtype=np.int64)
        if idx.max(initial=-1) >= len(y_all):
            raise ValueError(f"query index exceeds IFNet artifact length for fold {fold}")
        if not np.array_equal(y, y_all[idx]):
            raise ValueError(f"IFNet source artifact labels do not align for fold {fold}")
        m = metric_record(y, logits[idx])
        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "fold": int(fold),
            "method": "IFNet-SourceOnly",
            "model": "ifnet",
            "source_artifact": artifact_name,
            "support_split_source": "same_query_as_m0",
            "n_query": int(len(y)),
            **m,
        })
    return pd.DataFrame(rows)


def write_csv(path, rows_or_df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False)
    return df


def main():
    args = parse_args()
    out_prefix = default_prefix(args.dataset, args.k_per_class, args.seed)
    m0 = m0_prefix(args.dataset, args.k_per_class, args.seed)
    trial_preds_csv = Path(args.trial_preds_csv) if args.trial_preds_csv else Path(f"{m0}_trial_preds.csv")
    support_split_csv = Path(args.support_split_csv) if args.support_split_csv else Path(f"{m0}_support_split.csv")
    out_grid_csv = Path(args.out_grid_csv) if args.out_grid_csv else Path(f"{out_prefix}_grid.csv")
    out_best_csv = Path(args.out_best_csv) if args.out_best_csv else Path(f"{out_prefix}_best.csv")
    out_summary_csv = Path(args.out_summary_csv) if args.out_summary_csv else Path(f"{out_prefix}_summary.csv")
    out_ifnet_source_csv = Path(args.out_ifnet_source_csv) if args.out_ifnet_source_csv else Path(f"{m0}_ifnet_sourceonly_same_query.csv")
    out_protocol_csv = Path(args.out_protocol_csv) if args.out_protocol_csv else Path(f"{m0}_protocol_note.csv")

    df = pd.read_csv(trial_preds_csv)
    df = df[(df["dataset"].eq(args.dataset)) & (df["seed"].eq(args.seed))].copy()
    alphas = alpha_grid(args.alpha_step)

    grid_rows = []
    best_rows = []
    for pair_name, big_method, small_method in PAIR_SPECS:
        rows, best = sweep_pair(df, pair_name, big_method, small_method, alphas)
        grid_rows.extend(rows)
        best_rows.extend(best)

    grid = write_csv(out_grid_csv, grid_rows)
    best = write_csv(out_best_csv, best_rows)
    ifnet_source = load_ifnet_source_query(
        args.dataset,
        args.seed,
        support_split_csv,
        args.ifnet_source_artifact,
    )
    write_csv(out_ifnet_source_csv, ifnet_source)

    base_by_fold = (
        df.groupby(["method", "fold"])["correct"]
        .mean()
        .mul(100.0)
        .unstack("method")
    )
    base = base_by_fold.mean().to_dict()
    ifnet_ft = float(base["IFNet-FT"])
    summary_rows = []
    for pair, g in best.groupby("pair", sort=False):
        big_method = g["big_method"].iloc[0]
        paired = g.set_index("fold").sort_index()
        ifnet_fold = base_by_fold.loc[paired.index, "IFNet-FT"]
        big_fold = base_by_fold.loc[paired.index, big_method]
        best_single_fold = np.maximum(big_fold.to_numpy(), ifnet_fold.to_numpy())
        oracle_mean = float(g["acc"].mean())
        best_single = float(best_single_fold.mean())
        summary_rows.append({
            "pair": pair,
            "validity": "oracle_upper_bound_uses_target_query_labels",
            "n_folds": int(g["fold"].nunique()),
            "big_method": big_method,
            "small_method": "IFNet-FT",
            "big_acc": round(float(base[big_method]), 4),
            "ifnet_ft_acc": round(ifnet_ft, 4),
            "best_single_acc": round(best_single, 4),
            "oracle_alpha_acc": round(oracle_mean, 4),
            "oracle_minus_ifnet_ft": round(oracle_mean - ifnet_ft, 4),
            "oracle_minus_best_single": round(oracle_mean - best_single, 4),
            "wins_vs_ifnet_ft": int((paired["acc"].to_numpy() > ifnet_fold.to_numpy()).sum()),
            "wins_vs_best_single": int((paired["acc"].to_numpy() > best_single_fold).sum()),
            "mean_alpha": round(float(g["alpha"].mean()), 4),
            "alphas_by_fold": ";".join(
                f"{int(r.fold)}:{r.alpha:.2f}" for r in g.sort_values("fold").itertuples()
            ),
            "m1b_needed_by_1pp_rule": bool(oracle_mean - ifnet_ft >= 1.0),
        })
    summary = write_csv(out_summary_csv, summary_rows)

    protocol = pd.DataFrame([{
        "dataset": args.dataset,
        "seed": int(args.seed),
        "k_per_class": int(args.k_per_class),
        "support_query_policy": "random_stratified_k_per_class_within_loader_downstream_session",
        "session_note": "BNCI2014004 loader uses data_mode=session3; support/query here are random K-shot rows within that loaded downstream session, not session-aware split.",
        "m1a_validity": "OracleAlpha uses target query labels for alpha selection; upper-bound diagnostic only.",
        "ifnet_sourceonly_note": f"IFNet-SourceOnly computed from {args.ifnet_source_artifact} artifacts on the same M0 query rows.",
    }])
    write_csv(out_protocol_csv, protocol)

    print("OracleAlpha best per fold:", flush=True)
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\nSummary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nIFNet-SourceOnly same query:", flush=True)
    print(ifnet_source.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\nWrote {out_grid_csv}", flush=True)
    print(f"Wrote {out_best_csv}", flush=True)
    print(f"Wrote {out_summary_csv}", flush=True)
    print(f"Wrote {out_ifnet_source_csv}", flush=True)
    print(f"Wrote {out_protocol_csv}", flush=True)


if __name__ == "__main__":
    main()
