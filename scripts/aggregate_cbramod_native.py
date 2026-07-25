"""Aggregate CBraMod native adaptation CSVs into a compact five-task table."""
import argparse
import glob
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="native70")
    parser.add_argument("--input_dir", default="results/cbramod_native")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.input_dir, f"*_{args.preset}_train*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No CSVs matched {pattern}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if not df.empty:
            df["file"] = path
            frames.append(df)
    if not frames:
        raise SystemExit("Matched CSVs were empty")

    raw = pd.concat(frames, ignore_index=True)
    summary = (
        raw.groupby(["dataset", "preset", "train_percentage", "session"], dropna=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            bac_mean=("bac", "mean"),
            kappa_mean=("kappa", "mean"),
            n_rows=("acc", "size"),
            n_subjects=("subject", "nunique"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset", "session"])
    )
    for col in ["acc_mean", "acc_std", "bac_mean", "kappa_mean"]:
        summary[col] = summary[col].round(4)

    out = args.out or os.path.join(args.input_dir, f"summary_{args.preset}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
