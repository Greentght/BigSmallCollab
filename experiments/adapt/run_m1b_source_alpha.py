"""M1b source-validated scalar fusion for target-support adaptation.

Main line only:
    MIRepNet-Head + IFNet-FT

For each real target subject t, alpha is selected without seeing t's query
labels. Each remaining subject v is treated as a pseudo-target, while source
models are trained on subjects excluding both t and v. The selected alpha_t is
then applied to the already saved M0 query logits for real target t.
"""
import argparse
import copy
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

import config
import data
from models import get_adapter
from scripts.adapt.run_m1a_oracle_alpha import alpha_grid, logit_cols, metric_record, method_frame
from scripts.adapt.run_target_support_m0 import (
    BIG_MODEL,
    SMALL_MODEL,
    adapt_batch_size_for,
    adapt_weight_decay_for,
    count_params,
    finetune_ce,
    kshot_indices_per_class,
    model_cfg,
    prepare_ifnet_fullft,
    prepare_mirepnet_target_method,
    set_seed,
    target_arrays,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--support_draw", type=int, default=0)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--pseudo_subjects", type=int, nargs="+", default=None,
                   help="optional restriction for smoke tests; omit for formal M1b")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--torch_threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    p.add_argument("--base_epochs_mirepnet", type=int, default=None)
    p.add_argument("--base_epochs_ifnet", type=int, default=None)
    p.add_argument("--adapt_epochs", type=int, default=30)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_weight_decay", type=float, default=None)
    p.add_argument("--adapt_batch_size", type=int, default=None)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--trial_preds_csv", default=None)
    p.add_argument("--out_val_grid_csv", default=None)
    p.add_argument("--out_val_curve_csv", default=None)
    p.add_argument("--out_selected_csv", default=None)
    p.add_argument("--out_target_csv", default=None)
    p.add_argument("--out_checks_csv", default=None)
    p.add_argument("--out_summary_csv", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def default_prefix(dataset, k_per_class, seed):
    return Path("results") / "metrics" / f"{dataset}_target_support_m1b_source_alpha_k{k_per_class}_seed{seed}"


def m0_prefix(dataset, k_per_class, seed):
    return Path("results") / "metrics" / f"{dataset}_target_support_m0_k{k_per_class}_seed{seed}"


def device_from_gpu(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def pair_split(dataset, exclude_subjects, num_subjects):
    Xs, ys, subj = [], [], []
    excluded = set(int(s) for s in exclude_subjects)
    for s in range(num_subjects):
        if s in excluded:
            continue
        X, y = data.load_subject_raw(dataset, s)
        Xs.append(X)
        ys.append(y)
        subj.append(np.full(len(y), s, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(subj)


def build_pair_source(model_name, dataset, exclude_pair, seed, num_classes, num_subjects, device, args):
    X_src, y_src, subj_src = pair_split(dataset, exclude_pair, num_subjects)
    cfg = model_cfg(model_name, dataset, X_src, args)
    ad = get_adapter(model_name, device=device, **cfg)
    if model_name == BIG_MODEL:
        X_train = ad.ea_pad_per_subject(X_src, subj_src)
    else:
        X_train = X_src

    offset = 100000 + 997 * int(min(exclude_pair)) + 1291 * int(max(exclude_pair))
    if model_name == SMALL_MODEL:
        offset += 50000
    set_seed(seed + offset)
    model = ad.build(num_classes)
    model = finetune_ce(
        ad,
        model,
        X_train,
        y_src,
        num_classes,
        epochs=int(cfg.get("epochs", 50)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        batch_size=int(cfg.get("batch_size", 32)),
        seed=seed + offset,
        train_mode="default",
    )
    meta = {
        "source_train_subjects": ";".join(str(int(s)) for s in sorted(set(subj_src.tolist()))),
        "excluded_subjects": ";".join(str(int(s)) for s in sorted(exclude_pair)),
        "n_source_train": int(len(y_src)),
        "source_epochs": int(cfg.get("epochs", 50)),
        "source_lr": float(cfg.get("lr", 1e-3)),
        "source_weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "source_batch_size": int(cfg.get("batch_size", 32)),
        "total_params": int(count_params(model)[1]),
    }
    return ad, model, cfg, meta


def split_seed_for_subject(seed, subject, support_draw):
    return int(seed + 1009 * int(subject) + 100000 * int(support_draw))


def pseudo_target_split(dataset, subject, num_classes, k_per_class, seed, support_draw):
    X, y = data.load_subject_raw(dataset, int(subject))
    split_seed = split_seed_for_subject(seed, subject, support_draw)
    support_idx, query_idx = kshot_indices_per_class(y, num_classes, k_per_class, split_seed)
    return X, y, support_idx, query_idx, split_seed


def adapt_head_logits(big_ad, big_base, X_target, y_target, support_idx, query_idx,
                      pseudo_subject, num_classes, seed, args):
    X_sup, X_query = target_arrays(BIG_MODEL, big_ad, X_target, support_idx, query_idx, pseudo_subject)
    model = copy.deepcopy(big_base)
    params_meta = prepare_mirepnet_target_method(model, "MIRepNet-Head", args)
    wd = adapt_weight_decay_for(big_ad.cfg, args)
    bs = adapt_batch_size_for(big_ad.cfg, args)
    train_seed = seed + 200000 + 1009 * int(pseudo_subject)
    set_seed(train_seed)
    model = finetune_ce(
        big_ad,
        model,
        X_sup,
        y_target[support_idx],
        num_classes,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=wd,
        batch_size=bs,
        seed=train_seed,
        train_mode=params_meta["train_mode"],
    )
    _feats, logits = big_ad.infer(model, X_query)
    del model
    return logits


def adapt_ifnet_logits(small_ad, small_base, X_target, y_target, support_idx, query_idx,
                       pseudo_subject, num_classes, seed, args):
    X_sup, X_query = target_arrays(SMALL_MODEL, small_ad, X_target, support_idx, query_idx, pseudo_subject)
    model = copy.deepcopy(small_base)
    params_meta = prepare_ifnet_fullft(model)
    wd = adapt_weight_decay_for(small_ad.cfg, args)
    bs = adapt_batch_size_for(small_ad.cfg, args)
    train_seed = seed + 300000 + 1009 * int(pseudo_subject)
    set_seed(train_seed)
    model = finetune_ce(
        small_ad,
        model,
        X_sup,
        y_target[support_idx],
        num_classes,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        weight_decay=wd,
        batch_size=bs,
        seed=train_seed,
        train_mode=params_meta["train_mode"],
    )
    _feats, logits = small_ad.infer(model, X_query)
    del model
    return logits


def sweep_alpha(y, z_head, z_ifnet, alphas):
    rows = []
    for alpha in alphas:
        logits = (1.0 - alpha) * z_head + alpha * z_ifnet
        rows.append({"alpha": float(alpha), **metric_record(y, logits)})
    return rows


def target_method_logits(trial_preds, method, fold):
    df = method_frame(trial_preds, method)
    g = df[df["fold"].eq(fold)].sort_values("target_trial_index").reset_index(drop=True)
    if g.empty:
        raise ValueError(f"missing target logits for method={method} fold={fold}")
    cols = logit_cols(g)
    y = g["y"].to_numpy(dtype=np.int64)
    logits = g[cols].to_numpy(dtype=np.float64)
    return y, logits, g["target_trial_index"].to_numpy(dtype=np.int64)


def select_alpha(curve):
    best = curve.sort_values(["mean_val_acc", "alpha"], ascending=[False, False]).iloc[0]
    return float(best["alpha"]), float(best["mean_val_acc"])


def write_csv(path, rows_or_df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False)
    return df


def main():
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    dataset = args.dataset
    dcfg = config.load_dataset_config(dataset)
    num_classes = int(dcfg["num_classes"])
    num_subjects = int(dcfg["num_subjects"])
    folds = args.folds if args.folds is not None else list(range(num_subjects))
    pseudo_allowed = set(args.pseudo_subjects) if args.pseudo_subjects is not None else set(range(num_subjects))
    device = device_from_gpu(args.gpu)
    alphas = alpha_grid(args.alpha_step)

    prefix = default_prefix(dataset, args.k_per_class, args.seed)
    m0 = m0_prefix(dataset, args.k_per_class, args.seed)
    trial_preds_csv = Path(args.trial_preds_csv) if args.trial_preds_csv else Path(f"{m0}_trial_preds.csv")
    out_val_grid = Path(args.out_val_grid_csv) if args.out_val_grid_csv else Path(f"{prefix}_val_grid.csv")
    out_val_curve = Path(args.out_val_curve_csv) if args.out_val_curve_csv else Path(f"{prefix}_val_curve.csv")
    out_selected = Path(args.out_selected_csv) if args.out_selected_csv else Path(f"{prefix}_selected_alpha.csv")
    out_target = Path(args.out_target_csv) if args.out_target_csv else Path(f"{prefix}_target_results.csv")
    out_checks = Path(args.out_checks_csv) if args.out_checks_csv else Path(f"{prefix}_checks.csv")
    out_summary = Path(args.out_summary_csv) if args.out_summary_csv else Path(f"{prefix}_summary.csv")

    trial_preds = pd.read_csv(trial_preds_csv)
    trial_preds = trial_preds[(trial_preds["dataset"].eq(dataset)) & (trial_preds["seed"].eq(args.seed))].copy()

    val_rows = []
    check_rows = []
    requested_folds = set(int(f) for f in folds)
    subjects = list(range(num_subjects))

    print(
        f"[cfg] dataset={dataset} seed={args.seed} K/class={args.k_per_class} "
        f"folds={folds} adapt_epochs={args.adapt_epochs} alpha_step={args.alpha_step} device={device}",
        flush=True,
    )
    print("[cfg] main_line=MIRepNet-Head+IFNet-FT; alpha tie-break=max alpha toward IFNet", flush=True)

    for a, b in itertools.combinations(subjects, 2):
        orientations = []
        if a in requested_folds and b in pseudo_allowed:
            orientations.append((a, b))
        if b in requested_folds and a in pseudo_allowed:
            orientations.append((b, a))
        if not orientations:
            continue

        exclude_pair = tuple(sorted((a, b)))
        print(f"[pair {exclude_pair}] train source bases excluding {exclude_pair}", flush=True)
        big_ad, big_base, _big_cfg, big_meta = build_pair_source(
            BIG_MODEL, dataset, exclude_pair, args.seed, num_classes, num_subjects, device, args
        )
        small_ad, small_base, _small_cfg, small_meta = build_pair_source(
            SMALL_MODEL, dataset, exclude_pair, args.seed, num_classes, num_subjects, device, args
        )

        for target_t, pseudo_v in orientations:
            X_v, y_v, support_idx, query_idx, split_seed = pseudo_target_split(
                dataset, pseudo_v, num_classes, args.k_per_class, args.seed, args.support_draw
            )
            y_query = y_v[query_idx]
            support_counts = np.bincount(y_v[support_idx], minlength=num_classes)
            support_disjoint = len(set(support_idx.tolist()) & set(query_idx.tolist())) == 0
            train_subjects = set(int(x) for x in big_meta["source_train_subjects"].split(";") if x != "")
            excludes_ok = int(target_t) not in train_subjects and int(pseudo_v) not in train_subjects
            counts_ok = bool(np.all(support_counts == args.k_per_class))
            check_rows.append({
                "dataset": dataset,
                "seed": int(args.seed),
                "target_t": int(target_t),
                "pseudo_v": int(pseudo_v),
                "excluded_pair": ";".join(str(x) for x in exclude_pair),
                "source_train_subjects": big_meta["source_train_subjects"],
                "t_and_v_excluded": bool(excludes_ok),
                "support_query_disjoint": bool(support_disjoint),
                "support_counts": ";".join(str(int(x)) for x in support_counts),
                "support_counts_ok": counts_ok,
                "support_n": int(len(support_idx)),
                "query_n": int(len(query_idx)),
                "split_seed": int(split_seed),
            })
            if not excludes_ok or not support_disjoint or not counts_ok:
                raise ValueError(f"protocol check failed for target={target_t} pseudo={pseudo_v}")

            z_head = adapt_head_logits(
                big_ad, big_base, X_v, y_v, support_idx, query_idx,
                pseudo_v, num_classes, args.seed, args
            )
            z_ifnet = adapt_ifnet_logits(
                small_ad, small_base, X_v, y_v, support_idx, query_idx,
                pseudo_v, num_classes, args.seed, args
            )
            for row in sweep_alpha(y_query, z_head, z_ifnet, alphas):
                val_rows.append({
                    "dataset": dataset,
                    "seed": int(args.seed),
                    "target_t": int(target_t),
                    "pseudo_v": int(pseudo_v),
                    "excluded_pair": ";".join(str(x) for x in exclude_pair),
                    "alpha": row["alpha"],
                    "acc": row["acc"],
                    "kappa": row["kappa"],
                    "macro_f1": row["macro_f1"],
                    "n_query": int(len(y_query)),
                    "validity": "source_validation_no_target_query_labels",
                })
            print(
                f"[val] target={target_t} pseudo={pseudo_v} "
                f"head={sweep_alpha(y_query, z_head, z_ifnet, np.array([0.0]))[0]['acc']:.2f} "
                f"ifnet={sweep_alpha(y_query, z_head, z_ifnet, np.array([1.0]))[0]['acc']:.2f}",
                flush=True,
            )

        del big_base, small_base
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        write_csv(out_val_grid, val_rows)
        write_csv(out_checks, check_rows)

    val_grid = pd.DataFrame(val_rows)
    if val_grid.empty:
        raise ValueError("no validation rows were generated")

    curve = (
        val_grid.groupby(["target_t", "alpha"], as_index=False)
        .agg(mean_val_acc=("acc", "mean"), mean_val_kappa=("kappa", "mean"),
             n_pseudo=("pseudo_v", "nunique"))
    )
    selected_rows = []
    for target_t, g in curve.groupby("target_t", sort=True):
        alpha, val_acc = select_alpha(g)
        selected_rows.append({
            "dataset": dataset,
            "seed": int(args.seed),
            "target_t": int(target_t),
            "selected_alpha": alpha,
            "source_val_acc": val_acc,
            "n_pseudo": int(g["n_pseudo"].max()),
            "tie_break": "max_alpha_toward_ifnet",
        })
    selected = pd.DataFrame(selected_rows)

    target_rows = []
    for r in selected.itertuples():
        t = int(r.target_t)
        alpha = float(r.selected_alpha)
        y_head, z_head, idx_head = target_method_logits(trial_preds, "MIRepNet-Head", t)
        y_ifnet, z_ifnet, idx_ifnet = target_method_logits(trial_preds, "IFNet-FT", t)
        if not np.array_equal(y_head, y_ifnet) or not np.array_equal(idx_head, idx_ifnet):
            raise ValueError(f"M0 target logits are not aligned for fold {t}")
        logits = (1.0 - alpha) * z_head + alpha * z_ifnet
        fusion = metric_record(y_head, logits)
        head = metric_record(y_head, z_head)
        ifnet = metric_record(y_head, z_ifnet)
        alpha0 = metric_record(y_head, z_head)["acc"]
        alpha1 = metric_record(y_head, z_ifnet)["acc"]
        target_rows.append({
            "dataset": dataset,
            "seed": int(args.seed),
            "target_t": t,
            "selected_alpha": alpha,
            "source_val_acc": float(r.source_val_acc),
            "n_query": int(len(y_head)),
            "head_acc": head["acc"],
            "ifnet_ft_acc": ifnet["acc"],
            "fusion_acc": fusion["acc"],
            "gain_vs_ifnet_ft": fusion["acc"] - ifnet["acc"],
            "gain_vs_head": fusion["acc"] - head["acc"],
            "kappa": fusion["kappa"],
            "macro_f1": fusion["macro_f1"],
            "alpha0_reproduces_head": bool(abs(alpha0 - head["acc"]) < 1e-9),
            "alpha1_reproduces_ifnet": bool(abs(alpha1 - ifnet["acc"]) < 1e-9),
            "validity": "source_validated_alpha_no_target_query_labels",
        })
    target_df = pd.DataFrame(target_rows)

    mean_gain = float(target_df["gain_vs_ifnet_ft"].mean())
    wins = int((target_df["gain_vs_ifnet_ft"] > 0).sum())
    summary = pd.DataFrame([{
        "dataset": dataset,
        "seed": int(args.seed),
        "method": "M1b_SourceAlpha_Head_IFNetFT",
        "n_folds": int(target_df["target_t"].nunique()),
        "mean_head_acc": round(float(target_df["head_acc"].mean()), 4),
        "mean_ifnet_ft_acc": round(float(target_df["ifnet_ft_acc"].mean()), 4),
        "mean_fusion_acc": round(float(target_df["fusion_acc"].mean()), 4),
        "mean_gain_vs_ifnet_ft": round(mean_gain, 4),
        "wins_vs_ifnet_ft": wins,
        "passes_seed666_stop_rule": bool(mean_gain >= 1.0 and wins >= 6),
        "stop_rule": "continue_to_667_668_only_if_gain>=1pp_and_wins>=6of9",
        "selected_alphas": ";".join(
            f"{int(x.target_t)}:{x.selected_alpha:.2f}" for x in target_df.sort_values("target_t").itertuples()
        ),
    }])

    write_csv(out_val_grid, val_grid)
    write_csv(out_val_curve, curve)
    write_csv(out_selected, selected)
    write_csv(out_target, target_df)
    write_csv(out_checks, check_rows)
    write_csv(out_summary, summary)

    print("\nSelected alpha:", flush=True)
    print(selected.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\nTarget results:", flush=True)
    print(target_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\nSummary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nWrote {out_val_grid}", flush=True)
    print(f"Wrote {out_val_curve}", flush=True)
    print(f"Wrote {out_selected}", flush=True)
    print(f"Wrote {out_target}", flush=True)
    print(f"Wrote {out_checks}", flush=True)
    print(f"Wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()
