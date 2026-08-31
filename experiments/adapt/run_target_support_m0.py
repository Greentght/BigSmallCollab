"""M0 target-support adaptation baselines.

Protocol:
  * Outer LOSO target subject is never used in source training.
  * For the held-out target subject, sample K labeled support trials per class.
  * All methods share the exact same support/test indices.
  * No early stopping or hyperparameter search uses target test trials.

Default M0 cell:
    conda run --no-capture-output -n mirepnet python \
        experiments/adapt/run_target_support_m0.py \
        --dataset BNCI2014004 --k_per_class 10 --seeds 666 --gpu 0

Methods:
  SourceOnly         : MIRepNet LOSO source model, no target update
  IFNet-FT           : IFNet LOSO source model, full target fine-tune
  MIRepNet-Head      : MIRepNet source model, target update clshead only
  MIRepNet-LastBlock : MIRepNet source model, target update last transformer block + clshead
  MIRepNet-FullFT    : MIRepNet source model, full target fine-tune
  MIRepNet-LoRA      : MIRepNet source model, LoRA on attention q/v + clshead
"""
import argparse
import copy
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

import config
import data
from models import get_adapter
from models.mirepnet.lora import apply_lora

BIG_MODEL = "mirepnet"
SMALL_MODEL = "ifnet"
METHODS = [
    "SourceOnly",
    "IFNet-SourceOnly",
    "IFNet-FT",
    "MIRepNet-Head",
    "MIRepNet-LastBlock",
    "MIRepNet-FullFT",
    "MIRepNet-LoRA",
]
METHOD_SEED_OFFSET = {
    "SourceOnly": 0,
    "IFNet-SourceOnly": 7,
    "IFNet-FT": 11,
    "MIRepNet-Head": 23,
    "MIRepNet-LastBlock": 37,
    "MIRepNet-FullFT": 41,
    "MIRepNet-LoRA": 53,
}
METHOD_ALIASES = {
    "MIRepNet-SourceOnly": "SourceOnly",
    "all": "all",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BNCI2014004")
    p.add_argument("--k_per_class", type=int, default=10)
    p.add_argument("--support_draw", type=int, default=0)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[666])
    p.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help=(
            "Subset of methods to run. Use SourceOnly or MIRepNet-SourceOnly "
            "for the MIRepNet no-adaptation baseline."
        ),
    )
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
    p.add_argument("--out_csv", default=None)
    p.add_argument("--pred_csv", default=None)
    p.add_argument("--support_csv", default=None)
    p.add_argument("--complement_csv", default=None)
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_methods(methods):
    if methods is None:
        return list(METHODS)
    out = []
    for method in methods:
        method = METHOD_ALIASES.get(method, method)
        if method == "all":
            return list(METHODS)
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choices are {METHODS}")
        if method not in out:
            out.append(method)
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_tag(seeds):
    vals = [int(s) for s in seeds]
    if len(vals) == 1:
        return f"seed{vals[0]}"
    if vals == list(range(vals[0], vals[-1] + 1)):
        return f"seeds{vals[0]}-{vals[-1]}"
    return "seeds" + "-".join(str(s) for s in vals)


def default_paths(dataset, k_per_class, seeds):
    tag = f"{dataset}_target_support_m0_k{k_per_class}_{seed_tag(seeds)}"
    base = Path("results") / "metrics"
    return {
        "out_csv": base / f"{tag}.csv",
        "pred_csv": base / f"{tag}_trial_preds.csv",
        "support_csv": base / f"{tag}_support_split.csv",
        "complement_csv": base / f"{tag}_lora_ifnet_complement.csv",
        "summary_csv": base / f"{tag}_summary.csv",
    }


def device_from_gpu(gpu):
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def prepare_mirepnet_target_method(model, method, args):
    """Configure target-trainable parameters for one MIRepNet adaptation method."""
    model_device = next(model.parameters()).device
    n_lora_layers = 0
    if method == "MIRepNet-FullFT":
        set_requires_grad(model, True)
        target_update = "all"
        train_mode = "default"
    elif method == "MIRepNet-Head":
        set_requires_grad(model, False)
        for p in model.clshead.parameters():
            p.requires_grad = True
        target_update = "clshead"
        train_mode = "mirepnet_head"
    elif method == "MIRepNet-LastBlock":
        set_requires_grad(model, False)
        for p in model.transformer[-1].parameters():
            p.requires_grad = True
        for p in model.clshead.parameters():
            p.requires_grad = True
        target_update = "transformer_last_block+clshead"
        train_mode = "mirepnet_lastblock"
    elif method == "MIRepNet-LoRA":
        set_requires_grad(model, False)
        n_lora_layers = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            targets=args.lora_targets,
            dropout=args.lora_dropout,
        )
        for p in model.clshead.parameters():
            p.requires_grad = True
        # LoRALinear creates fresh parameters on CPU; move the whole wrapped model
        # back to the source model device before training.
        model.to(model_device)
        target_update = f"lora_{args.lora_targets}+clshead"
        train_mode = "mirepnet_lora"
    else:
        raise ValueError(f"not a MIRepNet target method: {method}")

    n_trainable, n_total = count_params(model)
    return {
        "target_update": target_update,
        "train_mode": train_mode,
        "n_lora_layers": n_lora_layers,
        "target_trainable_params": n_trainable,
        "total_params": n_total,
        "target_trainable_frac": n_trainable / n_total if n_total else 0.0,
    }


def prepare_ifnet_fullft(model):
    set_requires_grad(model, True)
    n_trainable, n_total = count_params(model)
    return {
        "target_update": "all",
        "train_mode": "default",
        "n_lora_layers": 0,
        "target_trainable_params": n_trainable,
        "total_params": n_total,
        "target_trainable_frac": n_trainable / n_total if n_total else 0.0,
    }


def set_train_mode(model, train_mode):
    """Keep frozen MIRepNet parts deterministic during parameter-efficient updates."""
    if train_mode == "default":
        model.train()
    elif train_mode == "mirepnet_head":
        model.eval()
        model.clshead.train()
    elif train_mode == "mirepnet_lastblock":
        model.eval()
        model.transformer[-1].train()
        model.clshead.train()
    elif train_mode == "mirepnet_lora":
        model.train()
        model.embedding.eval()
    else:
        raise ValueError(f"unknown train_mode: {train_mode}")


def finetune_ce(ad, model, X_tr, y_tr, num_classes, epochs, lr, weight_decay,
                batch_size, seed, train_mode="default"):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return model

    Xp = ad.preprocess(X_tr)
    y = torch.as_tensor(y_tr, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(Xp, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        set_train_mode(model, train_mode)
        for xb, yb in loader:
            xb = xb.to(ad.device)
            yb = yb.to(ad.device)
            _, logits = ad.forward(model, xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def model_cfg(model_name, dataset, X_src, args):
    cfg = dict(config.load_model_config(model_name))
    cfg.update(
        in_channels=X_src.shape[1],
        samples=X_src.shape[2],
        dataset_name=dataset,
    )
    if model_name == BIG_MODEL:
        cfg["skip_preprocess"] = True
        if args.base_epochs_mirepnet is not None:
            cfg["epochs"] = args.base_epochs_mirepnet
    if model_name == SMALL_MODEL and args.base_epochs_ifnet is not None:
        cfg["epochs"] = args.base_epochs_ifnet
    return cfg


def build_source_model(model_name, dataset, fold, seed, num_classes, device, args):
    X_src, y_src, subj_src, X_target, y_target = data.loso_split(dataset, fold)
    cfg = model_cfg(model_name, dataset, X_src, args)
    ad = get_adapter(model_name, device=device, **cfg)

    if model_name == BIG_MODEL:
        X_train = ad.ea_pad_per_subject(X_src, subj_src)
    else:
        X_train = X_src

    set_seed(seed)
    model = ad.build(num_classes)
    n_trainable, n_total = count_params(model)
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
        seed=seed,
        train_mode="default",
    )
    meta = {
        "source_trainable_params": n_trainable,
        "source_total_params": n_total,
        "source_epochs": int(cfg.get("epochs", 50)),
        "source_lr": float(cfg.get("lr", 1e-3)),
        "source_weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "source_batch_size": int(cfg.get("batch_size", 32)),
        "n_source_train": int(len(y_src)),
    }
    return ad, model, X_target, y_target, meta


def kshot_indices_per_class(y, num_classes, k_per_class, seed):
    rng = np.random.default_rng(seed)
    support = []
    for cls in range(num_classes):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) < k_per_class:
            raise ValueError(
                f"class {cls} has only {len(cls_idx)} trials, "
                f"cannot sample K={k_per_class}"
            )
        support.extend(rng.choice(cls_idx, size=k_per_class, replace=False).tolist())
    support = np.array(sorted(support), dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    mask[support] = False
    test = np.flatnonzero(mask).astype(np.int64)
    return support, test


def target_arrays(model_name, ad, X_target, support_idx, test_idx, fold):
    if model_name == BIG_MODEL:
        X_sup = ad.ea_pad_per_subject(
            X_target[support_idx], np.full(len(support_idx), fold, dtype=np.int64)
        )
        X_te = ad.ea_pad_per_subject(
            X_target[test_idx], np.full(len(test_idx), fold, dtype=np.int64)
        )
    else:
        X_sup = X_target[support_idx]
        X_te = X_target[test_idx]
    return X_sup, X_te


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def eval_logits(logits, y, num_classes):
    pred = logits.argmax(1)
    labels = list(range(num_classes))
    return {
        "acc": accuracy_score(y, pred) * 100.0,
        "bac": balanced_accuracy_score(y, pred),
        "kappa": cohen_kappa_score(y, pred, labels=labels),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro"),
    }


def infer_logits(ad, model, X):
    _feats, logits = ad.infer(model, X)
    return logits


def metric_row(dataset, seed, fold, method, model_name, target_update, params_meta,
               source_meta, k_per_class, support_draw, split_seed, y_test, logits):
    m = eval_logits(logits, y_test, int(logits.shape[1]))
    return {
        "dataset": dataset,
        "seed": int(seed),
        "fold": int(fold),
        "target_subject": int(fold),
        "method": method,
        "model": model_name,
        "target_update": target_update,
        "support_k_per_class": int(k_per_class),
        "support_n": int(k_per_class * logits.shape[1]),
        "support_draw": int(support_draw),
        "split_seed": int(split_seed),
        "test_n": int(len(y_test)),
        "support_query_policy": "random_stratified_k_per_class_within_loader_downstream_session",
        "session_split_policy": "random_kshot_within_loaded_downstream_session",
        "acc": round(m["acc"], 4),
        "bac": round(m["bac"], 6),
        "kappa": round(m["kappa"], 6),
        "macro_f1": round(m["macro_f1"], 6),
        "target_trainable_params": int(params_meta["target_trainable_params"]),
        "total_params": int(params_meta["total_params"]),
        "target_trainable_frac": round(float(params_meta["target_trainable_frac"]), 8),
        "n_lora_layers": int(params_meta.get("n_lora_layers", 0)),
        "source_epochs": int(source_meta["source_epochs"]),
        "source_lr": float(source_meta["source_lr"]),
        "source_weight_decay": float(source_meta["source_weight_decay"]),
        "source_batch_size": int(source_meta["source_batch_size"]),
        "n_source_train": int(source_meta["n_source_train"]),
        "adapt_epochs": 0 if method.endswith("SourceOnly") else int(params_meta["adapt_epochs"]),
        "adapt_lr": np.nan if method.endswith("SourceOnly") else float(params_meta["adapt_lr"]),
        "adapt_weight_decay": (
            np.nan if method.endswith("SourceOnly") else float(params_meta["adapt_weight_decay"])
        ),
        "adapt_batch_size": (
            0 if method.endswith("SourceOnly") else int(params_meta["adapt_batch_size"])
        ),
        "target_ea_policy": "split_self" if model_name == BIG_MODEL else "none",
    }


def prediction_rows(dataset, seed, fold, method, model_name, test_idx, y_test, logits):
    probs = softmax_np(logits)
    pred = logits.argmax(1)
    rows = []
    for row_i, raw_i in enumerate(test_idx):
        rec = {
            "dataset": dataset,
            "seed": int(seed),
            "fold": int(fold),
            "target_subject": int(fold),
            "method": method,
            "model": model_name,
            "test_row": int(row_i),
            "target_trial_index": int(raw_i),
            "y": int(y_test[row_i]),
            "pred": int(pred[row_i]),
            "correct": bool(pred[row_i] == y_test[row_i]),
            "confidence": float(probs[row_i].max()),
        }
        for c in range(logits.shape[1]):
            rec[f"logit_c{c}"] = float(logits[row_i, c])
            rec[f"prob_c{c}"] = float(probs[row_i, c])
        rows.append(rec)
    return rows


def support_rows(dataset, seed, fold, k_per_class, support_draw, split_seed,
                 support_idx, test_idx, y_target):
    rows = []
    for split, indices in (("support", support_idx), ("test", test_idx)):
        for i in indices:
            rows.append({
                "dataset": dataset,
                "seed": int(seed),
                "fold": int(fold),
                "target_subject": int(fold),
                "support_k_per_class": int(k_per_class),
                "support_draw": int(support_draw),
                "split_seed": int(split_seed),
                "split": split,
                "target_trial_index": int(i),
                "y": int(y_target[i]),
                "support_query_policy": "random_stratified_k_per_class_within_loader_downstream_session",
                "session_split_policy": "random_kshot_within_loaded_downstream_session",
            })
    return rows


def complement_row(dataset, seed, fold, k_per_class, support_draw, split_seed,
                   y_test, big_logits, small_logits):
    big_pred = big_logits.argmax(1)
    small_pred = small_logits.argmax(1)
    big_correct = big_pred == y_test
    small_correct = small_pred == y_test
    fused_pred = (big_logits + small_logits).argmax(1)
    big_acc = big_correct.mean() * 100.0
    small_acc = small_correct.mean() * 100.0
    oracle_acc = (big_correct | small_correct).mean() * 100.0
    return {
        "dataset": dataset,
        "seed": int(seed),
        "fold": int(fold),
        "target_subject": int(fold),
        "support_k_per_class": int(k_per_class),
        "support_draw": int(support_draw),
        "split_seed": int(split_seed),
        "test_n": int(len(y_test)),
        "big_method": "MIRepNet-LoRA",
        "small_method": "IFNet-FT",
        "big_acc": round(float(big_acc), 4),
        "small_acc": round(float(small_acc), 4),
        "big_correct_small_wrong": round(float((big_correct & ~small_correct).mean() * 100.0), 4),
        "big_wrong_small_correct": round(float((~big_correct & small_correct).mean() * 100.0), 4),
        "both_correct": round(float((big_correct & small_correct).mean() * 100.0), 4),
        "both_wrong": round(float((~big_correct & ~small_correct).mean() * 100.0), 4),
        "oracle_acc": round(float(oracle_acc), 4),
        "oracle_gap_vs_best": round(float(oracle_acc - max(big_acc, small_acc)), 4),
        "fixed_equal_logit_fusion_acc": round(float((fused_pred == y_test).mean() * 100.0), 4),
    }


def adapt_weight_decay_for(model_cfg_dict, args):
    if args.adapt_weight_decay is not None:
        return float(args.adapt_weight_decay)
    return float(model_cfg_dict.get("weight_decay", 1e-4))


def adapt_batch_size_for(model_cfg_dict, args):
    if args.adapt_batch_size is not None:
        return int(args.adapt_batch_size)
    return int(model_cfg_dict.get("batch_size", 32))


def with_adapt_meta(params_meta, cfg, args):
    out = dict(params_meta)
    out["adapt_epochs"] = int(args.adapt_epochs)
    out["adapt_lr"] = float(args.adapt_lr)
    out["adapt_weight_decay"] = adapt_weight_decay_for(cfg, args)
    out["adapt_batch_size"] = adapt_batch_size_for(cfg, args)
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(metrics_rows, complement_rows_list, summary_csv):
    rows = []
    df = pd.DataFrame(metrics_rows)
    if not df.empty:
        for method, g in df.groupby("method", sort=False):
            rows.append({
                "scope": "method_mean",
                "method": method,
                "n": int(len(g)),
                "mean_acc": round(float(g["acc"].mean()), 4),
                "std_acc": round(float(g["acc"].std(ddof=1)), 4) if len(g) > 1 else 0.0,
                "mean_kappa": round(float(g["kappa"].mean()), 6),
                "mean_target_trainable_params": round(float(g["target_trainable_params"].mean()), 2),
                "mean_target_trainable_frac": round(float(g["target_trainable_frac"].mean()), 8),
            })
        piv = df.pivot_table(index=["dataset", "seed", "fold"], columns="method", values="acc")
        for method in ["MIRepNet-Head", "MIRepNet-LastBlock", "MIRepNet-FullFT", "MIRepNet-LoRA", "IFNet-FT"]:
            if method in piv and "SourceOnly" in piv:
                delta = (piv[method] - piv["SourceOnly"]).dropna()
                rows.append({
                    "scope": "gain_vs_sourceonly",
                    "method": method,
                    "n": int(len(delta)),
                    "mean_acc": round(float(delta.mean()), 4),
                    "std_acc": round(float(delta.std(ddof=1)), 4) if len(delta) > 1 else 0.0,
                    "win_rate": round(float((delta > 0).mean()), 6),
                    "nonnegative_rate": round(float((delta >= 0).mean()), 6),
                })
        if {"MIRepNet-LoRA", "MIRepNet-FullFT"}.issubset(set(piv.columns)):
            delta = (piv["MIRepNet-LoRA"] - piv["MIRepNet-FullFT"]).dropna()
            rows.append({
                "scope": "lora_minus_fullft",
                "method": "MIRepNet-LoRA",
                "n": int(len(delta)),
                "mean_acc": round(float(delta.mean()), 4),
                "std_acc": round(float(delta.std(ddof=1)), 4) if len(delta) > 1 else 0.0,
                "win_rate": round(float((delta > 0).mean()), 6),
                "nonnegative_rate": round(float((delta >= 0).mean()), 6),
            })
    comp = pd.DataFrame(complement_rows_list)
    if not comp.empty:
        rows.append({
            "scope": "lora_ifnet_complement_mean",
            "method": "MIRepNet-LoRA_vs_IFNet-FT",
            "n": int(len(comp)),
            "big_acc": round(float(comp["big_acc"].mean()), 4),
            "small_acc": round(float(comp["small_acc"].mean()), 4),
            "oracle_acc": round(float(comp["oracle_acc"].mean()), 4),
            "oracle_gap_vs_best": round(float(comp["oracle_gap_vs_best"].mean()), 4),
            "fixed_equal_logit_fusion_acc": round(float(comp["fixed_equal_logit_fusion_acc"].mean()), 4),
        })
    if rows:
        write_csv(summary_csv, rows)
    return pd.DataFrame(rows)


def run_fold(dataset, dcfg, fold, seed, device, args):
    num_classes = int(dcfg["num_classes"])
    split_seed = int(seed + 1009 * fold + 100000 * args.support_draw)
    selected_methods = set(args.methods)

    print(f"[fold {fold} seed {seed}] train MIRepNet source", flush=True)
    big_ad, big_base, X_target, y_target, big_source_meta = build_source_model(
        BIG_MODEL, dataset, fold, seed, num_classes, device, args
    )
    support_idx, test_idx = kshot_indices_per_class(
        y_target, num_classes, args.k_per_class, split_seed
    )
    y_support = y_target[support_idx]
    y_test = y_target[test_idx]
    X_big_sup, X_big_te = target_arrays(BIG_MODEL, big_ad, X_target, support_idx, test_idx, fold)
    big_cfg = big_ad.cfg

    metrics_rows = []
    pred_rows = []
    support_split_rows = support_rows(
        dataset, seed, fold, args.k_per_class, args.support_draw, split_seed,
        support_idx, test_idx, y_target
    )
    logits_for_complement = {}

    source_logits = infer_logits(big_ad, big_base, X_big_te)
    source_params = {
        "target_trainable_params": 0,
        "total_params": count_params(big_base)[1],
        "target_trainable_frac": 0.0,
        "n_lora_layers": 0,
        "adapt_epochs": 0,
        "adapt_lr": np.nan,
        "adapt_weight_decay": np.nan,
        "adapt_batch_size": 0,
    }
    if "SourceOnly" in selected_methods:
        metrics_rows.append(metric_row(
            dataset, seed, fold, "SourceOnly", BIG_MODEL, "none",
            source_params, big_source_meta, args.k_per_class, args.support_draw,
            split_seed, y_test, source_logits
        ))
        pred_rows.extend(prediction_rows(
            dataset, seed, fold, "SourceOnly", BIG_MODEL, test_idx, y_test, source_logits
        ))
        print(
            f"fold={fold} seed={seed} SourceOnly | acc={metrics_rows[-1]['acc']:.2f} "
            f"kappa={metrics_rows[-1]['kappa']:.4f}",
            flush=True,
        )

    for method in ["MIRepNet-Head", "MIRepNet-LastBlock", "MIRepNet-FullFT", "MIRepNet-LoRA"]:
        if method not in selected_methods:
            continue
        set_seed(seed + METHOD_SEED_OFFSET[method] + 1000 * fold)
        model = copy.deepcopy(big_base)
        params_meta = prepare_mirepnet_target_method(model, method, args)
        params_meta = with_adapt_meta(params_meta, big_cfg, args)
        model = finetune_ce(
            big_ad,
            model,
            X_big_sup,
            y_support,
            num_classes,
            epochs=args.adapt_epochs,
            lr=args.adapt_lr,
            weight_decay=params_meta["adapt_weight_decay"],
            batch_size=params_meta["adapt_batch_size"],
            seed=seed + METHOD_SEED_OFFSET[method] + 1000 * fold,
            train_mode=params_meta["train_mode"],
        )
        logits = infer_logits(big_ad, model, X_big_te)
        metrics_rows.append(metric_row(
            dataset, seed, fold, method, BIG_MODEL, params_meta["target_update"],
            params_meta, big_source_meta, args.k_per_class, args.support_draw,
            split_seed, y_test, logits
        ))
        pred_rows.extend(prediction_rows(
            dataset, seed, fold, method, BIG_MODEL, test_idx, y_test, logits
        ))
        if method == "MIRepNet-LoRA":
            logits_for_complement["big"] = logits
        print(
            f"fold={fold} seed={seed} {method} | acc={metrics_rows[-1]['acc']:.2f} "
            f"kappa={metrics_rows[-1]['kappa']:.4f} "
            f"trainable={params_meta['target_trainable_params']}",
            flush=True,
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[fold {fold} seed {seed}] train IFNet source", flush=True)
    small_ad, small_base, X_target_small, y_target_small, small_source_meta = build_source_model(
        SMALL_MODEL, dataset, fold, seed, num_classes, device, args
    )
    if not np.array_equal(y_target, y_target_small):
        raise ValueError("target labels are not aligned between MIRepNet and IFNet loaders")
    X_small_sup, X_small_te = target_arrays(SMALL_MODEL, small_ad, X_target_small, support_idx, test_idx, fold)
    small_cfg = small_ad.cfg

    small_source_logits = infer_logits(small_ad, small_base, X_small_te)
    small_source_params = {
        "target_trainable_params": 0,
        "total_params": count_params(small_base)[1],
        "target_trainable_frac": 0.0,
        "n_lora_layers": 0,
        "adapt_epochs": 0,
        "adapt_lr": np.nan,
        "adapt_weight_decay": np.nan,
        "adapt_batch_size": 0,
    }
    if "IFNet-SourceOnly" in selected_methods:
        metrics_rows.append(metric_row(
            dataset, seed, fold, "IFNet-SourceOnly", SMALL_MODEL, "none",
            small_source_params, small_source_meta, args.k_per_class, args.support_draw,
            split_seed, y_test, small_source_logits
        ))
        pred_rows.extend(prediction_rows(
            dataset, seed, fold, "IFNet-SourceOnly", SMALL_MODEL, test_idx, y_test, small_source_logits
        ))
        print(
            f"fold={fold} seed={seed} IFNet-SourceOnly | acc={metrics_rows[-1]['acc']:.2f} "
            f"kappa={metrics_rows[-1]['kappa']:.4f}",
            flush=True,
        )

    if "IFNet-FT" in selected_methods:
        set_seed(seed + METHOD_SEED_OFFSET["IFNet-FT"] + 1000 * fold)
        small_model = copy.deepcopy(small_base)
        params_meta = prepare_ifnet_fullft(small_model)
        params_meta = with_adapt_meta(params_meta, small_cfg, args)
        small_model = finetune_ce(
            small_ad,
            small_model,
            X_small_sup,
            y_support,
            num_classes,
            epochs=args.adapt_epochs,
            lr=args.adapt_lr,
            weight_decay=params_meta["adapt_weight_decay"],
            batch_size=params_meta["adapt_batch_size"],
            seed=seed + METHOD_SEED_OFFSET["IFNet-FT"] + 1000 * fold,
            train_mode=params_meta["train_mode"],
        )
        small_logits = infer_logits(small_ad, small_model, X_small_te)
        metrics_rows.append(metric_row(
            dataset, seed, fold, "IFNet-FT", SMALL_MODEL, params_meta["target_update"],
            params_meta, small_source_meta, args.k_per_class, args.support_draw,
            split_seed, y_test, small_logits
        ))
        pred_rows.extend(prediction_rows(
            dataset, seed, fold, "IFNet-FT", SMALL_MODEL, test_idx, y_test, small_logits
        ))
        logits_for_complement["small"] = small_logits
        print(
            f"fold={fold} seed={seed} IFNet-FT | acc={metrics_rows[-1]['acc']:.2f} "
            f"kappa={metrics_rows[-1]['kappa']:.4f} "
            f"trainable={params_meta['target_trainable_params']}",
            flush=True,
        )
        del small_model

    complement_rows = []
    if {"big", "small"}.issubset(logits_for_complement):
        complement_rows.append(complement_row(
            dataset, seed, fold, args.k_per_class, args.support_draw, split_seed,
            y_test, logits_for_complement["big"], logits_for_complement["small"]
        ))

    del big_base, small_base
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics_rows, pred_rows, support_split_rows, complement_rows


def main():
    args = parse_args()
    args.methods = normalize_methods(args.methods)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    dataset = args.dataset
    dcfg = config.load_dataset_config(dataset)
    folds = args.folds if args.folds is not None else list(range(int(dcfg["num_subjects"])))
    device = device_from_gpu(args.gpu)

    paths = default_paths(dataset, args.k_per_class, args.seeds)
    out_csv = Path(args.out_csv) if args.out_csv else paths["out_csv"]
    pred_csv = Path(args.pred_csv) if args.pred_csv else paths["pred_csv"]
    support_csv = Path(args.support_csv) if args.support_csv else paths["support_csv"]
    complement_csv = Path(args.complement_csv) if args.complement_csv else paths["complement_csv"]
    summary_csv = Path(args.summary_csv) if args.summary_csv else paths["summary_csv"]

    if args.overwrite:
        metric_rows = []
        pred_rows = []
        split_rows = []
        comp_rows = []
    else:
        metric_rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
        pred_rows = pd.read_csv(pred_csv).to_dict("records") if pred_csv.exists() else []
        split_rows = pd.read_csv(support_csv).to_dict("records") if support_csv.exists() else []
        comp_rows = pd.read_csv(complement_csv).to_dict("records") if complement_csv.exists() else []

    completed = {
        (r["dataset"], int(r["seed"]), int(r["fold"]), r["method"])
        for r in metric_rows
    }

    print(
        f"[cfg] dataset={dataset} K/class={args.k_per_class} seeds={args.seeds} "
        f"folds={folds} adapt_epochs={args.adapt_epochs} adapt_lr={args.adapt_lr:g} "
        f"LoRA(r={args.lora_rank}, alpha={args.lora_alpha:g}, dropout={args.lora_dropout:g}, "
        f"targets={args.lora_targets}) device={device}",
        flush=True,
    )
    print(
        f"[cfg] methods={','.join(args.methods)}",
        flush=True,
    )

    for seed in args.seeds:
        for fold in folds:
            fold = int(fold)
            expected = {(dataset, int(seed), fold, m) for m in args.methods}
            if expected.issubset(completed) and not args.overwrite:
                print(f"[skip] dataset={dataset} fold={fold} seed={seed}", flush=True)
                continue
            mr, pr, sr, cr = run_fold(dataset, dcfg, fold, int(seed), device, args)
            metric_rows.extend(mr)
            pred_rows.extend(pr)
            split_rows.extend(sr)
            comp_rows.extend(cr)
            completed.update((r["dataset"], int(r["seed"]), int(r["fold"]), r["method"]) for r in mr)
            write_csv(out_csv, metric_rows)
            write_csv(pred_csv, pred_rows)
            write_csv(support_csv, split_rows)
            write_csv(complement_csv, comp_rows)
            summarize(metric_rows, comp_rows, summary_csv)
            print(f"[write] {out_csv}", flush=True)

    summary = summarize(metric_rows, comp_rows, summary_csv)
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        print("\nStudent/adapted model means:", flush=True)
        print(
            df.groupby("method", sort=False)[["acc", "kappa", "target_trainable_params", "target_trainable_frac"]]
            .mean()
            .round(4)
            .to_string(),
            flush=True,
        )
    if comp_rows:
        comp = pd.DataFrame(comp_rows)
        print("\nMIRepNet-LoRA vs IFNet-FT complement:", flush=True)
        print(
            comp[[
                "big_acc",
                "small_acc",
                "oracle_acc",
                "oracle_gap_vs_best",
                "fixed_equal_logit_fusion_acc",
            ]]
            .mean()
            .round(4)
            .to_string(),
            flush=True,
        )
    if not summary.empty:
        print("\nSummary:", flush=True)
        print(summary.to_string(index=False), flush=True)
    print(f"\nWrote {out_csv}", flush=True)
    print(f"Wrote {pred_csv}", flush=True)
    print(f"Wrote {support_csv}", flush=True)
    print(f"Wrote {complement_csv}", flush=True)
    print(f"Wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
