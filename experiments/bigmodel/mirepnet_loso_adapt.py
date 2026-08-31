"""MIRepNet LOSO end-to-end adaptation metrics for hyperparameter search.

Each held-out subject is evaluated by finetuning MIRepNet on all remaining
subjects. EA is applied per subject before the mixed LOSO train set is passed to
MIRepNet, matching ``scripts/export/export_teacher_loso.py`` and avoiding a
single transductive whitening covariance over multiple people.

Example:
    conda run -n mirepnet python experiments/bigmodel/mirepnet_loso_adapt.py \
        --dataset BNCI2014001-4 --gpu 0 --seeds 666 --epochs 30 --out /tmp/mirep_loso.csv
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

import config
import data
from models import get_adapter

MODEL = "mirepnet"

ALIASES = {
    "BNCI2014001_4c": "BNCI2014001-4",
    "BNCI2014001-4": "BNCI2014001-4",
    "14001-4": "BNCI2014001-4",
    "BNCI2014001_2c": "BNCI2014001",
    "BNCI2014001_2": "BNCI2014001",
    "BNCI2014001": "BNCI2014001",
    "14001-2": "BNCI2014001",
    "BNCI2014004": "BNCI2014004",
    "14004": "BNCI2014004",
    "AlexMI_2c": "AlexMI",
    "AlexMI": "AlexMI",
    "BNCI2015001": "BNCI2015001",
    "15001": "BNCI2015001",
}


def canonical_dataset_name(name):
    return ALIASES.get(name, name)


def set_seed(seed):
    from collab.seed import set_seed as _seed  # unified impl (was: torch.default_generator.manual_seed)
    _seed(seed)


def existing_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path)
    if df.empty:
        return set()
    return set(zip(df["dataset"], df["subject"], df["seed"]))


def run_fold(dataset, dcfg, mcfg, fold, seed, device):
    X_tr, y_tr, subj_tr, X_te, y_te = data.loso_split(dataset, fold)
    adapter_cfg = dict(mcfg)
    adapter_cfg.update(
        in_channels=X_tr.shape[1],
        samples=X_tr.shape[2],
        dataset_name=dataset,
        skip_preprocess=True,
    )
    ad = get_adapter(MODEL, device=device, **adapter_cfg)

    Xp_tr = ad.ea_pad_per_subject(X_tr, subj_tr)
    Xp_te = ad.ea_pad_per_subject(X_te, np.full(len(y_te), fold))

    set_seed(seed)
    model = ad.build(dcfg["num_classes"])
    model = ad.finetune(model, Xp_tr, y_tr, dcfg["num_classes"])

    _feats, logits = ad.infer(model, Xp_te)
    pred = logits.argmax(1)
    metrics = {
        "acc": accuracy_score(y_te, pred) * 100.0,
        "bac": balanced_accuracy_score(y_te, pred),
        "kappa": cohen_kappa_score(y_te, pred),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--torch_threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    return p.parse_args()


def main():
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    dataset = canonical_dataset_name(args.dataset)
    dcfg = config.load_dataset_config(dataset)
    mcfg = config.load_model_config(MODEL)
    for key in ("epochs", "lr", "weight_decay", "batch_size"):
        val = getattr(args, key)
        if val is not None:
            mcfg[key] = val

    folds = args.folds if args.folds is not None else list(range(dcfg["num_subjects"]))
    seeds = args.seeds if args.seeds is not None else dcfg["seeds"]
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = (f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available()
              else "cpu")
    out_path = args.out or os.path.join(
        "results", "mirepnet_loso", f"{dataset}_loso.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    if os.path.exists(out_path) and not args.overwrite:
        rows = pd.read_csv(out_path).to_dict("records")
    completed = set() if args.overwrite else existing_keys(out_path)

    print(
        f"[cfg] dataset={dataset} protocol=loso epochs={mcfg.get('epochs')} "
        f"bs={mcfg.get('batch_size')} lr={mcfg.get('lr'):g} "
        f"wd={mcfg.get('weight_decay'):g} gpu={device} threads={args.torch_threads}",
        flush=True,
    )
    for seed in seeds:
        for fold in folds:
            fold = int(fold)
            key = (dataset, fold, seed)
            if key in completed:
                print(f"[skip] {dataset} fold{fold} seed{seed}", flush=True)
                continue
            metrics = run_fold(dataset, dcfg, mcfg, fold, seed, device)
            row = {
                "dataset": dataset,
                "model": MODEL,
                "protocol": "loso",
                "subject": fold,
                "fold": fold,
                "seed": seed,
                "epochs": int(mcfg.get("epochs")),
                "batch_size": int(mcfg.get("batch_size")),
                "lr": float(mcfg.get("lr")),
                "weight_decay": float(mcfg.get("weight_decay")),
                "acc": round(metrics["acc"], 4),
                "bac": round(metrics["bac"], 6),
                "kappa": round(metrics["kappa"], 6),
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(
                f"{dataset} fold{fold} seed{seed} | acc={metrics['acc']:.2f}% "
                f"bac={metrics['bac']:.4f} kappa={metrics['kappa']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        print(
            f"==== {dataset} loso: acc={df.acc.mean():.2f}% "
            f"bac={df.bac.mean():.4f} kappa={df.kappa.mean():.4f} "
            f"n={len(df)} -> {out_path} ====",
            flush=True,
        )


if __name__ == "__main__":
    main()
