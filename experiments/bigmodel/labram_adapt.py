"""LaBraM native-preprocessing downstream adaptation for the paper-5 MI tasks.

Parallel to ``cbramod_adapt.py`` but faithful to LaBraM's own pipeline:

  * preprocessing = band-pass 0.1-75 Hz + notch 50 Hz + resample to 200 Hz +
    divide by 100 (µV -> 0.1 mV units, LaBraM's ``normalization``); *no* CAR.
    Reshape to ``(channels, seconds, 200)`` (1-s patches of 200 samples).
  * model = pretrained ``labram_base_patch200_200`` backbone + its native
    ``Linear`` head, full fine-tune. Per-channel positional embeddings are
    selected via ``input_chans`` (indices into LaBraM's ``standard_1020``
    montage, computed from each dataset's channel names).
  * optimizer = AdamW with LaBraM's layer-wise lr decay (get_parameter_groups +
    LayerDecayValueAssigner from ``optim_factory``); official finetune hp
    (lr 5e-4, layer_decay 0.9, wd 0.05, drop_path 0.1, smoothing 0.1).

Protocol mirrors CBraMod native (``native70`` = 70/30 within-subject, seeds
666/667/668) so the two foundation models are directly comparable.
"""
import argparse
import glob
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root

from models.labram.modeling_finetune import labram_base_patch200_200  # noqa: E402
from models.labram.optim_factory import (  # noqa: E402
    get_parameter_groups, LayerDecayValueAssigner)
from models.labram.montage import get_input_chans  # noqa: E402
import paths  # noqa: E402
from data.preproc import bandpass as _bandpass, notch as _notch  # noqa: E402

PRETRAIN = paths.weight_path("labram")


def _load_channel_names():
    """Channel-name lists (uppercase, matching X.npy channel order)."""
    from data import channels as cl
    return {
        "BNCI2014001_4c": cl.BNCI2014001_chn_names,
        "BNCI2014001_2c": cl.BNCI2014001_chn_names,
        "BNCI2014004": cl.BNCI2014004_chn_names,
        "AlexMI_2c": cl.AlexMI_chn_names,
        "BNCI2015001": cl.BNCI2015001_chn_names,
    }


@dataclass(frozen=True)
class DatasetCfg:
    source: str
    data_dir: str
    fs: int
    seconds: int
    num_classes: int
    keep: tuple = ()
    session: str = None
    l_freq: float = 0.1
    h_freq: float = 75.0
    notch_freq: float = 50.0
    target_fs: int = 200
    norm: str = "none"  # LaBraM does no spatial CAR; "car" available for ablation


_DATA_ROOT = os.environ.get('DATA_ROOT', '/data1/llx')  # raw-data root override

DATASETS = {
    "BNCI2014001_4c": DatasetCfg("BNCI2014001", os.path.join(_DATA_ROOT, "BNCI2014001"), 250, 4, 4),
    "BNCI2014001_2c": DatasetCfg("BNCI2014001", os.path.join(_DATA_ROOT, "BNCI2014001"), 250, 4, 2,
                                 keep=("left_hand", "right_hand")),
    "BNCI2014004": DatasetCfg("BNCI2014004", os.path.join(_DATA_ROOT, "BNCI2014004"), 250, 4, 2),
    "AlexMI_2c": DatasetCfg("AlexMI", os.path.join(_DATA_ROOT, "AlexMI"), 512, 3, 2,
                            keep=("right_hand", "feet")),
    "BNCI2015001": DatasetCfg("BNCI2015001", os.path.join(_DATA_ROOT, "BNCI2015001"), 512, 4, 2),
}

ALIASES = {
    "BNCI2014001": "BNCI2014001_4c",
    "BNCI2014001-4": "BNCI2014001_4c",
    "14001-4": "BNCI2014001_4c",
    "14001-2": "BNCI2014001_2c",
    "14004": "BNCI2014004",
    "15001": "BNCI2015001",
}

PAPER80_SESSIONS = {
    "BNCI2014001_4c": "session_T",
    "BNCI2014001_2c": "session_T",
    "BNCI2014004": "session_3",
    "BNCI2015001": "session_A",
}


def canonical_dataset_name(name):
    return ALIASES.get(name, name)


def set_seed(seed):
    from collab.seed import set_seed as _seed  # unified impl
    _seed(seed)


def load_dataset(cfg):
    x = np.load(os.path.join(cfg.data_dir, "X.npy"))
    y_raw = np.load(os.path.join(cfg.data_dir, "labels.npy"), allow_pickle=True)
    meta_files = glob.glob(os.path.join(cfg.data_dir, "*.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata CSV found in {cfg.data_dir}")
    meta = pd.read_csv(meta_files[0])
    if cfg.keep:
        keep_mask = np.isin(y_raw.astype(str), np.asarray(cfg.keep))
        x, y_raw = x[keep_mask], y_raw[keep_mask]
        meta = meta[keep_mask].reset_index(drop=True)
    if cfg.session:
        if "session" not in meta.columns:
            raise ValueError(f"{cfg.source} has no session column for {cfg.session}")
        session_mask = (meta["session"].astype(str) == cfg.session).values
        x, y_raw = x[session_mask], y_raw[session_mask]
        meta = meta[session_mask].reset_index(drop=True)
    y = LabelEncoder().fit_transform(y_raw)
    return x, y, meta


def preprocess(x, cfg, scale_divisor):
    """(N, ch, T) -> (N, ch, seconds, 200), faithful LaBraM native format."""
    if cfg.target_fs != 200:
        raise ValueError("LaBraM pretrained patch size expects target_fs=200.")
    src_len = int(cfg.seconds * cfg.fs)
    if x.shape[-1] < src_len:
        raise ValueError(
            f"{cfg.source} has only {x.shape[-1]} samples, need {src_len} "
            f"for {cfg.seconds}s at {cfg.fs}Hz")
    x = x[:, :, :src_len].astype(np.float32, copy=False)
    if cfg.norm == "car":
        x = x - x.mean(axis=1, keepdims=True)
    elif cfg.norm not in ("none", "None", None):
        raise ValueError(f"Unsupported norm: {cfg.norm}")
    x = _bandpass(x, cfg.fs, cfg.l_freq, cfg.h_freq)
    x = _notch(x, cfg.fs, cfg.notch_freq)
    x = resample(x, cfg.seconds * cfg.target_fs, axis=-1)
    x = (x / scale_divisor).astype(np.float32)
    n, ch, _ = x.shape
    return x.reshape(n, ch, cfg.seconds, 200)


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr):
    if epochs <= 0:
        return None
    min_factor = 0.0 if min_lr is None else min_lr / base_lr

    def lr_factor(epoch_idx):
        step = epoch_idx + 1
        if warmup_epochs > 0 and step <= warmup_epochs:
            return max(step / warmup_epochs, min_factor)
        denom = max(1, epochs - warmup_epochs)
        progress = (step - warmup_epochs) / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def class_weight_tensor(y, num_classes, device):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(num_classes, drop_path, device):
    model = labram_base_patch200_200(
        num_classes=num_classes, drop_path_rate=drop_path, use_mean_pooling=True,
        init_values=0.1, qkv_bias=True, use_abs_pos_emb=True, use_rel_pos_bias=False)
    ckpt = torch.load(PRETRAIN, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
    model_sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[labram] loaded {len(sd)} tensors; {len(missing)} missing, "
          f"{len(unexpected)} unexpected", flush=True)
    return model.to(device)


def build_optimizer(model, args):
    """AdamW with LaBraM's layer-wise lr decay (faithful to run_class_finetuning)."""
    num_layers = model.get_num_layers()
    assigner = LayerDecayValueAssigner(
        [args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)])
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = get_parameter_groups(
        model, args.weight_decay, skip, assigner.get_layer_id, assigner.get_scale)
    for g in groups:                       # bake lr_scale into the base lr
        g["lr"] = args.lr * g.get("lr_scale", 1.0)
    return torch.optim.AdamW(groups, lr=args.lr, eps=args.opt_eps)


def _predict(model, x, batch_size, device, input_chans):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(x[i:i + batch_size].to(device), input_chans=input_chans)
            preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()


def run_subject(x_sub, y_sub, cfg, args, seed, device, input_chans):
    """Fine-tune with a validation split; report test at the best-val epoch.

    Faithful to LaBraM's downstream recipe (best-checkpoint-by-validation) rather
    than CBraMod-native's fixed-epoch run — LaBraM overfits within-subject MI in a
    few epochs, so a fixed epoch count is fragile.
    """
    set_seed(seed)
    idx = np.arange(len(y_sub))
    train_idx, test_idx = train_test_split(
        idx, test_size=1.0 - args.train_percentage, stratify=y_sub, random_state=seed)
    # carve validation out of the training portion for best-epoch selection
    tr_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_split, stratify=y_sub[train_idx], random_state=seed)

    x_tr = torch.from_numpy(preprocess(x_sub[tr_idx], cfg, args.scale_divisor)).float()
    x_val = torch.from_numpy(preprocess(x_sub[val_idx], cfg, args.scale_divisor)).float()
    x_test = torch.from_numpy(preprocess(x_sub[test_idx], cfg, args.scale_divisor)).float()
    y_tr_np = y_sub[tr_idx]
    y_tr = torch.from_numpy(y_tr_np).long()
    y_val, y_test = y_sub[val_idx], y_sub[test_idx]

    model = build_model(cfg.num_classes, args.drop_path, device)
    optimizer = build_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr)
    weight = class_weight_tensor(y_tr_np, cfg.num_classes, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=args.batch_size,
        shuffle=True, num_workers=args.dataloader_workers, drop_last=False)

    best_val, best = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, input_chans=input_chans), yb)
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        val_bac = balanced_accuracy_score(y_val, _predict(model, x_val, args.batch_size, device, input_chans))
        if val_bac > best_val:
            pred = _predict(model, x_test, args.batch_size, device, input_chans)
            best_val = val_bac
            best = {
                "acc": accuracy_score(y_test, pred) * 100.0,
                "bac": balanced_accuracy_score(y_test, pred),
                "kappa": cohen_kappa_score(y_test, pred),
                "best_epoch": ep + 1,
                "val_bac": val_bac,
            }
    best.update(n_train=int(len(tr_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def existing_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path)
    if df.empty:
        return set()
    return set(zip(df["dataset"], df["subject"], df["seed"]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help=f"One of: {', '.join(DATASETS)}")
    p.add_argument("--preset", choices=["native70", "paper80"], default="native70")
    p.add_argument("--subjects", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[666, 667, 668])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--train_percentage", type=float, default=None)
    p.add_argument("--val_split", type=float, default=0.2,
                   help="fraction of the train portion held out for best-epoch selection")
    p.add_argument("--session", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dataloader_workers", type=int, default=0)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--class_weights", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--layer_decay", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--opt_eps", type=float, default=1e-8)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--clip_grad_norm", type=float, default=3.0)
    p.add_argument("--scale_divisor", type=float, default=100.0)

    p.add_argument("--l_freq", type=float, default=None)
    p.add_argument("--h_freq", type=float, default=None)
    p.add_argument("--notch_freq", type=float, default=None)
    p.add_argument("--norm_method", default=None, choices=["car", "none"])
    return p.parse_args()


def main():
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {args.dataset!r}; canonical={dataset!r}")

    cfg = DATASETS[dataset]
    train_percentage = args.train_percentage
    session = args.session
    if args.preset == "native70":
        train_percentage = 0.7 if train_percentage is None else train_percentage
    elif args.preset == "paper80":
        train_percentage = 0.8 if train_percentage is None else train_percentage
        session = PAPER80_SESSIONS.get(dataset, session)
    args.train_percentage = train_percentage

    cfg = replace(
        cfg, session=session,
        l_freq=cfg.l_freq if args.l_freq is None else args.l_freq,
        h_freq=cfg.h_freq if args.h_freq is None else args.h_freq,
        notch_freq=cfg.notch_freq if args.notch_freq is None else args.notch_freq,
        norm=cfg.norm if args.norm_method is None else args.norm_method,
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ch_names = _load_channel_names()[dataset]
    input_chans = get_input_chans(ch_names)

    tag = f"{args.preset}_train{args.train_percentage:g}"
    if cfg.session:
        tag += f"_{cfg.session}"
    out_path = args.out or os.path.join("results", "labram", f"{dataset}_{tag}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x, y, meta = load_dataset(cfg)
    subjects = args.subjects or sorted(meta["subject"].unique())
    completed = set() if args.overwrite else existing_keys(out_path)
    rows = pd.read_csv(out_path).to_dict("records") if (
        os.path.exists(out_path) and not args.overwrite) else []

    print(
        f"[cfg] dataset={dataset} source={cfg.source} preset={args.preset} "
        f"train={args.train_percentage:g} session={cfg.session} "
        f"nch={len(ch_names)} prep={cfg.norm}+{cfg.l_freq}-{cfg.h_freq}Hz"
        f"{'+notch'+str(cfg.notch_freq) if cfg.notch_freq else ''}"
        f"+resample{cfg.target_fs}+/ {args.scale_divisor:g} "
        f"epochs={args.epochs} bs={args.batch_size} lr={args.lr:g} "
        f"layer_decay={args.layer_decay:g} wd={args.weight_decay:g} gpu={device}",
        flush=True)

    for seed in args.seeds:
        for subject in subjects:
            subject = int(subject)
            key = (dataset, subject, seed)
            if key in completed:
                print(f"[skip] {dataset} S{subject} seed{seed}", flush=True)
                continue
            mask = (meta["subject"] == subject).values
            metrics = run_subject(x[mask], y[mask], cfg, args, seed, device, input_chans)
            row = {
                "dataset": dataset, "source": cfg.source, "preset": args.preset,
                "subject": subject, "seed": seed, "session": cfg.session or "",
                "train_percentage": args.train_percentage, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "layer_decay": args.layer_decay, "weight_decay": args.weight_decay,
                "drop_path": args.drop_path, "label_smoothing": args.label_smoothing,
                "l_freq": cfg.l_freq, "h_freq": cfg.h_freq,
                "notch_freq": cfg.notch_freq or "",
                "acc": round(metrics["acc"], 4), "bac": round(metrics["bac"], 6),
                "kappa": round(metrics["kappa"], 6),
                "best_epoch": metrics["best_epoch"], "val_bac": round(metrics["val_bac"], 6),
                "n_train": metrics["n_train"], "n_val": metrics["n_val"],
                "n_test": metrics["n_test"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(
                f"{dataset} S{subject} seed{seed} | acc={metrics['acc']:.2f}% "
                f"bac={metrics['bac']:.4f} kappa={metrics['kappa']:.4f} "
                f"(best_ep={metrics['best_epoch']} val_bac={metrics['val_bac']:.3f})", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(
            f"==== {dataset} {args.preset}: acc={df.acc.mean():.2f}% "
            f"bac={df.bac.mean():.4f} kappa={df.kappa.mean():.4f} "
            f"n={len(df)} -> {out_path} ====", flush=True)


if __name__ == "__main__":
    main()
