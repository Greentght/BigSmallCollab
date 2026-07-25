"""CBraMod native-preprocessing downstream adaptation for the paper-5 MI tasks.

This runner intentionally follows the "native preprocessing" step recorded in
PROGRESS.md: CAR, band-pass filtering, optional notch filtering, resample to
200 Hz, divide by 10, then reshape to (channels, one-second patches, 200).
It uses the native all_patch_reps classifier head rather than the earlier
avg-pool adapter.
"""
import argparse
import glob
import math
import os
import random
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from backbones.cbramod.cbramod import CBraMod  # noqa: E402
from core import paths  # noqa: E402

PRETRAIN = paths.weight_path("cbramod")


@dataclass(frozen=True)
class DatasetCfg:
    source: str
    data_dir: str
    fs: int
    seconds: int
    num_classes: int
    keep: tuple[str, ...] = ()
    session: str | None = None
    l_freq: float = 0.3
    h_freq: float = 50.0
    notch_freq: float | None = None
    target_fs: int = 200
    norm: str = "car"
    pad_to_seconds: int = 0   # benchmark adjust_time_length: keep full signal, pad-repeat to N s
    pipeline: str = "native"  # native=CAR->filter->resample; benchmark=resample->trim/pad->filter->CAR


DATASETS = {
    "BNCI2014001_4c": DatasetCfg(
        source="BNCI2014001",
        data_dir="/data1/llx/BNCI2014001",
        fs=250,
        seconds=4,
        num_classes=4,
    ),
    "BNCI2014001_2c": DatasetCfg(
        source="BNCI2014001",
        data_dir="/data1/llx/BNCI2014001",
        fs=250,
        seconds=4,
        num_classes=2,
        keep=("left_hand", "right_hand"),
    ),
    "BNCI2014004": DatasetCfg(
        source="BNCI2014004",
        data_dir="/data1/llx/BNCI2014004",
        fs=250,
        seconds=4,
        num_classes=2,
    ),
    "AlexMI_2c": DatasetCfg(
        source="AlexMI",
        data_dir="/data1/llx/AlexMI",
        fs=512,
        seconds=3,
        num_classes=2,
        keep=("right_hand", "feet"),
    ),
    "BNCI2015001": DatasetCfg(
        source="BNCI2015001",
        data_dir="/data1/llx/BNCI2015001",
        fs=512,
        seconds=4,
        num_classes=2,
    ),
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


class CBraModClassifier(nn.Module):
    def __init__(self, num_classes, n_ch, n_patch, dropout, pretrain=PRETRAIN, head="mlp"):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        if pretrain:
            self.backbone.load_state_dict(torch.load(pretrain, map_location="cpu"))
        self.backbone.proj_out = nn.Identity()
        flat_dim = n_ch * n_patch * 200
        if head == "mlp":            # official all_patch_reps 3-layer head (default)
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flat_dim, 800),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(800, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, num_classes),
            )
        elif head == "linear":       # benchmark LinearLayers: Dropout -> single Linear
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Dropout(dropout),
                nn.Linear(flat_dim, num_classes),
            )
        else:
            raise ValueError(f"unknown head {head!r}; use 'mlp' or 'linear'")

    def forward(self, x):
        return self.classifier(self.backbone(x))


def canonical_dataset_name(name):
    return ALIASES.get(name, name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _bandpass(x, fs, l_freq, h_freq):
    nyq = fs / 2.0
    high = min(h_freq, nyq - 1e-3)
    if l_freq <= 0 and high >= nyq:
        return x
    if l_freq <= 0:
        b, a = butter(4, high / nyq, btype="low")
    else:
        b, a = butter(4, [l_freq / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _notch(x, fs, notch_freq):
    if notch_freq is None or notch_freq <= 0 or notch_freq >= fs / 2.0:
        return x
    b, a = iirnotch(w0=notch_freq, Q=30, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def _pad_or_trim_repeat(x, target):
    """Match benchmark _trim_or_pad_time: trim to target, or pad by repeating from
    the start (wrap) when shorter. x: (N, ch, T) -> (N, ch, target)."""
    cur = x.shape[-1]
    if cur >= target:
        return x[..., :target]
    out = np.zeros(x.shape[:-1] + (target,), dtype=x.dtype)
    out[..., :cur] = x
    rem, pos = target - cur, cur
    while rem > 0:
        fill = min(rem, cur)
        out[..., pos:pos + fill] = x[..., :fill]
        pos += fill
        rem -= fill
    return out


def _preprocess_benchmark(x, cfg, scale_divisor, target_sec):
    """Benchmark order (utils/preprocessing.py): resample -> adjust_time_length
    (trim/pad-repeat) -> bandpass -> notch -> normalize(CAR last). Filtering thus
    happens at 200 Hz, not 250 Hz."""
    x = np.asarray(x, dtype=np.float32)
    n_dst = int(round(x.shape[-1] * cfg.target_fs / cfg.fs))   # full-signal resample
    x = resample(x, n_dst, axis=-1)
    x = _pad_or_trim_repeat(x, target_sec * cfg.target_fs)      # adjust_time_length
    x = _bandpass(x, cfg.target_fs, cfg.l_freq, cfg.h_freq)     # filter @200 Hz
    x = _notch(x, cfg.target_fs, cfg.notch_freq)
    if cfg.norm in ("car", "car_z"):                            # CAR last
        x = x - x.mean(axis=1, keepdims=True)
    elif cfg.norm not in ("none", "None", None, "z_score"):
        raise ValueError(f"Unsupported norm: {cfg.norm}")
    if cfg.norm in ("z_score", "car_z"):
        mu = x.mean(axis=-1, keepdims=True); sd = x.std(axis=-1, keepdims=True)
        x = (x - mu) / (sd + 1e-8)
    else:
        x = x / scale_divisor
    x = x.astype(np.float32)
    n, ch, _ = x.shape
    return x.reshape(n, ch, target_sec, 200)


def preprocess(x, cfg, scale_divisor):
    """(N, ch, T) -> (N, ch, seconds, 200), faithful CBraMod native format."""
    if cfg.target_fs != 200:
        raise ValueError("CBraMod pretrained patch size expects target_fs=200.")
    pad_to = getattr(cfg, "pad_to_seconds", 0) or 0
    if getattr(cfg, "pipeline", "native") == "benchmark":
        target_sec = pad_to if (pad_to and pad_to > cfg.seconds) else cfg.seconds
        return _preprocess_benchmark(x, cfg, scale_divisor, target_sec)
    if pad_to and pad_to > cfg.seconds:
        # benchmark-style: keep FULL raw signal, CAR/filter, resample preserving
        # true duration, then pad-repeat/trim to pad_to seconds (=adjust_time_length).
        x = np.asarray(x, dtype=np.float32)
        if cfg.norm in ("car", "car_z"):
            x = x - x.mean(axis=1, keepdims=True)
        x = _bandpass(x, cfg.fs, cfg.l_freq, cfg.h_freq)
        x = _notch(x, cfg.fs, cfg.notch_freq)
        n_dst = int(round(x.shape[-1] * cfg.target_fs / cfg.fs))   # true-duration resample
        x = resample(x, n_dst, axis=-1)
        if cfg.norm in ("z_score", "car_z"):
            mu = x.mean(axis=-1, keepdims=True); sd = x.std(axis=-1, keepdims=True)
            x = (x - mu) / (sd + 1e-8)
        else:
            x = x / scale_divisor
        x = _pad_or_trim_repeat(x.astype(np.float32), pad_to * cfg.target_fs)
        n, ch, _ = x.shape
        return x.reshape(n, ch, pad_to, 200)
    src_len = int(cfg.seconds * cfg.fs)
    if x.shape[-1] < src_len:
        raise ValueError(
            f"{cfg.source} has only {x.shape[-1]} samples, need {src_len} "
            f"for {cfg.seconds}s at {cfg.fs}Hz"
        )
    x = x[:, :, :src_len].astype(np.float32, copy=False)
    if cfg.norm in ("car", "car_z"):        # spatial common-average reference
        x = x - x.mean(axis=1, keepdims=True)
    elif cfg.norm not in ("none", "None", None, "z_score"):
        raise ValueError(f"Unsupported norm: {cfg.norm}")
    x = _bandpass(x, cfg.fs, cfg.l_freq, cfg.h_freq)
    x = _notch(x, cfg.fs, cfg.notch_freq)
    x = resample(x, cfg.seconds * cfg.target_fs, axis=-1)
    if cfg.norm in ("z_score", "car_z"):    # per-channel temporal z-score (last step)
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        x = (x - mu) / (sd + 1e-8)
    else:                                    # amplitude scaling (÷scale; scale=1 => CAR-only)
        x = x / scale_divisor
    x = x.astype(np.float32)
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


def _split_indices(y_sub, args, seed):
    """random = stratified random (ours); fewshot_first = per-class take-first
    train_percentage in original order (benchmark split_dataset_fewshot: deterministic)."""
    idx = np.arange(len(y_sub))
    if args.split_method == "fewshot_first":
        tr, te = [], []
        for c in np.unique(y_sub):
            ci = np.where(y_sub == c)[0]              # original trial order within class
            k = max(1, int(len(ci) * args.train_percentage))
            tr.extend(ci[:k].tolist()); te.extend(ci[k:].tolist())
        return np.array(sorted(tr)), np.array(sorted(te))
    return train_test_split(
        idx, test_size=1.0 - args.train_percentage, stratify=y_sub, random_state=seed)


def run_subject(x_sub, y_sub, cfg, args, seed, device):
    set_seed(seed)
    train_idx, test_idx = _split_indices(y_sub, args, seed)
    x_train = torch.from_numpy(preprocess(x_sub[train_idx], cfg, args.scale_divisor)).float()
    x_test = torch.from_numpy(preprocess(x_sub[test_idx], cfg, args.scale_divisor)).float()
    y_train_np = y_sub[train_idx]
    y_train = torch.from_numpy(y_train_np).long()
    y_test = y_sub[test_idx]

    model = CBraModClassifier(
        cfg.num_classes,
        n_ch=x_train.shape[1],
        n_patch=x_train.shape[2],   # auto-adapts to pad_to_seconds
        dropout=args.dropout,
        head=args.head,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        eps=args.opt_eps,
    )
    scheduler = make_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr
    )
    weight = class_weight_tensor(y_train_np, cfg.num_classes, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
    )
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_test), args.batch_size):
            logits = model(x_test[i : i + args.batch_size].to(device))
            preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    metrics = {
        "acc": accuracy_score(y_test, pred) * 100.0,
        "bac": balanced_accuracy_score(y_test, pred),
        "kappa": cohen_kappa_score(y_test, pred),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def existing_keys(out_path):
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path)
    if df.empty:
        return set()
    return set(zip(df["dataset"], df["subject"], df["seed"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help=f"One of: {', '.join(DATASETS)}")
    parser.add_argument("--preset", choices=["native70", "paper80"], default="native70")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[666, 667, 668])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--train_percentage", type=float, default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--opt_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--scale_divisor", type=float, default=10.0)
    parser.add_argument("--head", choices=["mlp", "linear"], default="mlp",
                        help="mlp=official all_patch_reps 3-layer; linear=benchmark single Linear")
    parser.add_argument("--pad_to_seconds", type=int, default=0,
                        help="benchmark adjust_time_length: keep full signal, pad-repeat to N seconds")
    parser.add_argument("--pipeline", choices=["native", "benchmark"], default="native",
                        help="native=CAR->filter->resample; benchmark=resample->trim/pad->filter->CAR")
    parser.add_argument("--split_method", choices=["random", "fewshot_first"], default="random",
                        help="random=stratified random; fewshot_first=benchmark per-class take-first")

    parser.add_argument("--l_freq", type=float, default=None)
    parser.add_argument("--h_freq", type=float, default=None)
    parser.add_argument("--notch_freq", type=float, default=None)
    parser.add_argument("--norm_method", default=None,
                        choices=["car", "z_score", "car_z", "none"],
                        help="normalization; car+scale_divisor=1 == EEGFMBench CAR-only")
    return parser.parse_args()


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
        cfg,
        session=session,
        l_freq=cfg.l_freq if args.l_freq is None else args.l_freq,
        h_freq=cfg.h_freq if args.h_freq is None else args.h_freq,
        notch_freq=cfg.notch_freq if args.notch_freq is None else args.notch_freq,
        norm=cfg.norm if args.norm_method is None else args.norm_method,
        pad_to_seconds=args.pad_to_seconds,
        pipeline=args.pipeline,
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tag = f"{args.preset}_train{args.train_percentage:g}"
    if cfg.session:
        tag += f"_{cfg.session}"
    out_path = args.out or os.path.join(
        "results", "cbramod_native", f"{dataset}_{tag}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x, y, meta = load_dataset(cfg)
    subjects = args.subjects or sorted(meta["subject"].unique())
    completed = set() if args.overwrite else existing_keys(out_path)
    rows = []
    if os.path.exists(out_path) and not args.overwrite:
        rows = pd.read_csv(out_path).to_dict("records")

    print(
        f"[cfg] dataset={dataset} source={cfg.source} preset={args.preset} "
        f"train={args.train_percentage:g} session={cfg.session} "
        f"prep=CAR+{cfg.l_freq}-{cfg.h_freq}Hz"
        f"{'+notch'+str(cfg.notch_freq) if cfg.notch_freq else ''}"
        f"+resample{cfg.target_fs}+/ {args.scale_divisor:g} "
        f"epochs={args.epochs} bs={args.batch_size} lr={args.lr:g} "
        f"wd={args.weight_decay:g} gpu={device}",
        flush=True,
    )
    for seed in args.seeds:
        for subject in subjects:
            subject = int(subject)
            key = (dataset, subject, seed)
            if key in completed:
                print(f"[skip] {dataset} S{subject} seed{seed}", flush=True)
                continue
            mask = (meta["subject"] == subject).values
            metrics = run_subject(x[mask], y[mask], cfg, args, seed, device)
            row = {
                "dataset": dataset,
                "source": cfg.source,
                "preset": args.preset,
                "subject": subject,
                "seed": seed,
                "session": cfg.session or "",
                "train_percentage": args.train_percentage,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "label_smoothing": args.label_smoothing,
                "l_freq": cfg.l_freq,
                "h_freq": cfg.h_freq,
                "notch_freq": cfg.notch_freq or "",
                "acc": round(metrics["acc"], 4),
                "bac": round(metrics["bac"], 6),
                "kappa": round(metrics["kappa"], 6),
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(
                f"{dataset} S{subject} seed{seed} | "
                f"acc={metrics['acc']:.2f}% bac={metrics['bac']:.4f} "
                f"kappa={metrics['kappa']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        print(
            f"==== {dataset} {args.preset}: "
            f"acc={df.acc.mean():.2f}% bac={df.bac.mean():.4f} "
            f"kappa={df.kappa.mean():.4f} n={len(df)} -> {out_path} ====",
            flush=True,
        )


if __name__ == "__main__":
    main()
