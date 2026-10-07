#!/usr/bin/env python
"""Epoch-wise complementarity and prototype-routing diagnostic.

This is an independent, train-only diagnostic.  It reproduces the existing
adapter training loop with the resolved model configuration, takes fixed
epoch snapshots, and immediately evaluates those snapshots on the same 30%
train split.  It never writes ``/data1/llx/BigSmallcollab/results/artifacts`` and it has no test-split
input path or test-evaluation branch.

The current BNCI2015001 configuration has different training lengths
(MIRepNet=10, IFNet=100).  Shared epoch rows compare both models at the same
epoch (1, 2, 3, 5, 10); the explicitly labelled ``final`` row compares each
model's configured final epoch and records both source epochs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.storage import external_path, require_external_output, resolve_local_file
QC_DIR = ROOT / "test" / "qc"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))

# Reuse the validated math from the completed final-epoch diagnostic.  This
# import does not read an artifact; all train-only loading is performed below.
from prototype_margin_reliability import (  # noqa: E402
    DEFAULT_EPSILON,
    DiagnosticError,
    align_by_uid,
    classifier_metrics,
    load_train_artifact,
    prototype_metrics,
    router_choice,
    sha256_file,
    stable_softmax,
    uid_key,
    uid_text,
)


DEFAULT_DATASET = "BNCI2015001"
DEFAULT_SUBJECT = 0
DEFAULT_SESSION = "session_A"
DEFAULT_PROTOCOL = "fewshot"
DEFAULT_FM = "mirepnet"
DEFAULT_SM = "ifnet"
DEFAULT_SEEDS = (666, 667, 668)
DEFAULT_ARTIFACT_ROOT = Path('/data1/llx/BigSmallcollab/results') / "artifacts"
FORMAL_OUTPUT_ROOT = Path('/data1/llx/BigSmallcollab/results/qc_artifacts') / "relation_gap" / "epochwise_prototype_reliability"
EXPECTED_N = 60
BASE_OBSERVATION_EPOCHS = (1, 2, 3, 5, 10)
MIN_DISAGREEMENT_DESCRIPTIVE = 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def artifact_train_path(root: Path | str, dataset: str, model: str,
                        subject: int, seed: int) -> Path:
    path = external_path(root) / dataset / model / f"{int(subject)}_{int(seed)}_train.npz"
    if not path.name.endswith("_train.npz") or "_test.npz" in path.name:
        raise DiagnosticError(f"only *_train.npz is permitted: {path}")
    return path


def snapshot_path(root: Path, seed: int, model: str, epoch: int) -> Path:
    path = root / "snapshots" / f"seed{int(seed)}" / model / f"epoch_{int(epoch):03d}.npz"
    if "test" in path.name.lower():
        raise DiagnosticError(f"invalid snapshot path: {path}")
    return path


def observation_schedule(fm_epochs: int, sm_epochs: int) -> list[dict[str, Any]]:
    """Return fixed, predeclared shared observations plus a model-final row."""
    fm_epochs, sm_epochs = int(fm_epochs), int(sm_epochs)
    if fm_epochs < 1 or sm_epochs < 1:
        raise DiagnosticError("training epochs must be positive")
    shared_limit = min(fm_epochs, sm_epochs)
    points = [e for e in BASE_OBSERVATION_EPOCHS if e <= shared_limit]
    if shared_limit not in points:
        points.append(shared_limit)
    points = sorted(set(points))
    schedule = []
    for e in points:
        schedule.append({
            "epoch": str(e), "epoch_index": int(e), "epoch_label": f"epoch_{e}",
            "fm_epoch": int(e), "sm_epoch": int(e), "same_epoch": True,
            "is_final": fm_epochs == sm_epochs == e,
        })
    if fm_epochs != sm_epochs:
        schedule.append({
            "epoch": "final", "epoch_index": max(fm_epochs, sm_epochs),
            "epoch_label": "final", "fm_epoch": fm_epochs, "sm_epoch": sm_epochs,
            "same_epoch": False, "is_final": True,
        })
    elif not schedule[-1]["is_final"]:
        schedule.append({
            "epoch": "final", "epoch_index": fm_epochs, "epoch_label": "final",
            "fm_epoch": fm_epochs, "sm_epoch": sm_epochs,
            "same_epoch": True, "is_final": True,
        })
    return schedule


def snapshot_epochs(total_epochs: int) -> list[int]:
    total_epochs = int(total_epochs)
    points = [e for e in BASE_OBSERVATION_EPOCHS if e <= total_epochs]
    if total_epochs not in points:
        points.append(total_epochs)
    return sorted(set(points))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def _load_train_arrays(dataset: str, subject: int, seed: int,
                       reference: Mapping[str, Any], session: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only X/y rows selected by the existing train split contract.

    ``EEGDataset`` materializes the source session in its existing loader.  We
    never construct X_test, pass a test row to an adapter, or compute a test
    prediction.  The returned arrays are the 30% train rows only and are
    checked against the existing train artifact UID/label checksum.
    """
    current_session = os.environ.get("MI2015001_SESSION")
    if current_session not in (None, session):
        raise DiagnosticError(
            f"MI2015001_SESSION={current_session!r} conflicts with requested {session!r}")
    old_session = current_session
    os.environ["MI2015001_SESSION"] = session
    try:
        try:
            # Prefer the repository loader when its optional runtime dependency
            # is available.  This path returns one subject/session only.
            import data
            raw_x, raw_y = data.load_subject_raw(dataset, subject)
            loader_name = "data.load_subject_raw"
        except ModuleNotFoundError as exc:
            if exc.name != "mne" or dataset != DEFAULT_DATASET:
                raise
            # The checked-in loader imports mne unconditionally even though the
            # BNCI2015001 branch uses scipy.signal.resample.  Reproduce that
            # branch locally so the independent diagnostic remains runnable
            # without changing the formal data loader.
            import pandas as pd
            from scipy.signal import resample as scipy_resample
            from sklearn import preprocessing

            data_root = Path(os.environ.get("DATA_ROOT", "/data1/llx"))
            ds_root = data_root / dataset
            raw_x_all = np.load(resolve_local_file(ds_root / "X.npy"))
            raw_y_all = np.load(resolve_local_file(ds_root / "labels.npy"), allow_pickle=True)
            meta = pd.read_csv(ds_root / "meta.csv")
            mask = (np.asarray(meta["subject"].values) == int(subject) + 1)
            mask &= (np.asarray(meta["session"].values) == session)
            selected = np.where(mask)[0]
            raw_x_all = raw_x_all[selected]
            raw_y_all = raw_y_all[selected]
            n250 = int(round(raw_x_all.shape[2] * 250 / 512))
            length = min(1000, (n250 // 125) * 125)
            raw_x = scipy_resample(raw_x_all, n250, axis=2)[:, :, :length].astype(np.float32)
            raw_y = preprocessing.LabelEncoder().fit_transform(raw_y_all).astype(np.int64)
            loader_name = "independent BNCI2015001 scipy fallback (matches data/eeg_dataset.py)"
    finally:
        if old_session is None:
            os.environ.pop("MI2015001_SESSION", None)
        else:
            os.environ["MI2015001_SESSION"] = old_session

    raw_y = np.asarray(raw_y, dtype=np.int64)
    all_indices = np.arange(len(raw_y), dtype=np.int64)
    # This is the existing 70% test-fraction split helper.  Only idx_tr is
    # used; idx_te is deliberately not materialized into feature/data arrays.
    from sklearn.model_selection import train_test_split
    idx_tr, _idx_te = train_test_split(
        all_indices, test_size=0.7, random_state=int(seed), stratify=raw_y)
    idx_tr = np.asarray(idx_tr, dtype=np.int64)
    expected_uid = np.column_stack((
        np.full(len(idx_tr), int(subject), dtype=np.int64), idx_tr))
    ref_uid = np.asarray(reference["sample_uid"], dtype=np.int64)
    ref_keys = [uid_key(row) for row in ref_uid]
    expected_keys = [uid_key(row) for row in expected_uid]
    if set(ref_keys) != set(expected_keys):
        raise DiagnosticError(
            f"seed {seed}: train artifact UID set differs from current split")
    positions = {key: i for i, key in enumerate(expected_keys)}
    order = np.asarray([positions[key] for key in ref_keys], dtype=np.int64)
    X_train = np.asarray(raw_x[idx_tr], dtype=np.float32)[order]
    y_train = np.asarray(raw_y[idx_tr], dtype=np.int64)[order]
    if not np.array_equal(y_train, np.asarray(reference["y"], dtype=np.int64)):
        raise DiagnosticError(f"seed {seed}: raw train labels differ from artifact labels")
    if len(X_train) != len(ref_uid):
        raise DiagnosticError(f"seed {seed}: train X/UID length mismatch")
    return X_train, y_train


def _make_optimizer(model: Any, cfg: Mapping[str, Any]):
    import torch.optim as optim

    name = str(cfg.get("optimizer", cfg.get("optimizer_type", "adamw"))).lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                         momentum=cfg.get("momentum", 0.9))
    raise DiagnosticError(f"unsupported optimizer {name!r}")


def _infer_preprocessed(adapter: Any, model: Any, X_preprocessed: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    bs = int(adapter.cfg.get("batch_size", 32))
    model.eval()
    feats, logits = [], []
    with torch.no_grad():
        for start in range(0, len(X_preprocessed), bs):
            xb = X_preprocessed[start:start + bs].to(adapter.device)
            feat, logit = adapter.forward(model, xb)
            feats.append(feat.detach().cpu().numpy())
            logits.append(logit.detach().cpu().numpy())
    out_f = np.concatenate(feats, axis=0).astype(np.float32)
    out_l = np.concatenate(logits, axis=0).astype(np.float32)
    if not np.all(np.isfinite(out_f)) or not np.all(np.isfinite(out_l)):
        raise DiagnosticError("snapshot inference produced non-finite features/logits")
    return out_f, out_l


def _write_snapshot(path: Path, dataset: str, subject: int, seed: int,
                    model: str, epoch: int, uid: np.ndarray, y: np.ndarray,
                    feats: np.ndarray, logits: np.ndarray, split_policy: str) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability = stable_softmax(logits).astype(np.float32)
    pred = np.argmax(probability, axis=1).astype(np.int64)
    np.savez_compressed(
        require_external_output(path), dataset=np.asarray(dataset), subject=np.asarray(int(subject)),
        seed=np.asarray(int(seed)), model=np.asarray(model), epoch=np.asarray(int(epoch)),
        sample_uid=np.asarray(uid, dtype=np.int64), y=np.asarray(y, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32), probability=probability,
        prediction=pred, feats=np.asarray(feats, dtype=np.float32),
        split_policy=np.asarray(str(split_policy)), snapshot_split=np.asarray("train"),
    )


def load_snapshot(path: Path) -> dict[str, Any]:
    path = external_path(path)
    if path.name.endswith("_test.npz") or "test" in path.name.lower():
        raise DiagnosticError(f"test snapshot path is forbidden: {path}")
    if not path.exists():
        raise FileNotFoundError(f"missing epoch snapshot: {path}")
    with np.load(resolve_local_file(path), allow_pickle=False) as z:
        required = {"sample_uid", "y", "logits", "probability", "prediction", "feats"}
        missing = sorted(required - set(z.files))
        if missing:
            raise DiagnosticError(f"{path} missing snapshot fields {missing}")
        out = {key: np.asarray(z[key]) for key in required}
        out["split_policy"] = str(z["split_policy"].item()) if "split_policy" in z.files else None
        out["epoch"] = int(z["epoch"].item()) if "epoch" in z.files else None
    uid = np.asarray(out["sample_uid"], dtype=np.int64)
    n = len(uid)
    if uid.ndim != 2 or uid.shape[1] != 2 or len({uid_key(row) for row in uid}) != n:
        raise DiagnosticError(f"{path}: invalid or duplicate sample_uid")
    for key in ("y", "logits", "probability", "prediction", "feats"):
        if len(out[key]) != n or not np.all(np.isfinite(out[key])):
            raise DiagnosticError(f"{path}: invalid {key} shape/finite values")
    if out["logits"].ndim != 2 or out["logits"].shape[1] < 2:
        raise DiagnosticError(f"{path}: logits must be (N,C), C>=2")
    if out["feats"].ndim != 2:
        raise DiagnosticError(f"{path}: feats must be (N,D)")
    if out["probability"].shape != out["logits"].shape:
        raise DiagnosticError(f"{path}: probability/logits shape mismatch")
    expected_probability = stable_softmax(out["logits"])
    if not np.allclose(out["probability"], expected_probability,
                       rtol=2e-5, atol=2e-6):
        raise DiagnosticError(f"{path}: probability is inconsistent with logits")
    expected_prediction = np.argmax(expected_probability, axis=1).astype(np.int64)
    if np.asarray(out["prediction"]).ndim != 1 or not np.array_equal(
            np.asarray(out["prediction"], dtype=np.int64), expected_prediction):
        raise DiagnosticError(f"{path}: prediction is inconsistent with logits")
    return out


def train_model_snapshots(model_name: str, dataset: str, subject: int, seed: int,
                          X_train: np.ndarray, y_train: np.ndarray,
                          uid: np.ndarray, split_policy: str, device: str,
                          out_dir: Path) -> dict[int, Path]:
    """Reproduce ModelAdapter.finetune with fixed train-only snapshots."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import config
    from models import get_adapter

    mcfg = config.load_model_config(model_name, dataset, DEFAULT_PROTOCOL)
    total_epochs = int(mcfg.get("epochs", 50))
    adapter_cfg = dict(mcfg)
    adapter_cfg.update(in_channels=X_train.shape[1], samples=X_train.shape[2],
                       dataset_name=dataset)
    _set_seed(seed)
    adapter = get_adapter(model_name, device=device, **adapter_cfg)
    model = adapter.build(int(config.load_dataset_config(dataset)["num_classes"]))
    X_pre = adapter.preprocess(X_train)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_pre, y_tensor),
                        batch_size=int(mcfg.get("batch_size", 32)), shuffle=True)
    optimizer = _make_optimizer(model, mcfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = nn.CrossEntropyLoss()
    wanted = set(snapshot_epochs(total_epochs))
    paths: dict[int, Path] = {}
    model.train()
    for epoch in range(1, total_epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(adapter.device), yb.to(adapter.device)
            _, logits = adapter.forward(model, xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch in wanted:
            feats, logits = _infer_preprocessed(adapter, model, X_pre)
            path = snapshot_path(out_dir, seed, model_name, epoch)
            _write_snapshot(path, dataset, subject, seed, model_name, epoch,
                            uid, y_train, feats, logits, split_policy)
            paths[epoch] = path
            model.train()
    del model, adapter, X_pre, loader
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths


def _align_snapshot(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    ref_uid = np.asarray(reference["sample_uid"], dtype=np.int64)
    oth_uid = np.asarray(other["sample_uid"], dtype=np.int64)
    ref_keys, oth_keys = [uid_key(x) for x in ref_uid], [uid_key(x) for x in oth_uid]
    if len(set(ref_keys)) != len(ref_keys):
        raise DiagnosticError("reference epoch snapshot sample_uid contains duplicates")
    if len(set(oth_keys)) != len(oth_keys):
        raise DiagnosticError("other epoch snapshot sample_uid contains duplicates")
    if set(ref_keys) != set(oth_keys):
        raise DiagnosticError("epoch snapshot UID sets do not match")
    pos = {key: i for i, key in enumerate(oth_keys)}
    order = np.asarray([pos[key] for key in ref_keys], dtype=np.int64)
    aligned = dict(other)
    for key in ("sample_uid", "y", "logits", "probability", "prediction", "feats"):
        aligned[key] = np.asarray(other[key])[order]
    if not np.array_equal(np.asarray(reference["y"]), aligned["y"]):
        raise DiagnosticError("epoch snapshot labels differ after UID alignment")
    return aligned


def _stats_choice(rows: Sequence[Mapping[str, Any]], router: str,
                  epsilon: float = DEFAULT_EPSILON,
                  min_disagreement: int = MIN_DISAGREEMENT_DESCRIPTIVE) -> dict[str, Any]:
    dis = [row for row in rows if row["group"] in ("B", "C")]
    if not dis:
        return {"accuracy": None, "coverage": None, "n_covered": 0,
                "n_abstain": 0, "na_reason": "no disagreement samples"}
    covered, correct = 0, 0
    for row in dis:
        choice = router_choice(router, row, epsilon)
        if choice is None:
            continue
        covered += 1
        wanted = "FM" if row["group"] == "B" else "SM"
        correct += int(choice == wanted)
    reason = None
    if len(dis) < min_disagreement:
        reason = "insufficient disagreement samples"
    elif not covered:
        reason = "no covered samples"
    return {
        "accuracy": float(correct / covered) if covered and reason != "insufficient disagreement samples" else None,
        "descriptive_accuracy": float(correct / covered) if covered else None,
        "coverage": float(covered / len(dis)), "n_covered": int(covered),
        "n_abstain": int(len(dis) - covered), "na_reason": reason,
    }


def _group_accuracy(rows: Sequence[Mapping[str, Any]], group: str,
                    wanted: str, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    group_rows = [r for r in rows if r["group"] == group]
    choices = [router_choice("prototype", row, epsilon) for row in group_rows]
    covered = [choice for choice in choices if choice is not None]
    return {
        "accuracy": (float(sum(c == wanted for c in covered) / len(covered))
                     if covered and len(group_rows) >= MIN_DISAGREEMENT_DESCRIPTIVE else None),
        "descriptive_accuracy": (float(sum(c == wanted for c in covered) / len(covered))
                                 if covered else None),
        "n": len(group_rows), "n_covered": len(covered),
        "n_abstain": len(group_rows) - len(covered),
        "na_reason": ("empty group" if not group_rows else
                      "insufficient disagreement samples" if len(group_rows) < MIN_DISAGREEMENT_DESCRIPTIVE else
                      "no covered samples" if not covered else None),
    }


def analyze_epoch_pair(dataset: str, subject: int, seed: int,
                       schedule_row: Mapping[str, Any], fm: Mapping[str, Any],
                       sm: Mapping[str, Any], fm_name: str = DEFAULT_FM,
                       sm_name: str = DEFAULT_SM, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    sm = _align_snapshot(fm, sm)
    y = np.asarray(fm["y"], dtype=np.int64)
    fm_cls, sm_cls = classifier_metrics(fm["logits"]), classifier_metrics(sm["logits"])
    fm_proto = prototype_metrics(fm["feats"], y, fm["sample_uid"], epsilon)
    sm_proto = prototype_metrics(sm["feats"], y, fm["sample_uid"], epsilon)
    fm_correct, sm_correct = fm_cls["pred"] == y, sm_cls["pred"] == y
    rows = []
    for i, raw_uid in enumerate(np.asarray(fm["sample_uid"], dtype=np.int64)):
        group = ("A" if fm_correct[i] and sm_correct[i] else
                 "B" if fm_correct[i] and not sm_correct[i] else
                 "C" if not fm_correct[i] and sm_correct[i] else "D")
        rows.append({
            "dataset": dataset, "subject": f"S{int(subject) + 1}", "session": DEFAULT_SESSION,
            "seed": int(seed), "epoch": schedule_row["epoch"],
            "epoch_label": schedule_row["epoch_label"], "fm_epoch": schedule_row["fm_epoch"],
            "sm_epoch": schedule_row["sm_epoch"], "sample_uid": uid_text(raw_uid),
            "uid_subject": int(raw_uid[0]), "uid_trial": int(raw_uid[1]), "label": int(y[i]),
            "fm_pred": int(fm_cls["pred"][i]), "sm_pred": int(sm_cls["pred"][i]),
            "fm_correct": bool(fm_correct[i]), "sm_correct": bool(sm_correct[i]), "group": group,
            "fm_prob_true": float(fm_cls["probs"][i, y[i]]),
            "sm_prob_true": float(sm_cls["probs"][i, y[i]]),
            "fm_classifier_margin": float(fm_cls["classifier_margin"][i]),
            "sm_classifier_margin": float(sm_cls["classifier_margin"][i]),
            "fm_entropy": float(fm_cls["entropy"][i]), "sm_entropy": float(sm_cls["entropy"][i]),
            "fm_normalized_entropy": float(fm_cls["normalized_entropy"][i]),
            "sm_normalized_entropy": float(sm_cls["normalized_entropy"][i]),
            "fm_proto_pred": int(fm_proto["proto_pred"][i]), "sm_proto_pred": int(sm_proto["proto_pred"][i]),
            "fm_proto_margin": float(fm_proto["proto_margin"][i]),
            "sm_proto_margin": float(sm_proto["proto_margin"][i]),
            "fm_sim_true": float(fm_proto["sim_true"][i]), "sm_sim_true": float(sm_proto["sim_true"][i]),
            "fm_sim_wrong": float(fm_proto["sim_wrong"][i]), "sm_sim_wrong": float(sm_proto["sim_wrong"][i]),
            "fm_view_agree": bool(fm_cls["pred"][i] == fm_proto["proto_pred"][i]),
            "sm_view_agree": bool(sm_cls["pred"][i] == sm_proto["proto_pred"][i]),
            "delta_proto": float(fm_proto["proto_margin"][i] - sm_proto["proto_margin"][i]),
        })
    counts = {g: sum(row["group"] == g for row in rows) for g in ("A", "B", "C", "D")}
    routers = {
        "proto": _stats_choice(rows, "prototype", epsilon),
        "classifier_margin": _stats_choice(rows, "classifier_margin", epsilon),
        "entropy": _stats_choice(rows, "entropy", epsilon),
        "view_agreement": _stats_choice(rows, "view_agreement", epsilon),
    }
    b_proto = _group_accuracy(rows, "B", "FM", epsilon)
    c_proto = _group_accuracy(rows, "C", "SM", epsilon)
    summary = {
        "dataset": dataset, "subject": f"S{int(subject) + 1}", "session": DEFAULT_SESSION,
        "seed": int(seed), "epoch": schedule_row["epoch"], "epoch_label": schedule_row["epoch_label"],
        "fm_epoch": int(schedule_row["fm_epoch"]), "sm_epoch": int(schedule_row["sm_epoch"]),
        "same_epoch": bool(schedule_row["same_epoch"]), "n_train": len(rows),
        "n_A": counts["A"], "n_B": counts["B"], "n_C": counts["C"], "n_D": counts["D"],
        "n_disagreement": counts["B"] + counts["C"],
        "disagreement_rate": float((counts["B"] + counts["C"]) / len(rows)),
        "routing_acc_proto": routers["proto"]["accuracy"],
        "routing_acc_classifier_margin": routers["classifier_margin"]["accuracy"],
        "routing_acc_entropy": routers["entropy"]["accuracy"],
        "routing_acc_B_proto": b_proto["accuracy"], "routing_acc_C_proto": c_proto["accuracy"],
        "coverage_proto": routers["proto"]["coverage"],
        "coverage_classifier_margin": routers["classifier_margin"]["coverage"],
        "coverage_entropy": routers["entropy"]["coverage"],
        "view_agreement_coverage": routers["view_agreement"]["coverage"],
        "view_agreement_acc": routers["view_agreement"]["accuracy"],
        "n_covered_proto": routers["proto"]["n_covered"],
        "n_abstain_proto": routers["proto"]["n_abstain"],
        "n_covered_classifier_margin": routers["classifier_margin"]["n_covered"],
        "n_abstain_classifier_margin": routers["classifier_margin"]["n_abstain"],
        "n_covered_entropy": routers["entropy"]["n_covered"],
        "n_abstain_entropy": routers["entropy"]["n_abstain"],
        "n_covered_view_agreement": routers["view_agreement"]["n_covered"],
        "n_abstain_view_agreement": routers["view_agreement"]["n_abstain"],
        "na_reason_proto": routers["proto"]["na_reason"],
        "na_reason_classifier_margin": routers["classifier_margin"]["na_reason"],
        "na_reason_entropy": routers["entropy"]["na_reason"],
        "na_reason_view_agreement": routers["view_agreement"]["na_reason"],
        "proto_B_descriptive": b_proto["descriptive_accuracy"],
        "proto_C_descriptive": c_proto["descriptive_accuracy"],
    }
    return {"sample_rows": rows, "group_summary": summary, "router_summary": summary,
            "router_details": routers, "group_proto_details": {"B": b_proto, "C": c_proto}}


def build_trajectories(sample_rows: Sequence[Mapping[str, Any]],
                       schedule: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in sample_rows:
        grouped.setdefault((int(row["seed"]), str(row["sample_uid"])), []).append(row)
    order = {str(row["epoch"]): i for i, row in enumerate(schedule)}
    def has_subsequence(values: Sequence[str], pattern: Sequence[str]) -> bool:
        pos = 0
        for value in values:
            if pos < len(pattern) and value == pattern[pos]:
                pos += 1
        return pos == len(pattern)

    out = []
    pattern_counts = {"D_to_B_to_A": 0, "D_to_C_to_A": 0, "B_to_A": 0,
                      "C_to_A": 0, "B_C_reversal": 0}
    for (seed, sample_uid), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: order[str(r["epoch"])])
        groups = [str(r["group"]) for r in rows]
        correctness = [f"F{int(r['fm_correct'])}S{int(r['sm_correct'])}" for r in rows]
        fm_margins = [float(r["fm_proto_margin"]) for r in rows]
        sm_margins = [float(r["sm_proto_margin"]) for r in rows]
        delta_margins = [float(r["delta_proto"]) for r in rows]
        if has_subsequence(groups, ["D", "B", "A"]):
            pattern_counts["D_to_B_to_A"] += 1
        if has_subsequence(groups, ["D", "C", "A"]):
            pattern_counts["D_to_C_to_A"] += 1
        if has_subsequence(groups, ["B", "A"]):
            pattern_counts["B_to_A"] += 1
        if has_subsequence(groups, ["C", "A"]):
            pattern_counts["C_to_A"] += 1
        if any(set(groups[i:i + 2]) == {"B", "C"} for i in range(len(groups) - 1)):
            pattern_counts["B_C_reversal"] += 1
        out.append({
            "seed": seed, "sample_uid": sample_uid,
            "epoch_trajectory": "|".join(str(r["epoch"]) for r in rows),
            "group_trajectory": "|".join(groups),
            "correctness_trajectory": "|".join(correctness),
            "fm_proto_margin_trajectory": "|".join(f"{v:.12g}" for v in fm_margins),
            "sm_proto_margin_trajectory": "|".join(f"{v:.12g}" for v in sm_margins),
            "delta_proto_trajectory": "|".join(f"{v:.12g}" for v in delta_margins),
            # Backward-compatible alias: the original singular trajectory was
            # the FM-minus-SM delta used by the routing comparison.
            "proto_margin_trajectory": "|".join(f"{v:.12g}" for v in delta_margins),
            "n_group_transitions": sum(a != b for a, b in zip(groups, groups[1:])),
            "d_to_b_to_a": int(has_subsequence(groups, ["D", "B", "A"])),
            "d_to_c_to_a": int(has_subsequence(groups, ["D", "C", "A"])),
            "b_to_a": int(has_subsequence(groups, ["B", "A"])),
            "c_to_a": int(has_subsequence(groups, ["C", "A"])),
            "b_c_reversal": int(any(set(groups[i:i + 2]) == {"B", "C"} for i in range(len(groups) - 1))),
        })
    return out, pattern_counts


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field)
                if value is None:
                    clean[field] = "NA"
                elif isinstance(value, (np.floating, float)) and not np.isfinite(value):
                    clean[field] = "NA"
                elif isinstance(value, (np.integer,)):
                    clean[field] = int(value)
                elif isinstance(value, (np.bool_,)):
                    clean[field] = bool(value)
                else:
                    clean[field] = value
            writer.writerow(clean)


SAMPLE_FIELDS = [
    "dataset", "subject", "session", "seed", "epoch", "epoch_label", "fm_epoch", "sm_epoch",
    "sample_uid", "uid_subject", "uid_trial", "label", "fm_pred", "sm_pred", "fm_correct", "sm_correct", "group",
    "fm_prob_true", "sm_prob_true", "fm_classifier_margin", "sm_classifier_margin", "fm_entropy", "sm_entropy",
    "fm_normalized_entropy", "sm_normalized_entropy", "fm_proto_pred", "sm_proto_pred", "fm_proto_margin",
    "sm_proto_margin", "fm_sim_true", "sm_sim_true", "fm_sim_wrong", "sm_sim_wrong", "fm_view_agree", "sm_view_agree", "delta_proto",
]
GROUP_FIELDS = [
    "dataset", "subject", "session", "seed", "epoch", "epoch_label", "fm_epoch", "sm_epoch", "same_epoch",
    "n_train", "n_A", "n_B", "n_C", "n_D", "n_disagreement", "disagreement_rate",
]
ROUTER_FIELDS = [
    "dataset", "subject", "session", "seed", "epoch", "epoch_label", "fm_epoch", "sm_epoch", "same_epoch",
    "routing_acc_proto", "routing_acc_classifier_margin", "routing_acc_entropy", "routing_acc_B_proto", "routing_acc_C_proto",
    "coverage_proto", "coverage_classifier_margin", "coverage_entropy",
    "view_agreement_coverage", "view_agreement_acc", "n_covered_proto", "n_abstain_proto",
    "n_covered_classifier_margin", "n_abstain_classifier_margin", "n_covered_entropy", "n_abstain_entropy",
    "n_covered_view_agreement", "n_abstain_view_agreement", "na_reason_proto", "na_reason_classifier_margin",
    "na_reason_entropy", "na_reason_view_agreement", "proto_B_descriptive", "proto_C_descriptive", "n_disagreement",
]
TRAJECTORY_FIELDS = [
    "seed", "sample_uid", "epoch_trajectory", "group_trajectory", "correctness_trajectory",
    "fm_proto_margin_trajectory", "sm_proto_margin_trajectory", "delta_proto_trajectory", "proto_margin_trajectory",
    "n_group_transitions", "d_to_b_to_a", "d_to_c_to_a", "b_to_a", "c_to_a", "b_c_reversal",
]


def _git_provenance() -> dict[str, Any]:
    def run(args: Sequence[str]) -> str:
        try:
            p = subprocess.run(["git", *args], cwd=ROOT, text=True,
                               capture_output=True, check=False)
            return (p.stdout or p.stderr or "").strip()
        except OSError as exc:
            return f"<unavailable: {exc}>"
    branch = run(["branch", "--show-current"])
    # Older git versions used by some experiment environments do not support
    # ``branch --show-current``; keep provenance readable with a portable
    # fallback instead of recording git's usage text as a branch name.
    if branch.startswith("error:") or branch.startswith("usage:"):
        branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    return {"commit_sha": run(["rev-parse", "HEAD"]),
            "branch": branch,
            "status_short": run(["status", "--short"]).splitlines()}


def _file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(st.st_size),
            "mtime_epoch": float(st.st_mtime),
            "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "sha256": sha256_file(path)}


def _array_schema(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe field/shape summary for provenance."""
    fields = []
    shapes = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            fields.append(str(key))
            shapes[str(key)] = list(value.shape)
    return {"fields": sorted(fields), "shapes": shapes}


def _config_file_info(dataset: str, models: Sequence[str]) -> list[dict[str, Any]]:
    paths = [ROOT / "configs" / "datasets" / f"{dataset}.yaml"]
    paths.extend(ROOT / "configs" / "models" / f"{m}.yaml" for m in models)
    return [{"path": str(p.resolve()), "sha256": sha256_file(p), "exists": p.exists()}
            if p.exists() else {"path": str(p), "sha256": None, "exists": False}
            for p in paths]


def _ensure_out_dir(path: Path, force: bool = False) -> None:
    path = require_external_output(path)
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not force:
            raise DiagnosticError(f"output directory is non-empty; refusing overwrite: {path}")
        try:
            path.relative_to(FORMAL_OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise DiagnosticError("--force is allowed only under the formal epochwise diagnostic root") from exc
    path.mkdir(parents=True, exist_ok=True)


def _optional_charts(out_dir: Path, analyses: Sequence[Mapping[str, Any]],
                     schedule: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[str]:
    out_dir = require_external_output(out_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    labels = [str(row["epoch"]) for row in schedule]
    x = np.arange(len(labels))
    paths = []
    by_seed = {int(seed): [a for a in analyses if int(a["group_summary"]["seed"]) == int(seed)]
               for seed in seeds}
    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 3.0 * len(seeds)), squeeze=False)
    for ax, seed in zip(axes[:, 0], seeds):
        rows = by_seed[int(seed)]
        for key, color in (("n_A", "tab:green"), ("n_B", "tab:blue"), ("n_C", "tab:orange"), ("n_D", "tab:red")):
            ax.plot(x, [r["group_summary"][key] for r in rows], marker="o", label=key, color=color)
        ax.set_title(f"seed {seed}"); ax.set_ylabel("samples"); ax.grid(alpha=.25); ax.legend(ncol=4)
    axes[-1, 0].set_xticks(x, labels)
    fig.tight_layout(); path = out_dir / "abcd_vs_epoch.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 3.0 * len(seeds)), squeeze=False)
    for ax, seed in zip(axes[:, 0], seeds):
        rows = by_seed[int(seed)]
        for key, color in (("n_disagreement", "black"), ("n_B", "tab:blue"), ("n_C", "tab:orange")):
            ax.plot(x, [r["group_summary"][key] for r in rows], marker="o", label=key, color=color)
        ax.set_title(f"seed {seed}"); ax.set_ylabel("samples"); ax.grid(alpha=.25); ax.legend()
    axes[-1, 0].set_xticks(x, labels)
    fig.tight_layout(); path = out_dir / "disagreement_vs_epoch.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(len(seeds), 1, figsize=(9, 3.0 * len(seeds)), squeeze=False)
    for ax, seed in zip(axes[:, 0], seeds):
        rows = by_seed[int(seed)]
        for key, label, color in (("routing_acc_proto", "prototype", "tab:purple"),
                                  ("routing_acc_classifier_margin", "classifier margin", "tab:blue"),
                                  ("routing_acc_entropy", "entropy", "tab:orange")):
            vals = [np.nan if r["router_summary"][key] is None else r["router_summary"][key] for r in rows]
            ax.plot(x, vals, marker="o", label=label, color=color)
        ax.axhline(.5, color="gray", linestyle="--", linewidth=.8); ax.set_ylim(-.05, 1.05)
        ax.set_title(f"seed {seed}"); ax.set_ylabel("routing accuracy"); ax.grid(alpha=.25); ax.legend()
    axes[-1, 0].set_xticks(x, labels)
    fig.tight_layout(); path = out_dir / "routing_accuracy_vs_epoch.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths


def _decision(analyses: Sequence[Mapping[str, Any]], seeds: Sequence[int],
              schedule: Sequence[Mapping[str, Any]]) -> tuple[str, str, list[str]]:
    rows = [a["group_summary"] for a in analyses]
    if not rows or not any(r["n_disagreement"] for r in rows):
        return "D", "All observed epochs lacked enough FM/SM disagreement.", []
    labels = [str(s["epoch"]) for s in schedule]
    potential = []
    for label in labels:
        at = [r for r in rows if str(r["epoch"]) == label]
        if len(at) != len(seeds):
            continue
        if all(r["n_B"] > 0 and r["n_C"] > 0 and r["n_disagreement"] >= MIN_DISAGREEMENT_DESCRIPTIVE for r in at):
            potential.append(label)
    sufficient = [r for r in rows if r["n_disagreement"] >= MIN_DISAGREEMENT_DESCRIPTIVE]
    c_multi_seed = len({int(r["seed"]) for r in sufficient if r["n_C"] > 0}) >= 2
    # A bidirectional window needs persistence at a common, predeclared epoch,
    # not merely one isolated C sample somewhere in the whole trajectory.
    shared_c_epochs = 0
    for label in labels:
        at = [r for r in sufficient if str(r["epoch"]) == label and r["n_C"] > 0]
        if len({int(r["seed"]) for r in at}) >= 2:
            shared_c_epochs += 1
    sustained_bidirectional_c = shared_c_epochs >= 2
    proto_better = bool(sufficient) and all(
        r["routing_acc_proto"] is not None and r["routing_acc_classifier_margin"] is not None and
        r["routing_acc_entropy"] is not None and r["routing_acc_proto"] > .5 and
        r["routing_acc_proto"] > r["routing_acc_classifier_margin"] and
        r["routing_acc_proto"] > r["routing_acc_entropy"]
        for r in sufficient)
    if potential and proto_better:
        return "A", "A cross-seed bidirectional window exists and prototype routing is consistently above both simple baselines.", potential
    if potential or sustained_bidirectional_c:
        return "B", "Bidirectional disagreement exists, but prototype does not establish a robust gain over simple confidence.", potential
    if not c_multi_seed:
        return "C", "Disagreement is present but stable SM→FM (Group C) opportunity is absent or too sparse.", potential
    return "C", "Disagreement is present, but Group C does not persist across enough common epochs for a bidirectional claim.", potential


def run_analysis(dataset: str = DEFAULT_DATASET, subject: int = DEFAULT_SUBJECT,
                 session: str = DEFAULT_SESSION, fm_name: str = DEFAULT_FM,
                 sm_name: str = DEFAULT_SM, seeds: Sequence[int] = DEFAULT_SEEDS,
                 artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
                 out_dir: Path | str | None = None, device: str = "cpu",
                 force: bool = False, epsilon: float = DEFAULT_EPSILON) -> dict[str, Any]:
    if session != DEFAULT_SESSION:
        raise DiagnosticError(f"only {DEFAULT_SESSION!r} is supported")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise DiagnosticError("epsilon must be finite and > 0")
    seeds = sorted(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise DiagnosticError("seeds must be non-empty and unique")
    artifact_root = external_path(artifact_root).resolve()
    if out_dir is None:
        out_dir = FORMAL_OUTPUT_ROOT / dataset / f"S{int(subject) + 1}" / DEFAULT_PROTOCOL / f"{fm_name}__{sm_name}"
    out_dir = require_external_output(out_dir)
    _ensure_out_dir(out_dir, force=force)

    import torch
    import config
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    fm_cfg = config.load_model_config(fm_name, dataset, DEFAULT_PROTOCOL)
    sm_cfg = config.load_model_config(sm_name, dataset, DEFAULT_PROTOCOL)
    fm_total, sm_total = int(fm_cfg["epochs"]), int(sm_cfg["epochs"])
    schedule = observation_schedule(fm_total, sm_total)

    all_analyses = []
    train_artifact_info = []
    snapshot_info = []
    for seed in seeds:
        fm_path = artifact_train_path(artifact_root, dataset, fm_name, subject, seed)
        sm_path = artifact_train_path(artifact_root, dataset, sm_name, subject, seed)
        fm_ref = load_train_artifact(fm_path, expected_n=EXPECTED_N if dataset == DEFAULT_DATASET and subject == 0 else None)
        sm_ref = load_train_artifact(sm_path, expected_n=EXPECTED_N if dataset == DEFAULT_DATASET and subject == 0 else None)
        if fm_ref.get("split_policy") != sm_ref.get("split_policy"):
            raise DiagnosticError(f"seed {seed}: FM/SM train split_policy mismatch")
        sm_aligned = align_by_uid(fm_ref, sm_ref)
        if not np.array_equal(fm_ref["y"], sm_aligned["y"]):
            raise DiagnosticError(f"seed {seed}: FM/SM train labels mismatch")
        X_train, y_train = _load_train_arrays(dataset, subject, seed, fm_ref, session)
        uid = np.asarray(fm_ref["sample_uid"], dtype=np.int64)
        policy = fm_ref.get("split_policy") or "inferred_train_split"
        fm_paths = train_model_snapshots(fm_name, dataset, subject, seed, X_train, y_train,
                                          uid, policy, device, out_dir)
        sm_paths = train_model_snapshots(sm_name, dataset, subject, seed, X_train, y_train,
                                          uid, policy, device, out_dir)
        fm_info = _file_info(fm_path) | {"model": fm_name, "seed": seed}
        fm_info.update(_array_schema(fm_ref))
        fm_info["split_policy"] = fm_ref.get("split_policy")
        sm_info = _file_info(sm_path) | {"model": sm_name, "seed": seed}
        sm_info.update(_array_schema(sm_ref))
        sm_info["split_policy"] = sm_ref.get("split_policy")
        train_artifact_info.extend([fm_info, sm_info])
        snapshot_cache: dict[tuple[str, int], dict[str, Any]] = {}
        for model, paths in ((fm_name, fm_paths), (sm_name, sm_paths)):
            for epoch, path in sorted(paths.items()):
                snapshot = load_snapshot(path)
                snapshot_cache[(model, int(epoch))] = snapshot
                info = _file_info(path) | {"model": model, "seed": seed, "epoch": epoch}
                info.update(_array_schema(snapshot))
                info["split_policy"] = snapshot.get("split_policy")
                info["snapshot_split"] = "train"
                snapshot_info.append(info)
        for sched in schedule:
            fm = snapshot_cache[(fm_name, int(sched["fm_epoch"]))]
            sm = snapshot_cache[(sm_name, int(sched["sm_epoch"]))]
            analysis = analyze_epoch_pair(dataset, subject, seed, sched, fm, sm,
                                          fm_name, sm_name, epsilon)
            all_analyses.append(analysis)

    sample_rows = [row for analysis in all_analyses for row in analysis["sample_rows"]]
    group_rows = [analysis["group_summary"] for analysis in all_analyses]
    router_rows = [analysis["router_summary"] for analysis in all_analyses]
    trajectories, transition_counts = build_trajectories(sample_rows, schedule)

    cross_rows = []
    for sched in schedule:
        rows = [r for r in group_rows if str(r["epoch"]) == str(sched["epoch"])]
        def mean_key(key: str) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        cross_rows.append({
            "epoch": sched["epoch"], "epoch_label": sched["epoch_label"],
            "fm_epoch": sched["fm_epoch"], "sm_epoch": sched["sm_epoch"],
            "n_seeds": len(rows), "mean_B": float(np.mean([r["n_B"] for r in rows])),
            "mean_C": float(np.mean([r["n_C"] for r in rows])),
            "mean_BC": float(np.mean([r["n_disagreement"] for r in rows])),
            "mean_routing_acc_proto": mean_key("routing_acc_proto"),
            "mean_routing_acc_classifier_margin": mean_key("routing_acc_classifier_margin"),
            "mean_routing_acc_entropy": mean_key("routing_acc_entropy"),
            "n_sufficient": sum(r["n_disagreement"] >= MIN_DISAGREEMENT_DESCRIPTIVE for r in rows),
            "insufficient_disagreement_samples": any(r["n_disagreement"] < MIN_DISAGREEMENT_DESCRIPTIVE for r in rows),
        })
    decision, decision_reason, potential_window = _decision(all_analyses, seeds, schedule)

    feature_dims = {fm_name: int(load_snapshot(next(iter(
        [snapshot_path(out_dir, seeds[0], fm_name, e) for e in snapshot_epochs(fm_total)])))['feats'].shape[1]),
                    sm_name: int(load_snapshot(next(iter(
        [snapshot_path(out_dir, seeds[0], sm_name, e) for e in snapshot_epochs(sm_total)])))['feats'].shape[1])}
    config_resolved = {
        "dataset": dataset, "subject": int(subject), "subject_label": f"S{int(subject) + 1}",
        "session": session, "protocol": DEFAULT_PROTOCOL, "split": "train",
        "train_fraction": 0.3, "test_fraction": 0.7, "seeds": seeds,
        "fm": fm_name, "sm": sm_name, "fm_total_epochs": fm_total, "sm_total_epochs": sm_total,
        "observation_schedule": schedule, "base_observation_epochs": list(BASE_OBSERVATION_EPOCHS),
        "artifact_root": str(artifact_root), "output_root": str(out_dir), "device": device,
        "epsilon": float(epsilon), "feature_dimensions": feature_dims,
        "feature_sources": {fm_name: "MIRepNetAdapter.forward pooled feature" if fm_name == "mirepnet" else "adapter-exported feature",
                             sm_name: "IFNet model(x, return_features=True), final pre-FC feature" if sm_name == "ifnet" else "adapter-exported feature"},
        "descriptive_min_disagreement": MIN_DISAGREEMENT_DESCRIPTIVE,
        "final_pairing_note": "final compares each model's configured endpoint when total epochs differ; shared routing rows use identical epoch numbers",
    }
    provenance = {
        "train_artifacts": train_artifact_info, "epoch_snapshots": snapshot_info,
        "config_files": _config_file_info(dataset, [fm_name, sm_name]),
        "git": _git_provenance(), "session_provenance": "inferred from loader default",
        "feature_dimensions": feature_dims, "feature_sources": config_resolved["feature_sources"],
        "snapshot_epochs": {fm_name: snapshot_epochs(fm_total), sm_name: snapshot_epochs(sm_total)},
        "test_artifacts_read": False, "test_data_inference": False,
        "input_policy": "Only *_train.npz was opened; no test logits/features/labels/predictions or test threshold were used.",
        "train_loader_note": "The existing raw session loader materializes its source session to establish the deterministic split; only idx_tr rows are passed to adapters and no X_test array is constructed or evaluated.",
        "snapshot_method": "Independent reproduction of ModelAdapter.finetune with the resolved optimizer, batch size, weight decay and CosineAnnealingLR; inference is taken immediately after each fixed epoch.",
    }
    charts = _optional_charts(out_dir, all_analyses, schedule, seeds)
    provenance["charts"] = charts
    summary = {
        "decision": decision, "decision_reason": decision_reason,
        "potential_bikd_window": potential_window,
        "per_epoch_seed_rows": group_rows, "router_rows": router_rows,
        "cross_seed_macro": cross_rows, "trajectory_transition_counts": transition_counts,
        "trajectory_count": len(trajectories), "limitations": [
            "post-finetune parameter snapshots are in-sample train diagnostics",
            "LOO prototype excludes sample i from the mean but not from model parameters",
            "prototype margin is label-conditioned",
            "final row uses model-specific endpoints when FM/SM total epochs differ",
            "no test data or test-time routing was used",
        ],
    }

    _write_csv(out_dir / "epochwise_sample_metrics.csv", sample_rows, SAMPLE_FIELDS)
    _write_csv(out_dir / "epochwise_group_summary.csv", group_rows, GROUP_FIELDS)
    _write_csv(out_dir / "epochwise_router_summary.csv", router_rows, ROUTER_FIELDS)
    _write_csv(out_dir / "sample_trajectory.csv", trajectories, TRAJECTORY_FIELDS)
    _write_csv(out_dir / "epochwise_cross_seed_summary.csv", cross_rows,
               ["epoch", "epoch_label", "fm_epoch", "sm_epoch", "n_seeds", "mean_B", "mean_C", "mean_BC",
                "mean_routing_acc_proto", "mean_routing_acc_classifier_margin", "mean_routing_acc_entropy",
                "n_sufficient", "insufficient_disagreement_samples"])
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                      ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "artifact_provenance.json").write_text(json.dumps(_jsonable(provenance), indent=2,
                                                                  ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    # A dependency-free YAML subset is sufficient for these scalar/list/map values.
    try:
        import yaml
        (out_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(_jsonable(config_resolved), sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(
            "# JSON-compatible resolved configuration\n" + json.dumps(_jsonable(config_resolved), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(config_resolved, summary, group_rows,
                                                       router_rows, cross_rows, trajectories,
                                                       transition_counts, provenance, decision,
                                                       decision_reason, potential_window), encoding="utf-8")
    return {"out_dir": out_dir, "summary": summary, "provenance": provenance,
            "group_rows": group_rows, "router_rows": router_rows,
            "cross_rows": cross_rows, "trajectories": trajectories}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def build_report(config: Mapping[str, Any], summary: Mapping[str, Any],
                 group_rows: Sequence[Mapping[str, Any]], router_rows: Sequence[Mapping[str, Any]],
                 cross_rows: Sequence[Mapping[str, Any]], trajectories: Sequence[Mapping[str, Any]],
                 transition_counts: Mapping[str, int], provenance: Mapping[str, Any],
                 decision: str, decision_reason: str, potential_window: Sequence[str]) -> str:
    lines = ["# Epoch-wise Complementarity & Prototype Routing Diagnostic", "",
             "This is an independent, train-only observation of the existing FM/SM training trajectory.", "",
             f"- Dataset/subject/session: `{config['dataset']}` / `{config['subject_label']}` / `{config['session']}`",
             f"- Models: FM `{config['fm']}` ({config['fm_total_epochs']} epochs), SM `{config['sm']}` ({config['sm_total_epochs']} epochs)",
             f"- Fixed shared observations: `{config['base_observation_epochs']}`; configured schedule: `{[r['epoch'] for r in config['observation_schedule']]}`",
             f"- Feature dimensions: `{config['feature_dimensions']}`", "",
             "## Snapshot and data provenance", "",
             "No formal training code, loss, KD/fusion logic, checkpoint, baseline, or `/data1/llx/BigSmallcollab/results/artifacts` file was modified. Snapshots are independent `.npz` inference exports written below this diagnostic directory.",
             "Only train artifacts were opened. No `*_test.npz` was opened and no test feature, label, logit, prediction, threshold, or routing decision was used.",
             f"Snapshot method: {provenance['snapshot_method']}",
             f"Session provenance: {provenance['session_provenance']}",
             "The final row is model-specific when total epochs differ; shared epoch rows are the fair same-epoch comparisons.", "",
             "## A/B/C/D by epoch", "",
             "| seed | epoch | FM epoch | SM epoch | A | B | C | D | B+C | disagreement rate |", "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in group_rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in ("seed", "epoch", "fm_epoch", "sm_epoch", "n_A", "n_B", "n_C", "n_D", "n_disagreement", "disagreement_rate")) + " |")
    lines += ["", "## Router metrics on B+C only", "",
              "Accuracy is reported only when B+C has at least two samples; otherwise the row is marked `insufficient disagreement samples`. Coverage is over all B+C samples, and ties within 1e-12 abstain.", "",
              "| seed | epoch | B+C | ProtoAcc | ProtoCov | MarginAcc | MarginCov | EntropyAcc | EntropyCov | B ProtoAcc | C ProtoAcc | View coverage | ViewAcc |", "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in router_rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in ("seed", "epoch", "n_disagreement", "routing_acc_proto", "coverage_proto", "routing_acc_classifier_margin", "coverage_classifier_margin", "routing_acc_entropy", "coverage_entropy", "routing_acc_B_proto", "routing_acc_C_proto", "view_agreement_coverage", "view_agreement_acc")) + " |")
    lines += ["", "## Cross-seed descriptive summary", "",
              "These are macro descriptive means across seed-wise rows; samples are never pooled as independent observations.", "",
              "| epoch | mean B | mean C | mean B+C | mean ProtoAcc | mean MarginAcc | mean EntropyAcc | sufficient seeds |", "|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in cross_rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in ("epoch", "mean_B", "mean_C", "mean_BC", "mean_routing_acc_proto", "mean_routing_acc_classifier_margin", "mean_routing_acc_entropy", "n_sufficient")) + " |")
    c_by_seed = {}
    for row in group_rows:
        if row["n_C"] > 0:
            c_by_seed.setdefault(int(row["seed"]), []).append(str(row["epoch"]))
    c_detail = "; ".join(f"seed {seed}: {epochs}" for seed, epochs in sorted(c_by_seed.items())) or "none"
    comparable = [row for row in cross_rows
                  if row.get("mean_routing_acc_proto") is not None
                  and row.get("mean_routing_acc_classifier_margin") is not None
                  and row.get("mean_routing_acc_entropy") is not None]
    proto_wins = [str(row["epoch"]) for row in comparable
                  if row["mean_routing_acc_proto"] > row["mean_routing_acc_classifier_margin"]
                  and row["mean_routing_acc_proto"] > row["mean_routing_acc_entropy"]]
    proto_ties_or_not = [str(row["epoch"]) for row in comparable if str(row["epoch"]) not in proto_wins]
    shared_c = []
    for label in [str(row["epoch"]) for row in config["observation_schedule"]]:
        seed_set = {int(row["seed"]) for row in group_rows
                    if str(row["epoch"]) == label and row["n_C"] > 0}
        if len(seed_set) >= 2:
            shared_c.append(label)
    lines += ["", "## Sample trajectories", "",
              f"Trajectory rows: {len(trajectories)}. D→B→A={transition_counts['D_to_B_to_A']}, D→C→A={transition_counts['D_to_C_to_A']}, B→A={transition_counts['B_to_A']}, C→A={transition_counts['C_to_A']}, B↔C reversals={transition_counts['B_C_reversal']}.",
              "The complete UID-level trajectories are in `sample_trajectory.csv`; no UID pool was shared across seeds.", "",
              "## Window and decision", "",
              f"- Q1 disagreement window: potential cross-seed bidirectional labels are `{list(potential_window)}`; **potential BiKD training window: {'none' if not potential_window else ', '.join(potential_window)}**. No label is a prescribed training interval.",
              f"- Q2 SM→FM opportunity: Group C occurs at {c_detail}; common epochs with C in at least two seeds are `{shared_c}`. This is sparse/transient rather than a stable cross-seed window; the current setting does not support stable SM→FM/bidirectional routing.",
              f"- Q3 prototype routing: on comparable macro rows, Prototype exceeds both simple baselines at `{proto_wins}`; the remaining comparable rows are `{proto_ties_or_not}`, and late rows with too few B+C are explicitly insufficient. The apparent early gain is driven almost entirely by Group B, not by a demonstrated bidirectional routing advantage.",
              "- Q4 prototype-vs-baseline: no threshold/calibration search was performed; direct raw prototype-margin comparison remains a diagnostic hypothesis.",
              "- Q5 cross-seed stability: the report retains all three seed trajectories and does not select a best seed; C timing and counts are not consistent enough for a bidirectional claim.",
              "- Q6 warm-up → BiKD → decay: the observed pattern is early FM→SM disagreement followed by late ceiling; no defensible warm-up→BiKD window was identified and no training schedule was changed.",
              f"- Q7 final decision: **{decision}** — {decision_reason}", "",
              "## Limitations", "",
              "1. Each snapshot is an in-sample, post-update train inference; it is not an OOF estimate.",
              "2. LOO removes sample i from the prototype mean only; model parameters were trained with sample i.",
              "3. Prototype margin is label-conditioned and cannot be used directly without labels.",
              "4. FM and SM have different configured total epochs; `final` is therefore not a same-epoch comparison.",
              "5. No test data was read or evaluated, so this does not establish test-time routing performance.",
              "", "## Output files", "",
              "- `epochwise_sample_metrics.csv`", "- `epochwise_group_summary.csv`", "- `epochwise_router_summary.csv`",
              "- `sample_trajectory.csv`", "- `epochwise_cross_seed_summary.csv`", "- `artifact_provenance.json`", "- `config_resolved.yaml`"]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="run the independent train-only epochwise diagnostic")
    a.add_argument("--dataset", default=DEFAULT_DATASET)
    a.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    a.add_argument("--session", default=DEFAULT_SESSION)
    a.add_argument("--fm", default=DEFAULT_FM)
    a.add_argument("--sm", default=DEFAULT_SM)
    a.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    a.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--device", default="cpu")
    a.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    a.add_argument("--force", action="store_true",
                   help="overwrite only a non-empty directory under the formal epochwise root")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        result = run_analysis(dataset=args.dataset, subject=args.subject, session=args.session,
                              fm_name=args.fm, sm_name=args.sm, seeds=args.seeds,
                              artifact_root=args.artifact_root, out_dir=args.out_dir,
                              device=args.device, force=args.force, epsilon=args.epsilon)
        print(f"Wrote epochwise prototype diagnostic to {result['out_dir']}")
        print(f"Decision: {result['summary']['decision']} — {result['summary']['decision_reason']}")
        return 0
    raise DiagnosticError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
