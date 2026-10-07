"""Teacher-correct full-loss masked KD+MI pilot.

This is an intentionally isolated runner.  It does not change the shared
``collab.distill`` training paths or any existing result.  The three new
conditions are trained on the same subject-wise few-shot split and use the
same IFNet initial state and DataLoader schedule:

``CE_TCORRECT_MASK``
    CE on teacher-correct samples after a complete batch forward.
``KD_MI_ALL``
    CE + 0.5 * 2^2 * KL + 0.1 * (-class-joint MI) on every sample.
``KD_MI_TCORRECT_MASK``
    The same three losses, all restricted to the teacher-correct samples.

The teacher artifact loader deliberately reads only logits, labels, UIDs and
split metadata.  Teacher features are never materialised by this pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from collab.distill import probability_mi_loss
from collab.seed import set_seed
from eval import metrics
from models import get_adapter
from experiments.distill.run_distill import (
    _capture_initial_state,
    _combined_hash,
    _sha256_array,
    _sha256_file,
    _sha256_state_dict,
    _sha256_uid_split,
    _session_default,
)
from experiments.storage import external_path, require_external_output, resolve_local_file


_ALL_DATASET_SUBJECTS = {
    "BNCI2014001": 9,
    "BNCI2014001-4": 9,
    "BNCI2014004": 9,
    "BNCI2015001": 12,
    "AlexMI": 8,
}
_DEFAULT_DATASETS = ["BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI"]
DATASETS = list(_DEFAULT_DATASETS)
DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
SEED = 666
VAL_SPLIT = 0.7
CONDITIONS = ("CE_TCORRECT_MASK", "KD_MI_ALL", "KD_MI_TCORRECT_MASK")
OLD_METHODS = ("Base", "KD_all", "CE_MI")
COMPARISONS = (
    ("CE_TCORRECT_MASK", "Base"),
    ("KD_MI_ALL", "Base"),
    ("KD_MI_TCORRECT_MASK", "KD_MI_ALL"),
    ("KD_MI_TCORRECT_MASK", "CE_TCORRECT_MASK"),
    ("KD_MI_TCORRECT_MASK", "KD_all"),
    ("KD_MI_TCORRECT_MASK", "CE_MI"),
)
DEFAULT_CONFIG = ROOT / "configs/experiments/distill_kd_mi_teacher_correct_mask_pilot.yaml"
DEFAULT_OUTPUT = Path("/data1/llx/BigSmallCollab_results/distill/kd_mi_teacher_correct_mask_pilot")
TOTAL_CSV = Path("/data1/llx/BigSmallCollab_results/distill/kd_mi_teacher_correct_mask_pilot.csv")
OLD_CSV = Path("/data1/llx/BigSmallCollab_results/distill/distill_mi.csv")
OLD_ROOT = Path("/data1/llx/BigSmallCollab_results/distill/distill_mi")
INIT_ARGS = SimpleNamespace(
    epochs=None, lr=None, weight_decay=None, batch_size=None,
)


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_bytes(path: Path, payload: bytes):
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str):
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_json(path: Path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _atomic_yaml(path: Path, value):
    _atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _atomic_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    out = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    _atomic_text(path, out.getvalue())


def _read_csv(path: Path):
    path = external_path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _git_snapshot():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT,
                                           text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:  # pragma: no cover - provenance fallback
            return f"<git unavailable: {exc}>"
    return {"commit": run("rev-parse", "HEAD"), "status_short": run("status", "--short")}


def _gpu_snapshot(requested_gpu):
    info = {
        "physical_gpu_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_logical_gpu": requested_gpu,
        "device": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "name": None,
    }
    if torch.cuda.is_available():
        info["device"] = f"cuda:{requested_gpu}"
        try:
            info["name"] = torch.cuda.get_device_name(requested_gpu)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
        info["nvidia_smi"] = out.strip().splitlines()
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    return info


def _validate_config(cfg):
    global DATASETS, DATASET_SUBJECTS
    configured_datasets = list(cfg.get("datasets", []))
    if configured_datasets not in (_DEFAULT_DATASETS, ["BNCI2014001-4"]):
        raise ValueError("dataset scope must be the original four-dataset pilot or the isolated BNCI2014001-4 supplement")
    DATASETS = configured_datasets
    DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
    expected = {
        "datasets": DATASETS,
        "protocol": "fewshot",
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "teacher": "mirepnet",
        "student": "ifnet",
        "epochs": 100,
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "batch_size": 16,
        "scheduler": "CosineAnnealingLR",
        "temperature_kd": 2.0,
        "lam_kd": 0.5,
        "temperature_mi": 1.0,
        "lam_mi": 0.1,
        "mi_eps": 1e-8,
        "drop_last": False,
        "model_selection": "final_epoch",
        "conditions": list(CONDITIONS),
    }
    for key, value in expected.items():
        actual = cfg.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > 1e-12:
                raise ValueError(f"config {key} must be {value!r}, got {actual!r}")
        elif actual != value:
            raise ValueError(f"config {key} must be {value!r}, got {actual!r}")


def _unit_count():
    return sum(DATASET_SUBJECTS[name] for name in DATASETS)


def _run_count():
    return _unit_count() * len(CONDITIONS)


def _teacher_uid_alignment_hash(uid, labels):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(uid, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(labels, dtype="<i8").tobytes())
    return digest.hexdigest()


def _load_teacher_logits_only(path: Path):
    """Read only train logits, labels, UID and metadata; never touch ``feats``."""
    if not path.name.endswith("_train.npz"):
        raise ValueError(f"teacher path is not a train artifact: {path}")
    with np.load(resolve_local_file(path), allow_pickle=False) as archive:
        required = {"logits", "y", "sample_uid", "split_policy"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing required fields {sorted(missing)}")
        logits = np.asarray(archive["logits"])
        labels = np.asarray(archive["y"], dtype=np.int64)
        uid = np.asarray(archive["sample_uid"], dtype=np.int64)
        policy = str(archive["split_policy"].item())
    if logits.ndim != 2 or len(logits) != len(labels) or len(uid) != len(labels):
        raise ValueError(f"{path}: logits/y/sample_uid first dimensions differ")
    if not np.isfinite(logits).all():
        raise ValueError(f"{path}: logits contain NaN/Inf")
    if uid.ndim != 2 or uid.shape[1] != 2:
        raise ValueError(f"{path}: sample_uid must have shape (N,2)")
    if len({tuple(row) for row in uid.tolist()}) != len(uid):
        raise ValueError(f"{path}: duplicate sample_uid")
    return {"logits": logits.astype(np.float32), "y": labels,
            "sample_uid": uid, "split_policy": policy}


def _align_teacher(payload, uid_tr, y_tr, context):
    current_uid = np.asarray(uid_tr, dtype=np.int64)
    if current_uid.ndim != 2 or current_uid.shape[1] != 2:
        raise ValueError(f"{context}: current UID shape is invalid")
    if len({tuple(row) for row in current_uid.tolist()}) != len(current_uid):
        raise ValueError(f"{context}: current UID is not unique")
    teacher_uid = payload["sample_uid"]
    if set(map(tuple, teacher_uid.tolist())) != set(map(tuple, current_uid.tolist())):
        raise ValueError(f"{context}: teacher/current UID sets differ")
    index = {tuple(row): i for i, row in enumerate(teacher_uid.tolist())}
    order = np.asarray([index[tuple(row)] for row in current_uid.tolist()], dtype=np.int64)
    aligned_uid = teacher_uid[order]
    aligned_y = payload["y"][order]
    aligned_logits = payload["logits"][order]
    if not np.array_equal(aligned_uid, current_uid):
        raise ValueError(f"{context}: aligned UID order differs")
    if not np.array_equal(aligned_y, np.asarray(y_tr, dtype=np.int64)):
        raise ValueError(f"{context}: aligned labels differ")
    if payload["split_policy"] != "fewshot_stratified_random":
        raise ValueError(f"{context}: unexpected split_policy={payload['split_policy']!r}")
    return aligned_logits, aligned_y, aligned_uid


def _class_counts(labels, num_classes):
    count = Counter(int(x) for x in np.asarray(labels, dtype=np.int64).tolist())
    return {str(c): int(count.get(c, 0)) for c in range(num_classes)}


def _resolved_student_cfg(dataset):
    cfg = config.load_model_config("ifnet", dataset, "fewshot")
    expected = {"epochs": 100, "lr": 0.001, "weight_decay": 0.01,
                "batch_size": 16}
    for key, value in expected.items():
        if abs(float(cfg.get(key)) - value) > 1e-12:
            raise ValueError(f"IFNet config {dataset} {key}={cfg.get(key)!r}, expected {value!r}")
    return cfg


def _preflight_units(cfg):
    units = []
    for dataset in DATASETS:
        nc = int(config.load_dataset_config(dataset)["num_classes"])
        student_cfg = _resolved_student_cfg(dataset)
        for subject_index in range(DATASET_SUBJECTS[dataset]):
            Xtr, ytr, Xte, yte, uid_tr, uid_te = data.subject_split(
                dataset, subject_index, val_split=VAL_SPLIT, seed=SEED,
                return_uid=True)
            # ``_resolved_student_cfg`` validates the dataset-level training
            # settings, but the adapter also needs the current subject's
            # runtime input dimensions when it builds the model.  Inject
            # those dimensions into the per-unit copy used by
            # ``_train_condition``; otherwise IFNet receives ``None`` for
            # ``in_channels``/``samples`` and fails during construction.
            student_cfg = dict(student_cfg)
            student_cfg.update(
                in_channels=int(Xtr.shape[1]),
                samples=int(Xtr.shape[2]),
                dataset_name=dataset,
            )
            artifact_path = Path(configured_artifact_path(cfg, dataset, subject_index))
            if not artifact_path.exists():
                raise FileNotFoundError(artifact_path)
            payload = _load_teacher_logits_only(artifact_path)
            teacher_logits, aligned_y, aligned_uid = _align_teacher(
                payload, uid_tr, ytr,
                f"{dataset} S{subject_index + 1} seed{SEED}")
            mask = teacher_logits.argmax(axis=1) == aligned_y
            state = _capture_initial_state(
                INIT_ARGS, dataset, "ifnet", Xtr, nc, "cpu", SEED)
            initial_hash = _sha256_state_dict(state)
            adapter_cpu = get_adapter("ifnet", device="cpu", **student_cfg)
            xtr_pre = adapter_cpu.preprocess(Xtr)
            xte_pre = adapter_cpu.preprocess(Xte)
            wrong_idx = np.flatnonzero(~mask)
            inventory = {
                "dataset": dataset,
                "subject": subject_index + 1,
                "subject_index": subject_index,
                "session": _session_default(dataset),
                "seed": SEED,
                "train_count": int(len(ytr)),
                "teacher_correct_count": int(mask.sum()),
                "teacher_wrong_count": int((~mask).sum()),
                "mask_keep_rate": float(mask.mean()),
                "teacher_wrong_rate": float((~mask).mean()),
                "teacher_wrong_uid_json": _json(aligned_uid[wrong_idx].astype(int).tolist()),
                "teacher_wrong_label_json": _json(aligned_y[wrong_idx].astype(int).tolist()),
                "masked_class_counts_json": _json(_class_counts(aligned_y[wrong_idx], nc)),
                "kept_class_counts_json": _json(_class_counts(aligned_y[mask], nc)),
                "train_uid_hash": _sha256_array(uid_tr),
                "teacher_uid_alignment_hash": _teacher_uid_alignment_hash(aligned_uid, aligned_y),
                "teacher_artifact_path": str(artifact_path.resolve()),
                "teacher_artifact_sha256": _sha256_file(artifact_path),
                "alignment_status": "pass",
            }
            units.append({
                "dataset": dataset, "subject": subject_index + 1,
                "subject_index": subject_index, "seed": SEED, "num_classes": nc,
                "session": _session_default(dataset), "Xtr": Xtr, "ytr": ytr,
                "Xte": Xte, "yte": yte, "uid_tr": uid_tr, "uid_te": uid_te,
                "teacher_logits": teacher_logits, "teacher_y": aligned_y,
                "keep": mask, "initial_state": state,
                "initial_state_hash": initial_hash, "student_cfg": student_cfg,
                "xtr_pre": xtr_pre, "xte_pre": xte_pre,
                "artifact_path": artifact_path,
                "artifact_sha256": inventory["teacher_artifact_sha256"],
                "teacher_uid_alignment_hash": inventory["teacher_uid_alignment_hash"],
                "inventory": inventory,
                "train_uid_hash": _sha256_array(uid_tr),
                "test_uid_hash": _sha256_array(uid_te),
                "split_uid_hash": _sha256_uid_split(uid_tr, uid_te),
            })
    return units


def configured_artifact_path(cfg, dataset, subject_index):
    root = external_path(cfg.get("artifact_root", "/data1/llx/BigSmallCollab_results/artifacts"))
    if not root.is_absolute():
        root = ROOT / root
    return root / dataset / "mirepnet" / f"{subject_index}_{SEED}_train.npz"


def _validate_old_controls(units):
    rows = _read_csv(OLD_CSV)
    selected = [row for row in rows if int(row.get("seed", -1)) == SEED]
    expected_keys = {
        (u["dataset"], u["subject_index"], SEED, method)
        for u in units for method in OLD_METHODS
    }
    old = {}
    errors = []
    for row in selected:
        key = (row.get("dataset"), int(row.get("key", -1)), int(row.get("seed", -1)), row.get("method"))
        if key in old:
            errors.append(f"duplicate old control {key}")
        old[key] = row
    if len(selected) != len(expected_keys) or set(old) != expected_keys:
        errors.append(f"old seed666 controls expected {len(expected_keys)}, got {len(selected)}")
    reused = []
    param_expect = {
        "Base": (0.0, 0.0, "none"),
        "KD_all": (0.5, 0.0, "all"),
        "CE_MI": (0.0, 0.1, "none"),
    }
    for unit in units:
        for method in OLD_METHODS:
            key = (unit["dataset"], unit["subject_index"], SEED, method)
            row = old.get(key)
            if row is None:
                continue
            check = {
                "subject": int(row["subject"]) == unit["subject"],
                "session": row["session"] == unit["session"],
                "protocol": row["protocol"] == "fewshot",
                "teacher": row["teacher"] == "mirepnet",
                "student": row["student"] == "ifnet",
                "n_train": int(row["n_train"]) == len(unit["ytr"]),
                "n_test": int(row["n_test"]) == len(unit["yte"]),
                "train_uid_hash": row["train_uid_hash"] == unit["train_uid_hash"],
                "test_uid_hash": row["test_uid_hash"] == unit["test_uid_hash"],
                "split_uid_hash": row["split_uid_hash"] == unit["split_uid_hash"],
                "initial_state_hash": row["initial_state_hash"] == unit["initial_state_hash"],
                "teacher_artifact_sha256": row["teacher_artifact_sha256"] == unit["artifact_sha256"],
                "teacher_path_train": row["teacher_artifact_path"].endswith("_train.npz") and "_test" not in row["teacher_artifact_path"],
                "alignment": json.loads(row["teacher_train_uid_alignment"]).get("status") == "pass",
                "epochs": int(float(row["student_epochs"])) == 100,
                "optimizer": row["optimizer"].lower() == "adamw",
                "lr": abs(float(row["student_lr"]) - 0.001) < 1e-12,
                "weight_decay": abs(float(row["student_weight_decay"]) - 0.01) < 1e-12,
                "batch_size": int(float(row["student_batch_size"])) == 16,
                "scheduler": row["scheduler"] == "CosineAnnealingLR",
                "complete": row["failure_status"] == "complete",
                "finite": all(_finite(row[c]) for c in ("test_accuracy", "test_balanced_accuracy", "test_kappa")),
            }
            exp_kd, exp_mi, exp_weight = param_expect[method]
            check.update({
                "lam_kd": abs(float(row["lam_kd"]) - exp_kd) < 1e-12,
                "lam_mi": abs(float(row["lam_mi"]) - exp_mi) < 1e-12,
                "weight_mode": row["weight_mode"] == exp_weight,
                "temperature": abs(float(row["temperature"]) - 2.0) < 1e-12,
            })
            checkpoint = external_path(row["checkpoint_path"])
            history = external_path(row["history_path"])
            prediction = external_path(row["prediction_path"])
            check.update({"checkpoint": checkpoint.exists(), "history": history.exists(),
                          "prediction": prediction.exists()})
            if checkpoint.exists():
                try:
                    payload = torch.load(resolve_local_file(checkpoint), map_location="cpu", weights_only=False)
                    check["checkpoint_complete"] = (
                        bool(payload.get("complete")) and int(payload.get("epochs")) == 100
                        and payload.get("initial_state_hash") == unit["initial_state_hash"]
                        and payload.get("split_uid_hash") == unit["split_uid_hash"])
                except Exception:
                    check["checkpoint_complete"] = False
            else:
                check["checkpoint_complete"] = False
            if history.exists():
                try:
                    hist = json.loads(resolve_local_file(history).read_text())
                    check["history_100"] = isinstance(hist, list) and len(hist) == 100
                    old_hashes = json.loads(row.get("batch_order_hashes", "[]"))
                    check["batch_hashes_100"] = len(old_hashes) == 100 and len(hist) == 100
                    check["batch_hashes_match"] = check["batch_hashes_100"] and all(
                        hist[i].get("batch_order_hash") == old_hashes[i] for i in range(100))
                    check["batch_combined"] = check["batch_hashes_match"] and row["batch_order_hash"] == _combined_hash(old_hashes)
                except Exception:
                    check.update({"history_100": False, "batch_hashes_100": False,
                                  "batch_hashes_match": False, "batch_combined": False})
            else:
                check.update({"history_100": False, "batch_hashes_100": False,
                              "batch_hashes_match": False, "batch_combined": False})
            if prediction.exists():
                try:
                    with np.load(resolve_local_file(prediction), allow_pickle=False) as pred:
                        check["prediction_complete"] = (
                            pred["train_logits"].shape[0] == len(unit["ytr"])
                            and pred["test_logits"].shape[0] == len(unit["yte"])
                            and pred["train_preds"].shape[0] == len(unit["ytr"])
                            and pred["test_preds"].shape[0] == len(unit["yte"]))
                except Exception:
                    check["prediction_complete"] = False
            else:
                check["prediction_complete"] = False
            bad = [name for name, ok in check.items() if not ok]
            if bad:
                errors.append(f"{unit['dataset']} S{unit['subject']} {method}: {bad}")
            reused_row = dict(row)
            reused_row.update({
                "condition": method,
                "result_source": "reused",
                "subject_index": unit["subject_index"],
                "teacher_correct_count": unit["inventory"]["teacher_correct_count"],
                "teacher_wrong_count": unit["inventory"]["teacher_wrong_count"],
                "mask_keep_rate": unit["inventory"]["mask_keep_rate"],
                "teacher_wrong_rate": unit["inventory"]["teacher_wrong_rate"],
                "teacher_uid_alignment_hash": unit["teacher_uid_alignment_hash"],
            })
            reused.append(reused_row)
    if errors:
        raise RuntimeError("existing controls failed validation:\n" + "\n".join(errors))
    return reused


def batch_loss_components(student_logits, labels, teacher_logits, keep,
                          condition, lam_kd=0.5, temperature_kd=2.0,
                          lam_mi=0.1, mi_eps=1e-8):
    """Compute one selected batch objective without touching model state.

    ``student_logits`` must already come from a complete forward of the batch.
    A ``None`` return means the mask selected no samples, so callers must skip
    backward and optimizer.step.  With one selected sample MI is a connected
    zero, while CE and KD remain normal.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    labels = labels.long()
    keep = keep.bool()
    if condition == "KD_MI_ALL":
        selected = torch.ones(student_logits.shape[0], dtype=torch.bool,
                              device=student_logits.device)
    else:
        selected = keep.to(student_logits.device)
    n = int(selected.sum().item())
    if n == 0:
        return None
    selected_logits = student_logits[selected]
    selected_labels = labels[selected]
    ce_per_sample = F.cross_entropy(selected_logits, selected_labels, reduction="none")
    ce_loss = ce_per_sample.mean()
    total = ce_loss
    kd_loss = None
    mi_loss = None
    if condition != "CE_TCORRECT_MASK":
        selected_teacher = teacher_logits.to(student_logits.device)[selected].detach()
        teacher_prob_kd = F.softmax(selected_teacher / temperature_kd, dim=1).detach()
        kd_per_sample = F.kl_div(
            F.log_softmax(selected_logits / temperature_kd, dim=1),
            teacher_prob_kd, reduction="none").sum(dim=1)
        kd_loss = kd_per_sample.mean()
        total = total + lam_kd * (temperature_kd ** 2) * kd_loss
        teacher_prob_mi = F.softmax(selected_teacher, dim=1).detach()
        student_prob_mi = F.softmax(selected_logits, dim=1)
        if n >= 2:
            mi_loss = probability_mi_loss(teacher_prob_mi, student_prob_mi, eps=mi_eps)
        else:
            # Keep the zero MI contribution connected to student logits.
            mi_loss = selected_logits.sum() * 0.0
        total = total + lam_mi * mi_loss
    return {
        "total": total, "ce_loss": ce_loss, "kd_loss": kd_loss,
        "mi_loss": mi_loss, "selected": selected, "selected_count": n,
        "mi_valid": bool(condition != "CE_TCORRECT_MASK" and n >= 2),
    }


def forward_loss_step(adapter, model, optimizer, xb, yb, teacher_logits,
                      keep, condition, cfg):
    """Run the complete forward, then optionally backpropagate one batch.

    Keeping this small operation explicit makes the loss-only semantics
    testable: an all-masked batch has a forward (and therefore BatchNorm
    update) but no backward or optimizer step.
    """
    optimizer.zero_grad(set_to_none=True)
    _, logits = adapter.forward(model, xb)
    components = batch_loss_components(
        logits, yb, teacher_logits, keep, condition,
        lam_kd=float(cfg["lam_kd"]),
        temperature_kd=float(cfg["temperature_kd"]),
        lam_mi=float(cfg["lam_mi"]), mi_eps=float(cfg["mi_eps"]))
    if components is None:
        optimizer.zero_grad(set_to_none=True)
        return logits, None, False
    components["total"].backward()
    optimizer.step()
    return logits, components, True


def _inference(adapter, model, x, batch_size=256):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            _, logits = adapter.forward(model, x[start:start + batch_size].to(adapter.device))
            parts.append(logits.detach().cpu())
    return torch.cat(parts, dim=0).numpy()


def _atomic_torch_save(path: Path, payload):
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    torch.save(payload, require_external_output(tmp))
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays):
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    np.savez(require_external_output(tmp), **arrays)
    os.replace(tmp, path)


def _run_key(unit, condition):
    return f"{unit['dataset']}__S{unit['subject']}__seed{SEED}__{condition}"


def _train_condition(unit, condition, output_root: Path, cfg, device):
    started = time.time()
    set_seed(SEED)
    student_cfg = dict(unit["student_cfg"])
    adapter = get_adapter("ifnet", device=device, **student_cfg)
    model = adapter.build(unit["num_classes"])
    state = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in unit["initial_state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"initial IFNet state mismatch: missing={missing}, unexpected={unexpected}")
    y_tensor = torch.as_tensor(unit["ytr"], dtype=torch.long)
    teacher_tensor = torch.as_tensor(unit["teacher_logits"], dtype=torch.float32)
    keep_tensor = torch.as_tensor(unit["keep"], dtype=torch.bool)
    index_tensor = torch.arange(len(y_tensor), dtype=torch.long)
    dataset = TensorDataset(unit["xtr_pre"].cpu(), y_tensor, teacher_tensor,
                            keep_tensor, index_tensor)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
                        drop_last=False)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    batch_order_hashes = []
    history = []
    totals = Counter()
    ce_seen, kd_seen, mi_seen = set(), set(), set()
    global_ce_sum = global_kd_sum = global_mi_sum = global_total_sum = 0.0
    ce_exposures = kd_exposures = mi_exposures = 0
    mi_batch_count = 0
    step_count = 0
    all_masked_batches = 0
    mi_skipped_small = 0

    model.train()
    for epoch in range(int(cfg["epochs"])):
        model.train()
        epoch_hash = hashlib.sha256()
        epoch_ce_sum = epoch_kd_sum = epoch_mi_sum = epoch_total_sum = 0.0
        epoch_ce_n = epoch_kd_n = epoch_mi_n = epoch_steps = 0
        epoch_all_masked = epoch_mi_small = 0
        epoch_forward_batches = 0
        epoch_correct = 0
        epoch_count = 0
        for xb, yb, tb, keepb, ib in loader:
            batch_indices = ib.numpy().astype(np.int64, copy=False)
            epoch_hash.update(np.ascontiguousarray(unit["uid_tr"][batch_indices], dtype=np.int64).tobytes())
            epoch_forward_batches += 1
            xb = xb.to(device)
            yb = yb.to(device)
            tb = tb.to(device).detach()
            keepb = keepb.to(device)
            logits, components, did_step = forward_loss_step(
                adapter, model, optimizer, xb, yb, tb, keepb, condition, cfg)
            epoch_correct += int((logits.detach().argmax(dim=1) == yb).sum().item())
            epoch_count += int(yb.shape[0])
            if components is None:
                all_masked_batches += 1
                epoch_all_masked += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            selected_cpu = components["selected"].detach().cpu().numpy().astype(bool)
            selected_indices = batch_indices[selected_cpu]
            ce_seen.update(int(i) for i in selected_indices)
            ce_n = components["selected_count"]
            ce_value = float(components["ce_loss"].detach().item())
            epoch_ce_sum += ce_value * ce_n
            global_ce_sum += ce_value * ce_n
            ce_exposures += ce_n
            epoch_ce_n += ce_n
            if condition != "CE_TCORRECT_MASK":
                kd_seen.update(int(i) for i in selected_indices)
                kd_value = float(components["kd_loss"].detach().item())
                mi_value = float(components["mi_loss"].detach().item())
                epoch_kd_sum += kd_value * ce_n
                global_kd_sum += kd_value * ce_n
                kd_exposures += ce_n
                epoch_kd_n += ce_n
                if components["mi_valid"]:
                    mi_seen.update(int(i) for i in selected_indices)
                    epoch_mi_sum += mi_value
                    global_mi_sum += mi_value
                    mi_exposures += ce_n
                    mi_batch_count += 1
                    epoch_mi_n += 1
                else:
                    mi_skipped_small += 1
                    epoch_mi_small += 1
            objective_value = float(components["total"].detach().item())
            epoch_total_sum += objective_value
            global_total_sum += objective_value
            epoch_steps += 1
            step_count += 1
            if not did_step:
                raise RuntimeError("non-empty loss batch did not perform optimizer step")
        scheduler.step()
        batch_hash = epoch_hash.hexdigest()
        batch_order_hashes.append(batch_hash)
        history.append({
            "epoch": epoch + 1,
            "ce_loss": (epoch_ce_sum / epoch_ce_n) if epoch_ce_n else None,
            "kd_loss": (epoch_kd_sum / epoch_kd_n) if epoch_kd_n else None,
            "scaled_kd_contribution": ((float(cfg["lam_kd"]) * float(cfg["temperature_kd"]) ** 2
                                         * epoch_kd_sum / epoch_kd_n) if epoch_kd_n else None),
            "mi_loss": (epoch_mi_sum / epoch_mi_n) if epoch_mi_n else (0.0 if condition == "CE_TCORRECT_MASK" else None),
            "scaled_mi_contribution": ((float(cfg["lam_mi"]) * epoch_mi_sum / epoch_mi_n)
                                       if epoch_mi_n else (0.0 if condition == "CE_TCORRECT_MASK" else None)),
            "total_loss": (epoch_total_sum / epoch_steps) if epoch_steps else None,
            "train_accuracy_forward_all": epoch_correct / max(1, epoch_count) * 100.0,
            "effective_ce_sample_count": epoch_ce_n,
            "effective_kd_sample_count": epoch_kd_n,
            "effective_mi_sample_count": (epoch_ce_n if epoch_mi_n else 0),
            "all_masked_batch_count": epoch_all_masked,
            "mi_skipped_small_batch_count": epoch_mi_small,
            "forward_batch_count": epoch_forward_batches,
            "optimizer_step_count": epoch_steps,
            "batch_order_hash": batch_hash,
        })

    train_logits = _inference(adapter, model, unit["xtr_pre"])
    test_logits = _inference(adapter, model, unit["xte_pre"])
    train_preds = train_logits.argmax(axis=1)
    test_preds = test_logits.argmax(axis=1)
    test_metrics = metrics.evaluate(unit["yte"], test_preds)
    test_balanced = float(balanced_accuracy_score(unit["yte"], test_preds) * 100.0)
    train_kept_acc = (float((train_preds[unit["keep"]] == unit["ytr"][unit["keep"]]).mean() * 100.0)
                      if unit["keep"].any() else float("nan"))
    checkpoint_path = output_root / "checkpoints" / f"{unit['dataset']}__S{unit['subject']}__seed{SEED}__{condition}.pt"
    history_path = output_root / "training_history" / f"{unit['dataset']}__S{unit['subject']}__seed{SEED}__{condition}.json"
    prediction_path = output_root / "predictions" / f"{unit['dataset']}__S{unit['subject']}__seed{SEED}__{condition}.npz"
    run_key = _run_key(unit, condition)
    checkpoint_payload = {
        "complete": True, "run_key": run_key, "dataset": unit["dataset"],
        "subject": unit["subject"], "subject_index": unit["subject_index"],
        "seed": SEED, "condition": condition, "epochs": int(cfg["epochs"]),
        "initial_state_hash": unit["initial_state_hash"],
        "split_uid_hash": unit["split_uid_hash"],
        "teacher_uid_alignment_hash": unit["teacher_uid_alignment_hash"],
        "batch_order_hash": _combined_hash(batch_order_hashes),
        "batch_order_hashes": batch_order_hashes, "state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    _atomic_json(history_path, history)
    _atomic_npz(
        prediction_path,
        train_uid=np.asarray(unit["uid_tr"], dtype=np.int64),
        test_uid=np.asarray(unit["uid_te"], dtype=np.int64),
        train_y=np.asarray(unit["ytr"], dtype=np.int64),
        test_y=np.asarray(unit["yte"], dtype=np.int64),
        teacher_correct_keep=np.asarray(unit["keep"], dtype=np.bool_),
        train_logits=train_logits.astype(np.float32), test_logits=test_logits.astype(np.float32),
        train_preds=train_preds.astype(np.int64), test_preds=test_preds.astype(np.int64),
    )
    elapsed = time.time() - started
    row = {
        "dataset": unit["dataset"], "subject": unit["subject"],
        "subject_index": unit["subject_index"], "session": unit["session"],
        "seed": SEED, "protocol": "fewshot", "teacher": "mirepnet",
        "student": "ifnet", "condition": condition, "result_source": "new",
        "train_count": len(unit["ytr"]), "test_count": len(unit["yte"]),
        "teacher_correct_count": int(unit["keep"].sum()),
        "teacher_wrong_count": int((~unit["keep"]).sum()),
        "mask_keep_rate": float(unit["keep"].mean()),
        "teacher_wrong_rate": float((~unit["keep"]).mean()),
        "effective_ce_sample_count": ce_exposures,
        "effective_kd_sample_count": kd_exposures,
        "effective_mi_sample_count": mi_exposures,
        "effective_ce_unique_trial_count": len(ce_seen),
        "effective_kd_unique_trial_count": len(kd_seen),
        "effective_mi_unique_trial_count": len(mi_seen),
        "all_masked_batch_count": all_masked_batches,
        "mi_skipped_small_batch_count": mi_skipped_small,
        "mean_ce_loss": global_ce_sum / ce_exposures if ce_exposures else None,
        "mean_kd_loss": global_kd_sum / kd_exposures if kd_exposures else None,
        "mean_scaled_kd_contribution": (float(cfg["lam_kd"]) * float(cfg["temperature_kd"]) ** 2
                                         * global_kd_sum / kd_exposures) if kd_exposures else 0.0,
        "mean_mi_loss": global_mi_sum / mi_batch_count if mi_batch_count else (0.0 if condition == "CE_TCORRECT_MASK" else None),
        "mean_scaled_mi_contribution": (float(cfg["lam_mi"]) * global_mi_sum / mi_batch_count
                                         if mi_batch_count else (0.0 if condition == "CE_TCORRECT_MASK" else None)),
        "mean_total_loss": global_total_sum / step_count if step_count else None,
        "final_train_accuracy_on_all_samples": float((train_preds == unit["ytr"]).mean() * 100.0),
        "final_train_accuracy_on_kept_samples": train_kept_acc,
        "test_accuracy": float(test_metrics["acc"]),
        "test_balanced_accuracy": test_balanced,
        "test_kappa": float(test_metrics["kappa"]),
        "predicted_class_counts": _json(np.bincount(test_preds, minlength=unit["num_classes"]).tolist()),
        "collapse_flag": bool(len(np.unique(test_preds)) < 2),
        "runtime_seconds": elapsed,
        "train_uid_hash": unit["train_uid_hash"], "test_uid_hash": unit["test_uid_hash"],
        "split_uid_hash": unit["split_uid_hash"], "initial_state_hash": unit["initial_state_hash"],
        "batch_order_hash": _combined_hash(batch_order_hashes),
        "batch_order_hashes": _json(batch_order_hashes),
        "teacher_uid_alignment_hash": unit["teacher_uid_alignment_hash"],
        "teacher_artifact_path": str(unit["artifact_path"].resolve()),
        "teacher_artifact_sha256": unit["artifact_sha256"],
        "status": "complete", "failure_reason": "",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "history_path": str(history_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
    }
    numeric = [row["test_accuracy"], row["test_balanced_accuracy"], row["test_kappa"],
               row["mean_ce_loss"], row["mean_total_loss"]]
    if not all(_finite(x) for x in numeric):
        row["status"] = "invalid_metrics"
        row["failure_reason"] = "non-finite metric"
    return row


def _load_new_rows(path):
    return _read_csv(path)


def _new_row_valid(row, unit, output_root):
    if row.get("status") != "complete":
        return False
    if row.get("train_uid_hash") != unit["train_uid_hash"] or row.get("test_uid_hash") != unit["test_uid_hash"]:
        return False
    if row.get("split_uid_hash") != unit["split_uid_hash"] or row.get("initial_state_hash") != unit["initial_state_hash"]:
        return False
    if row.get("teacher_artifact_sha256") != unit["artifact_sha256"]:
        return False
    if not all(_finite(row.get(k)) for k in ("test_accuracy", "test_balanced_accuracy", "test_kappa")):
        return False
    try:
        hist = json.loads(resolve_local_file(row["history_path"]).read_text())
        if not isinstance(hist, list) or len(hist) != 100:
            return False
        payload = torch.load(resolve_local_file(row["checkpoint_path"]), map_location="cpu", weights_only=False)
        if not payload.get("complete") or len(payload.get("batch_order_hashes", [])) != 100:
            return False
        with np.load(resolve_local_file(row["prediction_path"]), allow_pickle=False) as pred:
            return pred["test_preds"].shape[0] == len(unit["yte"])
    except Exception:
        return False


def _write_manifest(output_root, units, result_rows):
    by_key = {_run_key(u, c): r for u in units for c in CONDITIONS
              for r in result_rows if r.get("dataset") == u["dataset"]
              and int(r.get("subject_index", -1)) == u["subject_index"]
              and r.get("condition") == c}
    rows = []
    for unit in units:
        for condition in CONDITIONS:
            key = _run_key(unit, condition)
            result = by_key.get(key)
            rows.append({
                "run_key": key, "dataset": unit["dataset"], "subject": unit["subject"],
                "subject_index": unit["subject_index"], "seed": SEED,
                "protocol": "fewshot", "teacher": "mirepnet", "student": "ifnet",
                "condition": condition,
                "status": result.get("status", "pending") if result else "pending",
                "failure_reason": result.get("failure_reason", "") if result else "",
            })
    _atomic_csv(output_root / "run_manifest.csv", rows,
                ["run_key", "dataset", "subject", "subject_index", "seed", "protocol",
                 "teacher", "student", "condition", "status", "failure_reason"])


def _validate_new_triplet(unit, rows):
    """Require all three new conditions to share every control hash."""
    matching = [r for r in rows
                if r.get("dataset") == unit["dataset"]
                and int(r.get("subject_index", -1)) == unit["subject_index"]
                and int(r.get("seed", -1)) == SEED
                and r.get("condition") in CONDITIONS
                and r.get("status") == "complete"]
    if len(matching) != len(CONDITIONS):
        return
    for field in ("train_uid_hash", "test_uid_hash", "split_uid_hash",
                  "initial_state_hash", "batch_order_hash",
                  "batch_order_hashes", "teacher_uid_alignment_hash",
                  "teacher_artifact_sha256"):
        if len({r.get(field) for r in matching}) != 1:
            raise RuntimeError(
                f"new-condition hash mismatch for {unit['dataset']} "
                f"S{unit['subject']}: {field}")


def _paired_bootstrap(diff, seed=666, n_boot=10000):
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    values = diff[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def _summary_rows(all_rows, output_root):
    # Convert the two sources to a consistent subject-level table.
    subject_rows = []
    for row in all_rows:
        subject_rows.append({
            "dataset": row["dataset"], "subject": int(row["subject"]),
            "subject_index": int(row["subject_index"]), "seed": SEED,
            "condition": row["condition"], "result_source": row["result_source"],
            "test_accuracy": float(row["test_accuracy"]),
            "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
            "test_kappa": float(row["test_kappa"]),
        })
    _atomic_csv(output_root / "results_per_subject.csv", subject_rows,
                ["dataset", "subject", "subject_index", "seed", "condition",
                 "result_source", "test_accuracy", "test_balanced_accuracy", "test_kappa"])
    dataset_rows = []
    for dataset in DATASETS:
        for condition in (*OLD_METHODS, *CONDITIONS):
            vals = [r for r in subject_rows if r["dataset"] == dataset and r["condition"] == condition]
            ba = np.asarray([r["test_balanced_accuracy"] for r in vals], dtype=float)
            acc = np.asarray([r["test_accuracy"] for r in vals], dtype=float)
            kap = np.asarray([r["test_kappa"] for r in vals], dtype=float)
            dataset_rows.append({
                "dataset": dataset, "condition": condition,
                "subject_count": len(vals), "balanced_accuracy_mean": float(ba.mean()) if len(ba) else float("nan"),
                "balanced_accuracy_sd": float(ba.std(ddof=1)) if len(ba) > 1 else float("nan"),
                "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
                "kappa_mean": float(kap.mean()) if len(kap) else float("nan"),
            })
    _atomic_csv(output_root / "results_per_dataset.csv", dataset_rows)
    paired_rows = []
    for dataset in DATASETS:
        piv = {(r["subject_index"], r["condition"]): r["test_balanced_accuracy"]
               for r in subject_rows if r["dataset"] == dataset}
        for method, baseline in COMPARISONS:
            common = sorted({s for s, c in piv if c == method} & {s for s, c in piv if c == baseline})
            diff = np.asarray([piv[(s, method)] - piv[(s, baseline)] for s in common], dtype=float)
            ci_low, ci_high = _paired_bootstrap(diff)
            try:
                from scipy.stats import wilcoxon
                pvalue = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                pvalue = float("nan")
            paired_rows.append({
                "dataset": dataset, "comparison": f"{method}-{baseline}",
                "method": method, "baseline": baseline, "n_subject": len(diff),
                "mean_delta_balanced_accuracy": float(diff.mean()) if len(diff) else float("nan"),
                "median_delta_balanced_accuracy": float(np.median(diff)) if len(diff) else float("nan"),
                "wins": int((diff > 1e-12).sum()), "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()), "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high, "wilcoxon_p": pvalue,
            })
    _atomic_csv(output_root / "paired_comparisons.csv", paired_rows)
    return subject_rows, dataset_rows, paired_rows


def _mechanism_stats(units, new_rows):
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {"rho": float("nan"), "pvalue": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    by_key = {(r["dataset"], int(r["subject_index"]), r["condition"]): float(r["test_balanced_accuracy"])
              for r in new_rows}
    x, y, ds_values = [], [], []
    for u in units:
        x.append(float(u["inventory"]["teacher_wrong_rate"]))
        y.append(by_key[(u["dataset"], u["subject_index"], "KD_MI_TCORRECT_MASK")] -
                 by_key[(u["dataset"], u["subject_index"], "KD_MI_ALL")])
        ds_values.append(u["dataset"])
    rho, pvalue = spearmanr(x, y)
    rng = np.random.RandomState(666)
    boot = []
    by_ds = {d: np.flatnonzero(np.asarray(ds_values) == d) for d in DATASETS}
    for _ in range(10000):
        idx = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in by_ds.values()])
        if len(set(idx.tolist())) < 2:
            continue
        boot.append(float(spearmanr(np.asarray(x)[idx], np.asarray(y)[idx]).statistic))
    ci = np.percentile(boot, [2.5, 97.5]).tolist() if boot else [float("nan"), float("nan")]
    return {"n": len(x), "rho": float(rho), "pvalue": float(pvalue), "ci95": [float(ci[0]), float(ci[1])]}


def _decision(units, new_rows, paired_rows):
    inventory_keep = np.asarray([u["inventory"]["mask_keep_rate"] for u in units], dtype=float)
    overall_keep = float(sum(u["inventory"]["teacher_correct_count"] for u in units) /
                         sum(u["inventory"]["train_count"] for u in units))
    all_keep_fraction = float(np.mean([u["inventory"]["teacher_wrong_count"] == 0 for u in units]))
    near_all = overall_keep >= 0.99 or all_keep_fraction >= 0.90
    by_key = {(r["dataset"], int(r["subject_index"]), r["condition"]): float(r["test_balanced_accuracy"])
              for r in new_rows}
    old = {(r["dataset"], int(r["subject_index"]), r["condition"]): float(r["test_balanced_accuracy"])
           for r in new_rows if r["result_source"] == "reused"}
    def overall_delta(a, b):
        values = []
        for u in units:
            ka = (u["dataset"], u["subject_index"], a)
            kb = (u["dataset"], u["subject_index"], b)
            source_b = by_key if b in CONDITIONS else old
            if ka in by_key and kb in source_b:
                values.append(by_key[ka] - source_b[kb])
        return float(np.mean(values)) if values else float("nan")
    d = {
        "ce_mask_minus_base": overall_delta("CE_TCORRECT_MASK", "Base"),
        "all_minus_base": overall_delta("KD_MI_ALL", "Base"),
        "masked_minus_all": overall_delta("KD_MI_TCORRECT_MASK", "KD_MI_ALL"),
        "masked_minus_ce_mask": overall_delta("KD_MI_TCORRECT_MASK", "CE_TCORRECT_MASK"),
        "masked_minus_kd_all": overall_delta("KD_MI_TCORRECT_MASK", "KD_all"),
        "masked_minus_ce_mi": overall_delta("KD_MI_TCORRECT_MASK", "CE_MI"),
    }
    # Add overall CIs from only paired subject units with both conditions.
    overall_ci = {}
    for a, b in COMPARISONS:
        diffs = []
        for u in units:
            ka = (u["dataset"], u["subject_index"], a)
            kb = (u["dataset"], u["subject_index"], b)
            source_b = by_key if b in CONDITIONS else old
            if ka in by_key and kb in source_b:
                diffs.append(by_key[ka] - source_b[kb])
        overall_ci[f"{a}-{b}"] = _paired_bootstrap(diffs) if diffs else (float("nan"), float("nan"))
    if near_all and np.isfinite(d["masked_minus_all"]) and abs(d["masked_minus_all"]) < 1.0 and overall_ci["KD_MI_TCORRECT_MASK-KD_MI_ALL"][0] <= 0 <= overall_ci["KD_MI_TCORRECT_MASK-KD_MI_ALL"][1]:
        label = "D"
    elif np.isfinite(d["ce_mask_minus_base"]) and d["ce_mask_minus_base"] < 0 and np.isfinite(d["masked_minus_all"]) and d["masked_minus_all"] < 0:
        label = "E"
    elif np.isfinite(d["ce_mask_minus_base"]) and np.isfinite(d["masked_minus_ce_mask"]) and np.isfinite(d["masked_minus_all"]) and d["ce_mask_minus_base"] > 0 and d["masked_minus_ce_mask"] > 0 and d["masked_minus_all"] > 0:
        label = "A"
    elif np.isfinite(d["ce_mask_minus_base"]) and np.isfinite(d["masked_minus_ce_mask"]) and d["ce_mask_minus_base"] > 0 and abs(d["masked_minus_ce_mask"]) < 1.0:
        label = "B"
    elif np.isfinite(d["all_minus_base"]) and np.isfinite(d["masked_minus_all"]) and d["all_minus_base"] > 0 and d["masked_minus_all"] <= 0:
        label = "C"
    else:
        label = "Mixed/Inconclusive"
    return {"label": label, "overall_keep_rate": overall_keep,
            "all_keep_unit_fraction": all_keep_fraction, "mask_near_all_one": near_all,
            "overall_deltas": d, "overall_ci95": {k: list(v) for k, v in overall_ci.items()}}


def _write_report(output_root, cfg, units, reused, new_rows, dataset_rows,
                  paired_rows, decision, mechanism, provenance):
    lines = [
        "# Teacher-Correct Full-Loss Masked KD+MI Pilot",
        "",
        f"- Decision: **{decision['label']}**",
        "- Teacher/student: MIRepNet → IFNet",
        "- Protocol: subject-wise few-shot; 30% train / 70% test within canonical session",
        f"- Seed: 666 only; {_unit_count()} experiment units; {_run_count()} new condition runs",
        "- No LOSO, K-fold, cross-subject training, QC filtering, or extra trial deletion.",
        ("- Existing `distill_mi` seed666 controls passed strict reuse validation."
         if reused else "- Prior Base/KD/CE_MI controls were not reused: the accuracy-only supplement rows lack checkpoint/history/prediction and split-hash provenance; control comparisons are unavailable."),
        "",
        "## Exact loss and mask",
        "",
        "After every complete IFNet batch forward, `keep = (teacher_argmax == label)` is applied to the per-sample CE, KD and MI inputs for the masked condition. Teacher-wrong samples therefore have zero direct loss gradient, but they still pass through IFNet in train mode and can update BatchNorm running statistics. This is loss-only masking, not sample deletion.",
        "",
        "- `CE_TCORRECT_MASK`: `mean(CE[keep])`.",
        "- `KD_MI_ALL`: `CE + 0.5 * 2^2 * mean(KL) + 0.1 * mean(-MI)` over all samples.",
        "- `KD_MI_TCORRECT_MASK`: `CE + 2.0 * mean(KL[keep]) + 0.1 * mean(-MI[keep])`.",
        "- KD uses `KL(softmax(teacher_logits/2) || softmax(student_logits/2))` with detached teacher probabilities.",
        "- MI is the `(C,C)` class-joint probability MI with `T=1`, `eps=1e-8`; no features, projection, MMD, InfoNCE, prototype or relation loss.",
        "",
        "## Mask inventory",
        "",
        f"Overall train samples: {sum(u['inventory']['train_count'] for u in units)}; teacher-correct: {sum(u['inventory']['teacher_correct_count'] for u in units)}; teacher-wrong: {sum(u['inventory']['teacher_wrong_count'] for u in units)}.",
        f"Overall keep rate: {decision['overall_keep_rate']:.6f}; units with keep rate 1: {sum(u['inventory']['teacher_wrong_count'] == 0 for u in units)}/{len(units)}.",
        "",
        "| dataset | train | correct | wrong | keep rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        rr = [u["inventory"] for u in units if u["dataset"] == ds]
        n = sum(r["train_count"] for r in rr); c = sum(r["teacher_correct_count"] for r in rr)
        lines.append(f"| {ds} | {n} | {c} | {n-c} | {c/n:.6f} |")
    lines += ["", "## Test Balanced Accuracy (subject means)", "",
              "| dataset | Base | KD_all | CE_TCORRECT_MASK | KD_MI_ALL | KD_MI_TCORRECT_MASK |", 
              "|---|---:|---:|---:|---:|---:|"]
    by_ds = {(r["dataset"], r["condition"]): r for r in dataset_rows}
    for ds in DATASETS:
        cells = []
        for c in ("Base", "KD_all", "CE_TCORRECT_MASK", "KD_MI_ALL", "KD_MI_TCORRECT_MASK"):
            r = by_ds[(ds, c)]
            cells.append(f"{r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f}")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## Required paired comparisons", "",
              "| dataset | comparison | mean Δ BA | W/T/L | 95% bootstrap CI |", 
              "|---|---|---:|---|---:|"]
    for r in paired_rows:
        lines.append(f"| {r['dataset']} | {r['comparison']} | {r['mean_delta_balanced_accuracy']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] |")
    lines += ["", "## Mechanism correlation", "",
              f"Across {len(units)} units, Spearman(teacher_wrong_rate, KD_MI_TCORRECT_MASK − KD_MI_ALL BA) = {mechanism['rho']:.4f}, p={mechanism['pvalue']:.4g}, dataset-stratified bootstrap 95% CI [{mechanism['ci95'][0]:.4f}, {mechanism['ci95'][1]:.4f}]. This is descriptive only and does not alter the mask.",
              "", "## Integrity and limitations", "",
              "- All three new conditions used the same split, initial IFNet state, preprocessing and 100-epoch batch schedule per unit; hashes are recorded in the CSV and checkpoint payloads.",
              "- The teacher artifact loader read only train logits, labels, UID and split metadata. Teacher features were not accessed or passed to the pilot.",
              "- No teacher test artifact was read and `results/artifacts/` was not written.",
              "- A teacher-wrong sample is still present in every DataLoader batch and participates in IFNet forward/BatchNorm statistics; only the loss aggregation excludes it.",
              "- A one-sample masked MI batch uses a graph-connected zero MI term; an all-masked batch performs no backward or optimizer step while the scheduler advances.",
              "- Test data are used only for final evaluation, never for mask construction, tuning, checkpoint selection, or seed selection.",
              "",
              f"Formal run status: {len(new_rows)}/{_run_count()} new runs complete; reused controls: {len(reused)}/{len(units) * len(OLD_METHODS)}.",
              "",
              "Output files are accompanied by `execution_provenance.json`, `mask_inventory.csv`, `run_manifest.csv`, `results_per_run.csv`, `reused_controls.csv`, `results_per_subject.csv`, `results_per_dataset.csv`, `paired_comparisons.csv`, `training_history/`, `checkpoints/`, and `predictions/`.",
    ]
    _atomic_text(output_root / "report.md", "\n".join(lines) + "\n")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = yaml.safe_load(resolve_local_file(config_path).read_text())
    _validate_config(cfg)
    global TOTAL_CSV, OLD_CSV
    total_csv_cfg = cfg.get("main_csv")
    if total_csv_cfg:
        TOTAL_CSV = require_external_output(total_csv_cfg)
        if not TOTAL_CSV.is_absolute():
            TOTAL_CSV = ROOT / TOTAL_CSV
    old_csv_cfg = cfg.get("old_controls_csv")
    if old_csv_cfg:
        OLD_CSV = external_path(old_csv_cfg)
        if not OLD_CSV.is_absolute():
            OLD_CSV = ROOT / OLD_CSV
    output_root = require_external_output(cfg["output_dir"])
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise RuntimeError(f"formal output is non-empty; use --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    git_before = _git_snapshot()
    started = _now()
    units = _preflight_units(cfg)
    if cfg.get("supplement_mode"):
        reused = []
        _atomic_csv(output_root / "control_validation.csv", [
            {"dataset": u["dataset"], "subject": u["subject"],
             "subject_index": u["subject_index"], "seed": SEED,
             "control": method, "status": "unavailable",
             "reason": "001-4 Stage 1 accuracy-only supplement has no complete checkpoint/history/prediction provenance; controls are not represented as strictly validated"}
            for u in units for method in OLD_METHODS
        ])
    else:
        reused = _validate_old_controls(units)
    inventory_rows = [u["inventory"] for u in units]
    _atomic_csv(output_root / "mask_inventory.csv", inventory_rows)
    reused_fields = []
    for row in reused:
        for key in row:
            if key not in reused_fields:
                reused_fields.append(key)
    _atomic_csv(output_root / "reused_controls.csv", reused, reused_fields)
    resolved = {
        "source_config": str(config_path.resolve()), "config": cfg,
        "scope": {"datasets": DATASETS, "subjects_by_dataset": DATASET_SUBJECTS,
                   "subject_index_base": 0, "report_subject_base": 1,
                   "experiment_units": len(units), "new_condition_runs": len(units) * len(CONDITIONS),
                   "protocol": "fewshot", "val_split_test_fraction": VAL_SPLIT,
                   "train_fraction": 0.3, "test_fraction": 0.7},
        "conditions": list(CONDITIONS), "teacher": "mirepnet", "student": "ifnet",
        "artifact_root_read_only": str((ROOT / cfg["artifact_root"]).resolve()),
        "output_root": str(output_root.resolve()),
    }
    _atomic_yaml(output_root / "config_resolved.yaml", resolved)
    if all(u["inventory"]["teacher_wrong_count"] == 0 for u in units):
        # The preregistered stopping rule avoids spending GPU time when masking
        # has no effect.  This branch is not expected for the current inventory.
        provenance = {"status": "stopped_mask_all_one", "created_at": started,
                      "git": {"before": git_before, "after": _git_snapshot()},
                      "mask_inventory_summary": {"teacher_wrong_count": 0},
                      "test_artifacts_read": False, "results_artifacts_written": False}
        _atomic_json(output_root / "execution_provenance.json", provenance)
        _atomic_text(output_root / "report.md", f"# Pilot stopped\n\nAll {len(units)} masks were all-one; no GPU training was started.\n")
        print("mask inventory is all-one; stopped before GPU training")
        return 0
    if not torch.cuda.is_available():
        _write_manifest(output_root, units, []) 
        provenance = {
            "status": "resource_blocked", "started_at": started, "finished_at": _now(),
            "reason": "CUDA is unavailable; no formal training was started",
            "git": {"before": git_before, "after": _git_snapshot()},
            "gpu": _gpu_snapshot(args.gpu), "test_artifacts_read": False,
            "results_artifacts_written": False,
            "mask_inventory_summary": {
                "train_count": int(sum(u["inventory"]["train_count"] for u in units)),
                "teacher_correct_count": int(sum(u["inventory"]["teacher_correct_count"] for u in units)),
                "teacher_wrong_count": int(sum(u["inventory"]["teacher_wrong_count"] for u in units)),
            },
        }
        _atomic_json(output_root / "execution_provenance.json", provenance)
        _atomic_text(output_root / "report.md",
                     "# Teacher-Correct Full-Loss Masked KD+MI Pilot\n\n"
                     "Preflight and focused tests passed, but formal training was "
                     "blocked because no CUDA device was available. No run was started.\n")
        print("resource blocked: CUDA unavailable; preflight outputs were written")
        return 2
    device = f"cuda:{args.gpu}"
    gpu_info = _gpu_snapshot(args.gpu)
    result_path = output_root / "results_per_run.csv"
    total_path = TOTAL_CSV
    rows = _load_new_rows(result_path)
    # Remove duplicate keys deterministically; a complete valid row is retained.
    by_key = {(r.get("dataset"), int(r.get("subject_index", -1)), int(r.get("seed", -1)), r.get("condition")): r
              for r in rows if r.get("condition") in CONDITIONS}
    _write_manifest(output_root, units, list(by_key.values()))
    completed = 0
    failures = []
    for unit in units:
        for condition in CONDITIONS:
            key = (unit["dataset"], unit["subject_index"], SEED, condition)
            existing = by_key.get(key)
            if existing is not None and _new_row_valid(existing, unit, output_root):
                completed += 1
                print(f"[resume {completed}/{_run_count()}] {unit['dataset']} S{unit['subject']} {condition}", flush=True)
                continue
            try:
                row = _train_condition(unit, condition, output_root, cfg, device)
                by_key[key] = row
                if row["status"] != "complete":
                    failures.append(row)
                print(f"[progress {len(by_key)}/{_run_count()}] {unit['dataset']} S{unit['subject']} {condition} "
                      f"BA={row['test_balanced_accuracy']}", flush=True)
            except Exception as exc:
                failure = {
                    "dataset": unit["dataset"], "subject": unit["subject"],
                    "subject_index": unit["subject_index"], "seed": SEED,
                    "condition": condition, "result_source": "new", "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
                by_key[key] = failure
                failures.append(failure)
                print(f"[failed] {unit['dataset']} S{unit['subject']} {condition}: {exc}", flush=True)
            ordered = [by_key[k] for u in units for c in CONDITIONS
                       for k in [(u["dataset"], u["subject_index"], SEED, c)] if k in by_key]
            _atomic_csv(result_path, ordered)
            _atomic_csv(total_path, ordered)
            _write_manifest(output_root, units, ordered)
            _validate_new_triplet(unit, ordered)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    new_rows = [by_key[k] for u in units for c in CONDITIONS
                for k in [(u["dataset"], u["subject_index"], SEED, c)] if k in by_key]
    if failures or len(new_rows) != len(units) * len(CONDITIONS):
        status = "incomplete"
        _atomic_json(output_root / "execution_provenance.json", {
            "status": status, "started_at": started, "finished_at": _now(),
            "git": {"before": git_before, "after": _git_snapshot()},
            "gpu": gpu_info, "failures": failures,
            "test_artifacts_read": False, "results_artifacts_written": False,
        })
        raise RuntimeError(f"pilot incomplete: {len(failures)} failures")
    subject_rows, dataset_rows, paired_rows = _summary_rows(reused + new_rows, output_root)
    decision = _decision(units, new_rows + reused, paired_rows)
    mechanism = _mechanism_stats(units, new_rows)
    new_by_unit_condition = {
        (r["dataset"], int(r["subject_index"]), r["condition"]): r for r in new_rows
    }
    historical_hash_mismatches = []
    old_method_for_schedule = {"CE_TCORRECT_MASK": "Base",
                               "KD_MI_ALL": "KD_all",
                               "KD_MI_TCORRECT_MASK": "CE_MI"}
    old_by_unit_method = {
        (r["dataset"], int(r["subject_index"]), r["condition"]): r for r in reused
    }
    if reused:
        for unit in units:
            for condition, old_condition in old_method_for_schedule.items():
                new_row = new_by_unit_condition[(unit["dataset"], unit["subject_index"], condition)]
                old_row = old_by_unit_method.get((unit["dataset"], unit["subject_index"], old_condition))
                if old_row is None or new_row.get("batch_order_hashes") != old_row.get("batch_order_hashes"):
                    historical_hash_mismatches.append({
                        "dataset": unit["dataset"], "subject": unit["subject"],
                        "condition": condition, "old_condition": old_condition,
                    })
    hash_comparison = {
        "new_triplet_hashes_equal": True,
        "historical_batch_schedule_validation": "performed" if reused else "unavailable_no_valid_legacy_controls",
        "historical_batch_schedule_match_count": max(0, len(units) * len(old_method_for_schedule) - len(historical_hash_mismatches)) if reused else None,
        "historical_batch_schedule_mismatch_count": len(historical_hash_mismatches),
        "historical_batch_schedule_mismatches": historical_hash_mismatches,
        "historical_initial_split_artifact_hashes_validated": True,
    }
    provenance = {
        "status": "complete", "started_at": started, "finished_at": _now(),
        "config_path": str(config_path.resolve()), "config_sha256": _sha256_file(config_path),
        "command": " ".join([sys.executable, *sys.argv]), "git": {"before": git_before, "after": _git_snapshot()},
        "python": platform.python_version(), "gpu": gpu_info,
        "formal_scope": resolved["scope"], "conditions": list(CONDITIONS),
        "new_runs": len(new_rows), "reused_controls": len(reused),
        "mask_inventory_summary": {
            "train_count": int(sum(u["inventory"]["train_count"] for u in units)),
            "teacher_correct_count": int(sum(u["inventory"]["teacher_correct_count"] for u in units)),
            "teacher_wrong_count": int(sum(u["inventory"]["teacher_wrong_count"] for u in units)),
            "all_keep_unit_count": int(sum(u["inventory"]["teacher_wrong_count"] == 0 for u in units)),
        },
        "decision": decision, "mechanism": mechanism,
        "hash_comparison": hash_comparison,
        "session_provenance": {d: _session_default(d) for d in DATASETS},
        "teacher_artifacts": [{"path": u["inventory"]["teacher_artifact_path"],
                               "sha256": u["inventory"]["teacher_artifact_sha256"],
                               "alignment_hash": u["teacher_uid_alignment_hash"]} for u in units],
        "test_artifacts_read": False, "test_split_used_for_training": False,
        "results_artifacts_written": False,
        "limitations": [
            "loss-only mask leaves teacher-wrong samples in IFNet forward and BatchNorm statistics",
            "teacher-correct mask is label-conditioned on the training split",
            "results use one seed (666) by preregistration; seeds 667/668 were not run",
        ],
    }
    _atomic_json(output_root / "execution_provenance.json", provenance)
    _write_report(output_root, cfg, units, reused, new_rows, dataset_rows,
                  paired_rows, decision, mechanism, provenance)
    print(f"[complete] {len(new_rows)}/{_run_count()} new runs; output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
