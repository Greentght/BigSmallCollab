"""Paired delayed KD: all-sample KD versus teacher-correct-only KD.

Every cell (dataset, subject, seed) gets one CE-only warm-up through epoch 10.
The complete epoch-10 model/optimizer/scheduler/RNG/sampler state is then loaded
into both continuations. CE always sees every train sample. Only the KD term is
masked in DELAYED_KD_TCORRECT.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import data
from data import split as split_utils
from models import get_adapter
from experiments.storage import external_path, require_external_output, resolve_local_file

DATASETS = ("BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI")
CONDITIONS = ("DELAYED_KD_ALL", "DELAYED_KD_TCORRECT")
LEGACY_CSV = Path("/data1/llx/BigSmallCollab_results/distill/prealign_delayed_kd_pilot.csv")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/delayed_kd_teacher_correct_paired.yaml")
    p.add_argument("--gpu", type=int, default=None, help="optional CUDA device; CPU is the default")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="run epoch-10 forks for one all-correct and one error-containing cell")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(resolve_local_file(path).read_text())
    expected = {
        "datasets": list(DATASETS), "protocol": "fewshot", "val_split": 0.7,
        "seed": 666, "teacher": "mirepnet", "student": "ifnet",
        "epochs": 100, "warmup_epochs": 10, "batch_size": 16,
        "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.01,
        "scheduler": "CosineAnnealingLR", "temperature_kd": 2.0,
        "lam_kd": 0.5, "drop_last": False, "model_selection": "final_epoch",
        "conditions": list(CONDITIONS),
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"config {key} must be {value!r}, got {cfg.get(key)!r}")
    cfg["artifact_root"] = str(external_path(cfg['artifact_root']))
    cfg["output_dir"] = str(require_external_output(cfg['output_dir']))
    return cfg


def _device(gpu):
    if gpu is None:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu was provided but CUDA is unavailable")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def _seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _sha256_file(path):
    path = resolve_local_file(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hash_parts(*parts):
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        h.update(part)
    return h.hexdigest()


def _hash_array(value):
    value = np.asarray(value)
    return _hash_parts(str(value.dtype), str(tuple(value.shape)),
                       np.ascontiguousarray(value).tobytes())


def _hash_uid_split(uid_tr, uid_te):
    return _hash_parts(b"train", np.asarray(uid_tr, dtype=np.int64).tobytes(),
                       b"test", np.asarray(uid_te, dtype=np.int64).tobytes())


def _hash_state(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(str(key).encode("utf-8"))
        value = state[key]
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
            h.update(str(array.dtype).encode("utf-8"))
            h.update(str(tuple(array.shape)).encode("utf-8"))
            h.update(np.ascontiguousarray(array).tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _schedule(n, batch_size, epochs, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = []
    for _ in range(int(epochs)):
        permutation = torch.randperm(n, generator=generator).numpy().astype(np.int64)
        result.append([permutation[i:i + batch_size]
                       for i in range(0, n, batch_size)])
    return result


def _schedule_hash(schedule, uid_tr):
    h = hashlib.sha256()
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    for epoch, batches in enumerate(schedule, start=1):
        h.update(np.asarray([epoch], dtype=np.int64).tobytes())
        for batch in batches:
            h.update(uid_tr[np.asarray(batch, dtype=np.int64)].tobytes())
    return h.hexdigest()


def _atomic_json(path, value):
    path = require_external_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True,
                              ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def _atomic_csv(path, rows):
    path = require_external_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_torch_save(path, payload):
    path = require_external_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, require_external_output(tmp))
    os.replace(tmp, path)


def _run_key(dataset, subject, seed):
    return f"{dataset}__S{int(subject) + 1}__seed{seed}"


def _read_mask(path):
    with np.load(resolve_local_file(path), allow_pickle=False) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def _prepare_masks(cfg):
    out = require_external_output(cfg["output_dir"])
    mask_dir = out / "teacher_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    unit_rows = []
    total_n = total_wrong = 0
    for dataset_name in DATASETS:
        dcfg = config.load_dataset_config(dataset_name)
        for subject in range(int(dcfg["num_subjects"])):
            X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
                dataset_name, subject, val_split=float(cfg["val_split"]),
                seed=int(cfg["seed"]), return_uid=True)
            y_tr = np.asarray(y_tr, dtype=np.int64)
            y_te = np.asarray(y_te, dtype=np.int64)
            uid_tr = np.asarray(uid_tr, dtype=np.int64)
            uid_te = np.asarray(uid_te, dtype=np.int64)
            for label, uid in (("train", uid_tr), ("test", uid_te)):
                if uid.ndim != 2 or uid.shape[1] != 2:
                    raise ValueError(f"{dataset_name} S{subject + 1}: invalid {label} UID shape {uid.shape}")
                if len({tuple(row) for row in uid.tolist()}) != len(uid):
                    raise ValueError(f"{dataset_name} S{subject + 1}: duplicate {label} UID")
            if set(map(tuple, uid_tr.tolist())) & set(map(tuple, uid_te.tolist())):
                raise ValueError(f"{dataset_name} S{subject + 1}: train/test UID overlap")
            if not set(np.unique(y_tr)).issubset(set(range(int(dcfg["num_classes"])))):
                raise ValueError(f"{dataset_name} S{subject + 1}: train label mapping outside configured classes")
            if not set(np.unique(y_te)).issubset(set(range(int(dcfg["num_classes"])))):
                raise ValueError(f"{dataset_name} S{subject + 1}: test label mapping outside configured classes")

            artifact_path = external_path(cfg["artifact_root"]) / dataset_name / "mirepnet" / f"{subject}_{cfg['seed']}_train.npz"
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            with np.load(resolve_local_file(artifact_path), allow_pickle=False) as payload:
                needed = {"logits", "y", "sample_uid", "split_policy"}
                missing = needed.difference(payload.files)
                if missing:
                    raise ValueError(f"{artifact_path}: missing fields {sorted(missing)}")
                logits_raw = np.asarray(payload["logits"], dtype=np.float32)
                labels_raw = np.asarray(payload["y"], dtype=np.int64)
                uid_raw = np.asarray(payload["sample_uid"], dtype=np.int64)
                split_policy = str(payload["split_policy"].item())
            if split_policy != split_utils.FEWSHOT_SPLIT_POLICY:
                raise ValueError(f"{artifact_path}: split policy {split_policy!r} is not current fewshot")
            if logits_raw.ndim != 2 or logits_raw.shape[0] != len(labels_raw) or len(labels_raw) != len(uid_raw):
                raise ValueError(f"{artifact_path}: logits/y/UID dimensions disagree")
            if uid_raw.ndim != 2 or uid_raw.shape[1] != 2:
                raise ValueError(f"{artifact_path}: UID must have shape (N,2)")
            if len({tuple(row) for row in uid_raw.tolist()}) != len(uid_raw):
                raise ValueError(f"{artifact_path}: duplicate teacher UID")
            if not np.isfinite(logits_raw).all():
                raise ValueError(f"{artifact_path}: teacher train logits contain NaN/Inf")
            if logits_raw.shape[1] != int(dcfg["num_classes"]):
                raise ValueError(f"{artifact_path}: {logits_raw.shape[1]} logits for {dcfg['num_classes']} configured classes")
            if set(map(tuple, uid_raw.tolist())) != set(map(tuple, uid_tr.tolist())):
                raise ValueError(f"{artifact_path}: teacher train UID set differs from current split")
            lookup = {tuple(uid): i for i, uid in enumerate(uid_raw.tolist())}
            order = np.asarray([lookup[tuple(uid)] for uid in uid_tr.tolist()], dtype=np.int64)
            aligned_logits = logits_raw[order]
            aligned_y = labels_raw[order]
            if not np.array_equal(aligned_y, y_tr):
                raise ValueError(f"{artifact_path}: teacher labels differ after UID alignment")
            if not np.array_equal(uid_raw[order], uid_tr):
                raise ValueError(f"{artifact_path}: UID reorder failed")

            mask = (aligned_logits.argmax(axis=1) == y_tr).astype(np.uint8)
            wrong_uids = uid_tr[mask == 0]
            key = _run_key(dataset_name, subject, int(cfg["seed"]))
            mask_path = mask_dir / f"{key}.npz"
            expected_hash = _hash_array(mask)
            if mask_path.exists():
                old = _read_mask(mask_path)
                if (not np.array_equal(old["train_sample_uid"], uid_tr)
                        or not np.array_equal(old["y_train"], y_tr)
                        or not np.array_equal(old["teacher_correct_mask"], mask)
                        or not np.array_equal(old["teacher_logits"], aligned_logits)):
                    raise ValueError(f"existing teacher mask cache disagrees with current UID-aligned artifact: {mask_path}")
            else:
                tmp = mask_path.with_name(f".{mask_path.name}.tmp-{os.getpid()}")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, train_sample_uid=uid_tr, y_train=y_tr,
                                        teacher_logits=aligned_logits,
                                        teacher_correct_mask=mask)
                os.replace(tmp, mask_path)
            artifact_sha = _sha256_file(artifact_path)
            row = {
                "dataset": dataset_name, "subject": subject + 1,
                "subject_index": subject, "seed": int(cfg["seed"]),
                "train_count": len(y_tr), "test_count": len(y_te),
                "train_labels": json.dumps(np.unique(y_tr).tolist()),
                "test_labels": json.dumps(np.unique(y_te).tolist()),
                "train_class_counts": json.dumps(np.bincount(y_tr, minlength=int(dcfg["num_classes"])).tolist()),
                "test_class_counts": json.dumps(np.bincount(y_te, minlength=int(dcfg["num_classes"])).tolist()),
                "correct_teacher_count": int(mask.sum()),
                "teacher_wrong_count": int((mask == 0).sum()),
                "teacher_correct_rate": float(mask.mean()),
                "mask_all_ones": bool(mask.all()),
                "wrong_sample_uids": json.dumps(wrong_uids.tolist()),
                "split_policy": split_policy,
                "train_uid_hash": _hash_array(uid_tr),
                "test_uid_hash": _hash_array(uid_te),
                "split_uid_hash": _hash_uid_split(uid_tr, uid_te),
                "teacher_uid_alignment_hash": _hash_parts(uid_tr.astype(np.int64).tobytes(), y_tr.astype(np.int64).tobytes()),
                "teacher_mask_hash": expected_hash,
                "teacher_train_logits_hash": _hash_array(aligned_logits),
                "teacher_artifact_path": str(artifact_path.resolve()),
                "teacher_artifact_sha256": artifact_sha,
                "teacher_mask_path": str(mask_path.resolve()),
                "teacher_reordered_to_current_uid": bool(not np.array_equal(uid_raw, uid_tr)),
            }
            unit_rows.append(row)
            total_n += len(mask)
            total_wrong += int((mask == 0).sum())
            print(f"MASK {dataset_name} S{subject + 1}: n={len(mask)} wrong={int((mask == 0).sum())} rate={mask.mean():.4f}", flush=True)

    n_all = sum(bool(row["mask_all_ones"]) for row in unit_rows)
    n_err = len(unit_rows) - n_all
    manifest = {
        "status": "complete", "datasets": list(DATASETS), "n_units": len(unit_rows),
        "train_uid_count": total_n, "teacher_wrong_count": total_wrong,
        "teacher_correct_rate": 1.0 - total_wrong / total_n,
        "all_one_units": n_all, "error_containing_units": n_err,
        "split": "fewshot_stratified_random; 30% train / 70% test; seed 666",
        "teacher_source": "MIRepNet train logits only, UID and label aligned to current train split",
        "mask_definition": "teacher logits argmax == current train label, fixed before training",
    }
    _atomic_csv(out / "teacher_mask_manifest.csv", unit_rows)
    _atomic_json(out / "teacher_mask_manifest.json", manifest)
    return unit_rows, manifest


def _load_training_unit(cfg, row, device):
    dataset_name = row["dataset"]
    subject = int(row["subject_index"])
    X_tr, y_tr, X_te, y_te, uid_tr, uid_te = data.subject_split(
        dataset_name, subject, val_split=float(cfg["val_split"]),
        seed=int(cfg["seed"]), return_uid=True)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)
    uid_tr = np.asarray(uid_tr, dtype=np.int64)
    uid_te = np.asarray(uid_te, dtype=np.int64)
    if _hash_array(uid_tr) != row["train_uid_hash"] or _hash_array(uid_te) != row["test_uid_hash"]:
        raise ValueError(f"{dataset_name} S{subject + 1}: UID changed since preflight")
    mask_data = _read_mask(row["teacher_mask_path"])
    if not np.array_equal(mask_data["train_sample_uid"], uid_tr) or not np.array_equal(mask_data["y_train"], y_tr):
        raise ValueError(f"{dataset_name} S{subject + 1}: saved mask no longer aligns with train UID/labels")
    if _hash_array(mask_data["teacher_correct_mask"]) != row["teacher_mask_hash"]:
        raise ValueError(f"{dataset_name} S{subject + 1}: mask hash changed")

    student_cfg = config.load_model_config("ifnet", dataset_name, protocol="fewshot")
    student_cfg.update(dataset_name=dataset_name, in_channels=int(X_tr.shape[1]),
                       samples=1000, epochs=int(cfg["epochs"]),
                       lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]),
                       batch_size=int(cfg["batch_size"]), optimizer="adamw")
    adapter = get_adapter("ifnet", device=str(device), **student_cfg)
    Xp_tr = adapter.preprocess(X_tr)
    Xp_te = adapter.preprocess(X_te)
    if len(Xp_tr) != len(y_tr) or len(Xp_te) != len(y_te):
        raise ValueError(f"{dataset_name} S{subject + 1}: IFNet preprocessing changed sample count")
    order = _schedule(len(y_tr), int(cfg["batch_size"]), int(cfg["epochs"]), int(cfg["seed"]))
    batch_hash = _schedule_hash(order, uid_tr)
    return {
        "dataset_cfg": config.load_dataset_config(dataset_name),
        "adapter": adapter, "Xp_tr": Xp_tr, "Xp_te": Xp_te,
        "y_tr": y_tr, "y_te": y_te, "uid_tr": uid_tr, "uid_te": uid_te,
        "train_uid_hash": row["train_uid_hash"],
        "test_uid_hash": row["test_uid_hash"],
        "teacher_logits": np.asarray(mask_data["teacher_logits"], dtype=np.float32),
        "teacher_mask": np.asarray(mask_data["teacher_correct_mask"], dtype=np.bool_),
        "schedule": order, "batch_order_hash": batch_hash,
        "split_uid_hash": row["split_uid_hash"],
    }


def _make_model(adapter, num_classes):
    return adapter.build(int(num_classes))


def _optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]))
    return optimizer, scheduler


def _train_epoch(model, optimizer, adapter, data_unit, cfg, device, epoch,
                 kd_condition, steps_limit=None):
    model.train()
    y_tensor = torch.as_tensor(data_unit["y_tr"], dtype=torch.long)
    teacher_tensor = torch.as_tensor(data_unit["teacher_logits"], dtype=torch.float32)
    mask_tensor = torch.as_tensor(data_unit["teacher_mask"], dtype=torch.bool)
    n_seen = n_kd_valid = n_batches = zero_kd_batches = 0
    ce_sum = kd_sum = total_sum = 0.0
    batches = data_unit["schedule"][epoch - 1]
    if steps_limit is not None:
        batches = batches[:steps_limit]
    for batch in batches:
        idx = np.asarray(batch, dtype=np.int64)
        xb = data_unit["Xp_tr"][idx].to(device)
        yb = y_tensor[idx].to(device)
        logits = adapter.forward(model, xb)[1]
        ce = F.cross_entropy(logits, yb)
        loss = ce
        kd = logits.sum() * 0.0
        if kd_condition is not None:
            T = float(cfg["temperature_kd"])
            teacher = teacher_tensor[idx].to(device)
            per_sample_kl = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.softmax(teacher.detach() / T, dim=1),
                reduction="none").sum(dim=1)
            kd_mask = (torch.ones_like(mask_tensor[idx]) if kd_condition == "all"
                       else mask_tensor[idx]).to(device=device, dtype=per_sample_kl.dtype)
            denominator = kd_mask.sum().clamp_min(1.0)
            kd = (per_sample_kl * kd_mask).sum() / denominator
            loss = ce + float(cfg["lam_kd"]) * (T ** 2) * kd
            n_valid = int(kd_mask.sum().item())
            n_kd_valid += n_valid
            if n_valid == 0:
                zero_kd_batches += 1
            kd_sum += float(kd.detach().item()) * n_valid
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        n = len(idx)
        n_seen += n
        n_batches += 1
        ce_sum += float(ce.detach().item()) * n
        total_sum += float(loss.detach().item()) * n
    if not n_seen:
        raise RuntimeError("training epoch processed no samples")
    if steps_limit is None and n_seen != len(data_unit["y_tr"]):
        raise RuntimeError(f"epoch {epoch}: CE processed {n_seen}/{len(data_unit['y_tr'])} training samples")
    return {
        "epoch": epoch, "phase": "ce_warmup" if kd_condition is None else "delayed_kd",
        "ce_loss": ce_sum / n_seen,
        "kd_loss_on_valid_samples": kd_sum / n_kd_valid if n_kd_valid else 0.0,
        "total_loss": total_sum / n_seen,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "ce_samples": n_seen, "kd_effective_samples": n_kd_valid,
        "kd_effective_rate": n_kd_valid / n_seen,
        "zero_kd_batches": zero_kd_batches, "batches": n_batches,
    }


def _warmup_path(out, key):
    return out / "checkpoints" / "epoch10" / f"{key}.pt"


def _create_warmup(out, key, dataset_name, subject, cfg, unit, device, max_epochs=10):
    path = _warmup_path(out, key)
    if path.exists():
        cached = torch.load(resolve_local_file(path), map_location="cpu", weights_only=False)
        if (cached.get("epoch") != int(cfg["warmup_epochs"])
                or cached.get("split_uid_hash") != unit["split_uid_hash"]
                or cached.get("train_uid_hash") != unit["train_uid_hash"]
                or cached.get("test_uid_hash") != unit["test_uid_hash"]
                or cached.get("batch_order_hash") != unit["batch_order_hash"]):
            raise ValueError(f"existing epoch-10 checkpoint provenance mismatch: {path}")
        return cached, _sha256_file(path)

    _seed(int(cfg["seed"]), device)
    model = _make_model(unit["adapter"], unit["dataset_cfg"]["num_classes"])
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer, scheduler = _optimizer(model, cfg)
    history = []
    run_epochs = min(int(cfg["warmup_epochs"]), int(max_epochs))
    for epoch in range(1, run_epochs + 1):
        entry = _train_epoch(model, optimizer, unit["adapter"], unit, cfg,
                             device, epoch, kd_condition=None)
        scheduler.step()
        entry["lr_after_scheduler_step"] = float(optimizer.param_groups[0]["lr"])
        history.append(entry)
    if run_epochs != int(cfg["warmup_epochs"]):
        return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "rng_state": _rng_state(device),
                "epoch": run_epochs, "next_epoch": run_epochs + 1,
                "next_batch_index": 0, "warmup_history": history,
                "initial_state_hash": _hash_state(init_state),
                "epoch10_student_state_hash": _hash_state(model.state_dict()),
                "split_uid_hash": unit["split_uid_hash"],
                "train_uid_hash": unit["train_uid_hash"],
                "test_uid_hash": unit["test_uid_hash"],
                "batch_order_hash": unit["batch_order_hash"],
                "schedule": unit["schedule"]}, "smoke-unpersisted"
    payload = {
        "protocol": "delayed_kd_teacher_correct_paired_v1",
        "dataset": dataset_name, "subject_index": int(subject),
        "subject": int(subject) + 1, "seed": int(cfg["seed"]),
        "epoch": run_epochs, "next_epoch": run_epochs + 1,
        "next_batch_index": 0,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": _rng_state(device),
        "initial_state_hash": _hash_state(init_state),
        "epoch10_student_state_hash": _hash_state(model.state_dict()),
        "split_uid_hash": unit["split_uid_hash"],
        "train_uid_hash": unit["train_uid_hash"],
        "test_uid_hash": unit["test_uid_hash"],
        "batch_order_hash": unit["batch_order_hash"],
        "sampler_state": {
            "kind": "precomputed_per_epoch_random_permutation",
            "next_epoch": run_epochs + 1, "next_batch_index": 0,
            "schedule": [[b.tolist() for b in ep] for ep in unit["schedule"]],
        },
        "warmup_history": history,
    }
    _atomic_torch_save(path, payload)
    saved = torch.load(resolve_local_file(path), map_location="cpu", weights_only=False)
    return saved, _sha256_file(path)


@torch.no_grad()
def _predict(adapter, model, Xp, batch_size, device):
    model.eval()
    parts = []
    for start in range(0, len(Xp), batch_size):
        logits = adapter.forward(model, Xp[start:start + batch_size].to(device))[1]
        parts.append(logits.detach().cpu())
    return torch.cat(parts, dim=0).numpy().astype(np.float32)


def _metrics(y, logits):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(logits).argmax(axis=1)
    return {
        "accuracy": float((pred == y).mean() * 100),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred) * 100),
        "kappa": float(cohen_kappa_score(y, pred)),
        "pred": pred,
    }


def _branch_paths(out, key, condition, smoke=False):
    root = out / "smoke_test" if smoke else out
    return {
        "checkpoint": root / "checkpoints" / "final" / f"{key}__{condition}.pt",
        "prediction": root / "predictions" / f"{key}__{condition}.npz",
        "history": root / "training_history" / f"{key}__{condition}.json",
        "result": root / "branch_results" / f"{key}__{condition}.json",
    }


def _branch_complete(paths):
    if not paths["result"].is_file() or not paths["checkpoint"].is_file() or not paths["prediction"].is_file() or not paths["history"].is_file():
        return None
    try:
        row = json.loads(resolve_local_file(paths["result"]).read_text())
        return row if row.get("status") == "complete" and int(row.get("epochs_completed", 0)) == 100 else None
    except Exception:
        return None


def _train_branch(out, key, condition, warmup, warmup_sha, unit, cfg, device,
                  max_kd_epochs=90, smoke=False):
    paths = _branch_paths(out, key, condition, smoke=smoke)
    if not smoke:
        prior = _branch_complete(paths)
        if prior is not None:
            return prior
    adapter = unit["adapter"]
    model = _make_model(adapter, unit["dataset_cfg"]["num_classes"])
    model.load_state_dict(warmup["model_state_dict"], strict=True)
    optimizer, scheduler = _optimizer(model, cfg)
    # Optimizer.load_state_dict may retain references to the supplied state
    # tensors; each branch must own an independent copy of the shared epoch-10
    # snapshot so the first continuation cannot advance the second one.
    optimizer.load_state_dict(deepcopy(warmup["optimizer_state_dict"]))
    scheduler.load_state_dict(deepcopy(warmup["scheduler_state_dict"]))
    _restore_rng(warmup["rng_state"], device)
    initial_branch_state_hash = _hash_state(model.state_dict())
    if initial_branch_state_hash != warmup["epoch10_student_state_hash"]:
        raise ValueError(f"{key}: branch did not restore the common epoch-10 student state")

    n_epochs = min(int(max_kd_epochs), int(cfg["epochs"]) - int(cfg["warmup_epochs"]))
    start = time.time()
    history = list(warmup["warmup_history"])
    kd_condition = "all" if condition == "DELAYED_KD_ALL" else "tcorrect"
    ce_samples = kd_samples = zero_kd_batches = total_kd_batches = 0
    for offset in range(n_epochs):
        epoch = int(cfg["warmup_epochs"]) + offset + 1
        entry = _train_epoch(model, optimizer, adapter, unit, cfg, device,
                             epoch, kd_condition=kd_condition)
        scheduler.step()
        entry["lr_after_scheduler_step"] = float(optimizer.param_groups[0]["lr"])
        history.append(entry)
        ce_samples += entry["ce_samples"]
        kd_samples += entry["kd_effective_samples"]
        zero_kd_batches += entry["zero_kd_batches"]
        total_kd_batches += entry["batches"]
        if (offset + 1) % 30 == 0 or offset + 1 == n_epochs:
            print(f"TRAIN {key} {condition}: epoch={epoch} kd_rate={entry['kd_effective_rate']:.4f}", flush=True)

    test_logits = _predict(adapter, model, unit["Xp_te"],
                           int(cfg["batch_size"]), device)
    test_metrics = _metrics(unit["y_te"], test_logits)
    train_logits = _predict(adapter, model, unit["Xp_tr"],
                            int(cfg["batch_size"]), device)
    train_metrics = _metrics(unit["y_tr"], train_logits)
    final_state_hash = _hash_state(model.state_dict())
    epochs_done = int(cfg["warmup_epochs"]) + n_epochs
    row = {
        "dataset": key.split("__")[0], "subject": int(key.split("__S")[1].split("__")[0]),
        "subject_index": int(key.split("__S")[1].split("__")[0]) - 1,
        "seed": int(cfg["seed"]), "condition": condition,
        "train_count": len(unit["y_tr"]), "test_count": len(unit["y_te"]),
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "kappa": test_metrics["kappa"], "train_accuracy": train_metrics["accuracy"],
        "teacher_wrong_count": int((~unit["teacher_mask"]).sum()),
        "teacher_correct_count": int(unit["teacher_mask"].sum()),
        "kd_effective_samples_per_epoch": int(unit["teacher_mask"].sum()) if condition == "DELAYED_KD_TCORRECT" else len(unit["teacher_mask"]),
        "kd_effective_rate": float(unit["teacher_mask"].mean()) if condition == "DELAYED_KD_TCORRECT" else 1.0,
        "kd_samples_processed": int(kd_samples), "ce_samples_processed_during_kd": int(ce_samples),
        "ce_samples_processed_warmup": int(len(unit["y_tr"]) * int(cfg["warmup_epochs"])),
        "ce_samples_processed_total": int(len(unit["y_tr"]) * epochs_done),
        "ce_sample_rate_per_epoch": 1.0,
        "zero_kd_batches": int(zero_kd_batches), "kd_batches": int(total_kd_batches),
        "warmup_checkpoint_sha256": warmup_sha,
        "epoch10_student_state_hash": warmup["epoch10_student_state_hash"],
        "branch_restored_state_hash": initial_branch_state_hash,
        "final_student_state_hash": final_state_hash,
        "split_uid_hash": unit["split_uid_hash"],
        "train_uid_hash": unit["train_uid_hash"],
        "test_uid_hash": unit["test_uid_hash"],
        "batch_order_hash": unit["batch_order_hash"],
        "epochs_completed": epochs_done,
        "runtime_seconds": time.time() - start,
        "status": "complete" if epochs_done == int(cfg["epochs"]) else "smoke_complete",
        "checkpoint_path": str(paths["checkpoint"].resolve()),
        "prediction_path": str(paths["prediction"].resolve()),
        "history_path": str(paths["history"].resolve()),
    }
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(paths["checkpoint"], {
        "complete": True, "condition": condition, "dataset": row["dataset"],
        "subject": row["subject"], "seed": int(cfg["seed"]),
        "epochs_completed": epochs_done,
        "student_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "warmup_checkpoint_sha256": warmup_sha,
        "epoch10_student_state_hash": warmup["epoch10_student_state_hash"],
        "final_student_state_hash": final_state_hash,
        "split_uid_hash": unit["split_uid_hash"],
        "train_uid_hash": unit["train_uid_hash"],
        "test_uid_hash": unit["test_uid_hash"],
        "batch_order_hash": unit["batch_order_hash"],
        "next_epoch": epochs_done + 1, "next_batch_index": 0,
        "rng_state": _rng_state(device),
    })
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["prediction"].with_name(f".{paths['prediction'].name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f, train_sample_uid=unit["uid_tr"], test_sample_uid=unit["uid_te"],
            y_train=unit["y_tr"], y_test=unit["y_te"],
            train_logits=train_logits, test_logits=test_logits,
            train_pred=train_metrics["pred"], test_pred=test_metrics["pred"],
            teacher_correct_mask=unit["teacher_mask"].astype(np.uint8),
        )
    os.replace(tmp, paths["prediction"])
    _atomic_json(paths["history"], {
        "dataset": row["dataset"], "subject": row["subject"],
        "seed": int(cfg["seed"]), "condition": condition,
        "common_warmup_epochs": 10, "continuation_epochs": n_epochs,
        "warmup_checkpoint_sha256": warmup_sha,
        "epoch10_student_state_hash": warmup["epoch10_student_state_hash"],
        "split_uid_hash": unit["split_uid_hash"],
        "train_uid_hash": unit["train_uid_hash"],
        "test_uid_hash": unit["test_uid_hash"],
        "batch_order_hash": unit["batch_order_hash"], "epochs": history,
    })
    _atomic_json(paths["result"], row)
    return row


def _verify_all_one_pair(out, key, warmup_sha):
    rows = {}
    logits = {}
    for condition in CONDITIONS:
        paths = _branch_paths(out, key, condition)
        rows[condition] = json.loads(resolve_local_file(paths["result"]).read_text())
        with np.load(resolve_local_file(paths["prediction"]), allow_pickle=False) as payload:
            logits[condition] = {
                "test_logits": np.asarray(payload["test_logits"]),
                "train_logits": np.asarray(payload["train_logits"]),
            }
    for row in rows.values():
        if row["warmup_checkpoint_sha256"] != warmup_sha:
            raise RuntimeError(f"{key}: branch warm-up checkpoint hashes differ")
        if row["epoch10_student_state_hash"] != rows[CONDITIONS[0]]["epoch10_student_state_hash"]:
            raise RuntimeError(f"{key}: branch epoch-10 student states differ")
        if row["batch_order_hash"] != rows[CONDITIONS[0]]["batch_order_hash"]:
            raise RuntimeError(f"{key}: branch batch-order hashes differ")
        if (row["split_uid_hash"] != rows[CONDITIONS[0]]["split_uid_hash"]
                or row["train_uid_hash"] != rows[CONDITIONS[0]]["train_uid_hash"]
                or row["test_uid_hash"] != rows[CONDITIONS[0]]["test_uid_hash"]):
            raise RuntimeError(f"{key}: branch train/test UID hashes differ")
    for metric in ("accuracy", "balanced_accuracy", "kappa", "train_accuracy"):
        if abs(float(rows[CONDITIONS[0]][metric]) - float(rows[CONDITIONS[1]][metric])) > 1e-12:
            raise RuntimeError(f"{key}: all-one mask consistency failed for {metric}")
    if rows[CONDITIONS[0]]["final_student_state_hash"] != rows[CONDITIONS[1]]["final_student_state_hash"]:
        raise RuntimeError(f"{key}: all-one mask final student states differ")
    for split in ("test_logits", "train_logits"):
        if not np.array_equal(logits[CONDITIONS[0]][split], logits[CONDITIONS[1]][split]):
            raise RuntimeError(f"{key}: all-one mask {split} differ")
    return True


def _read_rows(path):
    path = external_path(path)
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _persist_rows(out, rows):
    _atomic_csv(out / "results_per_run.csv", rows)


def _bootstrap_ci(values, seed=666, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        samples[i] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _summarize(out, unit_rows, result_rows):
    indexed = {(r["dataset"], int(r["subject"]), r["condition"]): r
               for r in result_rows if r.get("status") == "complete"}
    by_unit = []
    for u in unit_rows:
        k = (u["dataset"], int(u["subject"]))
        a = indexed.get((k[0], k[1], CONDITIONS[0]))
        b = indexed.get((k[0], k[1], CONDITIONS[1]))
        if a is None or b is None:
            continue
        by_unit.append({
            "dataset": k[0], "subject": k[1],
            "teacher_wrong_count": int(u["teacher_wrong_count"]),
            "teacher_correct_rate": float(u["teacher_correct_rate"]),
            "all_balanced_accuracy": float(a["balanced_accuracy"]),
            "tcorrect_balanced_accuracy": float(b["balanced_accuracy"]),
            "delta_balanced_accuracy": float(b["balanced_accuracy"]) - float(a["balanced_accuracy"]),
            "all_accuracy": float(a["accuracy"]),
            "tcorrect_accuracy": float(b["accuracy"]),
            "delta_accuracy": float(b["accuracy"]) - float(a["accuracy"]),
            "all_kappa": float(a["kappa"]), "tcorrect_kappa": float(b["kappa"]),
            "delta_kappa": float(b["kappa"]) - float(a["kappa"]),
            "all_one_mask": bool(u["mask_all_ones"]),
            "all_one_consistency_verified": bool(a.get("all_one_consistency_verified") in (True, "True") and b.get("all_one_consistency_verified") in (True, "True")),
            "kd_effective_rate_tcorrect": float(b["kd_effective_rate"]),
            "zero_kd_batches_tcorrect": int(b["zero_kd_batches"]),
        })
    def stats(subset):
        if not subset:
            return {"n": 0, "mean_delta_balanced_accuracy": None,
                    "balanced_accuracy_ci95": [None, None], "win_tie_loss": [0, 0, 0]}
        delta = np.asarray([r["delta_balanced_accuracy"] for r in subset], dtype=np.float64)
        wins = int((delta > 1e-12).sum())
        losses = int((delta < -1e-12).sum())
        ties = len(delta) - wins - losses
        return {
            "n": len(delta), "mean_delta_balanced_accuracy": float(delta.mean()),
            "balanced_accuracy_ci95": _bootstrap_ci(delta),
            "mean_delta_accuracy": float(np.mean([r["delta_accuracy"] for r in subset])),
            "accuracy_ci95": _bootstrap_ci([r["delta_accuracy"] for r in subset]),
            "mean_delta_kappa": float(np.mean([r["delta_kappa"] for r in subset])),
            "kappa_ci95": _bootstrap_ci([r["delta_kappa"] for r in subset]),
            "win_tie_loss": [wins, ties, losses],
        }
    wrong_units = [r for r in by_unit if r["teacher_wrong_count"] > 0]
    summary = {
        "primary_comparison": "DELAYED_KD_TCORRECT - DELAYED_KD_ALL; Balanced Accuracy percentage points",
        "n_completed_pairs": len(by_unit), "overall": stats(by_unit),
        "teacher_error_units": stats(wrong_units),
        "teacher_error_unit_count": len(wrong_units),
        "all_one_mask_unit_count": sum(r["all_one_mask"] for r in by_unit),
        "all_one_mask_consistency_failures": sum(r["all_one_mask"] and not r["all_one_consistency_verified"] for r in by_unit),
        "by_dataset": {ds: stats([r for r in by_unit if r["dataset"] == ds]) for ds in DATASETS},
        "paired_cells": by_unit,
        "interpretation_scope": "training-label teacher-correct mask only; does not evaluate automatic teacher-error recognition",
        "legacy_74_04_reference": {
            "mean_balanced_accuracy_pct": 74.04,
            "strict_paired_control": False,
            "reason": "legacy run did not save epoch-10 optimizer, scheduler, RNG, or sampler checkpoint state",
        },
    }
    _atomic_csv(out / "paired_comparison.csv", by_unit)
    _atomic_json(out / "summary.json", summary)
    lines = [
        "# Paired delayed KD teacher-correct-mask experiment", "",
        f"Completed pairs: {len(by_unit)} / 38", "",
        "Primary endpoint: `DELAYED_KD_TCORRECT - DELAYED_KD_ALL` Balanced Accuracy (percentage points).", "",
        f"Overall: mean {summary['overall']['mean_delta_balanced_accuracy']!s}; 95% paired bootstrap CI {summary['overall']['balanced_accuracy_ci95']}; Win/Tie/Loss {summary['overall']['win_tie_loss']}.",
        f"Cells with at least one teacher error: n={summary['teacher_error_unit_count']}, mean {summary['teacher_error_units']['mean_delta_balanced_accuracy']!s}; 95% CI {summary['teacher_error_units']['balanced_accuracy_ci95']}; W/T/L {summary['teacher_error_units']['win_tie_loss']}.", "",
        "| Dataset | n | Δ Balanced Accuracy | 95% paired bootstrap CI | W/T/L | Δ Accuracy | Δ Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        subset = [r for r in by_unit if r["dataset"] == ds]
        s = summary["by_dataset"][ds]
        lines.append(f"| {ds} | {s['n']} | {s['mean_delta_balanced_accuracy']:.3f} | {s['balanced_accuracy_ci95']} | {s['win_tie_loss']} | {s['mean_delta_accuracy']:.3f} | {s['mean_delta_kappa']:.4f} |")
    lines += ["", f"Teacher-correct mask: {sum(int(r['correct_teacher_count']) for r in unit_rows)} / {sum(int(r['train_count']) for r in unit_rows)} training samples; error samples={sum(int(r['teacher_wrong_count']) for r in unit_rows)}; error-containing units={sum(int(r['teacher_wrong_count']) > 0 for r in unit_rows)}; all-one units={sum(bool(r['mask_all_ones']) for r in unit_rows)}.", "", "All-one mask cells are consistency checks: both continuations are expected to match exactly. The mask uses the known training label and says nothing about automatic error detection.", "", "Historical 74.04% delayed-KD is shown only as a reference because its epoch-10 training state was not saved.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return summary


def _git_snapshot():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"<unavailable: {exc}>"
    return {"commit": run("rev-parse", "HEAD"), "status_short": run("status", "--short")}


def main():
    args = _args()
    cfg = _load_config(args.config)
    out = require_external_output(cfg["output_dir"])
    if out.exists() and any(out.iterdir()) and not (args.resume or args.force):
        raise FileExistsError(f"{out} is not empty; use --resume or --force")
    out.mkdir(parents=True, exist_ok=True)
    device = _device(args.gpu)
    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    _atomic_json(out / "config_resolved.json", cfg)
    unit_rows, mask_manifest = _prepare_masks(cfg)
    _atomic_json(out / "execution_provenance.json", {
        "git": _git_snapshot(), "device": str(device),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "config_path": str((ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)),
        "mask_manifest": mask_manifest,
        "legacy_delayed_kd_strict_pairing": False,
        "legacy_delayed_kd_reason": "No saved epoch-10 optimizer/scheduler/RNG/sampler checkpoint to verify.",
    })
    if args.preflight_only:
        print(json.dumps(mask_manifest, indent=2))
        return
    if args.smoke_test:
        if (out / "results_per_run.csv").exists() and not args.resume:
            raise RuntimeError("refusing to run smoke test after formal results exist unless --resume is specified")
        all_one = next((r for r in unit_rows if bool(r["mask_all_ones"])), None)
        with_errors = next((r for r in unit_rows if int(r["teacher_wrong_count"]) > 0), None)
        if all_one is None or with_errors is None:
            raise RuntimeError("smoke test needs one all-one cell and one teacher-error cell")
        for row in (all_one, with_errors):
            key = _run_key(row["dataset"], int(row["subject_index"]), int(cfg["seed"]))
            unit = _load_training_unit(cfg, row, device)
            warmup, warmup_sha = _create_warmup(out / "smoke_test", key, row["dataset"],
                                                 int(row["subject_index"]), cfg, unit,
                                                 device, max_epochs=10)
            if warmup["epoch"] != 10 or warmup["next_batch_index"] != 0:
                raise RuntimeError(f"{key}: smoke checkpoint is not at epoch 10 boundary")
            branch_rows = []
            for condition in CONDITIONS:
                branch_rows.append(_train_branch(out, key, condition, warmup,
                                                 warmup_sha, unit, cfg, device,
                                                 max_kd_epochs=3, smoke=True))
            if bool(row["mask_all_ones"]):
                a, b = branch_rows
                if a["final_student_state_hash"] != b["final_student_state_hash"]:
                    raise RuntimeError(f"{key}: smoke all-one branches did not match")
                p0, p1 = [_branch_paths(out, key, c, smoke=True)["prediction"] for c in CONDITIONS]
                with np.load(resolve_local_file(p0), allow_pickle=False) as a0, np.load(resolve_local_file(p1), allow_pickle=False) as a1:
                    if not np.array_equal(a0["test_logits"], a1["test_logits"]):
                        raise RuntimeError(f"{key}: smoke all-one test logits differ")
        _atomic_json(out / "smoke_test" / "smoke_status.json", {
            "status": "passed", "cells": [all_one["dataset"] + " S" + str(all_one["subject"]),
                                             with_errors["dataset"] + " S" + str(with_errors["subject"])],
            "warmup_epochs": 10, "continuation_epochs_per_branch": 3,
            "all_one_pair_bitwise_state_match": True,
        })
        print("SMOKE TEST PASSED", flush=True)
        return

    result_path = out / "results_per_run.csv"
    result_rows = _read_rows(result_path)
    by_key = {(r.get("dataset"), int(r.get("subject", 0)), r.get("condition")): r
              for r in result_rows if r.get("subject")}
    for row in unit_rows:
        dataset_name = row["dataset"]
        subject = int(row["subject_index"])
        key = _run_key(dataset_name, subject, int(cfg["seed"]))
        unit = _load_training_unit(cfg, row, device)
        warmup, warmup_sha = _create_warmup(out, key, dataset_name, subject,
                                            cfg, unit, device)
        if warmup["epoch"] != int(cfg["warmup_epochs"]):
            raise RuntimeError(f"{key}: common CE warm-up did not end at epoch 10")
        if warmup["batch_order_hash"] != unit["batch_order_hash"]:
            raise RuntimeError(f"{key}: epoch-10 checkpoint batch-order hash mismatch")
        if warmup["split_uid_hash"] != unit["split_uid_hash"]:
            raise RuntimeError(f"{key}: epoch-10 checkpoint UID split hash mismatch")
        for condition in CONDITIONS:
            prior = by_key.get((dataset_name, subject + 1, condition)) if args.resume else None
            if prior and prior.get("status") == "complete" and int(prior.get("epochs_completed", 0)) == 100:
                complete = _branch_complete(_branch_paths(out, key, condition))
                if complete is not None:
                    by_key[(dataset_name, subject + 1, condition)] = complete
                    continue
            trained = _train_branch(out, key, condition, warmup, warmup_sha,
                                    unit, cfg, device)
            by_key[(dataset_name, subject + 1, condition)] = trained
            result_rows = list(by_key.values())
            _persist_rows(out, result_rows)
        if bool(row["mask_all_ones"]):
            _verify_all_one_pair(out, key, warmup_sha)
            for condition in CONDITIONS:
                pair_row = by_key[(dataset_name, subject + 1, condition)]
                pair_row["all_one_consistency_verified"] = True
                paths = _branch_paths(out, key, condition)
                result_json = json.loads(resolve_local_file(paths["result"]).read_text())
                result_json["all_one_consistency_verified"] = True
                _atomic_json(paths["result"], result_json)
            _persist_rows(out, list(by_key.values()))
        print(f"CELL COMPLETE {key}: all BA={by_key[(dataset_name, subject + 1, CONDITIONS[0])]['balanced_accuracy']:.3f} tcorrect BA={by_key[(dataset_name, subject + 1, CONDITIONS[1])]['balanced_accuracy']:.3f}", flush=True)

    result_rows = list(by_key.values())
    _persist_rows(out, result_rows)
    _summarize(out, unit_rows, result_rows)
    print(f"EXPERIMENT COMPLETE: {len(result_rows)} conditions at {out}", flush=True)


if __name__ == "__main__":
    main()
