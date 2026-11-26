"""Train/evaluate the author-checkpoint NeuroGPT encoder in current LOSO.

Examples (fold IDs are zero-based):
  python -B -m experiments.finetune.run_neurogpt_loso --dataset BNCI2014001-4 \
      --fold 0 --seed 666 --gpu 0 --smoke
  python -B -m experiments.finetune.run_neurogpt_loso --dataset BNCI2014001-4 \
      --fold 0 --seed 666 --gpu 0 --resume
  python -B -m experiments.finetune.run_neurogpt_loso --summarize

The first invocation performs a source-only two-step smoke. Formal cells use
the configured 10,000 optimizer steps and evaluate the held-out subject once,
after the final step. Training checkpoints and per-UID prediction artifacts
are stored outside the checkout.
"""
from __future__ import annotations

import os

# Set backend limits before importing NumPy, SciPy, or PyTorch.
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import config as project_config
from data.loso_source import load_dataset_subjects
from experiments.storage import (
    PROJECT_DATA_ROOT,
    REPORTS_ROOT,
    require_external_output,
    require_report_output,
    resolve_local_file,
)
from models import get_adapter


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/loso/neurogpt.yaml"
EXPECTED_CHANNELS_22 = (
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
)
ARTIFACT_ROOT = PROJECT_DATA_ROOT / "results/neurogpt_loso"
WEIGHT_ROOT = PROJECT_DATA_ROOT / "weights/neurogpt_loso"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    files = (
        PROJECT_ROOT / "models/neurogpt/neurogpt.py",
        PROJECT_ROOT / "models/neurogpt/adapter.py",
        Path(__file__).resolve(),
    )
    return _json_hash({path.name: _sha256_file(path) for path in files})


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    require_external_output(path)
    tmp = require_external_output(path.with_name(path.name + f".{os.getpid()}.tmp"))
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class _BatchSchedule:
    """Deterministic shuffled, no-replacement source batches with tail batches."""
    def __init__(self, n: int, batch_size: int, seed: int):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + 100003)
        self.order = torch.empty(0, dtype=torch.long)
        self.position = 0
        self.epoch = 0

    def next(self) -> torch.Tensor:
        if self.order.numel() == 0 or self.position >= self.n:
            self.order = torch.randperm(self.n, generator=self.generator)
            self.position = 0
            self.epoch += 1
        end = min(self.position + self.batch_size, self.n)
        index = self.order[self.position:end]
        self.position = end
        return index

    def state_dict(self):
        return {
            "generator_state": self.generator.get_state(),
            "order": self.order,
            "position": self.position,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        self.generator.set_state(state["generator_state"])
        self.order = state["order"].clone().long()
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])


def _lr_at_step(step: int, cfg: dict) -> float:
    base = float(cfg["learning_rate"])
    total = int(cfg["max_steps"])
    warmup = int(cfg["warmup_steps"])
    if step < warmup:
        return base * float(step + 1) / max(warmup, 1)
    return base * max(0.0, float(total - step) / max(total - warmup, 1))


def _save_training_state(path, *, model, optimizer, next_step, schedule,
                         identity, device):
    payload = {
        "identity": identity,
        "next_step": int(next_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    _atomic_torch_save(path, payload)


def _restore_training_state(path, *, model, optimizer, schedule,
                            identity, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("identity") != identity:
        raise RuntimeError(f"resume checkpoint identity mismatch: {path}")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["next_step"])


def _metrics(y, logits):
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

    y = np.asarray(y, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
    }, probs, pred


def _load_spec(config_path: Path, dataset: str):
    exp = yaml.safe_load(Path(resolve_local_file(config_path)).read_text())
    model_path = PROJECT_ROOT / "configs/models/neurogpt.yaml"
    model_spec = yaml.safe_load(model_path.read_text())
    ds_spec = project_config.load_dataset_config(dataset)
    if dataset not in exp["datasets"]:
        raise ValueError(f"dataset {dataset!r} is not enabled in {config_path}")
    if dataset not in {"AlexMI"}:
        current = project_config.load_loso_dataset_config(dataset)
        if current.get("source_directory", "").split("/")[-1] != "wide_fre_band":
            raise RuntimeError(f"{dataset}: LOSO source is not wide_fre_band")
    return exp, model_spec, ds_spec


def _source_channel_order_verified(dataset: str) -> bool:
    if dataset not in {"BNCI2014001", "BNCI2014001-4"}:
        return False
    from data.channels import BNCI2014001_chn_names
    actual = tuple(str(name).upper() for name in BNCI2014001_chn_names)
    if actual != EXPECTED_CHANNELS_22:
        raise RuntimeError(
            "current BNCI2014001 channel order no longer matches the author matrix input order")
    return True


def _adapter_for(dataset, input_channels, device, exp, model_spec):
    matrix_path = exp["channel_adaptation"]["fixed_official_matrix"]["path"]
    matrix_sha = exp["channel_adaptation"]["fixed_official_matrix"]["sha256"]
    channel_order_verified = _source_channel_order_verified(dataset)
    adapter_cfg = {
        "dataset_name": dataset,
        "input_channels": int(input_channels),
        "channel_matrix": matrix_path,
        "channel_order_verified": channel_order_verified,
        "head_dropout": float(model_spec["downstream_head"]["dropout"]),
    }
    if channel_order_verified:
        if _sha256_file(matrix_path) != matrix_sha:
            raise RuntimeError(f"official NeuroGPT montage matrix SHA-256 mismatch: {matrix_path}")
    return get_adapter("neurogpt", device=device, **adapter_cfg), adapter_cfg


def _cell_paths(run_id, dataset, fold, seed):
    cell = f"subject_{fold + 1:02d}/seed_{seed}"
    artifact = ARTIFACT_ROOT / run_id / dataset / cell
    checkpoint = WEIGHT_ROOT / run_id / dataset / cell / "training_state.pt"
    return artifact, checkpoint


def _write_report_summary(run_id):
    """Rebuild the checkout-local CSV from identity-bearing external manifests."""
    rows = []
    root = ARTIFACT_ROOT / run_id
    for manifest_path in sorted(root.glob("*/subject_*/seed_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            continue
        metrics = manifest["target_metrics"]
        rows.append({
            "dataset": manifest["dataset"],
            "test_subject": manifest["test_subject"],
            "seed": manifest["seed"],
            "n_train": manifest["n_train"],
            "n_test": manifest["n_test"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "training_steps": manifest["training_steps"],
            "elapsed_sec": manifest["elapsed_sec"],
        })
    report_dir = REPORTS_ROOT / "loso/neurogpt" / run_id
    summary_path = require_report_output(report_dir / "per_subject.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / ".summary.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        temporary = summary_path.with_name(summary_path.name + f".{os.getpid()}.tmp")
        columns = ["dataset", "test_subject", "seed", "n_train", "n_test",
                   "accuracy", "balanced_accuracy", "kappa", "training_steps",
                   "elapsed_sec"]
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (
                row["dataset"], row["seed"], row["test_subject"])))
        os.replace(temporary, summary_path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return summary_path, len(rows)


def _cell_identity(dataset, fold, seed, data_obj, exp, model_spec,
                   dataset_spec, adapter_cfg, weight_sha, steps):
    train_uid = np.asarray(data_obj.trial_uid).astype(str)
    return {
        "dataset": dataset,
        "fold_index_base": 0,
        "test_subject_zero_based": int(fold),
        "seed": int(seed),
        "train_uid_sha256": _array_hash(train_uid),
        "train_label_sha256": _array_hash(data_obj.y),
        "source_manifest_sha256": data_obj.source_manifest_sha256,
        "source_identity": data_obj.identity,
        "label_sha256": _array_hash(data_obj.y),
        "input_shape": list(data_obj.x.shape[1:]),
        "num_classes": int(len(data_obj.class_mapping)),
        "training_config_sha256": _json_hash(exp["training"]),
        "experiment_config_sha256": _json_hash(exp),
        "model_config_sha256": _json_hash(model_spec),
        "dataset_config_sha256": _json_hash(dataset_spec),
        "adapter_config": adapter_cfg,
        "pretrained_checkpoint_sha256": weight_sha,
        "implementation_sha256": _implementation_hash(),
        "max_steps": int(steps),
        "normalization": "per_trial_per_channel_zscore_over_1000_timepoints",
        "test_policy": "target_used_once_after_final_step",
    }


def _train_one(args, exp, model_spec, dataset_spec):
    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested but CUDA is unavailable in this environment")
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu} is outside visible CUDA device range")
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    seed = int(args.seed)
    fold = int(args.fold)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    exp_file_sha = _sha256_file(config_path)
    if args.expected_config_sha256 and exp_file_sha != args.expected_config_sha256:
        raise RuntimeError("experiment YAML changed after queue submission")
    model_yaml_path = PROJECT_ROOT / "configs/models/neurogpt.yaml"
    model_file_sha = _sha256_file(model_yaml_path)
    if args.expected_model_config_sha256 and model_file_sha != args.expected_model_config_sha256:
        raise RuntimeError("NeuroGPT model YAML changed after queue submission")
    dataset_yaml_path = PROJECT_ROOT / f"configs/datasets/{dataset}.yaml"
    dataset_file_sha = _sha256_file(dataset_yaml_path)
    if (args.expected_dataset_config_sha256
            and dataset_file_sha != args.expected_dataset_config_sha256):
        raise RuntimeError(f"{dataset} YAML changed after queue submission")
    implementation_sha = _implementation_hash()
    if (args.expected_implementation_sha256
            and implementation_sha != args.expected_implementation_sha256):
        raise RuntimeError("NeuroGPT implementation changed after queue submission")
    n_subjects = int(dataset_spec["num_subjects"])
    if fold < 0 or fold >= n_subjects:
        raise ValueError(f"fold must be zero-based in [0,{n_subjects - 1}]")
    train_subjects_expected = sorted(set(range(n_subjects)) - {fold})
    source = load_dataset_subjects(dataset, train_subjects_expected)
    subjects = np.asarray(source.subjects, dtype=np.int64)
    if not len(subjects) or np.any(subjects == fold):
        raise RuntimeError(f"{dataset} fold={fold}: target subject entered source training data")
    if set(np.unique(subjects).tolist()) != set(train_subjects_expected):
        raise RuntimeError(f"{dataset} fold={fold}: source-subject partition mismatch")
    class_ids = sorted(np.unique(np.asarray(source.y, dtype=np.int64)).tolist())
    if class_ids != list(range(len(source.class_mapping))):
        raise RuntimeError(f"{dataset}: labels are not contiguous from zero: {class_ids}")
    if len(source.class_mapping) != int(dataset_spec["num_classes"]):
        raise RuntimeError(f"{dataset}: source class count differs from dataset config")

    official = exp["official_reference"]
    weight_path = Path(project_config.weight_path("neurogpt"))
    if not weight_path.is_file():
        raise FileNotFoundError(f"official NeuroGPT checkpoint missing: {weight_path}")
    weight_sha = _sha256_file(weight_path)
    if weight_sha != official["weight_sha256"]:
        raise RuntimeError(
            f"official NeuroGPT checkpoint identity mismatch: {weight_path} sha256={weight_sha}")
    steps = 2 if args.smoke else int(exp["training"]["max_steps"])
    train_x = np.asarray(source.x, dtype=np.float32)
    train_y = np.asarray(source.y, dtype=np.int64)
    train_subject_ids = subjects
    adapter, adapter_cfg = _adapter_for(
        dataset, train_x.shape[1], device, exp, model_spec)
    identity = _cell_identity(dataset, fold, seed, source, exp,
                              model_spec, dataset_spec, adapter_cfg,
                              weight_sha, steps)
    identity["experiment_yaml_sha256"] = exp_file_sha
    identity["model_yaml_sha256"] = model_file_sha
    identity["dataset_yaml_sha256"] = dataset_file_sha
    identity_hash = _json_hash(identity)
    run_id = args.run_id or exp["run_id"]
    if args.smoke:
        run_id = f"{run_id}_smoke_s{seed}_fold{fold:02d}"
    artifact_dir, checkpoint_path = _cell_paths(run_id, dataset, fold, seed)
    result_path = artifact_dir / "target_predictions.npz"
    manifest_path = artifact_dir / "manifest.json"
    progress_path = Path(args.progress) if args.progress else None

    if not args.smoke and manifest_path.is_file() and result_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("status") == "complete"
                and manifest.get("identity_sha256") == identity_hash):
            print(f"[reuse] {dataset} S{fold + 1:02d} seed={seed}: identity-matched complete result",
                  flush=True)
            return manifest["target_metrics"]
        raise RuntimeError(f"completed artifact exists with different identity: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)
    model = adapter.build(len(source.class_mapping))
    expected_encoder_count = 0
    if adapter.pretrained_report:
        expected_encoder_count = int(adapter.pretrained_report["loaded_encoder_tensors"])
    if expected_encoder_count < 1:
        raise RuntimeError("official NeuroGPT encoder weights were not loaded")
    # Convert all source inputs independently per trial; no cross-trial fit.
    x_train = adapter.preprocess(train_x)
    y_tensor = torch.as_tensor(train_y, dtype=torch.long)
    training_cfg = dict(exp["training"])
    training_cfg["max_steps"] = steps
    batch_size = int(training_cfg["batch_size"])
    schedule = _BatchSchedule(len(y_tensor), batch_size, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg["learning_rate"]),
        betas=tuple(float(x) for x in training_cfg["betas"]),
        eps=float(training_cfg["epsilon"]),
        weight_decay=float(training_cfg["weight_decay"]),
        foreach=False, fused=False,
    )
    criterion = nn.CrossEntropyLoss()
    first_step = 0
    if not args.smoke and args.resume and checkpoint_path.is_file():
        first_step = _restore_training_state(
            checkpoint_path, model=model, optimizer=optimizer,
            schedule=schedule, identity=identity, device=device)
        if first_step > steps:
            raise RuntimeError("resume state is beyond the configured final step")
        print(f"[resume] {dataset} S{fold + 1:02d} seed={seed} next_step={first_step}",
              flush=True)
    elif not args.smoke and checkpoint_path.exists():
        raise RuntimeError(
            f"training state exists; pass --resume to continue without restarting: {checkpoint_path}")
    model.train()
    started = time.time()
    running_loss = 0.0
    completed_step = first_step
    checkpoint_interval = int(training_cfg["checkpoint_interval_steps"])
    print(
        f"[run] NeuroGPT {dataset} S{fold + 1:02d} seed={seed} device={device} "
        f"train_n={len(train_y)} steps={steps} batch={batch_size} "
        f"pretrained_encoder_tensors={expected_encoder_count}", flush=True)
    for step in range(first_step, steps):
        if args.stop_file and Path(args.stop_file).exists():
            if not args.smoke:
                _save_training_state(
                    checkpoint_path, model=model, optimizer=optimizer,
                    next_step=step, schedule=schedule,
                    identity=identity, device=device)
            raise InterruptedError("queue stop file requested an orderly checkpoint")
        indices = schedule.next()
        xb = x_train[indices].to(device, non_blocking=False)
        yb = y_tensor[indices].to(device, non_blocking=False)
        learning_rate = _lr_at_step(step, training_cfg)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        _, logits = adapter.forward(model, xb)
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite NeuroGPT CE at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training_cfg["gradient_clip_norm"]))
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        completed_step = step + 1
        if progress_path and (completed_step % 100 == 0 or completed_step == steps):
            _atomic_json(progress_path, {
                "dataset": dataset, "fold": fold, "seed": seed,
                "step": completed_step, "steps": steps,
                "last_loss": float(loss.detach().cpu()),
                "mean_loss_since_start": running_loss / max(completed_step - first_step, 1),
                "learning_rate": learning_rate,
                "elapsed_sec": round(time.time() - started, 2),
            })
        if not args.smoke and (completed_step % checkpoint_interval == 0
                               or completed_step == steps):
            _save_training_state(
                checkpoint_path, model=model, optimizer=optimizer,
                next_step=completed_step, schedule=schedule,
                identity=identity, device=device)
        if completed_step % 1000 == 0 or completed_step == steps:
            print(f"[progress] {dataset} S{fold + 1:02d} seed={seed} "
                  f"step={completed_step}/{steps} loss={float(loss):.5f} "
                  f"lr={learning_rate:.3g}", flush=True)

    if args.smoke:
        # The smoke never indexes target labels, signals, or UIDs.
        _atomic_json(artifact_dir / "smoke.json", {
            "status": "passed", "identity_sha256": identity_hash,
            "dataset": dataset, "fold": fold, "seed": seed,
            "source_only": True, "steps": steps,
            "last_loss": float(loss.detach().cpu()),
            "encoder_weights_loaded": expected_encoder_count,
            "channel_mapping": adapter.pretrained_report["channel_mapping"],
        })
        checkpoint_path.unlink(missing_ok=True)
        print(f"[smoke] passed: loss={float(loss):.5f} gradients finite; target unused",
              flush=True)
        return {"status": "smoke_passed", "last_loss": float(loss.detach().cpu())}

    # Only after the final optimizer step do we materialize target-subject EEG,
    # labels, and UIDs. The source-only smoke never calls this reader.
    test_source = load_dataset_subjects(dataset, [fold])
    if set(np.unique(np.asarray(test_source.subjects, dtype=np.int64)).tolist()) != {fold}:
        raise RuntimeError(f"{dataset} fold={fold}: target reader returned another subject")
    if (test_source.source_manifest_sha256 != source.source_manifest_sha256
            or test_source.class_mapping != source.class_mapping):
        raise RuntimeError(f"{dataset} fold={fold}: source/target identities differ")
    model.eval()
    test_y = np.asarray(test_source.y, dtype=np.int64)
    test_uid = np.asarray(test_source.trial_uid, dtype=str)
    with torch.no_grad():
        # This is the only point at which the held-out subject is passed to the model.
        features, logits = adapter.infer(model, np.asarray(test_source.x, dtype=np.float32))
    metrics, probs, pred = _metrics(test_y, logits)
    elapsed_sec = round(time.time() - started, 3)
    target_uid_hash = _array_hash(test_uid)
    temp_npz = require_external_output(result_path.with_name(
        result_path.name + f".{os.getpid()}.tmp"))
    try:
        with temp_npz.open("wb") as stream:
            np.savez_compressed(
                stream, trial_uid=test_uid, y=test_y, pred=pred,
                logits=np.asarray(logits, dtype=np.float32),
                probs=np.asarray(probs, dtype=np.float32),
                features=np.asarray(features, dtype=np.float32),
            )
        os.replace(temp_npz, result_path)
    finally:
        temp_npz.unlink(missing_ok=True)
    manifest = {
        "status": "complete",
        "run_id": run_id,
        "dataset": dataset,
        "protocol": "LOSO",
        "test_subject": fold + 1,
        "test_subject_zero_based": fold,
        "seed": seed,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "train_subjects": sorted(int(x) + 1 for x in np.unique(train_subject_ids)),
        "source_manifest_sha256": source.source_manifest_sha256,
        "source_identity": source.identity,
        "source_path": str(source.source_path),
        "source_band": dataset_spec.get("loso", {}).get("source_bandpass_hz"),
        "session": exp["source_data"]["session_by_dataset"][dataset],
        "class_mapping": source.class_mapping,
        "train_uid_sha256": identity["train_uid_sha256"],
        "test_uid_sha256": target_uid_hash,
        "model": "NeuroGPT encoder-only",
        "model_config": model_spec,
        "training_config": training_cfg,
        "adapter_config": adapter_cfg,
        "pretrained_checkpoint": str(weight_path),
        "pretrained_checkpoint_sha256": weight_sha,
        "pretrained_load_report": adapter.pretrained_report,
        "identity": identity,
        "identity_sha256": identity_hash,
        "training_steps": completed_step,
        "selection_policy": "fixed_final_optimizer_step_no_validation",
        "target_metrics": metrics,
        "target_prediction_artifact": str(result_path),
        "elapsed_sec": elapsed_sec,
    }
    _atomic_json(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    # Successful folds retain only a compact completion record, not optimizer/model tensors.
    completion = {
        "status": "complete", "identity_sha256": identity_hash,
        "terminal_step": completed_step, "seed": seed, "fold": fold,
        "target_uid_sha256": target_uid_hash,
        "prediction_artifact": str(result_path),
    }
    _atomic_json(artifact_dir / "completed_state.json", completion)
    summary_path, completed = _write_report_summary(run_id)
    print(f"[done] {dataset} S{fold + 1:02d} seed={seed} "
          f"acc={metrics['accuracy']:.4f} bacc={metrics['balanced_accuracy']:.4f} "
          f"kappa={metrics['kappa']:.4f} elapsed={elapsed_sec:.1f}s "
          f"summary={summary_path} complete_cells={completed}", flush=True)
    del model, adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _write_xlsx(run_id):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    report_dir = REPORTS_ROOT / "loso/neurogpt" / run_id
    csv_path = require_report_output(report_dir / "per_subject.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"run summary CSV is missing: {csv_path}")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "LOSO Results"
    columns = ["dataset", "test_subject", "seed", "n_train", "n_test",
               "Accuracy (%)", "Balanced Accuracy (%)", "Kappa",
               "training_steps", "elapsed_sec"]
    sheet.append(columns)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in rows:
        sheet.append([
            row["dataset"], int(row["test_subject"]), int(row["seed"]),
            int(row["n_train"]), int(row["n_test"]),
            100.0 * float(row["accuracy"]),
            100.0 * float(row["balanced_accuracy"]),
            float(row["kappa"]), int(row["training_steps"]), float(row["elapsed_sec"]),
        ])
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False
    for col, width in enumerate((18, 14, 12, 12, 12, 18, 26, 12, 18, 15), 1):
        sheet.column_dimensions[get_column_letter(col)].width = width
    for row in sheet.iter_rows(min_row=2):
        for col in (6, 7):
            row[col - 1].number_format = "0.00"
        for cell in row:
            cell.alignment = Alignment(horizontal="center")
    notes = workbook.create_sheet("Notes")
    notes.sheet_view.showGridLines = False
    notes.append(["Item", "Definition"])
    notes.append(["Metric", "Accuracy and Balanced Accuracy are percentages; Kappa is unitless."])
    notes.append(["Aggregation", "One row per target subject and seed; summarize only complete subject/seed cells."])
    notes.append(["Split", "LOSO: one complete subject held out; all other subjects train; no validation set."])
    notes.append(["Prediction", "Encoder-only NeuroGPT, two 500-sample chunks concatenated before the 256-32-class head."])
    notes.append(["Adaptation", "Official 22x22 matrix for BNCI2014001 tasks; trainable C-native to 22 projection for other channel counts."])
    for cell in notes[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    notes.column_dimensions["A"].width = 20
    notes.column_dimensions["B"].width = 105
    for row in notes.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    path = require_report_output(report_dir / "summary.xlsx")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    workbook.save(temporary)
    os.replace(temporary, path)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id")
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int, help="zero-based held-out subject index")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--device", help="CPU smoke override; formal queue uses CUDA")
    parser.add_argument("--smoke", action="store_true", help="two source-only optimizer steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-model-config-sha256")
    parser.add_argument("--expected-dataset-config-sha256")
    parser.add_argument("--expected-implementation-sha256")
    parser.add_argument("--progress")
    parser.add_argument("--stop-file")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        exp = yaml.safe_load(Path(resolve_local_file(args.config)).read_text())
        run_id = args.run_id or exp["run_id"]
        csv_path, count = _write_report_summary(run_id)
        xlsx_path = _write_xlsx(run_id)
        print(f"[summary] cells={count} csv={csv_path} xlsx={xlsx_path}", flush=True)
        return 0
    if args.dataset is None or args.fold is None or args.seed is None:
        raise ValueError("--dataset, --fold (zero-based), and --seed are required")
    exp, model_spec, dataset_spec = _load_spec(Path(args.config), args.dataset)
    if int(args.seed) not in [int(s) for s in exp["seeds"]]:
        raise ValueError(f"seed {args.seed} is not in configured seeds {exp['seeds']}")
    try:
        _train_one(args, exp, model_spec, dataset_spec)
    except InterruptedError as exc:
        print(f"[paused] {exc}", flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
