"""Train a fresh IFNet from B0/E2 probabilities, then evaluate offline soft votes."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BIGSMALLCOLLAB_CPU_THREADS", "TORCH_NUM_THREADS",
              "TORCH_NUM_INTEROP_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from collab.ensemble_distillation import (
    categorical_kl, classification_metrics, mean_probabilities,
    paired_bootstrap, sha256_state_dict, sha256_uids, student_objective,
)
from experiments.storage import (
    DATA_CACHE_ROOT, REPORTS_ROOT, RESULTS_ROOT, WEIGHTS_ROOT,
    require_external_output, require_report_output, resolve_local_file,
)
from models import get_adapter


DEFAULT_CONFIG = ROOT / "configs/experiments/loso/e2_ifnet_distill_boost_ensemble_v1.yaml"
CONDITIONS = ("NEW_CE", "B0_KD", "E2_KD")
CONDITION_TARGET = {"NEW_CE": None, "B0_KD": "p_b0", "E2_KD": "p_e2"}
CODE_FILES = (Path(__file__), ROOT / "collab/ensemble_distillation.py")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(resolve_local_file(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False, default=str, allow_nan=True) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def atomic_torch(path: Path, payload) -> None:
    path = require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict], *, report: bool = False) -> None:
    path = require_report_output(path) if report else require_external_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def set_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def load_config(path: Path, expected_sha256: str | None = None) -> tuple[dict, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"config hash changed: expected {expected_sha256}, got {digest}")
    cfg = yaml.safe_load(path.read_text())
    expected = {
        "protocol_id": "E2_IFNET_DISTILL_BOOST_ENSEMBLE_LOSO_V1",
        "dataset": "BNCI2014004", "student_seed": 668,
        "teacher_bundle_seed": 666,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"config {key} must equal {value!r}")
    training = cfg["training"]
    fixed = {"epochs": 100, "batch_size": 16, "optimizer": "adamw",
             "lr": 0.001, "weight_decay": 0.01,
             "betas": [0.9, 0.999], "amsgrad": False,
             "foreach": False, "fused": False,
             "scheduler": "cosine_per_epoch", "eta_min": 0.0,
             "precision": "fp32", "gradient_accumulation": 1,
             "gradient_clip": None, "num_workers": 0}
    for key, value in fixed.items():
        if training.get(key) != value:
            raise ValueError(f"training.{key} must equal {value!r}")
    kd = cfg["distillation"]
    if (kd.get("conditions") != list(CONDITIONS)
            or float(kd.get("temperature")) != 1.0
            or float(kd.get("lambda_kd")) != 0.5
            or int(kd.get("start_epoch")) != 1
            or float(kd.get("ce_weight")) != 1.0):
        raise ValueError("distillation config does not match the locked pilot")
    return cfg, digest


def roots(cfg: dict, run_id: str) -> dict[str, Path]:
    if not run_id or not run_id.replace("_", "").isalnum():
        raise ValueError(f"invalid run_id: {run_id!r}")
    namespace = cfg["artifacts"]["cache_namespace"]
    report_file = require_report_output(
        REPORTS_ROOT / cfg["artifacts"]["report_namespace"] / run_id / "summary.json")
    return {
        "cache": require_external_output(DATA_CACHE_ROOT / namespace / run_id),
        "weights": require_external_output(WEIGHTS_ROOT / cfg["artifacts"]["weights_namespace"] / run_id),
        "results": require_external_output(RESULTS_ROOT / cfg["artifacts"]["results_namespace"] / run_id),
        "report": report_file.parent,
    }


def _fold_input_dir(cfg: dict, fold: int) -> Path:
    base = Path(cfg["reuse"]["student_cache_root"]).expanduser()
    matches = sorted(base.glob(f"target_{fold + 1:02d}_feedback_*_seed_668_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one seed-668 cache for fold {fold}: {matches}")
    return matches[0]


def _historical_prediction_path(cfg: dict, fold: int) -> Path:
    return Path(cfg["reuse"]["adaboost_root"]) / f"S{fold + 1:02d}" / "predictions.npz"


def _student_model_config(cfg: dict) -> tuple[dict, str]:
    path = Path(cfg["reuse"]["resolved_student_config"])
    resolved = yaml.safe_load(path.read_text())
    model_cfg = dict(resolved["student_config"])
    if (model_cfg.get("dataset_name") != "BNCI2014004"
            or int(model_cfg.get("in_channels", -1)) != 3
            or int(model_cfg.get("samples", -1)) != 1000
            or model_cfg.get("use_filter_bank") is not True
            or model_cfg.get("filter_bank_hz") != [[4.0, 16.0], [16.0, 40.0]]):
        raise RuntimeError("saved IFNet resolved configuration does not match this pilot")
    return model_cfg, sha256_file(path)


def _global_inputs(cfg: dict) -> dict:
    root = Path(cfg["reuse"]["student_input_root"])
    names = {"x": "X.npy", "y": "y.npy", "subjects": "subjects.npy",
             "trials": "trials.csv", "manifest": "manifest.json"}
    paths = {key: root / name for key, name in names.items()}
    expected = cfg["expected_hashes"]
    hash_key = {"x": "global_X.npy", "y": "global_y.npy",
                "subjects": "global_subjects.npy", "trials": "global_trials.csv"}
    actual_hashes = {}
    for key in ("x", "y", "subjects", "trials"):
        actual = sha256_file(paths[key])
        actual_hashes[key] = actual
        if actual != expected[hash_key[key]]:
            raise RuntimeError(f"global IFNet {key} hash mismatch: {actual}")
    manifest = json.loads(paths["manifest"].read_text())
    if (manifest.get("dataset") != "BNCI2014004"
            or manifest.get("model") != "ifnet"
            or manifest.get("ea_regime") != "disabled"
            or manifest.get("input_shape") != [1400, 3, 1000]):
        raise RuntimeError("global IFNet input manifest does not match the configured source")
    return {"paths": paths, "hashes": actual_hashes, "manifest": manifest}


def _historical_run_identity(cfg: dict) -> dict:
    run_path = Path(cfg["reuse"]["historical_run_config"])
    manifest_path = Path(cfg["reuse"]["historical_reuse_manifest"])
    run_record = json.loads(run_path.read_text())
    reuse_manifest = json.loads(manifest_path.read_text())
    raw = run_record.get("config", {})
    expected_teacher = {"name": "mirepnet", "seed": 666,
                        "reuse_seven_source_checkpoint": True}
    if run_record.get("run_id") != "adaboost_bnci2014004_seed666_v1":
        raise RuntimeError("historical AdaBoost run ID does not match the fixed teacher bundle")
    if (raw.get("protocol_id") != "adaboost_mirepnet_ifnet_loso_seed666_v1"
            or raw.get("dataset") != "BNCI2014004"
            or raw.get("session") != "session_3"
            or raw.get("experiment_seed") != 666
            or raw.get("teacher") != expected_teacher
            or raw.get("students", {}).get("learner_seeds") != [666, 667]
            or raw.get("comparisons", {}).get("independent_ensemble") != ["B0", "CE666", "CE667"]):
        raise RuntimeError("historical AdaBoost config does not match the documented E2 bundle")
    if (reuse_manifest.get("run_id") != run_record["run_id"]
            or sorted(reuse_manifest.get("folds", {}).keys()) != [str(i) for i in range(1, 10)]):
        raise RuntimeError("historical AdaBoost reuse manifest is incomplete or mismatched")
    for fold in range(1, 10):
        cell = reuse_manifest["folds"][str(fold)]
        if not all(key in cell for key in ("teacher", "ce", "new_student_learner_seeds")):
            raise RuntimeError(f"historical reuse manifest is incomplete for S{fold:02d}")
        if sorted(cell["ce"].keys()) != ["666", "667"]:
            raise RuntimeError(f"historical CE members are incomplete for S{fold:02d}")
        if cell["new_student_learner_seeds"] != {"B1": 666, "B2": 667}:
            raise RuntimeError(f"historical B1/B2 learner identity mismatch for S{fold:02d}")
        teacher = cell["teacher"]
        provenance = teacher.get("provenance", {})
        teacher_cfg = provenance.get("config", {})
        if (teacher.get("match_status") != "matched_seven_source_train_uids_and_current_source"
                or provenance.get("model") != "mirepnet"
                or provenance.get("seed") != 666
                or provenance.get("epochs") != 10
                or teacher_cfg.get("lr") != 0.001
                or teacher_cfg.get("apply_EA") is not True):
            raise RuntimeError(f"historical B0 teacher provenance mismatch for S{fold:02d}")
    return {
        "run_id": run_record["run_id"],
        "run_config_path": str(run_path),
        "run_config_sha256": sha256_file(run_path),
        "recorded_config_sha256": run_record.get("config_sha256"),
        "reuse_manifest_path": str(manifest_path),
        "reuse_manifest_sha256": sha256_file(manifest_path),
        "runner_source_sha256": run_record.get("runner_source_sha256"),
        "teacher": expected_teacher,
        "teacher_root": raw["reuse"]["teacher_root"],
        "student_seeds": [666, 667],
        "source_variant": raw.get("source_variant"),
    }


def _read_student_cache(cfg: dict, fold: int) -> tuple[Path, dict, dict]:
    folder = _fold_input_dir(cfg, fold)
    input_path = folder / "student_inputs.npz"
    schedule_path = folder / "train_schedule.npz"
    split_path = folder / "split_manifest.json"
    with np.load(resolve_local_file(input_path), allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    with np.load(resolve_local_file(schedule_path), allow_pickle=False) as source:
        schedule = {key: np.asarray(source[key]) for key in source.files}
    split = json.loads(split_path.read_text())
    for partition in ("train", "feedback"):
        x, y, uids = (arrays[f"{partition}_x"], arrays[f"{partition}_y"],
                      arrays[f"{partition}_uids"])
        if (x.ndim != 3 or x.shape[1:] != (6, 1000)
                or y.shape != (len(x),) or uids.shape != (len(x), 2)
                or not np.isfinite(x).all()):
            raise RuntimeError(f"invalid cached {partition} IFNet inputs in {input_path}")
    if (len(schedule.get("epoch_offsets", [])) != 101
            or len(schedule.get("batch_offsets", [])) < 2
            or schedule.get("indices", np.empty(0)).ndim != 1):
        raise RuntimeError(f"expected the saved 100-epoch schedule in {schedule_path}")
    if int(split.get("target_subject_index_zero_based", -1)) != fold:
        raise RuntimeError(f"split manifest target fold mismatch in {split_path}")
    return folder, arrays, {"schedule": schedule, "split": split,
                            "input_sha256": sha256_file(input_path),
                            "schedule_sha256": sha256_file(schedule_path),
                            "split_sha256": sha256_file(split_path)}


def _batches_for_epoch(schedule: dict, epoch: int) -> list[np.ndarray]:
    batch_offsets = schedule["batch_offsets"].astype(np.int64)
    epoch_offsets = schedule["epoch_offsets"].astype(np.int64)
    indices = schedule["indices"].astype(np.int64)
    begin, end = int(epoch_offsets[epoch]), int(epoch_offsets[epoch + 1])
    return [indices[int(batch_offsets[i]):int(batch_offsets[i + 1])]
            for i in range(begin, end)]


def _verify_schedule(schedule: dict, uids: np.ndarray, batch_size: int) -> str:
    if len(schedule["epoch_offsets"]) != 101:
        raise RuntimeError("saved UID schedule must cover exactly 100 epochs")
    digest = hashlib.sha256()
    n = len(uids)
    for epoch in range(100):
        batches = _batches_for_epoch(schedule, epoch)
        if not batches or sum(map(len, batches)) != n:
            raise RuntimeError(f"schedule epoch {epoch + 1} does not cover every train trial")
        flat = np.concatenate(batches).astype(np.int64)
        if (np.any(flat < 0) or np.any(flat >= n)
                or not np.array_equal(np.sort(flat), np.arange(n, dtype=np.int64))):
            raise RuntimeError(f"schedule epoch {epoch + 1} repeats or omits a train trial")
        if any(len(batch) > batch_size or len(batch) == 0 for batch in batches):
            raise RuntimeError(f"invalid batch size in schedule epoch {epoch + 1}")
        for batch in batches:
            digest.update(np.ascontiguousarray(uids[batch], dtype=np.int64).tobytes())
    actual = digest.hexdigest()
    recorded = str(schedule["uid_schedule_sha256"].item())
    if recorded != actual:
        raise RuntimeError(f"UID schedule hash mismatch: {recorded} != {actual}")
    return actual


def _load_probability_archive(cfg: dict, fold: int, *, training: bool) -> dict:
    path = _historical_prediction_path(cfg, fold)
    with np.load(resolve_local_file(path), allow_pickle=False) as archive:
        keys = ["train_uids", "train_y", "train_B0_probabilities",
                "train_CE666_probabilities", "train_CE667_probabilities",
                "train_E2_probabilities", "feedback_uids", "feedback_y",
                "feedback_B0_probabilities", "feedback_CE666_probabilities",
                "feedback_CE667_probabilities", "feedback_E2_probabilities"]
        if not training:
            keys += ["test_uids", "test_y", "test_B0_probabilities",
                     "test_CE666_probabilities", "test_CE667_probabilities",
                     "test_B1_probabilities", "test_B2_probabilities",
                     "test_E2_probabilities", "test_E2_prediction", "test_E4_prediction"]
        result = {key: np.asarray(archive[key]) for key in keys}
    for part in ("train", "feedback"):
        p_b0 = result[f"{part}_B0_probabilities"]
        members = [p_b0, result[f"{part}_CE666_probabilities"],
                   result[f"{part}_CE667_probabilities"]]
        p_e2 = mean_probabilities(*members)
        stored = result[f"{part}_E2_probabilities"]
        error = float(np.max(np.abs(p_e2.astype(np.float64) - stored.astype(np.float64))))
        if error > 1e-6:
            raise RuntimeError(f"saved {part} E2 does not match its three members: {error}")
        result[f"{part}_p_e2_rebuilt"] = p_e2
        result[f"{part}_e2_member_max_abs_error"] = error
    return result


def _state_dict_cpu(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _new_model(adapter, model_cfg: dict, init_state: dict, device: torch.device):
    model = adapter.build(2)
    model.load_state_dict(init_state, strict=True)
    model.to(device)
    return model


def _create_initial_state(cfg: dict, cfg_sha: str, model_cfg: dict,
                          model_cfg_sha: str, weight_root: Path) -> tuple[dict, str, Path]:
    path = weight_root / "initial_state_seed668.pt"
    adapter = get_adapter("ifnet", device="cpu", **model_cfg)
    if path.is_file():
        payload = torch.load(resolve_local_file(path), map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        digest = sha256_state_dict(state)
        if (payload.get("student_seed") != int(cfg["student_seed"])
                or payload.get("model_config_sha256") != model_cfg_sha
                or payload.get("state_sha256") != digest):
            raise RuntimeError(f"saved initial IFNet identity mismatch: {path}")
        return state, digest, path
    set_seed(int(cfg["student_seed"]), torch.device("cpu"))
    model = adapter.build(2)
    state = _state_dict_cpu(model)
    digest = sha256_state_dict(state)
    atomic_torch(path, {
        "state_dict": state, "state_sha256": digest,
        "student_seed": int(cfg["student_seed"]),
        "model_config_sha256": model_cfg_sha,
        "resolved_student_config_sha256": sha256_file(cfg["reuse"]["resolved_student_config"]),
        "config_sha256": cfg_sha,
    })
    return state, digest, path


def _global_uid_lookup(trials_path: Path) -> dict[tuple[int, int], int]:
    lookup = {}
    with trials_path.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            uid = (int(row["subject_zero_based"]), int(row["source_row"]))
            if uid in lookup:
                raise RuntimeError(f"duplicate global trial UID: {uid}")
            lookup[uid] = index
    return lookup


def _validate_train_cache_against_export(cfg: dict, fold: int, arrays: dict,
                                         model_cfg: dict, global_meta: dict) -> dict:
    paths = global_meta["paths"]
    X = np.load(resolve_local_file(paths["x"]), mmap_mode="r", allow_pickle=False)
    y = np.load(resolve_local_file(paths["y"]), mmap_mode="r", allow_pickle=False)
    lookup = _global_uid_lookup(paths["trials"])
    adapter = get_adapter("ifnet", device="cpu", **model_cfg)
    checks = {}
    for partition in ("train", "feedback"):
        uids = np.asarray(arrays[f"{partition}_uids"], dtype=np.int64)
        try:
            rows = np.asarray([lookup[tuple(map(int, uid))] for uid in uids], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"{partition} UID is absent from exported IFNet inputs: {exc}") from exc
        if not np.array_equal(np.asarray(y[rows], dtype=np.int64), arrays[f"{partition}_y"]):
            raise RuntimeError(f"global exported labels mismatch cached {partition} UIDs in fold {fold}")
        processed = adapter.preprocess(np.asarray(X[rows], dtype=np.float32)).numpy()
        cached = arrays[f"{partition}_x"]
        maximum = float(np.max(np.abs(processed.astype(np.float64) - cached.astype(np.float64))))
        mean = float(np.mean(np.abs(processed.astype(np.float64) - cached.astype(np.float64))))
        if maximum > 1e-5 or mean > 1e-6:
            raise RuntimeError(
                f"IFNet preprocessing mismatch for fold {fold} {partition}: "
                f"max={maximum}, mean={mean}")
        checks[partition] = {"n": int(len(rows)), "max_abs_error": maximum,
                             "mean_abs_error": mean}
    return checks


def preflight(cfg: dict, cfg_sha: str, run_id: str) -> dict:
    out = roots(cfg, run_id)
    model_cfg, model_cfg_sha = _student_model_config(cfg)
    global_meta = _global_inputs(cfg)
    historical_identity = _historical_run_identity(cfg)
    input_records = []
    historical_metrics = {"E2": [], "E4": []}
    no_test_input_access = True
    b1b2_weights = {"B1": 0, "B2": 0}
    for fold in cfg["folds"]:
        folder, arrays, info = _read_student_cache(cfg, int(fold))
        archive = _load_probability_archive(cfg, int(fold), training=False)
        schedule_hash = _verify_schedule(info["schedule"], arrays["train_uids"],
                                         int(cfg["training"]["batch_size"]))
        for part in ("train", "feedback"):
            if not np.array_equal(arrays[f"{part}_uids"], archive[f"{part}_uids"]):
                raise RuntimeError(f"{part} UID mismatch in fold {fold}")
            if not np.array_equal(arrays[f"{part}_y"], archive[f"{part}_y"]):
                raise RuntimeError(f"{part} labels mismatch in fold {fold}")
            for member in ("B0", "CE666", "CE667", "B1", "B2"):
                key = f"{part}_{member}_probabilities"
                # The fold archive already exposes the complete per-trial vectors.
                # Validate these even though the trainer reads only B0/E2 train targets.
                if key not in archive:
                    with np.load(resolve_local_file(_historical_prediction_path(cfg, int(fold))),
                                 allow_pickle=False) as all_predictions:
                        value = np.asarray(all_predictions[key])
                else:
                    value = archive[key]
                if value.shape != (len(arrays[f"{part}_y"]), 2):
                    raise RuntimeError(f"invalid {key} shape in fold {fold}")
                if not np.isfinite(value).all() or np.any(value < 0) or not np.allclose(
                        value.sum(axis=1), 1.0, atol=1e-6, rtol=0):
                    raise RuntimeError(f"invalid probabilities in {key}, fold {fold}")
        for member in ("B0", "CE666", "CE667", "B1", "B2", "E2"):
            key = f"test_{member}_probabilities"
            value = archive[key]
            if (value.shape != (len(archive["test_y"]), 2)
                    or not np.isfinite(value).all() or np.any(value < 0)
                    or not np.allclose(value.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
                raise RuntimeError(f"invalid archived test probability field {key}, fold {fold}")
        if archive["test_uids"].shape != (len(archive["test_y"]), 2):
            raise RuntimeError(f"invalid archived test UID shape in fold {fold}")
        if not np.array_equal(np.argmax(archive["test_E2_probabilities"], axis=1),
                              archive["test_E2_prediction"]):
            raise RuntimeError(f"historical E2 prediction mismatch in fold {fold}")
        test_y = archive["test_y"].astype(np.int64)
        historical_metrics["E2"].append(float(np.mean(
            np.argmax(archive["test_E2_probabilities"], axis=1) == test_y) * 100.0))
        historical_metrics["E4"].append(float(np.mean(
            archive["test_E4_prediction"].astype(np.int64) == test_y) * 100.0))
        # Verify archived B1/B2 learner records and record, but do not restore, their weights.
        for learner in ("B1", "B2"):
            learner_dir = Path(cfg["reuse"]["adaboost_root"]) / f"S{int(fold)+1:02d}" / learner
            if not (learner_dir / "completed_state.json").is_file():
                raise FileNotFoundError(learner_dir / "completed_state.json")
            if not (learner_dir / "history.json").is_file():
                raise FileNotFoundError(learner_dir / "history.json")
            with np.load(resolve_local_file(learner_dir / "predictions.npz"),
                         allow_pickle=False) as learner_predictions:
                with np.load(resolve_local_file(_historical_prediction_path(cfg, int(fold))),
                             allow_pickle=False) as full_archive:
                    main_probability = np.asarray(
                        full_archive[f"train_{learner}_probabilities"])
                if (not np.array_equal(learner_predictions["uids"], archive["train_uids"])
                        or not np.array_equal(learner_predictions["y"], archive["train_y"])
                        or not np.allclose(learner_predictions["train_probabilities"],
                                           main_probability,
                                           atol=1e-7, rtol=0)):
                    raise RuntimeError(f"{learner} learner archive identity mismatch in fold {fold}")
            b1b2_weights[learner] += sum(1 for path in learner_dir.glob("*.pt") if path.is_file())
        preprocessing = _validate_train_cache_against_export(
            cfg, int(fold), arrays, model_cfg, global_meta)
        target_dir = out["cache"] / "folds" / f"S{int(fold)+1:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        p_e2 = archive["train_p_e2_rebuilt"]
        atomic_npz(target_dir / "train_targets.npz",
                   train_uids=arrays["train_uids"], train_y=arrays["train_y"],
                   p_b0=archive["train_B0_probabilities"].astype(np.float32),
                   p_e2=p_e2.astype(np.float32))
        target_sha = sha256_file(target_dir / "train_targets.npz")
        # Feedback probabilities remain diagnostic-only and are not copied to the training cache.
        input_records.append({
            "fold": int(fold), "target_subject_id": int(fold) + 1,
            "feedback_subject_id": int((int(fold) + 1) % 9) + 1,
            "cache_dir": str(folder), "input_sha256": info["input_sha256"],
            "schedule_file_sha256": info["schedule_sha256"],
            "uid_schedule_sha256": schedule_hash,
            "train_uids_sha256": sha256_uids(arrays["train_uids"]),
            "feedback_uids_sha256": sha256_uids(arrays["feedback_uids"]),
            "targets_sha256": target_sha,
            "train_e2_member_max_abs_error": archive["train_e2_member_max_abs_error"],
            "feedback_e2_member_max_abs_error": archive["feedback_e2_member_max_abs_error"],
            "preprocessing_identity": preprocessing,
            "train_n": int(len(arrays["train_y"])),
            "feedback_n": int(len(arrays["feedback_y"])),
            "test_n_from_saved_probabilities": int(len(archive["test_y"])),
        })
    init_state, init_hash, init_path = _create_initial_state(
        cfg, cfg_sha, model_cfg, model_cfg_sha, out["weights"])
    historical_macro = {name: float(np.mean(values)) for name, values in historical_metrics.items()}
    for name, value in historical_macro.items():
        expected_value = float(cfg["expected_reference_accuracy_percent"][name])
        if round(value, 2) != round(expected_value, 2):
            raise RuntimeError(
                f"historical {name} macro Accuracy does not reproduce the plan: "
                f"{value:.6f}% vs {expected_value:.2f}%")
    b1b2_checkpoints = {"B1": 0, "B2": 0}
    # The completion records are fixed historical artifacts; their model tensors were pruned.
    record = {
        "status": "passed", "protocol_id": cfg["protocol_id"], "run_id": run_id,
        "config_sha256": cfg_sha,
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_FILES},
        "created_unix": time.time(), "student_seed": int(cfg["student_seed"]),
        "teacher_bundle_seed": int(cfg["teacher_bundle_seed"]),
        "model_config_sha256": model_cfg_sha,
        "historical_adaboost_identity": historical_identity,
        "historical_macro_accuracy_percent": historical_macro,
        "global_input_hashes": global_meta["hashes"],
        "global_input_manifest": global_meta["manifest"],
        "initial_state_path": str(init_path), "initial_state_sha256": init_hash,
        "folds": input_records, "b1b2_checkpoint_files_found": b1b2_checkpoints,
        "b1b2_completion_records": {"B1": 9, "B2": 9},
        "test_X_used": False,
        "test_labels_read_only_from_historical_prediction_archive_for_identity_check": True,
        "training_targets_contain_only_D_train": True,
        "note": ("B1/B2 checkpoints were not restored or retrained. New-student training "
                 "will read only D_train soft targets and may use D_feedback for read-only diagnostics."),
    }
    atomic_json(out["results"] / "preflight.json", record)
    atomic_json(out["report"] / "preflight.json", record)
    print(f"[preflight] PASS run={run_id} folds={len(input_records)}/9 "
          f"train_targets=D_train_only test_X_loaded={not no_test_input_access}", flush=True)
    return record


def _load_training_fold(cfg: dict, fold: int, out: dict) -> dict:
    folder, arrays, info = _read_student_cache(cfg, fold)
    archive = _load_probability_archive(cfg, fold, training=True)
    for part in ("train", "feedback"):
        if not np.array_equal(arrays[f"{part}_uids"], archive[f"{part}_uids"]):
            raise RuntimeError(f"{part} UID mismatch for fold {fold}")
        if not np.array_equal(arrays[f"{part}_y"], archive[f"{part}_y"]):
            raise RuntimeError(f"{part} y mismatch for fold {fold}")
    record = json.loads((out["results"] / "preflight.json").read_text())
    current_code = {str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_FILES}
    if record.get("code_sha256") != current_code:
        raise RuntimeError("implementation code changed after preflight; rerun --preflight")
    fold_record = next((row for row in record["folds"] if row["fold"] == fold), None)
    if fold_record is None or fold_record["input_sha256"] != info["input_sha256"]:
        raise RuntimeError(f"fold {fold} cache changed after preflight")
    target_path = out["cache"] / "folds" / f"S{fold+1:02d}" / "train_targets.npz"
    with np.load(resolve_local_file(target_path), allow_pickle=False) as targets:
        target = {key: np.asarray(targets[key]) for key in targets.files}
    if (not np.array_equal(target["train_uids"], arrays["train_uids"])
            or not np.array_equal(target["train_y"], arrays["train_y"])):
        raise RuntimeError(f"cached train-only target identity mismatch for fold {fold}")
    if not np.array_equal(target["p_b0"], archive["train_B0_probabilities"]):
        raise RuntimeError(f"cached B0 soft targets changed for fold {fold}")
    if not np.allclose(target["p_e2"], archive["train_p_e2_rebuilt"], atol=1e-7, rtol=0):
        raise RuntimeError(f"cached E2 soft targets changed for fold {fold}")
    schedule_hash = _verify_schedule(info["schedule"], arrays["train_uids"],
                                     int(cfg["training"]["batch_size"]))
    if schedule_hash != fold_record["uid_schedule_sha256"]:
        raise RuntimeError(f"saved UID schedule changed after preflight for fold {fold}")
    return {"folder": folder, "arrays": arrays, "info": info, "archive": archive,
            "target": target, "schedule_hash": schedule_hash}


def _metric_summary(model, adapter, X: torch.Tensor, y: torch.Tensor,
                    p_target: torch.Tensor | None, device: torch.device) -> tuple[dict, np.ndarray]:
    """Evaluate a cloned model so IFNet's constrained FC cannot mutate training state."""
    before = sha256_state_dict(model.state_dict())
    rng_before = capture_rng(device)
    clone = copy.deepcopy(model)
    clone.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            _, logits = adapter.forward(clone, X[start:start + 64].to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu())
    prediction = torch.cat(probs, dim=0).numpy().astype(np.float32)
    restore_rng(rng_before, device)
    after = sha256_state_dict(model.state_dict())
    if before != after:
        raise RuntimeError("read-only diagnostic changed the training model")
    y_np = y.detach().cpu().numpy().astype(np.int64)
    row = classification_metrics(y_np, prediction)
    if p_target is not None:
        target = p_target.detach().cpu().numpy().astype(np.float64)
        student = prediction.astype(np.float64)
        row["kl_to_target"] = float(np.sum(target * (
            np.log(np.maximum(target, 1e-300)) - np.log(np.maximum(student, 1e-300))), axis=1).mean())
        row["prediction_agreement_with_target"] = float(
            np.mean(np.argmax(target, axis=1) == np.argmax(student, axis=1)))
    else:
        row["kl_to_target"] = float("nan")
        row["prediction_agreement_with_target"] = float("nan")
    return row, prediction


def _torch_load(path: Path, device):
    try:
        return torch.load(resolve_local_file(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(resolve_local_file(path), map_location=device)


def _resume_payload(model, optimizer, scheduler, device, *, fingerprint,
                    next_epoch, next_batch, history, elapsed):
    return {
        "fingerprint": fingerprint, "model_state": _state_dict_cpu(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": capture_rng(device), "next_epoch": int(next_epoch),
        "next_batch": int(next_batch), "history": history,
        "elapsed_seconds": float(elapsed),
    }


def _emit_progress(path: str | None, payload: dict) -> None:
    if path:
        atomic_json(Path(path), payload)


def _train_condition(cfg: dict, cfg_sha: str, run_id: str, fold: int,
                     condition: str, fold_data: dict, model_cfg: dict,
                     initial_state: dict, initial_hash: str,
                     device: torch.device, *, progress: str | None,
                     stop_file: Path | None, smoke_steps: int | None = None) -> dict:
    out = roots(cfg, run_id)
    fold_name = f"S{fold + 1:02d}"
    fold_weights = out["weights"] / "folds" / fold_name
    fold_results = out["results"] / "folds" / fold_name
    fold_weights.mkdir(parents=True, exist_ok=True)
    fold_results.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter("ifnet", device=device, **model_cfg)
    model = _new_model(adapter, model_cfg, initial_state, device)
    model.train()
    train_x = torch.as_tensor(fold_data["arrays"]["train_x"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(fold_data["arrays"]["train_y"], dtype=torch.long, device=device)
    feedback_x = torch.as_tensor(fold_data["arrays"]["feedback_x"], dtype=torch.float32, device=device)
    feedback_y = torch.as_tensor(fold_data["arrays"]["feedback_y"], dtype=torch.long, device=device)
    target_np = fold_data["target"][CONDITION_TARGET[condition]] if CONDITION_TARGET[condition] else None
    target_train = (torch.as_tensor(target_np, dtype=torch.float32, device=device)
                    if target_np is not None else None)
    feedback_target_np = (fold_data["archive"]["feedback_B0_probabilities"]
                          if condition == "B0_KD" else
                          fold_data["archive"]["feedback_p_e2_rebuilt"]
                          if condition == "E2_KD" else None)
    feedback_target = (torch.as_tensor(feedback_target_np, dtype=torch.float32, device=device)
                       if feedback_target_np is not None else None)
    opt_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(float(x) for x in opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]), amsgrad=bool(opt_cfg["amsgrad"]),
        foreach=bool(opt_cfg["foreach"]), fused=bool(opt_cfg["fused"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(opt_cfg["epochs"]), eta_min=float(opt_cfg["eta_min"]))
    final_path = fold_weights / f"{condition}_final.pt"
    condition_manifest = fold_results / f"{condition}_complete.json"
    target_sha = sha256_file(out["cache"] / "folds" / fold_name / "train_targets.npz")
    fingerprint = json_hash({
        "run_id": run_id, "config_sha256": cfg_sha, "fold": fold,
        "condition": condition, "initial_state_sha256": initial_hash,
        "input_sha256": fold_data["info"]["input_sha256"],
        "schedule_sha256": fold_data["schedule_hash"], "target_sha256": target_sha,
    })
    if condition_manifest.is_file() and final_path.is_file():
        old = json.loads(condition_manifest.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(f"completed {condition} identity mismatch for fold {fold}")
        payload = _torch_load(final_path, "cpu")
        if payload.get("state_sha256") != sha256_state_dict(payload["state_dict"]):
            raise RuntimeError(f"final checkpoint hash mismatch: {final_path}")
        print(f"[reuse] fold={fold_name} condition={condition} completed", flush=True)
        return old

    resume_path = fold_weights / f"{condition}_training_state.pt"
    history: list[dict] = []
    elapsed_before = 0.0
    start_epoch, start_batch = 0, 0
    if resume_path.is_file():
        state = _torch_load(resume_path, device)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(f"resume fingerprint mismatch: {resume_path}")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch, start_batch = int(state["next_epoch"]), int(state["next_batch"])
        history = list(state.get("history", []))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        restore_rng(state["rng_state"], device)
        print(f"[resume] {fold_name} {condition} epoch={start_epoch + 1} batch={start_batch}",
              flush=True)
    else:
        set_seed(int(cfg["student_seed"]), device)

    started = time.monotonic()
    epochs = int(opt_cfg["epochs"])
    save_every = int(opt_cfg["checkpoint_every_batches"])
    diagnostics = set(int(x) for x in opt_cfg["diagnostic_epochs"])
    total_steps = sum(len(_batches_for_epoch(fold_data["info"]["schedule"], e))
                      for e in range(epochs))
    seen_steps = start_epoch * len(_batches_for_epoch(fold_data["info"]["schedule"], 0)) + start_batch
    for epoch in range(start_epoch, epochs):
        model.train()
        batches = _batches_for_epoch(fold_data["info"]["schedule"], epoch)
        batch_begin = start_batch if epoch == start_epoch else 0
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "n": 0}
        for batch_index in range(batch_begin, len(batches)):
            ids_np = batches[batch_index]
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            xb, yb = train_x[ids], train_y[ids]
            _, logits = adapter.forward(model, xb)
            lambda_kd = (0.0 if condition == "NEW_CE"
                         else float(cfg["distillation"]["lambda_kd"]))
            teacher_batch = target_train[ids] if lambda_kd != 0.0 and target_train is not None else None
            loss, ce, kd = student_objective(
                logits, yb, teacher_probabilities=teacher_batch,
                ce_weight=float(cfg["distillation"]["ce_weight"]),
                lambda_kd=lambda_kd)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss fold={fold_name} condition={condition} epoch={epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"non-finite gradient fold={fold_name} condition={condition} epoch={epoch+1}")
            optimizer.step()
            n = int(len(ids_np))
            sums["loss"] += float(loss.detach()) * n
            sums["ce"] += float(ce.detach()) * n
            sums["kd"] += float(kd.detach()) * n
            sums["n"] += n
            seen_steps += 1
            last_batch = batch_index + 1 == len(batches)
            if last_batch:
                lr_used = float(optimizer.param_groups[0]["lr"])
                scheduler.step()
                record = {"epoch": epoch + 1, "condition": condition,
                          "loss": sums["loss"] / max(1, sums["n"]),
                          "ce": sums["ce"] / max(1, sums["n"]),
                          "kd": sums["kd"] / max(1, sums["n"]),
                          "lr": lr_used, "seconds": time.monotonic() - started + elapsed_before}
                if (epoch + 1) in diagnostics:
                    for part, X, y, target in (
                            ("train", train_x, train_y, target_train),
                            ("feedback", feedback_x, feedback_y, feedback_target)):
                        summary, _ = _metric_summary(model, adapter, X, y, target, device)
                        for name, value in summary.items():
                            record[f"{part}_{name}"] = value
                history.append(record)
                start_batch = 0
            next_epoch = epoch + 1 if last_batch else epoch
            next_batch = 0 if last_batch else batch_index + 1
            should_save = (seen_steps % save_every == 0 or last_batch
                           or (stop_file is not None and stop_file.exists()))
            if should_save:
                atomic_torch(resume_path, _resume_payload(
                    model, optimizer, scheduler, device, fingerprint=fingerprint,
                    next_epoch=next_epoch, next_batch=next_batch, history=history,
                    elapsed=elapsed_before + time.monotonic() - started))
            _emit_progress(progress, {
                "status": "running", "run_id": run_id, "fold": fold,
                "fold_subject": fold + 1, "condition": condition,
                "epoch": epoch + 1, "batch": batch_index + 1,
                "batches_in_epoch": len(batches), "steps": seen_steps,
                "total_steps": total_steps, "device": str(device),
            })
            if stop_file is not None and stop_file.exists():
                _emit_progress(progress, {"status": "paused", "run_id": run_id,
                                          "fold": fold, "condition": condition,
                                          "epoch": epoch + 1, "batch": batch_index + 1})
                return {"status": "paused", "condition": condition,
                        "fold": fold, "next_epoch": next_epoch, "next_batch": next_batch}
        start_batch = 0

    final_state = _state_dict_cpu(model)
    final_hash = sha256_state_dict(final_state)
    atomic_torch(final_path, {
        "state_dict": final_state, "state_sha256": final_hash,
        "fingerprint": fingerprint, "condition": condition, "fold": fold,
        "student_seed": int(cfg["student_seed"]), "initial_state_sha256": initial_hash,
        "config_sha256": cfg_sha, "train_uid_sha256": sha256_uids(fold_data["arrays"]["train_uids"]),
    })
    atomic_json(condition_manifest, {
        "status": "completed", "condition": condition, "fold": fold,
        "fingerprint": fingerprint, "state_sha256": final_hash,
        "initial_state_sha256": initial_hash, "target_sha256": target_sha,
        "history": history, "training_seconds": elapsed_before + time.monotonic() - started,
    })
    resume_path.unlink(missing_ok=True)
    write_csv(fold_results / f"{condition}_history.csv", history)
    print(f"[condition-complete] fold={fold_name} condition={condition} "
          f"epoch={epochs} seconds={elapsed_before + time.monotonic() - started:.1f}", flush=True)
    return json.loads(condition_manifest.read_text())


def run_smoke(cfg: dict, cfg_sha: str, run_id: str, fold: int = 0) -> dict:
    """Train-only wiring smoke: CE, B0-KD, E2-KD plus lambda-zero equivalence."""
    set_threads()
    out = roots(cfg, run_id)
    model_cfg, model_cfg_sha = _student_model_config(cfg)
    model_root = out["weights"]
    initial_state, initial_hash, _ = _create_initial_state(
        cfg, cfg_sha, model_cfg, model_cfg_sha, model_root)
    data = _load_training_fold(cfg, fold, out)
    device = torch.device("cpu")
    adapter = get_adapter("ifnet", device=device, **model_cfg)
    X = torch.as_tensor(data["arrays"]["train_x"], dtype=torch.float32)
    y = torch.as_tensor(data["arrays"]["train_y"], dtype=torch.long)
    schedule = data["info"]["schedule"]
    batches = _batches_for_epoch(schedule, 0)[:2]
    targets = {
        "B0_KD": torch.as_tensor(data["target"]["p_b0"], dtype=torch.float32),
        "E2_KD": torch.as_tensor(data["target"]["p_e2"], dtype=torch.float32),
    }

    def short_run(condition: str, lam: float, *, target_name: str | None):
        model = _new_model(adapter, model_cfg, initial_state, device)
        set_seed(int(cfg["student_seed"]), device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                                weight_decay=float(cfg["training"]["weight_decay"]),
                                betas=tuple(cfg["training"]["betas"]),
                                eps=float(cfg["training"]["eps"]),
                                amsgrad=False, foreach=False, fused=False)
        total = 0.0
        for ids_np in batches:
            ids = torch.as_tensor(ids_np, dtype=torch.long)
            _, logits = adapter.forward(model, X[ids])
            target_batch = targets[target_name][ids] if lam != 0.0 and target_name is not None else None
            loss, ce, kd = student_objective(
                logits, y[ids], teacher_probabilities=target_batch,
                ce_weight=1.0, lambda_kd=lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("smoke produced non-finite gradient")
            opt.step()
            total += float(loss.detach())
        return _state_dict_cpu(model), total

    ce_state, ce_loss = short_run("NEW_CE", 0.0, target_name=None)
    zero_state, zero_loss = short_run("B0_KD", 0.0, target_name="B0_KD")
    if sha256_state_dict(ce_state) != sha256_state_dict(zero_state) or ce_loss != zero_loss:
        raise RuntimeError("lambda_kd=0 failed to reproduce the CE smoke trajectory")
    branch_records = {}
    for condition, target_name in (("B0_KD", "B0_KD"), ("E2_KD", "E2_KD")):
        state, loss = short_run(condition, float(cfg["distillation"]["lambda_kd"]),
                                target_name=target_name)
        branch_records[condition] = {"state_sha256": sha256_state_dict(state), "loss": loss}
    smoke = {
        "status": "passed", "train_only": True, "fold": fold,
        "student_seed": int(cfg["student_seed"]), "initial_state_sha256": initial_hash,
        "n_batches_each": len(batches), "lambda_zero_matches_ce": True,
        "ce_state_sha256": sha256_state_dict(ce_state), "kd_conditions": branch_records,
        "target_partition": "D_train_only", "test_inputs_loaded": False,
    }
    smoke_root = out["results"] / "smoke"
    atomic_json(smoke_root / "train_only_smoke.json", smoke)
    atomic_json(out["report"] / "train_only_smoke.json", smoke)
    print(f"[smoke] PASS train-only fold=S{fold+1:02d} batches={len(batches)} "
          "lambda0_equals_CE; B0/E2 KD branches finite", flush=True)
    return smoke


def _fold_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def run_fold(cfg: dict, cfg_sha: str, run_id: str, fold: int,
             gpu: int | None, *, progress: str | None,
             stop_file: Path | None) -> int:
    if fold not in [int(value) for value in cfg["folds"]]:
        raise ValueError(f"fold {fold} is not configured")
    set_threads()
    device = torch.device("cpu") if gpu is None else torch.device(f"cuda:{gpu}")
    if gpu is not None:
        if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device {gpu} is unavailable")
        torch.cuda.set_device(gpu)
    out = roots(cfg, run_id)
    if not (out["results"] / "preflight.json").is_file():
        raise RuntimeError("preflight has not passed; run --preflight first")
    lock = _fold_lock(out["weights"] / "folds" / f"S{fold+1:02d}" / ".fold.lock")
    try:
        model_cfg, model_cfg_sha = _student_model_config(cfg)
        initial_state, initial_hash, initial_path = _create_initial_state(
            cfg, cfg_sha, model_cfg, model_cfg_sha, out["weights"])
        fold_data = _load_training_fold(cfg, fold, out)
        for condition in CONDITIONS:
            result = _train_condition(
                cfg, cfg_sha, run_id, fold, condition, fold_data, model_cfg,
                initial_state, initial_hash, device, progress=progress,
                stop_file=stop_file)
            if result.get("status") == "paused":
                return 75
        fold_weights = out["weights"] / "folds" / f"S{fold+1:02d}"
        fold_results = out["results"] / "folds" / f"S{fold+1:02d}"
        state = {
            "status": "completed", "fold": fold, "run_id": run_id,
            "config_sha256": cfg_sha, "student_seed": int(cfg["student_seed"]),
            "teacher_bundle_seed": int(cfg["teacher_bundle_seed"]),
            "initial_state_sha256": initial_hash, "initial_state_path": str(initial_path),
            "uid_schedule_sha256": fold_data["schedule_hash"],
            "train_uids_sha256": sha256_uids(fold_data["arrays"]["train_uids"]),
            "feedback_uids_sha256": sha256_uids(fold_data["arrays"]["feedback_uids"]),
            "conditions": {},
        }
        for condition in CONDITIONS:
            payload = _torch_load(fold_weights / f"{condition}_final.pt", "cpu")
            state["conditions"][condition] = {
                "state_sha256": payload["state_sha256"],
                "complete_record_sha256": sha256_file(fold_results / f"{condition}_complete.json"),
            }
        atomic_json(fold_results / "training_complete.json", state)
        atomic_torch(fold_weights / "completed_state.pt", {
            "fold": fold, "run_id": run_id, "terminal_rng": capture_rng(device),
            "state_sha256": json_hash(state),
        })
        _emit_progress(progress, {"status": "completed", "run_id": run_id,
                                 "fold": fold, "completed_folds": [fold],
                                 "completed_conditions": list(CONDITIONS)})
        print(f"[fold-complete] S{fold+1:02d} C/K/D trained; target test inputs not loaded",
              flush=True)
        return 0
    finally:
        lock.close()


def evaluate_all(cfg: dict, cfg_sha: str, run_id: str) -> dict:
    """Only called after all nine training folds are complete; loads target X here."""
    set_threads()
    out = roots(cfg, run_id)
    preflight_record = json.loads((out["results"] / "preflight.json").read_text())
    if preflight_record.get("status") != "passed":
        raise RuntimeError("preflight record is missing or failed")
    fold_complete = []
    for fold in cfg["folds"]:
        path = out["results"] / "folds" / f"S{int(fold)+1:02d}" / "training_complete.json"
        if not path.is_file():
            raise RuntimeError(f"target evaluation is blocked; training incomplete: {path}")
        record = json.loads(path.read_text())
        if record.get("status") != "completed" or record.get("config_sha256") != cfg_sha:
            raise RuntimeError(f"fold completion identity mismatch: {path}")
        fold_complete.append(record)

    model_cfg, _ = _student_model_config(cfg)
    adapter = get_adapter("ifnet", device="cpu", **model_cfg)
    input_meta = _global_inputs(cfg)
    paths = input_meta["paths"]
    X_all = np.load(resolve_local_file(paths["x"]), mmap_mode="r", allow_pickle=False)
    y_all = np.load(resolve_local_file(paths["y"]), mmap_mode="r", allow_pickle=False)
    lookup = _global_uid_lookup(paths["trials"])
    fold_rows: list[dict] = []
    comparison_vectors: dict[tuple[str, str], dict[int, float]] = {}
    predictions_output = out["results"] / "predictions_test.npz"
    pred_arrays: dict[str, np.ndarray] = {}
    historical_records = []
    for fold in cfg["folds"]:
        fold = int(fold)
        fold_name = f"S{fold+1:02d}"
        archive = _load_probability_archive(cfg, fold, training=False)
        uids = archive["test_uids"].astype(np.int64)
        labels = archive["test_y"].astype(np.int64)
        try:
            rows = np.asarray([lookup[tuple(map(int, uid))] for uid in uids], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"test UID cannot be found in global IFNet inputs: {exc}") from exc
        if not np.array_equal(y_all[rows], labels):
            raise RuntimeError(f"test labels disagree with saved historical prediction for {fold_name}")
        # This is the first stage that reads target EEG input values.
        X_raw = np.asarray(X_all[rows], dtype=np.float32)
        X_test = adapter.preprocess(X_raw)
        p_new = {}
        fold_weight_dir = out["weights"] / "folds" / fold_name
        for condition in CONDITIONS:
            payload = _torch_load(fold_weight_dir / f"{condition}_final.pt", "cpu")
            state_dict = payload["state_dict"]
            if payload.get("state_sha256") != sha256_state_dict(state_dict):
                raise RuntimeError(f"student checkpoint hash mismatch: {condition} {fold_name}")
            model = adapter.build(2)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            values = []
            with torch.no_grad():
                for start in range(0, len(X_test), 64):
                    _, logits = adapter.forward(model, X_test[start:start+64])
                    values.append(torch.softmax(logits, dim=1).cpu().numpy())
            p_new[condition] = np.concatenate(values, axis=0).astype(np.float32)

        p_b0 = archive["test_B0_probabilities"].astype(np.float32)
        p_ce666 = archive["test_CE666_probabilities"].astype(np.float32)
        p_ce667 = archive["test_CE667_probabilities"].astype(np.float32)
        p_b1 = archive["test_B1_probabilities"].astype(np.float32)
        p_b2 = archive["test_B2_probabilities"].astype(np.float32)
        p_e2 = mean_probabilities(p_b0, p_ce666, p_ce667)
        if np.max(np.abs(p_e2.astype(np.float64) - archive["test_E2_probabilities"].astype(np.float64))) > 1e-6:
            raise RuntimeError(f"target E2 probability reconstruction mismatch for {fold_name}")
        p_b12 = mean_probabilities(p_b1, p_b2)
        p_fc = mean_probabilities(p_new["NEW_CE"], p_b1, p_b2)
        p_fk = mean_probabilities(p_new["B0_KD"], p_b1, p_b2)
        p_fd = mean_probabilities(p_new["E2_KD"], p_b1, p_b2)
        condition_probs = {
            "B0": p_b0, "CE666": p_ce666, "CE667": p_ce667,
            "B1": p_b1, "B2": p_b2, "E2": p_e2,
            "B1_B2_SOFT": p_b12,
            "NEW_CE": p_new["NEW_CE"], "B0_KD": p_new["B0_KD"],
            "E2_KD": p_new["E2_KD"],
            "NEW_CE_B1_B2_SOFT": p_fc,
            "B0_KD_B1_B2_SOFT": p_fk,
            "E2_KD_B1_B2_SOFT": p_fd,
        }
        hard_overrides = {"E4": archive["test_E4_prediction"].astype(np.int64)}
        condition_probs["E4"] = np.eye(2, dtype=np.float32)[hard_overrides["E4"]]
        for condition, probabilities in condition_probs.items():
            metrics = classification_metrics(labels, probabilities,
                                             prediction=hard_overrides.get(condition))
            row = {"dataset": "BNCI2014004", "session": "session_3",
                   "target_subject_id": fold + 1, "fold": fold,
                   "student_seed": int(cfg["student_seed"]),
                   "teacher_bundle_seed": int(cfg["teacher_bundle_seed"]),
                   "condition": condition, **metrics}
            fold_rows.append(row)
            comparison_vectors.setdefault((condition, "accuracy"), {})[fold] = metrics["accuracy"]
            comparison_vectors.setdefault((condition, "balanced_accuracy"), {})[fold] = metrics["balanced_accuracy"]
        pred_arrays[f"S{fold+1:02d}_uids"] = uids
        pred_arrays[f"S{fold+1:02d}_y"] = labels
        for condition, probabilities in condition_probs.items():
            pred_arrays[f"S{fold+1:02d}_{condition}_probabilities"] = probabilities
        pred_arrays[f"S{fold+1:02d}_E4_prediction"] = hard_overrides["E4"]
        pred_arrays[f"S{fold+1:02d}_E2_KD_probabilities"] = p_new["E2_KD"]
        historical_records.append({"fold": fold, "target_subject_id": fold + 1,
                                   "test_uid_sha256": sha256_uids(uids),
                                   "test_n": int(len(labels)),
                                   "e2_member_reconstruction_max_abs_error": float(np.max(
                                       np.abs(p_e2.astype(np.float64) - archive["test_E2_probabilities"].astype(np.float64))))})

    atomic_npz(predictions_output, **pred_arrays)
    metric_index = {(row["target_subject_id"], row["condition"]): row for row in fold_rows}
    comparisons = [
        ("E2_KD_minus_NEW_CE", "E2_KD", "NEW_CE"),
        ("E2_KD_minus_B0_KD", "E2_KD", "B0_KD"),
        ("E2_KD_B1_B2_SOFT_minus_NEW_CE_B1_B2_SOFT", "E2_KD_B1_B2_SOFT", "NEW_CE_B1_B2_SOFT"),
        ("E2_KD_B1_B2_SOFT_minus_B0_KD_B1_B2_SOFT", "E2_KD_B1_B2_SOFT", "B0_KD_B1_B2_SOFT"),
        ("E2_KD_B1_B2_SOFT_minus_E2", "E2_KD_B1_B2_SOFT", "E2"),
        ("E2_KD_B1_B2_SOFT_minus_E4", "E2_KD_B1_B2_SOFT", "E4"),
        ("E2_KD_B1_B2_SOFT_minus_B1_B2_SOFT", "E2_KD_B1_B2_SOFT", "B1_B2_SOFT"),
        ("E2_KD_B1_B2_SOFT_minus_E2_KD", "E2_KD_B1_B2_SOFT", "E2_KD"),
    ]
    differences = []
    bootstrap_rng = np.random.default_rng(int(cfg["evaluation"]["bootstrap_seed"]))
    index_table = bootstrap_rng.integers(0, 9, size=(int(cfg["evaluation"]["bootstrap_draws"]), 9))
    for metric in ("accuracy", "balanced_accuracy"):
        for name, left, right in comparisons:
            diff = np.asarray([
                100.0 * (metric_index[(subject, left)][metric]
                         - metric_index[(subject, right)][metric])
                for subject in range(1, 10)], dtype=np.float64)
            mean, low, high, _ = paired_bootstrap(diff,
                draws=int(cfg["evaluation"]["bootstrap_draws"]),
                seed=int(cfg["evaluation"]["bootstrap_seed"]), index_table=index_table)
            differences.append({
                "comparison": name, "metric": metric, "n_seed": 1,
                "student_seed": int(cfg["student_seed"]), "n_subject": 9,
                "mean_difference_pp": mean, "ci95_low_pp": low, "ci95_high_pp": high,
                "win": int(np.sum(diff > 1e-6)), "tie": int(np.sum(np.abs(diff) <= 1e-6)),
                "loss": int(np.sum(diff < -1e-6)),
                "subject_differences_pp": json.dumps(diff.tolist()),
            })
    summary = []
    for condition in dict.fromkeys(row["condition"] for row in fold_rows):
        rows = [row for row in fold_rows if row["condition"] == condition]
        summary.append({
            "dataset": "BNCI2014004", "condition": condition,
            "n_student_seed": 1, "student_seed": int(cfg["student_seed"]),
            "teacher_bundle_seed": int(cfg["teacher_bundle_seed"]), "n_subject": len(rows),
            "accuracy_mean_percent": float(np.mean([row["accuracy"] for row in rows]) * 100),
            "balanced_accuracy_mean_percent": float(np.mean([row["balanced_accuracy"] for row in rows]) * 100),
            "accuracy_subject_sd_pp": float(np.std([row["accuracy"] for row in rows], ddof=1) * 100),
            "balanced_accuracy_subject_sd_pp": float(np.std([row["balanced_accuracy"] for row in rows], ddof=1) * 100),
        })
    external_metrics = out["results"] / "fold_metrics.csv"
    external_differences = out["results"] / "paired_differences.csv"
    external_summary = out["results"] / "summary.csv"
    write_csv(external_metrics, fold_rows)
    write_csv(external_differences, differences)
    write_csv(external_summary, summary)
    write_csv(out["report"] / "fold_metrics.csv", fold_rows, report=True)
    write_csv(out["report"] / "paired_differences.csv", differences, report=True)
    write_csv(out["report"] / "summary.csv", summary, report=True)
    report_record = {
        "status": "completed", "run_id": run_id, "protocol_id": cfg["protocol_id"],
        "config_sha256": cfg_sha, "student_seed": int(cfg["student_seed"]),
        "teacher_bundle_seed": int(cfg["teacher_bundle_seed"]), "n_subject": 9,
        "n_student_seed": 1, "folds": historical_records,
        "b1_b2_weights_restored": False,
        "evaluation_type": "offline probability composition on historical target trial UIDs",
        "target_input_used_after_all_training_completed": True,
        "target_ea_note": "Historical B0/E2/E4 use target-subject unlabeled EA; new IFNet inputs use no EA.",
        "results": {"fold_metrics": str(external_metrics),
                    "paired_differences": str(external_differences),
                    "summary": str(external_summary),
                    "test_predictions": str(predictions_output)},
    }
    atomic_json(out["results"] / "evaluation_complete.json", report_record)
    atomic_json(out["report"] / "summary.json", report_record)
    _write_markdown(out["report"] / "report.md", summary, differences, report_record)
    _write_xlsx(out["report"] / "summary.xlsx", summary, differences, report_record)
    # Keep D for future use. C and K are comparison-only checkpoints; all their
    # per-trial predictions and metrics have been written and hash-verified.
    for fold in cfg["folds"]:
        fold_dir = out["weights"] / "folds" / f"S{int(fold)+1:02d}"
        for condition in ("NEW_CE", "B0_KD"):
            checkpoint = fold_dir / f"{condition}_final.pt"
            payload = _torch_load(checkpoint, "cpu")
            if payload.get("state_sha256") != sha256_state_dict(payload["state_dict"]):
                raise RuntimeError(f"refusing to prune unverified checkpoint: {checkpoint}")
            checkpoint.unlink()
            (fold_dir / f"{condition}_training_state.pt").unlink(missing_ok=True)
        (fold_dir / "E2_KD_training_state.pt").unlink(missing_ok=True)
    print(f"[evaluate-complete] run={run_id} subjects=9 test rows={len(fold_rows)}; "
          f"reports={out['report']}", flush=True)
    return report_record


def _write_markdown(path: Path, summary: list[dict], differences: list[dict], metadata: dict) -> None:
    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E2 → IFNet distillation → B1/B2 soft-vote pilot",
        "",
        "Single student seed 668; fixed teacher bundle seed 666; 9 LOSO target subjects.",
        "Accuracy/BA are equal-weight means over target subjects; bootstrap resamples subjects.",
        "Historical B0/E2/E4 use target-subject unlabeled EA; new IFNet uses no EA.",
        "B1/B2 checkpoint weights were not restored; this is offline probability composition on saved trial UIDs.",
        "",
        "## Subject-macro means",
        "",
        "| Condition | Accuracy (%) | Balanced accuracy (%) |",
        "|---|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['accuracy_mean_percent']:.2f} | "
                     f"{row['balanced_accuracy_mean_percent']:.2f} |")
    lines.extend(["", "## Paired comparisons", "",
                  "| Comparison | Metric | Mean Δ (pp) | 95% bootstrap CI | W/T/L |",
                  "|---|---|---:|---:|---:|"])
    for row in differences:
        lines.append(f"| {row['comparison']} | {row['metric']} | "
                     f"{row['mean_difference_pp']:.2f} | "
                     f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] | "
                     f"{row['win']}/{row['tie']}/{row['loss']} |")
    lines.extend(["", f"Run ID: `{metadata['run_id']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_xlsx(path: Path, summary: list[dict], differences: list[dict], metadata: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    path = require_report_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    teacher_fill = PatternFill("solid", fgColor="FAC090")
    baseline_fill = PatternFill("solid", fgColor="93CDDD")
    red_fill = PatternFill("solid", fgColor="FFC7CE")
    green_fill = PatternFill("solid", fgColor="C6EFCE")
    yellow_fill = PatternFill("solid", fgColor="FFF2CC")
    metric_sources = (("Accuracy", "accuracy_mean_percent"),
                      ("Balanced Accuracy", "balanced_accuracy_mean_percent"))
    for sheet_name, field in metric_sources:
        ws = wb.active if wb.active.title == "Sheet" else wb.create_sheet()
        ws.title = sheet_name
        ws.sheet_view.showGridLines = False
        ws.append(["student", "method", "004"])
        for cell in ws[1]:
            cell.fill, cell.font = header_fill, header_font
            cell.alignment = Alignment(horizontal="center")
        order = ["B0", "CE666", "CE667", "E2", "E4", "B1", "B2", "B1_B2_SOFT",
                 "NEW_CE", "B0_KD", "E2_KD", "NEW_CE_B1_B2_SOFT",
                 "B0_KD_B1_B2_SOFT", "E2_KD_B1_B2_SOFT"]
        lookup = {row["condition"]: row[field] for row in summary}
        comparison_references = {
            "B0_KD": ("B0", "NEW_CE"),
            "E2_KD": ("E2", "NEW_CE"),
            "B0_KD_B1_B2_SOFT": ("B0", "NEW_CE_B1_B2_SOFT"),
            "E2_KD_B1_B2_SOFT": ("E2", "NEW_CE_B1_B2_SOFT"),
        }
        student_name = {"B0": "MIRepNet", "CE666": "IFNet", "CE667": "IFNet",
                        "E2": "Ensemble", "E4": "Ensemble", "B1": "IFNet",
                        "B2": "IFNet", "B1_B2_SOFT": "Ensemble",
                        "NEW_CE": "IFNet", "B0_KD": "IFNet", "E2_KD": "IFNet",
                        "NEW_CE_B1_B2_SOFT": "Ensemble",
                        "B0_KD_B1_B2_SOFT": "Ensemble",
                        "E2_KD_B1_B2_SOFT": "Ensemble"}
        for condition in order:
            if condition in lookup:
                ws.append([student_name[condition], condition, round(float(lookup[condition]), 2)])
        for row in ws.iter_rows(min_row=2, min_col=1, max_col=3):
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            row[2].number_format = "0.00"
            condition = row[1].value
            if condition in {"B0", "E2"}:
                row[2].fill = teacher_fill
            elif condition in {"NEW_CE", "NEW_CE_B1_B2_SOFT"}:
                row[2].fill = baseline_fill
            elif condition in comparison_references:
                teacher_name, baseline_name = comparison_references[condition]
                value = lookup.get(condition)
                teacher_value = lookup.get(teacher_name)
                baseline_value = lookup.get(baseline_name)
                if (value is not None and teacher_value is not None
                        and baseline_value is not None):
                    above_teacher = float(value) > float(teacher_value)
                    above_baseline = float(value) > float(baseline_value)
                    if above_teacher and above_baseline:
                        row[2].fill = red_fill
                    elif above_teacher:
                        row[2].fill = green_fill
                    elif above_baseline:
                        row[2].fill = yellow_fill
        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 34
        ws.column_dimensions["C"].width = 16
        note_row = ws.max_row + 2
        notes = [
            "Single student seed=668; teacher bundle seed=666; 9 LOSO target subjects.",
            "Values are subject-macro means. Accuracy and Balanced Accuracy are computed separately.",
            "Historical B0/E2/E4 use target-subject unlabeled EA; new IFNet inputs use no EA.",
            "B1/B2 model checkpoints are absent; combined scores are offline probabilities on saved historical trial UIDs.",
            "The final B1/B2/D vote is equal soft probability averaging; historical AdaBoost alpha0 is not reused.",
            "Comparison colors: method cells are compared with the paired reference rows described in the report; equality is not counted as greater.",
        ]
        for i, note in enumerate(notes, start=note_row):
            ws.cell(i, 1, note)
            ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=3)
            ws.cell(i, 1).alignment = Alignment(wrap_text=True, vertical="top")
        legend_row = note_row + len(notes) + 1
        legend = [
            ("Red", "Greater than both the matched teacher reference and the matched none baseline", "FFC7CE"),
            ("Green", "Greater than the matched teacher reference only", "C6EFCE"),
            ("Yellow", "Greater than the matched none baseline only", "FFF2CC"),
        ]
        for offset, (label, meaning, color) in enumerate(legend):
            current = legend_row + offset
            ws.cell(current, 1, label)
            ws.cell(current, 1).fill = PatternFill("solid", fgColor=color)
            ws.cell(current, 1).font = Font(bold=True)
            ws.cell(current, 1).alignment = Alignment(horizontal="center")
            ws.cell(current, 2, meaning)
            ws.merge_cells(start_row=current, start_column=2, end_row=current, end_column=3)
            ws.cell(current, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws = wb.create_sheet("Paired differences")
    ws.sheet_view.showGridLines = False
    columns = ["comparison", "metric", "mean Δ (pp)", "CI 2.5%", "CI 97.5%", "W", "T", "L", "n_seed", "n_subject"]
    ws.append(columns)
    for cell in ws[1]:
        cell.fill, cell.font = header_fill, header_font
        cell.alignment = Alignment(horizontal="center")
    for row in differences:
        ws.append([row["comparison"], row["metric"], row["mean_difference_pp"],
                   row["ci95_low_pp"], row["ci95_high_pp"], row["win"], row["tie"],
                   row["loss"], row["n_seed"], row["n_subject"]])
    for col, width in enumerate((48, 22, 16, 14, 14, 8, 8, 8, 10, 12), start=1):
        ws.column_dimensions[chr(64 + col)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    wb.save(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-config-sha256", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fold", type=int, default=None, help="zero-based LOSO target fold")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--progress", default=None)
    parser.add_argument("--stop-file", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg, cfg_sha = load_config(Path(args.config), args.expected_config_sha256)
        run_id = args.run_id or cfg["run_id"]
        if args.preflight:
            preflight(cfg, cfg_sha, run_id)
            return 0
        if args.smoke:
            if args.fold is not None and args.fold != 0:
                raise ValueError("train-only smoke is fixed to fold 0")
            run_smoke(cfg, cfg_sha, run_id, 0)
            return 0
        if args.evaluate:
            evaluate_all(cfg, cfg_sha, run_id)
            return 0
        if args.fold is None:
            raise ValueError("--fold is required for training")
        return run_fold(cfg, cfg_sha, run_id, int(args.fold), args.gpu,
                        progress=args.progress,
                        stop_file=Path(args.stop_file) if args.stop_file else None)
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
