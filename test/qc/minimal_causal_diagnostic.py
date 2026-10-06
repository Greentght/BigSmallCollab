#!/usr/bin/env python3
"""Minimal causal diagnostic for the BNCI2015001 S1 EEGNet collapse.

This is deliberately independent from the relation-gap experiment.  It runs
only four conditions on the existing seed-666 few-shot split:

    base_full, base_clean, kd_full, kd_clean

The training loop mirrors ``collab.distill.distill_student``.  In particular,
all four conditions use the same model construction, dry-run, AdamW optimizer,
CosineAnnealingLR scheduler, preprocessing, DataLoader construction, and
number of epochs.  The only loss change between base and KD is the vanilla
all-sample logit KD term.  The clean mask is applied before TensorDataset and
DataLoader construction.

The bad UID is supplied by the command line (``--exclude-uid SUBJECT TRIAL``),
so the project code does not globally hard-code a particular trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import data  # noqa: E402
from collab import artifacts  # noqa: E402
from collab.seed import set_seed  # noqa: E402
from models import get_adapter  # noqa: E402


DATASET = "BNCI2015001"
PROTOCOL = "fewshot"
SUBJECT = 0
SEED = 666
VAL_SPLIT = 0.7
STUDENT = "eegnet"
TEACHER = "mirepnet"
KD_METHOD = "KD_all"
KD_LAM = 0.5
KD_TEMPERATURE = 2.0
NUM_CLASSES = 2


def uid_key(uid: Sequence[int] | np.ndarray) -> tuple[int, int]:
    """Return the canonical, JSON-friendly UID key."""
    value = tuple(int(x) for x in np.asarray(uid).tolist())
    if len(value) != 2:
        raise ValueError(f"sample_uid must have two entries, got {value!r}")
    return value


def uid_list(uids: np.ndarray) -> list[list[int]]:
    return [[int(x[0]), int(x[1])] for x in np.asarray(uids)]


def _uid_set(uids: np.ndarray, context: str) -> set[tuple[int, int]]:
    arr = np.asarray(uids)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{context}: sample_uid must have shape (N, 2), got {arr.shape}")
    keys = [uid_key(x) for x in arr]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"{context}: sample_uid is not unique; duplicates={duplicates}")
    return set(keys)


def filter_by_uid(sample_uid: np.ndarray,
                  exclude_uids: Iterable[Sequence[int]]) -> np.ndarray:
    """Return a pre-DataLoader boolean mask for UID-based sample filtering."""
    _uid_set(sample_uid, "filter input")
    excluded = {uid_key(x) for x in exclude_uids}
    return np.asarray([uid_key(x) not in excluded for x in sample_uid], dtype=bool)


def align_by_uid(reference_uid: np.ndarray, candidate_uid: np.ndarray,
                 arrays: Mapping[str, np.ndarray], context: str = "UID alignment") -> dict[str, np.ndarray]:
    """Align candidate arrays to ``reference_uid`` and reject set/order errors."""
    reference = np.asarray(reference_uid, dtype=np.int64)
    candidate = np.asarray(candidate_uid, dtype=np.int64)
    reference_keys = list(map(uid_key, reference))
    candidate_keys = list(map(uid_key, candidate))
    reference_set = _uid_set(reference, f"{context} reference")
    candidate_set = _uid_set(candidate, f"{context} candidate")
    if reference_set != candidate_set:
        missing = sorted(reference_set - candidate_set)
        extra = sorted(candidate_set - reference_set)
        raise ValueError(f"{context}: UID set mismatch; missing={missing}, extra={extra}")
    positions = {key: index for index, key in enumerate(candidate_keys)}
    order = np.asarray([positions[key] for key in reference_keys], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if len(arr) != len(candidate):
            raise ValueError(
                f"{context}: {name} length {len(arr)} != UID length {len(candidate)}")
        out[name] = arr[order]
    return out


def assert_condition_uids(full_uid: np.ndarray, clean_uid: np.ndarray,
                          excluded_uids: Iterable[Sequence[int]]) -> None:
    """Assert that the clean mask removes exactly the requested UID set."""
    full = _uid_set(full_uid, "full condition")
    clean = _uid_set(clean_uid, "clean condition")
    excluded = {uid_key(x) for x in excluded_uids}
    if clean != full - excluded:
        raise AssertionError(
            f"clean UID set is not full-minus-excluded: missing={sorted((full - excluded) - clean)}, "
            f"unexpected={sorted(clean - (full - excluded))}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args, cwd=ROOT, check=False, capture_output=True, text=True)
        return result.stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False) + "\n")


def _append_log(path: Path, line: str) -> None:
    with path.open("a", buffering=1) as handle:
        handle.write(line.rstrip() + "\n")
    print(line, flush=True)


def _runtime_config(X_train: np.ndarray) -> dict:
    cfg = config.load_model_config(STUDENT, DATASET, PROTOCOL)
    cfg.update(
        in_channels=int(X_train.shape[1]),
        samples=int(X_train.shape[2]),
        dataset_name=DATASET,
    )
    return cfg


def _load_split_and_teacher(artifact_root: Path, excluded_uids: list[tuple[int, int]]):
    X_train, y_train, X_test, y_test, uid_train, uid_test = data.subject_split(
        DATASET, SUBJECT, val_split=VAL_SPLIT, seed=SEED, return_uid=True)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    uid_train = np.asarray(uid_train, dtype=np.int64)
    uid_test = np.asarray(uid_test, dtype=np.int64)

    if len(uid_train) != 60 or len(uid_test) != 140:
        raise AssertionError(
            f"expected 60/140 split, got {len(uid_train)}/{len(uid_test)}")
    _uid_set(uid_train, "raw train split")
    _uid_set(uid_test, "raw test split")
    if _uid_set(uid_train, "raw train split") & _uid_set(uid_test, "raw test split"):
        raise AssertionError("raw train/test UID sets overlap")
    if not set(excluded_uids).issubset(_uid_set(uid_train, "raw train split")):
        raise AssertionError(
            f"excluded UID(s) are not all in the full train split: {excluded_uids}")

    teacher_train_path = Path(artifacts.artifact_path(
        DATASET, TEACHER, SUBJECT, SEED, "train", root=str(artifact_root)))
    teacher_test_path = Path(artifacts.artifact_path(
        DATASET, TEACHER, SUBJECT, SEED, "test", root=str(artifact_root)))
    teacher_train = artifacts.load(
        DATASET, TEACHER, SUBJECT, SEED, "train", root=str(artifact_root))
    teacher_test = artifacts.load(
        DATASET, TEACHER, SUBJECT, SEED, "test", root=str(artifact_root))
    if teacher_train.get("split_policy") != "fewshot_stratified_random":
        raise AssertionError(
            f"unexpected teacher train split policy: {teacher_train.get('split_policy')!r}")
    if teacher_test.get("split_policy") != "fewshot_stratified_random":
        raise AssertionError(
            f"unexpected teacher test split policy: {teacher_test.get('split_policy')!r}")
    for name, artifact in (("teacher train", teacher_train), ("teacher test", teacher_test)):
        if "sample_uid" not in artifact:
            raise AssertionError(f"{name} artifact has no sample_uid")
        _uid_set(artifact["sample_uid"], name)

    aligned_train = align_by_uid(
        uid_train, teacher_train["sample_uid"],
        {"y": teacher_train["y"], "feats": teacher_train["feats"],
         "logits": teacher_train["logits"]},
        context="teacher train")
    aligned_test = align_by_uid(
        uid_test, teacher_test["sample_uid"], {"y": teacher_test["y"]},
        context="teacher test")
    if not np.array_equal(aligned_train["y"], y_train):
        raise AssertionError("teacher train labels do not match raw train labels")
    if not np.array_equal(aligned_test["y"], y_test):
        raise AssertionError("teacher test labels do not match raw test labels")
    if aligned_train["feats"].ndim != 2 or aligned_train["logits"].ndim != 2:
        raise AssertionError("teacher features/logits must be two-dimensional")
    if aligned_train["feats"].shape[0] != 60 or aligned_train["logits"].shape[0] != 60:
        raise AssertionError("teacher train arrays must contain 60 rows")
    aligned_train["sample_uid"] = uid_train.copy()

    full_mask = np.ones(len(uid_train), dtype=bool)
    clean_mask = filter_by_uid(uid_train, excluded_uids)
    assert_condition_uids(uid_train, uid_train[clean_mask], excluded_uids)
    if int(clean_mask.sum()) != 59:
        raise AssertionError(f"clean train count must be 59, got {clean_mask.sum()}")
    if not np.array_equal(uid_test, uid_test.copy()):
        raise AssertionError("test UID order changed unexpectedly")

    return {
        "X_train": X_train, "y_train": y_train, "uid_train": uid_train,
        "X_test": X_test, "y_test": y_test, "uid_test": uid_test,
        "teacher_train": aligned_train, "teacher_test": aligned_test,
        "teacher_train_path": teacher_train_path,
        "teacher_test_path": teacher_test_path,
        "teacher_train_sha256": _sha256(teacher_train_path),
        "teacher_test_sha256": _sha256(teacher_test_path),
        "full_mask": full_mask, "clean_mask": clean_mask,
    }


def _train_condition(condition: str, X_train: np.ndarray, y_train: np.ndarray,
                     uid_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                     teacher: Mapping[str, np.ndarray], runtime_cfg: Mapping,
                     output_dir: Path, device: str, epochs: int, kd: bool,
                     log_path: Path) -> tuple[dict, list[dict]]:
    """Train one condition using the existing offline-KD engine semantics."""
    # This call is intentionally immediately before model construction.  It
    # resets model, projection, dropout and DataLoader-shuffle initialization.
    set_seed(SEED)
    adapter = get_adapter(STUDENT, device=device, **dict(runtime_cfg))
    model = adapter.build(NUM_CLASSES)

    Xp = adapter.preprocess(X_train).to(adapter.device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    feat_teacher = torch.as_tensor(np.asarray(teacher["feats"]), dtype=torch.float32)
    logits_teacher = torch.as_tensor(np.asarray(teacher["logits"]), dtype=torch.float32)
    if len(feat_teacher) != len(y_tensor) or len(logits_teacher) != len(y_tensor):
        raise AssertionError(f"{condition}: teacher/student train counts differ")
    # KD_all is an all-sample term; the base condition uses the same ones vector
    # but sets lam_kd=0, retaining the exact original Base/KD data path.
    sample_weight = torch.ones(len(y_tensor), dtype=torch.float32)
    second_weight = torch.zeros(len(y_tensor), dtype=torch.float32)

    # Match collab.distill.distill_student: probe the feature dimension before
    # constructing the TensorDataset.  The probe is deliberately in train mode,
    # as in the existing implementation, so BatchNorm behavior is unchanged.
    with torch.no_grad():
        feature_probe, _ = adapter.forward(model, Xp[:2])
    projection = nn.Linear(feature_probe.shape[1], feat_teacher.shape[1]).to(adapter.device)

    dataset = TensorDataset(
        Xp.cpu(), y_tensor, feat_teacher, logits_teacher, sample_weight, second_weight)
    loader = DataLoader(
        dataset, batch_size=int(runtime_cfg["batch_size"]), shuffle=True)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=float(runtime_cfg["lr"]), weight_decay=float(runtime_cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    temperature = KD_TEMPERATURE
    lam_kd = KD_LAM if kd else 0.0
    _append_log(log_path, f"condition={condition}")
    _append_log(log_path, f"seed={SEED} train_count={len(y_train)} test_count={len(y_test)}")
    _append_log(log_path, f"kd_method={'KD_all' if kd else 'none'} lam_kd={lam_kd} temperature={temperature}")
    _append_log(log_path, f"epochs={epochs} lr={runtime_cfg['lr']} weight_decay={runtime_cfg['weight_decay']} batch_size={runtime_cfg['batch_size']} optimizer=AdamW scheduler=CosineAnnealingLR")
    _append_log(log_path, f"excluded_uid_applied_before_dataloader={condition.endswith('_clean')}")

    history: list[dict] = []
    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        correct = 0
        seen = 0
        for xb, yb, fb, lb, wb, _w2b in loader:
            xb = xb.to(adapter.device)
            yb = yb.to(adapter.device)
            fb = fb.to(adapter.device)
            lb = lb.to(adapter.device)
            wb = wb.to(adapter.device)
            _feat_student, logits = adapter.forward(model, xb)
            loss = F.cross_entropy(logits, yb)
            if lam_kd > 0:
                kd_per_sample = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(lb / temperature, dim=1),
                    reduction="none").sum(dim=1)
                denom = wb.sum()
                if denom <= 0:
                    raise AssertionError(f"{condition}: KD sample-weight denominator is zero")
                loss = loss + lam_kd * (temperature * temperature) * (
                    wb * kd_per_sample).sum() / denom
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = int(yb.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_n
            correct += int((logits.detach().argmax(1) == yb).sum().cpu())
            seen += batch_n
        scheduler.step()
        row = {
            "condition": condition,
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": 100.0 * correct / seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _append_log(
            log_path,
            f"epoch={epoch:03d} train_loss={row['train_loss']:.8f} train_accuracy={row['train_accuracy']:.4f} lr={row['learning_rate']:.10g}")

    model.eval()
    with torch.no_grad():
        _test_features, test_logits = adapter.infer(model, X_test)
    test_pred = np.asarray(test_logits).argmax(axis=1).astype(np.int64)
    accuracy = float((test_pred == y_test).mean() * 100.0)
    balanced_accuracy = float(balanced_accuracy_score(y_test, test_pred) * 100.0)
    kappa = float(cohen_kappa_score(y_test, test_pred))
    predicted_counts = np.bincount(test_pred, minlength=NUM_CLASSES)
    final_history = history[-1]

    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "condition": condition,
        "seed": SEED,
        "model": STUDENT,
        "model_state_dict": model.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "runtime_config": _to_builtin(dict(runtime_cfg)),
        "kd_method": KD_METHOD if kd else "none",
        "lam_kd": lam_kd,
        "temperature": temperature,
        "epochs": epochs,
        "final_train_loss": final_history["train_loss"],
        "final_train_accuracy": final_history["train_accuracy"],
    }, condition_dir / "model.pt")
    with (condition_dir / "train_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        condition_dir / "test_predictions.npz",
        sample_uid=np.asarray(teacher["test_uid"], dtype=np.int64),
        y=y_test,
        logits=np.asarray(test_logits, dtype=np.float32),
        pred=test_pred,
    )
    _append_log(log_path, f"test_accuracy={accuracy:.4f} balanced_accuracy={balanced_accuracy:.4f} kappa={kappa:.6f}")
    _append_log(log_path, f"predicted_class_0_count={int(predicted_counts[0])} predicted_class_1_count={int(predicted_counts[1])}")

    result = {
        "condition": condition,
        "seed": SEED,
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "removed_uid": "",
        "kd_method": KD_METHOD if kd else "none",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "kappa": kappa,
        "final_train_loss": float(final_history["train_loss"]),
        "final_train_accuracy": float(final_history["train_accuracy"]),
        "predicted_class_0_count": int(predicted_counts[0]),
        "predicted_class_1_count": int(predicted_counts[1]),
    }
    # The caller replaces this with the exact CLI manifest representation.  Do
    # not infer removed UIDs from row positions in the training loop.
    result["removed_uid"] = "" if condition.endswith("_full") else "manifest"
    return result, history


def _write_csv(path: Path, rows: list[Mapping], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition_result(result: dict, excluded_uids: list[tuple[int, int]]) -> dict:
    result = dict(result)
    result["removed_uid"] = "" if result["condition"].endswith("_full") else ";".join(
        f"({subject},{trial})" for subject, trial in excluded_uids)
    return result


def _write_report(output_dir: Path, results: list[dict], resolved: Mapping,
                  split_manifest: Mapping, excluded_uids: list[tuple[int, int]],
                  prior_result_path: Path) -> None:
    def fmt(value, digits=2):
        return f"{float(value):.{digits}f}"

    by_condition = {row["condition"]: row for row in results}
    table_rows = []
    for row in results:
        collapsed = (row["predicted_class_0_count"] == 0 or
                     row["predicted_class_1_count"] == 0)
        table_rows.append(
            f"| {row['condition']} | {row['train_count']} | {fmt(row['accuracy'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['kappa'], 4)} | "
            f"{'是' if collapsed else '否'} |")
    full = by_condition.get("base_full")
    clean = by_condition.get("base_clean")
    kd_full = by_condition.get("kd_full")
    kd_clean = by_condition.get("kd_clean")
    lines = [
        "# BNCI2015001 S1 EEGNet 坏 trial 最小因果诊断",
        "",
        "本实验只运行 `base_full`、`base_clean`、`kd_full`、`kd_clean` 四个条件，未运行 relation-gap、Top1/Top3、随机删除、多 seed 或其他学生模型。准确率和 balanced accuracy 均为百分比；kappa 为原始比例。",
        "",
        "## 结果",
        "",
        "| condition | train n | test accuracy | balanced accuracy | kappa | 是否坍塌 |",
        "| --------- | ------: | ------------: | ----------------: | ----: | :------: |",
        *table_rows,
        "",
        "坍塌定义：最终测试预测只包含一个类别（对应 predicted class count 为 0/140）。",
        "",
        "## 固定配置与数据检查",
        "",
        f"- 数据集/被试/协议/seed：`{DATASET}` / S1（内部 index 0）/ `{PROTOCOL}` / `{SEED}`。",
        f"- 原始划分：`val_split={VAL_SPLIT}`，训练 {split_manifest['full_train_count']} 条，测试 {split_manifest['test_count']} 条，policy=`{split_manifest['split_policy']}`；四个条件测试 UID 集合和顺序相同。",
        f"- 排除 UID：{', '.join(f'({a},{b})' for a,b in excluded_uids)}；只在 `base_clean`/`kd_clean` 构建 TensorDataset 前过滤，clean 训练集为 59 条，未补入测试样本。",
        f"- MIRepNet teacher artifact：`{resolved['teacher_artifact_train']}`；原文件 SHA256=`{resolved['teacher_artifact_train_sha256']}`。KD clean 使用输出目录内的 59 条过滤副本，没有覆盖原 artifact。",
        "",
        "## KD 方法和参数",
        "",
        "固定为现有统一 runner 中的标准 `KD_all`：对全部训练样本使用 logits KD，不使用 teacher-correct mask、MMD 或新的温度/权重。四个条件沿用 `collab.distill.distill_student` 的训练路径；base 条件把 `lam_kd` 设为 0，因此只剩 CE。",
        f"- `lam_kd={KD_LAM}`，`temperature={KD_TEMPERATURE}`，KD 权重模式 `all`。",
        f"- EEGNet 配置：epochs={resolved['training']['formal_epochs_from_config']}，lr={resolved['training']['lr']}，weight_decay={resolved['training']['weight_decay']}，batch_size={resolved['training']['batch_size']}；AdamW + CosineAnnealingLR(T_max=epochs)。",
        "- 模型选择：最后一个 epoch 的模型直接在固定 140 条测试集评测；没有用测试集选 epoch。",
        f"- 现有结果证据：`{prior_result_path}` 中 S1/seed666 的 KD_all/KD_masked/MMD/KD_MMD 均为 50%，本实验不把其他方法挑出来替代 KD_all。",
        "",
        "## 因果解释",
        "",
        f"- `base_full` vs `base_clean`：准确率变化为 {fmt(full['accuracy'])} -> {fmt(clean['accuracy'])}（差值 {fmt(clean['accuracy']-full['accuracy'])} 个百分点）。这直接检验保留/删除 UID 的影响。",
        f"- `kd_full` vs `base_full`：{fmt(kd_full['accuracy'])} vs {fmt(full['accuracy'])}；`kd_clean` vs `base_clean`：{fmt(kd_clean['accuracy'])} vs {fmt(clean['accuracy'])}。",
        "- 只有在正式结果呈现预先定义的方向时，才能说坏 trial 对坍塌有因果作用或 KD 有缓解作用；若没有复现，不调整配置制造预期结果。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "conda run -n mirepnet python test/qc/minimal_causal_diagnostic.py \\",
        "  --exclude-uid 0 1 --device cpu \\",
        "  --artifact-root results/artifacts \\",
        "  --output-dir test/qc/artifacts/qc_v1/test-15001-S1-seed666",
        "```",
        "",
        "smoke test 可加 `--epochs 2 --output-dir /tmp/test-15001-S1-seed666-smoke`；该结果不能代替正式 100 epoch 结果。",
    ]
    if full and not (45.0 <= full["accuracy"] <= 55.0):
        lines += [
            "",
            "## 原始坍塌未复现时的配置说明",
            "",
            "当前工作树的 `configs/models/eegnet.yaml` 对 BNCI2015001/fewshot 解析为 `weight_decay=0.1`；初始 Git commit 中只有 flat legacy 配置，默认 `weight_decay=0.0001`。已有 CSV 没有保存其训练超参数，因此若本次 `base_full` 未呈现约 50% 的坍塌，这个差异以及已有结果缺少完整 provenance 都必须作为可能原因报告，不能据此修改正式实验参数。",
        ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-uid", nargs=2, type=int, required=True,
                        metavar=("SUBJECT", "TRIAL"),
                        help="UID to remove in clean conditions")
    parser.add_argument("--artifact-root", type=Path,
                        default=ROOT / "results" / "artifacts")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "test" / "qc" / "artifacts" / "qc_v1" / "test-15001-S1-seed666")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None,
                        help="only for smoke tests; omit for resolved formal epochs")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-existing", action="store_true",
                        help="allow writing into this diagnostic output directory")
    return parser.parse_args(argv)


def run(args) -> int:
    if DATASET != "BNCI2015001" or SUBJECT != 0 or SEED != 666:
        raise AssertionError("fixed diagnostic constants were changed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    excluded_uids = [uid_key(args.exclude_uid)]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(
            f"refusing to overwrite non-empty diagnostic output: {output_dir}; "
            "choose a new directory or pass --allow-existing for this directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model_cfg = _runtime_config(np.zeros((60, 13, 1000), dtype=np.float32))
    epochs = int(args.epochs if args.epochs is not None else resolved_model_cfg["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.epochs is not None:
        resolved_model_cfg["epochs_override"] = int(args.epochs)

    _append_log(output_dir / "run.log", "static split/artifact check starting")
    data_pack = _load_split_and_teacher(artifact_root, excluded_uids)
    clean_mask = data_pack["clean_mask"]
    full_uid = data_pack["uid_train"]
    clean_uid = full_uid[clean_mask]
    test_uid = data_pack["uid_test"]
    if len(clean_uid) != 59 or uid_key(excluded_uids[0]) in {uid_key(x) for x in clean_uid}:
        raise AssertionError("clean UID filtering assertion failed")

    split_manifest = {
        "dataset": DATASET,
        "subject": "S1",
        "subject_index": SUBJECT,
        "protocol": PROTOCOL,
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "split_policy": "fewshot_stratified_random",
        "session": os.environ.get("MI2015001_SESSION", "session_A"),
        "source": "data.subject_split",
        "raw_train_shape": list(data_pack["X_train"].shape),
        "raw_test_shape": list(data_pack["X_test"].shape),
        "full_train_count": int(len(full_uid)),
        "clean_train_count": int(len(clean_uid)),
        "test_count": int(len(test_uid)),
        "full_train_uids": uid_list(full_uid),
        "clean_train_uids": uid_list(clean_uid),
        "test_uids": uid_list(test_uid),
        "full_train_label_counts": np.bincount(data_pack["y_train"], minlength=NUM_CLASSES).tolist(),
        "clean_train_label_counts": np.bincount(data_pack["y_train"][clean_mask], minlength=NUM_CLASSES).tolist(),
        "test_label_counts": np.bincount(data_pack["y_test"], minlength=NUM_CLASSES).tolist(),
        "assertions": {
            "full_is_60_and_contains_excluded": len(full_uid) == 60 and uid_key(excluded_uids[0]) in {uid_key(x) for x in full_uid},
            "clean_is_59_and_excludes_uid": len(clean_uid) == 59 and uid_key(excluded_uids[0]) not in {uid_key(x) for x in clean_uid},
            "test_is_140": len(test_uid) == 140,
            "clean_test_uid_overlap": bool(_uid_set(clean_uid, "clean") & _uid_set(test_uid, "test")),
        },
    }
    if split_manifest["assertions"]["clean_test_uid_overlap"]:
        raise AssertionError("clean train contains a test UID")
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "removed_uid_manifest.json", {
        "excluded_uids": uid_list(np.asarray(excluded_uids, dtype=np.int64)),
        "source": "command line --exclude-uid",
        "applied_conditions": ["base_clean", "kd_clean"],
        "full_conditions": ["base_full", "kd_full"],
        "filter_stage": "before TensorDataset, Sampler, and DataLoader construction",
        "physical_source_data_deleted": False,
        "teacher_clean_copy": str(output_dir / "teacher_kd_clean_train.npz"),
    })

    teacher_full = data_pack["teacher_train"]
    teacher_clean = {key: value[clean_mask] for key, value in teacher_full.items()}
    filtered_teacher_path = output_dir / "teacher_kd_clean_train.npz"
    np.savez_compressed(
        filtered_teacher_path,
        feats=np.asarray(teacher_clean["feats"], dtype=np.float32),
        logits=np.asarray(teacher_clean["logits"], dtype=np.float32),
        y=np.asarray(teacher_clean["y"], dtype=np.int64),
        sample_uid=np.asarray(clean_uid, dtype=np.int64),
        split_policy=np.asarray("fewshot_stratified_random"),
    )
    filtered_teacher = np.load(filtered_teacher_path)
    if not np.array_equal(filtered_teacher["sample_uid"], clean_uid):
        raise AssertionError("filtered teacher UID order does not match clean student UID order")
    if not np.array_equal(filtered_teacher["y"], data_pack["y_train"][clean_mask]):
        raise AssertionError("filtered teacher labels do not match clean student labels")

    prior_result_path = ROOT / "results" / "BNCI2015001_fewshot_distill_mirepnet_to_eegnet.csv"
    config_payload = {
        "experiment": "minimal_bad_trial_causal_diagnostic",
        "dataset": DATASET,
        "subject": "S1",
        "subject_index": SUBJECT,
        "protocol": PROTOCOL,
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "student": STUDENT,
        "teacher": TEACHER,
        "kd_method": KD_METHOD,
        "kd_parameters": {
            "lam_kd": KD_LAM,
            "temperature": KD_TEMPERATURE,
            "weight_mode": "all",
            "lam_feat": 0.0,
            "lam_mmd": 0.0,
            "teacher_correct_only": False,
        },
        "training": {
            "epochs": epochs,
            "formal_epochs_from_config": int(resolved_model_cfg["epochs"]),
            "lr": resolved_model_cfg["lr"],
            "weight_decay": resolved_model_cfg["weight_decay"],
            "batch_size": resolved_model_cfg["batch_size"],
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": epochs,
            "preprocessing": "models.base._SmallAdapter.preprocess (identity float32 tensor)",
            "model_selection": "last epoch; fixed test set only for final evaluation",
            "device": args.device,
        },
        "model_config_path": str(ROOT / "configs" / "models" / "eegnet.yaml"),
        "dataset_config_path": str(ROOT / "configs" / "datasets" / f"{DATASET}.yaml"),
        "teacher_artifact_train": str(data_pack["teacher_train_path"]),
        "teacher_artifact_test": str(data_pack["teacher_test_path"]),
        "teacher_artifact_train_sha256": data_pack["teacher_train_sha256"],
        "teacher_artifact_test_sha256": data_pack["teacher_test_sha256"],
        "prior_result_path": str(prior_result_path),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "session": os.environ.get("MI2015001_SESSION", "session_A"),
        "excluded_uids": uid_list(np.asarray(excluded_uids, dtype=np.int64)),
    }
    _write_json(output_dir / "config_resolved.json", config_payload)
    # Keep the requested YAML filename while retaining JSON's exact provenance
    # in the same directory for values such as the dirty git status.
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(_to_builtin(config_payload), sort_keys=False, allow_unicode=True))
    _append_log(output_dir / "run.log", "static split/artifact check passed")
    if args.check_only:
        _append_log(output_dir / "run.log", "check-only requested; no model was trained")
        return 0

    runtime_cfg = dict(resolved_model_cfg)
    results = []
    all_history = []
    conditions = [
        ("base_full", data_pack["full_mask"], False),
        ("base_clean", data_pack["clean_mask"], False),
        ("kd_full", data_pack["full_mask"], True),
        ("kd_clean", data_pack["clean_mask"], True),
    ]
    for condition, mask, kd in conditions:
        condition_teacher = {
            key: np.asarray(value)[mask] for key, value in teacher_full.items()}
        condition_uids = full_uid[mask]
        condition_teacher["test_uid"] = data_pack["uid_test"].copy()
        if not np.array_equal(condition_uids, condition_teacher["sample_uid"]):
            raise AssertionError(f"{condition}: teacher/student UID mismatch before DataLoader")
        condition_log = output_dir / f"{condition}.log"
        if condition_log.exists():
            condition_log.unlink()
        row, history = _train_condition(
            condition,
            data_pack["X_train"][mask], data_pack["y_train"][mask], condition_uids,
            data_pack["X_test"], data_pack["y_test"], condition_teacher,
            runtime_cfg, output_dir, args.device, epochs, kd, condition_log)
        results.append(_condition_result(row, excluded_uids))
        all_history.extend(history)

    result_fields = [
        "condition", "seed", "train_count", "test_count", "removed_uid",
        "kd_method", "accuracy", "balanced_accuracy", "kappa",
        "final_train_loss", "final_train_accuracy",
        "predicted_class_0_count", "predicted_class_1_count",
    ]
    _write_csv(output_dir / "results.csv", results, result_fields)
    _write_csv(output_dir / "train_history.csv", all_history, list(all_history[0]))
    _write_report(output_dir, results, config_payload, split_manifest,
                  excluded_uids, prior_result_path)
    _append_log(output_dir / "run.log", f"formal run complete; output={output_dir}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
