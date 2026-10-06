"""Independent seed-666 Teacher Prototype-Gated KD/MI pilot with 10-epoch CE warm-up."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
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
from experiments.distill import run_distill as legacy

_spec = importlib.util.spec_from_file_location(
    "_prototype_margin_reliability", ROOT / "test/qc/prototype_margin_reliability.py")
if _spec is None or _spec.loader is None:
    raise ImportError("prototype diagnostic helper cannot be loaded")
_proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proto)

_ALL_DATASET_SUBJECTS = {"BNCI2014001": 9, "BNCI2014001-4": 9,
                         "BNCI2014004": 9, "BNCI2015001": 12, "AlexMI": 8}
_DEFAULT_DATASETS = ["BNCI2014001", "BNCI2014004", "BNCI2015001", "AlexMI"]
DATASETS = list(_DEFAULT_DATASETS)
DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
CONDITIONS = ("KD_PROTO", "MI_PROTO", "KD_MI_PROTO")
OLD_CONTROLS = ("KD_all", "CE_MI", "KD_MI_ALL")
COMPARISONS = (("KD_PROTO", "KD_all"), ("MI_PROTO", "CE_MI"),
               ("KD_MI_PROTO", "KD_PROTO"), ("KD_MI_PROTO", "MI_PROTO"),
               ("KD_MI_PROTO", "KD_MI_ALL"))
SEED = 666
VAL_SPLIT = 0.7
EPOCHS = 100
WARMUP_EPOCHS = 10
PROTO_EPS = 1e-12
INIT_ARGS = SimpleNamespace(epochs=None, lr=None, weight_decay=None, batch_size=None)
DEFAULT_CONFIG = ROOT / "configs/experiments/distill_kd_mi_proto_mask_warmup10.yaml"
DEFAULT_OUTPUT = ROOT / "results/distill/kd_mi_proto_mask_warmup10"
MAIN_CSV = ROOT / "results/distill/kd_mi_proto_mask_warmup10.csv"
OLD_MI_CSV = ROOT / "results/distill/distill_mi.csv"
OLD_MASK_ROOT = ROOT / "results/distill/kd_mi_teacher_correct_mask_pilot"


def _json(x):
    def conv(v):
        if isinstance(v, Path): return str(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v) if np.isfinite(v) else None
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, dict): return {str(k): conv(y) for k, y in v.items()}
        if isinstance(v, (tuple, list)): return [conv(y) for y in v]
        return v
    return json.dumps(conv(x), sort_keys=True, separators=(",", ":"))


def _atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as f:
        tmp = Path(f.name); f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_text(path, text): _atomic_bytes(path, text.encode())
def _atomic_json(path, value): _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
def _atomic_yaml(path, value): _atomic_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _atomic_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    out = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows: writer.writerow({k: row.get(k, "") for k in fields})
    _atomic_text(path, out.getvalue())


def _read_csv(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))


def _finite(x):
    try: return bool(np.isfinite(float(x)))
    except (TypeError, ValueError): return False


def _sha_array(x): return legacy._sha256_array(np.asarray(x))
def _sha_file(x): return legacy._sha256_file(x)
def _sha_state(x): return legacy._sha256_state_dict(x)
def _combine(x): return legacy._combined_hash(x)
def _uid_split_hash(a, b): return legacy._sha256_uid_split(a, b)
def _session(dataset): return legacy._session_default(dataset)


def _alignment_hash(uid, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(uid, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(y, dtype="<i8").tobytes())
    return h.hexdigest()


def _git_snapshot():
    def run(*args):
        try: return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception as e: return f"<git unavailable: {e}>"
    return {"commit": run("rev-parse", "HEAD"), "status_short": run("status", "--short")}


def _gpu_snapshot(gpu):
    out = {"physical_gpu_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "requested_logical_gpu": gpu, "device": None,
           "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
           "cuda_available": bool(torch.cuda.is_available()), "name": None}
    if torch.cuda.is_available():
        out["device"] = f"cuda:{gpu}"
        try: out["name"] = torch.cuda.get_device_name(gpu)
        except Exception: pass
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        out["nvidia_smi"] = text.strip().splitlines()
    except Exception as e: out["nvidia_smi_error"] = str(e)
    return out


def _validate_config(cfg):
    global DATASETS, DATASET_SUBJECTS
    configured = list(cfg.get("datasets", []))
    if configured not in (_DEFAULT_DATASETS, ["BNCI2014001-4"]):
        raise ValueError("dataset scope must be the original four datasets or the isolated BNCI2014001-4 supplement")
    DATASETS = configured
    DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
    expected = {"datasets": DATASETS, "protocol": "fewshot", "val_split": .7,
                "seed": 666, "teacher": "mirepnet", "student": "ifnet",
                "epochs": 100, "warmup_epochs": 10, "optimizer": "adamw", "lr": .001,
                "weight_decay": .01, "batch_size": 16,
                "scheduler": "CosineAnnealingLR", "temperature_kd": 2.,
                "lam_kd": .5, "temperature_mi": 1., "lam_mi": .1,
                "mi_eps": 1e-8, "prototype_epsilon": 1e-12,
                "drop_last": False, "model_selection": "final_epoch",
                "conditions": list(CONDITIONS)}
    for key, value in expected.items():
        actual = cfg.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > 1e-12:
                raise ValueError(f"config {key} must be {value}, got {actual}")
        elif actual != value: raise ValueError(f"config {key} must be {value}, got {actual}")
    if any("AGREE" in str(x).upper() for x in cfg.get("conditions", [])):
        raise ValueError("agreement-only conditions are prohibited")
    if not DATASET_SUBJECTS or any(name not in _ALL_DATASET_SUBJECTS for name in DATASETS): raise AssertionError("invalid dataset scope")


def _unit_count(): return sum(DATASET_SUBJECTS[name] for name in DATASETS)
def _run_count(): return _unit_count() * len(CONDITIONS)


def _student_cfg(dataset):
    cfg = config.load_model_config("ifnet", dataset, "fewshot")
    for key, value in {"epochs": 100, "lr": .001, "weight_decay": .01, "batch_size": 16}.items():
        if abs(float(cfg.get(key)) - value) > 1e-12:
            raise ValueError(f"IFNet {dataset} {key}={cfg.get(key)!r}; expected {value}")
    return cfg


def _artifact_path(cfg, dataset, subject_index):
    root = Path(cfg.get("artifact_root", "results/artifacts"))
    if not root.is_absolute(): root = ROOT / root
    path = root / dataset / "mirepnet" / f"{subject_index}_{SEED}_train.npz"
    _proto.validate_train_path(path)
    return path


def _prototype_support(pm, teacher_pred, labels):
    classes = np.asarray(pm["classes"], dtype=np.int64)
    z = np.asarray(pm["z"], dtype=np.float64)
    preds = np.empty(len(z), dtype=np.int64); support = np.empty(len(z), dtype=np.float64)
    for i, y in enumerate(np.asarray(labels, dtype=np.int64)):
        sims = np.asarray([np.dot(z[i], pm["loo_proto_by_row"][i]) if int(c) == int(y)
                           else np.dot(z[i], pm["full_proto"][int(c)]) for c in classes])
        preds[i] = int(classes[np.argmax(sims)])
        pos = int(np.where(classes == int(teacher_pred[i]))[0][0])
        support[i] = float(sims[pos] - np.max(np.delete(sims, pos)))
    return preds, support


def _prototype_hash(pm, support, proto_pred, reliable):
    h = hashlib.sha256()
    full = np.asarray([pm["full_proto"][int(c)] for c in pm["classes"]], dtype=np.float64)
    for name, value in (("z", pm["z"]), ("full", full), ("support", support),
                        ("pred", proto_pred), ("reliable", reliable)):
        h.update(name.encode()); h.update(_sha_array(value).encode())
    return h.hexdigest()


def _class_counts(y, nc):
    c = Counter(np.asarray(y, dtype=np.int64).tolist())
    return {str(i): int(c.get(i, 0)) for i in range(nc)}


def _preflight(cfg):
    units, inventory = [], []
    for dataset in DATASETS:
        nc = int(config.load_dataset_config(dataset)["num_classes"])
        base_cfg = _student_cfg(dataset)
        for subject_index in range(DATASET_SUBJECTS[dataset]):
            Xtr, ytr, Xte, yte, uid_tr, uid_te = data.subject_split(
                dataset, subject_index, val_split=VAL_SPLIT, seed=SEED, return_uid=True)
            uid_tr = np.asarray(uid_tr, dtype=np.int64); uid_te = np.asarray(uid_te, dtype=np.int64)
            if uid_tr.ndim != 2 or uid_tr.shape[1] != 2 or len(set(map(tuple, uid_tr.tolist()))) != len(uid_tr):
                raise ValueError(f"{dataset} S{subject_index + 1}: invalid train UID")
            path = _artifact_path(cfg, dataset, subject_index)
            payload = _proto.load_train_artifact(path, expected_n=len(ytr))
            aligned = _proto.align_by_uid({"sample_uid": uid_tr, "y": np.asarray(ytr, dtype=np.int64)}, payload)
            y = np.asarray(ytr, dtype=np.int64); logits = np.asarray(aligned["logits"], dtype=np.float32)
            feats = np.asarray(aligned["feats"], dtype=np.float64)
            pm = _proto.prototype_metrics(feats, y, uid_tr, epsilon=PROTO_EPS)
            tpred = logits.argmax(1).astype(np.int64)
            ppred, support = _prototype_support(pm, tpred, y)
            reliable = (ppred == tpred) & (support > PROTO_EPS)
            base_cfg = dict(base_cfg)
            base_cfg.update(in_channels=int(Xtr.shape[1]), samples=int(Xtr.shape[2]), dataset_name=dataset)
            init = legacy._capture_initial_state(INIT_ARGS, dataset, "ifnet", Xtr, nc, "cpu", SEED)
            adapter = get_adapter("ifnet", device="cpu", **base_cfg)
            xtr_pre = adapter.preprocess(Xtr); xte_pre = adapter.preprocess(Xte)
            inv = {"dataset": dataset, "subject": subject_index + 1, "subject_index": subject_index,
                   "session": _session(dataset), "seed": SEED, "train_count": len(y),
                   "class_counts": _json(_class_counts(y, nc)),
                   "teacher_predicted_class_counts": _json(_class_counts(tpred, nc)),
                   "teacher_proto_predicted_class_counts": _json(_class_counts(ppred, nc)),
                   "teacher_proto_reliable_count": int(reliable.sum()),
                   "teacher_proto_unreliable_count": int((~reliable).sum()),
                   "teacher_proto_reliable_rate": float(reliable.mean()),
                   "teacher_classifier_proto_agreement_rate": float((ppred == tpred).mean()),
                   "teacher_support_margin_mean": float(support.mean()),
                   "teacher_support_margin_std": float(support.std(ddof=1)) if len(support) > 1 else np.nan,
                   "teacher_support_margin_min": float(support.min()), "teacher_support_margin_max": float(support.max()),
                   "unreliable_uid": _json(uid_tr[~reliable].astype(int).tolist()),
                   "unreliable_label": _json(y[~reliable].astype(int).tolist()),
                   "unreliable_teacher_pred": _json(tpred[~reliable].astype(int).tolist()),
                   "unreliable_proto_pred": _json(ppred[~reliable].astype(int).tolist()),
                   "train_uid_hash": _sha_array(uid_tr), "test_uid_hash": _sha_array(uid_te),
                   "split_uid_hash": _uid_split_hash(uid_tr, uid_te),
                   "teacher_uid_alignment_hash": _alignment_hash(uid_tr, y),
                   "teacher_artifact_path": str(path.resolve()), "teacher_artifact_sha256": _sha_file(path),
                   "teacher_logits_hash": _sha_array(logits), "teacher_feature_hash": _sha_array(feats),
                   "prototype_state_hash": _prototype_hash(pm, support, ppred, reliable),
                   "alignment_status": "pass", "status": "pass", "failure_reason": ""}
            inventory.append(inv)
            units.append({"dataset": dataset, "subject": subject_index + 1, "subject_index": subject_index,
                          "seed": SEED, "num_classes": nc, "session": _session(dataset), "Xtr": Xtr, "ytr": y,
                          "Xte": Xte, "yte": np.asarray(yte, dtype=np.int64), "uid_tr": uid_tr, "uid_te": uid_te,
                          "teacher_logits": logits, "teacher_feats": feats, "teacher_pred": tpred,
                          "teacher_proto_pred": ppred, "teacher_support": support, "teacher_proto_reliable": reliable,
                          "initial_state": init, "initial_state_hash": _sha_state(init), "student_cfg": base_cfg,
                          "xtr_pre": xtr_pre, "xte_pre": xte_pre, "artifact_path": path,
                          "artifact_sha256": inv["teacher_artifact_sha256"], "teacher_uid_alignment_hash": inv["teacher_uid_alignment_hash"],
                          "prototype_state_hash": inv["prototype_state_hash"],
                          "preprocessing_hash": _combine([_sha_array(xtr_pre.numpy()), _sha_array(xte_pre.numpy())]),
                          "train_uid_hash": inv["train_uid_hash"], "test_uid_hash": inv["test_uid_hash"],
                          "split_uid_hash": inv["split_uid_hash"], "inventory": inv})
    if len(units) != _unit_count(): raise RuntimeError(f"expected {_unit_count()} units, got {len(units)}")
    return units, inventory


def masked_components(student_logits, labels, teacher_logits, keep, condition,
                      lam_kd=.5, temperature_kd=2., lam_mi=.1, mi_eps=1e-8):
    """Full CE plus KD/MI on a post-forward dynamic prototype gate."""
    if condition not in CONDITIONS: raise ValueError(condition)
    keep = keep.to(student_logits.device).bool(); labels = labels.long()
    ce = F.cross_entropy(student_logits, labels); zero = student_logits.sum() * 0.
    n = int(keep.sum().item()); kd = mi = None
    if condition in ("KD_PROTO", "KD_MI_PROTO"):
        if n:
            t = teacher_logits.detach()[keep]; s = student_logits[keep]
            tp = F.softmax(t / temperature_kd, dim=1).detach()
            per = F.kl_div(F.log_softmax(s / temperature_kd, dim=1), tp,
                           reduction="none").sum(1)
            kd = per.mean()
        else: kd = zero
    if condition in ("MI_PROTO", "KD_MI_PROTO"):
        if n >= 2:
            t = teacher_logits.detach()[keep]; s = student_logits[keep]
            mi = probability_mi_loss(F.softmax(t, 1).detach(), F.softmax(s, 1), eps=mi_eps)
        else: mi = zero
    total = ce
    if kd is not None: total = total + lam_kd * temperature_kd ** 2 * kd
    if mi is not None: total = total + lam_mi * mi
    return {"total": total, "ce_loss": ce, "kd_loss": kd, "mi_loss": mi,
            "selected_count": n, "mi_valid": bool(mi is not None and n >= 2)}


def mask_from_predictions(teacher_pred, student_pred, reliable):
    teacher_pred = np.asarray(teacher_pred); student_pred = np.asarray(student_pred)
    reliable = np.asarray(reliable, dtype=bool)
    agreement = teacher_pred == student_pred
    rescue = (~agreement) & reliable
    return agreement, rescue, agreement | rescue


def _atomic_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    torch.save(payload, tmp); os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as f:
        tmp = Path(f.name)
    np.savez(tmp, **arrays); os.replace(tmp, path)


def _run_key(unit, condition):
    return f"{unit['dataset']}__S{unit['subject']}__seed{SEED}__{condition}"


def _inference(adapter, model, x, batch_size=256):
    model.eval(); pieces=[]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            _, logits = adapter.forward(model, x[start:start+batch_size].to(adapter.device))
            pieces.append(logits.detach().cpu())
    return torch.cat(pieces, dim=0).numpy()


def _epoch_metric(unit, condition, epoch, agreement, rescue, keep, kd, mi,
                  all_masked, mi_small, batch_hash):
    n=len(unit['ytr']); dis=n-int(agreement.sum()); rn=int(rescue.sum())
    return {'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],
            'seed':SEED,'condition':condition,'epoch':epoch,'train_count':n,
            'prediction_agreement_count':int(agreement.sum()),'prediction_agreement_rate':float(agreement.mean()),
            'prediction_disagreement_count':dis,'prediction_disagreement_rate':float(dis/n),
            'prototype_rescue_count':rn,'prototype_rescue_rate':float(rn/dis) if dis else '',
            'prototype_rescue_rate_reason':'' if dis else 'no_disagreement','prototype_rescue_rate_all':float(rn/n),
            'mask_keep_count':int(keep.sum()),'mask_keep_rate':float(keep.mean()),
            'mask_drop_count':int((~keep).sum()),'mask_drop_rate':float((~keep).mean()),
            'effective_kd_sample_count':kd if kd is not None else '',
            'effective_kd_sample_count_reason':'' if kd is not None else 'condition_has_no_kd',
            'effective_mi_sample_count':mi if mi is not None else '',
            'effective_mi_sample_count_reason':'' if mi is not None else 'condition_has_no_mi',
            'distill_all_masked_batch_count':int(all_masked),
            'mi_skipped_small_batch_count':mi_small if mi_small is not None else '',
            'mi_skipped_small_batch_count_reason':'' if mi_small is not None else 'condition_has_no_mi',
            'batch_order_hash':batch_hash}


def _train_condition(unit, condition, output_root, cfg, device):
    """Train 100 epochs: CE-only warm-up first, then MASK_PROTO distillation."""
    started = time.time(); set_seed(SEED)
    adapter = get_adapter('ifnet', device=device, **dict(unit['student_cfg']))
    model = adapter.build(unit['num_classes'])
    state = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in unit['initial_state'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'initial state mismatch: {missing}, {unexpected}')
    y = torch.as_tensor(unit['ytr'], dtype=torch.long)
    teacher = torch.as_tensor(unit['teacher_logits'], dtype=torch.float32)
    teacher_pred = torch.as_tensor(unit['teacher_pred'], dtype=torch.long)
    reliable = torch.as_tensor(unit['teacher_proto_reliable'], dtype=torch.bool)
    index = torch.arange(len(y), dtype=torch.long)
    ds = TensorDataset(unit['xtr_pre'].cpu(), y, teacher, teacher_pred, reliable, index)
    generator = torch.Generator(); generator.set_state(torch.get_rng_state())
    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False, generator=generator)
    optimizer = optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    n = len(y); seen = np.zeros(n, dtype=np.int64)
    student_trace = np.empty((EPOCHS, n), dtype=np.int64)
    agreement_trace = np.zeros((EPOCHS, n), bool)
    rescue_trace = np.zeros((EPOCHS, n), bool)
    keep_trace = np.zeros((EPOCHS, n), bool)
    history = []; epoch_rows = []; order_hashes = []; totals = Counter()
    has_kd = condition in ('KD_PROTO', 'KD_MI_PROTO')
    has_mi = condition in ('MI_PROTO', 'KD_MI_PROTO')
    for ep in range(EPOCHS):
        active = ep >= WARMUP_EPOCHS
        model.train(); digest = hashlib.sha256(); seen_ep = np.zeros(n, dtype=np.int64)
        ag = np.zeros(n, bool); rs = np.zeros(n, bool); kp = np.zeros(n, bool)
        kd_count = 0 if (active and has_kd) else None
        mi_count = 0 if (active and has_mi) else None
        all_masked = 0; mi_small = 0 if (active and has_mi) else None
        ce_sum = total_sum = kd_sum = mi_sum = 0.; ce_n = steps = mi_batches = 0
        for xb, yb, tb, tpb, rb, ib in loader:
            idx = ib.numpy().astype(np.int64, copy=False); seen[idx] += 1
            digest.update(np.ascontiguousarray(unit['uid_tr'][idx], dtype=np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            # Always forward the complete batch before any prediction/mask operation.
            _, logits = adapter.forward(model, xb.to(device)); sp = logits.detach().argmax(1)
            agree = tpb.to(device).eq(sp); rescue = (~agree) & rb.to(device)
            keep = (agree | rescue) if active else torch.ones_like(agree, dtype=torch.bool)
            if active:
                comp = masked_components(
                    logits, yb.to(device), tb.to(device).detach(), keep, condition,
                    lam_kd=float(cfg['lam_kd']), temperature_kd=float(cfg['temperature_kd']),
                    lam_mi=float(cfg['lam_mi']), mi_eps=float(cfg['mi_eps']))
            else:
                ce = F.cross_entropy(logits, yb.to(device))
                comp = {'total': ce, 'ce_loss': ce, 'kd_loss': None, 'mi_loss': None,
                        'selected_count': None, 'mi_valid': False}
            comp['total'].backward(); optimizer.step()
            a = agree.detach().cpu().numpy(); r = rescue.detach().cpu().numpy(); k = keep.detach().cpu().numpy()
            ag[idx] = a; rs[idx] = r; kp[idx] = k; seen_ep[idx] += 1
            student_trace[ep, idx] = sp.detach().cpu().numpy()
            agreement_trace[ep, idx] = a; rescue_trace[ep, idx] = r; keep_trace[ep, idx] = k
            ce_val = float(comp['ce_loss'].detach()); obj = float(comp['total'].detach())
            ce_sum += ce_val * len(idx); ce_n += len(idx); total_sum += obj; steps += 1
            totals['ce_sum'] += ce_val * len(idx); totals['ce_n'] += len(idx)
            totals['total_sum'] += obj; totals['steps'] += 1
            selected = int(comp['selected_count']) if active else 0
            if active and selected == 0: all_masked += 1
            if active and has_kd and selected:
                kd_count += selected; value = float(comp['kd_loss'].detach())
                kd_sum += value * selected; totals['kd_sum'] += value * selected; totals['kd_n'] += selected
            if active and has_mi:
                if selected >= 2:
                    mi_count += selected; mi_batches += 1; value = float(comp['mi_loss'].detach())
                    mi_sum += value; totals['mi_sum'] += value; totals['mi_batches'] += 1; totals['mi_n'] += selected
                elif selected == 1:
                    mi_small += 1; totals['mi_small'] += 1
        if not np.all(seen_ep == 1):
            raise RuntimeError(f'{unit["dataset"]} S{unit["subject"]} epoch {ep + 1}: UID schedule invalid')
        scheduler.step(); bh = digest.hexdigest(); order_hashes.append(bh)
        metric = _epoch_metric(unit, condition, ep + 1, ag, rs, kp, kd_count, mi_count, all_masked, mi_small, bh)
        metric.update({'phase': 'distill' if active else 'warmup',
                       'distillation_active': bool(active), 'warmup_epochs': WARMUP_EPOCHS})
        epoch_rows.append(metric)
        history.append({'epoch': ep + 1, 'phase': 'distill' if active else 'warmup',
                        'distillation_active': bool(active), 'warmup_epochs': WARMUP_EPOCHS,
                        'ce_loss': ce_sum / ce_n, 'kd_loss': kd_sum / kd_count if kd_count else None,
                        'mi_loss': mi_sum / mi_batches if mi_batches else None,
                        'scaled_kd_contribution': 2. * kd_sum / kd_count if kd_count else None,
                        'scaled_mi_contribution': .1 * mi_sum / mi_batches if mi_batches else None,
                        'total_loss': total_sum / steps, 'batch_order_hash': bh,
                        'effective_ce_sample_count': ce_n, 'effective_kd_sample_count': kd_count,
                        'effective_mi_sample_count': mi_count,
                        'distill_all_masked_batch_count': all_masked,
                        'mi_skipped_small_batch_count': mi_small,
                        'prediction_agreement_rate': float(ag.mean()),
                        'prototype_rescue_rate': float(rs.sum() / (~ag).sum()) if (~ag).sum() else None,
                        'mask_keep_rate': float(kp.mean())})
    if not np.all(seen == EPOCHS):
        raise RuntimeError('not every train UID appeared once per epoch')
    train_logits = _inference(adapter, model, unit['xtr_pre']); test_logits = _inference(adapter, model, unit['xte_pre'])
    train_preds = train_logits.argmax(1).astype(np.int64); test_preds = test_logits.argmax(1).astype(np.int64)
    final_agree = unit['teacher_pred'] == train_preds
    final_rescue = (~final_agree) & unit['teacher_proto_reliable']; final_keep = final_agree | final_rescue
    run_key = _run_key(unit, condition); trace = output_root / 'mask_traces' / f'{run_key}.npz'
    _atomic_npz(trace, sample_uid=unit['uid_tr'], labels=unit['ytr'], teacher_pred=unit['teacher_pred'],
                teacher_proto_pred=unit['teacher_proto_pred'], teacher_proto_reliable=unit['teacher_proto_reliable'],
                teacher_support_margin=unit['teacher_support'], epoch=np.arange(1, EPOCHS + 1),
                student_pred=student_trace, agreement_mask=agreement_trace,
                prototype_rescue_mask=rescue_trace, keep_mask=keep_trace)
    trace_sha = _sha_file(trace)
    history_path = output_root / 'training_history' / f'{run_key}.json'
    prediction_path = output_root / 'predictions' / f'{run_key}.npz'
    checkpoint_path = output_root / 'checkpoints' / f'{run_key}.pt'
    _atomic_json(history_path, history)
    _atomic_npz(prediction_path, train_uid=unit['uid_tr'], test_uid=unit['uid_te'], train_y=unit['ytr'],
                test_y=unit['yte'], train_logits=train_logits.astype(np.float32), test_logits=test_logits.astype(np.float32),
                train_preds=train_preds, test_preds=test_preds, final_agreement=final_agree,
                final_prototype_rescue=final_rescue, final_mask_keep=final_keep)
    order_hash = _combine(order_hashes)
    _atomic_torch(checkpoint_path, {'complete': True, 'run_key': run_key, 'dataset': unit['dataset'],
        'subject': unit['subject'], 'subject_index': unit['subject_index'], 'seed': SEED,
        'condition': condition, 'epochs': EPOCHS, 'warmup_epochs': WARMUP_EPOCHS,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'train_uid_hash': unit['train_uid_hash'], 'test_uid_hash': unit['test_uid_hash'],
        'split_uid_hash': unit['split_uid_hash'], 'initial_state_hash': unit['initial_state_hash'],
        'preprocessing_hash': unit['preprocessing_hash'], 'batch_order_hash': order_hash,
        'batch_order_hashes': order_hashes, 'teacher_uid_alignment_hash': unit['teacher_uid_alignment_hash'],
        'prototype_state_hash': unit['prototype_state_hash'], 'mask_trace_sha256': trace_sha})
    ev = metrics.evaluate(unit['yte'], test_preds); ba = float(balanced_accuracy_score(unit['yte'], test_preds) * 100.)
    distill_rows = [r for r in epoch_rows if r.get('distillation_active')]
    mean_distill_keep = float(np.mean([r['mask_keep_rate'] for r in distill_rows]))
    mean_distill_agree = float(np.mean([r['prediction_agreement_rate'] for r in distill_rows]))
    rescue_values = [float(r['prototype_rescue_rate']) for r in distill_rows if r['prototype_rescue_rate'] != '']
    mean_distill_rescue = float(np.mean(rescue_values)) if rescue_values else ''
    row = {'dataset': unit['dataset'], 'subject': unit['subject'], 'subject_index': unit['subject_index'],
        'session': unit['session'], 'seed': SEED, 'protocol': 'fewshot', 'teacher': 'mirepnet', 'student': 'ifnet',
        'condition': condition, 'result_source': 'new', 'warmup_epochs': WARMUP_EPOCHS,
        'distillation_start_epoch': WARMUP_EPOCHS + 1, 'total_epochs': EPOCHS,
        'train_count': len(unit['ytr']), 'test_count': len(unit['yte']),
        'teacher_proto_reliable_count': int(unit['teacher_proto_reliable'].sum()),
        'teacher_proto_unreliable_count': int((~unit['teacher_proto_reliable']).sum()),
        'teacher_proto_reliable_rate': float(unit['teacher_proto_reliable'].mean()),
        'final_epoch_prediction_agreement_rate': float(final_agree.mean()),
        'final_epoch_prototype_rescue_rate': float(final_rescue.sum() / (~final_agree).sum()) if (~final_agree).sum() else '',
        'final_epoch_mask_keep_rate': float(final_keep.mean()),
        'mean_epoch_prediction_agreement_rate': float(np.mean([r['prediction_agreement_rate'] for r in epoch_rows])),
        'mean_epoch_prototype_rescue_rate': float(np.nanmean([float(r['prototype_rescue_rate']) if r['prototype_rescue_rate'] != '' else np.nan for r in epoch_rows])),
        'mean_epoch_mask_keep_rate': float(np.mean([r['mask_keep_rate'] for r in epoch_rows])),
        'mean_distill_epoch_prediction_agreement_rate': mean_distill_agree,
        'mean_distill_epoch_prototype_rescue_rate': mean_distill_rescue,
        'mean_distill_epoch_mask_keep_rate': mean_distill_keep,
        'final_train_eval_agreement_rate': float(final_agree.mean()),
        'final_train_eval_prototype_rescue_rate': float(final_rescue.sum() / (~final_agree).sum()) if (~final_agree).sum() else '',
        'final_train_eval_mask_keep_rate': float(final_keep.mean()),
        'final_train_eval_kept_uid': _json(unit['uid_tr'][final_keep].astype(int).tolist()),
        'final_train_eval_masked_uid': _json(unit['uid_tr'][~final_keep].astype(int).tolist()),
        'effective_kd_sample_exposures': int(totals['kd_n']) if has_kd else '',
        'effective_mi_sample_exposures': int(totals['mi_n']) if has_mi else '',
        'distill_all_masked_batch_count': int(sum(r['distill_all_masked_batch_count'] for r in epoch_rows)),
        'mi_skipped_small_batch_count': int(totals['mi_small']) if has_mi else '',
        'mean_ce_loss': totals['ce_sum'] / totals['ce_n'],
        'mean_kd_loss': totals['kd_sum'] / totals['kd_n'] if has_kd and totals['kd_n'] else '',
        'mean_mi_loss': totals['mi_sum'] / totals['mi_batches'] if has_mi and totals['mi_batches'] else '',
        'mean_scaled_kd_contribution': 2. * totals['kd_sum'] / totals['kd_n'] if has_kd and totals['kd_n'] else '',
        'mean_scaled_mi_contribution': .1 * totals['mi_sum'] / totals['mi_batches'] if has_mi and totals['mi_batches'] else '',
        'mean_total_loss': totals['total_sum'] / totals['steps'],
        'final_train_accuracy': float((train_preds == unit['ytr']).mean() * 100.), 'test_accuracy': float(ev['acc']),
        'test_balanced_accuracy': ba, 'test_kappa': float(ev['kappa']),
        'predicted_class_counts': _json(np.bincount(test_preds, minlength=unit['num_classes']).tolist()),
        'collapse_flag': bool(len(np.unique(test_preds)) < 2), 'runtime_seconds': time.time() - started,
        'train_uid_hash': unit['train_uid_hash'], 'test_uid_hash': unit['test_uid_hash'],
        'split_uid_hash': unit['split_uid_hash'], 'initial_state_hash': unit['initial_state_hash'],
        'preprocessing_hash': unit['preprocessing_hash'], 'batch_order_hash': order_hash,
        'batch_order_hashes': _json(order_hashes), 'teacher_uid_alignment_hash': unit['teacher_uid_alignment_hash'],
        'teacher_artifact_sha256': unit['artifact_sha256'], 'teacher_logits_hash': unit['inventory']['teacher_logits_hash'],
        'teacher_feature_hash': unit['inventory']['teacher_feature_hash'], 'prototype_state_hash': unit['prototype_state_hash'],
        'mask_trace_path': str(trace.resolve()), 'mask_trace_sha256': trace_sha, 'epochs_completed': EPOCHS,
        'status': 'complete', 'failure_reason': '', 'checkpoint_path': str(checkpoint_path.resolve()),
        'history_path': str(history_path.resolve()), 'prediction_path': str(prediction_path.resolve())}
    return row, epoch_rows

def _validate_control_row(row, unit, name):
    check={}
    for key,value in [('dataset',unit['dataset']),('subject_index',unit['subject_index']),('subject',unit['subject']),('session',unit['session']),('seed',SEED),('protocol','fewshot'),('teacher','mirepnet'),('student','ifnet')]:
        check[key]=(key == 'subject_index' and key not in row) or str(row.get(key,''))==str(value)
    check['train_count']=str(row.get('n_train',row.get('train_count','')))==str(len(unit['ytr']))
    check['test_count']=str(row.get('n_test',row.get('test_count','')))==str(len(unit['yte']))
    for key,value in [('train_uid_hash',unit['train_uid_hash']),('test_uid_hash',unit['test_uid_hash']),('split_uid_hash',unit['split_uid_hash']),('initial_state_hash',unit['initial_state_hash']),('teacher_artifact_sha256',unit['artifact_sha256'])]: check[key]=row.get(key)==value
    path=row.get('teacher_artifact_path',''); check['train_artifact_only']=path.endswith('_train.npz') and '_test.npz' not in path
    alignment=row.get('teacher_train_uid_alignment',row.get('teacher_uid_alignment_hash','')); check['alignment']=('status' not in alignment or 'pass' in alignment or alignment==unit['teacher_uid_alignment_hash'])
    check['complete']=row.get('failure_status',row.get('status'))=='complete'; check['finite']=all(_finite(row.get(k)) for k in ('test_accuracy','test_balanced_accuracy','test_kappa'))
    if name in ('KD_all','CE_MI'):
        check['epochs']=str(row.get('student_epochs','')) in ('100','100.0'); check['optimizer']=row.get('optimizer','').lower()=='adamw'; check['lr']=_finite(row.get('student_lr')) and abs(float(row['student_lr'])-.001)<1e-12; check['weight_decay']=_finite(row.get('student_weight_decay')) and abs(float(row['student_weight_decay'])-.01)<1e-12; check['batch_size']=str(row.get('student_batch_size','')) in ('16','16.0'); check['scheduler']=row.get('scheduler')=='CosineAnnealingLR'
        try: check['loss_params']=abs(float(row['lam_kd'])-(.5 if name=='KD_all' else 0.))<1e-12 and abs(float(row['lam_mi'])-(.1 if name=='CE_MI' else 0.))<1e-12 and abs(float(row['temperature'])-2.)<1e-12
        except Exception: check['loss_params']=False
    else:
        check['condition']=row.get('condition')=='KD_MI_ALL'; check['epochs']=str(row.get('epochs_completed','100')) in ('100','100.0')
        try:
            hist=json.loads(Path(row['history_path']).read_text()); check['history_unmasked']=len(hist)==100 and all(int(x.get('effective_ce_sample_count',-1))==len(unit['ytr']) and int(x.get('effective_kd_sample_count',-1))==len(unit['ytr']) and int(x.get('effective_mi_sample_count',-1))==len(unit['ytr']) and int(x.get('all_masked_batch_count',-1))==0 for x in hist)
        except Exception: check['history_unmasked']=False
    for key in ('checkpoint_path','history_path','prediction_path'): check[key]=bool(row.get(key)) and Path(row[key]).exists()
    if check.get('checkpoint_path'):
        try:
            p=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=True); check['checkpoint_complete']=bool(p.get('complete')) and int(p.get('epochs',-1))==100
        except Exception: check['checkpoint_complete']=False
    else: check['checkpoint_complete']=False
    if check.get('history_path'):
        try: check['history_100']=len(json.loads(Path(row['history_path']).read_text()))==100
        except Exception: check['history_100']=False
    else: check['history_100']=False
    if check.get('prediction_path'):
        try:
            with np.load(row['prediction_path'],allow_pickle=False) as p: check['prediction_complete']=p['test_preds'].shape[0]==len(unit['yte'])
        except Exception: check['prediction_complete']=False
    else: check['prediction_complete']=False
    return check,[k for k,v in check.items() if not v]


def _validate_controls(units, output_root):
    old_mi=_read_csv(OLD_MI_CSV); old_mask=_read_csv(OLD_MASK_ROOT/'results_per_run.csv')
    lookup={
        'KD_all':{(r.get('dataset'),int(r.get('key',-1))):r for r in old_mi if int(r.get('seed',-1))==SEED and r.get('method')=='KD_all'},
        'CE_MI':{(r.get('dataset'),int(r.get('key',-1))):r for r in old_mi if int(r.get('seed',-1))==SEED and r.get('method')=='CE_MI'},
        'KD_MI_ALL':{(r.get('dataset'),int(r.get('subject_index',-1))):r for r in old_mask if int(r.get('seed',-1))==SEED and r.get('condition')=='KD_MI_ALL'}}
    validation=[]; reused=[]
    for unit in units:
        for name in OLD_CONTROLS:
            row=lookup[name].get((unit['dataset'],unit['subject_index']))
            if row is None: check,bad={'row_present':False},['row_present']
            else: check,bad=_validate_control_row(row,unit,name); check['row_present']=True
            validation.append({'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],'seed':SEED,'control':name,'valid':not bad,'failed_checks':_json(bad),'checks':_json(check)})
            if row is not None and not bad:
                out=dict(row); out.update({'condition':name,'result_source':'reused','subject_index':unit['subject_index'],'train_count':len(unit['ytr']),'test_count':len(unit['yte']),'teacher_proto_reliable_count':int(unit['teacher_proto_reliable'].sum()),'teacher_proto_unreliable_count':int((~unit['teacher_proto_reliable']).sum()),'teacher_proto_reliable_rate':float(unit['teacher_proto_reliable'].mean()),'teacher_uid_alignment_hash':unit['teacher_uid_alignment_hash'],'prototype_state_hash':unit['prototype_state_hash']}); reused.append(out)
    _atomic_csv(output_root/'control_validation.csv',validation)
    return reused,validation


def _new_valid(row, unit):
    if row.get('status')!='complete' or str(row.get('epochs_completed'))!='100': return False
    for key,value in [('train_uid_hash',unit['train_uid_hash']),('test_uid_hash',unit['test_uid_hash']),('split_uid_hash',unit['split_uid_hash']),('initial_state_hash',unit['initial_state_hash']),('preprocessing_hash',unit['preprocessing_hash']),('teacher_artifact_sha256',unit['artifact_sha256']),('teacher_uid_alignment_hash',unit['teacher_uid_alignment_hash']),('prototype_state_hash',unit['prototype_state_hash'])]:
        if row.get(key)!=value:return False
    if not all(_finite(row.get(k)) for k in ('test_accuracy','test_balanced_accuracy','test_kappa')):return False
    try:
        hist=json.loads(Path(row['history_path']).read_text()); p=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=True)
        if len(hist)!=100 or not p.get('complete') or int(p.get('epochs',-1))!=100:return False
        with np.load(row['prediction_path'],allow_pickle=False) as a:
            if a['test_preds'].shape[0]!=len(unit['yte']):return False
        return Path(row['mask_trace_path']).exists() and _sha_file(Path(row['mask_trace_path']))==row.get('mask_trace_sha256')
    except Exception:return False


def _write_manifest(output_root,units,rows):
    lookup={(r.get('dataset'),int(r.get('subject_index',-1)),r.get('condition')):r for r in rows}; out=[]
    for u in units:
        for c in CONDITIONS:
            r=lookup.get((u['dataset'],u['subject_index'],c)); out.append({'run_key':_run_key(u,c),'dataset':u['dataset'],'subject':u['subject'],'subject_index':u['subject_index'],'seed':SEED,'condition':c,'status':r.get('status','pending') if r else 'pending','epochs_completed':r.get('epochs_completed',0) if r else 0,'failure_reason':r.get('failure_reason','') if r else ''})
    _atomic_csv(output_root/'run_manifest.csv',out)


def _bootstrap(diff,seed=666,count=10000):
    x=np.asarray(diff,dtype=float)
    if not len(x):return [np.nan,np.nan]
    rng=np.random.RandomState(seed); means=x[rng.randint(0,len(x),size=(count,len(x)))].mean(1)
    return [float(v) for v in np.percentile(means,[2.5,97.5])]


def _summaries(units,new_rows,reused,output_root):
    combined=[]
    for row in list(reused)+list(new_rows):
        r=dict(row); r['subject']=int(r.get('subject',0)); r['subject_index']=int(r.get('subject_index',r.get('key',-1))); r['condition']=r.get('condition',r.get('method')); r['test_accuracy']=float(r.get('test_accuracy',r.get('acc'))); r['test_balanced_accuracy']=float(r.get('test_balanced_accuracy')); r['test_kappa']=float(r.get('test_kappa',r.get('kappa'))); combined.append(r)
    _atomic_csv(output_root/'combined_results_per_run.csv',combined)
    subjects=[{'dataset':r['dataset'],'subject':r['subject'],'subject_index':r['subject_index'],'seed':SEED,'condition':r['condition'],'result_source':r.get('result_source','reused'),'test_accuracy':r['test_accuracy'],'test_balanced_accuracy':r['test_balanced_accuracy'],'test_kappa':r['test_kappa']} for r in combined]
    _atomic_csv(output_root/'results_per_subject.csv',subjects)
    datasets=[]
    for ds in DATASETS:
        for c in (*OLD_CONTROLS,*CONDITIONS):
            vals=[r for r in subjects if r['dataset']==ds and r['condition']==c]; ba=np.asarray([r['test_balanced_accuracy'] for r in vals],float); ac=np.asarray([r['test_accuracy'] for r in vals],float); ka=np.asarray([r['test_kappa'] for r in vals],float)
            datasets.append({'dataset':ds,'condition':c,'subject_count':len(vals),'balanced_accuracy_mean':float(ba.mean()) if len(ba) else np.nan,'balanced_accuracy_sd':float(ba.std(ddof=1)) if len(ba)>1 else np.nan,'accuracy_mean':float(ac.mean()) if len(ac) else np.nan,'accuracy_sd':float(ac.std(ddof=1)) if len(ac)>1 else np.nan,'kappa_mean':float(ka.mean()) if len(ka) else np.nan})
    _atomic_csv(output_root/'results_per_dataset.csv',datasets)
    by={(r['dataset'],r['subject_index'],r['condition']):r['test_balanced_accuracy'] for r in subjects}; comparisons=[]
    for ds in DATASETS:
        for method,base in COMPARISONS:
            ids=sorted({s for d,s,c in by if d==ds and c==method}&{s for d,s,c in by if d==ds and c==base})
            if not ids: comparisons.append({'dataset':ds,'comparison':f'{method}-{base}','status':'unavailable','n_subject':0}); continue
            diff=np.asarray([by[(ds,s,method)]-by[(ds,s,base)] for s in ids],float); ci=_bootstrap(diff)
            try:
                from scipy.stats import wilcoxon
                p=float(wilcoxon(diff).pvalue) if np.any(diff!=0) else 1.
            except Exception:p=np.nan
            comparisons.append({'dataset':ds,'comparison':f'{method}-{base}','status':'available','n_subject':len(diff),'mean_delta_balanced_accuracy':float(diff.mean()),'wins':int((diff>1e-12).sum()),'ties':int((np.abs(diff)<=1e-12).sum()),'losses':int((diff< -1e-12).sum()),'bootstrap_seed':666,'bootstrap_count':10000,'bootstrap_ci95_low':ci[0],'bootstrap_ci95_high':ci[1],'wilcoxon_p':p})
    _atomic_csv(output_root/'paired_comparisons.csv',comparisons)
    return combined,subjects,datasets,comparisons


def _report(output_root,units,inventory,reused,new_rows,datasets,comparisons,controls_ok):
    lines=['# Teacher Prototype-Gated KD/MI Pilot — 10-epoch CE warm-up','',f'- {len(units)} subject-wise few-shot units; seed 666 only; new runs {len(new_rows)}/{len(units)*len(CONDITIONS)}; reused controls {len(reused)}/{len(units)*len(OLD_CONTROLS)}; controls all valid: {controls_ok}.','- Fixed schedule: epochs 1–10 are CE-only warm-up; epochs 11–100 use the same `MASK_PROTO` rule. No student prototype is used.','- New conditions only: `KD_PROTO`, `MI_PROTO`, `KD_MI_PROTO`.','- Complete-batch IFNet forward precedes the dynamic gate. CE always uses all samples; only KD/MI use `keep` during epochs 11–100. Masked rows remain in forward/BatchNorm and are not deleted.','','## MASK_PROTO','', '`keep = agreement | ((~agreement) & teacher_proto_reliable)`, with `student_pred = argmax(student_logits.detach())`. Teacher reliability is `(teacher_proto_pred == teacher_pred) and (teacher_support_margin > 1e-12)`. Teacher features are normalized, class prototypes are normalized means, and the true-class prototype is leave-one-out; wrong classes use full prototypes. This is an in-sample, label-conditioned teacher view used only as a Boolean gate.','','## Prototype inventory','','| dataset | train | reliable | unreliable | rate |','|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        rr=[r for r in inventory if r['dataset']==ds]; n=sum(int(r['train_count']) for r in rr); rel=sum(int(r['teacher_proto_reliable_count']) for r in rr); lines.append(f'| {ds} | {n} | {rel} | {n-rel} | {rel/n:.6f} |')
    lines += ['','## Test Balanced Accuracy (mean ± SD)','', '| dataset | KD_all | CE_MI | KD_MI_ALL | KD_PROTO | MI_PROTO | KD_MI_PROTO |','|---|---:|---:|---:|---:|---:|---:|']
    look={(r['dataset'],r['condition']):r for r in datasets}
    for ds in DATASETS:
        cells=[]
        for c in (*OLD_CONTROLS,*CONDITIONS):
            r=look[(ds,c)]; cells.append('NA' if not _finite(r['balanced_accuracy_mean']) else f"{r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f}")
        lines.append(f'| {ds} | '+' | '.join(cells)+' |')
    lines += ['','## Paired comparisons','', '| dataset | comparison | mean Δ BA | W/T/L | 95% bootstrap CI |','|---|---|---:|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['dataset']} | {r['comparison']} | unavailable | — | — |" if r.get('status')!='available' else f"| {r['dataset']} | {r['comparison']} | {r['mean_delta_balanced_accuracy']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] |")
    lines += ['','## Limitations','', 'The same MASK_PROTO rule does not imply identical dynamic UID masks because each condition follows a different student trajectory. The MI term is the class-space (C,C) joint probability MI; KD is the existing T=2 KL with 0.5*T² scaling. Teacher logits/features are detached and only IFNet is optimized. No test artifact is read; test is used only for final evaluation. Prototype reliability is label-conditioned and in-sample, not OOF.','','Output files include `prototype_inventory.csv`, `control_validation.csv`, `run_manifest.csv`, `results_per_run.csv`, `reused_controls.csv`, `combined_results_per_run.csv`, `results_per_subject.csv`, `results_per_dataset.csv`, `paired_comparisons.csv`, `epoch_mask_metrics.csv`, `mask_traces/`, `training_history/`, `checkpoints/`, `predictions/`, and `execution_provenance.json`.']
    _atomic_text(output_root/'report.md','\n'.join(lines)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--resume',action='store_true'); ap.add_argument('--preflight-only',action='store_true',help='write inventory/control validation without training'); args=ap.parse_args(argv)
    global OLD_CONTROLS, MAIN_CSV
    config_path=Path(args.config); config_path=ROOT/config_path if not config_path.is_absolute() else config_path; cfg=yaml.safe_load(config_path.read_text()); _validate_config(cfg)
    if cfg.get('supplement_mode'): OLD_CONTROLS=()
    if cfg.get('main_csv'):
        MAIN_CSV=Path(cfg['main_csv']); MAIN_CSV=ROOT/MAIN_CSV if not MAIN_CSV.is_absolute() else MAIN_CSV
    output_root=Path(cfg.get('output_dir',DEFAULT_OUTPUT)); output_root=ROOT/output_root if not output_root.is_absolute() else output_root
    if output_root.exists() and any(output_root.iterdir()) and not args.resume: raise RuntimeError(f'formal output is non-empty; use --resume: {output_root}')
    output_root.mkdir(parents=True,exist_ok=True);
    if MAIN_CSV.exists() and not args.resume and MAIN_CSV.stat().st_size > 0: raise RuntimeError(f'new main CSV is non-empty; use --resume: {MAIN_CSV}')
    git_before=_git_snapshot(); _atomic_text(output_root/'git_status_before.txt',git_before['status_short']+'\n'); started=datetime.now(timezone.utc).isoformat()
    units,inventory=_preflight(cfg); _atomic_csv(output_root/'prototype_inventory.csv',inventory); reused,validation=_validate_controls(units,output_root); _atomic_csv(output_root/'reused_controls.csv',reused)
    resolved={'source_config':str(config_path.resolve()),'config':cfg,'scope':{'datasets':DATASETS,'subjects_by_dataset':DATASET_SUBJECTS,'experiment_units':len(units),'new_runs':len(units)*len(CONDITIONS),'subject_index_base':0,'report_subject_base':1,'protocol':'fewshot','train_fraction':.3,'test_fraction':.7},'conditions':list(CONDITIONS),'teacher':'mirepnet','student':'ifnet','artifact_root_read_only':str((ROOT/cfg['artifact_root']).resolve()),'output_root':str(output_root.resolve()),'no_test_artifacts_read':True}; _atomic_yaml(output_root/'config_resolved.yaml',resolved)
    controls_ok=(not OLD_CONTROLS) or len(reused)==len(units)*len(OLD_CONTROLS)
    if args.preflight_only:
        _write_manifest(output_root,units,[])
        _atomic_json(output_root/'execution_provenance.json',{'status':'preflight_complete','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'git':{'before':git_before,'after':_git_snapshot()},'new_runs':0,'reused_controls':len(reused),'control_validation':{'valid':controls_ok,'rows':len(validation)},'test_artifacts_read':False,'results_artifacts_written':False})
        _atomic_text(output_root/'report.md', f'# Teacher Prototype-Gated KD/MI Pilot\n\nPreflight-only: {len(units)} units and {len(reused)}/{len(units)*len(OLD_CONTROLS)} old controls validated; no formal run was started because this invocation was explicitly preflight-only.\n')
        print(f'[preflight-only] units={len(units)} controls={len(reused)}/{len(units)*len(OLD_CONTROLS)} output={output_root}')
        return 0
    if not torch.cuda.is_available():
        _write_manifest(output_root,units,[]); _atomic_json(output_root/'execution_provenance.json',{'status':'resource_blocked','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'reason':'CUDA unavailable; no formal run started','git':{'before':git_before,'after':_git_snapshot()},'gpu':_gpu_snapshot(args.gpu),'new_runs':0,'reused_controls':len(reused),'test_artifacts_read':False,'results_artifacts_written':False}); _report(output_root,units,inventory,reused,[],[],[],controls_ok); return 2
    device=f'cuda:{args.gpu}'; gpu_info=_gpu_snapshot(args.gpu); result_path=output_root/'results_per_run.csv'; metrics_path=output_root/'epoch_mask_metrics.csv'; existing=_read_csv(result_path); by={(r.get('dataset'),int(r.get('subject_index',-1)),r.get('condition')):r for r in existing if r.get('condition') in CONDITIONS}; metric_rows=_read_csv(metrics_path); metric_by={(r.get('dataset'),int(r.get('subject_index',-1)),r.get('condition'),int(r.get('epoch',-1))):r for r in metric_rows}; _write_manifest(output_root,units,list(by.values())); failures=[]
    for unit in units:
        for condition in CONDITIONS:
            key=(unit['dataset'],unit['subject_index'],condition); old=by.get(key)
            if old is not None and _new_valid(old,unit): print(f'[resume {len(by)}/{_run_count()}] {unit["dataset"]} S{unit["subject"]} {condition}',flush=True); continue
            try:
                row,epochs=_train_condition(unit,condition,output_root,cfg,device); by[key]=row
                for r in epochs: metric_by[(r['dataset'],int(r['subject_index']),r['condition'],int(r['epoch']))]=r
                print(f'[progress {len(by)}/{_run_count()}] {unit["dataset"]} S{unit["subject"]} {condition} BA={row["test_balanced_accuracy"]}',flush=True)
            except Exception as exc:
                row={'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],'seed':SEED,'condition':condition,'result_source':'new','status':'failed','epochs_completed':0,'failure_reason':f'{type(exc).__name__}: {exc}'}; by[key]=row; failures.append(row); print(f'[failed] {unit["dataset"]} S{unit["subject"]} {condition}: {exc}',flush=True)
            ordered=[by[k] for u in units for c in CONDITIONS for k in [(u['dataset'],u['subject_index'],c)] if k in by]; _atomic_csv(result_path,ordered); _atomic_csv(MAIN_CSV,ordered); _atomic_csv(metrics_path,[metric_by[k] for u in units for c in CONDITIONS for e in range(1,101) for k in [(u['dataset'],u['subject_index'],c,e)] if k in metric_by]); _write_manifest(output_root,units,ordered)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    new_rows=[by[k] for u in units for c in CONDITIONS for k in [(u['dataset'],u['subject_index'],c)] if k in by]
    if len(new_rows)!=_run_count() or failures or any(r.get('status')!='complete' for r in new_rows):
        _atomic_json(output_root/'execution_provenance.json',{'status':'incomplete','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'git':{'before':git_before,'after':_git_snapshot()},'gpu':gpu_info,'new_runs':len(new_rows),'failures':failures,'reused_controls':len(reused),'test_artifacts_read':False,'results_artifacts_written':False}); raise RuntimeError(f'pilot incomplete: {len(new_rows)}/{_run_count()} rows; failures={len(failures)}')
    combined,subjects,datasets,comparisons=_summaries(units,new_rows,reused,output_root); triplets=[]
    for u in units:
        rr=[r for r in new_rows if r['dataset']==u['dataset'] and int(r['subject_index'])==u['subject_index']]; fields=('train_uid_hash','test_uid_hash','split_uid_hash','initial_state_hash','preprocessing_hash','batch_order_hash','batch_order_hashes'); triplets.append({'dataset':u['dataset'],'subject':u['subject'],'hashes_equal':all(len({r.get(f) for r in rr})==1 for f in fields)})
    _atomic_json(output_root/'execution_provenance.json',{'status':'complete','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'config_path':str(config_path.resolve()),'config_sha256':_sha_file(config_path),'command':' '.join([sys.executable,*sys.argv]),'git':{'before':git_before,'after':_git_snapshot()},'python':platform.python_version(),'gpu':gpu_info,'scope':resolved['scope'],'new_runs':len(new_rows),'reused_controls':len(reused),'control_validation':{'valid':controls_ok,'rows':len(validation)},'new_triplet_hashes':triplets,'session_provenance':{d:_session(d) for d in DATASETS},'test_artifacts_read':False,'test_split_used_for_training':False,'results_artifacts_written':False,'limitations':['label-conditioned in-sample prototype gate','masked rows remain in forward/BatchNorm','same rule does not imply identical dynamic masks','only seed 666 was run']})
    _report(output_root,units,inventory,reused,new_rows,datasets,comparisons,controls_ok); print(f'[complete] {len(new_rows)}/{_run_count()} new runs; combined={len(combined)}; output={output_root}'); return 0


if __name__=='__main__': raise SystemExit(main())
