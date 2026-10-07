"""Matched MIRepNet-to-IFNet 10-epoch warm-up plus KD/MI run."""
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
from experiments.storage import external_path, require_external_output, resolve_local_file

_spec = importlib.util.spec_from_file_location(
    "_prototype_margin_reliability", ROOT / "test/qc/prototype_margin_reliability.py")
if _spec is None or _spec.loader is None:
    raise ImportError("prototype diagnostic helper cannot be loaded")
_proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proto)

_ALL_DATASET_SUBJECTS = {"BNCI2014001": 9, "BNCI2014001-4": 9,
                         "BNCI2014004": 9, "BNCI2015001": 12, "AlexMI": 8}
_DEFAULT_DATASETS = ["BNCI2014001", "BNCI2014001-4", "BNCI2014004", "BNCI2015001", "AlexMI"]
DATASETS = list(_DEFAULT_DATASETS)
DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
CONDITIONS = ("WARMUP10_KD_MI",)
OLD_CONTROLS = ()
COMPARISONS = ()
SEED = 666
VAL_SPLIT = 0.7
EPOCHS = 100
PROTO_EPS = 1e-12
INIT_ARGS = SimpleNamespace(epochs=None, lr=None, weight_decay=None, batch_size=None)
DEFAULT_CONFIG = ROOT / "configs/experiments/distill_warmup_mi_combo_seed666.yaml"
DEFAULT_OUTPUT = Path("/data1/llx/BigSmallcollab/results/distill/warmup_mi_combo_seed666")
MAIN_CSV = Path("/data1/llx/BigSmallcollab/results/distill/warmup_mi_combo_seed666.csv")
OLD_MI_CSV = Path("/data1/llx/BigSmallcollab/results/distill/distill_mi.csv")
OLD_MASK_ROOT = Path("/data1/llx/BigSmallcollab/results/distill/kd_mi_teacher_correct_mask_pilot")


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
    path = require_external_output(path)
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
    path = external_path(path)
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
    if configured != _DEFAULT_DATASETS:
        raise ValueError(f"dataset scope must be all five tasks: {_DEFAULT_DATASETS}")
    DATASETS = configured
    DATASET_SUBJECTS = {name: _ALL_DATASET_SUBJECTS[name] for name in DATASETS}
    expected = {"datasets": DATASETS, "protocol": "fewshot", "val_split": .7,
                "seed": SEED, "teacher": "mirepnet", "student": "ifnet",
                "epochs": 100, "optimizer": "adamw", "lr": .001,
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
    if not DATASET_SUBJECTS or any(name not in _ALL_DATASET_SUBJECTS for name in DATASETS):
        raise AssertionError("invalid dataset scope")


def _unit_count():
    return sum(DATASET_SUBJECTS[name] for name in DATASETS)


def _run_count():
    return _unit_count() * len(CONDITIONS)


def _student_cfg(dataset):
    cfg = config.load_model_config("ifnet", dataset, "fewshot")
    for key, value in {"epochs": 100, "lr": .001, "weight_decay": .01, "batch_size": 16}.items():
        if abs(float(cfg.get(key)) - value) > 1e-12:
            raise ValueError(f"IFNet {dataset} {key}={cfg.get(key)!r}; expected {value}")
    return cfg


def _artifact_path(cfg, dataset, subject_index):
    root = external_path(cfg.get("artifact_root", "/data1/llx/BigSmallcollab/results/artifacts"))
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
                      kd_active=True, mi_active=True, lam_kd=.5, temperature_kd=2.,
                      lam_mi=.1, mi_eps=1e-8):
    """Full CE plus the condition's KD/MI terms; gates only affect KD rows."""
    if condition not in CONDITIONS: raise ValueError(condition)
    keep = keep.to(student_logits.device).bool(); labels = labels.long()
    ce = F.cross_entropy(student_logits, labels); zero = student_logits.sum() * 0.
    n = int(keep.sum().item()); kd = mi = None
    if condition in ("KD_ALL", "WARMUP10", "KD_MI_ALL", "KD_PROTO", "WARMUP10_KD_MI") and kd_active:
        if n:
            t = teacher_logits.detach()[keep]; s = student_logits[keep]
            tp = F.softmax(t / temperature_kd, dim=1).detach()
            per = F.kl_div(F.log_softmax(s / temperature_kd, dim=1), tp,
                           reduction="none").sum(1)
            kd = per.mean()
        else: kd = zero
    if condition in ("KD_MI_ALL", "WARMUP10_KD_MI") and mi_active:
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
    path = require_external_output(path)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    torch.save(payload, require_external_output(tmp)); os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    path = require_external_output(path)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as f:
        tmp = Path(f.name)
    np.savez(require_external_output(tmp), **arrays); os.replace(tmp, path)


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
    started=time.time(); set_seed(SEED)
    adapter=get_adapter('ifnet', device=device, **dict(unit['student_cfg']))
    model=adapter.build(unit['num_classes'])
    state={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in unit['initial_state'].items()}
    missing,unexpected=model.load_state_dict(state, strict=False)
    if missing or unexpected: raise ValueError(f'initial state mismatch: {missing}, {unexpected}')
    y=torch.as_tensor(unit['ytr'], dtype=torch.long)
    teacher=torch.as_tensor(unit['teacher_logits'], dtype=torch.float32)
    teacher_pred=torch.as_tensor(unit['teacher_pred'], dtype=torch.long)
    reliable=torch.as_tensor(unit['teacher_proto_reliable'], dtype=torch.bool)
    index=torch.arange(len(y), dtype=torch.long)
    ds=TensorDataset(unit['xtr_pre'].cpu(),y,teacher,teacher_pred,reliable,index)
    generator=torch.Generator(); generator.set_state(torch.get_rng_state())
    loader=DataLoader(ds,batch_size=16,shuffle=True,drop_last=False,generator=generator)
    optimizer=optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    n=len(y); seen=np.zeros(n,dtype=np.int64)
    student_trace=np.empty((EPOCHS,n),dtype=np.int64); agreement_trace=np.zeros((EPOCHS,n),bool)
    rescue_trace=np.zeros((EPOCHS,n),bool); keep_trace=np.zeros((EPOCHS,n),bool)
    history=[]; epoch_rows=[]; order_hashes=[]; totals=Counter()
    for ep in range(EPOCHS):
        model.train(); digest=hashlib.sha256(); seen_ep=np.zeros(n,dtype=np.int64)
        ag=np.zeros(n,bool); rs=np.zeros(n,bool); kp=np.zeros(n,bool)
        condition_has_kd=condition in ('KD_ALL','WARMUP10','KD_MI_ALL','KD_PROTO','WARMUP10_KD_MI')
        warming=condition in ('WARMUP10','WARMUP10_KD_MI') and ep < 10
        has_kd=condition_has_kd and not warming
        has_mi=condition in ('KD_MI_ALL','WARMUP10_KD_MI') and not warming
        kd_count=0 if has_kd else None; mi_count=0 if has_mi else None; all_masked=0; mi_small=0 if has_mi else None
        ce_sum=total_sum=kd_sum=mi_sum=0.; ce_n=steps=mi_batches=0
        for xb,yb,tb,tpb,rb,ib in loader:
            idx=ib.numpy().astype(np.int64,copy=False); seen[idx]+=1
            digest.update(np.ascontiguousarray(unit['uid_tr'][idx],dtype=np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            # Full batch forward precedes every mask operation.
            _,logits=adapter.forward(model,xb.to(device)); sp=logits.detach().argmax(1)
            agree=tpb.to(device).eq(sp); rescue=(~agree)&rb.to(device)
            keep=(agree|rescue) if condition=='KD_PROTO' else torch.ones_like(agree)
            comp=masked_components(logits,yb.to(device),tb.to(device).detach(),keep,condition,
                                   kd_active=has_kd,mi_active=has_mi,
                                   lam_kd=float(cfg['lam_kd']),temperature_kd=float(cfg['temperature_kd']),
                                   lam_mi=float(cfg['lam_mi']),mi_eps=float(cfg['mi_eps']))
            comp['total'].backward(); optimizer.step()
            a=agree.detach().cpu().numpy(); r=rescue.detach().cpu().numpy(); k=keep.detach().cpu().numpy()
            ag[idx]=a; rs[idx]=r; kp[idx]=k; seen_ep[idx]+=1; student_trace[ep,idx]=sp.detach().cpu().numpy()
            agreement_trace[ep,idx]=a; rescue_trace[ep,idx]=r; keep_trace[ep,idx]=k
            selected=int(comp['selected_count']); ce=float(comp['ce_loss'].detach()); obj=float(comp['total'].detach())
            ce_sum+=ce*len(idx); ce_n+=len(idx); total_sum+=obj; steps+=1
            totals['ce_sum']+=ce*len(idx); totals['ce_n']+=len(idx); totals['total_sum']+=obj; totals['steps']+=1
            if selected==0: all_masked+=1
            if has_kd and selected:
                kd_count+=selected; value=float(comp['kd_loss'].detach()); kd_sum+=value*selected; totals['kd_sum']+=value*selected; totals['kd_n']+=selected
            if has_mi:
                if selected>=2:
                    mi_count+=selected; mi_batches+=1; value=float(comp['mi_loss'].detach()); mi_sum+=value; totals['mi_sum']+=value; totals['mi_batches']+=1; totals['mi_n']+=selected
                elif selected==1: mi_small+=1; totals['mi_small']+=1
        if not np.all(seen_ep==1): raise RuntimeError(f'{unit["dataset"]} S{unit["subject"]} epoch {ep+1}: UID schedule invalid')
        scheduler.step(); bh=digest.hexdigest(); order_hashes.append(bh)
        epoch_rows.append(_epoch_metric(unit,condition,ep+1,ag,rs,kp,kd_count,mi_count,all_masked,mi_small,bh))
        history.append({'epoch':ep+1,'ce_loss':ce_sum/ce_n,'kd_loss':kd_sum/kd_count if kd_count else None,
                        'mi_loss':mi_sum/mi_batches if mi_batches else None,'scaled_kd_contribution':2.*kd_sum/kd_count if kd_count else None,
                        'scaled_mi_contribution':.1*mi_sum/mi_batches if mi_batches else None,'total_loss':total_sum/steps,
                        'batch_order_hash':bh,'effective_ce_sample_count':ce_n,'effective_kd_sample_count':kd_count,
                        'effective_mi_sample_count':mi_count,'distill_all_masked_batch_count':all_masked,
                        'mi_skipped_small_batch_count':mi_small,'prediction_agreement_rate':float(ag.mean()),
                        'prototype_rescue_rate':float(rs.sum()/(~ag).sum()) if (~ag).sum() else None,'mask_keep_rate':float(kp.mean())})
    if not np.all(seen==EPOCHS): raise RuntimeError('not every train UID appeared once per epoch')
    train_logits=_inference(adapter,model,unit['xtr_pre']); test_logits=_inference(adapter,model,unit['xte_pre'])
    train_preds=train_logits.argmax(1).astype(np.int64); test_preds=test_logits.argmax(1).astype(np.int64)
    final_agree=unit['teacher_pred']==train_preds; final_rescue=(~final_agree)&unit['teacher_proto_reliable']
    final_keep=(final_agree|final_rescue) if condition=='KD_PROTO' else np.ones_like(final_agree,dtype=bool)
    run_key=_run_key(unit,condition); trace=output_root/'mask_traces'/f'{run_key}.npz'
    _atomic_npz(trace,sample_uid=unit['uid_tr'],labels=unit['ytr'],teacher_pred=unit['teacher_pred'],teacher_proto_pred=unit['teacher_proto_pred'],teacher_proto_reliable=unit['teacher_proto_reliable'],teacher_support_margin=unit['teacher_support'],epoch=np.arange(1,EPOCHS+1),student_pred=student_trace,agreement_mask=agreement_trace,prototype_rescue_mask=rescue_trace,keep_mask=keep_trace)
    trace_sha=_sha_file(trace); history_path=output_root/'training_history'/f'{run_key}.json'; prediction_path=output_root/'predictions'/f'{run_key}.npz'; checkpoint_path=output_root/'checkpoints'/f'{run_key}.pt'
    _atomic_json(history_path,history)
    _atomic_npz(prediction_path,train_uid=unit['uid_tr'],test_uid=unit['uid_te'],train_y=unit['ytr'],test_y=unit['yte'],train_logits=train_logits.astype(np.float32),test_logits=test_logits.astype(np.float32),train_preds=train_preds,test_preds=test_preds,final_agreement=final_agree,final_prototype_rescue=final_rescue,final_mask_keep=final_keep)
    order_hash=_combine(order_hashes)
    _atomic_torch(checkpoint_path,{'complete':True,'run_key':run_key,'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],'seed':SEED,'condition':condition,'epochs':EPOCHS,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'train_uid_hash':unit['train_uid_hash'],'test_uid_hash':unit['test_uid_hash'],'split_uid_hash':unit['split_uid_hash'],'initial_state_hash':unit['initial_state_hash'],'preprocessing_hash':unit['preprocessing_hash'],'batch_order_hash':order_hash,'batch_order_hashes':order_hashes,'teacher_uid_alignment_hash':unit['teacher_uid_alignment_hash'],'prototype_state_hash':unit['prototype_state_hash'],'mask_trace_sha256':trace_sha})
    ev=metrics.evaluate(unit['yte'],test_preds); ba=float(balanced_accuracy_score(unit['yte'],test_preds)*100.)
    has_kd=condition in ('KD_ALL','WARMUP10','KD_MI_ALL','KD_PROTO','WARMUP10_KD_MI'); has_mi=condition=='WARMUP10_KD_MI'
    row={'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],'session':unit['session'],'seed':SEED,'protocol':'fewshot','teacher':'mirepnet','student':'ifnet','condition':condition,'result_source':'new','train_count':len(unit['ytr']),'test_count':len(unit['yte']),'teacher_proto_reliable_count':int(unit['teacher_proto_reliable'].sum()),'teacher_proto_unreliable_count':int((~unit['teacher_proto_reliable']).sum()),'teacher_proto_reliable_rate':float(unit['teacher_proto_reliable'].mean()),'final_epoch_prediction_agreement_rate':float(final_agree.mean()),'final_epoch_prototype_rescue_rate':float(final_rescue.sum()/(~final_agree).sum()) if (~final_agree).sum() else '','final_epoch_mask_keep_rate':float(final_keep.mean()),'mean_epoch_prediction_agreement_rate':float(np.mean([r['prediction_agreement_rate'] for r in epoch_rows])),'mean_epoch_prototype_rescue_rate':float(np.nanmean([float(r['prototype_rescue_rate']) if r['prototype_rescue_rate']!='' else np.nan for r in epoch_rows])),'mean_epoch_mask_keep_rate':float(np.mean([r['mask_keep_rate'] for r in epoch_rows])),'final_train_eval_agreement_rate':float(final_agree.mean()),'final_train_eval_prototype_rescue_rate':float(final_rescue.sum()/(~final_agree).sum()) if (~final_agree).sum() else '','final_train_eval_mask_keep_rate':float(final_keep.mean()),'final_train_eval_kept_uid':_json(unit['uid_tr'][final_keep].astype(int).tolist()),'final_train_eval_masked_uid':_json(unit['uid_tr'][~final_keep].astype(int).tolist()),'effective_kd_sample_exposures':int(totals['kd_n']) if has_kd else '','effective_mi_sample_exposures':int(totals['mi_n']) if has_mi else '','distill_all_masked_batch_count':int(sum(r['distill_all_masked_batch_count'] for r in epoch_rows)),'mi_skipped_small_batch_count':int(totals['mi_small']) if has_mi else '','mean_ce_loss':totals['ce_sum']/totals['ce_n'],'mean_kd_loss':totals['kd_sum']/totals['kd_n'] if has_kd and totals['kd_n'] else '','mean_mi_loss':totals['mi_sum']/totals['mi_batches'] if has_mi and totals['mi_batches'] else '','mean_scaled_kd_contribution':2.*totals['kd_sum']/totals['kd_n'] if has_kd and totals['kd_n'] else '','mean_scaled_mi_contribution':.1*totals['mi_sum']/totals['mi_batches'] if has_mi and totals['mi_batches'] else '','mean_total_loss':totals['total_sum']/totals['steps'],'final_train_accuracy':float((train_preds==unit['ytr']).mean()*100.),'test_accuracy':float(ev['acc']),'test_balanced_accuracy':ba,'test_kappa':float(ev['kappa']),'predicted_class_counts':_json(np.bincount(test_preds,minlength=unit['num_classes']).tolist()),'collapse_flag':bool(len(np.unique(test_preds))<2),'runtime_seconds':time.time()-started,'train_uid_hash':unit['train_uid_hash'],'test_uid_hash':unit['test_uid_hash'],'split_uid_hash':unit['split_uid_hash'],'initial_state_hash':unit['initial_state_hash'],'preprocessing_hash':unit['preprocessing_hash'],'batch_order_hash':order_hash,'batch_order_hashes':_json(order_hashes),'teacher_uid_alignment_hash':unit['teacher_uid_alignment_hash'],'teacher_artifact_sha256':unit['artifact_sha256'],'teacher_logits_hash':unit['inventory']['teacher_logits_hash'],'teacher_feature_hash':unit['inventory']['teacher_feature_hash'],'prototype_state_hash':unit['prototype_state_hash'],'mask_trace_path':str(trace.resolve()),'mask_trace_sha256':trace_sha,'epochs_completed':EPOCHS,'status':'complete','failure_reason':'','checkpoint_path':str(checkpoint_path.resolve()),'history_path':str(history_path.resolve()),'prediction_path':str(prediction_path.resolve())}
    return row,epoch_rows


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
            hist=json.loads(resolve_local_file(row['history_path']).read_text()); check['history_unmasked']=len(hist)==100 and all(int(x.get('effective_ce_sample_count',-1))==len(unit['ytr']) and int(x.get('effective_kd_sample_count',-1))==len(unit['ytr']) and int(x.get('effective_mi_sample_count',-1))==len(unit['ytr']) and int(x.get('all_masked_batch_count',-1))==0 for x in hist)
        except Exception: check['history_unmasked']=False
    for key in ('checkpoint_path','history_path','prediction_path'): check[key]=bool(row.get(key)) and external_path(row[key]).exists()
    if check.get('checkpoint_path'):
        try:
            p=torch.load(resolve_local_file(row['checkpoint_path']),map_location='cpu',weights_only=True); check['checkpoint_complete']=bool(p.get('complete')) and int(p.get('epochs',-1))==100
        except Exception: check['checkpoint_complete']=False
    else: check['checkpoint_complete']=False
    if check.get('history_path'):
        try: check['history_100']=len(json.loads(resolve_local_file(row['history_path']).read_text()))==100
        except Exception: check['history_100']=False
    else: check['history_100']=False
    if check.get('prediction_path'):
        try:
            with np.load(resolve_local_file(row['prediction_path']),allow_pickle=False) as p: check['prediction_complete']=p['test_preds'].shape[0]==len(unit['yte'])
        except Exception: check['prediction_complete']=False
    else: check['prediction_complete']=False
    return check,[k for k,v in check.items() if not v]


def _validate_controls(units, output_root):
    # All student controls are trained in this run. No historical score is reused.
    _atomic_csv(output_root/'reused_controls.csv',[])
    return [],[]


def _teacher_baselines(units, output_root):
    """Score the frozen teacher on the exact held-out UIDs used by this run."""
    rows=[]
    for unit in units:
        path=Path('/data1/llx/BigSmallcollab/results/artifacts')/unit['dataset']/'mirepnet'/f"{unit['subject_index']}_{SEED}_test.npz"
        if not path.exists():
            raise FileNotFoundError(f'missing teacher test artifact: {path}')
        with np.load(resolve_local_file(path),allow_pickle=False) as a:
            payload={k:np.asarray(a[k]) for k in ('sample_uid','logits','feats','y')}
        aligned=_proto.align_by_uid({'sample_uid':unit['uid_te'],'y':unit['yte']},payload)
        pred=np.asarray(aligned['logits']).argmax(1).astype(np.int64)
        ev=metrics.evaluate(unit['yte'],pred)
        rows.append({'dataset':unit['dataset'],'subject':unit['subject'],'subject_index':unit['subject_index'],
                     'seed':SEED,'model':'MIRepNet_teacher','test_count':len(unit['yte']),
                     'test_accuracy':float(ev['acc']),'test_balanced_accuracy':float(balanced_accuracy_score(unit['yte'],pred)*100.),
                     'test_kappa':float(ev['kappa']),'test_uid_hash':unit['test_uid_hash'],
                     'teacher_test_artifact':str(path.resolve()),'teacher_test_artifact_sha256':_sha_file(path),
                     'uid_alignment':'pass'})
    _atomic_csv(output_root/'teacher_baselines_per_subject.csv',rows)
    return rows


def _new_valid(row, unit):
    if row.get('status')!='complete' or str(row.get('epochs_completed'))!='100': return False
    for key,value in [('train_uid_hash',unit['train_uid_hash']),('test_uid_hash',unit['test_uid_hash']),('split_uid_hash',unit['split_uid_hash']),('initial_state_hash',unit['initial_state_hash']),('preprocessing_hash',unit['preprocessing_hash']),('teacher_artifact_sha256',unit['artifact_sha256']),('teacher_uid_alignment_hash',unit['teacher_uid_alignment_hash']),('prototype_state_hash',unit['prototype_state_hash'])]:
        if row.get(key)!=value:return False
    if not all(_finite(row.get(k)) for k in ('test_accuracy','test_balanced_accuracy','test_kappa')):return False
    try:
        hist=json.loads(resolve_local_file(row['history_path']).read_text()); p=torch.load(resolve_local_file(row['checkpoint_path']),map_location='cpu',weights_only=True)
        if len(hist)!=100 or not p.get('complete') or int(p.get('epochs',-1))!=100:return False
        with np.load(resolve_local_file(row['prediction_path']),allow_pickle=False) as a:
            if a['test_preds'].shape[0]!=len(unit['yte']):return False
        return Path(row['mask_trace_path']).exists() and _sha_file(Path(row['mask_trace_path']))==row.get('mask_trace_sha256')
    except Exception:return False


def _write_manifest(output_root,units,rows):
    lookup={(r.get('dataset'),int(r.get('subject_index',-1)),r.get('condition')):r for r in rows}; out=[]
    for u in units:
        for c in CONDITIONS:
            r=lookup.get((u['dataset'],u['subject_index'],c)); out.append({'run_key':_run_key(u,c),'dataset':u['dataset'],'subject':u['subject'],'subject_index':u['subject_index'],'seed':SEED,'condition':c,'status':r.get('status','pending') if r else 'pending','epochs_completed':r.get('epochs_completed',0) if r else 0,'failure_reason':r.get('failure_reason','') if r else ''})
    _atomic_csv(output_root/'run_manifest.csv',out)


def _bootstrap(diff,seed=None,count=10000):
    x=np.asarray(diff,dtype=float)
    if not len(x):return [np.nan,np.nan]
    if seed is None: seed=SEED
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
            diff=np.asarray([by[(ds,s,method)]-by[(ds,s,base)] for s in ids],float); ci=_bootstrap(diff,seed=SEED)
            try:
                from scipy.stats import wilcoxon
                p=float(wilcoxon(diff).pvalue) if np.any(diff!=0) else 1.
            except Exception:p=np.nan
            comparisons.append({'dataset':ds,'comparison':f'{method}-{base}','status':'available','n_subject':len(diff),'mean_delta_balanced_accuracy':float(diff.mean()),'wins':int((diff>1e-12).sum()),'ties':int((np.abs(diff)<=1e-12).sum()),'losses':int((diff< -1e-12).sum()),'bootstrap_seed':SEED,'bootstrap_count':10000,'bootstrap_ci95_low':ci[0],'bootstrap_ci95_high':ci[1],'wilcoxon_p':p})
    _atomic_csv(output_root/'paired_comparisons.csv',comparisons)
    macro=[]; task_source={'BNCI2014001+BNCI2014001-4':['BNCI2014001','BNCI2014001-4'],
                           'BNCI2014004':['BNCI2014004'],'BNCI2015001':['BNCI2015001'],'AlexMI':['AlexMI']}
    for method,base in COMPARISONS:
        by_ds={}
        for ds in DATASETS:
            ss=sorted({s for d,s,c in by if d==ds and c==method}&{s for d,s,c in by if d==ds and c==base})
            by_ds[ds]={s:by[(ds,s,method)]-by[(ds,s,base)] for s in ss}
        source_values={}
        shared=set(by_ds['BNCI2014001']) & set(by_ds['BNCI2014001-4'])
        source_values['BNCI2014001+BNCI2014001-4']={s:(by_ds['BNCI2014001'][s]+by_ds['BNCI2014001-4'][s])/2 for s in shared}
        for ds in ('BNCI2014004','BNCI2015001','AlexMI'): source_values[ds]=by_ds[ds]
        means={k:float(np.mean(list(v.values()))) for k,v in source_values.items() if v}
        rng=np.random.RandomState(SEED); draws=np.zeros(10000,dtype=float)
        for j in range(len(draws)):
            source_means=[]
            for values in source_values.values():
                vals=np.asarray(list(values.values()),dtype=float)
                if len(vals): source_means.append(float(vals[rng.randint(0,len(vals),size=len(vals))].mean()))
            draws[j]=float(np.mean(source_means)) if source_means else np.nan
        vals=list(means.values())
        macro.append({'comparison':f'{method}-{base}','source_count':len(vals),
                      'equal_weight_source_macro_delta_ba':float(np.mean(vals)) if vals else np.nan,
                      'positive_sources':int(sum(v>1e-12 for v in vals)),
                      'negative_sources':int(sum(v< -1e-12 for v in vals)),
                      'tied_sources':int(sum(abs(v)<=1e-12 for v in vals)),
                      'bootstrap_seed':SEED,'bootstrap_count':len(draws),
                      'bootstrap_ci95_low':float(np.percentile(draws,2.5)) if vals else np.nan,
                      'bootstrap_ci95_high':float(np.percentile(draws,97.5)) if vals else np.nan,
                      'source_deltas':_json(means),
                      'ci_interpretation':f'within-source subject resampling; fixed sources and seed {SEED}'})
    _atomic_csv(output_root/'source_macro_comparisons.csv',macro)
    return combined,subjects,datasets,comparisons


def _report(output_root, units, inventory, reused, new_rows, datasets, comparisons, controls_ok):
    baseline_root = (Path('/data1/llx/BigSmallcollab/results/distill/seed666_three_ablation') if SEED == 666
                     else ROOT / f'/data1/llx/BigSmallcollab/results/distill/three_seed_ablation_seed{SEED}')
    baseline_rows = _read_csv(baseline_root / 'results_per_run.csv')
    baseline = {(r['dataset'], int(r['subject_index']), r['condition']): r for r in baseline_rows}
    combo = {(r['dataset'], int(r['subject_index'])): r for r in new_rows}
    columns = ('BASE_CE', 'KD_ALL', 'WARMUP10', 'KD_MI_ALL', 'WARMUP10_KD_MI')
    lines = [f'# IFNet warm-up + KD + MI combination, seed {SEED}', '',
             f'- Five tasks, {len(units)} task-subject units; each combination run trains for 100 epochs.',
             '- Epochs 1–10 use CE only. Epochs 11–100 use CE + all-sample logits KD + class-probability MI.',
             '- KD uses temperature 2 and weight 0.5. MI uses probability mutual information at temperature 1 and weight 0.1.',
             '- No feature alignment or prototype gate is used. Comparisons are paired by task and subject against the matched three-seed ablation runs.', '',
             '## Test balanced accuracy (mean ± SD, %)', '',
             '| task | IFNet CE | KD_ALL | warm-up only | KD+MI | warm-up + KD+MI |',
             '|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        cells = []
        for cond in columns:
            vals = []
            for subj in range(DATASET_SUBJECTS[ds]):
                row = combo[(ds, subj)] if cond == 'WARMUP10_KD_MI' else baseline.get((ds, subj, cond))
                if row and _finite(row.get('test_balanced_accuracy')):
                    vals.append(float(row['test_balanced_accuracy']))
            cells.append(f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}' if vals else 'NA')
        lines.append(f'| {ds} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired deltas against the two basic baselines', '',
              '| baseline | four-source macro Δ BA (pp) | source directions (+/−/=) |', '|---|---:|---:|']
    source_sets = (('BNCI2014001', 'BNCI2014001-4'), ('BNCI2014004',),
                   ('BNCI2015001',), ('AlexMI',))
    for cond in ('BASE_CE', 'KD_ALL'):
        task_delta = {}
        for ds in DATASETS:
            task_delta[ds] = {subj: float(combo[(ds, subj)]['test_balanced_accuracy']) -
                                      float(baseline[(ds, subj, cond)]['test_balanced_accuracy'])
                              for subj in range(DATASET_SUBJECTS[ds])}
        source_means = []
        for source in source_sets:
            if len(source) == 1:
                values = list(task_delta[source[0]].values())
            else:
                shared = sorted(set(task_delta[source[0]]) & set(task_delta[source[1]]))
                values = [np.mean([task_delta[source[0]][subj], task_delta[source[1]][subj]])
                          for subj in shared]
            source_means.append(float(np.mean(values)))
        macro_delta = float(np.mean(source_means))
        directions = (sum(x > 1e-12 for x in source_means),
                      sum(x < -1e-12 for x in source_means),
                      sum(abs(x) <= 1e-12 for x in source_means))
        lines.append(f'| {cond} | {macro_delta:+.3f} | {directions[0]}/{directions[1]}/{directions[2]} |')
    lines += ['', 'This report is seed-specific. See the three-seed summary for cross-seed uncertainty and the final paired comparison.',
              '', 'Output files include `results_per_run.csv`, `epoch_mask_metrics.csv`, `run_manifest.csv`, `prototype_inventory.csv`, `teacher_baselines_per_subject.csv`, and `execution_provenance.json`.']
    _atomic_text(output_root/'report.md', '\n'.join(lines) + '\n')

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--resume',action='store_true'); ap.add_argument('--preflight-only',action='store_true',help='write inventory/control validation without training'); args=ap.parse_args(argv)
    global OLD_CONTROLS, MAIN_CSV
    global SEED
    config_path=Path(args.config); config_path=ROOT/config_path if not config_path.is_absolute() else config_path; cfg=yaml.safe_load(resolve_local_file(config_path).read_text()); SEED=int(cfg.get('seed',SEED)); _validate_config(cfg)
    if cfg.get('supplement_mode'): OLD_CONTROLS=()
    if cfg.get('main_csv'):
        MAIN_CSV=require_external_output(cfg['main_csv'])
    output_root=require_external_output(cfg.get('output_dir',DEFAULT_OUTPUT))
    if output_root.exists() and any(output_root.iterdir()) and not args.resume: raise RuntimeError(f'formal output is non-empty; use --resume: {output_root}')
    output_root.mkdir(parents=True,exist_ok=True);
    if MAIN_CSV.exists() and not args.resume and MAIN_CSV.stat().st_size > 0: raise RuntimeError(f'new main CSV is non-empty; use --resume: {MAIN_CSV}')
    git_before=_git_snapshot(); _atomic_text(output_root/'git_status_before.txt',git_before['status_short']+'\n'); started=datetime.now(timezone.utc).isoformat()
    units,inventory=_preflight(cfg); _atomic_csv(output_root/'prototype_inventory.csv',inventory); teacher_baselines=_teacher_baselines(units,output_root); reused,validation=_validate_controls(units,output_root); _atomic_csv(output_root/'reused_controls.csv',reused)
    resolved={'source_config':str(config_path.resolve()),'config':cfg,'scope':{'datasets':DATASETS,'subjects_by_dataset':DATASET_SUBJECTS,'experiment_units':len(units),'new_runs':len(units)*len(CONDITIONS),'subject_index_base':0,'report_subject_base':1,'protocol':'fewshot','train_fraction':.3,'test_fraction':.7},'conditions':list(CONDITIONS),'teacher':'mirepnet','student':'ifnet','artifact_root_read_only':str((ROOT/cfg['artifact_root']).resolve()),'output_root':str(output_root.resolve()),'teacher_test_artifacts_read_for_reference_only':True,'test_split_used_for_training':False}; _atomic_yaml(output_root/'config_resolved.yaml',resolved)
    controls_ok=(not OLD_CONTROLS) or len(reused)==len(units)*len(OLD_CONTROLS)
    if args.preflight_only:
        _write_manifest(output_root,units,[])
        _atomic_json(output_root/'execution_provenance.json',{'status':'preflight_complete','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'git':{'before':git_before,'after':_git_snapshot()},'new_runs':0,'reused_controls':len(reused),'control_validation':{'valid':controls_ok,'rows':len(validation)},'teacher_test_artifacts_read_for_reference_only':True,'test_split_used_for_training':False,'student_result_artifacts_written':False})
        _atomic_text(output_root/'report.md', f'# MIRepNet → IFNet seed-{SEED} matched ablations\n\nPreflight-only: {len(units)} units validated; no formal student run was started.\n')
        print(f'[preflight-only] units={len(units)} output={output_root}')
        return 0
    if not torch.cuda.is_available():
        _write_manifest(output_root,units,[]); _atomic_json(output_root/'execution_provenance.json',{'status':'resource_blocked','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'reason':'CUDA unavailable; no formal run started','git':{'before':git_before,'after':_git_snapshot()},'gpu':_gpu_snapshot(args.gpu),'new_runs':0,'reused_controls':len(reused),'teacher_test_artifacts_read_for_reference_only':True,'test_split_used_for_training':False,'student_result_artifacts_written':False}); _report(output_root,units,inventory,reused,[],[],[],controls_ok); return 2
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
    _atomic_json(output_root/'execution_provenance.json',{'status':'complete','started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'config_path':str(config_path.resolve()),'config_sha256':_sha_file(config_path),'runner_path':str(Path(__file__).resolve()),'runner_sha256':_sha_file(Path(__file__)),'command':' '.join([sys.executable,*sys.argv]),'git':{'before':git_before,'after':_git_snapshot()},'python':platform.python_version(),'gpu':gpu_info,'scope':resolved['scope'],'new_runs':len(new_rows),'reused_controls':len(reused),'control_validation':{'valid':controls_ok,'rows':len(validation)},'new_triplet_hashes':triplets,'session_provenance':{d:_session(d) for d in DATASETS},'teacher_test_artifacts_read_for_reference_only':True,'test_split_used_for_training':False,'student_result_artifacts_written':True,'limitations':['label-conditioned in-sample prototype gate','masked rows remain in forward/BatchNorm','same rule does not imply identical dynamic masks',f'only seed {SEED} was run']})
    _report(output_root,units,inventory,reused,new_rows,datasets,comparisons,controls_ok); print(f'[complete] {len(new_rows)}/{_run_count()} new runs; combined={len(combined)}; output={output_root}'); return 0


if __name__=='__main__': raise SystemExit(main())
