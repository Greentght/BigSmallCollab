"""Fixed CE-warmup and native-feature KD paths for the YAML runner.

Keep this recipe separate from the source files frozen by manual queue jobs.
It shares the existing seed, sampling, model and metric utilities.
"""
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from collab import distill as shared
from experiments.distill import run_distill as legacy


def _method_kwargs(args, method="KD_warmup10"):
    kwargs, weight_mode = legacy._method_kwargs("KD_all", args, {}, None)
    if method == "KD_warmup10":
        kwargs["distill_warmup_epochs"] = 10
    elif method == "KD_feature":
        kwargs["distill_warmup_epochs"] = 0
        kwargs["lam_feat"] = args.lam_feat
    elif method == "KD_warmup10_feature":
        kwargs["distill_warmup_epochs"] = 10
        kwargs["lam_feat"] = args.lam_feat
    else:
        raise ValueError(f"unsupported KD method {method!r}")
    return kwargs, weight_mode


def distill_student(student_adapter, num_classes, X_tr, y_tr, feat_t, log_t, X_te,
                    lam_kd=0.5, temperature=2.0, epochs=100, lr=0.001,
                    weight_decay=0.01, batch_size=16, balanced_batch=False,
                    seed=None, subject_ids=None, initial_state_dict=None,
                    sample_uid=None, return_training_details=False,
                    distill_warmup_epochs=10, lam_feat=0.0,
                    teacher_correct_only=False):
    """Unmasked CE+KD and optional native cosine alignment after the CE prefix."""
    if (isinstance(distill_warmup_epochs, (bool, np.bool_))
            or not isinstance(distill_warmup_epochs, (int, np.integer))
            or distill_warmup_epochs < 0):
        raise ValueError("distill_warmup_epochs must be a non-negative integer")
    if distill_warmup_epochs >= epochs:
        raise ValueError("distill_warmup_epochs must be less than epochs")
    if teacher_correct_only:
        raise ValueError("this KD path uses all training samples")
    if not np.isfinite(lam_feat) or lam_feat < 0:
        raise ValueError("lam_feat must be non-negative and finite")
    if not np.isfinite(lam_kd) or lam_kd <= 0:
        raise ValueError("lam_kd must be positive and finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    teacher = np.asarray(log_t, dtype=np.float32)
    if teacher.shape != (len(y_tr), num_classes) or not np.isfinite(teacher).all():
        raise ValueError("teacher logits must be finite and match training labels/classes")
    features = None
    if lam_feat > 0:
        if feat_t is None:
            raise ValueError("KD+feature requires teacher features")
        features = np.asarray(feat_t, dtype=np.float32)
        if (features.ndim != 2 or features.shape[0] != len(y_tr)
                or features.shape[1] < 1 or not np.isfinite(features).all()):
            raise ValueError("teacher features must be a finite (N, D) array")
    uid = None if sample_uid is None else np.asarray(sample_uid, dtype=np.int64)
    if uid is not None and (uid.shape != (len(y_tr), 2)
                            or len({tuple(row) for row in uid.tolist()}) != len(uid)):
        raise ValueError("sample_uid must contain unique (N, 2) trial UIDs")
    if seed is not None:
        shared._set_seed(seed)
    device = student_adapter.device
    model = student_adapter.build(num_classes)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    projector = None
    if features is not None:
        feature_dim = next((getattr(model, name) for name in
                            ("feat_dim", "flatten_size", "feat_size")
                            if hasattr(model, name)), None)
        if feature_dim is None:
            raise ValueError("student model does not expose its native feature dimension")
        # No train-mode probe: preserve student BN/dropout state and batch RNG.
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed or 0) + 7919)
            projector = torch.nn.Linear(int(feature_dim), features.shape[1]).to(device)
    values = student_adapter.preprocess(X_tr).to(device)
    labels = torch.as_tensor(y_tr, dtype=torch.long)
    feature_tensor = (torch.from_numpy(features) if features is not None
                      else torch.empty((len(labels), 0), dtype=torch.float32))
    dataset = TensorDataset(values.cpu(), labels, torch.from_numpy(teacher), feature_tensor,
                            torch.arange(len(labels), dtype=torch.long))
    if balanced_batch and subject_ids is not None:
        sampler = shared.ClassSubjectBalancedSampler(
            np.asarray(y_tr), np.asarray(subject_ids), batch_size,
            num_classes, seed=(seed or 0))
        loader = DataLoader(dataset, batch_sampler=sampler)
    elif balanced_batch:
        sampler = shared.BalancedBatchSampler(
            np.asarray(y_tr), batch_size, num_classes, seed=(seed or 0))
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    parameters = list(model.parameters())
    if projector is not None:
        parameters.extend(projector.parameters())
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, batch_hashes = [], []
    for epoch in range(epochs):
        model.train()
        kd_active = epoch >= distill_warmup_epochs
        total_sum = ce_sum = kd_sum = feature_sum = 0.0
        correct = count = 0
        order_hash = hashlib.sha256()
        for xb, yb, teacher_logits, teacher_features, indices in loader:
            indices = indices.numpy().astype(np.int64, copy=False)
            order_hash.update((indices if uid is None else uid[indices]).tobytes())
            xb, yb = xb.to(device), yb.to(device)
            student_features, logits = student_adapter.forward(model, xb)
            ce = F.cross_entropy(logits, yb)
            kd = ce.new_zeros(())
            feature_loss = ce.new_zeros(())
            if kd_active:
                per_sample = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits.to(device) / temperature, dim=1),
                    reduction="none").sum(dim=1)
                kd = temperature ** 2 * per_sample.sum() / len(yb)
                if projector is not None:
                    feature_loss = (1 - F.cosine_similarity(
                        projector(student_features), teacher_features.to(device).detach(),
                        dim=1, eps=1e-8)).mean()
            loss = ce + lam_kd * kd + lam_feat * feature_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = len(yb)
            count += n
            correct += int((logits.detach().argmax(dim=1) == yb).sum().item())
            total_sum += float(loss.detach().item()) * n
            ce_sum += float(ce.detach().item()) * n
            kd_sum += float(kd.detach().item()) * n
            feature_sum += float(feature_loss.detach().item()) * n
        scheduler.step()
        batch_hashes.append(order_hash.hexdigest())
        history.append({
            "epoch": epoch + 1, "kd_active": kd_active,
            "total_loss": total_sum / max(1, count),
            "ce_loss": ce_sum / max(1, count), "kd_loss": kd_sum / max(1, count),
            "feature_loss": feature_sum / max(1, count),
            "mi_loss": None, "full_train_mi": None,
            "train_accuracy": correct / max(1, count) * 100.0,
            "n_train": count, "batch_order_hash": batch_hashes[-1],
        })
    _, test_logits = student_adapter.infer(model, X_te)
    predictions = test_logits.argmax(1)
    if not return_training_details:
        return predictions
    _, train_logits = student_adapter.infer(model, X_tr)
    return predictions, {
        "model": model, "projector": projector, "training_history": history,
        "batch_order_hashes": batch_hashes,
        "train_logits": train_logits, "test_logits": test_logits,
        "train_preds": train_logits.argmax(1), "test_preds": predictions,
        "full_train_mi_history": [],
    }


def run_method(args, dataset, protocol, teacher, student, teacher_artifact,
               key, seed, method, X_tr, y_tr, X_te, y_te, subj_tr, tch,
               nc, device, initial_state_dict=None, uid_tr=None, uid_te=None,
               return_details=False):
    """Return the normal YAML result/artifact schema for warmup or feature KD."""
    cfg = legacy._student_runtime_cfg(student, dataset, protocol, args, X_tr)
    adapter = legacy.get_adapter(student, device=device, **cfg)
    params, weight_mode = _method_kwargs(args, method)
    teacher_features = tch['feats'] if params['lam_feat'] > 0 else None
    training = {name: cfg[name] for name in ("epochs", "lr", "weight_decay", "batch_size")}
    predictions, details = distill_student(
        adapter, nc, X_tr, y_tr, teacher_features, tch["logits"], X_te,
        lam_kd=params["lam_kd"], temperature=params["temperature"],
        lam_feat=params['lam_feat'],
        distill_warmup_epochs=params["distill_warmup_epochs"],
        seed=seed, balanced_batch=(protocol == "loso" and subj_tr is not None),
        subject_ids=subj_tr, initial_state_dict=initial_state_dict,
        sample_uid=uid_tr, return_training_details=True, **training)
    metrics = legacy.metrics.evaluate(y_te, predictions)
    balanced_accuracy = float(legacy.balanced_accuracy_score(y_te, predictions) * 100.0)
    finite = all(legacy._finite_metric(value) for value in
                 (metrics["acc"], metrics["kappa"], balanced_accuracy))
    last = details["training_history"][-1]
    diagnostics = legacy._final_train_mi_and_agreement(
        tch["logits"], details["train_logits"], details["train_preds"])
    row = {
        "dataset": dataset, "subject": int(key) + 1, "key": int(key),
        "session": legacy._session_default(dataset), "protocol": protocol,
        "teacher": teacher, "student": student, "teacher_artifact": teacher_artifact,
        "seed": int(seed), "method": method,
        "condition_label": {
            "KD_warmup10": "10-epoch CE warmup + Vanilla KD",
            "KD_feature": "KD + native cosine feature alignment",
            "KD_warmup10_feature": "10-epoch CE warmup + KD + native cosine feature alignment",
        }[method],
        "acc": metrics["acc"], "test_accuracy": metrics["acc"],
        "test_balanced_accuracy": round(balanced_accuracy, 4),
        "kappa": metrics["kappa"], "test_kappa": metrics["kappa"],
        "n_train": len(y_tr), "n_test": len(y_te),
        "train_count": len(y_tr), "test_count": len(y_te), "num_classes": int(nc),
        "teacher_train_acc_pct": round(legacy._teacher_acc(tch, y_tr), 2),
        "teacher_train_uid_alignment": json.dumps(tch.get("_uid_alignment", {}), sort_keys=True),
        "teacher_artifact_path": tch.get("_artifact_info", {}).get("path", ""),
        "teacher_artifact_sha256": tch.get("_artifact_info", {}).get("sha256", ""),
        "teacher_checkpoint_sha256": tch.get("_manifest", {}).get("teacher_checkpoint_sha256", ""),
        "train_uid_hash": legacy._sha256_array(uid_tr) if uid_tr is not None else "",
        "test_uid_hash": legacy._sha256_array(uid_te) if uid_te is not None else "",
        "split_uid_hash": (legacy._sha256_uid_split(uid_tr, uid_te)
                           if uid_tr is not None and uid_te is not None else ""),
        "initial_state_hash": (legacy._sha256_state_dict(initial_state_dict)
                               if initial_state_dict is not None else ""),
        "batch_order_hash": legacy._combined_hash(details["batch_order_hashes"]),
        "batch_order_hashes": json.dumps(details["batch_order_hashes"]),
        "lam_kd": params["lam_kd"], "temperature": params["temperature"],
        "distill_warmup_epochs": params['distill_warmup_epochs'],
        "lam_feat": params['lam_feat'], "lam_mi": 0.0, "lam_mmd": 0.0,
        "feature_metric": "cosine" if params['lam_feat'] > 0 else "",
        "feature_transform": "native_penultimate" if params['lam_feat'] > 0 else "",
        "projector_state_hash": (legacy._sha256_state_dict(details['projector'].state_dict())
                                 if details['projector'] is not None else ""),
        "weight_mode": weight_mode, "student_epochs": cfg["epochs"],
        "student_lr": cfg["lr"], "student_weight_decay": cfg["weight_decay"],
        "student_batch_size": cfg["batch_size"], "optimizer": "adamw",
        "scheduler": "CosineAnnealingLR",
        "preprocessing": json.dumps({"use_filter_bank": bool(cfg.get("use_filter_bank", False)),
                                     "in_channels": X_tr.shape[1], "samples": X_tr.shape[2]}, sort_keys=True),
        "final_train_loss": last["total_loss"], "final_train_accuracy": last["train_accuracy"],
        "final_ce_loss": last["ce_loss"], "final_mi_loss": None,
        "final_feature_loss": last['feature_loss'],
        "full_train_mi": diagnostics.get("full_train_mi", ""),
        "full_train_mi_last": diagnostics.get("full_train_mi", ""),
        "full_train_mi_history": "",
        "teacher_student_prediction_agreement": diagnostics.get("teacher_student_prediction_agreement", ""),
        "mean_abs_probability_diff": diagnostics.get("mean_abs_probability_diff", ""),
        "predicted_class_counts": json.dumps(np.bincount(predictions, minlength=nc).tolist()),
        "collapse_flag": bool(len(np.unique(predictions)) < 2),
        "failure_status": "complete" if finite else "invalid_metrics",
        "failure_reason": "" if finite else "non-finite evaluation metric",
    }
    for field, source in {
        "teacher_target_manifest_path": "_manifest_path",
        "teacher_target_manifest_sha256": "_manifest_sha256",
        "source_manifest_sha256": "_source_manifest_sha256",
        "student_input_manifest_sha256": "_student_input_manifest_sha256",
        "student_input_policy_sha256": "_student_input_policy_sha256",
        "teacher_input_manifest_sha256": "_input_manifest_sha256",
        "teacher_input_policy_sha256": "_input_policy_sha256",
    }.items():
        row[field] = tch.get(source, "")
    return (row, details) if return_details else row
