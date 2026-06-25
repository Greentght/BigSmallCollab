"""Offline KD + feature-align: train a small student against a *cached, frozen*
teacher (any big model).

Ports the combined loss from MIRepNet's ``run_align_combo.train_student``:

    L = CE(s, y)
      + lam_kd   * T^2 * KL( log_softmax(s/T) || softmax(t/T) )      # dark knowledge
      + lam_feat * ( 1 - cos( proj(f_s), f_t ) ).mean()              # penultimate align

The teacher's train-split ``logits`` and ``feats`` are read from a cached artifact
(exported once, in the teacher's own env), so the big model is never loaded here —
this is what lets a CBraMod/LaBraM teacher distill into a student that lives in a
different conda env. ``student_adapter`` provides ``preprocess`` + ``forward`` ->
``(feat_s, logits)``; a ``Linear`` projects student feat dim to the teacher's.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def distill_student(student_adapter, num_classes, X_tr, y_tr, feat_t, log_t,
                    X_te, lam_kd=0.5, lam_feat=0.5, temperature=2.0,
                    epochs=50, lr=1e-3, weight_decay=0.01, batch_size=16):
    """Train a student via CE (+ optional KD + feature-align). Returns test
    predictions ``(N_te,)`` as a numpy array. Set lam_kd=lam_feat=0 for the
    plain-student baseline (identical training path, no teacher signal)."""
    device = student_adapter.device
    model = student_adapter.build(num_classes)

    Xtr = student_adapter.preprocess(X_tr).to(device)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)
    ft = torch.as_tensor(np.asarray(feat_t), dtype=torch.float32)
    lt = torch.as_tensor(np.asarray(log_t), dtype=torch.float32)

    # student feat dim from a dry-run forward
    with torch.no_grad():
        f_probe, _ = student_adapter.forward(model, Xtr[:2])
    proj = nn.Linear(f_probe.shape[1], ft.shape[1]).to(device)

    loader = DataLoader(TensorDataset(Xtr.cpu(), ytr, ft, lt),
                        batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(list(model.parameters()) + list(proj.parameters()),
                      lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = temperature

    model.train()
    for _ in range(epochs):
        for xb, yb, fb, lb in loader:
            xb, yb, fb, lb = (xb.to(device), yb.to(device),
                              fb.to(device), lb.to(device))
            feat_s, logits = student_adapter.forward(model, xb)
            loss = F.cross_entropy(logits, yb)
            if lam_kd > 0:
                loss = loss + lam_kd * (T * T) * F.kl_div(
                    F.log_softmax(logits / T, dim=1),
                    F.softmax(lb / T, dim=1), reduction='batchmean')
            if lam_feat > 0:
                loss = loss + lam_feat * (
                    1 - F.cosine_similarity(proj(feat_s), fb, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    _, logits_te = student_adapter.infer(model, X_te)
    return logits_te.argmax(1)
