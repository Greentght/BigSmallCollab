"""Sample-level asymmetric bidirectional distillation (LOSO).

Big model B (MIRepNet) and small model S (IFNet) are trained *jointly* (both live
in the mirepnet env). Per batch, routing masks decide who teaches whom, using
stop-grad predictions on the true label:

    m_{B->S} = 1[ y_B == y  and  y_S != y ]   # big right, small wrong -> B teaches S
    m_{S->B} = 1[ y_S == y  and  y_B != y ]   # small right, big wrong -> S teaches B

    L = CE_S + CE_B
      + lam_bs * mean_{m_BS} KL( softmax(B/T) || S )      # forward, routed
      + lam_sb * mean_{m_SB} KL( softmax(S/T) || B )      # reverse, routed

Asymmetric: lam_sb < lam_bs (the small model only nudges the big one on the few
complementary samples). Motivated by LOSO finding: (B wrong, S correct) ~ 15.9%.
Both KD targets are detached; the masks only route, never leak gradient.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .distill import _set_seed, ClassSubjectBalancedSampler


def _routed_kd(student_logits, teacher_logits, mask, T):
    """mean over masked samples of KL(softmax(teacher/T) || student), teacher detached."""
    kd = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                  F.softmax(teacher_logits.detach() / T, dim=1),
                  reduction='none').sum(dim=1)
    s = mask.sum()
    return (mask * kd).sum() / s if s > 0 else student_logits.sum() * 0.0


@torch.no_grad()
def _infer(adapter, model, Xp, bs=64):
    model.eval()
    device = adapter.device
    out = []
    for i in range(0, len(Xp), bs):
        _, lg = adapter.forward(model, Xp[i:i + bs].to(device))
        out.append(lg.cpu())
    return torch.cat(out).argmax(1).numpy()


def bidirectional_distill(big_ad, small_ad, num_classes,
                          X_tr, y_tr, subj_tr, X_te, test_subject,
                          lam_bs=1.0, lam_sb=0.1, temperature=2.0,
                          epochs=100, lr_big=1e-4, lr_small=1e-3,
                          weight_decay=0.01, batch_size=16, seed=666):
    """Joint bidirectional routed distillation. Returns (S_preds, B_preds) on the
    held-out test subject. big_ad must be built with skip_preprocess=True (data is
    EA'd per subject here). lam_sb=0 -> unidirectional (forward routed only)."""
    device = small_ad.device
    _set_seed(seed)
    B = big_ad.build(num_classes)
    S = small_ad.build(num_classes)

    # B: per-subject EA + 45ch pad (then adapter passes through). S: raw.
    Xp_B = torch.as_tensor(big_ad.ea_pad_per_subject(X_tr, subj_tr),
                           dtype=torch.float32)
    Xp_S = small_ad.preprocess(X_tr)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)

    loader = DataLoader(TensorDataset(Xp_B, Xp_S, ytr),
                        batch_sampler=ClassSubjectBalancedSampler(
                            np.asarray(y_tr), np.asarray(subj_tr),
                            batch_size, num_classes, seed=seed))
    opt = optim.AdamW([{'params': B.parameters(), 'lr': lr_big},
                       {'params': S.parameters(), 'lr': lr_small}],
                      weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = temperature

    B.train(); S.train()
    for _ in range(epochs):
        for xbB, xbS, yb in loader:
            xbB, xbS, yb = xbB.to(device), xbS.to(device), yb.to(device)
            _, lB = big_ad.forward(B, xbB)
            _, lS = small_ad.forward(S, xbS)
            with torch.no_grad():
                bc = lB.argmax(1) == yb
                sc = lS.argmax(1) == yb
                m_bs = (bc & ~sc).float()      # big right, small wrong
                m_sb = (sc & ~bc).float()      # small right, big wrong
            loss = F.cross_entropy(lS, yb) + F.cross_entropy(lB, yb)
            if lam_bs > 0:
                loss = loss + lam_bs * (T * T) * _routed_kd(lS, lB, m_bs, T)
            if lam_sb > 0:
                loss = loss + lam_sb * (T * T) * _routed_kd(lB, lS, m_sb, T)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    Xp_B_te = torch.as_tensor(big_ad.ea_pad_per_subject(
        X_te, np.full(len(X_te), test_subject)), dtype=torch.float32)
    Xp_S_te = small_ad.preprocess(X_te)
    return _infer(small_ad, S, Xp_S_te), _infer(big_ad, B, Xp_B_te)
