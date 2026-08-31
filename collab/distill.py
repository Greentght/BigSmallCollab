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

from .seed import set_seed as _set_seed  # back-compat alias (collab.{bdeeg,bidirectional,mutual} import it)


def _sim_matrix(feat):
    """B x B cosine-similarity (Gram) matrix of L2-normalized features."""
    z = F.normalize(feat, dim=1)
    return z @ z.t()


def _entropy_at_T(logits, Tp):
    """Per-sample Shannon entropy of softmax(logits / Tp)."""
    p = F.softmax(logits / Tp, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(1)


def _logit_pearson_dist(s_logits, t_logits, eps=1e-8):
    """Per-sample Pearson-correlation distance between student and teacher
    *logit vectors* (paper's L_inter): for each sample i, correlate the C logits
    of the student against the C logits of the teacher (mean-centred over the
    class dim), and return d_p = 1 - rho_p per sample. The batch mean is taken by
    the caller. Teacher logits are treated as constants (caller detaches).
    NOTE: for C=2 the correlation of two points is always +/-1, so d_p is
    degenerate (in {0, 2}); this term is only informative for C>=3 classes."""
    s = s_logits - s_logits.mean(dim=1, keepdim=True)
    t = t_logits - t_logits.mean(dim=1, keepdim=True)
    num = (s * t).sum(dim=1)
    den = s.norm(dim=1) * t.norm(dim=1) + eps
    return 1.0 - num / den


class BalancedBatchSampler:
    """Yields class-balanced index batches (~batch_size/C per class), so every
    batch has intra-class pairs for relational distillation. Samples with
    replacement within a class if it is smaller than the per-class quota."""

    def __init__(self, labels, batch_size, num_classes, seed=0):
        self.by_class = [np.where(labels == c)[0] for c in range(num_classes)]
        self.per_class = max(1, batch_size // num_classes)
        self.n_batches = max(1, len(labels) // batch_size)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for idx in self.by_class:
                if len(idx) == 0:
                    continue
                rep = len(idx) < self.per_class
                batch += list(self.rng.choice(idx, self.per_class, replace=rep))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class ClassSubjectBalancedSampler:
    """Class-balanced batches whose per-class samples are spread evenly across
    subjects (round-robin over subjects within each class), so no single training
    subject dominates the gradient/prototype in cross-subject (LOSO) training."""

    def __init__(self, labels, subjects, batch_size, num_classes, seed=0):
        labels = np.asarray(labels); subjects = np.asarray(subjects)
        # per class: list of per-subject index arrays
        self.class_subj = []
        for c in range(num_classes):
            subs = [np.where((labels == c) & (subjects == s))[0]
                    for s in np.unique(subjects)]
            self.class_subj.append([a for a in subs if len(a) > 0])
        self.per_class = max(1, batch_size // num_classes)
        self.n_batches = max(1, len(labels) // batch_size)
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for subs in self.class_subj:
                if not subs:
                    continue
                for k in range(self.per_class):        # round-robin over subjects
                    idx = subs[(k + self.rng.randint(len(subs))) % len(subs)]
                    batch.append(int(self.rng.choice(idx)))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def _dkd_terms(s_logits, t_logits, y, T, eps=1e-7):
    """Decoupled-KD split of KL(teacher || student) at temperature T into
    per-sample (TCKD, NCKD). TCKD = binary KL on {target, non-target} mass
    (knowledge about the *ground-truth* class t); NCKD = KL over the C-1
    non-target classes renormalized (dark knowledge = inter-class relations).
    For C=2 there is a single non-target class -> NCKD == 0 by construction.
    Zhao et al., CVPR 2022."""
    ps = F.softmax(s_logits / T, dim=1)
    pt = F.softmax(t_logits / T, dim=1)
    B, C = ps.shape
    idx = y.view(-1, 1)
    pt_t = pt.gather(1, idx).squeeze(1)
    ps_t = ps.gather(1, idx).squeeze(1)
    tckd = (pt_t * torch.log((pt_t + eps) / (ps_t + eps))
            + (1 - pt_t) * torch.log((1 - pt_t + eps) / (1 - ps_t + eps)))
    nt = torch.ones_like(ps, dtype=torch.bool).scatter_(1, idx, False)
    pt_nt = pt[nt].view(B, C - 1) / (1 - pt_t).clamp_min(eps).view(-1, 1)
    ps_nt = ps[nt].view(B, C - 1) / (1 - ps_t).clamp_min(eps).view(-1, 1)
    nckd = (pt_nt * (torch.log(pt_nt + eps) - torch.log(ps_nt + eps))).sum(1)
    return tckd, nckd


def distill_student(student_adapter, num_classes, X_tr, y_tr, feat_t, log_t,
                    X_te, lam_kd=0.5, lam_feat=0.5, temperature=2.0,
                    epochs=50, lr=1e-3, weight_decay=0.01, batch_size=16,
                    teacher_correct_only=True, sample_weight=None,
                    dkd=False, w_target=None, w_nontarget=None,
                    dkd_alpha=1.0, dkd_beta=1.0,
                    feat_proto=False, return_train_preds=False,
                    relational=None, balanced_batch=False, seed=None,
                    subject_ids=None, ea_kd=False, ea_temp=3.0,
                    pearson=None):
    """Train a student via CE (+ optional KD + feature-align). Returns test
    predictions ``(N_te,)`` as a numpy array. Set lam_kd=lam_feat=0 for the
    plain-student baseline (identical training path, no teacher signal).

    The KD and feature-align terms are combined with a *per-train-sample weight*
    ``w_i`` (the CE term always spans all samples): each batch uses a weighted
    mean ``sum_i w_i * loss_i / sum_i w_i``. The weight vector unifies the
    reliability variants:
      - ``sample_weight=None`` + ``teacher_correct_only=True`` -> hard mask
        ``w_i = 1[teacher_argmax==y]`` (teacher-correct-only);
      - ``sample_weight=None`` + ``teacher_correct_only=False`` -> ``w_i = 1``
        (plain / all-sample alignment);
      - explicit ``sample_weight`` (N_tr,) -> continuous weights, e.g. the
        entropy/MC-dropout adaptive weight ``w_i = 1 - H_T(x_i)/log C``.
    A sample the teacher is unreliable on thus contributes little/no alignment
    signal, so the student doesn't chase mistaken soft-labels / feature direction.
    """
    device = student_adapter.device
    if seed is not None:
        _set_seed(seed)
    model = student_adapter.build(num_classes)

    Xtr = student_adapter.preprocess(X_tr).to(device)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)
    ft = torch.as_tensor(np.asarray(feat_t), dtype=torch.float32)
    lt = torch.as_tensor(np.asarray(log_t), dtype=torch.float32)

    if sample_weight is not None:
        w = torch.as_tensor(np.asarray(sample_weight), dtype=torch.float32)
    elif teacher_correct_only:
        w = (lt.argmax(dim=1) == ytr).float()
    else:
        w = torch.ones(len(ytr), dtype=torch.float32)

    # second weight column: DKD non-target gate (unused when dkd=False)
    if dkd:
        wt = (torch.ones(len(ytr)) if w_target is None
              else torch.as_tensor(np.asarray(w_target), dtype=torch.float32))
        wnt = (torch.ones(len(ytr)) if w_nontarget is None
               else torch.as_tensor(np.asarray(w_nontarget), dtype=torch.float32))
        w, w2 = wt, wnt
    else:
        w2 = torch.zeros(len(ytr), dtype=torch.float32)

    # class-prototype target: teacher class means over the whole train split
    # (teacher is frozen/cached -> a static, batch-noise-free prototype; no EMA
    # needed). feat term aligns proj(feat_s) to M[y] instead of the teacher's
    # per-sample feature fb.
    M = None
    if feat_proto:
        M = torch.stack([ft[ytr == c].mean(0) if (ytr == c).any()
                         else torch.zeros(ft.shape[1])
                         for c in range(num_classes)]).to(device)  # (C, D_T)

    # student feat dim from a dry-run forward
    with torch.no_grad():
        f_probe, _ = student_adapter.forward(model, Xtr[:2])
    proj = nn.Linear(f_probe.shape[1], ft.shape[1]).to(device)

    dataset = TensorDataset(Xtr.cpu(), ytr, ft, lt, w, w2)
    if balanced_batch and subject_ids is not None:
        loader = DataLoader(dataset, batch_sampler=ClassSubjectBalancedSampler(
            np.asarray(y_tr), np.asarray(subject_ids), batch_size,
            num_classes, seed=(seed or 0)))
    elif balanced_batch:
        loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(
            np.asarray(y_tr), batch_size, num_classes, seed=(seed or 0)))
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(list(model.parameters()) + list(proj.parameters()),
                      lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = temperature

    model.train()
    for _ in range(epochs):
        for xb, yb, fb, lb, wb, w2b in loader:
            xb, yb, fb, lb, wb, w2b = (xb.to(device), yb.to(device),
                                       fb.to(device), lb.to(device),
                                       wb.to(device), w2b.to(device))
            feat_s, logits = student_adapter.forward(model, xb)
            loss = F.cross_entropy(logits, yb)

            if dkd and lam_kd > 0:
                # per-sample: w_target*alpha*TCKD + w_nontarget*beta*NCKD
                tckd, nckd = _dkd_terms(logits, lb, yb, T)
                dkd_loss = (wb * dkd_alpha * tckd + w2b * dkd_beta * nckd).mean()
                loss = loss + lam_kd * (T * T) * dkd_loss
            elif ea_kd and (lam_kd > 0 or lam_feat > 0):
                # EA-KD (Entropy-based Adaptive KD): per-sample weight
                #   w_i = 1/2 * H_T(x_i) * (1 + H_S(x_i)/logC)   (entropies @ T')
                # teacher entropy = knowledge value; student entropy = current
                # need. wb carries an optional extra gate (e.g. correctness mask
                # -> CorrectMask x EA). Weight is stop-grad (a coefficient). The
                # same weight scales BOTH the KD and (for EA_Combo) the feat term.
                Ht = _entropy_at_T(lb, ea_temp)
                Hs = _entropy_at_T(logits.detach(), ea_temp)
                logC = float(np.log(num_classes))
                w_ea = wb * 0.5 * Ht * (1.0 + Hs / logC)
                denom = w_ea.sum() + 1e-8
                if lam_kd > 0:
                    kd = F.kl_div(F.log_softmax(logits / T, dim=1),
                                  F.softmax(lb / T, dim=1),
                                  reduction='none').sum(dim=1)
                    loss = loss + lam_kd * (T * T) * (w_ea * kd).sum() / denom
                if lam_feat > 0:
                    target = M[yb] if feat_proto else fb
                    fa = 1 - F.cosine_similarity(proj(feat_s), target, dim=1)
                    loss = loss + lam_feat * (w_ea * fa).sum() / denom
            else:
                wsum = wb.sum()
                if wsum > 0:
                    if lam_kd > 0:
                        kd = F.kl_div(F.log_softmax(logits / T, dim=1),
                                      F.softmax(lb / T, dim=1),
                                      reduction='none').sum(dim=1)  # per-sample KL
                        loss = loss + lam_kd * (T * T) * (wb * kd).sum() / wsum
                    if lam_feat > 0:
                        target = M[yb] if feat_proto else fb   # prototype vs per-sample
                        fa = 1 - F.cosine_similarity(proj(feat_s), target, dim=1)
                        loss = loss + lam_feat * (wb * fa).sum() / wsum

            # relational distillation: align batch B x B similarity structure
            # (teacher detached). intra_inter splits same-class vs diff-class
            # pairs and normalizes each by its own pair count.
            if relational is not None:
                As = _sim_matrix(feat_s)
                At = _sim_matrix(fb).detach()
                Bn = As.shape[0]
                offdiag = ~torch.eye(Bn, dtype=torch.bool, device=device)
                diff2 = (As - At) ** 2
                if relational['mode'] == 'sim':
                    loss = loss + relational['lam_sim'] * diff2[offdiag].mean()
                else:  # intra_inter
                    same = (yb.view(-1, 1) == yb.view(1, -1)) & offdiag
                    diff = (yb.view(-1, 1) != yb.view(1, -1))
                    if same.any():
                        loss = loss + relational['lam_intra'] * diff2[same].mean()
                    if diff.any():
                        loss = loss + relational['lam_inter'] * diff2[diff].mean()

            # Pearson logit-distance regularizer (paper's L_inter): pull the
            # student's per-sample logit vector into correlation with the
            # teacher's. Teacher detached. 'masked' -> weight by wb (e.g.
            # teacher-correct), else plain batch mean as in the paper.
            if pearson is not None:
                dp = _logit_pearson_dist(logits, lb.detach())
                if pearson.get('masked'):
                    wsum = wb.sum()
                    if wsum > 0:
                        loss = loss + pearson['lam'] * (wb * dp).sum() / wsum
                else:
                    loss = loss + pearson['lam'] * dp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    _, logits_te = student_adapter.infer(model, X_te)
    if return_train_preds:
        _, logits_tr = student_adapter.infer(model, X_tr)
        return logits_te.argmax(1), logits_tr.argmax(1)
    return logits_te.argmax(1)
