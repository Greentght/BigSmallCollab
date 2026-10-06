"""Offline KD + feature-align: train a small student against a *cached, frozen*
teacher (any big model).

Ports the combined loss from MIRepNet's ``run_align_combo.train_student``:

    L = CE(s, y)
      + lam_kd   * T^2 * KL( log_softmax(s/T) || softmax(t/T) )      # dark knowledge
      + lam_feat * ( 1 - cos( proj(f_s), f_t ) ).mean()              # penultimate align
      + lam_mmd  * MMD( proj(f_s), f_t )                             # feature distribution align

The optional CE+MI pilot uses only class probabilities (no teacher features):
``CE + lam_mi * probability_mi_loss(softmax(t), softmax(s))``. Its joint
distribution is ``(C, C)`` and is estimated independently for each mini-batch.

The teacher's train-split ``logits`` and ``feats`` are read from a cached artifact
(exported once, in the teacher's own env), so the big model is never loaded here —
this is what lets a CBraMod/LaBraM teacher distill into a student that lives in a
different conda env. ``student_adapter`` provides ``preprocess`` + ``forward`` ->
``(feat_s, logits)``; a ``Linear`` is created only when a feature-alignment term
is enabled.
"""
import hashlib

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


def probability_mi_loss(teacher_prob, student_prob, eps=1e-8):
    """Negative mutual information of teacher/student class probabilities.

    The joint distribution is over the *class* axes, hence its shape is
    ``(C, C)`` rather than ``(B, B)``.  Teacher probabilities are always
    treated as constants; the student probabilities retain their gradient.
    This is intentionally the exact batch-wise estimator used by the pilot
    CE+MI condition (raw-logit softmax, temperature one).
    """
    if teacher_prob.ndim != 2 or student_prob.ndim != 2:
        raise ValueError("teacher_prob and student_prob must be 2-D (B,C)")
    if teacher_prob.shape != student_prob.shape:
        raise ValueError(
            f"teacher/student probability shapes differ: "
            f"{tuple(teacher_prob.shape)} vs {tuple(student_prob.shape)}")
    if teacher_prob.shape[0] < 1 or teacher_prob.shape[1] < 2:
        raise ValueError("probability tensors require B>=1 and C>=2")
    if not np.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and > 0")

    teacher_prob = teacher_prob.detach()
    batch_size = teacher_prob.shape[0]
    joint = teacher_prob.T @ student_prob / batch_size
    joint = joint / joint.sum()

    teacher_marginal = joint.sum(dim=1, keepdim=True)
    student_marginal = joint.sum(dim=0, keepdim=True)

    mi = (
        joint
        * torch.log(
            (joint + eps)
            / (teacher_marginal * student_marginal + eps)
        )
    ).sum()
    return -mi


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


def _rbf_kernel(x, y, sigmas):
    """Multi-scale RBF kernel matrix."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    dist2 = (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)
    k = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * float(sigma) * float(sigma))
        k = k + torch.exp(-gamma * dist2)
    return k / len(sigmas)


def _mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0, 4.0),
             normalize=True, labels=None, num_classes=None):
    """Biased RBF MMD^2. Optional class-conditional mode averages per-class MMD."""
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
    if labels is not None:
        vals = []
        for c in range(int(num_classes)):
            mask = labels == c
            if mask.any():
                vals.append(_mmd_rbf(
                    x[mask], y[mask], sigmas=sigmas,
                    normalize=False, labels=None, num_classes=None))
        return torch.stack(vals).mean() if vals else x.sum() * 0.0
    kxx = _rbf_kernel(x, x, sigmas).mean()
    kyy = _rbf_kernel(y, y, sigmas).mean()
    kxy = _rbf_kernel(x, y, sigmas).mean()
    return kxx + kyy - 2.0 * kxy


def distill_student(student_adapter, num_classes, X_tr, y_tr, feat_t, log_t,
                    X_te, lam_kd=0.5, lam_feat=0.5, temperature=2.0,
                    epochs=50, lr=1e-3, weight_decay=0.01, batch_size=16,
                    teacher_correct_only=True, sample_weight=None,
                    dkd=False, w_target=None, w_nontarget=None,
                    dkd_alpha=1.0, dkd_beta=1.0,
                    feat_proto=False, return_train_preds=False,
                    relational=None, balanced_batch=False, seed=None,
                    subject_ids=None, ea_kd=False, ea_temp=3.0,
                    pearson=None, lam_mmd=0.0, mmd_sigmas=(0.5, 1.0, 2.0, 4.0),
                    mmd_normalize=True, mmd_class_conditional=False,
                    probability_mi=False, lam_mi=0.0,
                    return_mi_history=False, initial_state_dict=None,
                    sample_uid=None, return_training_details=False):
    """Train a student via CE (+ optional KD + feature/MMD align). Returns test
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
    probability_mi = bool(probability_mi)
    if not np.isfinite(float(lam_mi)) or float(lam_mi) < 0:
        raise ValueError("lam_mi must be finite and >= 0")
    if probability_mi:
        if lam_kd != 0 or lam_feat != 0 or lam_mmd != 0:
            raise ValueError("CE+MI cannot combine lam_kd, lam_feat or lam_mmd")
        if dkd or ea_kd or relational is not None or pearson is not None or feat_proto:
            raise ValueError("CE+MI cannot combine KD/feature/relational signals")
        if balanced_batch or subject_ids is not None:
            raise ValueError("CE+MI is subject-wise few-shot only; balanced or alternate sampling is forbidden")
        if teacher_correct_only or sample_weight is not None:
            raise ValueError("CE+MI does not use teacher-correct masks or sample weights")
    if seed is not None:
        _set_seed(seed)
    model = student_adapter.build(num_classes)
    if initial_state_dict is not None:
        state = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in initial_state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"initial_state_dict does not match student model: "
                f"missing={list(missing)}, unexpected={list(unexpected)}")

    Xtr = student_adapter.preprocess(X_tr).to(device)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)
    uid_arr = None
    if sample_uid is not None:
        uid_arr = np.asarray(sample_uid, dtype=np.int64)
        if uid_arr.ndim != 2 or uid_arr.shape[1] != 2:
            raise ValueError(
                f"sample_uid must have shape (N, 2), got {uid_arr.shape}")
        if len(uid_arr) != len(ytr):
            raise ValueError("sample_uid and labels have different lengths")
        if len({tuple(row) for row in uid_arr.tolist()}) != len(uid_arr):
            raise ValueError("sample_uid must be unique")
    needs_teacher_features = (
        lam_feat > 0 or lam_mmd > 0 or relational is not None
        or feat_proto)
    # Do not even materialize a teacher feature object for Base, vanilla KD,
    # or CE+MI.  This keeps those paths logits/labels-only when a caller hands
    # in a lazy artifact view rather than the runner's usual ``None``.
    ft = None
    if needs_teacher_features:
        if feat_t is None:
            raise ValueError("the selected feature path requires teacher features")
        ft = torch.as_tensor(np.asarray(feat_t), dtype=torch.float32)
    needs_teacher_logits = (
        probability_mi or lam_kd > 0 or dkd or ea_kd
        or pearson is not None or teacher_correct_only)
    lt = None
    if needs_teacher_logits:
        if log_t is None:
            raise ValueError("the selected KD/reliability path requires teacher logits")
        lt = torch.as_tensor(np.asarray(log_t), dtype=torch.float32)
    if ft is not None and len(ft) != len(ytr):
        raise ValueError("teacher features and labels have different lengths")
    if lt is not None and len(lt) != len(ytr):
        raise ValueError("teacher logits and labels have different lengths")
    if probability_mi and lt is None:
        raise ValueError("CE+MI requires cached teacher logits")

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

    # Create a feature projection only for paths that actually consume teacher
    # features.  Base, vanilla KD and CE+MI must not instantiate or touch one.
    needs_projection = lam_feat > 0 or lam_mmd > 0
    proj = None
    if needs_projection:
        with torch.no_grad():
            f_probe, _ = student_adapter.forward(model, Xtr[:2])
        proj = nn.Linear(f_probe.shape[1], ft.shape[1]).to(device)

    # Keep the tensor tuple shape stable for the legacy branches while using
    # empty placeholders when a method has no teacher feature/logit signal.
    # These placeholders are never read by Base or CE+MI.
    ft_batch = ft if ft is not None else torch.empty((len(ytr), 0), dtype=torch.float32)
    lt_batch = lt if lt is not None else torch.empty((len(ytr), 0), dtype=torch.float32)

    sample_index = torch.arange(len(ytr), dtype=torch.long)
    dataset = TensorDataset(Xtr.cpu(), ytr, ft_batch, lt_batch, w, w2,
                            sample_index)
    if balanced_batch and subject_ids is not None:
        loader = DataLoader(dataset, batch_sampler=ClassSubjectBalancedSampler(
            np.asarray(y_tr), np.asarray(subject_ids), batch_size,
            num_classes, seed=(seed or 0)))
    elif balanced_batch:
        loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(
            np.asarray(y_tr), batch_size, num_classes, seed=(seed or 0)))
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)
    opt_params = list(model.parameters())
    if proj is not None:
        opt_params += list(proj.parameters())
    opt = optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = temperature
    mi_history = []
    training_history = []
    batch_order_hashes = []

    def _full_train_mi():
        """Full-train, no-grad MI monitor; not the stochastic batch loss."""
        model.eval()
        logits_parts = []
        with torch.no_grad():
            for start in range(0, len(Xtr), batch_size):
                _, logits_part = student_adapter.forward(
                    model, Xtr[start:start + batch_size].to(device))
                logits_parts.append(logits_part)
            student_logits = torch.cat(logits_parts, dim=0)
            teacher_prob = F.softmax(lt.to(device), dim=1).detach()
            student_prob = F.softmax(student_logits, dim=1)
            # ``probability_mi_loss`` is negative MI for optimization; expose
            # the positive MI value in the epoch-end monitor/history.
            return float((-probability_mi_loss(teacher_prob, student_prob)).item())

    model.train()
    for epoch_index in range(epochs):
        # The full-train MI monitor switches the model to eval mode; restore
        # training mode before the next epoch so dropout/normalization retain
        # the same semantics as the legacy loop.
        model.train()
        epoch_total = 0.0
        epoch_ce = 0.0
        epoch_mi = 0.0
        epoch_correct = 0
        epoch_count = 0
        order_hasher = hashlib.sha256()
        for xb, yb, fb, lb, wb, w2b, ib in loader:
            batch_indices = ib.detach().cpu().numpy().astype(np.int64, copy=False)
            if uid_arr is None:
                order_hasher.update(batch_indices.tobytes())
            else:
                order_hasher.update(uid_arr[batch_indices].tobytes())
            xb, yb, fb, lb, wb, w2b = (xb.to(device), yb.to(device),
                                       fb.to(device), lb.to(device),
                                       wb.to(device), w2b.to(device))
            feat_s, logits = student_adapter.forward(model, xb)
            ce_loss = F.cross_entropy(logits, yb)
            loss = ce_loss
            mi_loss_value = None

            if probability_mi:
                # Raw-logit (T=1) probabilities; the teacher side is a
                # constant and the MI estimate is batch-wise stochastic.
                teacher_prob = F.softmax(lb, dim=1).detach()
                student_prob = F.softmax(logits, dim=1)
                mi_loss_value = probability_mi_loss(teacher_prob, student_prob)
                loss = loss + lam_mi * mi_loss_value
            elif dkd and lam_kd > 0:
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
            if lam_mmd > 0:
                loss = loss + lam_mmd * _mmd_rbf(
                    proj(feat_s), fb.detach(), sigmas=mmd_sigmas,
                    normalize=mmd_normalize,
                    labels=(yb if mmd_class_conditional else None),
                    num_classes=num_classes)
            opt.zero_grad(); loss.backward(); opt.step()
            batch_count = int(yb.shape[0])
            epoch_count += batch_count
            epoch_total += float(loss.detach().item()) * batch_count
            epoch_ce += float(ce_loss.detach().item()) * batch_count
            if mi_loss_value is not None:
                epoch_mi += float(mi_loss_value.detach().item()) * batch_count
            epoch_correct += int((logits.detach().argmax(dim=1) == yb).sum().item())
        sched.step()
        full_train_mi = None
        if probability_mi:
            full_train_mi = _full_train_mi()
            mi_history.append(full_train_mi)
        batch_order_hashes.append(order_hasher.hexdigest())
        training_history.append({
            'epoch': int(epoch_index + 1),
            'total_loss': epoch_total / max(1, epoch_count),
            'ce_loss': epoch_ce / max(1, epoch_count),
            'mi_loss': (epoch_mi / max(1, epoch_count)
                        if mi_loss_value is not None else None),
            'full_train_mi': full_train_mi,
            'train_accuracy': epoch_correct / max(1, epoch_count) * 100.0,
            'n_train': int(epoch_count),
            'batch_order_hash': batch_order_hashes[-1],
        })

    _, logits_te = student_adapter.infer(model, X_te)
    preds = logits_te.argmax(1)
    if return_training_details:
        _, logits_tr = student_adapter.infer(model, X_tr)
        train_preds = logits_tr.argmax(1)
        details = {
            'model': model,
            'training_history': training_history,
            'batch_order_hashes': batch_order_hashes,
            'train_logits': logits_tr,
            'test_logits': logits_te,
            'train_preds': train_preds,
            'test_preds': preds,
            'full_train_mi_history': mi_history,
        }
        return preds, details
    if return_mi_history:
        return preds, mi_history
    if return_train_preds:
        _, logits_tr = student_adapter.infer(model, X_tr)
        return preds, logits_tr.argmax(1)
    return preds
