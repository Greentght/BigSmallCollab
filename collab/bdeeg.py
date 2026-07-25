"""BD-EEG: difference-aware, direction-asymmetric bidirectional distillation for
cross-subject (LOSO) EEG, adapted from Kweon et al. (2021) "Bidirectional
Distillation for Top-K Recommender System".

Kweon et al. route knowledge by *rank discrepancy* between two co-trained models
and use a wider selection big->small than small->big. Motor-imagery has only
2-4 classes, so class rank is nearly info-free; we replace rank discrepancy with
the **true-class cross-entropy difference**:

    e_B = -log p_B(y),  e_S = -log p_S(y)
    d_{B->S} = e_S - e_B   (>0  => big fits the true class better than small)
    d_{S->B} = e_B - e_S   (>0  => small fits the true class better than big)

Direction-asymmetric knowledge selection (the core of the method):
  * big -> small : BROAD, continuous.   w = 1[y_B==y & d_{B->S}>0] * tanh(d/gamma)
      covers "big right / small wrong" AND "both right but big clearly surer".
  * small -> big : STRICT.              m = 1[y_S==y & y_B!=y], then keep only the
      top-rho% by d_{S->B} (small confidently complementary), with lam_sb < lam_bs.

Training follows the paper: both models share a CE warm-up, every group forks
from the *same* warm-up checkpoint, and routing weights are refreshed once per
epoch on the FULL source-train set (not per-batch), so routing reflects the whole
distribution rather than a single batch. Weights are computed with stop-grad and
only from source-train true labels -> no held-out leakage.

Groups (all fork from the shared warm-up):
  G0_CE        : both update, no distill                 (training-length control)
  G1_FixKD     : big frozen, all-sample B->S             (traditional offline KD)
  SymDML       : both update, all-sample both dirs (eq)  (symmetric mutual learning)
  StrictRouted : both update, binary correctness routing both dirs (B right/S wrong,
                 S right/B wrong)                        (Strict Routed DML)
  BD_EEG       : broad continuous B->S + strict top-rho% S->B   -- CORE
  BD_EEG_Swap  : strict top-rho% B->S + broad continuous S->B   -- swap ablation

Per-direction spec is (mode, lam_key): mode in {all, routed, broad, strict}.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .distill import _set_seed, ClassSubjectBalancedSampler

# each group: update_big, and per-direction (mode, lam_key) or None
GROUPS = {
    'G0_CE':        dict(update_big=True,  bs=None,                 sb=None),
    'G1_FixKD':     dict(update_big=False, bs=('all', 'lam_bs'),    sb=None),
    'SymDML':       dict(update_big=True,  bs=('all', 'lam_bs'),    sb=('all', 'lam_bs')),
    'StrictRouted': dict(update_big=True,  bs=('routed', 'lam_bs'), sb=('routed', 'lam_sb')),
    'BD_EEG':       dict(update_big=True,  bs=('broad', 'lam_bs'),  sb=('strict', 'lam_sb')),
    'BD_EEG_Swap':  dict(update_big=True,  bs=('strict', 'lam_sb'), sb=('broad', 'lam_bs')),
}


def _kd(student_logits, teacher_logits, T):
    """Per-sample KL( softmax(teacher/T) || log_softmax(student/T) ). Teacher is
    the detached source; gradient flows only into `student_logits`."""
    return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits.detach() / T, dim=1),
                    reduction='none').sum(dim=1)


@torch.no_grad()
def _ce_state(ad, model, Xp, y, bs=64):
    """Full-set forward -> (e[i]=-log p(y_i), yhat[i]). Used for per-epoch routing."""
    model.eval()
    es, yh = [], []
    y = torch.as_tensor(y, dtype=torch.long)
    for i in range(0, len(Xp), bs):
        _, lg = ad.forward(model, Xp[i:i + bs].to(ad.device))
        lp = F.log_softmax(lg, dim=1).cpu()
        es.append(-lp.gather(1, y[i:i + bs].view(-1, 1)).squeeze(1))
        yh.append(lg.argmax(1).cpu())
    return torch.cat(es), torch.cat(yh)


@torch.no_grad()
def _infer(ad, model, Xp, bs=64):
    model.eval()
    out = []
    for i in range(0, len(Xp), bs):
        _, lg = ad.forward(model, Xp[i:i + bs].to(ad.device))
        out.append(lg.cpu())
    return torch.cat(out).argmax(1).numpy()


def _dir_weights(mode, direction, eB, eS, yhB, yhS, y, gamma, rho):
    """Per-sample routing weight vector over the FULL train set for one direction.

    direction 'BS' (big->small, updates S): source=big, advantage d = eS - eB.
    direction 'SB' (small->big, updates B): source=small, advantage d = eB - eS.
    """
    if direction == 'BS':
        d = eS - eB
        src_correct = (yhB == y); tgt_wrong = (yhS != y)
    else:
        d = eB - eS
        src_correct = (yhS == y); tgt_wrong = (yhB != y)

    if mode == 'all':
        return torch.ones_like(d)
    if mode == 'routed':                       # binary correctness routing
        return (src_correct & tgt_wrong).float()
    if mode == 'broad':                        # wide, continuous CE-difference weight
        gate = (src_correct & (d > 0)).float()
        return gate * torch.tanh(d.clamp_min(0) / gamma)
    if mode == 'strict':                       # top-rho% of the complementary pool
        m = (src_correct & tgt_wrong)
        w = torch.zeros_like(d)
        idx = torch.nonzero(m, as_tuple=True)[0]
        if len(idx) > 0:
            k = max(1, int(round(rho * len(idx))))
            order = torch.argsort(d[idx], descending=True)[:k]
            w[idx[order]] = 1.0
        return w
    raise ValueError(mode)


def bd_eeg_fold(big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, test_subj,
                groups=None, warmup=20, total=100, ramp=10,
                lr_big=1e-4, lr_small=1e-3, wd=0.01, bs=16, T=2.0,
                lam_bs=1.0, lam_sb=0.25, gamma=1.0, rho=0.5, seed=666):
    """Warm-up + all groups for one LOSO fold. Returns
    (results{group->(S_pred,B_pred)}, diag[list of per-epoch dicts])."""
    groups = groups or GROUPS
    dev = small_ad.device
    lam = {'lam_bs': lam_bs, 'lam_sb': lam_sb}
    _set_seed(seed)
    B = big_ad.build(nc)
    S = small_ad.build(nc)
    Xp_B = torch.as_tensor(big_ad.ea_pad_per_subject(X_tr, subj_tr), dtype=torch.float32)
    Xp_S = small_ad.preprocess(X_tr)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)
    idx_all = torch.arange(len(ytr))
    ds = TensorDataset(Xp_B, Xp_S, ytr, idx_all)

    def loader():
        return DataLoader(ds, batch_sampler=ClassSubjectBalancedSampler(
            np.asarray(y_tr), np.asarray(subj_tr), bs, nc, seed=seed))

    # ---- shared warm-up (CE only) ----
    optW = optim.AdamW([{'params': B.parameters(), 'lr': lr_big},
                        {'params': S.parameters(), 'lr': lr_small}], weight_decay=wd)
    B.train(); S.train()
    for _ in range(warmup):
        for xbB, xbS, yb, _ in loader():
            xbB, xbS, yb = xbB.to(dev), xbS.to(dev), yb.to(dev)
            _, lB = big_ad.forward(B, xbB)
            _, lS = small_ad.forward(S, xbS)
            loss = F.cross_entropy(lB, yb) + F.cross_entropy(lS, yb)
            optW.zero_grad(); loss.backward(); optW.step()
    snapB, snapS = copy.deepcopy(B.state_dict()), copy.deepcopy(S.state_dict())

    results, diag = {}, []
    collab = total - warmup
    Xp_B_te = torch.as_tensor(big_ad.ea_pad_per_subject(
        X_te, np.full(len(X_te), test_subj)), dtype=torch.float32)
    Xp_S_te = small_ad.preprocess(X_te)

    for gname, gc in groups.items():
        B.load_state_dict(snapB); S.load_state_dict(snapS)
        pg = [{'params': S.parameters(), 'lr': lr_small}]
        if gc['update_big']:
            pg.append({'params': B.parameters(), 'lr': lr_big})
        opt = optim.AdamW(pg, weight_decay=wd)
        has_distill = gc['bs'] is not None or gc['sb'] is not None
        for ep in range(collab):
            g = min(1.0, (ep + 1) / max(1, ramp))
            # ---- per-epoch routing refresh on the full source-train set ----
            w_bs = w_sb = None
            if has_distill:
                eB, yhB = _ce_state(big_ad, B, Xp_B, ytr)
                eS, yhS = _ce_state(small_ad, S, Xp_S, ytr)
                if gc['bs'] is not None:
                    w_bs = _dir_weights(gc['bs'][0], 'BS', eB, eS, yhB, yhS,
                                        ytr, gamma, rho).to(dev)
                if gc['sb'] is not None and gc['update_big']:
                    w_sb = _dir_weights(gc['sb'][0], 'SB', eB, eS, yhB, yhS,
                                        ytr, gamma, rho).to(dev)
            S.train(); B.train() if gc['update_big'] else B.eval()
            for xbB, xbS, yb, ib in loader():
                xbB, xbS, yb, ib = xbB.to(dev), xbS.to(dev), yb.to(dev), ib.to(dev)
                if gc['update_big']:
                    _, lB = big_ad.forward(B, xbB)
                else:
                    with torch.no_grad():
                        _, lB = big_ad.forward(B, xbB)
                _, lS = small_ad.forward(S, xbS)
                Bsz = lS.shape[0]
                loss = F.cross_entropy(lS, yb)
                if gc['update_big']:
                    loss = loss + F.cross_entropy(lB, yb)
                if w_bs is not None:                          # B -> S (updates S)
                    loss = loss + lam[gc['bs'][1]] * g * (T * T) * \
                        (w_bs[ib] * _kd(lS, lB, T)).sum() / Bsz
                if w_sb is not None:                          # S -> B (updates B)
                    loss = loss + lam[gc['sb'][1]] * g * (T * T) * \
                        (w_sb[ib] * _kd(lB, lS, T)).sum() / Bsz
                opt.zero_grad(); loss.backward(); opt.step()
            drow = dict(group=gname, epoch=ep)
            if w_bs is not None:
                drow.update(n_bs=int((w_bs > 0).sum()), w_bs_mean=float(w_bs.mean()))
            if w_sb is not None:
                drow.update(n_sb=int((w_sb > 0).sum()), w_sb_mean=float(w_sb.mean()))
            diag.append(drow)
        results[gname] = (_infer(small_ad, S, Xp_S_te),
                          _infer(big_ad, B, Xp_B_te))
    return results, diag
