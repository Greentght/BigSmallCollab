"""CR-AMD: Complementarity-Routed Asymmetric Mutual Distillation (LOSO).

Shared warm-up (both models CE only) -> snapshot -> fork into groups, so every
group starts from an identical warm-up checkpoint. Distillation terms are
normalized by BATCH SIZE (not mask count) and ramped up over the first `ramp`
collaboration epochs. Per-epoch TRAIN prediction-state proportions are logged
(r_BS/r_SB/r_BB/r_WW) to see whether the complementary (B wrong, S correct) pool
survives as the models fit the source subjects.

Groups (min set G0/G1/G3/G5/G6/G7):
  G0_CE      : both update, no distill               (training-length control)
  G1_FixKD   : big frozen, all-sample B->S           (traditional offline KD)
  G3_SymDML  : both update, all-sample both dirs (eq) (symmetric mutual learning)
  G5_Routed  : both update, routed B->S only         (reliability one-way)
  G6_CRAMD   : both update, routed B->S + routed S->B (weak) -- core method
  G7_DisagCE : both update, S's CE up-weighted on the routed-B->S samples, NO KL
               (ablation: does the gain come merely from re-weighting the hard,
               teacher-can-fix samples, rather than from the teacher's soft
               probabilities?)  L_S = mean_i (1 + alpha*m_{B->S,i}) * CE_i .

Routing masks (stop-grad, on the SOURCE-TRAIN true labels -> no test leakage):
  m_{B->S} = 1[y_B==y & y_S!=y];  m_{S->B} = 1[y_S==y & y_B!=y].
For 2 classes this correctness routing is IDENTICAL to the user's margin-sign
routing m_{B->S}=1[M_B>0]1[M_S<0] with M = (2y-1)*(z_1 - z_0): argmax==y <=> the
true-class logit is the larger one <=> M>0. So the strict-disagreement router
here IS the 2-class margin router.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .distill import _set_seed, ClassSubjectBalancedSampler

GROUPS = {
    'G0_CE':     dict(update_big=True,  fwd='none',   rev='none', feat='none'),
    'G1_FixKD':  dict(update_big=False, fwd='all',    rev='none', feat='none'),
    'G3_SymDML': dict(update_big=True,  fwd='all',    rev='all',  feat='none'),
    'G5_Routed': dict(update_big=True,  fwd='routed', rev='none', feat='none'),
    'G6_CRAMD':  dict(update_big=True,  fwd='routed', rev='routed', feat='none'),
    'G7_DisagCE': dict(update_big=True, fwd='disag_ce', rev='none', feat='none'),
    # --- feature-level bidirectional alignment (D0-motivated) ---
    # routed feat align preserves complementarity (only pull on one-correct samples);
    # feat='all' is the homogenization control; G8k adds logit KD on top.
    'G8_FeatBiDir': dict(update_big=True, fwd='none',   rev='none',   feat='routed'),
    'G8k_FeatKD':   dict(update_big=True, fwd='routed', rev='routed', feat='routed'),
    'G9_FeatAll':   dict(update_big=True, fwd='none',   rev='none',   feat='all'),
}


def _kd(student_logits, teacher_logits, T):
    return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits.detach() / T, dim=1),
                    reduction='none').sum(dim=1)


def _feat_align(proj, f_src, f_tgt):
    """Routed feature alignment loss (per-sample, un-reduced): 1 - cos(proj(f_src),
    f_tgt.detach()). proj maps f_src's dim to f_tgt's dim. Pulls the source model's
    feature toward the (complementary) target model's feature — the target is
    stop-grad, so only the source model + projector move toward it."""
    ps = proj(f_src)
    ps = F.normalize(ps, dim=1)
    ft = F.normalize(f_tgt.detach(), dim=1)
    return 1.0 - (ps * ft).sum(dim=1)


@torch.no_grad()
def _infer(ad, model, Xp, bs=64):
    model.eval()
    out = []
    for i in range(0, len(Xp), bs):
        _, lg = ad.forward(model, Xp[i:i + bs].to(ad.device))
        out.append(lg.cpu())
    return torch.cat(out).argmax(1).numpy()


def cr_amd_fold(big_ad, small_ad, nc, X_tr, y_tr, subj_tr, X_te, test_subj,
                groups=None, warmup=20, total=100, ramp=10,
                lr_big=1e-4, lr_small=1e-3, wd=0.01, bs=16, T=2.0,
                lam_bs=0.5, lam_sb=0.1, alpha_ce=1.0, lam_feat=0.5,
                lr_proj=1e-3, seed=666):
    """Run warm-up + all groups for one LOSO fold. Returns
    (results{group->(S_pred,B_pred)}, diag[list of per-epoch train-quadrant dicts])."""
    groups = groups or GROUPS
    dev = small_ad.device
    _set_seed(seed)
    B = big_ad.build(nc)
    S = small_ad.build(nc)
    Xp_B = torch.as_tensor(big_ad.ea_pad_per_subject(X_tr, subj_tr), dtype=torch.float32)
    Xp_S = small_ad.preprocess(X_tr)
    ytr = torch.as_tensor(y_tr, dtype=torch.long)
    ds = TensorDataset(Xp_B, Xp_S, ytr)

    def loader():
        return DataLoader(ds, batch_sampler=ClassSubjectBalancedSampler(
            np.asarray(y_tr), np.asarray(subj_tr), bs, nc, seed=seed))

    # ---- shared warm-up (CE only) ----
    optW = optim.AdamW([{'params': B.parameters(), 'lr': lr_big},
                        {'params': S.parameters(), 'lr': lr_small}], weight_decay=wd)
    B.train(); S.train()
    for _ in range(warmup):
        for xbB, xbS, yb in loader():
            xbB, xbS, yb = xbB.to(dev), xbS.to(dev), yb.to(dev)
            _, lB = big_ad.forward(B, xbB)
            _, lS = small_ad.forward(S, xbS)
            loss = F.cross_entropy(lB, yb) + F.cross_entropy(lS, yb)
            optW.zero_grad(); loss.backward(); optW.step()
    snapB, snapS = copy.deepcopy(B.state_dict()), copy.deepcopy(S.state_dict())

    # penultimate-feature dims (one dummy forward) -> projector shapes
    with torch.no_grad():
        _xb, _xs, _ = next(iter(loader()))
        _fB, _ = big_ad.forward(B, _xb.to(dev))
        _fS, _ = small_ad.forward(S, _xs.to(dev))
        dB, dS = _fB.shape[1], _fS.shape[1]

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
        proj_s = proj_b = None
        if gc.get('feat', 'none') != 'none':
            proj_s = torch.nn.Linear(dS, dB).to(dev)   # S feat -> B dim (S learns from B)
            proj_b = torch.nn.Linear(dB, dS).to(dev)   # B feat -> S dim (B learns from S)
            pg.append({'params': list(proj_s.parameters()) + list(proj_b.parameters()),
                       'lr': lr_proj})
        opt = optim.AdamW(pg, weight_decay=wd)
        for ep in range(collab):
            g = min(1.0, (ep + 1) / max(1, ramp))
            S.train(); B.train() if gc['update_big'] else B.eval()
            nBS = nSB = nBB = nWW = ntot = 0
            for xbB, xbS, yb in loader():
                xbB, xbS, yb = xbB.to(dev), xbS.to(dev), yb.to(dev)
                if gc['update_big']:
                    fB, lB = big_ad.forward(B, xbB)
                else:
                    with torch.no_grad():
                        fB, lB = big_ad.forward(B, xbB)
                fS, lS = small_ad.forward(S, xbS)
                with torch.no_grad():
                    bc = lB.argmax(1) == yb; sc = lS.argmax(1) == yb
                    m_bs = (bc & ~sc).float(); m_sb = (sc & ~bc).float()
                    nBS += int((bc & ~sc).sum()); nSB += int((sc & ~bc).sum())
                    nBB += int((bc & sc).sum()); nWW += int((~bc & ~sc).sum())
                    ntot += len(yb)
                Bsz = lS.shape[0]
                # S's CE: plain mean, or (G7) up-weight the routed-B->S samples
                ce_s = F.cross_entropy(lS, yb, reduction='none')
                if gc['fwd'] == 'disag_ce':
                    loss = ((1.0 + alpha_ce * m_bs) * ce_s).mean()
                else:
                    loss = ce_s.mean()
                if gc['update_big']:
                    loss = loss + F.cross_entropy(lB, yb)
                if gc['fwd'] in ('all', 'routed'):           # B -> S (updates S)
                    mf = torch.ones_like(m_bs) if gc['fwd'] == 'all' else m_bs
                    loss = loss + lam_bs * g * (T * T) * (mf * _kd(lS, lB, T)).sum() / Bsz
                if gc['rev'] != 'none' and gc['update_big']: # S -> B (updates B)
                    mr = torch.ones_like(m_sb) if gc['rev'] == 'all' else m_sb
                    lam_r = lam_bs if gc['rev'] == 'all' else lam_sb
                    loss = loss + lam_r * g * (T * T) * (mr * _kd(lB, lS, T)).sum() / Bsz
                if gc.get('feat', 'none') != 'none':         # feature-level bidir align
                    mf = torch.ones_like(m_bs) if gc['feat'] == 'all' else m_bs
                    mr = torch.ones_like(m_sb) if gc['feat'] == 'all' else m_sb
                    # S learns B's feature where B is right & S wrong (updates S+proj_s)
                    loss = loss + lam_feat * g * (mf * _feat_align(proj_s, fS, fB)).sum() / Bsz
                    if gc['update_big']:                     # B learns S's feature where S right & B wrong
                        loss = loss + lam_feat * g * (mr * _feat_align(proj_b, fB, fS)).sum() / Bsz
                opt.zero_grad(); loss.backward(); opt.step()
            diag.append(dict(group=gname, epoch=ep, rBS=nBS / ntot, rSB=nSB / ntot,
                             rBB=nBB / ntot, rWW=nWW / ntot,
                             n_fwd=nBS, n_rev=nSB))
        results[gname] = (_infer(small_ad, S, Xp_S_te),
                          _infer(big_ad, B, Xp_B_te))
    return results, diag
