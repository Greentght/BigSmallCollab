"""Feature-level bidirectional alignment in the K-shot few-shot scenario.

Both LOSO-pretrained models are jointly fine-tuned on K support trials from the
test subject. The feature alignment loss pulls each model's penultimate features
toward the complementary model's features on correctness-routed samples:
  - big correct & small wrong  -> small learns big's feature (via proj_s)
  - small correct & big wrong  -> big learns small's feature (via proj_b)

Methods compared per (cell, subject, K, draw):
  ft_big     : independent CE fine-tune of big  (same as run_finetune_baseline)
  ft_small   : independent CE fine-tune of small
  ft_ens     : ensemble of independently fine-tuned models
  ft_bd_big  : joint CE+feat-bidir fine-tune -> big model
  ft_bd_small: joint CE+feat-bidir fine-tune -> small model
  ft_bd_ens  : ensemble of jointly fine-tuned models

Primary test: ft_bd_small > ft_small  (does small model benefit from big's features?)
Secondary:    ft_bd_big   > ft_big    (symmetric)
              ft_bd_ens   > ft_ens    (joint vs independent fine-tune ensemble)

Stats: subject-level paired Wilcoxon (across draws, averaged per subject).

    conda run -n mirepnet python -u scripts/bidir/run_featbidir_fewshot.py \\
        --cells BNCI2014001-4:mirepnet:ifnet --Ks 20 30 --gpu 2
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit

import config
import data
from models import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(ROOT, 'results', 'metrics')

_SPLIT_CACHE = {}


def cached_loso_split(dataset, fold):
    key = (dataset, fold)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = data.loso_split(dataset, fold)
    return _SPLIT_CACHE[key]


def build_loso_base(model_name, dataset, fold, seed, nc, mcfg, device, base_epochs=None):
    """Train the LOSO base, return (adapter, model, Xp_te [preprocessed], y_te)."""
    X_tr, y_tr, _subj, X_te, y_te = cached_loso_split(dataset, fold)
    cfg = dict(mcfg); cfg.update(in_channels=X_tr.shape[1], samples=X_tr.shape[2],
                                 dataset_name=dataset)
    if base_epochs is not None:
        cfg['epochs'] = base_epochs
    torch.manual_seed(seed); np.random.seed(seed)
    ad = get_adapter(model_name, device=device, **cfg)
    model = ad.build(nc)
    model = ad.finetune(model, X_tr, y_tr, nc)
    # preprocess the FULL held-out test set once so EA is stable
    Xp_te = ad.preprocess(X_te)
    return ad, model, Xp_te, y_te


@torch.no_grad()
def _infer_pp(ad, model, Xp, bs=64):
    """Inference on already-preprocessed tensor Xp; skips re-preprocessing."""
    dev = ad.device
    model.eval()
    feats, logits = [], []
    for i in range(0, len(Xp), bs):
        f, lg = ad.forward(model, Xp[i:i+bs].to(dev))
        feats.append(f.cpu().numpy())
        logits.append(lg.cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def _softmax_np(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _finetune_indep(ad, base_model, Xp_sup, y_sup_t, ft_epochs, ft_lr, wd, seed):
    """Independent CE fine-tune from LOSO base; Xp_sup is already preprocessed."""
    torch.manual_seed(seed)
    dev = ad.device
    model = copy.deepcopy(base_model)
    opt = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)
    Xb = Xp_sup.to(dev)
    yb = y_sup_t.to(dev)
    model.train()
    for _ in range(ft_epochs):
        _, logits = ad.forward(model, Xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def featbidir_joint_finetune(adb, ads, mb, ms,
                              Xpb_sup, Xps_sup, y_sup_t,
                              ft_epochs, ft_lr, wd, lam_feat, seed):
    """Joint CE + correctness-routed feature-bidir fine-tune on K support trials.

    Xpb_sup / Xps_sup are already-preprocessed tensors for big / small.
    Routing is dynamic (uses current model predictions each epoch).
    Projectors proj_s (dS->dB) and proj_b (dB->dS) are throw-away adapters.
    """
    torch.manual_seed(seed)
    dev = adb.device
    B = copy.deepcopy(mb); S = copy.deepcopy(ms)
    B.to(dev); S.to(dev)

    # probe feature dims
    with torch.no_grad():
        _fB, _ = adb.forward(B, Xpb_sup[:2].to(dev))
        _fS, _ = ads.forward(S, Xps_sup[:2].to(dev))
        dB, dS = _fB.shape[1], _fS.shape[1]

    proj_s = torch.nn.Linear(dS, dB).to(dev)
    proj_b = torch.nn.Linear(dB, dS).to(dev)

    opt = optim.AdamW([
        {'params': B.parameters(), 'lr': ft_lr},
        {'params': S.parameters(), 'lr': ft_lr},
        {'params': list(proj_s.parameters()) + list(proj_b.parameters()), 'lr': ft_lr},
    ], weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)

    Xb = Xpb_sup.to(dev); Xs = Xps_sup.to(dev); yb = y_sup_t.to(dev)
    Bsz = float(len(yb))

    for _ in range(ft_epochs):
        B.train(); S.train()
        fB, lB = adb.forward(B, Xb)
        fS, lS = ads.forward(S, Xs)

        with torch.no_grad():
            bc = lB.argmax(1) == yb; sc = lS.argmax(1) == yb
            m_bs = (bc & ~sc).float()   # big right, small wrong -> small learns B feat
            m_sb = (sc & ~bc).float()   # small right, big wrong -> big learns S feat

        loss = F.cross_entropy(lB, yb) + F.cross_entropy(lS, yb)

        if m_bs.sum() > 0:
            pS = F.normalize(proj_s(fS), dim=1)
            fBd = F.normalize(fB.detach(), dim=1)
            loss = loss + lam_feat * (m_bs * (1.0 - (pS * fBd).sum(dim=1))).sum() / Bsz

        if m_sb.sum() > 0:
            pB = F.normalize(proj_b(fB), dim=1)
            fSd = F.normalize(fS.detach(), dim=1)
            loss = loss + lam_feat * (m_sb * (1.0 - (pB * fSd).sum(dim=1))).sum() / Bsz

        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    return B, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='BNCI2014001-4:mirepnet:ifnet',
                    help='comma of dataset:big:small')
    ap.add_argument('--Ks', type=int, nargs='+', default=[20, 30])
    ap.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--subjects', type=int, nargs='+', default=None)
    ap.add_argument('--base_epochs', type=int, default=None)
    ap.add_argument('--ft_epochs', type=int, default=30)
    ap.add_argument('--ft_lr', type=float, default=5e-4)
    ap.add_argument('--lam_feat', type=float, default=0.5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    cells = [tuple(c.split(':')) for c in a.cells.split(',')]
    os.makedirs(METRICS, exist_ok=True)
    tag = a.tag or 'v1'
    out_csv = os.path.join(METRICS, f'featbidir_fewshot_{tag}.csv')

    import pandas as pd
    rows = []

    for ds, big, small in cells:
        dcfg = config.load_dataset_config(ds)
        nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']
        bcfg, scfg = config.load_model_config(big), config.load_model_config(small)
        subjects = a.subjects if a.subjects is not None else list(range(n_sub))

        for seed in a.seeds:
            for t in subjects:
                adb, mb, Xpb_te, y = build_loso_base(big,   ds, t, seed, nc, bcfg, device, a.base_epochs)
                ads_, ms, Xps_te, _ = build_loso_base(small, ds, t, seed, nc, scfg, device, a.base_epochs)

                if np.bincount(y, minlength=nc).min() < 2:
                    continue

                for K in a.Ks:
                    if len(y) <= K + nc:
                        continue
                    for di, (tr, te) in enumerate(StratifiedShuffleSplit(
                            n_splits=a.draws, train_size=K, random_state=seed).split(np.zeros(len(y)), y)):
                        yt = y[te]
                        Xpb_sup = Xpb_te[tr]; Xps_sup = Xps_te[tr]
                        y_sup_t = torch.as_tensor(y[tr], dtype=torch.long)

                        # independent CE fine-tune
                        mb_i = _finetune_indep(adb, mb, Xpb_sup, y_sup_t,
                                               a.ft_epochs, a.ft_lr, a.wd, seed)
                        ms_i = _finetune_indep(ads_, ms, Xps_sup, y_sup_t,
                                               a.ft_epochs, a.ft_lr, a.wd, seed)
                        _, lb_i = _infer_pp(adb, mb_i, Xpb_te[te])
                        _, ls_i = _infer_pp(ads_, ms_i, Xps_te[te])
                        pb_i = _softmax_np(lb_i); ps_i = _softmax_np(ls_i)
                        ft_big   = (pb_i.argmax(1) == yt).mean() * 100
                        ft_small = (ps_i.argmax(1) == yt).mean() * 100
                        ft_ens   = ((pb_i + ps_i).argmax(1) == yt).mean() * 100

                        # joint feat-bidir fine-tune
                        mb_bd, ms_bd = featbidir_joint_finetune(
                            adb, ads_, mb, ms, Xpb_sup, Xps_sup, y_sup_t,
                            a.ft_epochs, a.ft_lr, a.wd, a.lam_feat, seed)
                        _, lb_bd = _infer_pp(adb, mb_bd, Xpb_te[te])
                        _, ls_bd = _infer_pp(ads_, ms_bd, Xps_te[te])
                        pb_bd = _softmax_np(lb_bd); ps_bd = _softmax_np(ls_bd)
                        ft_bd_big   = (pb_bd.argmax(1) == yt).mean() * 100
                        ft_bd_small = (ps_bd.argmax(1) == yt).mean() * 100
                        ft_bd_ens   = ((pb_bd + ps_bd).argmax(1) == yt).mean() * 100

                        rows.append(dict(
                            dataset=ds, big=big, small=small, seed=seed,
                            subject=t, K=K, draw=di,
                            ft_big=round(ft_big, 2),
                            ft_small=round(ft_small, 2),
                            ft_ens=round(ft_ens, 2),
                            ft_bd_big=round(ft_bd_big, 2),
                            ft_bd_small=round(ft_bd_small, 2),
                            ft_bd_ens=round(ft_bd_ens, 2),
                        ))

                    print(f'[{ds} {big}x{small}] seed{seed} S{t} K{K} done', flush=True)

                del mb, ms, mb_i, ms_i, mb_bd, ms_bd
                if device != 'cpu':
                    torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    from scipy.stats import wilcoxon
    print('\n=== feat-bidir vs independent fine-tune (subject-level Wilcoxon) ===')
    for (ds, big, small, K), g in df.groupby(['dataset', 'big', 'small', 'K']):
        gs = g.groupby('subject')[['ft_big', 'ft_small', 'ft_ens',
                                    'ft_bd_big', 'ft_bd_small', 'ft_bd_ens']].mean()
        print(f'\n  {ds} {big}x{small} K={K}  (n={len(gs)} subjects)')
        for col_new, col_base in [('ft_bd_big',   'ft_big'),
                                   ('ft_bd_small', 'ft_small'),
                                   ('ft_bd_ens',   'ft_ens')]:
            delta = gs[col_new] - gs[col_base]
            wins = int((delta > 0).sum())
            try:
                _, p = wilcoxon(gs[col_new], gs[col_base])
            except ValueError:
                p = 1.0
            print(f'    {col_new} vs {col_base}: mean Δ={delta.mean():+.2f}  '
                  f'wins={wins}/{len(gs)}  Wilcoxon p={p:.3f}')


if __name__ == '__main__':
    main()
