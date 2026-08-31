"""Feature-level bidirectional alignment — within-subject scenario.

Both big and small models are trained from pretrained weights on the subject's
own training data. Two split ratios are tested:
  val_split=0.3 : 70% train / 30% test  (standard)
  val_split=0.7 : 30% train / 70% test  (low-resource)

Methods per (subject, split, seed):
  ce_big    : CE-only big model   (each model's own config epochs/lr)
  ce_small  : CE-only small model
  bd_big    : joint CE+feat-bidir big model
  bd_small  : joint CE+feat-bidir small model
  bd_ens    : ensemble of jointly trained models

Correctness routing is dynamic (uses current batch predictions):
  m_bs = big correct & small wrong  -> small model aligns to big's feature
  m_sb = small correct & big wrong  -> big model aligns to small's feature

Primary test: bd_small > ce_small  (subject-level paired Wilcoxon over seeds+splits)

    conda run -n mirepnet python -u experiments/bidir/run_featbidir_within.py \\
        --datasets BNCI2014001-4 --big mirepnet --small ifnet --gpu 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
import data
from models import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = os.path.join(ROOT, 'results', 'metrics')


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval_acc(ad, model, Xp_te, y_te, bs=64):
    dev = ad.device
    model.eval()
    preds = []
    for i in range(0, len(Xp_te), bs):
        _, lg = ad.forward(model, Xp_te[i:i+bs].to(dev))
        preds.append(lg.argmax(1).cpu().numpy())
    return (np.concatenate(preds) == y_te).mean() * 100


def _train_ce(ad, model, Xp_tr, ytr_t, epochs, lr, wd, bs):
    """CE-only training on preprocessed tensors; returns trained model."""
    dev = ad.device
    loader = DataLoader(TensorDataset(Xp_tr, ytr_t), batch_size=bs, shuffle=True,
                        drop_last=False)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            _, lg = ad.forward(model, xb)
            loss = F.cross_entropy(lg, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def _joint_bidir(adb, ads, B, S, Xpb_tr, Xps_tr, ytr_t,
                 epochs, lr_big, lr_small, wd_big, wd_small, bs, lam_feat):
    """Joint CE + correctness-routed feature-bidir training on preprocessed data."""
    dev = adb.device

    # probe feature dims once
    with torch.no_grad():
        _fB, _ = adb.forward(B, Xpb_tr[:2].to(dev))
        _fS, _ = ads.forward(S, Xps_tr[:2].to(dev))
        dB, dS = _fB.shape[1], _fS.shape[1]

    proj_s = torch.nn.Linear(dS, dB).to(dev)   # S -> B dim (S learns from B)
    proj_b = torch.nn.Linear(dB, dS).to(dev)   # B -> S dim (B learns from S)

    loader = DataLoader(TensorDataset(Xpb_tr, Xps_tr, ytr_t),
                        batch_size=bs, shuffle=True, drop_last=False)
    opt = optim.AdamW([
        {'params': B.parameters(), 'lr': lr_big,   'weight_decay': wd_big},
        {'params': S.parameters(), 'lr': lr_small, 'weight_decay': wd_small},
        {'params': list(proj_s.parameters()) + list(proj_b.parameters()),
         'lr': lr_small, 'weight_decay': wd_small},
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    B.train(); S.train()
    for _ in range(epochs):
        for xbB, xbS, yb in loader:
            xbB, xbS, yb = xbB.to(dev), xbS.to(dev), yb.to(dev)
            fB, lB = adb.forward(B, xbB)
            fS, lS = ads.forward(S, xbS)

            with torch.no_grad():
                bc = lB.argmax(1) == yb; sc = lS.argmax(1) == yb
                m_bs = (bc & ~sc).float()   # big right, small wrong
                m_sb = (sc & ~bc).float()   # small right, big wrong

            Bsz = float(len(yb))
            loss = F.cross_entropy(lB, yb) + F.cross_entropy(lS, yb)

            if m_bs.sum() > 0:
                pS = F.normalize(proj_s(fS), dim=1)
                fBd = F.normalize(fB.detach(), dim=1)
                loss = loss + lam_feat * (m_bs * (1.0 - (pS * fBd).sum(1))).sum() / Bsz

            if m_sb.sum() > 0:
                pB = F.normalize(proj_b(fB), dim=1)
                fSd = F.normalize(fS.detach(), dim=1)
                loss = loss + lam_feat * (m_sb * (1.0 - (pB * fSd).sum(1))).sum() / Bsz

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    return B, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['BNCI2014001-4'])
    ap.add_argument('--big',   default='mirepnet')
    ap.add_argument('--small', default='ifnet')
    ap.add_argument('--val_splits', type=float, nargs='+', default=[0.3, 0.7],
                    help='test fraction(s): 0.3=70/30, 0.7=30/70')
    ap.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    ap.add_argument('--subjects', type=int, nargs='+', default=None)
    ap.add_argument('--lam_feat', type=float, default=0.5)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--tag', default='v1')
    a = ap.parse_args()

    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    os.makedirs(METRICS, exist_ok=True)
    out_csv = os.path.join(METRICS, f'featbidir_within_{a.tag}.csv')

    import pandas as pd
    rows = []

    for ds in a.datasets:
        dcfg = config.load_dataset_config(ds)
        nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']
        bcfg = config.load_model_config(a.big)
        scfg = config.load_model_config(a.small)
        subjects = a.subjects if a.subjects is not None else list(range(n_sub))

        # model hyperparams
        epochs_big   = bcfg.get('epochs', 10)
        epochs_small = scfg.get('epochs', 100)
        epochs_joint = max(epochs_big, epochs_small)   # both train for the longer schedule
        lr_big,   wd_big   = bcfg.get('lr', 1e-3), bcfg.get('weight_decay', 1e-6)
        lr_small, wd_small = scfg.get('lr', 1e-3), scfg.get('weight_decay', 1e-2)
        bs_big, bs_small   = bcfg.get('batch_size', 8), scfg.get('batch_size', 16)
        # NOTE: ce_big uses epochs_big (model's natural setting).
        # ce_big_long uses epochs_joint (same as joint training) to isolate the
        # epoch-count effect from the feature-alignment effect for big model.
        bs_joint = min(bs_big, bs_small)   # use the smaller batch size

        for val_split in a.val_splits:
            split_label = f'{round((1-val_split)*100)}tr_{round(val_split*100)}te'
            for seed in a.seeds:
                _set_seed(seed)
                for subj in subjects:
                    X_tr, y_tr, X_te, y_te = data.subject_split(ds, subj, val_split, seed)
                    if np.bincount(y_tr, minlength=nc).min() < 2:
                        continue

                    # build adapters (config needs in_channels/samples/dataset_name)
                    bcfg2 = dict(bcfg); bcfg2.update(in_channels=X_tr.shape[1],
                                                      samples=X_tr.shape[2],
                                                      dataset_name=ds)
                    scfg2 = dict(scfg); scfg2.update(in_channels=X_tr.shape[1],
                                                      samples=X_tr.shape[2],
                                                      dataset_name=ds)
                    adb = get_adapter(a.big,   device=device, **bcfg2)
                    ads = get_adapter(a.small,  device=device, **scfg2)

                    # preprocess once per split (EA is per-set; compute on train/test separately)
                    Xpb_tr = adb.preprocess(X_tr); Xpb_te = adb.preprocess(X_te)
                    Xps_tr = ads.preprocess(X_tr); Xps_te = ads.preprocess(X_te)
                    ytr_t  = torch.as_tensor(y_tr, dtype=torch.long)

                    # ---- CE-only baselines ----
                    _set_seed(seed)
                    B_ce = adb.build(nc)
                    _train_ce(adb, B_ce, Xpb_tr, ytr_t, epochs_big, lr_big, wd_big, bs_big)
                    ce_big = _eval_acc(adb, B_ce, Xpb_te, y_te)

                    # big model trained for epochs_joint (same as joint) — isolates epoch effect
                    _set_seed(seed)
                    B_ce_long = adb.build(nc)
                    _train_ce(adb, B_ce_long, Xpb_tr, ytr_t, epochs_joint, lr_big, wd_big, bs_big)
                    ce_big_long = _eval_acc(adb, B_ce_long, Xpb_te, y_te)

                    _set_seed(seed)
                    S_ce = ads.build(nc)
                    _train_ce(ads, S_ce, Xps_tr, ytr_t, epochs_small, lr_small, wd_small, bs_small)
                    ce_small = _eval_acc(ads, S_ce, Xps_te, y_te)

                    # ---- joint feat-bidir ----
                    _set_seed(seed)
                    B_bd = adb.build(nc)
                    S_bd = ads.build(nc)
                    _joint_bidir(adb, ads, B_bd, S_bd,
                                 Xpb_tr, Xps_tr, ytr_t,
                                 epochs_joint, lr_big, lr_small,
                                 wd_big, wd_small, bs_joint, a.lam_feat)
                    bd_big   = _eval_acc(adb, B_bd, Xpb_te, y_te)
                    bd_small = _eval_acc(ads, S_bd, Xps_te, y_te)
                    with torch.no_grad():
                        pb = torch.zeros(len(y_te), nc)
                        ps = torch.zeros(len(y_te), nc)
                        for i in range(0, len(y_te), 64):
                            _, _lb = adb.forward(B_bd, Xpb_te[i:i+64].to(device))
                            _, _ls = ads.forward(S_bd, Xps_te[i:i+64].to(device))
                            pb[i:i+64] = _lb.cpu(); ps[i:i+64] = _ls.cpu()
                    from torch.nn.functional import softmax
                    pb = softmax(pb, dim=1).numpy(); ps = softmax(ps, dim=1).numpy()
                    bd_ens = ((pb + ps).argmax(1) == y_te).mean() * 100

                    rows.append(dict(
                        dataset=ds, big=a.big, small=a.small,
                        split=split_label, val_split=val_split,
                        seed=seed, subject=subj,
                        ce_big=round(ce_big, 2),
                        ce_big_long=round(ce_big_long, 2),
                        ce_small=round(ce_small, 2),
                        bd_big=round(bd_big, 2),
                        bd_small=round(bd_small, 2),
                        bd_ens=round(bd_ens, 2),
                        delta_big=round(bd_big - ce_big_long, 2),   # fair: same epoch count
                        delta_small=round(bd_small - ce_small, 2),
                    ))
                    print(f'[{ds} {a.big}x{a.small}] split={split_label} '
                          f'seed{seed} S{subj} | '
                          f'ce_big={ce_big:.1f}({ce_big_long:.1f}) ce_small={ce_small:.1f} '
                          f'bd_big={bd_big:.1f} bd_small={bd_small:.1f} '
                          f'Δbig(fair)={bd_big-ce_big_long:+.1f} Δsmall={bd_small-ce_small:+.1f}',
                          flush=True)

                    del B_ce, B_ce_long, S_ce, B_bd, S_bd
                    if device != 'cpu':
                        torch.cuda.empty_cache()

                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    from scipy.stats import wilcoxon
    print('\n=== feat-bidir vs CE-only (subject-level Wilcoxon) ===')
    print('  [bd_big vs ce_big_long: fair comparison — both use epochs_joint epochs]')
    for (ds, big, small, split), g in df.groupby(['dataset', 'big', 'small', 'split']):
        gs = g.groupby('subject')[['ce_big', 'ce_big_long', 'ce_small',
                                    'bd_big', 'bd_small', 'bd_ens']].mean()
        print(f'\n  {ds} {big}x{small} {split}  (n={len(gs)} subjects)')
        for col_new, col_base in [('bd_big',   'ce_big_long'),   # fair: same epoch count
                                   ('bd_small', 'ce_small'),
                                   ('bd_ens',   'ce_small')]:
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
