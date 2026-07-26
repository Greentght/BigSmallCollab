"""Balance-gated few-shot selection — the corrected main line.

F+T fusion is not an unconditional win: it helps when the big and small models are
comparably strong (MIRepNet pairs) and DILUTES the strong model when they are far
apart (weak-CBraMod pairs). So the deployable result is a *selector* that, using
only the K support trials, decides per subject whether to use the adapted big head,
the adapted small head, or their fusion — capturing the fusion gain on balanced
pairs while never eating the dilution loss.

Methods (all adapted on the K labeled test-subject trials, over frozen features):
  head_big / head_small : single-model adaptation (controls)
  fusion                : concat linear fusion head
  cv_sel2               : pick head_big vs head_small by support-CV (realistic
                          single-model selection — the honest baseline)
  cv_sel3               : pick big vs small vs fusion by support-CV (portfolio)
  bal_gate              : interpretable — if the support-CV accuracy gap
                          |cv_big - cv_small| < tau -> fusion, else the stronger
                          single. Tests hypothesis: fuse only when balanced.
  best_single           : per-subject oracle max(head_big, head_small) (upper ref)

Reports subject-level paired Wilcoxon of cv_sel3 / bal_gate vs cv_sel2 (does adding
fusion to the toolkit help?) and vs head_big (naive "always adapt the foundation
model"), pooled across cells + per cell, plus the interpretability check that the
balance gap predicts fusion's benefit.

    conda run -n mirepnet python scripts/run_balance_gate.py --Ks 20 30 --tau 0.1
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon, binomtest, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit

import config
from collab import artifacts
from collab import fusion as fz
from collab.router import _softmax_np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, 'results', 'metrics')
LB = os.path.join(ROOT, 'results', 'leaderboard.csv')

DEFAULT_CELLS = [
    ('BNCI2014001-4', 'mirepnet'), ('BNCI2014001-4', 'cbramod_native'),
    ('BNCI2014004', 'mirepnet'), ('BNCI2014004', 'cbramod_native'),
    ('BNCI2015001', 'mirepnet'), ('BNCI2015001', 'cbramod_native'),
    ('AlexMI', 'mirepnet'), ('AlexMI', 'cbramod_native'),
]
SMALLS = ['ifnet', 'eegnet', 'adfcnn']


def eval_subject(bf, sf, y, tr, te, nc, tau):
    """One draw for one subject: return {method: acc} + (balance_gap, fusion_benefit)."""
    yt = y[te]
    def acc(p): return (p == yt).mean() * 100
    hb = fz.head_single(bf[tr], y[tr], bf[te], nc)
    hs = fz.head_single(sf[tr], y[tr], sf[te], nc)
    fu = fz.fusion_concat(bf[tr], sf[tr], y[tr], bf[te], sf[te], nc, hidden=0)
    a_hb, a_hs, a_fu = acc(hb), acc(hs), acc(fu)
    cb = fz.cv_acc(bf[tr], y[tr], nc)
    cs = fz.cv_acc(sf[tr], y[tr], nc)
    cf = fz.cv_acc(np.concatenate([bf[tr], sf[tr]], 1), y[tr], nc)
    # selectors (using only support-CV estimates)
    sel2 = a_hb if cb >= cs else a_hs
    best3 = max([(cb, a_hb), (cs, a_hs), (cf, a_fu)], key=lambda x: x[0])[1]
    gate = a_fu if abs(cb - cs) < tau else (a_hb if cb >= cs else a_hs)
    out = dict(head_big=a_hb, head_small=a_hs, fusion=a_fu, cv_sel2=sel2,
               cv_sel3=best3, bal_gate=gate, best_single=max(a_hb, a_hs))
    return out, abs(cb - cs), a_fu - max(a_hb, a_hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ks', type=int, nargs='+', default=[20, 30])
    ap.add_argument('--seeds', type=int, nargs='+', default=[666, 667, 668])
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--cells', default=None, help='comma dataset:big to restrict')
    a = ap.parse_args()
    cells = DEFAULT_CELLS
    if a.cells:
        cells = [(c.split(':')[0], c.split(':')[1]) for c in a.cells.split(',')]

    METHODS = ['head_big', 'head_small', 'fusion', 'cv_sel2', 'cv_sel3', 'bal_gate', 'best_single']
    rows = []
    gaps, benes = [], []
    for ds, big in cells:
        try:
            dcfg = config.load_dataset_config(ds)
        except Exception:
            continue
        nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']
        for small in SMALLS:
            bdir, sdir = f'{big}_loso', f'{small}_loso'
            for K in a.Ks:
                per_sub = {m: {} for m in METHODS}
                have = False
                for seed in a.seeds:
                    for t in range(n_sub):
                        try:
                            P, y = artifacts.load_aligned(ds, [bdir, sdir], t, seed, 'test')
                        except (FileNotFoundError, ValueError):
                            continue
                        bf, sf = P[bdir]['feats'], P[sdir]['feats']
                        if np.bincount(y, minlength=nc).min() < 2 or len(y) <= K + nc:
                            continue
                        have = True
                        for di, (tr, te) in enumerate(StratifiedShuffleSplit(
                                n_splits=a.draws, train_size=K, random_state=seed).split(bf, y)):
                            res, gap, bene = eval_subject(bf, sf, y, tr, te, nc, a.tau)
                            for m in METHODS:
                                per_sub[m].setdefault(t, []).append(res[m])
                            gaps.append(gap); benes.append(bene)
                if not have:
                    continue
                sm = {m: np.array([np.mean(per_sub[m][t]) for t in sorted(per_sub[m])]) for m in METHODS}
                def paired(a_, b_):
                    try:
                        return wilcoxon(a_, b_)[1]
                    except ValueError:
                        return 1.0
                rows.append(dict(
                    dataset=ds, big=big, small=small, K=K, n_sub=len(sm['cv_sel2']),
                    head_big=round(sm['head_big'].mean(), 2),
                    head_small=round(sm['head_small'].mean(), 2),
                    fusion=round(sm['fusion'].mean(), 2),
                    cv_sel2=round(sm['cv_sel2'].mean(), 2),
                    cv_sel3=round(sm['cv_sel3'].mean(), 2),
                    bal_gate=round(sm['bal_gate'].mean(), 2),
                    best_single=round(sm['best_single'].mean(), 2),
                    d_sel3_vs_sel2=round(sm['cv_sel3'].mean() - sm['cv_sel2'].mean(), 2),
                    d_gate_vs_sel2=round(sm['bal_gate'].mean() - sm['cv_sel2'].mean(), 2),
                    d_gate_vs_headbig=round(sm['bal_gate'].mean() - sm['head_big'].mean(), 2),
                    p_sel3_vs_sel2=round(paired(sm['cv_sel3'], sm['cv_sel2']), 4),
                    p_gate_vs_sel2=round(paired(sm['bal_gate'], sm['cv_sel2']), 4)))

    os.makedirs(METRICS, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(METRICS, 'balance_gate.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 240, 'display.max_columns', 40)
    print(df.to_string(index=False))

    print('\n=== aggregate over cells ===')
    for K in a.Ks:
        d = df[df.K == K]
        if d.empty:
            continue
        for col, ref in [('d_sel3_vs_sel2', 'cv_sel3 vs cv_sel2'),
                         ('d_gate_vs_sel2', 'bal_gate vs cv_sel2'),
                         ('d_gate_vs_headbig', 'bal_gate vs head_big(always-foundation)')]:
            pos = int((d[col] > 0).sum()); n = len(d)
            bt = binomtest(pos, n, 0.5, alternative='greater').pvalue
            print(f'  K={K}  {ref:42s} mean Δ={d[col].mean():+5.2f}  '
                  f'{pos}/{n} cells positive  sign-test p={bt:.4f}')
    if gaps:
        rho, p = spearmanr(gaps, benes)
        print(f'\n  interpretability: Spearman(balance_gap, fusion_benefit) = {rho:+.3f} '
              f'(p={p:.1e})  [negative ⇒ fusion helps when models balanced]')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
