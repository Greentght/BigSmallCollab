"""Analyze BD-EEG LOSO results. Aggregates seeds->per-fold means, prints per-group
S_acc/B_acc (mean over folds), and paired Wilcoxon (per-fold) + Holm for the key
contrasts, for BOTH the small model (S) and big model (B). Reports accuracy %.

    conda run -n mirepnet python scripts/analyze_bdeeg.py --dataset BNCI2014001-4
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired(df, metric, a, b):
    """Per-fold paired diff a-b (mean over seeds first). Returns (mean_a, mean_b,
    delta, n_win, n, wilcoxon_p)."""
    piv = (df[df.group.isin([a, b])]
           .groupby(['fold', 'group'])[metric].mean().unstack('group'))
    piv = piv.dropna(subset=[a, b])
    va, vb = piv[a].values, piv[b].values
    d = va - vb
    try:
        p = wilcoxon(va, vb).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return va.mean(), vb.mean(), d.mean(), int((d > 0).sum()), len(d), p


def holm(pairs):
    """Holm-Bonferroni on list of (label, p). Returns dict label->adjusted p."""
    order = sorted(pairs, key=lambda x: x[1])
    m = len(order); adj = {}; running = 0.0
    for i, (lab, p) in enumerate(order):
        running = max(running, (m - i) * p)
        adj[lab] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='BNCI2014001-4')
    ap.add_argument('--tag', default='bdeeg', help='exact tag, or "bdeeg_f*" glob')
    a = ap.parse_args()
    patt = f'results/metrics/{a.dataset}_loso_{a.tag}_mirepnet_ifnet.csv'
    files = sorted(glob.glob(patt))
    if not files:
        raise SystemExit(f'no CSVs match {patt}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f'== {len(files)} file(s): {patt} ==')
    print(f'seeds={sorted(df.seed.unique())}  folds={sorted(df.fold.unique())}  '
          f'groups={list(df.group.unique())}\n')

    # per-group means (over fold x seed) and per-fold-mean +/- std
    g = df.groupby('group')
    tab = g[['S_acc', 'B_acc', 'S_kappa', 'B_kappa']].mean().round(2)
    print('Per-group mean (over fold x seed):')
    print(tab.to_string(), '\n')

    contrasts = [
        ('BD_EEG', 'G0_CE',        'core vs no-distill control'),
        ('BD_EEG', 'G1_FixKD',     'core vs traditional one-way KD'),
        ('BD_EEG', 'SymDML',       'diff-aware selection useful?'),
        ('BD_EEG', 'StrictRouted', 'continuous CE-diff vs binary routing?'),
        ('BD_EEG', 'BD_EEG_Swap',  'asymmetry direction correct?'),
    ]
    for metric, who in [('S_acc', 'SMALL model S'), ('B_acc', 'BIG model B')]:
        print(f'--- {who}: paired per-fold Wilcoxon (Delta = A - B, acc%) ---')
        rows, ps = [], []
        for A, Bg, desc in contrasts:
            if A not in df.group.values or Bg not in df.group.values:
                continue
            ma, mb, dl, win, n, p = paired(df, metric, A, Bg)
            rows.append((f'{A} - {Bg}', ma, mb, dl, win, n, p, desc))
            ps.append((f'{A} - {Bg}', p))
        adj = holm(ps)
        for lab, ma, mb, dl, win, n, p, desc in rows:
            print(f'  {lab:24s} {ma:5.2f} vs {mb:5.2f}  Delta={dl:+5.2f}  '
                  f'win {win}/{n}  p={p:.3f} Holm={adj[lab]:.3f}  # {desc}')
        print()


if __name__ == '__main__':
    main()
