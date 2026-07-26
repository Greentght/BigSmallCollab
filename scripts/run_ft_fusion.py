"""F+T main experiment — few-shot subject-adaptive feature fusion, rigorously.

For a (big, small, dataset) cell under LOSO, adapt lightweight heads on K labeled
trials of each *test subject* over the cached frozen features and fuse. Sweeps K,
ablates the fusion mechanism, and runs the subject-level paired statistics the
project requires (fixed seeds, paired Wilcoxon vs the strongest control, Holm over
the K grid). No base model is retrained — reuses D0 LOSO test artifacts.

Primary hypothesis: fusion (both models' features) beats adapting the *stronger*
model alone at the same label budget K (head_big), i.e. the collaboration adds
value beyond just "fine-tune the big model on K trials".

    conda run -n mirepnet python scripts/run_ft_fusion.py --big mirepnet --small ifnet \
        --dataset BNCI2014001-4 --Ks 5 10 20 30 --seeds 666 667 668 --draws 5
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit

import config
from collab import artifacts
from collab import fusion as fz
from collab.router import _softmax_np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = os.path.join(ROOT, 'results', 'leaderboard.csv')
METRICS = os.path.join(ROOT, 'results', 'metrics')


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


LINEAR_METHODS = ['fixed_big', 'fixed_small', 'fixed_avg', 'head_big', 'head_small',
                  'fusion_lr', 'oracle']
TORCH_METHODS = ['fusion_mlp', 'gated', 'mutual']


def eval_methods(bf_tr, sf_tr, y_tr, bf_te, sf_te, y_te, nc, pb_te, ps_te, seed,
                 device, methods):
    """Fit the requested methods on the K-trial support -> {method: (acc, kappa)}."""
    out = {}
    def rec(name, pred):
        out[name] = ((pred == y_te).mean() * 100,
                     cohen_kappa_score(y_te, pred, labels=list(range(nc))))
    if 'fixed_big' in methods: rec('fixed_big', pb_te.argmax(1))
    if 'fixed_small' in methods: rec('fixed_small', ps_te.argmax(1))
    if 'fixed_avg' in methods: rec('fixed_avg', (pb_te + ps_te).argmax(1))
    if 'head_big' in methods: rec('head_big', fz.head_single(bf_tr, y_tr, bf_te, nc, seed=seed, device=device))
    if 'head_small' in methods: rec('head_small', fz.head_single(sf_tr, y_tr, sf_te, nc, seed=seed, device=device))
    if 'fusion_lr' in methods: rec('fusion_lr', fz.fusion_concat(bf_tr, sf_tr, y_tr, bf_te, sf_te, nc, hidden=0, seed=seed, device=device))
    if 'fusion_mlp' in methods: rec('fusion_mlp', fz.fusion_concat(bf_tr, sf_tr, y_tr, bf_te, sf_te, nc, hidden=64, seed=seed, device=device))
    if 'gated' in methods: rec('gated', fz.fusion_gated(bf_tr, sf_tr, y_tr, bf_te, sf_te, nc, seed=seed, device=device))
    if 'mutual' in methods: rec('mutual', fz.fusion_mutual(bf_tr, sf_tr, y_tr, bf_te, sf_te, nc, seed=seed, device=device))
    if 'oracle' in methods:
        rec('oracle', np.where((pb_te.argmax(1) == y_te), pb_te.argmax(1),
                               np.where(ps_te.argmax(1) == y_te, ps_te.argmax(1), pb_te.argmax(1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--big', default='mirepnet')
    ap.add_argument('--small', default='ifnet')
    ap.add_argument('--dataset', default='BNCI2014001-4')
    ap.add_argument('--Ks', type=int, nargs='+', default=[5, 10, 20, 30])
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--linear_only', action='store_true',
                    help='skip torch fusion methods (fast sklearn sweep)')
    a = ap.parse_args()
    METHODS = LINEAR_METHODS if a.linear_only else LINEAR_METHODS[:-1] + TORCH_METHODS + ['oracle']
    import torch
    device = (f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
    dcfg = config.load_dataset_config(a.dataset)
    seeds = a.seeds or dcfg['seeds']
    nc, n_sub = dcfg['num_classes'], dcfg['num_subjects']
    bdir, sdir = f'{a.big}_loso', f'{a.small}_loso'

    # per (K, method): subj -> list of (acc,kappa) over seeds×draws
    accs = {K: {m: {} for m in METHODS} for K in a.Ks}
    rows = []
    for seed in seeds:
        for t in range(n_sub):
            try:
                per, y = artifacts.load_aligned(a.dataset, [bdir, sdir], t, seed, 'test')
            except (FileNotFoundError, ValueError):
                continue
            bf, sf = per[bdir]['feats'], per[sdir]['feats']
            pb, ps = _softmax_np(per[bdir]['logits']), _softmax_np(per[sdir]['logits'])
            if np.bincount(y, minlength=nc).min() < 2:
                continue
            for K in a.Ks:
                if len(y) <= K + nc:
                    continue
                sss = StratifiedShuffleSplit(n_splits=a.draws, train_size=K, random_state=seed)
                for di, (tr, te) in enumerate(sss.split(bf, y)):
                    res = eval_methods(bf[tr], sf[tr], y[tr], bf[te], sf[te], y[te],
                                       nc, pb[te], ps[te], seed * 100 + di, device, METHODS)
                    for m, (ac, kp) in res.items():
                        accs[K][m].setdefault(t, []).append((ac, kp))
                        rows.append(dict(dataset=a.dataset, big=a.big, small=a.small,
                                         seed=seed, subject=t, K=K, draw=di,
                                         method=m, acc=round(ac, 3), kappa=round(kp, 4)))
        print(f'[seed {seed}] done', flush=True)

    os.makedirs(METRICS, exist_ok=True)
    tag = f'_{a.tag}' if a.tag else ''
    long_csv = os.path.join(METRICS, f'ft_fusion_{a.dataset}_{a.big}_{a.small}{tag}.csv')
    import pandas as pd
    pd.DataFrame(rows).to_csv(long_csv, index=False)

    # aggregate + stats
    print(f'\n=== F+T fusion: {a.big} x {a.small} / {a.dataset} / LOSO '
          f'({len(seeds)} seeds x {a.draws} draws, {n_sub} subjects) ===')
    print(f'{"K":>4} | ' + ' '.join(f'{m[:10]:>10}' for m in METHODS))
    primary_p = []
    for K in a.Ks:
        subj_mean = {m: np.array([np.mean([v[0] for v in accs[K][m][t]])
                                  for t in sorted(accs[K][m])]) for m in METHODS}
        line = f'{K:>4} | ' + ' '.join(f'{subj_mean[m].mean():>10.2f}' for m in METHODS)
        print(line)
        # primary: fusion_lr vs head_big (strongest single-model adaptation)
        if len(subj_mean['fusion_lr']) >= 3:
            try:
                _, p = wilcoxon(subj_mean['fusion_lr'], subj_mean['head_big'])
            except ValueError:
                p = 1.0
            primary_p.append((K, subj_mean['fusion_lr'].mean() - subj_mean['head_big'].mean(),
                              int((subj_mean['fusion_lr'] > subj_mean['head_big']).sum()),
                              len(subj_mean['fusion_lr']), p))
    print('\n  Primary test  fusion_lr − head_big  (subject-level paired Wilcoxon, Holm over K):')
    if primary_p:
        praw = [x[4] for x in primary_p]
        padj = holm(praw)
        for (K, d, w, n, p), pa in zip(primary_p, padj):
            sig = '***' if pa < 0.05 else ('*' if p < 0.05 else '')
            print(f'    K={K:>3}  Δ={d:+5.2f}  wins {w}/{n}  p={p:.4f}  p_holm={pa:.4f} {sig}')

    # leaderboard rows (mean over subjects/seeds/draws, per K, key methods)
    write_header = not os.path.exists(LB) or os.path.getsize(LB) == 0
    with open(LB, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'protocol', 'method', 'variant', 'big', 'small',
                        'seed', 'acc', 'f1', 'kappa', 'wake_rate', 'params',
                        'vs_best_baseline_p', 'significant'])
        pmap = {K: p for (K, *_), p in zip(primary_p, holm([x[4] for x in primary_p]))} if primary_p else {}
        for K in a.Ks:
            for m in ['fusion_lr', 'fusion_mlp', 'gated', 'mutual', 'head_big', 'fixed_big', 'oracle']:
                if m not in accs[K] or not accs[K][m]:
                    continue
                vals = [np.mean([v[0] for v in accs[K][m][t]]) for t in sorted(accs[K][m])]
                kps = [np.mean([v[1] for v in accs[K][m][t]]) for t in sorted(accs[K][m])]
                if not vals:
                    continue
                pa = pmap.get(K, '')
                sig = ('yes' if (m == 'fusion_lr' and pa != '' and pa < 0.05) else '')
                w.writerow([a.dataset, 'loso_fewshot', 'F+T', f'{m}_K{K}', a.big, a.small,
                            f'mean{len(seeds)}x{a.draws}', round(np.mean(vals), 2), '',
                            round(np.mean(kps), 4), '', '',
                            round(pa, 4) if (m == 'fusion_lr' and pa != '') else '', sig])
    print(f'\nWrote {long_csv}\nAppended F+T rows to {LB}')


if __name__ == '__main__':
    main()
