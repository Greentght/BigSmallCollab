"""Subject-level paired analysis of the few-shot Pearson (L_inter) experiment.

Per [[fix-seeds-for-small-gains]]: average seeds within subject first, then do
subject-level paired comparisons (unit = subject, n=9). For each --shots level
we report, for the key contrasts, mean/median Delta acc%, 95% bootstrap CI,
win count, paired Wilcoxon p, and Holm-corrected p across the contrasts.

    conda run -n mirepnet python experiments/distill/analyze_fewshot_pearson.py \
        results/metrics/BNCI2014001-4_fewshot_pearson_mirepnet_to_ifnet.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# (name, condition_a, condition_b) -> tests a - b
CONTRASTS = [
    ('Pearson  - base', 'Pearson', 'base'),
    ('KD       - base', 'KD', 'base'),
    ('KD+Pears - base', 'KD+Pearson', 'base'),
    ('KD+Pears - KD', 'KD+Pearson', 'KD'),
]


def _boot_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    means = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def analyze(csv):
    df = pd.read_csv(csv)
    df['cond'] = df['condition'].str.replace(r'^[^_]+_', '', regex=True)
    student = df['condition'].iloc[0].split('_')[0]
    print(f"\n==== {csv}  (student={student}, "
          f"{df['subject'].nunique()} subjects, seeds={sorted(df['seed'].unique())}) ====")
    for shots, g in df.groupby('shots'):
        # per-subject mean over seeds
        piv = g.groupby(['subject', 'cond'])['acc'].mean().unstack('cond')
        means = piv.mean()
        print(f"\n--- shots={shots} (n_subj={len(piv)}) | mean acc%: "
              + ', '.join(f"{c}={means[c]:.2f}" for c in
                          ['base', 'KD', 'Pearson', 'KD+Pearson'] if c in means))
        rows, praw = [], []
        for name, a, b in CONTRASTS:
            if a not in piv or b not in piv:
                continue
            d = (piv[a] - piv[b]).dropna().values
            lo, hi = _boot_ci(d)
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            rows.append([name, d.mean(), np.median(d), lo, hi,
                         int((d > 0).sum()), len(d), p])
            praw.append(p)
        padj = holm(np.array(praw))
        print(f"  {'contrast':<16} {'dAcc':>7} {'med':>7} {'CI95':>16} "
              f"{'wins':>6} {'p':>7} {'pHolm':>7}")
        for r, pa in zip(rows, padj):
            name, mean, med, lo, hi, w, n, p = r
            flag = '*' if (lo > 0 or hi < 0) and pa < 0.05 else (
                '~' if (lo > 0 or hi < 0) else ' ')
            print(f"  {name:<16} {mean:>+7.2f} {med:>+7.2f} "
                  f"[{lo:>+6.2f},{hi:>+6.2f}] {w:>3}/{n:<2} {p:>7.3f} {pa:>7.3f} {flag}")
    print("\n  * = CI excludes 0 AND Holm p<.05 ; ~ = CI excludes 0 only (suggestive)")


if __name__ == '__main__':
    for csv in sys.argv[1:]:
        analyze(csv)
