"""Subject-level paired analysis of the few-shot bidirectional-distillation runs.

Per [[fix-seeds-for-small-gains]]: average seeds within subject first, then
subject-level paired tests (unit=subject, n=9), Wilcoxon + bootstrap CI + Holm.
Primary metric = small model S_acc; also checks the reverse S->B nudge on B_acc.

    conda run -n mirepnet python scripts/analyze_bidir_fewshot.py \
        results/metrics/BNCI2014004_fewshot_bidirfs_sb0.1_mirepnet_ifnet.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# metric -> list of (label, group_a, group_b): tests a - b
CONTRASTS = {
    'S_acc': [
        ('G5_Routed  - G0 (1way KD vs base)', 'G5_Routed', 'G0_CE'),
        ('G6_CRAMD   - G0 (bidir vs base)',   'G6_CRAMD', 'G0_CE'),
        ('G3_SymDML  - G0 (sym DML vs base)', 'G3_SymDML', 'G0_CE'),
        ('G7_DisagCE - G0 (hardCE vs base)',  'G7_DisagCE', 'G0_CE'),
        ('G6_CRAMD   - G5 (add reverse S->B)', 'G6_CRAMD', 'G5_Routed'),
        ('G5_Routed  - G7 (KD probs vs hardCE)', 'G5_Routed', 'G7_DisagCE'),
    ],
    'B_acc': [   # does the reverse routed S->B help the BIG model?
        ('G6_CRAMD - G0 (S->B nudge vs base)', 'G6_CRAMD', 'G0_CE'),
        ('G6_CRAMD - G5 (S->B nudge vs 1way)', 'G6_CRAMD', 'G5_Routed'),
    ],
}


def _boot_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    return tuple(np.percentile([rng.choice(d, len(d), replace=True).mean()
                                for _ in range(n)], [2.5, 97.5]))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def analyze(csv):
    df = pd.read_csv(csv)
    print(f"\n{'='*72}\n{csv}\n  {df['subject'].nunique()} subjects, "
          f"seeds={sorted(df['seed'].unique())}, groups={sorted(df['group'].unique())}")
    for metric in ('S_acc', 'B_acc'):
        print(f"\n################  metric = {metric}  ################")
        for shots, g in df.groupby('shots'):
            piv = g.groupby(['subject', 'group'])[metric].mean().unstack('group')
            means = piv.mean()
            order = ['G0_CE', 'G1_FixKD', 'G3_SymDML', 'G5_Routed', 'G6_CRAMD', 'G7_DisagCE']
            print(f"\n--- shots={shots} (n={len(piv)}) | mean {metric}: "
                  + ', '.join(f"{c.split('_')[0]}={means[c]:.2f}"
                              for c in order if c in means))
            rows, praw = [], []
            for label, a, b in CONTRASTS[metric]:
                if a not in piv or b not in piv:
                    continue
                d = (piv[a] - piv[b]).dropna().values
                lo, hi = _boot_ci(d)
                try:
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                rows.append((label, d.mean(), np.median(d), lo, hi,
                             int((d > 0).sum()), len(d), p)); praw.append(p)
            for r, pa in zip(rows, holm(praw)):
                label, mean, med, lo, hi, w, n, p = r
                flag = '*' if (lo > 0 or hi < 0) and pa < 0.05 else (
                    '~' if (lo > 0 or hi < 0) else ' ')
                print(f"  {label:<38} {mean:>+6.2f} med{med:>+6.2f} "
                      f"[{lo:>+6.2f},{hi:>+6.2f}] {w:>2}/{n} p{p:>6.3f} pH{pa:>6.3f} {flag}")
    print("\n  * = CI excl 0 & Holm p<.05 ; ~ = CI excl 0 only (suggestive)")


if __name__ == '__main__':
    for csv in sys.argv[1:]:
        analyze(csv)
