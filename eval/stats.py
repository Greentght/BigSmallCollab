"""Subject-level paired statistics — the project's model-selection protocol.

The discipline (from repeated hard lessons, see [[fix-seeds-for-small-gains]]):
~1% gains are only credible after **fixed seeds + subject-level paired Wilcoxon +
Holm correction + a confidence interval**, never from comparing raw means or a
small-sample probe. This module makes that the default, so a new experiment gets
it from one call instead of re-implementing ``analyze_*.py`` each time.

Data model: long-form rows, one per (unit, seed, condition):
    dataset, subject|fold, seed, condition|group, acc, kappa, [extra...]
The **pairing unit** is the subject (``subject`` in within-subject CSVs, ``fold``
in LOSO CSVs); seeds are averaged into a per-unit mean first (a subject is one
paired observation, not one-per-seed — that would fake-inflate n). ``acc`` is
reported as a percentage, always alongside kappa.
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

UNIT_COLS = ('subject', 'fold')          # whichever is present = pairing unit
COND_COLS = ('condition', 'group')       # whichever is present = method label


def load_metrics(pattern):
    """Concatenate every metrics CSV matching a glob ``pattern`` into one frame."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no metrics CSV matches {pattern!r}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _pick(df, names, kind):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f'no {kind} column in {list(df.columns)} (looked for {names})')


def holm(pairs):
    """Holm-Bonferroni step-down. ``pairs`` = list of (label, p). Returns
    ``{label: adjusted_p}`` (monotone, capped at 1.0)."""
    order = sorted(pairs, key=lambda x: x[1])
    m = len(order)
    adj, running = {}, 0.0
    for i, (lab, p) in enumerate(order):
        running = max(running, (m - i) * p)
        adj[lab] = min(1.0, running)
    return adj


def _bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of paired per-unit differences.
    Fixed ``seed`` so the interval is reproducible (project discipline)."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_stats(df, method, baseline, metric='acc', unit_col=None,
                 cond_col=None, seed_col='seed', n_boot=10000, ci_seed=0):
    """One paired contrast ``method`` vs ``baseline`` on ``metric``.

    Averages seeds -> per-unit mean, pairs on the shared units, and returns a dict:
    ``mean_method, mean_baseline, delta`` (mean of per-unit diffs), ``median_diff,
    n_win, n, pvalue`` (paired Wilcoxon), ``ci_low, ci_high`` (bootstrap CI of the
    mean delta). ``delta`` is in the metric's units (acc = percentage points).
    """
    unit_col = unit_col or _pick(df, UNIT_COLS, 'unit')
    cond_col = cond_col or _pick(df, COND_COLS, 'condition')
    sub = df[df[cond_col].isin([method, baseline])]
    piv = (sub.groupby([unit_col, cond_col])[metric].mean()
           .unstack(cond_col))
    if method not in piv.columns or baseline not in piv.columns:
        raise KeyError(f'{method!r} or {baseline!r} not in {list(piv.columns)}')
    piv = piv.dropna(subset=[method, baseline])
    a, b = piv[method].values, piv[baseline].values
    d = a - b
    if len(d) == 0:
        raise ValueError(f'no shared units between {method} and {baseline}')
    try:
        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = _bootstrap_ci(d, n_boot=n_boot, seed=ci_seed)
    return {
        'method': method, 'baseline': baseline, 'metric': metric,
        'mean_method': float(a.mean()), 'mean_baseline': float(b.mean()),
        'delta': float(d.mean()), 'median_diff': float(np.median(d)),
        'n_win': int((d > 0).sum()), 'n': int(len(d)), 'pvalue': p,
        'ci_low': lo, 'ci_high': hi,
    }


def compare(df, baseline, methods, metric='acc', unit_col=None, cond_col=None,
            seed_col='seed', n_boot=10000, ci_seed=0):
    """Run a family of contrasts ``methods`` vs ``baseline`` on one ``metric`` and
    apply Holm across the family. Returns a tidy DataFrame (one row per method)
    with an added ``holm_p`` column, sorted by the raw p-value."""
    rows = [paired_stats(df, m, baseline, metric, unit_col, cond_col, seed_col,
                         n_boot, ci_seed) for m in methods]
    adj = holm([(r['method'], r['pvalue']) for r in rows])
    for r in rows:
        r['holm_p'] = adj[r['method']]
    return (pd.DataFrame(rows)
            .sort_values('pvalue')
            .reset_index(drop=True))


def summarize(df, metrics=('acc', 'kappa'), cond_col=None, unit_col=None):
    """Per-condition mean of each metric (seeds and units pooled). Returns a
    DataFrame indexed by condition; acc stays in percent."""
    cond_col = cond_col or _pick(df, COND_COLS, 'condition')
    cols = [m for m in metrics if m in df.columns]
    return df.groupby(cond_col)[cols].mean().round(4)


def report_contrasts(df, baseline, methods, metrics=('acc', 'kappa'),
                     unit_col=None, cond_col=None, seed_col='seed',
                     n_boot=10000, ci_seed=0, star=(0.05, 0.10)):
    """Print the full model-selection report and return
    ``{metric: compare(...) DataFrame}``.

    For each metric: the per-unit paired Wilcoxon delta vs ``baseline``, its
    bootstrap CI, win count, raw p and Holm-adjusted p. ``**`` marks Holm<star[0],
    ``.`` marks Holm<star[1]. acc deltas are percentage points.
    """
    unit_col = unit_col or _pick(df, UNIT_COLS, 'unit')
    cond_col = cond_col or _pick(df, COND_COLS, 'condition')
    n_units = df[unit_col].nunique()
    seeds = sorted(df[seed_col].unique()) if seed_col in df.columns else []
    print(f'== units({unit_col})={n_units}  seeds={seeds}  '
          f'baseline={baseline} ==')
    print('\nPer-condition mean:')
    print(summarize(df, metrics, cond_col).to_string(), '\n')

    out = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        res = compare(df, baseline, methods, metric, unit_col, cond_col,
                      seed_col, n_boot, ci_seed)
        out[metric] = res
        unit = 'pp' if metric == 'acc' else ''
        print(f'--- {metric}: paired Wilcoxon vs {baseline} '
              f'(Delta = method - baseline, seeds->per-{unit_col} mean) ---')
        for _, r in res.iterrows():
            mark = ('**' if r['holm_p'] < star[0]
                    else '. ' if r['holm_p'] < star[1] else '  ')
            print(f'  {mark}{r["method"]:28s} '
                  f'{r["mean_method"]:6.2f} vs {r["mean_baseline"]:6.2f}  '
                  f'Delta={r["delta"]:+6.3f}{unit}  '
                  f'CI[{r["ci_low"]:+.3f},{r["ci_high"]:+.3f}]  '
                  f'win {r["n_win"]}/{r["n"]}  '
                  f'p={r["pvalue"]:.3f} Holm={r["holm_p"]:.3f}')
        print()
    return out
