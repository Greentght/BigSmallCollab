"""Compare re-run metrics CSVs against their historical ground truth.

Matches files by basename between the two globs, then per condition checks:

    deterministic  — historical rows are bit-reproducible: per (unit, seed,
                     condition) the acc% must match exactly (max abs diff == 0)
    ci             — historically nondeterministic lines: unit is paired and
                     the mean per-unit diff (repro - hist) must fall inside
                     the bootstrap CI; PASS iff CI covers 0 or |mean| <= tol

Prints a per-condition table and exits 0 only if every check passed.

Usage:
    python eval/compare_repro.py --hist 'results/metrics/*.csv' \
        --repro 'results/metrics_repro/*.csv' --mode deterministic
    python eval/compare_repro.py --hist 'results/metrics/*.csv' \
        --repro 'results/metrics_repro/*.csv' --mode ci --tol 0.5
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from eval import stats as estats

UNIT_COLS = ('subject', 'fold')
COND_COLS = ('condition', 'group')


def _norm_name(b):
    """Normalize historical filename variants to the current unified model names
    (e.g. ``cbramodnative`` -> ``cbramod`` from the pre-rename CSV names)."""
    return b.replace('cbramodnative', 'cbramod')


def _match_files(hist_pat, repro_pat):
    hist = {_norm_name(os.path.basename(f)): f
            for f in glob.glob(hist_pat)}
    repro = {_norm_name(os.path.basename(f)): f
             for f in glob.glob(repro_pat)}
    if not hist:
        raise SystemExit(f'no historical CSV matches {hist_pat!r}')
    if not repro:
        raise SystemExit(f'no re-run CSV matches {repro_pat!r}')
    missing = sorted(set(hist) - set(repro))
    extra = sorted(set(repro) - set(hist))
    if missing:
        print(f'[warn] re-run missing {len(missing)} files: {missing}')
    if extra:
        print(f'[warn] re-run has {len(extra)} extra files (not compared): {extra}')
    return {name: (hist[name], repro[name]) for name in sorted(set(hist) & set(repro))}


def _cols(df):
    unit = next(c for c in UNIT_COLS if c in df.columns)
    cond = next(c for c in COND_COLS if c in df.columns)
    return unit, cond


def check_deterministic(hist, repro):
    unit, cond = _cols(hist)
    keys = [unit, 'seed', cond] if 'seed' in hist.columns else [unit, cond]
    m = pd.merge(hist, repro, on=keys, suffixes=('_h', '_r'), how='outer')
    n_rows = len(m)
    m['acc_diff'] = (m['acc_r'] - m['acc_h']).abs()
    max_diff = float(m['acc_diff'].max()) if n_rows else float('nan')
    return max_diff, n_rows


def check_ci(hist, repro, tol=0.5, n_boot=2000):
    unit, cond = _cols(hist)
    # seeds -> per-unit mean first (project discipline), then pair units
    def agg(d):
        return d.groupby([unit, cond], as_index=False)['acc'].mean()
    m = pd.merge(agg(hist), agg(repro), on=[unit, cond],
                 suffixes=('_h', '_r'))
    diff = (m['acc_r'] - m['acc_h']).values
    n_units = len(diff)
    lo, hi = estats._bootstrap_ci(diff, n_boot=n_boot)
    mean_diff = float(diff.mean())
    ok = (lo <= 0 <= hi) or abs(mean_diff) <= tol
    return mean_diff, lo, hi, ok, n_units


def main():
    ap = argparse.ArgumentParser(prog='eval/compare_repro.py')
    ap.add_argument('--hist', required=True, help='glob of historical CSVs')
    ap.add_argument('--repro', required=True, help='glob of re-run CSVs')
    ap.add_argument('--mode', choices=('deterministic', 'ci'),
                    default='deterministic')
    ap.add_argument('--tol', type=float, default=0.5,
                    help='ci mode: |mean diff| tolerance in acc pp')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()

    matched = _match_files(a.hist, a.repro)
    print(f'{a.mode} mode | tol={a.tol}pp | {len(matched)} matched files\n')
    rows, all_ok = [], True
    for name, (hf, rf) in matched.items():
        hist, repro = pd.read_csv(hf), pd.read_csv(rf)
        unit, cond = _cols(hist)
        for c in sorted(set(hist[cond]) & set(repro[cond])):
            h, r = hist[hist[cond] == c], repro[repro[cond] == c]
            if a.mode == 'deterministic':
                max_diff, n_rows = check_deterministic(h, r)
                ok = max_diff == 0.0
                row = dict(file=name, condition=c, n_rows=n_rows,
                           max_acc_diff=max_diff, verdict='PASS' if ok else 'FAIL')
            else:
                mean_diff, lo, hi, ok, n_units = check_ci(h, r, a.tol, a.n_boot)
                row = dict(file=name, condition=c, n_units=n_units,
                           mean_diff=round(mean_diff, 4),
                           ci_low=round(lo, 4), ci_high=round(hi, 4),
                           verdict='PASS' if ok else 'FAIL')
            rows.append(row)
            all_ok &= ok

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    fails = df[df.verdict == 'FAIL'] if len(df) else df
    print(f'\n{len(df) - len(fails)}/{len(df)} PASS')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
