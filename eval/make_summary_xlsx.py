"""Summarize metrics CSVs into one xlsx workbook.

Replaces the historical hand-made ``results/*_summary.xlsx`` files with
in-repo code. Reads every metrics CSV matched by a glob (each file's rows
get a ``source`` column = its filename), then writes three sheets:

    by_method  — per source x method mean acc%/kappa (seeds+units pooled)
    stats    — eval.stats.compare vs the auto-picked baseline (method
               ending in base/Base), one row per method, incl. Holm p
    diff     — when both --repro and --hist globs are given: per source x
               method acc% means side by side and their difference

Usage:
    python eval/make_summary_xlsx.py \
        --hist 'results/*.csv' \
        --repro 'results_repro/*.csv' \
        [--out results/summary_repro.xlsx] [--n_boot 2000]

Only basic row/column writes (no styling) so old openpyxl works; if xlsx
writing fails it falls back to writing each sheet as a CSV next to --out.
"""
import argparse
import glob
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from eval import stats as estats
from experiments.storage import (RESULTS_ROOT, external_path,
                                 require_external_output, resolve_local_file)

UNIT_COLS = ('subject', 'fold')


def _load(pattern):
    """Load every CSV matching ``pattern`` with a ``source`` column."""
    files = sorted(glob.glob(str(external_path(pattern))))
    if not files:
        raise FileNotFoundError(f'no CSV matches {pattern!r}')
    parts = [pd.read_csv(resolve_local_file(f)).assign(source=os.path.basename(f))
             for f in files]
    return pd.concat(parts, ignore_index=True)


def _unit_col(df):
    return next(c for c in UNIT_COLS if c in df.columns)


def _method_col(df):
    return estats._method_col(df)


def by_method(df):
    method = _method_col(df)
    cols = [c for c in ('acc', 'kappa') if c in df.columns]
    return (df.groupby(['source', method])[cols].mean()
            .round(4).reset_index())


def stats_sheet(df, n_boot=2000):
    """Per-source paired contrasts vs the auto-picked baseline."""
    method = _method_col(df)
    unit = _unit_col(df)
    rows = []
    for source, sub in df.groupby('source'):
        methods = list(sub[method].unique())
        base = next((c for c in methods if str(c).lower().endswith('base')), None)
        if base is None or len(methods) < 2:
            continue
        res = estats.compare(sub, base, [c for c in methods if c != base],
                             metric='acc', n_boot=n_boot)
        for _, r in res.iterrows():
            row = dict(r.to_dict())
            row['source'] = source
            rows.append(row)
    return pd.DataFrame(rows).round(4)


def diff_sheet(hist, repro):
    h = by_method(hist).rename(columns={'acc': 'acc_hist', 'kappa': 'kappa_hist'})
    r = by_method(repro).rename(columns={'acc': 'acc_repro', 'kappa': 'kappa_repro'})
    method = _method_col(hist)
    m = pd.merge(h, r, on=['source', method], how='outer')
    m['acc_diff'] = (m['acc_repro'] - m['acc_hist']).round(4)
    if 'kappa_repro' in m and 'kappa_hist' in m:
        m['kappa_diff'] = (m['kappa_repro'] - m['kappa_hist']).round(4)
    return m


def _write(path, sheets):
    """xlsx via pandas/openpyxl; fall back to per-sheet CSV on failure."""
    path = require_external_output(path)
    try:
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name, index=False)
        print(f'Wrote {path}')
        return
    except Exception as e:
        print(f'[xlsx failed: {e}] falling back to CSV next to {path}')
    stem, _ = os.path.splitext(path)
    for name, df in sheets.items():
        df.to_csv(require_external_output(f'{stem}_{name}.csv'), index=False)
        print(f'Wrote {stem}_{name}.csv')


def main():
    ap = argparse.ArgumentParser(prog='eval/make_summary_xlsx.py')
    ap.add_argument('--hist', default=None, help='glob of historical metrics CSVs')
    ap.add_argument('--repro', default=None, help='glob of re-run metrics CSVs')
    ap.add_argument('--out', default=None,
                    help=f'output path (default {RESULTS_ROOT}/summary_<date>.xlsx)')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()

    if not a.hist and not a.repro:
        raise SystemExit('need at least one of --hist/--repro')

    sheets = {}
    hist = repro = None
    if a.hist:
        hist = _load(a.hist)
    if a.repro:
        repro = _load(a.repro)
    for label, df in (('hist', hist), ('repro', repro)):
        if df is not None:
            sheets[f'by_method_{label}'] = by_method(df)
            sheets[f'stats_{label}'] = stats_sheet(df, a.n_boot)
    if hist is not None and repro is not None:
        sheets['diff'] = diff_sheet(hist, repro)

    out = require_external_output(
        a.out or RESULTS_ROOT / f'summary_{datetime.now():%Y%m%d_%H%M%S}.xlsx')
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, sheets)


if __name__ == '__main__':
    main()
