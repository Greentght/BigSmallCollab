"""CLI for the paired-stats report — replaces the per-experiment analyze_*.py.

    python -m eval 'results/metrics/BNCI2014004_maskablation_*_to_eegnet.csv'
    python -m eval '<glob>' --baseline eegnet_base --methods eegnet_KD_all eegnet_Combo_all
    python -m eval '<glob>' --metrics acc kappa

If ``--baseline`` is omitted, the condition whose name ends in ``base``/``Base`` is
used; ``--methods`` defaults to every other condition.
"""
import argparse

from .stats import load_metrics, report_contrasts, COND_COLS, _pick


def main():
    ap = argparse.ArgumentParser(prog='python -m eval')
    ap.add_argument('pattern', help='glob of metrics CSV(s)')
    ap.add_argument('--baseline', default=None,
                    help='baseline condition (default: the *base/*Base one)')
    ap.add_argument('--methods', nargs='*', default=None,
                    help='methods to test (default: all non-baseline conditions)')
    ap.add_argument('--metrics', nargs='*', default=['acc', 'kappa'])
    ap.add_argument('--n_boot', type=int, default=10000)
    a = ap.parse_args()

    df = load_metrics(a.pattern)
    cond_col = _pick(df, COND_COLS, 'condition')
    conds = list(df[cond_col].unique())

    baseline = a.baseline
    if baseline is None:
        cand = [c for c in conds if str(c).lower().endswith('base')]
        if len(cand) != 1:
            raise SystemExit(
                f'cannot auto-pick baseline from {conds}; pass --baseline')
        baseline = cand[0]
    methods = a.methods or [c for c in conds if c != baseline]
    report_contrasts(df, baseline=baseline, methods=methods,
                     metrics=tuple(a.metrics), n_boot=a.n_boot)


if __name__ == '__main__':
    main()
