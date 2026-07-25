"""Framework evaluation & statistics layer.

One place for the subject-level paired-stats protocol the project re-derives in
every ``analyze_*`` script (see PROGRESS.md and the [[fix-seeds-for-small-gains]]
memory): aggregate seeds -> per-subject mean, paired Wilcoxon across subjects,
Holm-Bonferroni over the contrast family, fixed-seed bootstrap CI, and acc%-first
reporting. Consume the long-form metrics CSVs written by the run scripts
(columns: ``dataset, subject|fold, seed, condition|group, acc, kappa, ...``).

    from eval import report_contrasts
    df = load_metrics('results/metrics/BNCI2014004_maskablation_*_to_eegnet.csv')
    report_contrasts(df, baseline='eegnet_base',
                     methods=['eegnet_KD_all', 'eegnet_KD_masked'])
"""
from .stats import (
    load_metrics, holm, paired_stats, compare, summarize, report_contrasts,
)

__all__ = [
    'load_metrics', 'holm', 'paired_stats', 'compare', 'summarize',
    'report_contrasts',
]
