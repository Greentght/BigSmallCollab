"""Shared evaluation metrics: accuracy, Cohen's kappa, paired Wilcoxon.

Kappa is the headline metric for the kappa-track experiments (matches MIRepNet's
``result/kappa/*``); the paired Wilcoxon signed-rank test is the agreed
model-selection test over ``(subject, seed)`` pairs (see MIRepNet CLAUDE.md).
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate(y_true, y_pred):
    """Return {'acc': pct, 'kappa': float} for one prediction vector."""
    return {
        'acc': round(accuracy_score(y_true, y_pred) * 100, 2),
        'kappa': round(cohen_kappa_score(y_true, y_pred), 4),
    }


def preds_from_logits(logits):
    return np.asarray(logits).argmax(axis=1)


def paired_wilcoxon(scores_a, scores_b):
    """Paired Wilcoxon signed-rank over matched (subject, seed) scores.

    Returns {'stat', 'pvalue', 'median_diff', 'n_wins_a'}. Use to compare two
    fusion rules rather than comparing means (CLAUDE.md model-selection step 1).
    """
    from scipy.stats import wilcoxon

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('score vectors must be paired (same shape)')
    diff = a - b
    if np.allclose(diff, 0):
        return {'stat': 0.0, 'pvalue': 1.0, 'median_diff': 0.0,
                'n_wins_a': 0}
    stat, p = wilcoxon(a, b)
    return {
        'stat': float(stat),
        'pvalue': float(p),
        'median_diff': float(np.median(diff)),
        'n_wins_a': int((diff > 0).sum()),
    }
