"""Shared per-prediction metrics: accuracy and Cohen's kappa.

Kappa is the headline metric for the kappa-track experiments (matches MIRepNet's
``result/kappa/*``). Subject-level paired statistics (Wilcoxon / Holm / bootstrap
CI over matched units) live in the :mod:`eval` layer — use ``eval.paired_stats`` /
``eval.report_contrasts`` for model selection, not this module.
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
