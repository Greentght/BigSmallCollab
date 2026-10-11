"""Pure calculations for the diagnostic-only c metric experiment."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


WINDOWS = {
    'c_1_100': (0, 100),
    'c_1_30': (0, 30),
    'c_11_100': (10, 100),
}


def c_components(prob_ce, prob_kd, labels, windows=WINDOWS):
    """Return D, B and c for each window using post-epoch probabilities.

    ``prob_*`` are [epochs, samples, classes] arrays for epochs 1 onward;
    epoch 0 must not be included. Each window restarts its prescribed linear
    weights at (E-1)/E and ends at zero.
    """
    prob_ce = np.asarray(prob_ce, dtype=np.float64)
    prob_kd = np.asarray(prob_kd, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if prob_ce.ndim != 3 or prob_kd.shape != prob_ce.shape:
        raise ValueError('CE/KD probabilities must have matching [epochs,samples,classes] shapes')
    if prob_ce.shape[1] != len(labels):
        raise ValueError('probability sample axis does not match labels')
    if len(labels) and (labels.min() < 0 or labels.max() >= prob_ce.shape[2]):
        raise ValueError('labels are outside the probability class axis')
    if not np.isfinite(prob_ce).all() or not np.isfinite(prob_kd).all():
        raise ValueError('probabilities contain NaN or infinity')
    if not windows:
        raise ValueError('at least one c window is required')

    sample_index = np.arange(len(labels))
    true_ce = prob_ce[:, sample_index, labels]
    true_kd = prob_kd[:, sample_index, labels]
    output = {}
    for name, (start, stop) in windows.items():
        if not 0 <= start < stop <= len(prob_ce):
            raise ValueError(f'invalid c window {name}: [{start},{stop})')
        p_ce = true_ce[start:stop]
        p_kd = true_kd[start:stop]
        length = stop - start
        weights = np.arange(length - 1, -1, -1, dtype=np.float64) / length
        distance = np.abs(p_kd - p_ce).sum(axis=0)
        benefit_magnitude = (weights[:, None] * p_kd).sum(axis=0)
        score = distance * benefit_magnitude
        normalized = (distance / length) * (benefit_magnitude / weights.sum())
        output[name] = {
            'D': distance,
            'B': benefit_magnitude,
            'c': score,
            'c_normalized': normalized,
            'weight_sum': float(weights.sum()),
            'window_length': length,
        }
    return output


def late_signed_probability_gain(prob_ce, prob_kd, labels,
                                 start_epoch=81, end_epoch=100):
    """Mean signed true-class probability difference over inclusive epochs."""
    prob_ce = np.asarray(prob_ce, dtype=np.float64)
    prob_kd = np.asarray(prob_kd, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if prob_ce.ndim != 3 or prob_kd.shape != prob_ce.shape:
        raise ValueError('CE/KD probabilities must have matching [epochs,samples,classes] shapes')
    if prob_ce.shape[0] < end_epoch or prob_ce.shape[1] != len(labels):
        raise ValueError('probability dimensions do not cover the requested late window')
    if not 1 <= start_epoch <= end_epoch:
        raise ValueError('late epoch range must be positive and ordered')
    sample_index = np.arange(len(labels))
    delta = (prob_kd[:, sample_index, labels]
             - prob_ce[:, sample_index, labels])
    return delta[start_epoch - 1:end_epoch].mean(axis=0)


def strict_median_groups(values):
    """Split values strictly around the median, exposing ties as their own set."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('median grouping expects a vector')
    if not len(values):
        return {'threshold': float('nan'), 'low': np.zeros(0, dtype=bool),
                'high': np.zeros(0, dtype=bool), 'ties': np.zeros(0, dtype=bool)}
    threshold = float(np.median(values))
    return {'threshold': threshold, 'low': values < threshold,
            'high': values > threshold, 'ties': values == threshold}


def spearman_or_none(left, right):
    """Spearman correlation with constant/undersized inputs reported as None."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError('correlation vectors must have the same one-dimensional shape')
    if len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, None
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None, None
    result = spearmanr(left, right)
    rho, pvalue = float(result.statistic), float(result.pvalue)
    if not np.isfinite(rho) or not np.isfinite(pvalue):
        return None, None
    return rho, pvalue
