"""Test-time ensemble over cached logits (no model needed — pure arrays).

Ports the three strategies from MIRepNet's ``utils/utils.py``; the confidence-
gated rule is the confirmed best (see the Ensemble Strategies memory). Operating
on cached ``logits`` lets big + small models that live in incompatible conda
envs be fused after the fact.

Convention: a "big" side (one foundation model, or the mean of several) gates how
much to trust the averaged "small" side. ``num_classes`` is inferred from logits.
"""
import numpy as np


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gate(big_logits, small_logits_list):
    """Confidence-gated fusion (best strategy).

        a = clamp((conf_big - 1/C) / (1 - 1/C), 0, 1)
        p = a * p_big + (1 - a) * mean(p_small_i)

    ``big_logits`` may be a single array or a list (averaged into the big side).
    Returns fused probabilities ``(N, C)``.
    """
    p_big = _mean_probs(big_logits)
    C = p_big.shape[1]
    conf_big = p_big.max(axis=1, keepdims=True)
    a = np.clip((conf_big - 1.0 / C) / (1.0 - 1.0 / C), 0.0, 1.0)
    p_small = _mean_probs(small_logits_list)
    return a * p_big + (1.0 - a) * p_small


def conf_weighted(logits_list):
    """Per-sample confidence-weighted fusion across all models (ablation)."""
    probs = [_softmax(np.asarray(l)) for l in logits_list]
    confs = [p.max(axis=1, keepdims=True) for p in probs]   # each (N,1)
    w = np.concatenate(confs, axis=1)                        # (N, M)
    w = w / w.sum(axis=1, keepdims=True)
    out = np.zeros_like(probs[0])
    for i, p in enumerate(probs):
        out += w[:, i:i + 1] * p
    return out


def voting(logits_list, tie_breaker=0):
    """Hard majority vote; ties broken by model index ``tie_breaker`` (default
    the first, conventionally the big model)."""
    preds = np.stack([np.asarray(l).argmax(1) for l in logits_list], axis=0)  # (M,N)
    C = np.asarray(logits_list[0]).shape[1]
    N = preds.shape[1]
    out = np.empty(N, dtype=np.int64)
    for n in range(N):
        counts = np.bincount(preds[:, n], minlength=C)
        top = np.flatnonzero(counts == counts.max())
        out[n] = preds[tie_breaker, n] if len(top) > 1 else top[0]
    return out


def _softmax_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _mean_probs(logits):
    arrs = _softmax_list(logits)
    return np.mean([_softmax(np.asarray(a)) for a in arrs], axis=0)
