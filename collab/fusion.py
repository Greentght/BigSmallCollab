"""Reusable post-hoc fusion utilities for aligned big/small artifacts."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    """Set RNGs used by the fusion head training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def as_feature_matrix(feats):
    """Return features as float32 (N, D), flattening non-batch dims."""
    x = np.asarray(feats, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)


def concat_features(big_feats, small_feats):
    """Concatenate row-aligned big/small features."""
    big = as_feature_matrix(big_feats)
    small = as_feature_matrix(small_feats)
    if len(big) != len(small):
        raise ValueError(
            f'feature length mismatch: big={len(big)} small={len(small)}')
    return np.concatenate([big, small], axis=1).astype(np.float32)


def standardize_fit(x, eps=1e-6):
    """Standardize using train-set statistics."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x, mean, std):
    """Apply train-set standardization statistics to another split."""
    x = np.asarray(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def softmax(logits):
    """Stable numpy softmax over class dimension."""
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def preds_from_probs(probs):
    return np.asarray(probs).argmax(axis=1)


def preds_from_logits(logits):
    return np.asarray(logits).argmax(axis=1)


def accuracy_fraction(logits, y):
    pred = preds_from_logits(logits)
    return float((pred == np.asarray(y)).mean())


def avg_prob_fusion(big_logits, small_logits, big_temperature=1.0,
                    small_temperature=1.0, big_weight=0.5):
    """Fuse temperature-scaled probabilities with a configurable big weight."""
    if float(big_temperature) <= 0 or float(small_temperature) <= 0:
        raise ValueError("probability temperatures must be > 0")
    if not 0.0 <= float(big_weight) <= 1.0:
        raise ValueError("big_weight must be in [0, 1]")
    p_big = softmax(np.asarray(big_logits) / float(big_temperature))
    p_small = softmax(np.asarray(small_logits) / float(small_temperature))
    if p_big.shape != p_small.shape:
        raise ValueError(
            f"probability shape mismatch: big={p_big.shape} small={p_small.shape}")
    return (float(big_weight) * p_big
            + (1.0 - float(big_weight)) * p_small).astype(np.float32)


def gate_conf_acc(big_train_logits, small_train_logits, y_train,
                  big_test_logits, small_test_logits,
                  alpha=1.0, beta=1.0, big_temperature=1.0,
                  small_temperature=1.0, eps=1e-8):
    """Confidence and subject-train-accuracy gated probability fusion.

    Reliability is computed only from the train/support artifact labels. Test
    labels are intentionally not used here.
    """
    p_big = softmax(np.asarray(big_test_logits) / float(big_temperature))
    p_small = softmax(np.asarray(small_test_logits) / float(small_temperature))
    if p_big.shape != p_small.shape:
        raise ValueError(
            f'probability shape mismatch: big={p_big.shape} small={p_small.shape}')

    conf_big = np.maximum(p_big.max(axis=1), eps)
    conf_small = np.maximum(p_small.max(axis=1), eps)
    acc_big = max(accuracy_fraction(big_train_logits, y_train), eps)
    acc_small = max(accuracy_fraction(small_train_logits, y_train), eps)

    score_big = np.power(conf_big, alpha) * np.power(acc_big, beta)
    score_small = np.power(conf_small, alpha) * np.power(acc_small, beta)
    denom = score_big + score_small
    w_big = np.divide(score_big, denom,
                      out=np.full_like(score_big, 0.5, dtype=np.float32),
                      where=denom > eps)
    fused = w_big[:, None] * p_big + (1.0 - w_big[:, None]) * p_small
    info = {
        'train_acc_big_pct': round(acc_big * 100.0, 2),
        'train_acc_small_pct': round(acc_small * 100.0, 2),
        'mean_w_big': round(float(w_big.mean()), 6),
        'std_w_big': round(float(w_big.std()), 6),
    }
    return fused.astype(np.float32), info


class ConcatMLP(nn.Module):
    """Small fusion head trained on frozen, row-aligned artifact features."""

    def __init__(self, input_dim, num_classes, hidden=128, dropout=0.2):
        super().__init__()
        hidden = int(hidden)
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def train_concat_mlp(big_train_feats, small_train_feats, y_train,
                     big_test_feats, small_test_feats, num_classes,
                     seed=0, device='cpu', epochs=100, lr=1e-3,
                     weight_decay=1e-4, batch_size=32, hidden=128,
                     dropout=0.2):
    """Train a concat MLP on train features and return test logits."""
    set_seed(seed)

    z_train = concat_features(big_train_feats, small_train_feats)
    z_test = concat_features(big_test_feats, small_test_feats)
    z_train, mean, std = standardize_fit(z_train)
    z_test = standardize_apply(z_test, mean, std)

    y_arr = np.asarray(y_train, dtype=np.int64)
    if len(z_train) != len(y_arr):
        raise ValueError(
            f'concat train length {len(z_train)} != y length {len(y_arr)}')

    device = torch.device(device)
    model = ConcatMLP(z_train.shape[1], num_classes, hidden=hidden,
                      dropout=dropout).to(device)
    x_tensor = torch.as_tensor(z_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_arr, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=float(weight_decay))
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(z_test, dtype=torch.float32).to(device))
    info = {
        'concat_input_dim': int(z_train.shape[1]),
        'concat_hidden': int(hidden),
    }
    return logits.cpu().numpy().astype(np.float32), info
