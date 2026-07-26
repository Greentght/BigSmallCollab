"""Framework-owned EEG preprocessing primitives.

Vendored from MIRepNet's ``utils/utils.py`` (only the two transforms the adapters
actually use) so the framework no longer reaches into that repo for its data
layer. These operate on raw numpy epochs ``(N, C, T)``:

  - ``EA`` — Euclidean Alignment whitening (per-set reference covariance), the
    MIRepNet-signature transductive alignment.
  - ``pad_missing_channels_diff`` — inverse-distance channel remap onto a target
    montage (used to pad a dataset's native channels up to the 45-ch template).
  - ``bandpass`` / ``notch`` — zero-phase (filtfilt) Butterworth band-pass and
    IIR notch, shared by the foundation-model adapters/scripts (CBraMod / LaBraM).

Channel names / scalp positions live in :mod:`core.channels`.
"""
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, filtfilt, iirnotch
from scipy.spatial.distance import cdist

from data.channels import channel_positions

# Canonical raw-epoch sample rate, and the resample target the transformer
# foundation models (CBraMod / LaBraM) expect. Shared so the adapters don't each
# re-declare them.
SRC_FS = 250
DST_FS = 200


def EA(x):
    """Euclidean Alignment: whiten each trial by the set's mean covariance^{-1/2}.

    ``x`` is ``(N, C, T)``; returns the aligned array of the same shape. This is
    transductive (uses all N trials' covariance), so it must be applied per set
    (per subject-group), matching MIRepNet's ``process_and_replace_loader``.
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def pad_missing_channels_diff(x, target_channels, actual_channels):
    """Remap ``x`` (N, C, T) from ``actual_channels`` onto ``target_channels``.

    A present target channel is copied through; a missing one is inverse-distance
    interpolated from the actual channels' scalp positions. Returns
    ``(N, len(target_channels), T)``.
    """
    B, C, T = x.shape
    num_target = len(target_channels)
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])
    target_pos = np.array([channel_positions[ch] for ch in target_channels])
    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)
            weights /= weights.sum()
            W[i] = weights
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]
    return padded


def bandpass(x, fs, l_freq, h_freq):
    """Zero-phase order-4 Butterworth band-pass along the last axis. ``l_freq<=0``
    with ``h_freq>=Nyquist`` is a no-op; ``l_freq<=0`` alone gives a low-pass."""
    nyq = fs / 2.0
    high = min(h_freq, nyq - 1e-3)
    if l_freq <= 0 and high >= nyq:
        return x
    if l_freq <= 0:
        b, a = butter(4, high / nyq, btype='low')
    else:
        b, a = butter(4, [l_freq / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)


def notch(x, fs, notch_freq):
    """Zero-phase IIR notch (Q=30) at ``notch_freq`` along the last axis. No-op if
    the frequency is unset or outside (0, Nyquist)."""
    if notch_freq is None or notch_freq <= 0 or notch_freq >= fs / 2.0:
        return x
    b, a = iirnotch(w0=notch_freq, Q=30, fs=fs)
    return filtfilt(b, a, x, axis=-1)
