"""
IFNet V2 adapted for MIRepNet framework.
Same interface as ResidualEEGNet:
    model = IFNet(in_channels, samples, num_classes)
    logits = model(x)              # x: (B, C, T) or (B, 1, C, T)
    feat, logits = model(x, return_features=True)

Note: IFNet expects filter-bank input. If use_filter_bank=True (default),
      it applies 2-band bandpass filtering internally, doubling channel count.
      If use_filter_bank=False, raw input is used directly.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt


# ======================== Weight Constraint Layers ========================

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


# ======================== Building Blocks ========================

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))


class InterFre(nn.Module):
    def forward(self, x):
        return F.gelu(sum(x))


# ======================== Stem ========================

class Stem(nn.Module):
    def __init__(self, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        super().__init__()
        self.out_planes = out_planes
        self.patch_size = patch_size
        self.radix = radix

        mid_planes = out_planes * radix
        self.sconv = Conv(
            nn.Conv1d(in_planes, mid_planes, 1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(mid_planes))

        self.tconv = nn.ModuleList()
        ks = kernel_size
        for _ in range(radix):
            self.tconv.append(Conv(
                nn.Conv1d(out_planes, out_planes, ks, 1,
                          groups=out_planes, padding=ks // 2, bias=False),
                bn=nn.BatchNorm1d(out_planes)))
            ks //= 2

        self.interFre = InterFre()
        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def _band_power(self, b, N, T):
        """Per-band pooled feature: same power+pool as the fused path, applied to
        one frequency branch -> (N, out_planes * n_patch)."""
        bb = b.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        return self.power(bb).flatten(1)

    def forward(self, x, return_bands=False):
        N, C, T = x.shape
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(xi) for xi, m in zip(out, self.tconv)]  # per-band [N, out_planes, T]
        bands = [self._band_power(b, N, T) for b in out] if return_bands else None
        out = self.interFre(out)
        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        if return_bands:
            return out, bands
        return out


# ======================== Filter Bank ========================

def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')


def apply_filter_bank(x_np, fs=250, bands=((4, 16), (16, 40))):
    """
    Apply bandpass filter bank and concatenate along channel dim.
    x_np: (N, C, T) numpy array
    Returns: (N, C*len(bands), T) numpy array
    """
    filtered = []
    for low, high in bands:
        b, a = _butter_bandpass(low, high, fs)
        # filtfilt along time axis
        filt = filtfilt(b, a, x_np, axis=2).astype(np.float32)
        filtered.append(filt)
    return np.concatenate(filtered, axis=1)


# ======================== IFNet Main ========================

class IFNet(nn.Module):
    """
    IFNet V2 with same interface as ResidualEEGNet.
    Input:  (B, C, T) or (B, 1, C, T)
    Output: logits (B, num_classes)

    Args:
        in_channels: number of EEG channels (e.g. 22)
        samples: time points (e.g. 750)
        num_classes: number of classes
        embed_dim: feature dimension (default 64)
        kernel_size: temporal conv kernel (default 63)
        patch_size: temporal pooling size (default 125)
        radix: number of frequency bands (default 2)
        use_filter_bank: if True, apply 2-band filter bank, doubling channels
    """
    def __init__(self, in_channels, samples, num_classes,
                 embed_dim=64, kernel_size=63, patch_size=125, radix=2,
                 use_filter_bank=False):
        super().__init__()
        self.use_filter_bank = use_filter_bank
        self.num_classes = num_classes

        if use_filter_bank:
            actual_channels = in_channels * radix
            actual_radix = radix
        else:
            actual_channels = in_channels
            # radix must divide in_channels; fall back to 1 if not
            actual_radix = radix if in_channels % radix == 0 else 1

        self.radix = actual_radix
        self.stem = Stem(actual_channels, embed_dim, kernel_size,
                         patch_size=patch_size, radix=actual_radix)

        feat_dim = embed_dim * (samples // patch_size)
        self.feat_dim = feat_dim
        self.fc = LinearWithConstraint(feat_dim, num_classes, max_norm=0.5)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False, return_bands=False):
        # Handle 4D input (B, 1, C, T) -> (B, C, T)
        if len(x.shape) == 4:
            x = x.squeeze(1)

        if return_bands:
            out, bands = self.stem(x, return_bands=True)  # bands: list of per-band feats
            feat = out.flatten(1)
            logits = self.fc(feat)
            return feat, logits, bands

        out = self.stem(x)              # (B, embed_dim, T//patch_size)
        feat = out.flatten(1)           # (B, feat_dim)
        logits = self.fc(feat)          # (B, num_classes)

        if return_features:
            return feat, logits
        return logits
