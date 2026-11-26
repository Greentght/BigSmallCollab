"""Equation (2) HFD adapted to the native CBraMod and EEGNet operators.

The paper uses the peer's next ViT attention or CNN layer on the aligned
first-stage feature, then matches that output to the peer's native output.
The EEG models need explicit channel/time adapters; neither adapter cuts the
input trial into separate classification examples.
"""
from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


def _frozen_call(module: nn.Module, *args):
    """Differentiate through the input, but not through the peer's parameters.

Clone buffers so the additional CNN pass cannot update the peer's BatchNorm
running statistics. Its training/eval mode remains the native module's mode.
    """
    state = {name: value.detach() for name, value in module.named_parameters()}
    state.update({name: value.detach().clone() for name, value in module.named_buffers()})
    return functional_call(module, state, args, strict=True)


class _CBraAttention(nn.Module):
    """The actual pre-norm, spatial/temporal attention in CBraMod layer 0."""

    def __init__(self, layer):
        super().__init__()
        self.norm = layer.norm1
        self.spatial = layer.self_attn_s
        self.temporal = layer.self_attn_t
        self.dropout = layer.dropout1

    def forward(self, x):
        x = self.norm(x)
        batch, channels, patches, dim = x.shape
        spatial = x[..., :dim // 2].transpose(1, 2).contiguous().view(
            batch * patches, channels, dim // 2)
        temporal = x[..., dim // 2:].contiguous().view(
            batch * channels, patches, dim // 2)
        spatial = self.spatial(spatial, spatial, spatial, need_weights=False)[0]
        temporal = self.temporal(temporal, temporal, temporal, need_weights=False)[0]
        spatial = spatial.contiguous().view(batch, patches, channels, dim // 2).transpose(1, 2)
        temporal = temporal.contiguous().view(batch, channels, patches, dim // 2)
        return self.dropout(torch.cat((spatial, temporal), dim=-1))


class _EEGNetSecondLayer(nn.Module):
    """EEGNet's actual spatial convolution through the second pooling layer."""

    def __init__(self, model):
        super().__init__()
        self.conv = model.conv2
        self.norm = model.batchnorm2
        self.pool = model.pooling2

    def forward(self, x):
        return self.pool(F.elu(self.norm(self.conv(x))))


class CrossOperatorHFD(nn.Module):
    """Bidirectional HFD with native cross-model transformation operators.

    F1^V: CBraMod patch_embedding, [B,3,4,200].
    F2^V: the first native CBraMod attention output, [B,3,4,200].
    F1^C: EEGNet batchnorm1, [B,8,3,1001].
    F2^C: EEGNet pooling2, [B,16,1,250].

    The paper's channel/spatial projection is adapted to three electrodes and
    the native EEG time axes. A 200->64 projection decodes eight time points
    per CBraMod patch into eight CNN channels; interpolation restores the CNN
    feature axis. In the reverse direction 32 time points are sampled and an
    8->25 pointwise projection forms the original 25x8 CBraMod token layout.
    """

    def __init__(self, big: nn.Module, small: nn.Module, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + 1000003)
            self.big_to_small = nn.Linear(200, 64, bias=False)
            self.small_to_big = nn.Linear(8, 25, bias=False)
        # Do not register peer model parameters as part of HFD or checkpoints.
        object.__setattr__(self, '_big_ref', weakref.ref(big))
        object.__setattr__(self, '_small_ref', weakref.ref(small))
        object.__setattr__(self, '_attention', _CBraAttention(big.backbone.encoder.layers[0]))
        object.__setattr__(self, '_cnn_layer', _EEGNetSecondLayer(small))
        self.features = {}
        modules = {
            'big_first': big.backbone.patch_embedding,
            'big_second': big.backbone.encoder.layers[0].dropout1,
            'small_first': small.batchnorm1,
            'small_second': small.pooling2,
        }
        self.handles = [layer.register_forward_hook(self._capture(name))
                        for name, layer in modules.items()]

    def _capture(self, name):
        def hook(_module, _inputs, output):
            self.features[name] = output
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    @staticmethod
    def _indices(device, dtype):
        # EEGNet's even kernel gives feature index 250*t + 0.5.
        positions = (torch.arange(32, device=device, dtype=dtype) / 8) * 250 + 0.5
        left = positions.floor().long()
        return left, positions - left.to(dtype=dtype)

    def cnn_to_vit(self, feature):
        if tuple(feature.shape[1:]) != (8, 3, 1001):
            raise ValueError(f'Unexpected EEGNet feature shape: {feature.shape}')
        batch = feature.shape[0]
        left, frac = self._indices(feature.device, feature.dtype)
        sampled = torch.lerp(feature.index_select(-1, left),
                             feature.index_select(-1, left + 1), frac)
        sampled = sampled.reshape(batch, 8, 3, 4, 8).permute(0, 2, 3, 4, 1)
        projected = self.small_to_big(sampled)
        return projected.permute(0, 1, 2, 4, 3).reshape(batch, 3, 4, 200)

    def vit_to_cnn(self, feature):
        if tuple(feature.shape[1:]) != (3, 4, 200):
            raise ValueError(f'Unexpected CBraMod feature shape: {feature.shape}')
        batch = feature.shape[0]
        projected = self.big_to_small(feature).reshape(batch, 3, 4, 8, 8)
        projected = projected.permute(0, 4, 1, 2, 3).reshape(batch, 8, 3, 32)
        # Grid sample locations refer to 0, 0.125, ... 3.875 seconds.
        # Clamp outside that range instead of introducing future values.
        pos = ((torch.arange(1001, device=feature.device, dtype=feature.dtype) - 0.5)
               * (8.0 / 250.0)).clamp(0, 31)
        left = pos.floor().long()
        right = (left + 1).clamp(max=31)
        frac = pos - left.to(feature.dtype)
        # Repeated gather indices would require a CUDA atomic scatter in the
        # backward pass. A fixed interpolation matrix gives the same linear
        # interpolation and a reproducible matrix-multiplication backward.
        weights = (F.one_hot(left, 32).to(feature.dtype) * (1 - frac[:, None]) +
                   F.one_hot(right, 32).to(feature.dtype) * frac[:, None])
        return torch.matmul(projected, weights.transpose(0, 1))

    def losses(self):
        required = {'big_first', 'big_second', 'small_first', 'small_second'}
        if set(self.features) != required:
            raise RuntimeError('Missing native first/second-stage HFD features')
        big_first = self.features['big_first']
        small_first = self.features['small_first']
        # Take snapshots before cross-calls trigger the same native hooks.
        big_target = self.features['big_second'].detach()
        small_target = self.features['small_second'].detach()
        if tuple(big_target.shape[1:]) != (3, 4, 200) or tuple(small_target.shape[1:]) != (16, 1, 250):
            raise ValueError(f'Unexpected HFD target shapes: {big_target.shape}, {small_target.shape}')
        cross_cnn = _frozen_call(self._cnn_layer, self.vit_to_cnn(big_first))
        cross_attention = _frozen_call(self._attention, self.cnn_to_vit(small_first))
        big_loss = (1 - F.cosine_similarity(cross_cnn, small_target, dim=1, eps=1e-8)).mean()
        small_loss = (1 - F.cosine_similarity(cross_attention, big_target, dim=-1, eps=1e-8)).mean()
        return big_loss, small_loss
