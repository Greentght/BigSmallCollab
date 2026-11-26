"""Paper Eq. (2) cross-operator HFD for the project's nine model pairs.

Model-specific views expose an actual first-stage feature and the output of
that model's next attention/CNN operator. Trainable 1x1 linear maps and
physical-time interpolation adapt their different EEG dimensions. The two
ADFCNN spectral branches both participate and their losses are averaged.
"""
from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from collab.hfd_cross_operator import _CBraAttention, _EEGNetSecondLayer


def frozen_call(module, x):
    """Freeze the peer and copy buffers/weights that its forward may mutate."""
    state = {name: value.detach().clone() for name, value in module.named_parameters()}
    state.update({name: value.detach().clone() for name, value in module.named_buffers()})
    return functional_call(module, state, (x,), strict=True)


def resample_time(x, source, destination):
    """Piecewise linear sampling of [B,C,T,D] on explicit time coordinates.

    Dense fixed weights give a deterministic CUDA backward for repeated
    interpolation indices. Coordinates beyond the source are clamped.
    """
    source = source.to(device=x.device, dtype=x.dtype)
    destination = destination.to(device=x.device, dtype=x.dtype).clamp(source[0], source[-1])
    right = torch.searchsorted(source, destination).clamp(1, len(source) - 1)
    left = right - 1
    fraction = (destination - source[left]) / (source[right] - source[left])
    weights = (F.one_hot(left, len(source)).to(dtype=x.dtype) * (1 - fraction[:, None]) +
               F.one_hot(right, len(source)).to(dtype=x.dtype) * fraction[:, None])
    return torch.matmul(x.transpose(-1, -2), weights.transpose(0, 1)).transpose(-1, -2)


class _ConformerAttention(nn.Module):
    """The native LayerNorm + attention used by MIRepNet or NeuroGPT."""
    def __init__(self, residual_attention):
        super().__init__()
        self.norm = residual_attention.fn[0]
        self.attention = residual_attention.fn[1]

    def forward(self, x):
        return self.attention(self.norm(x))


class _IFNetNext(nn.Module):
    """IFNet's native frequency branches, fusion and GELU."""
    def __init__(self, model):
        super().__init__()
        self.tconv = model.stem.tconv
        self.interfre = model.stem.interFre
        self.width = model.stem.out_planes

    def forward(self, x):
        parts = torch.split(x, self.width, dim=1)
        if len(parts) != len(self.tconv):
            raise ValueError('IFNet branch count differs from its native stem')
        return self.interfre([layer(part) for layer, part in zip(self.tconv, parts)])


class CrossOperatorHFDAll(nn.Module):
    """Cross attention/CNN operators for one big model and one small model."""

    def __init__(self, big_name, small_name, big, small, channels, seed):
        super().__init__()
        if big_name not in {'mirepnet', 'cbramod', 'neurogpt'}:
            raise ValueError(f'Unsupported big model: {big_name}')
        if small_name not in {'ifnet', 'eegnet', 'adfcnn'}:
            raise ValueError(f'Unsupported small model: {small_name}')
        self.big_name, self.small_name, self.channels = big_name, small_name, int(channels)
        object.__setattr__(self, '_big_ref', weakref.ref(big))
        object.__setattr__(self, '_small_ref', weakref.ref(small))
        self.features = {}
        if big_name == 'cbramod':
            big_first = big.backbone.patch_embedding
            first_layer = big.backbone.encoder.layers[0]
            big_second = first_layer.dropout1
            big_operator = _CBraAttention(first_layer)
            self.big_dim = 200
        elif big_name == 'mirepnet':
            big_first = big.embedding
            attention = big.transformer[0][0]
            big_second = attention.fn[1]
            big_operator = _ConformerAttention(attention)
            self.big_dim = big.embedding.projection[0].out_channels
        else:
            big_first = big.encoder.patch_embedding
            attention = big.encoder.transformer[0][0]
            big_second = attention.fn[1]
            big_operator = _ConformerAttention(attention)
            self.big_dim = big.encoder.patch_embedding.projection[0].out_channels
        object.__setattr__(self, '_big_operator', big_operator)

        if small_name == 'eegnet':
            self.small_channels = self.channels
            self.small_dims = [8]
            small_first = [small.batchnorm1]
            small_second = [small.pooling2]
            small_operators = [_EEGNetSecondLayer(small)]
        elif small_name == 'ifnet':
            self.small_channels = 1  # IFNet's native stem already mixes electrodes.
            self.small_dims = [small.stem.out_planes * small.stem.radix]
            small_first = [small.stem.sconv]
            small_second = [small.stem.interFre]
            small_operators = [_IFNetNext(small)]
        else:
            self.small_channels = self.channels
            self.small_dims = [8, 8]
            small_first = [small.backbone.spectral_1, small.backbone.spectral_2]
            small_second = [small.backbone.spatial_1, small.backbone.spatial_2]
            small_operators = small_second
        object.__setattr__(self, '_small_operators', small_operators)

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + 1000003)
            if big_name == 'cbramod':
                # Preserve per-electrode correspondence where both models
                # still have an electrode axis; IFNet needs a learned map
                # because its first layer has already mixed the electrodes.
                if self.small_channels == self.channels:
                    big_shapes = [(self.big_dim, 8 * d) for d in self.small_dims]
                    small_shapes = [(d, 25) for d in self.small_dims]
                else:
                    big_shapes = [(self.channels * self.big_dim, 8 * d) for d in self.small_dims]
                    small_shapes = [(d, self.channels * 25) for d in self.small_dims]
            else:
                big_shapes = [(self.big_dim, self.small_channels * d) for d in self.small_dims]
                small_shapes = [(self.small_channels * d, self.big_dim) for d in self.small_dims]
            self.big_to_small = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in big_shapes])
            self.small_to_big = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in small_shapes])

        hooks = [('big_first', big_first), ('big_second', big_second)]
        for i, (first, second) in enumerate(zip(small_first, small_second)):
            hooks.extend([(f'small_first_{i}', first), (f'small_second_{i}', second)])
        self.handles = [module.register_forward_hook(self._capture(name)) for name, module in hooks]

    def _capture(self, name):
        def hook(_module, _inputs, output):
            # MIRepNet's residual block performs an in-place addition after
            # the attention output. Preserve the actual pre-residual target.
            self.features[name] = (output.clone() if name == 'big_second' and
                                   self.big_name == 'mirepnet' else output)
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def _big_times(self, device):
        if self.big_name == 'cbramod':
            return torch.arange(32, device=device, dtype=torch.float32) / 8
        if self.big_name == 'mirepnet':
            return (torch.arange(61, device=device, dtype=torch.float32) * 15 + 49) / 250
        times = (torch.arange(27, device=device, dtype=torch.float32) * 15 + 49) / 250
        return torch.cat((times, times + 2))

    def _small_times(self, branch, length, device):
        if self.small_name == 'eegnet':
            offset = -0.5
        elif self.small_name == 'adfcnn' and branch == 1:
            offset = 0.5  # even 30-sample 'same' convolution
        else:
            offset = 0.0
        return (torch.arange(length, device=device, dtype=torch.float32) + offset) / 250

    def _small_canonical(self, feature, branch):
        if self.small_name == 'ifnet':
            if feature.ndim != 3 or feature.shape[1] != self.small_dims[branch]:
                raise ValueError(f'Unexpected IFNet feature shape: {feature.shape}')
            return feature.transpose(1, 2).unsqueeze(1)
        if feature.ndim != 4 or feature.shape[1:3] != (self.small_dims[branch], self.channels):
            raise ValueError(f'Unexpected {self.small_name} feature shape: {feature.shape}')
        return feature.permute(0, 2, 3, 1)

    def _big_canonical(self, feature, batch):
        if self.big_name == 'cbramod':
            if tuple(feature.shape) != (batch, self.channels, 4, self.big_dim):
                raise ValueError(f'Unexpected CBraMod feature shape: {feature.shape}')
            return feature
        expected = ((batch, 61, self.big_dim) if self.big_name == 'mirepnet'
                    else (batch * 2, 27, self.big_dim))
        if tuple(feature.shape) != expected:
            raise ValueError(f'Unexpected {self.big_name} feature shape: {feature.shape}')
        return feature.reshape(batch, -1, self.big_dim)

    def small_to_big_input(self, feature, branch, batch):
        value = self._small_canonical(feature, branch)
        value = resample_time(value, self._small_times(branch, value.shape[2], value.device),
                              self._big_times(value.device))
        projector = self.small_to_big[branch]
        if self.big_name == 'cbramod':
            if self.small_channels == self.channels:
                value = projector(value)  # [B,C,32,25]
            else:
                value = projector(value.squeeze(1)).reshape(batch, 32, self.channels, 25)
                value = value.permute(0, 2, 1, 3)
            return value.reshape(batch, self.channels, 4, 8, 25).permute(
                0, 1, 2, 4, 3).reshape(batch, self.channels, 4, 200)
        value = value.permute(0, 2, 1, 3).reshape(batch, -1, self.small_channels * self.small_dims[branch])
        value = projector(value)
        if self.big_name == 'neurogpt':
            value = value.reshape(batch * 2, 27, self.big_dim)
        return value

    def big_to_small_input(self, feature, branch, batch, small_length):
        big = self._big_canonical(feature, batch)
        dim = self.small_dims[branch]
        projector = self.big_to_small[branch]
        if self.big_name == 'cbramod':
            if self.small_channels == self.channels:
                value = projector(big).reshape(batch, self.channels, 32, dim)
            else:
                big = big.permute(0, 2, 1, 3).reshape(batch, 4, self.channels * self.big_dim)
                value = projector(big).reshape(batch, 1, 32, dim)
        else:
            value = projector(big).reshape(batch, -1, self.small_channels, dim).permute(0, 2, 1, 3)
        value = resample_time(value, self._big_times(value.device),
                              self._small_times(branch, small_length, value.device))
        if self.small_name == 'ifnet':
            return value.squeeze(1).transpose(1, 2)
        return value.permute(0, 3, 1, 2)

    def losses(self):
        keys = {'big_first', 'big_second'} | {
            f'small_{stage}_{i}' for i in range(len(self.small_dims)) for stage in ('first', 'second')}
        if set(self.features) != keys:
            raise RuntimeError(f'Missing native HFD features: {keys - set(self.features)}')
        # Snapshot all targets before calling operators that trigger hooks.
        big_first = self.features['big_first']
        big_target = self.features['big_second'].detach()
        first = [self.features[f'small_first_{i}'] for i in range(len(self.small_dims))]
        target = [self.features[f'small_second_{i}'].detach() for i in range(len(self.small_dims))]
        batch = first[0].shape[0]
        self._big_canonical(big_first, batch)
        big_losses, small_losses = [], []
        for i, feature in enumerate(first):
            small_in = self.big_to_small_input(big_first, i, batch, feature.shape[-1])
            big_in = self.small_to_big_input(feature, i, batch)
            cross_cnn = frozen_call(self._small_operators[i], small_in)
            cross_attention = frozen_call(self._big_operator, big_in)
            if cross_cnn.shape != target[i].shape or cross_attention.shape != big_target.shape:
                raise ValueError(f'Cross-operator output mismatch: {cross_cnn.shape}/{target[i].shape}, '
                                 f'{cross_attention.shape}/{big_target.shape}')
            big_losses.append((1 - F.cosine_similarity(cross_cnn, target[i], dim=1, eps=1e-8)).mean())
            small_losses.append((1 - F.cosine_similarity(cross_attention, big_target, dim=-1, eps=1e-8)).mean())
        return torch.stack(big_losses).mean(), torch.stack(small_losses).mean()
