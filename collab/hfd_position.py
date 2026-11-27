"""Cross-operator HFD at shallow and first deep-supervision positions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from collab.hfd_cross_all import CrossOperatorHFDAll, _ConformerAttention
from collab.hfd_cross_operator import _CBraAttention


class EEGNetAfterPooling2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dropout = model.dropout
        self.depthwise = model.conv3_depthwise
        self.pointwise = model.conv3_pointwise
        self.norm = model.batchnorm3
        self.pool = model.pooling3

    def forward(self, x):
        x = self.dropout(x)
        return self.pool(F.elu(self.norm(self.pointwise(self.depthwise(x)))))


class ADFCNNAttention(nn.Module):
    """ADFCNN after spatial_1, retaining the native parallel spatial_2 context."""
    def __init__(self, backbone):
        super().__init__()
        self.query = backbone.w_q
        self.key = backbone.w_k
        self.value = backbone.w_v
        self.dropout = backbone.drop

    def forward(self, first, second):
        x = torch.cat((first, second), dim=3)
        batch, channels, height, width = x.shape
        tokens = x.reshape(batch, channels, height * width).permute(0, 2, 1)
        q = F.normalize(self.query(tokens).permute(0, 2, 1), dim=-1)
        k = F.normalize(self.key(tokens).permute(0, 2, 1), dim=-1)
        v = self.value(tokens).permute(0, 2, 1)
        attention = ((q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)).softmax(dim=-1)
        updated = (attention @ v).reshape(batch, height * width, channels)
        updated = tokens + self.dropout(updated)
        return self.dropout(updated.reshape(batch, height, width, channels).permute(0, 3, 1, 2))


def frozen_call(module, *args):
    state = {name: value.detach().clone() for name, value in module.named_parameters()}
    state.update({name: value.detach().clone() for name, value in module.named_buffers()})
    return functional_call(module, state, args, strict=True)


class PositionHFD(CrossOperatorHFDAll):
    def __init__(self, big_name, small_name, big, small, channels, seed, big_deep, small_deep):
        super().__init__(big_name, small_name, big, small, channels, seed)
        self.close()
        self.big_deep = big_deep
        self.small_deep = small_deep and small_name != 'ifnet'
        if big_name == 'cbramod':
            big_first = big.backbone.encoder.layers[3] if big_deep else big.backbone.patch_embedding
            next_layer = big.backbone.encoder.layers[4 if big_deep else 0]
            big_second, big_operator = next_layer.dropout1, _CBraAttention(next_layer)
        else:
            transformer = big.transformer if big_name == 'mirepnet' else big.encoder.transformer
            big_first = (transformer[1] if big_deep else
                         big.embedding if big_name == 'mirepnet' else big.encoder.patch_embedding)
            attention = transformer[2 if big_deep else 0][0]
            big_second, big_operator = attention.fn[1], _ConformerAttention(attention)
        object.__setattr__(self, '_big_operator', big_operator)

        if self.small_deep and small_name == 'eegnet':
            self.small_channels, self.small_dims = 1, [16]
            small_first, small_second = [small.pooling2], [small.pooling3]
            small_operators = [EEGNetAfterPooling2(small)]
        elif self.small_deep and small_name == 'adfcnn':
            self.small_channels, self.small_dims = 1, [8]
            small_first, small_second = [small.backbone.spatial_1], [small.backbone]
            small_operators = [ADFCNNAttention(small.backbone)]
        else:
            if small_name == 'eegnet':
                from collab.hfd_cross_operator import _EEGNetSecondLayer
                self.small_channels, self.small_dims = channels, [8]
                small_first, small_second = [small.batchnorm1], [small.pooling2]
                small_operators = [_EEGNetSecondLayer(small)]
            elif small_name == 'ifnet':
                from collab.hfd_cross_all import _IFNetNext
                self.small_channels = 1
                self.small_dims = [small.stem.out_planes * small.stem.radix]
                small_first, small_second = [small.stem.sconv], [small.stem.interFre]
                small_operators = [_IFNetNext(small)]
            else:
                self.small_channels, self.small_dims = channels, [8, 8]
                small_first = [small.backbone.spectral_1, small.backbone.spectral_2]
                small_second = [small.backbone.spatial_1, small.backbone.spatial_2]
                small_operators = small_second
        object.__setattr__(self, '_small_operators', small_operators)

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + 1000003)
            if big_name == 'cbramod':
                if self.small_channels == channels:
                    big_shapes = [(self.big_dim, 8 * d) for d in self.small_dims]
                    small_shapes = [(d, 25) for d in self.small_dims]
                else:
                    big_shapes = [(channels * self.big_dim, 8 * d) for d in self.small_dims]
                    small_shapes = [(d, channels * 25) for d in self.small_dims]
            else:
                big_shapes = [(self.big_dim, self.small_channels * d) for d in self.small_dims]
                small_shapes = [(self.small_channels * d, self.big_dim) for d in self.small_dims]
            self.big_to_small = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in big_shapes])
            self.small_to_big = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in small_shapes])
        hooks = [('big_first', big_first), ('big_second', big_second)]
        for i, (first, second) in enumerate(zip(small_first, small_second)):
            hooks.extend([(f'small_first_{i}', first), (f'small_second_{i}', second)])
        if self.small_deep and small_name == 'adfcnn':
            hooks.append(('small_context', small.backbone.spatial_2))
        self.handles = [module.register_forward_hook(self._capture(name)) for name, module in hooks]

    def _capture(self, name):
        def hook(_module, _inputs, output):
            # MIRepNet adds residuals in-place after both attention and block hooks.
            self.features[name] = output.clone() if self.big_name == 'mirepnet' and name.startswith('big_') else output
        return hook

    def _small_canonical(self, feature, branch):
        if self.small_name == 'ifnet':
            return super()._small_canonical(feature, branch)
        if feature.ndim != 4 or tuple(feature.shape[1:3]) != (self.small_dims[branch], self.small_channels):
            raise ValueError(f'Unexpected {self.small_name} position feature: {feature.shape}')
        return feature.permute(0, 2, 3, 1)

    def _small_times(self, branch, length, device):
        if self.small_deep and self.small_name == 'eegnet':
            return (torch.arange(length, device=device, dtype=torch.float32) * 4 + 1.0) / 250
        if self.small_deep and self.small_name == 'adfcnn':
            return (torch.arange(length, device=device, dtype=torch.float32) * 32 + 15.5) / 250
        return super()._small_times(branch, length, device)

    def losses(self):
        keys = {'big_first', 'big_second'} | {
            f'small_{stage}_{i}' for i in range(len(self.small_dims)) for stage in ('first', 'second')}
        if self.small_deep and self.small_name == 'adfcnn':
            keys.add('small_context')
        if set(self.features) != keys:
            raise RuntimeError(f'Missing native features: {keys - set(self.features)}')
        big_first = self.features['big_first']
        big_target = self.features['big_second'].detach()
        first = [self.features[f'small_first_{i}'] for i in range(len(self.small_dims))]
        targets = [self.features[f'small_second_{i}'].detach() for i in range(len(self.small_dims))]
        context = self.features.get('small_context')
        batch = first[0].shape[0]
        big_losses, small_losses = [], []
        for index, feature in enumerate(first):
            small_input = self.big_to_small_input(big_first, index, batch, feature.shape[-1])
            big_input = self.small_to_big_input(feature, index, batch)
            extra = (context.detach(),) if context is not None else ()
            cross_small = frozen_call(self._small_operators[index], small_input, *extra)
            cross_big = frozen_call(self._big_operator, big_input)
            if cross_small.shape != targets[index].shape or cross_big.shape != big_target.shape:
                raise ValueError('Cross-operator outputs differ from native target shapes')
            big_losses.append((1 - F.cosine_similarity(cross_small, targets[index], dim=1, eps=1e-8)).mean())
            small_losses.append((1 - F.cosine_similarity(cross_big, big_target, dim=-1, eps=1e-8)).mean())
        return torch.stack(big_losses).mean(), torch.stack(small_losses).mean()
