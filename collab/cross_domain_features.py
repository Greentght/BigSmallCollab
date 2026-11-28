"""SemCKD and activation attention transfer for paired full-trial EEG models.

Core equations follow DefangChen/SemCKD (SelfA, AAEmbed, SemCKDLoss) and
szagoruyko/attention-transfer (utils.at_loss). Online detached peer targets,
rectangular EEG maps and padded final-batch Gram rows are explicit adaptations.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


BIG_LAYERS = ('patch_embedding', 'encoder_block4', 'encoder_block8')
SMALL_LAYERS = ('batchnorm1', 'pooling2', 'pooling3')


def match_resolution(source, target):
    """Pool each physical axis to the smaller existing extent, without upsampling."""
    size = tuple(min(a, b) for a, b in zip(source.shape[-2:], target.shape[-2:]))
    return F.adaptive_avg_pool2d(source, size), F.adaptive_avg_pool2d(target, size)


class GramEmbedding(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        dim = batch_size // 4
        if dim < 1:
            raise ValueError('SemCKD batch size must be at least four')
        self.batch_size = batch_size
        self.mlp = nn.Sequential(nn.Linear(batch_size, 2 * dim), nn.ReLU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, feature):
        flat = feature.flatten(1)
        gram = flat @ flat.t()
        if not 1 <= gram.shape[0] <= self.batch_size:
            raise ValueError('Unexpected batch size for SemCKD Gram embedding')
        # Real rows only; pad missing peer columns in a short final minibatch.
        gram = F.pad(gram, (0, self.batch_size - gram.shape[1]))
        return F.normalize(self.mlp(gram), dim=1, eps=1e-12)


class FeatureRegressor(nn.Sequential):
    def __init__(self, source_channels, target_channels):
        hidden = 2 * target_channels
        super().__init__(
            nn.Conv2d(source_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, target_channels, 1, bias=False),
        )


class SemanticDirection(nn.Module):
    """All layer pairs, instance Gram attention and original nonlinear regressors."""
    def __init__(self, source_channels, target_channels, batch_size, seed, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.source_count, self.target_count = len(source_channels), len(target_channels)
        # Independent CPU streams keep model/dropout RNG and the uniform-control
        # regressors identical to those in the learned-association condition.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.regressors = nn.ModuleList([
                nn.ModuleList([FeatureRegressor(a, b) for b in target_channels])
                for a in source_channels
            ])
            self.queries = nn.ModuleList([] if uniform else
                                         [GramEmbedding(batch_size) for _ in source_channels])
            self.keys = nn.ModuleList([] if uniform else
                                      [GramEmbedding(batch_size) for _ in target_channels])

    def forward(self, source, target):
        target = [value.detach() for value in target]
        batch = source[0].shape[0]
        if self.uniform:
            weights = source[0].new_full((batch, self.source_count, self.target_count),
                                        1.0 / self.target_count)
        else:
            query = torch.stack([net(x) for net, x in zip(self.queries, source)], dim=1)
            key = torch.stack([net(x) for net, x in zip(self.keys, target)], dim=2)
            weights = (query @ key).softmax(dim=-1)
        pair_losses = []
        for i, own in enumerate(source):
            row = []
            for j, peer in enumerate(target):
                own_map, peer_map = match_resolution(own, peer)
                row.append((self.regressors[i][j](own_map) - peer_map).square().flatten(1).mean(1))
            pair_losses.append(torch.stack(row, dim=-1))
        pair_losses = torch.stack(pair_losses, dim=1)
        loss = (weights * pair_losses).sum() / (batch * self.source_count)
        return loss, {
            'association': weights.detach().mean(0),
            'association_entropy': -(weights.detach() * weights.detach().clamp_min(1e-12).log()).sum(-1).mean(),
            'pair_mse': pair_losses.detach().mean(0),
        }


class AttentionDirection(nn.Module):
    """Original squared, L2-normalized activation-energy maps, three depth pairs."""
    def forward(self, source, target):
        losses = []
        for own, peer in zip(source, target):
            own, peer = match_resolution(own, peer.detach())
            a = F.normalize(own.square().mean(1).flatten(1), dim=1, eps=1e-12)
            b = F.normalize(peer.square().mean(1).flatten(1), dim=1, eps=1e-12)
            losses.append((a - b).square().mean())
        # The original code sums the layer losses rather than averaging them.
        terms = torch.stack(losses)
        return terms.sum(), {'layer_mse': terms.detach()}


class CrossDomainFeatures(nn.Module):
    def __init__(self, big, small, method, batch_size, seed):
        super().__init__()
        if method not in {'semckd', 'semckd_uniform', 'at'}:
            raise ValueError(f'Unknown feature method: {method}')
        self.method = method
        if method == 'at':
            self.directions = nn.ModuleList([AttentionDirection(), AttentionDirection()])
        else:
            self.directions = nn.ModuleList([
                SemanticDirection([200] * 3, [8, 16, 16], batch_size, seed + 200003,
                                  uniform=method == 'semckd_uniform'),
                SemanticDirection([8, 16, 16], [200] * 3, batch_size, seed + 300007,
                                  uniform=method == 'semckd_uniform'),
            ])
        self.features = {}
        modules = (big.backbone.patch_embedding, big.backbone.encoder.layers[3],
                   big.backbone.encoder.layers[7], small.batchnorm1, small.pooling2, small.pooling3)
        self.handles = [module.register_forward_hook(self._capture(index))
                        for index, module in enumerate(modules)]

    def _capture(self, index):
        def hook(_module, _inputs, output):
            self.features[index] = output
        return hook

    def take(self):
        if set(self.features) != set(range(6)):
            raise RuntimeError('All six features must come from the current full-trial forward')
        big = [self.features[i].permute(0, 3, 1, 2) for i in range(3)]
        small = [self.features[i] for i in range(3, 6)]
        self.clear()
        return big, small

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
