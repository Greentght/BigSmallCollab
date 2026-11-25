"""Shallow, time-aligned feature exchange between two EEG models."""

import torch
from torch import nn
from torch.nn import functional as F


class ShallowHFD(nn.Module):
    """Capture each model's features during its normal training forward."""

    DIMENSIONS = {'mirepnet_ifnet': (128, 64), 'cbramod_eegnet': (25, 8)}

    def __init__(self, pair, big, small, seed):
        super().__init__()
        if pair not in self.DIMENSIONS:
            raise ValueError(f'Unsupported HFD pair: {pair}')
        self.pair = pair
        self.features = {}
        big_dim, small_dim = self.DIMENSIONS[pair]
        # Do not advance the models' dropout or sampling streams.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also resets CUDA generators, which would make
            # dropout differ between the HFD and non-HFD conditions.  Only the
            # CPU generator is needed to initialize these CPU linear layers.
            torch.random.default_generator.manual_seed(int(seed) + 1000003)
            self.big_to_small = nn.Linear(big_dim, small_dim, bias=False)
            self.small_to_big = nn.Linear(small_dim, big_dim, bias=False)
        if pair == 'mirepnet_ifnet':
            layers = (big.embedding.pool, small.stem.interFre)
        else:
            layers = (big.backbone.patch_embedding.proj_in, small.batchnorm1)
        self.handles = [layer.register_forward_hook(self._capture(role))
                        for role, layer in zip(('big', 'small'), layers)]

    def _capture(self, role):
        def hook(_module, _inputs, output):
            self.features[role] = output
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def aligned(self):
        if set(self.features) != {'big', 'small'}:
            raise RuntimeError('Missing shallow features from current paired forward')
        big, small = self.features['big'], self.features['small']
        batch = big.shape[0]
        if self.pair == 'mirepnet_ifnet':
            if tuple(big.shape[1:]) != (128, 1, 61) or tuple(small.shape[1:]) != (64, 1000):
                raise ValueError(f'Unexpected HFD shapes: {big.shape}, {small.shape}')
            big = big.squeeze(2).transpose(1, 2)
            small = F.avg_pool1d(small, kernel_size=99, stride=15).transpose(1, 2)
        else:
            if tuple(big.shape[1:]) != (25, 12, 8) or tuple(small.shape[1:]) != (8, 3, 1001):
                raise ValueError(f'Unexpected HFD shapes: {big.shape}, {small.shape}')
            big = big.reshape(batch, 25, 3, 4, 8).permute(0, 2, 3, 4, 1)
            time = (torch.arange(4, device=small.device)[:, None] +
                    torch.arange(8, device=small.device)[None, :] / 8) * 250 + 0.5
            left = time.floor().long().flatten()
            fraction = (time.flatten() - left).to(dtype=small.dtype)
            small = torch.lerp(small.index_select(-1, left),
                               small.index_select(-1, left + 1), fraction)
            small = small.reshape(batch, 8, 3, 4, 8).permute(0, 2, 3, 4, 1)
        return big, small

    def losses(self):
        big, small = self.aligned()
        big_loss = (1 - F.cosine_similarity(
            self.big_to_small(big), small.detach(), dim=-1, eps=1e-8)).mean()
        small_loss = (1 - F.cosine_similarity(
            self.small_to_big(small), big.detach(), dim=-1, eps=1e-8)).mean()
        return big_loss, small_loss
