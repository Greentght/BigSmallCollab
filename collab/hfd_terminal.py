"""Cross-operator HFD whose targets are the native pre-classification features.

The normal forwards, classification heads, DS heads and native-logit DML stay
intact. The EEG adaptation uses a learned 32-knot temporal bottleneck between
the native patch grid and the CNN grid; knots are adapter coordinates, not
additional classification examples or claimed native transformer samples.
"""
from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F

from collab.hfd_cross_all import frozen_call, resample_time
from collab.hfd_position import EEGNetAfterPooling2


BIG_LAYERS = ('encoder_block11', 'encoder_block12_full_output')
SMALL_LAYERS = ('pooling2', 'pooling3_pre_dropout')


class TerminalDirection(nn.Module):
    def __init__(self, operator, role, channels, seed, device):
        super().__init__()
        self.role, self.channels = role, channels
        object.__setattr__(self, '_operator', operator)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            dims = (channels * 200, 8 * 16) if role == 0 else (8 * 16, channels * 200)
            self.projector = nn.Linear(*dims, bias=False)
        self.generator = torch.Generator(device=device).manual_seed(seed + 500009)

    @staticmethod
    def knots(device):
        # Eight learned reconstruction slots per native one-second patch.
        return (torch.arange(32, device=device, dtype=torch.float32) + .5) / 8

    @staticmethod
    def cnn_times(length, device):
        # Original 64-sample even conv (-0.5), followed by avg-pool(4).
        return (torch.arange(length, device=device, dtype=torch.float32) * 4 + 1) / 250

    def project(self, own, peer_first):
        batch = own.shape[0]
        if self.role == 0:
            if tuple(own.shape[1:]) != (self.channels, 4, 200):
                raise ValueError(f'Unexpected transformer terminal input: {own.shape}')
            x = own.permute(0, 2, 1, 3).reshape(batch, 4, self.channels * 200)
            x = self.projector(x).reshape(batch, 1, 32, 16)
            x = resample_time(x, self.knots(x.device), self.cnn_times(peer_first.shape[-1], x.device))
            return x.permute(0, 3, 1, 2)
        if tuple(own.shape[1:3]) != (16, 1):
            raise ValueError(f'Unexpected CNN terminal input: {own.shape}')
        x = own.permute(0, 2, 3, 1)
        x = resample_time(x, self.cnn_times(x.shape[2], x.device), self.knots(x.device))
        x = self.projector(x.reshape(batch, 4, 8 * 16))
        return x.reshape(batch, 4, self.channels, 200).permute(0, 2, 1, 3)

    def forward(self, source, target):
        # HFD differentiates through the peer operator's input, with frozen
        # parameter copies and cloned buffers. Native peer targets are detached.
        own, peer_first, peer_target = source[0], target[0].detach(), target[1].detach()
        x = self.project(own, peer_first)
        # Extra dropout must neither consume native-forward RNG nor mutate peer
        # BatchNorm buffers. A separately saved stream supports exact resumption.
        with torch.random.fork_rng(devices=[own.device]):
            torch.cuda.set_rng_state(self.generator.get_state(), own.device)
            cross = frozen_call(self._operator, x)
            self.generator.set_state(torch.cuda.get_rng_state(own.device))
        if cross.shape != peer_target.shape:
            raise ValueError(f'Terminal HFD output mismatch: {cross.shape}, {peer_target.shape}')
        feature_axis = 1 if self.role == 0 else -1
        loss = (1 - F.cosine_similarity(cross, peer_target, dim=feature_axis, eps=1e-8)).mean()
        return loss, {'cosine_distance': loss.detach(),
                      'cross_rms': cross.detach().square().mean().sqrt(),
                      'target_rms': peer_target.square().mean().sqrt()}


class TerminalHFD(nn.Module):
    def __init__(self, big, small, method, batch_size, seed):
        super().__init__()
        if method not in {'terminal_hfd_w01', 'terminal_hfd_w1'}:
            raise ValueError(f'Unknown terminal HFD condition: {method}')
        channels = big.classifier.in_features // (4 * 200)
        if big.feature_head != 'flatten' or channels not in (3, 22):
            raise ValueError('This pilot requires the matched native CBraMod flatten head')
        device = next(big.parameters()).device
        if device.type != 'cuda':
            raise ValueError('Terminal HFD uses a separately checkpointed CUDA dropout stream')
        self.directions = nn.ModuleList([
            TerminalDirection(EEGNetAfterPooling2(small), 0, channels, seed + 400009, device),
            TerminalDirection(big.backbone.encoder.layers[11], 1, channels, seed + 500009, device),
        ])
        self.features = {}
        object.__setattr__(self, '_small_ref', weakref.ref(small))
        modules = (big.backbone.encoder.layers[10], big.backbone.encoder.layers[11],
                   small.pooling2, small.pooling3)
        self.handles = [module.register_forward_hook(self._capture(i)) for i, module in enumerate(modules)]

    def _capture(self, index):
        def hook(_module, _inputs, output):
            self.features[index] = output
        return hook

    def take(self):
        if set(self.features) != set(range(4)):
            raise RuntimeError('All four terminal features must come from the same native forward')
        result = ([self.features[0], self.features[1]], [self.features[2], self.features[3]])
        self.clear()
        return result

    def rng_states(self):
        return [direction.generator.get_state() for direction in self.directions]

    def set_rng_states(self, states):
        for direction, state in zip(self.directions, states): direction.generator.set_state(state)

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        self.clear()
