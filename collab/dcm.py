"""Training-time auxiliary classifiers and dense mutual distillation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AuxiliaryClassifier(nn.Module):
    """Project temporal features to class logits while retaining temporal bins."""

    def __init__(self, channels: int, classes: int, bins: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(bins)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * bins, 128), nn.GELU(),
            nn.Linear(128, classes),
        )

    def forward(self, features):
        return self.classifier(self.pool(features))


class DCMHeads(nn.Module):
    """Capture two intermediate features from an existing model's forward pass."""

    def __init__(self, model: nn.Module, architecture: str, classes: int):
        super().__init__()
        self.architecture = architecture
        self.features = {}
        if architecture == 'cbramod':
            layers = [model.backbone.encoder.layers[3], model.backbone.encoder.layers[7]]
            channels = [200, 200]
        elif architecture == 'neurogpt':
            layers = [model.encoder.transformer[1], model.encoder.transformer[3]]
            channels = [40, 40]
        elif architecture == 'ifnet':
            layers = [model.stem.sconv, model.stem.interFre]
            channels = [model.stem.out_planes * model.stem.radix, model.stem.out_planes]
        elif architecture == 'eegnet':
            layers = [model.pooling2, model.pooling3]
            channels = [16, 16]
        else:
            raise ValueError(f'Unsupported architecture: {architecture}')
        self.heads = nn.ModuleList(AuxiliaryClassifier(c, classes) for c in channels)
        self.handles = [layer.register_forward_hook(self._capture(index))
                        for index, layer in enumerate(layers)]

    def _capture(self, index):
        def hook(_module, _inputs, output):
            self.features[index] = output
        return hook

    def forward(self, batch_size: int):
        if set(self.features) != {0, 1}:
            raise RuntimeError('Both auxiliary features must be captured by the current forward')
        logits = []
        for index, head in enumerate(self.heads):
            x = self.features[index]
            if self.architecture == 'cbramod':
                x = x.mean(dim=1).transpose(1, 2)
            elif self.architecture == 'neurogpt':
                x = x.reshape(batch_size, 2, x.shape[1], x.shape[2])
                x = x.reshape(batch_size, -1, x.shape[-1]).transpose(1, 2)
            elif self.architecture == 'eegnet':
                x = x.squeeze(2)
            logits.append(head(x))
        self.features.clear()
        return logits

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.features.clear()


def mutual_losses(outputs, labels, condition, temperature=2.0, alpha=0.5, weight=0.25):
    """Return independent directional objectives, with detached peer targets."""
    if condition not in {'none', 'dml_kl', 'dml_ds', 'dcm'}:
        raise ValueError(f'Unsupported DCM condition: {condition}')
    expected = 3 if condition in {'dml_ds', 'dcm'} else 1
    if len(outputs) != 2 or any(len(side) != expected for side in outputs):
        raise ValueError('Wrong number of classifier outputs for condition')
    losses = []
    for side in range(2):
        own, peer = outputs[side], outputs[1 - side]
        loss = F.cross_entropy(own[0], labels)
        if expected == 3:
            loss = loss + alpha * torch.stack([
                F.cross_entropy(logits, labels) for logits in own[1:]
            ]).mean()
        if condition != 'none':
            own_heads = own if condition == 'dcm' else own[:1]
            peer_heads = peer if condition == 'dcm' else peer[:1]
            terms = [F.kl_div(F.log_softmax(a / temperature, dim=-1),
                              F.softmax(b.detach() / temperature, dim=-1),
                              reduction='batchmean')
                     for a in own_heads for b in peer_heads]
            loss = loss + weight * temperature ** 2 * torch.stack(terms).mean()
        losses.append(loss)
    return losses
