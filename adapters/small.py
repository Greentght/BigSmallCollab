"""Adapters for the lightweight specialists: IFNet, ResidualEEGNet, ADFCNN.

All three share one contract in the MIRepNet repo:
    model(x, return_features=True) -> (feat[B,D], logits[B,C])   with x = (B, C, T)
and they consume the canonical raw epoch as-is (no EA / no channel padding — those
are MIRepNet-specific and verified to hurt the baselines). So a single base class
covers them; each concrete adapter only names its constructor.
"""
import torch

from core import paths
from .base import ModelAdapter


class _SmallAdapter(ModelAdapter):
    """Shared logic for (B,C,T) + return_features models."""

    def preprocess(self, X_raw):
        return torch.as_tensor(X_raw, dtype=torch.float32)

    def forward(self, model, x):
        feat, logits = model(x, return_features=True)
        return feat, logits


class IFNetAdapter(_SmallAdapter):
    name = 'ifnet'

    def build(self, num_classes):
        paths.add_repo('mirepnet')
        from model.IFNet import IFNet
        in_ch = self.cfg.get('in_channels')
        samples = self.cfg.get('samples', 1000)
        model = IFNet(in_channels=in_ch, samples=samples, num_classes=num_classes)
        return model.to(self.device)


class EEGNetAdapter(_SmallAdapter):
    name = 'eegnet'

    def build(self, num_classes):
        paths.add_repo('mirepnet')
        from model.ResidualEEGNet import ResidualEEGNet
        in_ch = self.cfg.get('in_channels')
        samples = self.cfg.get('samples', 1000)
        model = ResidualEEGNet(in_channels=in_ch, samples=samples,
                               num_classes=num_classes)
        return model.to(self.device)


class ADFCNNAdapter(_SmallAdapter):
    name = 'adfcnn'

    def build(self, num_classes):
        paths.add_repo('mirepnet')
        from model.ADFCNN import ADFCNN_Net
        in_ch = self.cfg.get('in_channels')
        samples = self.cfg.get('samples', 1000)
        model = ADFCNN_Net(in_channels=in_ch, samples=samples,
                           num_classes=num_classes)
        return model.to(self.device)
