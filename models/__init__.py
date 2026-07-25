"""Framework-owned lightweight specialist models (the KD/ensemble students).

Vendored from MIRepNet's ``model/`` so the small-model definitions live inside
the collaboration framework rather than being imported from an upstream repo.
Each exposes the uniform ``model(x, return_features=True) -> (feat, logits)``
contract that :class:`adapters.small._SmallAdapter` relies on.
"""
from .ifnet import IFNet
from .residual_eegnet import ResidualEEGNet
from .adfcnn import ADFCNN_Net

__all__ = ['IFNet', 'ResidualEEGNet', 'ADFCNN_Net']
