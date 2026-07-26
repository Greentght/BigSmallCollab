"""ADFCNN adapter (lightweight student)."""
from models.base import _SmallAdapter
from .adfcnn import ADFCNN_Net


class ADFCNNAdapter(_SmallAdapter):
    name = 'adfcnn'

    def build(self, num_classes):
        model = ADFCNN_Net(in_channels=self.cfg.get('in_channels'),
                           samples=self.cfg.get('samples', 1000),
                           num_classes=num_classes)
        return model.to(self.device)
