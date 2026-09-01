"""ResidualEEGNet adapter (lightweight student)."""
from models.base import _SmallAdapter
from .residual_eegnet import ResidualEEGNet


class EEGNetAdapter(_SmallAdapter):
    name = 'eegnet'

    def build(self, num_classes):
        model = ResidualEEGNet(in_channels=self.cfg.get('in_channels'),
                               samples=self.cfg.get('samples', 1000),
                               num_classes=num_classes)
        return model.to(self.device)


ADAPTER = EEGNetAdapter
