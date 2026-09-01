"""IFNet adapter (lightweight student)."""
from models.base import _SmallAdapter
from .ifnet import IFNet


class IFNetAdapter(_SmallAdapter):
    name = 'ifnet'

    def build(self, num_classes):
        model = IFNet(in_channels=self.cfg.get('in_channels'),
                      samples=self.cfg.get('samples', 1000),
                      num_classes=num_classes)
        return model.to(self.device)


ADAPTER = IFNetAdapter
