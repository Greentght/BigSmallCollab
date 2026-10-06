"""IFNet adapter (lightweight student)."""
import numpy as np
import torch
from models.base import _SmallAdapter
from .ifnet import IFNet, apply_filter_bank


class IFNetAdapter(_SmallAdapter):
    name = 'ifnet'

    def preprocess(self, X_raw):
        if self.cfg.get('use_filter_bank', False):
            # 双频：滤波成 2 个频段(4-16 / 16-40 Hz)并沿通道拼接 -> C*2。
            # 所有数据集到这一步都是 250Hz(005/ALEXMI 已在 eeg_dataset 里重采样)。
            x = np.asarray(X_raw, dtype=np.float64)
            x = apply_filter_bank(x, fs=self.cfg.get('sample_rate', 250))
            return torch.as_tensor(x, dtype=torch.float32)
        return super().preprocess(X_raw)

    def build(self, num_classes):
        model = IFNet(in_channels=self.cfg.get('in_channels'),
                      samples=self.cfg.get('samples', 1000),
                      num_classes=num_classes,
                      use_filter_bank=self.cfg.get('use_filter_bank', False))
        return model.to(self.device)


ADAPTER = IFNetAdapter
