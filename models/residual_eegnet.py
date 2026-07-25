import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualEEGNet(nn.Module):
    def __init__(self, in_channels=49, samples=1000, num_classes=2, dropoutRate=0.25, kernLength=64):
        super(ResidualEEGNet, self).__init__()
        self.num_classes = num_classes

        # EEGNet-8,2
        F1 = 8
        D = 2
        F2 = F1 * D

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling2 = nn.AvgPool2d((1, 4))

        # Block 2: SeparableConv2D = depthwise temporal conv + pointwise conv
        self.conv3_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv3_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling3 = nn.AvgPool2d((1, 8))

        self.dropout = nn.Dropout(dropoutRate)

        self.flatten_size = self._infer_flatten_size(in_channels, samples)
        self.fc1 = nn.Linear(self.flatten_size, num_classes)

    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
        x = self.dropout(x)
        return x

    def _infer_flatten_size(self, in_channels, samples):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, samples)
            feat = self._forward_features(dummy)
            return feat.reshape(1, -1).size(1)

    def forward(self, x, return_features=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self._forward_features(x)
        feat = x.reshape(x.size(0), -1)
        logits = self.fc1(feat)
        if return_features:
            return feat, logits
        return logits