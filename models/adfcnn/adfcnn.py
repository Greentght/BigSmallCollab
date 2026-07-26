"""
ADFCNN model adapted for MIRepNet framework.
Follows the same interface as ResidualEEGNet:
    model = ADFCNN_Net(in_channels, samples, num_classes)
    logits = model(x)              # x: (B, C, T) or (B, 1, C, T)
    feat, logits = model(x, return_features=True)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class ActSquare(nn.Module):
    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ADFCNN_Backbone(nn.Module):
    def __init__(self, num_channels, F1=8, D=1, drop_out=0.25):
        super().__init__()
        F2 = F1 * D

        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )

        # Spatial branch 1
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        # Spatial branch 2
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[num_channels, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            nn.AvgPool2d((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),
        )

        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)

        B, N, C = x_attention.shape
        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)
        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        return x


class ADFCNN_Net(nn.Module):
    """
    ADFCNN with the same interface as ResidualEEGNet.
    Input:  (B, C, T) or (B, 1, C, T)
    Output: logits (B, num_classes)
    """
    def __init__(self, in_channels, samples, num_classes, F1=8, D=1, drop_out=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ADFCNN_Backbone(num_channels=in_channels, F1=F1, D=D, drop_out=drop_out)

        # Infer classifier kernel size from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, samples)
            feat = self.backbone(dummy)
            # feat shape: (1, F1*D, 1, time_out)
            self.time_out = feat.shape[3]
            self.feat_size = feat.shape[1] * feat.shape[2] * feat.shape[3]

        F2 = F1 * D
        self.classifier = nn.Sequential(
            nn.Conv2d(F2, num_classes, (1, self.time_out)),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, return_features=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        feat_map = self.backbone(x)
        logits = self.classifier(feat_map)
        logits = logits.squeeze(3).squeeze(2)  # (B, num_classes)

        if return_features:
            feat = feat_map.reshape(feat_map.size(0), -1)
            return feat, logits
        return logits
