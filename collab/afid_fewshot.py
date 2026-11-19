"""Training-only feature interaction for heterogeneous EEG model pairs.

Each model keeps its original classifier. The fusion head and attention gates
are auxiliary components, so the student can be evaluated on its own.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AFIDFeatureFusion(nn.Module):
    """Capture two feature depths and train a cross- or self-gated fusion head."""

    def __init__(self, teacher_model, teacher_name, student_model, student_name,
                 mode='cross', bins=4, dim=32):
        super().__init__()
        if mode not in ('cross', 'self'):
            raise ValueError(f'Unsupported interaction mode: {mode}')
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_name = teacher_name.lower()
        self.student_name = student_name.lower()
        self.mode = mode
        self.bins = int(bins)
        self._features = {}
        self._handles = []

        if self.teacher_name == 'mirepnet':
            teacher_taps = [teacher_model.embedding, teacher_model.transformer[2]]
            teacher_dims = [teacher_model.clshead.in_features] * 2
        elif self.teacher_name == 'cbramod':
            teacher_taps = [teacher_model.backbone.patch_embedding,
                            teacher_model.backbone.encoder.layers[5]]
            teacher_dims = [200, 200]
        else:
            raise ValueError(f'Unsupported teacher: {teacher_name}')

        if self.student_name == 'ifnet':
            student_taps = [[student_model.stem.sconv], [student_model.stem.interFre]]
            student_dims = [student_model.stem.sconv.conv.out_channels,
                            student_model.stem.out_planes]
        elif self.student_name == 'eegnet':
            student_taps = [[student_model.batchnorm1], [student_model.pooling2]]
            student_dims = [student_model.conv1.out_channels,
                            student_model.conv3_pointwise.out_channels]
        elif self.student_name == 'adfcnn':
            backbone = student_model.backbone
            student_taps = [[backbone.spectral_1, backbone.spectral_2],
                            [backbone.spatial_1, backbone.spatial_2]]
            student_dims = [2 * backbone.spectral_1[0].out_channels,
                            2 * backbone.spectral_1[0].out_channels]
        else:
            raise ValueError(f'Unsupported student: {student_name}')

        for depth, layer in enumerate(teacher_taps):
            self._capture(layer, ('teacher', depth, 0))
        for depth, layers in enumerate(student_taps):
            for branch, layer in enumerate(layers):
                self._capture(layer, ('student', depth, branch))

        self.teacher_projection = nn.ModuleList(nn.Linear(d, dim) for d in teacher_dims)
        self.student_projection = nn.ModuleList(nn.Linear(d, dim) for d in student_dims)
        self.teacher_gates = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.student_gates = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        for gate in (*self.teacher_gates, *self.student_gates):
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)
        self.fusion_head = nn.Linear(4 * self.bins * dim, 2)

    def _capture(self, layer, key):
        def save(_module, _inputs, output):
            self._features[key] = output
        self._handles.append(layer.register_forward_hook(save))

    def _teacher_tokens(self, depth):
        feature = self._features[('teacher', depth, 0)]
        if self.teacher_name == 'mirepnet':
            feature = feature.transpose(1, 2)
        else:
            feature = feature.mean(dim=1).transpose(1, 2)
        return F.adaptive_avg_pool1d(feature, self.bins).transpose(1, 2)

    def _student_tokens(self, depth):
        features = [self._features[('student', depth, 0)]]
        if self.student_name == 'adfcnn':
            features.append(self._features[('student', depth, 1)])
        pooled = []
        for feature in features:
            if feature.ndim == 4:
                feature = feature.mean(dim=2)
            pooled.append(F.adaptive_avg_pool1d(feature, self.bins))
        return torch.cat(pooled, dim=1).transpose(1, 2)

    def forward(self, teacher_input, student_input):
        self._features.clear()
        _, teacher_logits = self.teacher_model(teacher_input)
        _, student_logits = self.student_model(student_input, return_features=True)
        fused = []
        for depth in range(2):
            teacher = self.teacher_projection[depth](self._teacher_tokens(depth))
            student = self.student_projection[depth](self._student_tokens(depth))
            teacher_source, student_source = (student, teacher) if self.mode == 'cross' else (teacher, student)
            teacher = teacher * (2 * torch.sigmoid(self.teacher_gates[depth](teacher_source)))
            student = student * (2 * torch.sigmoid(self.student_gates[depth](student_source)))
            fused.extend((teacher.flatten(1), student.flatten(1)))
        fused_logits = self.fusion_head(torch.cat(fused, dim=1))
        return teacher_logits, student_logits, fused_logits

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._features.clear()
