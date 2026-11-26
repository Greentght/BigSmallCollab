"""Training-only AFID interaction between LaBraM and ResidualEEGNet."""
import torch
from torch import nn
from torch.nn import functional as F


class LaBraMEEGNetAFID(nn.Module):
    def __init__(self, teacher, student, input_chans, classes, patches, *, mode='cross', bins=4, dim=32):
        super().__init__()
        if mode not in ('cross', 'self'):
            raise ValueError('mode must be cross or self')
        self.teacher = teacher
        self.student = student
        self.input_chans = input_chans
        self.patches = patches
        self.bins = bins
        self.mode = mode
        self.features = {}
        self.handles = []
        for key, layer in [('t0', teacher.patch_embed), ('t1', teacher.blocks[5]),
                           ('s0', student.batchnorm1), ('s1', student.pooling2)]:
            self.handles.append(layer.register_forward_hook(self._capture(key)))
        self.teacher_projection = nn.ModuleList(nn.Linear(teacher.embed_dim, dim) for _ in range(2))
        self.student_projection = nn.ModuleList([nn.Linear(student.conv1.out_channels, dim),
                                                  nn.Linear(student.conv3_pointwise.out_channels, dim)])
        self.teacher_gates = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.student_gates = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        for gate in (*self.teacher_gates, *self.student_gates):
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)
        self.fusion_head = nn.Linear(4 * bins * dim, classes)

    def _capture(self, key):
        def save(_module, _inputs, output):
            self.features[key] = output
        return save

    def forward(self, xt, xs):
        self.features.clear()
        tokens = self.teacher.forward_features(xt, input_chans=self.input_chans, return_all_tokens=True)
        lt = self.teacher.head(tokens)
        _, ls = self.student(xs, return_features=True)
        outputs = []
        for depth in range(2):
            t = self.features[f't{depth}']
            if depth:
                t = t[:, 1:, :]
            batch, _, dim = t.shape
            t = t.reshape(batch, -1, self.patches, dim).mean(dim=1).transpose(1, 2)
            t = F.adaptive_avg_pool1d(t, self.bins).transpose(1, 2)
            s = self.features[f's{depth}'].mean(dim=2)
            s = F.adaptive_avg_pool1d(s, self.bins).transpose(1, 2)
            t = self.teacher_projection[depth](t)
            s = self.student_projection[depth](s)
            source_t, source_s = (s, t) if self.mode == 'cross' else (t, s)
            tg = t * (2 * torch.sigmoid(self.teacher_gates[depth](source_t)))
            sg = s * (2 * torch.sigmoid(self.student_gates[depth](source_s)))
            outputs.extend((tg.flatten(1), sg.flatten(1)))
        return lt, ls, self.fusion_head(torch.cat(outputs, dim=1))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.features.clear()
