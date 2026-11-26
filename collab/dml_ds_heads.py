"""Additional deep-supervision heads for the few-shot mutual experiment."""
from torch import nn

from collab.dcm import AuxiliaryClassifier, DCMHeads


class AdditionalHeads(nn.Module):
    """Capture MIRepNet transformer stages or ADFCNN spatial branches."""

    def __init__(self, model, architecture, classes):
        super().__init__()
        self.architecture = architecture
        self.features = {}
        if architecture == 'mirepnet':
            layers = [model.transformer[1], model.transformer[3]]
            channels = model.clshead.in_features
        elif architecture == 'adfcnn':
            layers = [model.backbone.spatial_1, model.backbone.spatial_2]
            channels = model.backbone.w_q.in_features
        else:
            raise ValueError(f'Unsupported architecture: {architecture}')
        self.heads = nn.ModuleList(AuxiliaryClassifier(channels, classes) for _ in layers)
        self.handles = [layer.register_forward_hook(self._capture(index))
                        for index, layer in enumerate(layers)]

    def _capture(self, index):
        def hook(_module, _inputs, output):
            self.features[index] = output
        return hook

    def forward(self, batch_size):
        if set(self.features) != {0, 1}:
            raise RuntimeError('Both auxiliary features must be captured by the current forward')
        logits = []
        for index, head in enumerate(self.heads):
            x = self.features[index]
            if x.shape[0] != batch_size:
                raise ValueError('Auxiliary feature batch size mismatch')
            if self.architecture == 'mirepnet':
                x = x.transpose(1, 2)
            else:
                x = x.squeeze(2)
            logits.append(head(x))
        self.features.clear()
        return logits

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.features.clear()


def build_heads(model, architecture, classes):
    if architecture in ('mirepnet', 'adfcnn'):
        return AdditionalHeads(model, architecture, classes)
    return DCMHeads(model, architecture, classes)
