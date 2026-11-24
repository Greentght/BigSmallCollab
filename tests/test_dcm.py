import torch

from collab.dcm import DCMHeads, mutual_losses
from models.cbramod.adapter import _CBraModModel
from models.eegnet.residual_eegnet import ResidualEEGNet
from models.ifnet.ifnet import IFNet
from models.neurogpt.neurogpt import NeuroGPTClassifier


def test_auxiliary_heads_match_models_and_receive_gradients():
    torch.set_num_threads(1)
    models = [
        ('cbramod', _CBraModModel(4, 22), torch.randn(2, 22, 4, 200)),
        ('ifnet', IFNet(22, 1000, 4, use_filter_bank=True), torch.randn(2, 44, 1000)),
        ('neurogpt', NeuroGPTClassifier(4, 22), torch.randn(2, 22, 1000)),
        ('eegnet', ResidualEEGNet(22, 1000, 4), torch.randn(2, 22, 1000)),
    ]
    for name, model, x in models:
        heads = DCMHeads(model, name, 4)
        model(x)
        logits = heads(len(x))
        assert all(value.shape == (2, 4) for value in logits)
        sum(value.square().mean() for value in logits).backward()
        assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                   for head in heads.heads)
        heads.close()


def test_directional_losses_detach_peer_and_cover_all_heads():
    outputs = [[torch.randn(3, 4, requires_grad=True) for _ in range(3)] for _ in range(2)]
    losses = mutual_losses(outputs, torch.tensor([0, 1, 2]), 'dcm')
    grads = torch.autograd.grad(losses[0], outputs[0] + outputs[1], allow_unused=True)
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads[:3])
    assert all(grad is None for grad in grads[3:])
