"""Guard time indexing, gradient isolation, and RNG isolation for HFD."""

import torch
from torch import nn
from unittest.mock import patch

from collab.hfd import ShallowHFD


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Module()
        self.embedding.pool = nn.Identity()
        self.stem = nn.Module()
        self.stem.interFre = nn.Identity()
        self.backbone = nn.Module()
        self.backbone.patch_embedding = nn.Module()
        self.backbone.patch_embedding.proj_in = nn.Identity()
        self.batchnorm1 = nn.Identity()


def test_mirepnet_ifnet_time_alignment_and_isolation():
    torch.manual_seed(10)
    big, small = Stub(), Stub()
    before = torch.get_rng_state().clone()
    # torch.manual_seed silently re-seeds every CUDA device.  Even on a CPU
    # test host we can detect a regression by checking that call is absent.
    with patch('torch.cuda.manual_seed_all') as cuda_seed:
        hfd = ShallowHFD('mirepnet_ifnet', big, small, 666)
        cuda_seed.assert_not_called()
    assert torch.equal(before, torch.get_rng_state())
    b = torch.randn(2, 128, 1, 61, requires_grad=True)
    s = torch.arange(1000, dtype=torch.float32).reshape(1, 1, 1000).expand(2, 64, -1).clone().requires_grad_()
    big.embedding.pool(b)
    small.stem.interFre(s)
    bb, ss = hfd.aligned()
    assert bb.shape == (2, 61, 128) and ss.shape == (2, 61, 64)
    assert torch.equal(ss[0, :, 0], torch.arange(61) * 15 + 49)
    losses = hfd.losses()
    for i, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
        grads = torch.autograd.grad(losses[i], ((b, s)[i], (s, b)[i], projector.weight),
                                    retain_graph=True, allow_unused=True)
        assert grads[0] is not None and grads[1] is None and grads[2] is not None
        assert torch.isfinite(grads[0]).all() and torch.isfinite(grads[2]).all()
    hfd.close()
    assert len(big.embedding.pool._forward_hooks) == 0


def test_cbramod_eegnet_time_and_electrode_alignment():
    big, small = Stub(), Stub()
    hfd = ShallowHFD('cbramod_eegnet', big, small, 666)
    b = torch.randn(2, 25, 12, 8, requires_grad=True)
    electrode = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1) * 10000
    time = torch.arange(1001, dtype=torch.float32).reshape(1, 1, 1, 1001) - .5
    s = (electrode + time).expand(2, 8, -1, -1).clone().requires_grad_()
    big.backbone.patch_embedding.proj_in(b)
    small.batchnorm1(s)
    bb, ss = hfd.aligned()
    assert bb.shape == (2, 3, 4, 8, 25) and ss.shape == (2, 3, 4, 8, 8)
    expected = (torch.arange(4)[:, None] + torch.arange(8)[None, :] / 8) * 250
    for electrode_index in range(3):
        assert torch.allclose(ss[0, electrode_index, :, :, 0],
                              expected + 10000 * electrode_index)
    losses = hfd.losses()
    for i, projector in enumerate((hfd.big_to_small, hfd.small_to_big)):
        grads = torch.autograd.grad(losses[i], ((b, s)[i], (s, b)[i], projector.weight),
                                    retain_graph=True, allow_unused=True)
        assert grads[0] is not None and grads[1] is None and grads[2] is not None
    hfd.close()
