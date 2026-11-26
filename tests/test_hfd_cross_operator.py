"""Behavioral checks for the paper's two cross-operator HFD paths."""
import torch

from collab.hfd_cross_operator import CrossOperatorHFD, _frozen_call
from models.cbramod.cbramod import CBraMod
from models.eegnet.residual_eegnet import ResidualEEGNet


class Big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CBraMod(n_layer=2)

    def forward(self, x):
        return self.backbone(x)


def models(mode):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    big, small = Big(), ResidualEEGNet(in_channels=3, samples=1000)
    big.train(mode)
    small.train(mode)
    rng = torch.get_rng_state().clone()
    hfd = CrossOperatorHFD(big, small, seed=666)
    assert torch.equal(rng, torch.get_rng_state())
    return big, small, hfd


def captured(big, small, hfd):
    generator = torch.Generator().manual_seed(777)
    big(torch.randn(2, 3, 4, 200, generator=generator))
    small(torch.randn(2, 3, 1000, generator=generator))
    return dict(hfd.features)


def test_cross_operators_are_the_real_next_native_operators():
    big, small, hfd = models(False)
    try:
        f = captured(big, small, hfd)
        assert f['big_first'].shape == (2, 3, 4, 200)
        assert f['big_second'].shape == (2, 3, 4, 200)
        assert f['small_first'].shape == (2, 8, 3, 1001)
        assert f['small_second'].shape == (2, 16, 1, 250)
        assert torch.allclose(_frozen_call(hfd._attention, f['big_first']), f['big_second'])
        assert torch.allclose(_frozen_call(hfd._cnn_layer, f['small_first']), f['small_second'])
        assert hfd.cnn_to_vit(f['small_first']).shape == f['big_first'].shape
        assert hfd.vit_to_cnn(f['big_first']).shape == f['small_first'].shape
        assert set(hfd.state_dict()) == {'big_to_small.weight', 'small_to_big.weight'}
    finally:
        hfd.close()


def test_each_direction_updates_only_receiving_model_and_preserves_peer_buffers():
    big, small, hfd = models(True)
    try:
        f = captured(big, small, hfd)
        before = {name: buf.detach().clone() for name, buf in small.named_buffers()}
        losses = hfd.losses()
        for name, buf in small.named_buffers():
            assert torch.equal(before[name], buf), name
        for index, (own, peer, projector) in enumerate((
                (big, small, hfd.big_to_small), (small, big, hfd.small_to_big))):
            own_first = next(p for p in own.parameters() if p.requires_grad)
            peer_params = [p for p in peer.parameters() if p.requires_grad]
            grads = torch.autograd.grad(losses[index], [own_first, projector.weight] + peer_params,
                                        allow_unused=True, retain_graph=True)
            assert grads[0] is not None and grads[0].norm() > 0
            assert grads[1] is not None and grads[1].norm() > 0
            assert all(value is None for value in grads[2:])
            assert torch.isfinite(losses[index])
    finally:
        hfd.close()


def test_upsampling_uses_the_requested_physical_time_coordinates():
    big, small, hfd = models(False)
    try:
        feature = torch.randn(2, 3, 4, 200)
        projected = hfd.big_to_small(feature).reshape(2, 3, 4, 8, 8)
        projected = projected.permute(0, 4, 1, 2, 3).reshape(2, 8, 3, 32)
        pos = ((torch.arange(1001) - 0.5) * (8.0 / 250.0)).clamp(0, 31)
        left = pos.floor().long()
        right = (left + 1).clamp(max=31)
        expected = torch.lerp(projected.index_select(-1, left),
                              projected.index_select(-1, right), pos - left)
        assert torch.allclose(hfd.vit_to_cnn(feature), expected, atol=1e-6)
    finally:
        hfd.close()
