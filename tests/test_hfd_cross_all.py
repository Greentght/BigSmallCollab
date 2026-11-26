"""Actual native operators, output alignment and isolation for all nine pairs."""
from itertools import product

import torch

from collab.hfd_cross_all import CrossOperatorHFDAll, frozen_call, resample_time
from collab.hfd_cross_operator import CrossOperatorHFD
from models.adfcnn.adfcnn import ADFCNN_Net
from models.cbramod.cbramod import CBraMod
from models.eegnet.residual_eegnet import ResidualEEGNet
from models.ifnet.ifnet import IFNet
from models.mirepnet.mlm import mlm_mask
from models.neurogpt.neurogpt import NeuroGPTClassifier


class BigCBra(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CBraMod(n_layer=2)

    def forward(self, x):
        return self.backbone(x)


def construct(big_name, small_name, channels):
    if big_name == 'cbramod':
        big = BigCBra()
        big_input = torch.randn(2, channels, 4, 200)
    elif big_name == 'mirepnet':
        big = mlm_mask(emb_size=128, depth=2, n_classes=2)
        big_input = torch.randn(2, 45, 1000)
    else:
        big = NeuroGPTClassifier(num_classes=2, input_channels=channels)
        big_input = torch.randn(2, channels, 1000)
    if small_name == 'eegnet':
        small = ResidualEEGNet(in_channels=channels, samples=1000)
        small_input = torch.randn(2, channels, 1000)
    elif small_name == 'ifnet':
        small = IFNet(in_channels=channels, samples=1000, num_classes=2, use_filter_bank=True)
        small_input = torch.randn(2, channels * 2, 1000)
    else:
        small = ADFCNN_Net(in_channels=channels, samples=1000, num_classes=2)
        small_input = torch.randn(2, channels, 1000)
    big.eval()
    small.eval()
    return big, small, big_input, small_input


def test_all_nine_pairs_use_real_operators_and_isolate_gradients():
    torch.set_num_threads(1)
    for big_name, small_name in product(('mirepnet', 'cbramod', 'neurogpt'),
                                         ('ifnet', 'eegnet', 'adfcnn')):
        big, small, big_input, small_input = construct(big_name, small_name, channels=3)
        rng = torch.get_rng_state().clone()
        hfd = CrossOperatorHFDAll(big_name, small_name, big, small, channels=3, seed=666)
        try:
            assert torch.equal(rng, torch.get_rng_state())
            big(big_input)
            small(small_input)
            features = dict(hfd.features)
            assert torch.allclose(frozen_call(hfd._big_operator, features['big_first']),
                                  features['big_second'], atol=1e-6), (big_name, small_name)
            for i, op in enumerate(hfd._small_operators):
                assert torch.allclose(frozen_call(op, features[f'small_first_{i}']),
                                      features[f'small_second_{i}'], atol=1e-6), (big_name, small_name, i)
            # Re-run native forwards since the verification calls activate hooks.
            hfd.clear()
            big(big_input)
            small(small_input)
            features = dict(hfd.features)
            before = {name: buf.clone() for name, buf in small.named_buffers()}
            losses = hfd.losses()
            assert all(torch.isfinite(loss) for loss in losses)
            for name, buf in small.named_buffers():
                assert torch.equal(before[name], buf), (big_name, small_name, name)
            for i, (own, peer, projection) in enumerate((
                    (big, small, hfd.big_to_small), (small, big, hfd.small_to_big))):
                own_param = next(p for p in own.parameters() if p.requires_grad)
                own_proj = [p for p in projection.parameters()]
                peer_params = [p for p in peer.parameters() if p.requires_grad]
                grads = torch.autograd.grad(losses[i], [own_param] + own_proj + peer_params,
                                            allow_unused=True, retain_graph=True)
                assert grads[0] is not None and grads[0].norm() > 0, (big_name, small_name, i)
                assert all(g is not None and g.norm() > 0 for g in grads[1:1 + len(own_proj)])
                assert all(g is None for g in grads[1 + len(own_proj):])
            print('passed', big_name, small_name, flush=True)
        finally:
            hfd.close()


def test_interpolation_respects_input_times_and_clamps_endpoints():
    source = torch.tensor([1., 2., 3.])
    destination = torch.tensor([0., 1., 1.5, 2., 2.25, 4.])
    value = torch.tensor([[[[2.], [4.], [8.]]]], requires_grad=True)
    output = resample_time(value, source, destination)
    assert torch.allclose(output.flatten(), torch.tensor([2., 2., 3., 4., 5., 8.]))
    output.sum().backward()
    assert torch.allclose(value.grad.flatten(), torch.tensor([2.5, 2.25, 1.25]))


def test_existing_004_pair_is_numerically_compatible():
    torch.set_num_threads(1)
    big, small, big_input, small_input = construct('cbramod', 'eegnet', channels=3)
    previous = CrossOperatorHFD(big, small, seed=666)
    current = CrossOperatorHFDAll('cbramod', 'eegnet', big, small, channels=3, seed=666)
    try:
        for old, new in ((previous.big_to_small, current.big_to_small[0]),
                         (previous.small_to_big, current.small_to_big[0])):
            torch.testing.assert_close(old.weight, new.weight, rtol=0, atol=0)
        big(big_input)
        small(small_input)
        first = previous.losses()
        # Cross calls activate the native hooks. Capture fresh native features.
        big(big_input)
        small(small_input)
        second = current.losses()
        for old, new in zip(first, second):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)
    finally:
        previous.close()
        current.close()


def test_different_montages_preserve_native_output_shapes():
    torch.set_num_threads(1)
    for channels, big_name, small_name in ((22, 'neurogpt', 'adfcnn'),
                                           (13, 'cbramod', 'ifnet'),
                                           (16, 'mirepnet', 'eegnet')):
        big, small, big_input, small_input = construct(big_name, small_name, channels)
        hfd = CrossOperatorHFDAll(big_name, small_name, big, small, channels, seed=666)
        try:
            big(big_input)
            small(small_input)
            assert all(torch.isfinite(v) and v > 0 for v in hfd.losses())
        finally:
            hfd.close()
