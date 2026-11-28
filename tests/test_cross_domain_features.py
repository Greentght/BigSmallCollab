"""Scientific invariants for the two feature-transfer mechanisms."""
import torch

from collab.cross_domain_features import AttentionDirection, SemanticDirection


def test_attention_energy_ignores_channel_permutation_and_global_scale():
    torch.manual_seed(7)
    own = torch.randn(4, 3, 2, 5, requires_grad=True)
    peer = (own.detach()[:, [2, 0, 1]] * 3).requires_grad_()
    loss, _ = AttentionDirection()([own], [peer])
    assert loss.item() < 1e-12
    assert torch.autograd.grad(loss, peer, allow_unused=True)[0] is None


def test_semckd_short_batch_retains_every_example_and_detaches_peer():
    torch.manual_seed(8)
    net = SemanticDirection([2, 3], [4, 5], 16, 10)
    own = [torch.randn(3, c, 2, 7, requires_grad=True) for c in (2, 3)]
    peer = [torch.randn(3, c, 1, 4, requires_grad=True) for c in (4, 5)]
    loss, info = net(own, peer)
    grads = torch.autograd.grad(loss, own + peer + list(net.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads[:2])
    assert all(g is None for g in grads[2:4])
    assert any(g is not None and g.abs().sum() > 0 for g in grads[4:])
    assert info['association'].shape == (2, 2)
    assert torch.allclose(info['association'].sum(1), torch.ones(2))


def test_semckd_uniform_control_has_identical_regressors_and_preserves_rng():
    torch.manual_seed(13)
    rng = torch.get_rng_state().clone()
    learned = SemanticDirection([2, 3], [4, 5], 16, 19)
    uniform = SemanticDirection([2, 3], [4, 5], 16, 19, uniform=True)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in learned.regressors.state_dict().items():
        assert torch.equal(value, uniform.regressors.state_dict()[name])
    own = [torch.randn(3, c, 2, 7) for c in (2, 3)]
    peer = [torch.randn(3, c, 1, 4) for c in (4, 5)]
    loss, info = uniform(own, peer)
    assert torch.equal(info['association'], torch.full((2, 2), .5))
    assert torch.allclose(loss, info['pair_mse'].mean(), atol=1e-6)
