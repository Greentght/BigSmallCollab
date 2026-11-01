import numpy as np
import pytest
import torch
from torch import nn

from collab.mcc_diffkd import EpsilonPredictor, ddpm_schedule, ddim_denoise
from collab.mirepnet_t2s import (
    classifier_readout, derive_seed, fixed_teacher_projection, forward_native,
    layer_norm_no_affine, projection_properties, projection_readout_diagnostic,
    projection_tensor, project_teacher_tensor,
)


@pytest.mark.parametrize('dt,ds,orientation', [(7, 3, 'rows'), (3, 7, 'columns'), (4, 4, 'rows')])
def test_projection_uses_possible_orthogonality_orientation(dt, ds, orientation):
    p = fixed_teacher_projection(dt, ds, 42)
    assert p.shape == (ds, dt)
    np.testing.assert_array_equal(p, fixed_teacher_projection(dt, ds, 42))
    assert projection_properties(p)['orthogonality'] == orientation
    gram = p @ p.T if ds <= dt else p.T @ p
    np.testing.assert_allclose(gram, np.eye(min(dt, ds)), atol=1e-12)


def test_expansion_preserves_teacher_readout_before_auxiliary_layer_norm():
    rng = np.random.default_rng(12)
    f = rng.standard_normal((9, 3))
    w = rng.standard_normal((4, 3))
    b = rng.standard_normal(4)
    logits = f @ w.T + b
    result = projection_readout_diagnostic(f, logits, w, b, fixed_teacher_projection(3, 8, 13))
    assert result['linear_readout_relative_weight_residual'] < 1e-12
    assert result['logit_rmse'] < 1e-12
    assert result['prediction_agreement'] == 1.0
    assert result['before_auxiliary_layer_norm'] is True


def test_projection_has_no_trainable_parameter_and_rejects_trainable_matrix():
    p = projection_tensor(fixed_teacher_projection(3, 8, 19))
    assert not p.requires_grad
    assert not isinstance(p, nn.Parameter)
    x = torch.randn(5, 3, requires_grad=True)
    _, normalized = project_teacher_tensor(x, p)
    normalized[:, 0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert p.grad is None
    with pytest.raises(ValueError, match='fixed'):
        project_teacher_tensor(x, nn.Parameter(p.clone()))


def test_seed_streams_are_distinct_and_deterministic():
    assert derive_seed('P', 666, 0, 'BNCI2014004', 'ifnet') == derive_seed('P', 666, 0, 'BNCI2014004', 'ifnet')
    values = {derive_seed(stage, seed, fold, ds, model)
              for stage in ('P', 'prior') for seed in (666, 667)
              for fold in (0, 1) for ds in ('BNCI2014004', 'AlexMI')
              for model in ('ifnet', 'eegnet')}
    assert len(values) == 32


@pytest.mark.parametrize('dataset,channels,classes', [
    ('BNCI2014004', 3, 2), ('BNCI2014001', 22, 2),
    ('BNCI2014001-4', 22, 4), ('BNCI2015001', 13, 2), ('AlexMI', 16, 2),
])
@pytest.mark.parametrize('student,dimension', [('ifnet', 512), ('eegnet', 496), ('adfcnn', 552)])
def test_native_student_classifier_input_is_the_supervised_feature(dataset, channels, classes, student, dimension):
    import config
    from models import get_adapter
    cfg = config.load_model_config(student, dataset, 'loso')
    cfg.update(in_channels=channels, samples=1000, dataset_name=dataset)
    adapter = get_adapter(student, device='cpu', **cfg)
    model = adapter.build(classes).eval()
    x = torch.zeros(2, channels * (2 if cfg.get('use_filter_bank') else 1), 1000)
    with torch.no_grad():
        feat, logits, meta = forward_native(adapter, model, x, return_metadata=True)
    assert feat.shape == (2, dimension)
    assert logits.shape == (2, classes)
    assert meta['native_trial_dimension'] == dimension
    assert meta['pooling_added'] is False
    weight, bias = classifier_readout(model, student)
    assert weight.shape == (classes, dimension)
    expected = torch.nn.functional.linear(feat, weight, bias)
    if student == 'adfcnn':
        expected = expected.log_softmax(dim=1)
    torch.testing.assert_close(logits, expected)


def test_mirepnet_tap_is_its_native_256d_classifier_input():
    from models.mirepnet.adapter import MIRepNetAdapter
    from models.mirepnet.mlm import mlm_mask
    # Transformer depth does not change its native classifier-input contract.
    model = mlm_mask(emb_size=256, depth=1, n_classes=4, pretrain=None).eval()
    adapter = MIRepNetAdapter(device='cpu', dataset_name='BNCI2014001-4', skip_preprocess=True)
    with torch.no_grad():
        feature, logits, meta = forward_native(adapter, model, torch.zeros(2, 45, 1000), True)
    assert feature.shape == (2, 256)
    assert logits.shape == (2, 4)
    assert meta['module_path'] == 'clshead'


def test_tap_rejects_an_adapter_that_substitutes_a_different_feature():
    class Adapter:
        name = 'eegnet'
        def forward(self, model, x):
            return x[:, :2], model.fc1(x)
    model = nn.Module()
    model.fc1 = nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match='actual native classifier input'):
        forward_native(Adapter(), model, torch.randn(2, 3))


def test_frozen_five_step_denoiser_passes_gradients_to_native_student():
    torch.manual_seed(43)
    predictor = EpsilonPredictor(8, hidden_dimensions=12, time_dimensions=4).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    _, alpha, sigma = ddpm_schedule(1000, 1e-4, .02, torch.device('cpu'))
    native = torch.randn(4, 8, requires_grad=True)
    student = layer_norm_no_affine(native)
    denoised = ddim_denoise(student, torch.randn_like(student), predictor, alpha, sigma)
    torch.nn.functional.mse_loss(denoised, torch.randn_like(denoised)).backward()
    assert native.grad is not None
    assert torch.isfinite(native.grad).all() and native.grad.norm() > 0
    assert all(parameter.grad is None for parameter in predictor.parameters())
