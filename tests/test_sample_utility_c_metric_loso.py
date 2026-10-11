import numpy as np
import pytest

from experiments.distill.c_metric_diagnostics import (
    c_components, late_signed_probability_gain, strict_median_groups,
)


def _two_class_prob(true_probability):
    true_probability = np.asarray(true_probability, dtype=np.float64)
    return np.stack([true_probability, 1.0 - true_probability], axis=-1)


def test_c_components_use_window_local_decreasing_weights_and_normalization():
    ce = _two_class_prob([[0.2], [0.3], [0.4]])
    kd = _two_class_prob([[0.3], [0.5], [0.7]])
    result = c_components(ce, kd, np.asarray([0]),
                          {'c_1_100': (0, 3)})['c_1_100']

    distance = 0.1 + 0.2 + 0.3
    benefit = (2 / 3) * 0.3 + (1 / 3) * 0.5
    assert result['D'][0] == pytest.approx(distance)
    assert result['B'][0] == pytest.approx(benefit)
    assert result['c'][0] == pytest.approx(distance * benefit)
    assert result['c_normalized'][0] == pytest.approx(
        (distance / 3) * benefit)


def test_fixed_windows_exclude_epoch0_and_restart_weights_per_window():
    ce = _two_class_prob(np.full((100, 1), 0.4))
    kd_values = np.full((100, 1), 0.4)
    kd_values[:30] += 0.1
    kd_values[10:100] += 0.2
    kd = _two_class_prob(kd_values)
    result = c_components(ce, kd, np.asarray([0]))

    assert result['c_1_30']['window_length'] == 30
    assert result['c_11_100']['window_length'] == 90
    assert result['c_1_30']['D'][0] == pytest.approx(7.0)
    assert result['c_11_100']['D'][0] == pytest.approx(20.0)
    assert result['c_11_100']['B'][0] == pytest.approx(
        0.6 * np.arange(89, -1, -1).sum() / 90
        + 0.1 * np.arange(89, 69, -1).sum() / 90)


def test_late_gain_uses_epochs_81_through_100_inclusive():
    ce_values = np.full((100, 2), 0.2, dtype=np.float64)
    kd_values = np.full((100, 2), 0.2, dtype=np.float64)
    kd_values[80:100, 0] = 0.4
    kd_values[79, 0] = 0.9
    kd_values[80:100, 1] = 0.1
    ce = _two_class_prob(ce_values)
    kd = _two_class_prob(kd_values)
    gain = late_signed_probability_gain(ce, kd, np.asarray([0, 0]))

    assert gain.tolist() == pytest.approx([0.2, -0.1])


def test_strict_median_groups_report_ties_without_forcing_assignment():
    groups = strict_median_groups(np.asarray([0.0, 1.0, 1.0, 3.0]))
    assert groups['threshold'] == 1.0
    assert groups['low'].tolist() == [True, False, False, False]
    assert groups['ties'].tolist() == [False, True, True, False]
    assert groups['high'].tolist() == [False, False, False, True]
    all_ties = strict_median_groups(np.zeros(4))
    assert not all_ties['low'].any()
    assert not all_ties['high'].any()
    assert all_ties['ties'].all()


def test_invalid_shapes_nonfinite_values_and_bad_windows_are_rejected():
    probs = _two_class_prob(np.full((5, 2), 0.5))
    with pytest.raises(ValueError):
        c_components(probs, probs[:4], np.asarray([0, 1]))
    nonfinite = probs.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        c_components(nonfinite, probs, np.asarray([0, 1]))
    with pytest.raises(ValueError):
        c_components(probs, probs, np.asarray([0, 1]), {'bad': (0, 6)})
