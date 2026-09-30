"""Post-hoc Platt calibration and the calibration metrics (ECE, Brier)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.experiments.paper_analysis import brier_by_label, ece_by_label, log_loss
from src.experiments.posthoc_calibration import apply_platt, fit_platt


def overconfident_scores(seed: int = 0, tracks: int = 3000, labels: int = 3):
    """Correctly ranked but overconfident probabilities (logit scaled by 3)."""
    rng = np.random.default_rng(seed)
    true_logit = rng.uniform(-2.2, 2.2, size=(tracks, labels)) - 0.5
    target = (rng.random((tracks, labels)) < 1.0 / (1.0 + np.exp(-true_logit))).astype(np.float64)
    raw = 1.0 / (1.0 + np.exp(-3.0 * true_logit))
    return target, raw


def test_platt_scaling_is_monotone_and_leaves_the_ranking_unchanged():
    target, raw = overconfident_scores()
    parameters = fit_platt(target, raw)
    assert parameters.shape == (3, 2)
    assert (parameters[:, 0] > 0).all()  # positive slope: order preserving
    calibrated = apply_platt(raw, parameters)
    for label in range(target.shape[1]):
        order = np.argsort(raw[:, label], kind="stable")
        assert np.all(np.diff(calibrated[order, label]) >= 0.0)
        ap_raw = average_precision_score(target[:, label], raw[:, label])
        ap_calibrated = average_precision_score(target[:, label], calibrated[:, label])
        assert ap_calibrated == pytest.approx(ap_raw, abs=1e-12)


def test_platt_scaling_reduces_calibration_error_of_overconfident_scores():
    target, raw = overconfident_scores(seed=1)
    calibrated = apply_platt(raw, fit_platt(target, raw))
    assert ece_by_label(target, calibrated).mean() < 0.5 * ece_by_label(target, raw).mean()
    assert brier_by_label(target, calibrated).mean() < brier_by_label(target, raw).mean()


def test_degenerate_label_keeps_identity_parameters():
    target = np.zeros((20, 1))
    probability = np.linspace(0.05, 0.95, 20)[:, None]
    parameters = fit_platt(target, probability)
    np.testing.assert_array_equal(parameters, [[1.0, 0.0]])
    np.testing.assert_allclose(apply_platt(probability, parameters), probability, rtol=1e-9)


def test_ece_known_values():
    target = np.array([[1], [0], [0], [0], [1], [1], [1], [1]], dtype=float)
    probability = np.array([[0.25]] * 4 + [[0.95]] * 4)
    # bin of 0.25: 1/4 positives -> gap 0; bin of 0.95: 4/4 positives -> gap 0.05; each bin holds half the tracks
    assert ece_by_label(target, probability)[0] == pytest.approx(0.025)
    constant = np.full((10, 1), 0.9)
    half = np.array([[1.0]] * 5 + [[0.0]] * 5)
    assert ece_by_label(half, constant)[0] == pytest.approx(0.4)


def test_ece_uses_15_equal_width_bins_and_includes_probability_one():
    assert ece_by_label(np.zeros((3, 1)), np.ones((3, 1)))[0] == pytest.approx(1.0)
    # 0.30 and 0.36 fall into different 1/15-wide bins, so their errors do not cancel
    target = np.array([[1.0], [0.0]])
    probability = np.array([[0.30], [0.36]])
    assert ece_by_label(target, probability)[0] == pytest.approx(0.5 * 0.70 + 0.5 * 0.36)


def test_brier_and_log_loss():
    target = np.array([[1.0], [0.0]])
    probability = np.array([[0.8], [0.3]])
    assert brier_by_label(target, probability)[0] == pytest.approx((0.2**2 + 0.3**2) / 2)
    assert log_loss(target, probability) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
