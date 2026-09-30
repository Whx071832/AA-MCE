"""Per-label decision thresholds selected by F1 on validation data."""

import numpy as np
import pytest

from src.experiments.aa_core import THRESHOLD_GRID, fast_thresholds
from src.experiments.robust_multimodal_tagging import tune_thresholds


def test_threshold_grid_is_0_05_to_0_80_in_steps_of_0_025():
    assert len(THRESHOLD_GRID) == 31
    assert THRESHOLD_GRID[0] == pytest.approx(0.05)
    assert THRESHOLD_GRID[-1] == pytest.approx(0.80, abs=1e-6)
    np.testing.assert_allclose(np.diff(THRESHOLD_GRID), 0.025, atol=1e-6)


def test_toy_threshold_is_the_first_grid_value_with_maximal_f1():
    target = np.array([[0], [0], [1], [1], [1]], dtype=np.float32)
    probability = np.array([[0.10], [0.21], [0.36], [0.60], [0.90]])
    # Every threshold in (0.21, 0.36] separates the classes; ties resolve to the smallest grid value.
    assert tune_thresholds(target, probability)[0] == pytest.approx(0.225, abs=1e-6)
    assert fast_thresholds(target, probability)[0] == pytest.approx(0.225, abs=1e-6)


def test_thresholds_are_selected_per_label():
    target = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    probability = np.array([[0.70, 0.10], [0.66, 0.12], [0.31, 0.13], [0.20, 0.14]])
    thresholds = tune_thresholds(target, probability)
    assert 0.31 < thresholds[0] <= 0.66  # label 0 separates at a high threshold
    assert thresholds[1] <= 0.125  # label 1 needs a low threshold
    np.testing.assert_array_equal(fast_thresholds(target, probability), thresholds)


def test_vectorised_search_equals_the_reference_implementation():
    rng = np.random.default_rng(7)
    target = (rng.random((400, 6)) < np.array([0.05, 0.2, 0.4, 0.6, 0.0, 1.0])).astype(np.float32)
    probability = np.clip(0.5 * target + 0.6 * rng.random(target.shape), 0.0, 1.0)
    np.testing.assert_array_equal(fast_thresholds(target, probability), tune_thresholds(target, probability))
