"""Paired bootstrap over labels and Holm correction."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.common import _bootstrap
from src.experiments.full_feature_ablation import holm_by_family
from src.experiments.paper_analysis import holm


def test_bootstrap_is_seeded_and_reports_a_two_sided_p_value():
    rng = np.random.default_rng(0)
    delta = rng.normal(0.02, 0.01, size=50)  # one paired difference per label
    result = _bootstrap(delta, 20261101, 2000)
    assert result == _bootstrap(delta, 20261101, 2000)
    low, high, p_value = result
    assert 0 < low < delta.mean() < high
    assert p_value == 0.0  # every resampled mean is positive
    symmetric = np.array([-1.0, 1.0] * 25)
    assert _bootstrap(symmetric, 7, 2000)[2] > 0.5


def test_holm_step_down_adjustment_within_each_family():
    frame = pd.DataFrame(
        {
            "proposed": "AA-MCE",
            "condition": "observed",
            "metric": ["AP", "AP", "AP", "F1"],
            "baseline": ["b1", "b2", "b3", "b1"],
            "p_value": [0.01, 0.04, 0.03, 0.04],
        }
    )
    adjusted = holm_by_family(frame)["p_value_holm"].tolist()
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.04])  # F1 is its own family
    renamed = frame.rename(columns={"proposed": "family"})
    assert holm(renamed, ["family", "condition", "metric"])["p_value_holm"].tolist() == pytest.approx(adjusted)
