"""Validation-only selection of per-pattern component weights (AA-MCE)."""

import math

import numpy as np
import pytest

from src.experiments import aa_core
from src.experiments import availability_aware_ensemble as aa3

COMPONENTS = ["C1", "C2", "C3"]  # C1 = Concat-H384/MD10, C2 = Concat-H192/MD30, C3 = RAMT
LOCKED = {"C1": 0.6, "C2": 0.4}  # locked MCE weights (C3 = 0)
STEP = 0.1
MARGIN = 0.0005


def test_simplex_grid_size_and_membership():
    grid = aa_core.simplex_grid(3, STEP)
    assert len(grid) == math.comb(10 + 2, 2) == 66
    assert len(aa_core.simplex_grid(5, STEP)) == math.comb(10 + 4, 4) == 1001  # five-candidate upper bound
    assert all(abs(sum(weights) - 1.0) < 1e-9 for weights in grid)
    assert all(min(weights) >= 0.0 for weights in grid)
    assert all(float(round(value * 10, 6)).is_integer() for weights in grid for value in weights)
    assert (0.6, 0.4, 0.0) in grid  # the locked weights lie on the grid
    assert aa3.simplex_grid(3, STEP) == grid


def test_mix_is_a_weighted_sum_that_skips_zero_weights():
    probabilities = {"C1": np.full((2, 3), 0.2), "C2": np.full((2, 3), 0.6), "C3": np.full((2, 3), np.nan)}
    mixed = aa_core.mix(probabilities, {"C1": 0.5, "C2": 0.5, "C3": 0.0})
    np.testing.assert_allclose(mixed, 0.4)
    np.testing.assert_allclose(aa3.mix(probabilities, {"C1": 0.5, "C2": 0.5, "C3": 0.0}), mixed)


def synthetic_validation(informative: str | None, seed: int = 0):
    """Validation predictions for 2 seeds x 3 availability patterns x 3 components."""
    rng = np.random.default_rng(seed)
    patterns = aa_core.all_patterns(2)
    labels = [aa_core.pattern_name(p, ["m1", "m2"]) for p in patterns]
    target = (rng.random((150, 4)) < 0.3).astype(np.float32)
    eligible = {index: np.arange(len(target)) for index in range(len(patterns))}
    shared = rng.random(target.shape)
    predictions = {}
    for seed_key in (1, 2):
        predictions[seed_key] = {}
        for index in range(len(patterns)):
            if informative is None:
                # identical components: every weight vector ranks the tracks identically
                predictions[seed_key][index] = {name: shared for name in COMPONENTS}
            else:
                values = {name: rng.random(target.shape) for name in COMPONENTS}
                values[informative] = 0.7 * target + 0.3 * rng.random(target.shape)
                predictions[seed_key][index] = values
    return predictions, eligible, target, patterns, labels


def select(predictions, eligible, target, patterns, labels, margin=MARGIN, locked=LOCKED):
    return aa_core.select_pattern_weights(predictions, eligible, target, patterns, labels, COMPONENTS, locked, STEP, margin)


def test_locked_weights_are_kept_when_no_candidate_gains():
    weights, rows, grid = select(*synthetic_validation(informative=None), margin=0.0)
    assert len(grid) == 3 * 66
    for variant in ("AA-MCE", "AA-MCE-2"):
        for chosen in weights[variant].values():
            assert chosen == {"C1": 0.6, "C2": 0.4, "C3": 0.0}
    assert not rows["deviates_from_locked"].any()


def test_locked_weights_are_replaced_when_the_gain_exceeds_the_margin():
    weights, rows, _ = select(*synthetic_validation(informative="C3"))
    aa = rows.loc[rows["variant"] == "AA-MCE"]
    assert aa["deviates_from_locked"].all()
    assert (aa["validation_mAP"] - aa["locked_validation_mAP"] > MARGIN).all()
    assert all(chosen["C3"] > 0 for chosen in weights["AA-MCE"].values())
    # AA-MCE-2 may only re-weight the two locked components.
    assert all(chosen["C3"] == 0 for chosen in weights["AA-MCE-2"].values())


def test_margin_is_a_strict_threshold_on_the_validation_gain():
    data = synthetic_validation(informative="C3")
    _, rows, _ = select(*data)
    aa = rows.loc[rows["variant"] == "AA-MCE"]
    gains = aa["best_grid_validation_mAP"] - aa["locked_validation_mAP"]
    weights, rows_high, _ = select(*data, margin=float(gains.max()) + 1e-9)
    assert not rows_high.loc[rows_high["variant"] == "AA-MCE", "deviates_from_locked"].any()
    assert all(chosen == {"C1": 0.6, "C2": 0.4, "C3": 0.0} for chosen in weights["AA-MCE"].values())
    _, rows_low, _ = select(*data, margin=float(gains.min()) - 1e-9)
    assert rows_low.loc[rows_low["variant"] == "AA-MCE", "deviates_from_locked"].all()


def test_locked_weights_must_lie_on_the_grid():
    with pytest.raises(ValueError):
        select(*synthetic_validation(informative=None), locked={"C1": 0.65, "C2": 0.35})
