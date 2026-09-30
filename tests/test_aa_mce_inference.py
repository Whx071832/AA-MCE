"""AA-MCE inference is a per-pattern table lookup (weights and thresholds), on synthetic data."""

import numpy as np
import pandas as pd

from src.experiments import aa_core

COMPONENTS = ["C1", "C2", "C3"]
LOCKED = {"C1": 0.6, "C2": 0.4}
VIEWS = ["m1", "m2"]


def noisy(rng, target, strength):
    return {name: np.clip(strength[name] * target + (1.0 - strength[name]) * rng.random(target.shape), 0.0, 1.0) for name in COMPONENTS}


def test_evaluate_variants_mixes_each_track_with_its_pattern_weights(tmp_path):
    rng = np.random.default_rng(3)
    patterns = aa_core.all_patterns(2)  # (1, 1), (1, 0), (0, 1)
    pattern_labels = [aa_core.pattern_name(p, VIEWS) for p in patterns]
    labels = ["t0", "t1", "t2", "t3"]
    target_validation = (rng.random((120, 4)) < 0.35).astype(np.float32)
    target_test = (rng.random((90, 4)) < 0.35).astype(np.float32)
    eligible = {index: np.arange(len(target_validation)) for index in range(len(patterns))}
    # C3 is only informative when a modality is missing.
    validation = {
        11: {index: noisy(rng, target_validation, {"C1": 0.3, "C2": 0.3, "C3": 0.1 if index == 0 else 0.7}) for index in range(len(patterns))}
    }
    conditions = ["observed", "no_m1"]
    test = {11: {condition: noisy(rng, target_test, {"C1": 0.3, "C2": 0.3, "C3": 0.5}) for condition in conditions}}
    test_patterns = {
        "observed": np.zeros(len(target_test), dtype=np.int64),
        "no_m1": np.full(len(target_test), patterns.index((0, 1)), dtype=np.int64),
    }
    weights, _, _ = aa_core.select_pattern_weights(validation, eligible, target_validation, patterns, pattern_labels, COMPONENTS, LOCKED, 0.1, 0.0005)
    assert weights["AA-MCE"][patterns.index((0, 1))]["C3"] > 0

    mean_test = aa_core.evaluate_variants(
        output_root=tmp_path,
        labels=labels,
        patterns=patterns,
        pattern_labels=pattern_labels,
        observed_index=0,
        components=COMPONENTS,
        controls=["C3"],
        locked=LOCKED,
        pattern_weights=weights,
        validation_predictions=validation,
        eligible=eligible,
        target_validation=target_validation,
        test_predictions=test,
        test_patterns=test_patterns,
        target_test=target_test,
        conditions=conditions,
        test_track_ids=np.asarray([f"track{i}" for i in range(len(target_test))]),
        test_proposed=["AA-MCE"],
        replicates=50,
        bootstrap_seed=1,
    )
    assert {"MCE", "MCE+AAT", "AA-MCE-w", "AA-MCE-2", "AA-MCE", "C3", "C3+AAT"} <= set(mean_test)
    for condition in conditions:
        predictions = test[11][condition]
        # Fixed-weight MCE is the special case with the locked 0.6/0.4/0 weights for every pattern.
        np.testing.assert_allclose(mean_test["MCE"][condition], 0.6 * predictions["C1"] + 0.4 * predictions["C2"], rtol=1e-6)
        expected = aa_core.mix(predictions, weights["AA-MCE"][int(test_patterns[condition][0])])
        np.testing.assert_allclose(mean_test["AA-MCE"][condition], expected, rtol=1e-6)
        np.testing.assert_allclose(mean_test["C3"][condition], predictions["C3"], rtol=1e-6)

    thresholds = pd.read_csv(tmp_path / "pattern_thresholds.csv")
    assert set(thresholds["pattern"]) == set(pattern_labels)  # one threshold vector per pattern and label
    assert len(thresholds.loc[(thresholds["variant"] == "AA-MCE")]) == len(patterns) * len(labels)
    tests = pd.read_csv(tmp_path / "paired_label_bootstrap_holm.csv")
    assert set(tests["proposed"]) == {"AA-MCE"} and tests["p_value_holm"].between(0, 1).all()
    for name in ("metrics_per_seed_condition.csv", "metrics_summary.csv", "per_label_metrics.csv", "mean_test_predictions_all_conditions.parquet"):
        assert (tmp_path / name).exists()
