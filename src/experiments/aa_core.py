"""Shared core of the availability-aware ensemble (AA-MCE) campaigns.

The three-view Onion campaign (``availability_aware_ensemble``) implements the
procedure inline; the FMA and six-view campaigns call the functions below so
that the selection rule, the variants, the controls and the statistics are
identical across datasets:

* ``fast_thresholds`` -- vectorised per-label F1 threshold search on the same
  grid as ``robust_multimodal_tagging.tune_thresholds`` (identical result);
* ``select_pattern_weights`` -- validation-only mixing weights per availability
  pattern (locked mixture kept unless the gain exceeds the margin);
* ``evaluate_variants`` -- MCE, MCE + AA thresholds, AA-MCE (weights only /
  locked components only / full) and every control with and without AA
  thresholds; per-seed metrics, mean predictions, per-label metrics and
  paired label bootstrap tests with Holm correction.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .common import _bootstrap
from .full_feature_ablation import holm_by_family
from .robust_multimodal_tagging import summarise


THRESHOLD_GRID = np.arange(0.05, 0.81, 0.025, dtype=np.float32)


def fast_thresholds(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Per-label threshold maximising F1 on the validation grid (vectorised)."""
    thresholds = np.full(target.shape[1], 0.5, dtype=np.float32)
    positives = target > 0.5
    for label in range(target.shape[1]):
        predicted = probability[:, label][:, None] >= THRESHOLD_GRID[None, :]
        tp = (predicted & positives[:, label][:, None]).sum(axis=0)
        fp = (predicted & ~positives[:, label][:, None]).sum(axis=0)
        fn = (~predicted & positives[:, label][:, None]).sum(axis=0)
        denominator = 2 * tp + fp + fn
        f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)
        thresholds[label] = THRESHOLD_GRID[int(np.argmax(f1))]
    return thresholds


def all_patterns(count: int) -> list[tuple[int, ...]]:
    return [p for p in itertools.product((1, 0), repeat=count) if sum(p) > 0]


def pattern_name(pattern: tuple[int, ...], names: list[str]) -> str:
    return "+".join(name for name, flag in zip(names, pattern, strict=True) if flag)


def pattern_index(availability: np.ndarray, patterns: list[tuple[int, ...]], fallback: int) -> np.ndarray:
    lookup = {pattern: index for index, pattern in enumerate(patterns)}
    rows = availability.astype(int)
    return np.asarray([lookup.get(tuple(int(v) for v in row), fallback) for row in rows], dtype=np.int64)


def simplex_grid(count: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    return [tuple(round(v / units, 6) for v in combo) for combo in itertools.product(range(units + 1), repeat=count) if sum(combo) == units]


def mix(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(float(weight) * probabilities[name] for name, weight in weights.items() if weight > 0)


def metrics_tracked(target: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    predicted = probability >= thresholds
    valid_auc = np.logical_and(target.sum(axis=0) > 0, target.sum(axis=0) < len(target))
    return {
        "mAP": float(average_precision_score(target, probability, average="macro")),
        "MacroF1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "MicroF1": float(f1_score(target, predicted, average="micro", zero_division=0)),
        "MacroROC_AUC": float(roc_auc_score(target[:, valid_auc], probability[:, valid_auc], average="macro")),
    }


def select_pattern_weights(
    validation_predictions: dict[int, dict[int, dict[str, np.ndarray]]],
    eligible: dict[int, np.ndarray],
    target_validation: np.ndarray,
    patterns: list[tuple[int, ...]],
    pattern_labels: list[str],
    components: list[str],
    locked: dict[str, float],
    step: float,
    margin: float,
) -> tuple[dict[str, dict[int, dict[str, float]]], pd.DataFrame, pd.DataFrame]:
    seeds = sorted(validation_predictions)
    grid = simplex_grid(len(components), step)
    locked_combo = tuple(locked.get(name, 0.0) for name in components)
    if locked_combo not in grid:
        raise ValueError("locked weights must lie on the search grid")
    result: dict[str, dict[int, dict[str, float]]] = {"AA-MCE": {}, "AA-MCE-2": {}}
    grid_rows, weight_rows = [], []
    for index, pattern in enumerate(patterns):
        rows = eligible[index]
        pattern_target = target_validation[rows]
        scores: dict[tuple[float, ...], float] = {}
        for combo in grid:
            weights = dict(zip(components, combo, strict=True))
            values = [float(average_precision_score(pattern_target, mix(validation_predictions[seed][index], weights), average="macro")) for seed in seeds]
            scores[combo] = float(np.mean(values))
            grid_rows.append({"pattern": pattern_labels[index], **{f"w_{name}": w for name, w in weights.items()}, "validation_mAP": scores[combo]})
        locked_score = scores[locked_combo]
        two_component = lambda combo: all(w == 0 for name, w in zip(components, combo, strict=True) if name not in locked)  # noqa: E731
        for variant, allowed in (("AA-MCE", lambda combo: True), ("AA-MCE-2", two_component)):
            best = max((combo for combo in grid if allowed(combo)), key=lambda combo: scores[combo])
            chosen = best if scores[best] - locked_score > margin else locked_combo
            result[variant][index] = dict(zip(components, chosen, strict=True))
            weight_rows.append(
                {
                    "variant": variant,
                    "pattern": pattern_labels[index],
                    "eligible_validation_tracks": int(len(rows)),
                    **{f"w_{name}": w for name, w in result[variant][index].items()},
                    "validation_mAP": scores[chosen],
                    "locked_validation_mAP": locked_score,
                    "best_grid_validation_mAP": scores[best],
                    "deviates_from_locked": bool(chosen != locked_combo),
                }
            )
    return result, pd.DataFrame(weight_rows), pd.DataFrame(grid_rows)


def evaluate_variants(
    output_root: Path,
    labels: list[str],
    patterns: list[tuple[int, ...]],
    pattern_labels: list[str],
    observed_index: int,
    components: list[str],
    controls: list[str],
    locked: dict[str, float],
    pattern_weights: dict[str, dict[int, dict[str, float]]],
    validation_predictions: dict[int, dict[int, dict[str, np.ndarray]]],
    eligible: dict[int, np.ndarray],
    target_validation: np.ndarray,
    test_predictions: dict[int, dict[str, dict[str, np.ndarray]]],
    test_patterns: dict[str, np.ndarray],
    target_test: np.ndarray,
    conditions: list[str],
    test_track_ids: np.ndarray,
    test_proposed: list[str],
    replicates: int,
    bootstrap_seed: int,
    threshold_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = fast_thresholds,
) -> dict[str, dict[str, np.ndarray]]:
    seeds = sorted(validation_predictions)

    def locked_weights(_: int) -> dict[str, float]:
        return {name: locked.get(name, 0.0) for name in components}

    variants: dict[str, dict[str, Any]] = {
        "MCE": {"weights": locked_weights, "pattern_thresholds": False},
        "MCE+AAT": {"weights": locked_weights, "pattern_thresholds": True},
        "AA-MCE-w": {"weights": lambda index: pattern_weights["AA-MCE"][index], "pattern_thresholds": False},
        "AA-MCE-2": {"weights": lambda index: pattern_weights["AA-MCE-2"][index], "pattern_thresholds": True},
        "AA-MCE": {"weights": lambda index: pattern_weights["AA-MCE"][index], "pattern_thresholds": True},
    }
    for name in controls:
        variants[name] = {"single": name, "pattern_thresholds": False}
        variants[f"{name}+AAT"] = {"single": name, "pattern_thresholds": True}

    def probability_of(spec: dict[str, Any], predictions: dict[str, np.ndarray], index: int) -> np.ndarray:
        if "single" in spec:
            return predictions[spec["single"]]
        return mix(predictions, spec["weights"](index))

    def thresholds_by_track(per_pattern: dict[int, np.ndarray], track_pattern: np.ndarray) -> np.ndarray:
        result = np.empty((len(track_pattern), len(labels)), dtype=np.float32)
        for index in np.unique(track_pattern):
            result[track_pattern == index] = per_pattern[int(index)][None, :]
        return result

    metrics_rows, threshold_rows = [], []
    seed_thresholds: dict[str, dict[int, list[np.ndarray]]] = {name: {index: [] for index in range(len(patterns))} for name in variants}
    seed_test: dict[str, dict[str, list[np.ndarray]]] = {name: {condition: [] for condition in conditions} for name in variants}
    for seed in seeds:
        for name, spec in variants.items():
            per_pattern: dict[int, np.ndarray] = {}
            for index in range(len(patterns)):
                rows = eligible[index]
                per_pattern[index] = threshold_function(target_validation[rows], probability_of(spec, validation_predictions[seed][index], index))
                seed_thresholds[name][index].append(per_pattern[index])
                for label, value in zip(labels, per_pattern[index], strict=True):
                    threshold_rows.append({"variant": name, "seed": seed, "pattern": pattern_labels[index], "label": label, "threshold": float(value)})
            for condition in conditions:
                track_pattern = test_patterns[condition]
                predictions = test_predictions[seed][condition]
                if spec["pattern_thresholds"]:
                    probability = np.zeros((len(track_pattern), len(labels)), dtype=np.float32)
                    for index in np.unique(track_pattern):
                        selected = track_pattern == index
                        probability[selected] = probability_of(spec, {k: v[selected] for k, v in predictions.items()}, int(index))
                    thresholds = thresholds_by_track(per_pattern, track_pattern)
                else:
                    probability = probability_of(spec, predictions, observed_index)
                    thresholds = np.tile(per_pattern[observed_index][None, :], (len(track_pattern), 1))
                seed_test[name][condition].append(probability)
                metrics_rows.append({"model": name, "seed": seed, "condition": condition, **metrics_tracked(target_test, probability, thresholds)})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_root / "metrics_per_seed_condition.csv", index=False)
    summarise(metrics).to_csv(output_root / "metrics_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "pattern_thresholds.csv", index=False)

    mean_test = {name: {condition: np.mean(values, axis=0) for condition, values in per_condition.items()} for name, per_condition in seed_test.items()}
    mean_thresholds = {name: {index: np.mean(values, axis=0) for index, values in per_pattern.items()} for name, per_pattern in seed_thresholds.items()}
    per_label_rows, test_rows = [], []
    families = {proposed: [name for name in variants if name != proposed] for proposed in test_proposed}
    for condition in conditions:
        track_pattern = test_patterns[condition]
        ap_cache, f1_cache = {}, {}
        for name, spec in variants.items():
            probability = mean_test[name][condition]
            thresholds = thresholds_by_track(mean_thresholds[name], track_pattern) if spec["pattern_thresholds"] else np.tile(mean_thresholds[name][observed_index][None, :], (len(track_pattern), 1))
            predicted = probability >= thresholds
            ap_cache[name] = np.asarray([average_precision_score(target_test[:, label], probability[:, label]) for label in range(len(labels))])
            f1_cache[name] = np.asarray([f1_score(target_test[:, label], predicted[:, label], zero_division=0) for label in range(len(labels))])
            for label_id, label in enumerate(labels):
                per_label_rows.append({"model": name, "condition": condition, "label_id": label_id, "label": label, "AP": float(ap_cache[name][label_id]), "F1": float(f1_cache[name][label_id])})
        for proposed, baselines in families.items():
            for baseline in baselines:
                for metric, cache in (("AP", ap_cache), ("F1", f1_cache)):
                    delta = cache[proposed] - cache[baseline]
                    low, high, p_value = _bootstrap(delta, bootstrap_seed + len(test_rows), replicates)
                    test_rows.append({"condition": condition, "metric": metric, "proposed": proposed, "baseline": baseline, "mean_delta": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
    pd.DataFrame(per_label_rows).to_csv(output_root / "per_label_metrics.csv", index=False)
    holm_by_family(pd.DataFrame(test_rows)).to_csv(output_root / "paired_label_bootstrap_holm.csv", index=False)

    frames = []
    for name in variants:
        for condition in conditions:
            frame = pd.DataFrame(mean_test[name][condition], columns=[f"score_{label}" for label in labels])
            frame.insert(0, "track_id", test_track_ids)
            frame.insert(0, "condition", condition)
            frame.insert(0, "model", name)
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(output_root / "mean_test_predictions_all_conditions.parquet", index=False)
    return mean_test
