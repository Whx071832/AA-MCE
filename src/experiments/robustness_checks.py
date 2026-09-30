"""Reviewer-facing robustness checks that need no training.

1. **Track-level bootstrap** of the mAP difference for the main comparisons.
   The paper's primary tests resample *labels*; a reviewer may ask whether
   the conclusions survive resampling *test tracks* instead.  For every
   comparison and condition the test tracks are resampled with replacement
   (``replicates`` times), macro-AP is recomputed for both models on the same
   resample, and the two-sided bootstrap p-value and 95% CI of the difference
   are reported.  AP uses the precision-at-each-positive definition, which
   equals scikit-learn's value in the absence of score ties; the maximum
   absolute discrepancy on the full test set is recorded.

2. **Selection-margin sensitivity** of the AA-MCE weight search: from the
   persisted validation grids, which patterns deviate from the locked mixture
   for margins from 0 to 0.005, and the resulting component weights.  This is
   validation-only and involves no test evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .paper_analysis import fma_test_frame
from .common import load_config
from .robust_multimodal_tagging import build_dataset, targets


def ap_fast(target: np.ndarray, score: np.ndarray) -> float:
    positives = target.sum()
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="stable")
    hits = target[order] > 0.5
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / positives)


def macro_ap(target: np.ndarray, probability: np.ndarray) -> float:
    """Macro AP over labels, vectorised (same definition as ``ap_fast``)."""
    order = np.argsort(-probability, axis=0, kind="stable")
    hits = np.take_along_axis(target, order, axis=0) > 0.5
    precision = np.cumsum(hits, axis=0) / np.arange(1, len(hits) + 1)[:, None]
    positives = hits.sum(axis=0)
    per_label = np.where(positives > 0, (precision * hits).sum(axis=0) / np.maximum(positives, 1), 0.0)
    return float(per_label.mean())


def load_mean_predictions(run_root: Path, model: str, labels: list[str], track_ids: np.ndarray, conditions: list[str]) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(run_root / "mean_test_predictions_all_conditions.parquet")
    columns = [f"score_{label}" for label in labels]
    result = {}
    for condition in conditions:
        group = frame.loc[(frame["model"] == model) & (frame["condition"] == condition)]
        if not np.array_equal(group["track_id"].astype(str).to_numpy(), track_ids):
            raise RuntimeError(f"Track order mismatch: {run_root.name}/{model}/{condition}")
        result[condition] = group[columns].to_numpy(dtype=np.float64)
    return result


def track_bootstrap(
    target: np.ndarray,
    proposed: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    conditions: list[str],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(seed)
    n = len(target)
    for condition in conditions:
        full_delta = macro_ap(target, proposed[condition]) - macro_ap(target, baseline[condition])
        deltas = np.empty(replicates)
        for replicate in range(replicates):
            index = rng.integers(0, n, size=n)
            deltas[replicate] = macro_ap(target[index], proposed[condition][index]) - macro_ap(target[index], baseline[condition][index])
        low, high = np.quantile(deltas, [0.025, 0.975])
        p_value = min(1.0, 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean())))
        rows.append({"condition": condition, "mean_delta_mAP": full_delta, "ci95_low": float(low), "ci95_high": float(high), "p_value": p_value, "replicates": replicates})
    return rows


def margin_sensitivity(grid_path: Path, locked: dict[str, float], margins: list[float]) -> pd.DataFrame:
    grid = pd.read_csv(grid_path)
    weight_columns = [column for column in grid.columns if column.startswith("w_")]
    rows = []
    for pattern, block in grid.groupby("pattern", sort=False):
        locked_mask = np.ones(len(block), dtype=bool)
        for column in weight_columns:
            locked_mask &= np.isclose(block[column].to_numpy(), locked.get(column[2:], 0.0))
        locked_score = float(block.loc[locked_mask, "validation_mAP"].iloc[0])
        best = block.loc[block["validation_mAP"].idxmax()]
        for margin in margins:
            deviates = float(best["validation_mAP"]) - locked_score > margin
            chosen = best if deviates else block.loc[locked_mask].iloc[0]
            rows.append({"pattern": pattern, "margin": margin, "deviates": bool(deviates), "gain_over_locked": float(best["validation_mAP"]) - locked_score, **{column: float(chosen[column]) for column in weight_columns}})
    return pd.DataFrame(rows)


def run(project_root: Path, config: dict[str, Any], run_name: str, replicates: int) -> Path:
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    experiments = project_root / config["project"]["outputs_root"] / "experiments"
    summary: dict[str, Any] = {}

    # --- Onion -------------------------------------------------------------
    settings = config["robust_multimodal_tagging"]
    split_frames, labels, _ = build_dataset(project_root / config["project"]["processed_root"], int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
    target = targets(split_frames["test"], len(labels)).astype(np.float64)
    track_ids = split_frames["test"]["track_id"].astype(str).to_numpy()
    conditions = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
    sources = {
        "OptimizedEnsemble": experiments / "optimized_multimodal_tagging_final_v1",
        "FullConcatMD10-H384-BCE": experiments / "optimized_multimodal_tagging_final_v1",
        "RAMT": experiments / "full_feature_ablation_v1",
        "ConcatModDrop": experiments / "full_feature_ablation_v1",
        "EarlyConcat": experiments / "full_feature_ablation_v1",
        "AudioOnly": experiments / "full_feature_ablation_v1",
        "AttnFusion": experiments / "advanced_fusion_baselines_v1",
        "EvidFusion": experiments / "advanced_fusion_baselines_v1",
        "AA-MCE": experiments / "availability_aware_ensemble_v1",
    }
    predictions = {model: load_mean_predictions(root, model, labels, track_ids, conditions) for model, root in sources.items() if (root / "completion.json").exists()}
    discrepancy = max(abs(macro_ap(target, predictions["OptimizedEnsemble"][c]) - float(average_precision_score(target, predictions["OptimizedEnsemble"][c], average="macro"))) for c in conditions)
    summary["onion_ap_definition_max_abs_diff"] = float(discrepancy)
    comparisons = [("OptimizedEnsemble", b) for b in ["AudioOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion"] if b in predictions]
    if "AA-MCE" in predictions:
        comparisons += [("AA-MCE", "OptimizedEnsemble"), ("AA-MCE", "RAMT"), ("AA-MCE", "AttnFusion"), ("AA-MCE", "EvidFusion")]
    rows = []
    for number, (proposed, baseline) in enumerate(comparisons):
        for row in track_bootstrap(target, predictions[proposed], predictions[baseline], conditions, replicates, 20261501 + number):
            rows.append({"dataset": "onion", "proposed": proposed, "baseline": baseline, **row})
        print(f"onion track bootstrap complete {proposed} vs {baseline}", flush=True)
    onion_rows = pd.DataFrame(rows)

    # --- FMA ---------------------------------------------------------------
    fma_rows = pd.DataFrame()
    fma_root = experiments / "fma_cross_dataset_v1"
    if (fma_root / "completion.json").exists():
        test, fma_labels = fma_test_frame(project_root, config)
        fma_target = targets(test, len(fma_labels)).astype(np.float64)
        fma_ids = test["track_id"].astype(str).to_numpy()
        fma_conditions = ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"]
        fma_models = ["MCE", "TextOnly", "EarlyConcat", "ConcatModDrop", "ConcatMD10-H384", "AttnFusion", "EvidFusion"]
        fma_predictions = {model: load_mean_predictions(fma_root, model, fma_labels, fma_ids, fma_conditions) for model in fma_models}
        rows = []
        for number, baseline in enumerate(fma_models[1:]):
            for row in track_bootstrap(fma_target, fma_predictions["MCE"], fma_predictions[baseline], fma_conditions, replicates, 20261601 + number):
                rows.append({"dataset": "fma", "proposed": "MCE", "baseline": baseline, **row})
            print(f"fma track bootstrap complete MCE vs {baseline}", flush=True)
        fma_rows = pd.DataFrame(rows)
    pd.concat([onion_rows, fma_rows], ignore_index=True).to_csv(output_root / "track_level_bootstrap.csv", index=False)

    # --- margin sensitivity -------------------------------------------------
    frames = []
    for run_dir, locked in (
        ("availability_aware_ensemble_v1", {"FullConcatMD10-H384-BCE": 0.6, "FullConcatMD-BCE": 0.4}),
        ("availability_aware_ensemble_fma_v1", {"ConcatMD10-H384": 0.6, "ConcatModDrop": 0.4}),
        ("availability_aware_ensemble_sixview_v1", {"ConcatMD10-H384": 0.6, "ConcatModDrop": 0.4}),
    ):
        path = experiments / run_dir / "pattern_weight_grid.csv"
        if path.exists():
            frames.append(margin_sensitivity(path, locked, [0.0, 0.0005, 0.001, 0.002, 0.005]).assign(run=run_dir))
    if frames:
        sensitivity = pd.concat(frames, ignore_index=True)
        sensitivity.to_csv(output_root / "selection_margin_sensitivity.csv", index=False)
        summary["margin_sensitivity"] = {
            run: {str(margin): int(block.loc[block["margin"] == margin, "deviates"].sum()) for margin in sorted(block["margin"].unique())}
            for run, block in sensitivity.groupby("run")
        }
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["status"] = "complete"
    (output_root / "completion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", default="robustness_checks_v1")
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.replicates)


if __name__ == "__main__":
    main()
