"""E0/E5 post-hoc analyses over the frozen prediction files of every campaign.

The runner never trains anything.  It reads the persisted mean test
predictions of the completed campaigns and produces the analyses that the
redesigned protocol (``docs/REDESIGNED_RESEARCH_PLAN.md``) requires but that
the individual campaign runners do not emit:

* **E0 frozen-asset check** -- label order, test-track order and sample sizes
  are identical across every Onion campaign and match the data on disk; the
  same for the FMA campaign.
* **E5 calibration audit for every model** -- Brier, macro ECE (15 bins) and
  log loss per model and condition on both datasets, with paired label-level
  bootstrap tests of MCE against every baseline (Holm within family,
  condition and metric).
* **Worst-case / mean-missing robustness aggregates for every model** on both
  datasets (RQ2 "worst group").
* **E3 natural-missingness strata on FMA** -- the Echonest social descriptors
  exist for only a minority of tracks; test tracks are stratified by their
  availability and per-stratum mAP plus paired label-level tests are reported.

Outputs are written to ``outputs/experiments/<run-name>/``.
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

from .fma_cross_dataset import build_split
from .common import _bootstrap
from .common import load_config
from .robust_multimodal_tagging import build_dataset, targets


ONION_CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
FMA_CONDITIONS = ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"]

# Onion campaigns: (run key, run directory, models, family tag)
ONION_SOURCES: list[tuple[str, str, list[str] | None, str]] = [
    ("compact", "outputs/experiments/robust_multimodal_tagging_v3", None, "compact"),
    ("full", "outputs/experiments/full_feature_ablation_v1", None, "full"),
    ("final", "outputs/experiments/optimized_multimodal_tagging_final_v1", ["FullConcatMD10-H384-BCE", "FullConcatMD-BCE", "OptimizedEnsemble"], "full"),
    ("advanced", "outputs/experiments/advanced_fusion_baselines_v1", ["AttnFusion", "EvidFusion"], "full"),
]
ONION_PROPOSED = "OptimizedEnsemble"
FMA_RUN = "outputs/experiments/fma_cross_dataset_v1"
FMA_PROPOSED = "MCE"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def ece_by_label(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> np.ndarray:
    """Per-label expected calibration error with equal-width bins."""
    result = np.zeros(target.shape[1], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for label in range(target.shape[1]):
        column = probability[:, label]
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            selected = (column >= lower) & (column < (upper if upper < 1.0 else 1.00001))
            if selected.any():
                result[label] += selected.mean() * abs(column[selected].mean() - target[selected, label].mean())
    return result


def brier_by_label(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return ((probability - target) ** 2).mean(axis=0)


def log_loss(target: np.ndarray, probability: np.ndarray, epsilon: float = 1e-7) -> float:
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))


def ap_by_label(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return np.asarray([average_precision_score(target[:, label], probability[:, label]) for label in range(target.shape[1])])


def holm(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_holm"] = np.nan
    for _, indices in result.groupby(family_columns, sort=False).groups.items():
        ordered = result.loc[indices, "p_value"].sort_values().index.tolist()
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * float(result.at[index, "p_value"])))
            result.at[index, "p_value_holm"] = previous
    return result


# ---------------------------------------------------------------------------
# Loading frozen predictions
# ---------------------------------------------------------------------------
def load_predictions(run_root: Path, labels: list[str], models: list[str] | None) -> dict[str, dict[str, np.ndarray]]:
    """Mean test predictions per model/condition; verifies label order."""
    stored = json.loads((run_root / "labels.json").read_text(encoding="utf-8"))["labels"]
    if stored != labels:
        raise RuntimeError(f"Label order mismatch in {run_root}")
    frame = pd.read_parquet(run_root / "mean_test_predictions_all_conditions.parquet")
    score_columns = [f"score_{label}" for label in labels]
    result: dict[str, dict[str, np.ndarray]] = {}
    track_order: np.ndarray | None = None
    for (model, condition), group in frame.groupby(["model", "condition"], sort=False):
        if models is not None and model not in models:
            continue
        ids = group["track_id"].astype(str).to_numpy()
        if track_order is None:
            track_order = ids
        elif not np.array_equal(track_order, ids):
            raise RuntimeError(f"Track order mismatch inside {run_root}: {model}/{condition}")
        result.setdefault(str(model), {})[str(condition)] = group[score_columns].to_numpy(dtype=np.float64)
    result["__track_ids__"] = {"order": track_order}  # type: ignore[assignment]
    return result


def calibration_tables(
    target: np.ndarray,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    families: dict[str, str],
    conditions: list[str],
    proposed: str,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration metrics for every model and paired tests of ``proposed``."""
    metric_rows, test_rows = [], []
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for model, condition_values in predictions.items():
        for condition in conditions:
            probability = condition_values[condition]
            ece = ece_by_label(target, probability)
            brier = brier_by_label(target, probability)
            cache[(model, condition)] = {"ECE": ece, "Brier": brier}
            metric_rows.append(
                {
                    "model": model,
                    "family": families[model],
                    "condition": condition,
                    "Brier": float(brier.mean()),
                    "MacroECE15": float(ece.mean()),
                    "LogLoss": log_loss(target, probability),
                }
            )
    for condition in conditions:
        for baseline, family in families.items():
            if baseline == proposed:
                continue
            for metric in ("ECE", "Brier"):
                # Positive improvement means the proposed model is better calibrated.
                delta = cache[(baseline, condition)][metric] - cache[(proposed, condition)][metric]
                low, high, p_value = _bootstrap(delta, 20260960 + len(test_rows), replicates)
                test_rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "metric": metric,
                        "proposed": proposed,
                        "baseline": baseline,
                        "improvement_positive_is_better": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(metric_rows), holm(pd.DataFrame(test_rows), ["family", "condition", "metric"])


def robustness_aggregate(summary: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    selected = summary.loc[summary["metric"].isin(["mAP", "MacroF1"])]
    pivot = selected.pivot_table(index=["family", "model", "metric"], columns="condition", values="mean")
    missing = conditions[1:]
    rows = []
    for (family, model, metric), values in pivot.iterrows():
        rows.append(
            {
                "family": family,
                "model": model,
                "metric": metric,
                "observed": values["observed"],
                "missing_mean": values[missing].mean(),
                "worst_case": values[missing].min(),
                "worst_condition": str(values[missing].idxmin()),
                "worst_retention_percent": 100.0 * values[missing].min() / values["observed"],
                "keep_one_retention_percent": 100.0 * values["random_two_missing"] / values["observed"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Onion
# ---------------------------------------------------------------------------
def analyse_onion(project_root: Path, config: dict[str, Any], output_root: Path, replicates: int) -> dict[str, Any]:
    settings = config["robust_multimodal_tagging"]
    split_frames, labels, _ = build_dataset(
        project_root / config["project"]["processed_root"], int(settings["top_labels"]), int(settings["min_train_tag_frequency"])
    )
    target = targets(split_frames["test"], len(labels)).astype(np.float64)
    test_ids = split_frames["test"]["track_id"].astype(str).to_numpy()
    check: dict[str, Any] = {"labels": len(labels), "test_tracks": int(len(test_ids)), "runs": {}}

    predictions: dict[str, dict[str, np.ndarray]] = {}
    families: dict[str, str] = {}
    summaries = []
    for key, relative, models, family in ONION_SOURCES:
        run_root = project_root / relative
        if not (run_root / "completion.json").exists():
            check["runs"][key] = {"status": "missing"}
            continue
        loaded = load_predictions(run_root, labels, models)
        order = loaded.pop("__track_ids__")["order"]
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        check["runs"][key] = {
            "status": "complete",
            "track_order_matches_data": bool(np.array_equal(order, test_ids)),
            "labels_match": True,
            "samples": manifest.get("samples"),
            "models": sorted(loaded),
        }
        if not np.array_equal(order, test_ids):
            raise RuntimeError(f"Test track order of {key} differs from the data on disk")
        summary = pd.read_csv(run_root / "metrics_summary.csv")
        if models is not None:
            summary = summary.loc[summary["model"].isin(models)]
        summary = summary.assign(family=family, source=key)
        summaries.append(summary)
        for model, values in loaded.items():
            tagged = model if family == "full" else f"{model}@compact"
            predictions[tagged] = values
            families[tagged] = family
    if ONION_PROPOSED not in predictions:
        raise RuntimeError("Frozen MCE predictions are required")

    metrics, tests = calibration_tables(target, predictions, families, ONION_CONDITIONS, ONION_PROPOSED, replicates)
    metrics["model"] = metrics["model"].str.replace("@compact", "", regex=False)
    tests["baseline"] = tests["baseline"].str.replace("@compact", "", regex=False)
    metrics.to_csv(output_root / "onion_calibration_metrics.csv", index=False)
    tests.to_csv(output_root / "onion_calibration_paired_holm.csv", index=False)
    merged = pd.concat(summaries, ignore_index=True)
    robustness_aggregate(merged, ONION_CONDITIONS).to_csv(output_root / "onion_robustness_aggregate.csv", index=False)

    # Feature-representation tests: full features vs compact sketch, same
    # architecture, per label (positive delta favours the full features).
    representation_rows = []
    for model in sorted(predictions):
        compact_name = f"{model}@compact"
        if families.get(model) != "full" or compact_name not in predictions:
            continue
        for condition in ONION_CONDITIONS:
            delta = ap_by_label(target, predictions[model][condition]) - ap_by_label(target, predictions[compact_name][condition])
            low, high, p_value = _bootstrap(delta, 20260980 + len(representation_rows), replicates)
            representation_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "metric": "AP",
                    "mean_delta_full_minus_compact": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": p_value,
                }
            )
    if representation_rows:
        holm(pd.DataFrame(representation_rows), ["condition", "metric"]).to_csv(
            output_root / "onion_feature_representation_paired_holm.csv", index=False
        )
    check["feature_representation_tests"] = len(representation_rows)
    return check


# ---------------------------------------------------------------------------
# FMA
# ---------------------------------------------------------------------------
def fma_test_frame(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the FMA label set and test frame exactly as the E3 runner did."""
    settings = config["fma_cross_dataset"]
    processed_root = project_root / config["project"]["processed_root"]
    tracks = pd.read_parquet(
        processed_root / "external" / "fma_tracks.parquet", columns=["track_id", "source_track_id", "artist_id", "genres_all"]
    )
    genres = pd.read_parquet(processed_root / "external" / "fma_genres.parquet")
    genre_names = {str(row.genre_id): str(row.title) for row in genres.itertuples()}
    tracks["genre_titles"] = tracks["genres_all"].map(
        lambda values: sorted({genre_names[str(v)] for v in (values.tolist() if isinstance(values, np.ndarray) else list(values or [])) if str(v) in genre_names})
    )
    tracks = tracks.loc[tracks["genre_titles"].map(bool)].copy()
    tracks["split"] = tracks["artist_id"].astype(str).map(build_split)
    frequencies: dict[str, int] = {}
    for titles in tracks.loc[tracks["split"] == "train", "genre_titles"]:
        for title in titles:
            frequencies[title] = frequencies.get(title, 0) + 1
    labels = [
        title
        for title, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= int(settings["min_train_label_frequency"])
    ][: int(settings["top_labels"])]
    index = {label: position for position, label in enumerate(labels)}
    tracks["label_ids"] = tracks["genre_titles"].map(lambda titles: [index[t] for t in titles if t in index])
    tracks = tracks.loc[tracks["label_ids"].map(bool)].copy()
    test = tracks.loc[tracks["split"] == "test"].sort_values("track_id", kind="stable").reset_index(drop=True)
    return test, labels


def analyse_fma(project_root: Path, config: dict[str, Any], output_root: Path, replicates: int) -> dict[str, Any]:
    run_root = project_root / FMA_RUN
    if not (run_root / "completion.json").exists():
        return {"status": "missing"}
    test, labels = fma_test_frame(project_root, config)
    target = targets(test, len(labels)).astype(np.float64)
    test_ids = test["track_id"].astype(str).to_numpy()
    loaded = load_predictions(run_root, labels, None)
    order = loaded.pop("__track_ids__")["order"]
    if not np.array_equal(order, test_ids):
        raise RuntimeError("FMA test track order differs from the rebuilt frame")
    families = {model: "full" for model in loaded}
    metrics, tests = calibration_tables(target, loaded, families, FMA_CONDITIONS, FMA_PROPOSED, replicates)
    metrics.to_csv(output_root / "fma_calibration_metrics.csv", index=False)
    tests.to_csv(output_root / "fma_calibration_paired_holm.csv", index=False)
    summary = pd.read_csv(run_root / "metrics_summary.csv").assign(family="full", source="fma")
    robustness_aggregate(summary, FMA_CONDITIONS).to_csv(output_root / "fma_robustness_aggregate.csv", index=False)

    # --- natural missingness: Echonest social descriptors ------------------
    fma_root = project_root / "Data" / "fma" / "fma_metadata" / "fma_metadata"
    echonest = pd.read_csv(fma_root / "echonest.csv", header=[0, 1, 2], index_col=0)
    echonest_ids = set(int(value) for value in echonest.index)
    available = test["source_track_id"].astype(np.int64).map(lambda value: int(value) in echonest_ids).to_numpy()
    strata = {"echonest_available": available, "echonest_unavailable": ~available}
    stratum_rows, stratum_tests = [], []
    for stratum, mask in strata.items():
        stratum_target = target[mask]
        valid_labels = np.flatnonzero((stratum_target.sum(axis=0) > 0) & (stratum_target.sum(axis=0) < mask.sum()))
        ap_cache: dict[tuple[str, str], np.ndarray] = {}
        for model, condition_values in loaded.items():
            for condition in FMA_CONDITIONS:
                probability = condition_values[condition][mask]
                ap = np.asarray(
                    [average_precision_score(stratum_target[:, label], probability[:, label]) for label in valid_labels]
                )
                ap_cache[(model, condition)] = ap
                stratum_rows.append(
                    {
                        "stratum": stratum,
                        "tracks": int(mask.sum()),
                        "labels_evaluated": int(len(valid_labels)),
                        "model": model,
                        "condition": condition,
                        "mAP": float(ap.mean()),
                    }
                )
        for condition in FMA_CONDITIONS:
            for baseline in loaded:
                if baseline == FMA_PROPOSED:
                    continue
                delta = ap_cache[(FMA_PROPOSED, condition)] - ap_cache[(baseline, condition)]
                low, high, p_value = _bootstrap(delta, 20260970 + len(stratum_tests), replicates)
                stratum_tests.append(
                    {
                        "stratum": stratum,
                        "condition": condition,
                        "metric": "AP",
                        "proposed": FMA_PROPOSED,
                        "baseline": baseline,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                    }
                )
    pd.DataFrame(stratum_rows).to_csv(output_root / "fma_natural_missingness_strata.csv", index=False)
    holm(pd.DataFrame(stratum_tests), ["stratum", "condition", "metric"]).to_csv(
        output_root / "fma_natural_missingness_paired_holm.csv", index=False
    )
    return {
        "status": "complete",
        "labels": len(labels),
        "test_tracks": int(len(test_ids)),
        "track_order_matches_data": True,
        "labels_match": True,
        "echonest_available_test_tracks": int(available.sum()),
        "echonest_available_fraction": float(available.mean()),
        "models": sorted(loaded),
    }


# ---------------------------------------------------------------------------
# Tag groups: objective content vs subjective preference tags
# ---------------------------------------------------------------------------
TAG_GROUP_BASELINES = ["AudioOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion"]


def analyse_tag_groups(project_root: Path, config: dict[str, Any], output_root: Path, replicates: int) -> dict[str, Any]:
    """Per-group AP/F1 of every full-feature model and within-group paired tests."""
    groups: dict[str, list[str]] = {str(name): [str(tag) for tag in tags] for name, tags in config["tag_groups"].items()}
    frames = []
    for key, relative, models, family in ONION_SOURCES:
        path = project_root / relative / "per_label_metrics.csv"
        if family != "full" or not path.exists():
            continue
        frame = pd.read_csv(path)
        if models is not None:
            frame = frame.loc[frame["model"].isin(models)]
        frames.append(frame.assign(source=key))
    per_label = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model", "condition", "label"], keep="last")
    labels = sorted(per_label["label"].unique())
    assigned = [tag for tags in groups.values() for tag in tags]
    if sorted(assigned) != labels:
        missing = sorted(set(labels) - set(assigned))
        extra = sorted(set(assigned) - set(labels))
        raise RuntimeError(f"tag_groups must partition the label set; missing={missing} extra={extra}")
    group_of = {tag: name for name, tags in groups.items() for tag in tags}
    per_label["group"] = per_label["label"].map(group_of)

    metric_rows = []
    for (model, condition, group), block in per_label.groupby(["model", "condition", "group"], sort=False):
        metric_rows.append(
            {
                "model": model,
                "condition": condition,
                "group": group,
                "labels": int(len(block)),
                "mean_AP": float(block["AP"].mean()),
                "mean_F1": float(block["F1"].mean()) if block["F1"].notna().all() else np.nan,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_root / "onion_tag_group_metrics.csv", index=False)

    test_rows = []
    pivot = per_label.pivot_table(index=["condition", "label"], columns="model", values="AP")
    for condition in ONION_CONDITIONS:
        block = pivot.loc[condition]
        for group, tags in groups.items():
            proposed = block.loc[tags, ONION_PROPOSED].to_numpy(dtype=np.float64)
            for baseline in TAG_GROUP_BASELINES:
                if baseline not in block.columns:
                    continue
                delta = proposed - block.loc[tags, baseline].to_numpy(dtype=np.float64)
                low, high, p_value = _bootstrap(delta, 20260990 + len(test_rows), replicates)
                test_rows.append(
                    {
                        "condition": condition,
                        "group": group,
                        "labels": len(tags),
                        "metric": "AP",
                        "proposed": ONION_PROPOSED,
                        "baseline": baseline,
                        "mean_delta": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value": p_value,
                    }
                )
    holm(pd.DataFrame(test_rows), ["condition", "group", "metric"]).to_csv(output_root / "onion_tag_group_paired_holm.csv", index=False)

    frequencies = json.loads((project_root / ONION_SOURCES[0][1] / "labels.json").read_text(encoding="utf-8"))["train_frequencies"]
    observed = per_label.loc[(per_label["condition"] == "observed") & (per_label["model"] == ONION_PROPOSED)].set_index("label")["AP"]
    payload = {
        "groups": groups,
        "labels": {tag: {"group": group_of[tag], "train_frequency": int(frequencies[tag]), "mce_observed_AP": float(observed[tag])} for tag in labels},
    }
    (output_root / "tag_groups.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        name: {
            "labels": len(tags),
            "mce_observed_mean_AP": float(observed[tags].mean()),
            "median_train_frequency": float(np.median([frequencies[tag] for tag in tags])),
        }
        for name, tags in groups.items()
    }
    return {"status": "complete", "groups": summary, "tests": len(test_rows)}


# ---------------------------------------------------------------------------
# AA-MCE transfer runs versus the frozen component/baseline models
# ---------------------------------------------------------------------------
AAMCE_EXTRA = [
    # (AA run, frozen run, conditions, baselines, label source)
    ("availability_aware_ensemble_fma_v1", "fma_cross_dataset_v1", FMA_CONDITIONS, ["ConcatModDrop", "ConcatMD10-H384", "EarlyConcat", "UniformMean"], "fma"),
    ("availability_aware_ensemble_sixview_v1", "extended_modalities_v1", ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_view_missing", "random_half_views_missing", "keep_one_view"], ["ConcatModDrop", "ConcatMD10-H384", "EarlyConcat", "UniformMean"], "onion"),
]


def analyse_aamce_extra(project_root: Path, config: dict[str, Any], output_root: Path, replicates: int) -> dict[str, Any]:
    """AA-MCE (transfer runs) vs frozen models that are not variants of the AA run."""
    experiments = project_root / config["project"]["outputs_root"] / "experiments"
    rows, status = [], {}
    for aa_name, frozen_name, conditions, baselines, label_source in AAMCE_EXTRA:
        aa_root, frozen_root = experiments / aa_name, experiments / frozen_name
        if not (aa_root / "completion.json").exists() or not (frozen_root / "completion.json").exists():
            status[aa_name] = "missing"
            continue
        if label_source == "fma":
            test, labels = fma_test_frame(project_root, config)
        else:
            settings = config["robust_multimodal_tagging"]
            frames, labels, _ = build_dataset(project_root / config["project"]["processed_root"], int(settings["top_labels"]), int(settings["min_train_tag_frequency"]))
            test = frames["test"]
        target = targets(test, len(labels)).astype(np.float64)
        aa = load_predictions(aa_root, labels, ["AA-MCE", "MCE"])
        aa.pop("__track_ids__")
        frozen = load_predictions(frozen_root, labels, baselines)
        frozen.pop("__track_ids__")
        for condition in conditions:
            proposed_ap = ap_by_label(target, aa["AA-MCE"][condition])
            for baseline in baselines:
                delta = proposed_ap - ap_by_label(target, frozen[baseline][condition])
                low, high, p_value = _bootstrap(delta, 20261801 + len(rows), replicates)
                rows.append({"run": aa_name, "condition": condition, "metric": "AP", "proposed": "AA-MCE", "baseline": baseline, "mean_delta": float(delta.mean()), "ci95_low": low, "ci95_high": high, "p_value": p_value})
        status[aa_name] = "complete"
    if rows:
        holm(pd.DataFrame(rows), ["run", "condition", "metric"]).to_csv(output_root / "aamce_transfer_vs_frozen_paired_holm.csv", index=False)
    return status


def run(project_root: Path, config: dict[str, Any], run_name: str, replicates: int) -> Path:
    output_root = project_root / config["project"]["outputs_root"] / "experiments" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check = {
        "onion": analyse_onion(project_root, config, output_root, replicates),
        "fma": analyse_fma(project_root, config, output_root, replicates),
        "tag_groups": analyse_tag_groups(project_root, config, output_root, replicates),
        "aamce_transfer": analyse_aamce_extra(project_root, config, output_root, replicates),
    }
    (output_root / "frozen_asset_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    completion = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "bootstrap_replicates": replicates,
        "run_directory": str(output_root),
    }
    (output_root / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-name", default="paper_analysis_v1")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    run(args.project_root, load_config(args.project_root, args.config), args.run_name, args.replicates)


if __name__ == "__main__":
    main()
