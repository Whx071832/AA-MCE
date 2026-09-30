"""Collect every number used in the manuscript from the frozen result files.

Outputs (``--out``, default ``outputs/paper/data``):
* ``onion_summary.csv``   per-model mean/std of every metric and condition (3 views)
* ``onion_tests.csv``      AA-MCE vs every baseline, label-level paired bootstrap, Holm
* ``fma_summary.csv`` / ``fma_tests.csv`` and ``sixview_summary.csv`` / ``sixview_tests.csv``
* ``ablation.csv``         mechanism and availability-aware ablation rows (Onion)
* ``tag_groups.csv`` / ``tag_points.csv``  tag-group and per-tag results for AA-MCE
* ``numbers.json``         scalar facts quoted in the text

All tests use the proposed model's per-label AP/F1 from the AA-MCE runs and the
baselines' per-label AP/F1 from their own frozen runs (five-seed mean
predictions, validation-selected thresholds), with 5,000 bootstrap resamples
over labels and Holm correction within each dataset x condition x metric
family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.common import _bootstrap

EXP = Path("outputs/experiments")

ONION_CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
FMA_CONDITIONS = ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"]
SIX_CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_view_missing", "random_half_views_missing", "keep_one_view"]

# model -> (run directory, model name inside that run)
ONION_SOURCES = {
    "AudioOnly": ("full_feature_ablation_v1", "AudioOnly"),
    "LyricsOnly": ("full_feature_ablation_v1", "LyricsOnly"),
    "VisualOnly": ("full_feature_ablation_v1", "VisualOnly"),
    "UniformMean": ("full_feature_ablation_v1", "UniformMean"),
    "EarlyConcat": ("full_feature_ablation_v1", "EarlyConcat"),
    "ConcatModDrop": ("full_feature_ablation_v1", "ConcatModDrop"),
    "ReliabilityGate": ("full_feature_ablation_v1", "ReliabilityGate"),
    "RAMT": ("full_feature_ablation_v1", "RAMT"),
    "H192": ("optimized_multimodal_tagging_final_v1", "FullConcatMD-BCE"),
    "H384": ("optimized_multimodal_tagging_final_v1", "FullConcatMD10-H384-BCE"),
    "AttnFusion": ("advanced_fusion_baselines_v1", "AttnFusion"),
    "EvidFusion": ("advanced_fusion_baselines_v1", "EvidFusion"),
    "MCE": ("optimized_multimodal_tagging_final_v1", "OptimizedEnsemble"),
    "AA-MCE": ("availability_aware_ensemble_v1", "AA-MCE"),
}
FMA_SOURCES = {
    "AudioOnly": ("fma_cross_dataset_v1", "AudioOnly"),
    "TextOnly": ("fma_cross_dataset_v1", "TextOnly"),
    "SocialOnly": ("fma_cross_dataset_v1", "SocialOnly"),
    "UniformMean": ("fma_cross_dataset_v1", "UniformMean"),
    "EarlyConcat": ("fma_cross_dataset_v1", "EarlyConcat"),
    "ConcatModDrop": ("fma_cross_dataset_v1", "ConcatModDrop"),
    "ReliabilityGate": ("fma_cross_dataset_v1", "ReliabilityGate"),
    "RAMT": ("fma_cross_dataset_v1", "RAMT"),
    "H384": ("fma_cross_dataset_v1", "ConcatMD10-H384"),
    "AttnFusion": ("fma_cross_dataset_v1", "AttnFusion"),
    "EvidFusion": ("fma_cross_dataset_v1", "EvidFusion"),
    "MCE": ("fma_cross_dataset_v1", "MCE"),
    "AA-MCE": ("availability_aware_ensemble_fma_v1", "AA-MCE"),
}
SIX_SOURCES = {
    "AudioEssentiaOnly": ("extended_modalities_v1", "AudioEssentiaOnly"),
    "AudioIvecOnly": ("extended_modalities_v1", "AudioIvecOnly"),
    "LyricsTfidfOnly": ("extended_modalities_v1", "LyricsTfidfOnly"),
    "LyricsW2vOnly": ("extended_modalities_v1", "LyricsW2vOnly"),
    "VisualResnetOnly": ("extended_modalities_v1", "VisualResnetOnly"),
    "VisualIncpOnly": ("extended_modalities_v1", "VisualIncpOnly"),
    "UniformMean": ("extended_modalities_v1", "UniformMean"),
    "EarlyConcat": ("extended_modalities_v1", "EarlyConcat"),
    "ConcatModDrop": ("extended_modalities_v1", "ConcatModDrop"),
    "ReliabilityGate": ("extended_modalities_v1", "ReliabilityGate"),
    "RAMT": ("extended_modalities_v1", "RAMT"),
    "H384": ("extended_modalities_v1", "ConcatMD10-H384"),
    "AttnFusion": ("extended_modalities_v1", "AttnFusion"),
    "EvidFusion": ("extended_modalities_v1", "EvidFusion"),
    "MCE": ("extended_modalities_v1", "MCE"),
    "AA-MCE": ("availability_aware_ensemble_sixview_v1", "AA-MCE"),
}
TAG_GROUP_ORDER = ["genre_content", "era_origin", "mood_affect", "preference_context"]


def summary_frame(sources: dict[str, tuple[str, str]], conditions: list[str]) -> pd.DataFrame:
    rows = []
    for model, (run, name) in sources.items():
        summary = pd.read_csv(EXP / run / "metrics_summary.csv")
        block = summary.loc[summary["model"] == name]
        if block.empty:
            raise KeyError(f"{model}: {run}/{name}")
        for condition in conditions:
            for metric in ("mAP", "MacroF1", "MicroF1", "MacroROC_AUC"):
                entry = block.loc[(block["condition"] == condition) & (block["metric"] == metric)]
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "metric": metric,
                        "mean": float(entry["mean"].iloc[0]),
                        "std": float(entry["std"].iloc[0]),
                        "ci95": float(entry["ci95_across_seeds"].iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def per_label(sources: dict[str, tuple[str, str]], conditions: list[str]) -> pd.DataFrame:
    frames = []
    for model, (run, name) in sources.items():
        frame = pd.read_csv(EXP / run / "per_label_metrics.csv")
        frame = frame.loc[(frame["model"] == name) & (frame["condition"].isin(conditions))]
        if frame.empty:
            raise KeyError(f"per-label {model}: {run}/{name}")
        frames.append(frame.assign(model=model)[["model", "condition", "label", "AP", "F1"]])
    return pd.concat(frames, ignore_index=True)


def holm(frame: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["p_holm"] = np.nan
    for _, index in result.groupby(family, sort=False).groups.items():
        ordered = result.loc[index, "p"].sort_values().index.tolist()
        previous = 0.0
        for rank, i in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * float(result.at[i, "p"])))
            result.at[i, "p_holm"] = previous
    return result


def tests(labels: pd.DataFrame, conditions: list[str], proposed: str, seed: int, replicates: int) -> pd.DataFrame:
    rows = []
    pivot = {metric: labels.pivot_table(index=["condition", "label"], columns="model", values=metric) for metric in ("AP", "F1")}
    for condition in conditions:
        for metric in ("AP", "F1"):
            block = pivot[metric].loc[condition]
            for baseline in block.columns:
                if baseline == proposed:
                    continue
                delta = (block[proposed] - block[baseline]).dropna().to_numpy(dtype=float)
                low, high, p_value = _bootstrap(delta, seed + len(rows), replicates)
                rows.append({"condition": condition, "metric": metric, "baseline": baseline, "delta": float(delta.mean()), "ci_low": low, "ci_high": high, "p": p_value, "labels": len(delta)})
    return holm(pd.DataFrame(rows), ["condition", "metric"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/paper/data"))
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}

    # --- Onion three views -------------------------------------------------
    onion = summary_frame(ONION_SOURCES, ONION_CONDITIONS)
    onion.to_csv(out / "onion_summary.csv", index=False)
    onion_labels = per_label(ONION_SOURCES, ONION_CONDITIONS)
    onion_labels.to_csv(out / "onion_per_label.csv", index=False)
    onion_tests = tests(onion_labels, ONION_CONDITIONS, "AA-MCE", 20261901, args.replicates)
    onion_tests.to_csv(out / "onion_tests.csv", index=False)
    print("onion tests done", flush=True)

    # --- ablation rows -------------------------------------------------------
    aa = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "metrics_summary.csv")
    five = pd.read_csv(EXP / "availability_aware_ensemble_5c_v1" / "metrics_summary.csv")
    rows = []
    for label, frame, name in (
        ("MCE+AAT", aa, "MCE+AAT"),
        ("AA-MCE-w", aa, "AA-MCE-w"),
        ("AA-MCE-2", aa, "AA-MCE-2"),
        ("AA-MCE", aa, "AA-MCE"),
        ("AA-MCE-5", five, "AA-MCE"),
        ("RAMT+AAT", aa, "RAMT+AAT"),
        ("AttnFusion+AAT", aa, "AttnFusion+AAT"),
        ("EvidFusion+AAT", aa, "EvidFusion+AAT"),
    ):
        block = frame.loc[frame["model"] == name]
        for _, entry in block.iterrows():
            rows.append({"model": label, "condition": entry["condition"], "metric": entry["metric"], "mean": entry["mean"], "std": entry["std"], "ci95": entry["ci95_across_seeds"]})
    ablation = pd.concat([onion, pd.DataFrame(rows)], ignore_index=True)
    ablation.to_csv(out / "ablation_summary.csv", index=False)
    aa_tests = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "paired_label_bootstrap_holm.csv")
    aa_tests.to_csv(out / "aa_variant_tests.csv", index=False)
    five_tests = pd.read_csv(EXP / "availability_aware_ensemble_5c_v1" / "paired_label_bootstrap_holm.csv")
    five_tests.to_csv(out / "aa5_tests.csv", index=False)

    # --- FMA and six views ---------------------------------------------------
    fma = summary_frame(FMA_SOURCES, FMA_CONDITIONS)
    fma.to_csv(out / "fma_summary.csv", index=False)
    fma_labels = per_label(FMA_SOURCES, FMA_CONDITIONS)
    fma_tests = tests(fma_labels, FMA_CONDITIONS, "AA-MCE", 20262001, args.replicates)
    fma_tests.to_csv(out / "fma_tests.csv", index=False)
    print("fma tests done", flush=True)
    aa_fma = pd.read_csv(EXP / "availability_aware_ensemble_fma_v1" / "metrics_summary.csv")
    aa_fma.to_csv(out / "fma_aa_variants_summary.csv", index=False)

    six = summary_frame(SIX_SOURCES, SIX_CONDITIONS)
    six.to_csv(out / "sixview_summary.csv", index=False)
    six_labels = per_label(SIX_SOURCES, SIX_CONDITIONS)
    six_tests = tests(six_labels, SIX_CONDITIONS, "AA-MCE", 20262101, args.replicates)
    six_tests.to_csv(out / "sixview_tests.csv", index=False)
    print("six-view tests done", flush=True)
    aa_six = pd.read_csv(EXP / "availability_aware_ensemble_sixview_v1" / "metrics_summary.csv")
    aa_six.to_csv(out / "sixview_aa_variants_summary.csv", index=False)
    # three-view vs six-view AA-MCE / MCE on the shared conditions
    shared = ["observed", "no_audio", "no_lyrics", "no_visual"]
    three_six = []
    for model, three_name, six_name in (("MCE", "MCE", "MCE"), ("AA-MCE", "AA-MCE", "AA-MCE")):
        a = onion_labels.loc[onion_labels["model"] == three_name]
        b = six_labels.loc[six_labels["model"] == six_name]
        for condition in shared:
            da = a.loc[a["condition"] == condition].set_index("label")["AP"]
            db = b.loc[b["condition"] == condition].set_index("label")["AP"]
            delta = (db - da).dropna().to_numpy(float)
            low, high, p_value = _bootstrap(delta, 20262201 + len(three_six), args.replicates)
            three_six.append({"model": model, "condition": condition, "delta_six_minus_three": float(delta.mean()), "ci_low": low, "ci_high": high, "p": p_value})
    pd.DataFrame(three_six).to_csv(out / "three_vs_six_tests.csv", index=False)

    # --- tag groups for AA-MCE and references --------------------------------
    import yaml

    config = yaml.safe_load(open("configs/experiments.yaml", encoding="utf-8"))
    groups = {name: list(tags) for name, tags in config["tag_groups"].items()}
    group_of = {tag: name for name, tags in groups.items() for tag in tags}
    labels_payload = json.loads((EXP / "robust_multimodal_tagging_v3" / "labels.json").read_text(encoding="utf-8"))
    frequencies = labels_payload["train_frequencies"]
    tg = onion_labels.assign(group=onion_labels["label"].map(group_of))
    group_rows = tg.groupby(["model", "condition", "group"])["AP"].mean().reset_index()
    group_rows.to_csv(out / "tag_groups.csv", index=False)
    points = tg.loc[(tg["model"].isin(["AA-MCE", "AudioOnly"])) & (tg["condition"] == "observed")].pivot_table(index=["label", "group"], columns="model", values="AP").reset_index()
    points["train_frequency"] = points["label"].map(frequencies)
    points.to_csv(out / "tag_points.csv", index=False)
    # within-group tests AA-MCE vs strongest references (observed)
    group_tests = []
    obs = tg.loc[tg["condition"] == "observed"].pivot_table(index="label", columns="model", values="AP")
    for group in TAG_GROUP_ORDER:
        tags = groups[group]
        for baseline in ("AudioOnly", "EarlyConcat", "RAMT", "AttnFusion", "EvidFusion", "H384", "MCE"):
            delta = (obs.loc[tags, "AA-MCE"] - obs.loc[tags, baseline]).to_numpy(float)
            low, high, p_value = _bootstrap(delta, 20262301 + len(group_tests), args.replicates)
            group_tests.append({"group": group, "baseline": baseline, "delta": float(delta.mean()), "ci_low": low, "ci_high": high, "p": p_value, "labels": len(tags)})
    holm(pd.DataFrame(group_tests), ["group"]).to_csv(out / "tag_group_tests.csv", index=False)

    # --- scalar facts --------------------------------------------------------
    def get(frame: pd.DataFrame, model: str, condition: str, metric: str) -> float:
        return float(frame.loc[(frame["model"] == model) & (frame["condition"] == condition) & (frame["metric"] == metric), "mean"].iloc[0])

    for model in ONION_SOURCES:
        values = [get(onion, model, c, "mAP") for c in ONION_CONDITIONS[1:]]
        numbers[f"onion_{model}_missing_mean_mAP"] = float(np.mean(values))
        numbers[f"onion_{model}_worst_mAP"] = float(np.min(values))
    for model in FMA_SOURCES:
        values = [get(fma, model, c, "mAP") for c in FMA_CONDITIONS[1:]]
        numbers[f"fma_{model}_missing_mean_mAP"] = float(np.mean(values))
    for model in SIX_SOURCES:
        values = [get(six, model, c, "mAP") for c in SIX_CONDITIONS[1:]]
        numbers[f"six_{model}_missing_mean_mAP"] = float(np.mean(values))
    manifest_onion = json.loads((EXP / "full_feature_ablation_v1" / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_fma = json.loads((EXP / "fma_cross_dataset_v1" / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_six = json.loads((EXP / "extended_modalities_v1" / "run_manifest.json").read_text(encoding="utf-8"))
    numbers["onion_samples"] = manifest_onion["samples"]
    numbers["onion_artists"] = manifest_onion["artists"]
    numbers["onion_coverage"] = manifest_onion["coverage"]
    numbers["fma_samples"] = manifest_fma["samples"]
    numbers["fma_artists"] = manifest_fma["artists"]
    numbers["six_views"] = manifest_six["views"]
    numbers["six_coverage_test"] = manifest_six["coverage"]["test"]
    numbers["tag_frequencies_min_max"] = [min(frequencies.values()), max(frequencies.values())]
    check = json.loads((EXP / "paper_analysis_v1" / "frozen_asset_check.json").read_text(encoding="utf-8"))
    numbers["fma_echonest_fraction"] = check["fma"]["echonest_available_fraction"]
    (out / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
