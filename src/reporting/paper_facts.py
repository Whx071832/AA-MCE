"""Print every scalar quoted in the manuscript text (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.common import _bootstrap

DATA = Path("outputs/paper/data")
EXP = Path("outputs/experiments")


def v(frame, model, condition, metric, column="mean"):
    return float(frame.loc[(frame["model"] == model) & (frame["condition"] == condition) & (frame["metric"] == metric), column].iloc[0])


def main() -> None:
    facts: dict = {}
    onion = pd.read_csv(DATA / "onion_summary.csv")
    for model in onion["model"].unique():
        facts[f"onion/{model}"] = {c: round(v(onion, model, c, "mAP"), 4) for c in onion["condition"].unique()}
        facts[f"onion/{model}/F1"] = {c: round(v(onion, model, c, "MacroF1"), 4) for c in onion["condition"].unique()}
        facts[f"onion/{model}/obs"] = {m: f"{v(onion, model, 'observed', m):.4f}±{v(onion, model, 'observed', m, 'std'):.4f}" for m in ("mAP", "MacroF1", "MicroF1", "MacroROC_AUC")}
    aa = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "metrics_summary.csv")
    for model in aa["model"].unique():
        facts[f"aa/{model}"] = {c: (round(v(aa, model, c, "mAP"), 4), round(v(aa, model, c, "MacroF1"), 4)) for c in aa["condition"].unique()}
    five = pd.read_csv(EXP / "availability_aware_ensemble_5c_v1" / "metrics_summary.csv")
    facts["aa5/AA-MCE"] = {c: (round(v(five, "AA-MCE", c, "mAP"), 4), round(v(five, "AA-MCE", c, "MacroF1"), 4)) for c in five["condition"].unique()}
    # five vs three candidates, label level
    p3 = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "per_label_metrics.csv")
    p5 = pd.read_csv(EXP / "availability_aware_ensemble_5c_v1" / "per_label_metrics.csv")
    rows = []
    for condition in p3["condition"].unique():
        for metric in ("AP", "F1"):
            a = p5.loc[(p5["model"] == "AA-MCE") & (p5["condition"] == condition)].set_index("label")[metric]
            b = p3.loc[(p3["model"] == "AA-MCE") & (p3["condition"] == condition)].set_index("label")[metric]
            delta = (a - b).to_numpy(float)
            low, high, p = _bootstrap(delta, 20262401 + len(rows), 5000)
            rows.append({"condition": condition, "metric": metric, "delta": round(float(delta.mean()), 4), "p": p})
    facts["five_vs_three"] = rows
    # tag groups (AA-MCE) retention and top/bottom tags
    groups = pd.read_csv(DATA / "tag_groups.csv")
    g = groups.loc[groups["model"] == "AA-MCE"].pivot(index="group", columns="condition", values="AP")
    facts["tag_groups/AA-MCE"] = g.round(4).to_dict()
    facts["tag_groups/retention_keep1"] = (100 * g["random_two_missing"] / g["observed"]).round(1).to_dict()
    facts["tag_groups/retention_no_audio"] = (100 * g["no_audio"] / g["observed"]).round(1).to_dict()
    points = pd.read_csv(DATA / "tag_points.csv").sort_values("AA-MCE", ascending=False)
    facts["tags/top8"] = points.head(8)[["label", "AA-MCE", "train_frequency"]].round(3).values.tolist()
    facts["tags/bottom8"] = points.tail(8)[["label", "AA-MCE", "train_frequency"]].round(3).values.tolist()
    facts["tags/spearman_freq_ap"] = float(points[["train_frequency", "AA-MCE"]].corr(method="spearman").iloc[0, 1])
    gt = pd.read_csv(DATA / "tag_group_tests.csv")
    facts["tag_group_tests"] = gt[["group", "baseline", "delta", "p_holm"]].round(4).values.tolist()
    # FMA and six views
    for name in ("fma", "sixview"):
        s = pd.read_csv(DATA / f"{name}_summary.csv")
        for model in s["model"].unique():
            facts[f"{name}/{model}"] = {c: (round(v(s, model, c, "mAP"), 4), round(v(s, model, c, "MacroF1"), 4)) for c in s["condition"].unique()}
        facts[f"{name}/AA-MCE/obs_std"] = f"{v(s, 'AA-MCE', 'observed', 'mAP'):.4f}±{v(s, 'AA-MCE', 'observed', 'mAP', 'std'):.4f}"
        variants = pd.read_csv(DATA / f"{name}_aa_variants_summary.csv")
        for model in ("MCE", "MCE+AAT", "AA-MCE-w", "AA-MCE-2", "RAMT+AAT", "AttnFusion+AAT", "EvidFusion+AAT"):
            facts[f"{name}_aa/{model}"] = {c: (round(v(variants, model, c, "mAP"), 4), round(v(variants, model, c, "MacroF1"), 4)) for c in variants["condition"].unique()}
    numbers = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    facts["missing_means"] = {k: round(val, 4) for k, val in numbers.items() if "missing_mean" in k}
    facts["three_vs_six"] = pd.read_csv(DATA / "three_vs_six_tests.csv").round(4).values.tolist()
    # calibration
    cal = pd.read_csv(EXP / "posthoc_calibration_v1" / "calibration_metrics.csv")
    facts["calibration"] = cal.loc[cal["condition"].isin(["observed", "random_two_missing", "no_audio"])].round(4).values.tolist()
    ct = pd.read_csv(EXP / "posthoc_calibration_v1" / "calibration_paired_holm.csv")
    x = ct.loc[ct["family"] == "AA-MCE_cal-AA_vs_others_cal-AA"]
    facts["calibration_tests"] = x[["condition", "metric", "baseline", "improvement_positive_is_better", "p_value_holm"]].round(4).values.tolist()
    # efficiency
    eff = pd.read_csv(EXP / "paper_efficiency_v1" / "efficiency.csv")
    facts["efficiency"] = eff[["model", "device", "parameters", "median_latency_ms"]].round(3).values.tolist()
    # robustness checks
    tb = pd.read_csv(EXP / "robustness_checks_v1" / "track_level_bootstrap.csv")
    facts["track_bootstrap"] = tb[["dataset", "proposed", "baseline", "condition", "mean_delta_mAP", "p_value"]].round(4).values.tolist()
    # mechanism chain (RAMT as proposed, full-feature)
    mech = pd.read_csv(EXP / "full_feature_ablation_v1" / "paired_label_bootstrap_holm.csv")
    mech = mech.loc[(mech["proposed"] == "RAMT") & (mech["baseline"].isin(["EarlyConcat", "ConcatModDrop", "ReliabilityGate", "UniformMean"]))]
    facts["mechanism"] = mech[["baseline", "condition", "metric", "mean_delta", "p_value_holm"]].round(4).values.tolist()
    gates = pd.read_csv(EXP / "full_feature_ablation_v1" / "gate_weights.csv")
    gate_means = gates.loc[gates["model"] == "RAMT"].groupby(["condition", "modality"])["mean_gate"].mean().round(3)
    facts["gates"] = {f"{condition}/{modality}": value for (condition, modality), value in gate_means.items()}
    weights = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "pattern_weights.csv")
    facts["pattern_weights_onion"] = weights.loc[weights["variant"] == "AA-MCE"].round(4).values.tolist()
    w6 = pd.read_csv(EXP / "availability_aware_ensemble_sixview_v1" / "pattern_weights.csv")
    w6 = w6.loc[w6["variant"] == "AA-MCE"].copy()
    w6["gain"] = w6["validation_mAP"] - w6["locked_validation_mAP"]
    facts["sixview_weight_gain_range"] = [round(float(w6["gain"].min()), 4), round(float(w6["gain"].max()), 4)]
    wf = pd.read_csv(EXP / "availability_aware_ensemble_fma_v1" / "pattern_weights.csv")
    facts["pattern_weights_fma"] = wf.loc[wf["variant"] == "AA-MCE"].round(4).values.tolist()
    sweep = pd.read_csv(EXP / "optimized_tagging_validation_v2" / "validation_sweep.csv")
    facts["sweep"] = sweep[["model", "mAP"]].round(4).values.tolist()
    sel = json.loads((EXP / "optimized_tagging_validation_v2" / "ensemble_selection.json").read_text(encoding="utf-8"))
    facts["ensemble_selection"] = sel
    labels = json.loads((EXP / "robust_multimodal_tagging_v3" / "labels.json").read_text(encoding="utf-8"))
    facts["labels"] = labels["labels"]
    facts["label_freq_range"] = [min(labels["train_frequencies"].values()), max(labels["train_frequencies"].values())]
    fma_labels = json.loads((EXP / "fma_cross_dataset_v1" / "labels.json").read_text(encoding="utf-8"))["labels"]
    facts["fma_labels"] = fma_labels
    hist = pd.read_csv(EXP / "full_feature_ablation_v1" / "training_history.csv")
    facts["epochs_E1F"] = hist.groupby(["model", "seed"])["epoch"].max().groupby("model").mean().round(1).to_dict()
    completion = {}
    for run in ("full_feature_ablation_v1", "availability_aware_ensemble_v1", "extended_modalities_v1", "fma_cross_dataset_v1", "availability_aware_ensemble_sixview_v1", "availability_aware_ensemble_fma_v1", "advanced_fusion_baselines_v1"):
        completion[run] = json.loads((EXP / run / "completion.json").read_text(encoding="utf-8")).get("elapsed_seconds")
    facts["elapsed_seconds"] = completion
    (DATA / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, default=str)[:200])


if __name__ == "__main__":
    main()
