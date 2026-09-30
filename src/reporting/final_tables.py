"""Publication tables for the redesigned tagging study.

Writes CSV and Markdown tables to ``outputs/tables_final``.  Sources and
model lists are shared with the figure suite (``final_figures.PLOT_PARAMS``).

Tables 1-3 use the feature-matched Onion family (E1-F ablations, locked MCE
components, recent strong baselines, MCE).  Tables 4-5 are the FMA transfer.
Table 6 is the efficiency audit, Table 7 the calibration audit, Table 8 the
FMA natural-missingness strata and Table 9 the full-feature mechanism chain.
Table S1 documents the compact-feature (v3) family and the paired
feature-representation tests.

Usage::

    python -m src.reporting.final_tables --project-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_figures import PLOT_PARAMS, display, efficiency_table, has_full_family, load_summary, onion_models, onion_summary, run_path


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def significance_mark(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def holm_within(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    """Re-apply Holm within a family after merging tests from several runs."""
    result = frame.copy()
    result["p_value_holm"] = np.nan
    for _, indices in result.groupby(family_columns, sort=False).groups.items():
        ordered = result.loc[indices, "p_value"].sort_values().index.tolist()
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * float(result.at[index, "p_value"])))
            result.at[index, "p_value_holm"] = previous
    return result


def write_markdown(frame: pd.DataFrame, path: Path, title: str, note: str | None = None) -> None:
    lines = [f"### {title}", "", "| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    if note:
        lines.extend(["", note])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SIGNIFICANCE_NOTE = "Paired label-level bootstrap (5,000 resamples) on the five-seed mean predictions; Holm-adjusted within each condition and metric family. *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant."


# ---------------------------------------------------------------------------
# Tables 1-3: Onion feature-matched family
# ---------------------------------------------------------------------------
def table_onion_conditions(project_root: Path, output_dir: Path) -> None:
    summary = onion_summary(project_root, "full")
    family, extra = onion_models(project_root)
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    rows = []
    for model in family + extra:
        subset = summary.loc[(summary["model"] == model) & (summary["metric"] == "mAP")]
        row = {"Model": display(model)}
        for condition in conditions:
            entry = subset.loc[subset["condition"] == condition]
            row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0])) if not entry.empty else "—"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_1_onion_mAP_by_condition.csv", index=False)
    features = "all models use the full 1,034/1,000/4,096-d features" if has_full_family(project_root) else "compact-feature ablations"
    write_markdown(frame, output_dir / "Table_1_onion_mAP_by_condition.md", f"Table 1. Onion test mAP (mean ± std over five seeds) under the six availability conditions; {features}")


def table_onion_observed(project_root: Path, output_dir: Path) -> None:
    summary = onion_summary(project_root, "full")
    family, extra = onion_models(project_root)
    rows = []
    for model in family + extra:
        subset = summary.loc[(summary["model"] == model) & (summary["condition"] == "observed")]
        row = {"Model": display(model)}
        for metric, column in (("mAP", "mAP"), ("MacroF1", "Macro-F1"), ("MicroF1", "Micro-F1"), ("MacroROC_AUC", "Macro ROC-AUC")):
            entry = subset.loc[subset["metric"] == metric]
            row[column] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0])) if not entry.empty else "—"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_2_onion_observed_all_metrics.csv", index=False)
    write_markdown(frame, output_dir / "Table_2_onion_observed_all_metrics.md", "Table 2. Onion observed-condition results (mean ± std over five seeds)")


def onion_mce_tests(project_root: Path) -> pd.DataFrame:
    """MCE-vs-baseline tests merged from the frozen runs, Holm re-applied."""
    frames = []
    if has_full_family(project_root):
        sources = (("ablation", None), ("advanced", ["AttnFusion", "EvidFusion"]))
    else:
        sources = (("final", None), ("advanced", None))
    for key, baselines in sources:
        path = run_path(project_root, key) / "paired_label_bootstrap_holm.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame.loc[frame["proposed"] == "OptimizedEnsemble"]
        if baselines is not None:
            frame = frame.loc[frame["baseline"].isin(baselines)]
        frame = frame.assign(source=key)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["condition", "metric", "baseline"], keep="first")
    merged = merged.dropna(subset=["p_value"])
    return holm_within(merged, ["condition", "metric"])


def table_significance(project_root: Path, output_dir: Path) -> None:
    tests = onion_mce_tests(project_root)
    family, extra = onion_models(project_root)
    order = [m for m in family + extra if m != "OptimizedEnsemble"]
    for metric, suffix, title in (("AP", "", "AP"), ("F1", "_F1", "F1")):
        subset = tests.loc[tests["metric"] == metric]
        rows = []
        for baseline in order:
            entries = subset.loc[subset["baseline"] == baseline]
            if entries.empty:
                continue
            row = {"Baseline": display(baseline)}
            for condition, label in zip(PLOT_PARAMS["conditions_onion"], PLOT_PARAMS["conditions_onion_labels"], strict=True):
                entry = entries.loc[entries["condition"] == condition]
                row[label] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / f"Table_3{suffix}_onion_significance.csv", index=False)
        write_markdown(
            frame,
            output_dir / f"Table_3{suffix}_onion_significance.md",
            f"Table 3{'' if not suffix else 'b'}. Per-tag {title} delta of MCE vs. each baseline on Onion (Holm re-applied across all baselines within each condition)",
            SIGNIFICANCE_NOTE,
        )


# ---------------------------------------------------------------------------
# Tables 4-5: FMA transfer
# ---------------------------------------------------------------------------
def table_fma(project_root: Path, output_dir: Path) -> None:
    path = run_path(project_root, "fma") / "metrics_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    conditions = PLOT_PARAMS["conditions_fma"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_fma_labels"], strict=True))
    rows = []
    for model in PLOT_PARAMS["fig8_bar_models"]:
        subset = summary.loc[(summary["model"] == model) & (summary["metric"] == "mAP")]
        if subset.empty:
            continue
        row = {"Model": display(model)}
        for condition in conditions:
            entry = subset.loc[subset["condition"] == condition]
            row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0])) if not entry.empty else "—"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_4_fma_mAP_by_condition.csv", index=False)
    write_markdown(frame, output_dir / "Table_4_fma_mAP_by_condition.md", "Table 4. FMA cross-dataset test mAP (mean ± std over five seeds) with the locked recipe")

    tests_path = run_path(project_root, "fma") / "paired_label_bootstrap_holm.csv"
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        tests = tests.loc[tests["metric"] == "AP"]
        rows = []
        for baseline in [m for m in PLOT_PARAMS["fig8_bar_models"] if m != "MCE"]:
            subset = tests.loc[tests["baseline"] == baseline]
            if subset.empty:
                continue
            row = {"Baseline": display(str(baseline))}
            for condition in conditions:
                entry = subset.loc[subset["condition"] == condition]
                row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "Table_5_fma_significance.csv", index=False)
        write_markdown(
            frame,
            output_dir / "Table_5_fma_significance.md",
            "Table 5. FMA per-label AP delta of MCE vs. each baseline",
            SIGNIFICANCE_NOTE + " Negative deltas mean the baseline is better.",
        )


# ---------------------------------------------------------------------------
# Table 6: efficiency
# ---------------------------------------------------------------------------
def table_efficiency(project_root: Path, output_dir: Path) -> None:
    efficiency = efficiency_table(project_root).set_index("model")
    summary = onion_summary(project_root, "full")
    family, extra = onion_models(project_root)
    rows = []
    for model in family + extra:
        if model not in efficiency.index:
            continue
        entry = efficiency.loc[model]
        observed = summary.loc[(summary["model"] == model) & (summary["condition"] == "observed") & (summary["metric"] == "mAP")]
        rows.append(
            {
                "Model": display(model),
                "Parameters (M)": f"{entry['parameters'] / 1e6:.2f}",
                "Latency (ms / 1024 tracks)": f"{entry['median_latency_ms']:.2f}",
                "Throughput (tracks/s)": f"{entry['throughput_tracks_per_s']:.0f}",
                "Observed mAP": f"{float(observed['mean'].iloc[0]):.4f}" if not observed.empty else "—",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_6_efficiency.csv", index=False)
    write_markdown(
        frame,
        output_dir / "Table_6_efficiency.md",
        "Table 6. Parameters, warmed batch inference latency (median of 30 repeats, RTX 3090, batch 1,024) and observed mAP of the full-feature model family",
    )


# ---------------------------------------------------------------------------
# Table 7: calibration audit (Onion)
# ---------------------------------------------------------------------------
def table_calibration(project_root: Path, output_dir: Path) -> None:
    analysis = run_path(project_root, "analysis")
    metrics_path = analysis / "onion_calibration_metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["family"] == "full"]
    tests = pd.read_csv(analysis / "onion_calibration_paired_holm.csv")
    tests = tests.loc[tests["family"] == "full"]
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    for metric_column, test_metric, suffix, name in (("MacroECE15", "ECE", "", "Macro-ECE (15 bins)"), ("Brier", "Brier", "b", "Brier score")):
        rows = []
        for model in PLOT_PARAMS["fig7_models"]:
            subset = metrics.loc[metrics["model"] == model]
            if subset.empty:
                continue
            row = {"Model": display(model)}
            for condition in conditions:
                value = float(subset.loc[subset["condition"] == condition, metric_column].iloc[0])
                if model == "OptimizedEnsemble":
                    row[condition_labels[condition]] = f"{value:.4f}"
                else:
                    entry = tests.loc[(tests["baseline"] == model) & (tests["condition"] == condition) & (tests["metric"] == test_metric)]
                    improvement = float(entry["improvement_positive_is_better"].iloc[0])
                    mark = significance_mark(float(entry["p_value_holm"].iloc[0]))
                    row[condition_labels[condition]] = f"{value:.4f} ({'MCE better' if improvement > 0 else 'baseline better'} {mark})"
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / f"Table_7{suffix}_onion_calibration.csv", index=False)
        write_markdown(
            frame,
            output_dir / f"Table_7{suffix}_onion_calibration.md",
            f"Table 7{suffix}. Onion {name} of the five-seed mean predictions per availability condition (lower is better); parentheses give the direction and Holm-adjusted significance of the paired label-level difference against MCE",
            SIGNIFICANCE_NOTE,
        )


# ---------------------------------------------------------------------------
# Table 8: FMA natural-missingness strata
# ---------------------------------------------------------------------------
def table_fma_strata(project_root: Path, output_dir: Path) -> None:
    analysis = run_path(project_root, "analysis")
    strata_path = analysis / "fma_natural_missingness_strata.csv"
    if not strata_path.exists():
        return
    strata = pd.read_csv(strata_path)
    tests = pd.read_csv(analysis / "fma_natural_missingness_paired_holm.csv")
    strata = strata.loc[strata["condition"] == "observed"]
    tests = tests.loc[tests["condition"] == "observed"]
    keys = ["echonest_available", "echonest_unavailable"]
    counts = {key: int(strata.loc[strata["stratum"] == key, "tracks"].iloc[0]) for key in keys}
    labels_evaluated = {key: int(strata.loc[strata["stratum"] == key, "labels_evaluated"].iloc[0]) for key in keys}
    rows = []
    for model in PLOT_PARAMS["fig8_bar_models"]:
        if strata.loc[strata["model"] == model].empty:
            continue
        row = {"Model": display(model)}
        for key in keys:
            value = float(strata.loc[(strata["stratum"] == key) & (strata["model"] == model), "mAP"].iloc[0])
            column = f"Echonest {'present' if key == 'echonest_available' else 'absent'} (n={counts[key]:,}; {labels_evaluated[key]} labels)"
            if model == "MCE":
                row[column] = f"{value:.4f}"
            else:
                entry = tests.loc[(tests["stratum"] == key) & (tests["baseline"] == model)]
                row[column] = f"{value:.4f} (Δ {float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))})"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_8_fma_natural_missingness.csv", index=False)
    write_markdown(
        frame,
        output_dir / "Table_8_fma_natural_missingness.md",
        "Table 8. FMA observed-condition mAP within the natural-missingness strata (test tracks with and without Echonest social descriptors); Δ is the per-label AP gain of MCE over the model",
        SIGNIFICANCE_NOTE + " Only labels with at least one positive and one negative test track inside the stratum are evaluated.",
    )


# ---------------------------------------------------------------------------
# Table 9: full-feature mechanism chain (RAMT as the proposed model)
# ---------------------------------------------------------------------------
def table_mechanism(project_root: Path, output_dir: Path) -> None:
    path = run_path(project_root, "ablation") / "paired_label_bootstrap_holm.csv"
    if not path.exists():
        return
    tests = pd.read_csv(path)
    tests = tests.loc[tests["proposed"] == "RAMT"]
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    rows = []
    for baseline in ["UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate"]:
        for metric in ("AP", "F1"):
            subset = tests.loc[(tests["baseline"] == baseline) & (tests["metric"] == metric)]
            if subset.empty:
                continue
            row = {"Comparison": f"RAMT − {display(baseline)}", "Metric": metric}
            for condition in conditions:
                entry = subset.loc[subset["condition"] == condition]
                row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_9_onion_mechanism_chain.csv", index=False)
    write_markdown(
        frame,
        output_dir / "Table_9_onion_mechanism_chain.md",
        "Table 9. Full-feature mechanism chain on Onion: per-tag AP and F1 deltas of RAMT (reliability gate + modality dropout) against the ablations that remove one mechanism (Holm within condition and metric)",
        SIGNIFICANCE_NOTE,
    )


# ---------------------------------------------------------------------------
# Table S1: compact-feature family and feature-representation tests
# ---------------------------------------------------------------------------
def table_supplementary_compact(project_root: Path, output_dir: Path) -> None:
    if not has_full_family(project_root):
        return
    compact = onion_summary(project_root, "compact")
    full = pd.read_csv(run_path(project_root, "ablation") / "metrics_summary.csv")
    tests_path = run_path(project_root, "analysis") / "onion_feature_representation_paired_holm.csv"
    tests = pd.read_csv(tests_path) if tests_path.exists() else None
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    rows = []
    for model in PLOT_PARAMS["ablation_family"]:
        row = {"Model": display(model)}
        for condition in conditions:
            c = compact.loc[(compact["model"] == model) & (compact["condition"] == condition) & (compact["metric"] == "mAP")]
            f = full.loc[(full["model"] == model) & (full["condition"] == condition) & (full["metric"] == "mAP")]
            if c.empty or f.empty:
                row[condition_labels[condition]] = "—"
                continue
            text = f"{float(c['mean'].iloc[0]):.4f} → {float(f['mean'].iloc[0]):.4f}"
            if tests is not None:
                entry = tests.loc[(tests["model"] == model) & (tests["condition"] == condition)]
                if not entry.empty:
                    text += f" ({float(entry['mean_delta_full_minus_compact'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))})"
            row[condition_labels[condition]] = text
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_S1_feature_representation.csv", index=False)
    write_markdown(
        frame,
        output_dir / "Table_S1_feature_representation.md",
        "Table S1. Feature representation ablation: test mAP of each architecture with compact 256-d CountSketch features (v3, H128) → all raw coordinates (E1-F, H192); parentheses give the paired per-tag AP difference (full − compact) with Holm-adjusted significance",
        SIGNIFICANCE_NOTE,
    )


# ---------------------------------------------------------------------------
# Table 10: tag groups
# ---------------------------------------------------------------------------
def table_tag_groups(project_root: Path, output_dir: Path) -> None:
    analysis = run_path(project_root, "analysis")
    path = analysis / "onion_tag_group_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path)
    tests = pd.read_csv(analysis / "onion_tag_group_paired_holm.csv")
    group_labels = PLOT_PARAMS["fig10_group_labels"]
    groups = [g for g in group_labels if g in set(metrics["group"])]
    observed = metrics.loc[metrics["condition"] == "observed"]
    observed_tests = tests.loc[tests["condition"] == "observed"]
    family, extra = onion_models(project_root)
    rows = []
    for model in family + extra:
        subset = observed.loc[observed["model"] == model]
        if subset.empty:
            continue
        row = {"Model": display(model)}
        for group in groups:
            count = int(subset.loc[subset["group"] == group, "labels"].iloc[0])
            value = float(subset.loc[subset["group"] == group, "mean_AP"].iloc[0])
            entry = observed_tests.loc[(observed_tests["baseline"] == model) & (observed_tests["group"] == group)]
            text = f"{value:.4f}"
            if not entry.empty:
                text += f" (Δ {float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))})"
            row[f"{group_labels[group]} ({count} tags)"] = text
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_10_onion_tag_groups.csv", index=False)
    write_markdown(
        frame,
        output_dir / "Table_10_onion_tag_groups.md",
        "Table 10. Observed-condition mean AP per tag group on Onion (five-seed mean predictions); Δ is the within-group per-tag AP gain of MCE over the model where tested",
        SIGNIFICANCE_NOTE + " Tests use the tags of a group as units, so the six-tag era/origin group has low power.",
    )

    # Retention per group for MCE and the strongest baselines.
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    rows = []
    for model in ["EarlyConcat", "RAMT", "AttnFusion", "EvidFusion", "OptimizedEnsemble"]:
        subset = metrics.loc[metrics["model"] == model]
        if subset.empty:
            continue
        for group in groups:
            base = float(subset.loc[(subset["condition"] == "observed") & (subset["group"] == group), "mean_AP"].iloc[0])
            row = {"Model": display(model), "Group": group_labels[group]}
            for condition in conditions:
                value = float(subset.loc[(subset["condition"] == condition) & (subset["group"] == group), "mean_AP"].iloc[0])
                row[condition_labels[condition]] = f"{value:.4f} ({100.0 * value / base:.1f}%)"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_10b_onion_tag_group_retention.csv", index=False)
    write_markdown(frame, output_dir / "Table_10b_onion_tag_group_retention.md", "Table 10b. Mean AP per tag group under each availability condition (retention relative to the observed condition in parentheses)")


# ---------------------------------------------------------------------------
# Table 11: availability-aware ensemble
# ---------------------------------------------------------------------------
def table_aamce(project_root: Path, output_dir: Path) -> None:
    root = run_path(project_root, "aamce")
    path = root / "metrics_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    tests = pd.read_csv(root / "paired_label_bootstrap_holm.csv")
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    order = ["MCE", "MCE+AAT", "AA-MCE-w", "AA-MCE-2", "AA-MCE", "RAMT", "RAMT+AAT", "AttnFusion", "AttnFusion+AAT", "EvidFusion", "EvidFusion+AAT"]
    for metric, column, suffix in (("mAP", "mAP", ""), ("MacroF1", "Macro-F1", "b")):
        rows = []
        for model in order:
            subset = summary.loc[(summary["model"] == model) & (summary["metric"] == metric)]
            if subset.empty:
                continue
            row = {"Model": display(model)}
            for condition in conditions:
                entry = subset.loc[subset["condition"] == condition]
                row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0]))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / f"Table_11{suffix}_aamce_{metric}.csv", index=False)
        write_markdown(frame, output_dir / f"Table_11{suffix}_aamce_{metric}.md", f"Table 11{suffix}. Availability-aware ensemble on Onion: test {column} (mean ± std over five seeds) of MCE, its availability-aware variants and the strongest baselines with (+AA thresholds) and without availability-aware thresholds")
    # Paired tests of AA-MCE against every other variant.
    rows = []
    subset = tests.loc[tests["proposed"] == "AA-MCE"]
    for baseline in [m for m in order if m != "AA-MCE"]:
        for metric in ("AP", "F1"):
            entries = subset.loc[(subset["baseline"] == baseline) & (subset["metric"] == metric)]
            if entries.empty:
                continue
            row = {"Baseline": display(baseline), "Metric": metric}
            for condition in conditions:
                entry = entries.loc[entries["condition"] == condition]
                row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_11c_aamce_significance.csv", index=False)
    write_markdown(frame, output_dir / "Table_11c_aamce_significance.md", "Table 11c. Per-tag AP and F1 deltas of AA-MCE against each variant and baseline (Holm within condition and metric)", SIGNIFICANCE_NOTE)


# ---------------------------------------------------------------------------
# Table 12: six-view study
# ---------------------------------------------------------------------------
def table_sixview(project_root: Path, output_dir: Path) -> None:
    root = run_path(project_root, "sixview")
    path = root / "metrics_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    tests = pd.read_csv(root / "paired_label_bootstrap_holm.csv")
    conditions = PLOT_PARAMS["conditions_sixview"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_sixview_labels"], strict=True))
    rows = []
    for model in PLOT_PARAMS["fig12_bar_models"]:
        subset = summary.loc[(summary["model"] == model) & (summary["metric"] == "mAP")]
        if subset.empty:
            continue
        row = {"Model": display(model)}
        for condition in conditions:
            entry = subset.loc[subset["condition"] == condition]
            row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0]))
        rows.append(row)
    three = onion_summary(project_root, "full")
    row = {"Model": display("OptimizedEnsemble@3view")}
    for condition in conditions:
        entry = three.loc[(three["model"] == "OptimizedEnsemble") & (three["condition"] == condition) & (three["metric"] == "mAP")]
        row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0])) if not entry.empty else "—"
    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_12_six_view_mAP.csv", index=False)
    write_markdown(frame, output_dir / "Table_12_six_view_mAP.md", "Table 12. Six-view Onion study: test mAP (mean ± std over five seeds) under the seven predeclared availability conditions; the last row is the frozen three-view MCE on the shared sense-level conditions")

    rows = []
    subset = tests.loc[tests["metric"] == "AP"]
    for baseline in [m for m in PLOT_PARAMS["fig12_bar_models"] if m != "MCE"] + ["OptimizedEnsemble@3view"]:
        entries = subset.loc[subset["baseline"] == baseline]
        if entries.empty:
            continue
        row = {"Baseline": display(baseline)}
        for condition in conditions:
            entry = entries.loc[entries["condition"] == condition]
            row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_12b_six_view_significance.csv", index=False)
    write_markdown(frame, output_dir / "Table_12b_six_view_significance.md", "Table 12b. Six-view study: per-tag AP delta of the six-view MCE against every baseline and against the frozen three-view MCE (shared conditions only)", SIGNIFICANCE_NOTE)


# ---------------------------------------------------------------------------
# Table 13: AA-MCE on FMA and six views
# ---------------------------------------------------------------------------
def _aamce_tables(project_root: Path, output_dir: Path, run_key: str, number: str, dataset: str, conditions: list[str], labels: list[str]) -> None:
    root = run_path(project_root, run_key)
    path = root / "metrics_summary.csv"
    if not path.exists():
        return
    slug = dataset.lower().replace(" ", "_").replace("-", "_")
    summary = pd.read_csv(path)
    tests = pd.read_csv(root / "paired_label_bootstrap_holm.csv")
    condition_labels = dict(zip(conditions, labels, strict=True))
    order = ["MCE", "MCE+AAT", "AA-MCE-w", "AA-MCE-2", "AA-MCE", "RAMT", "RAMT+AAT", "AttnFusion", "AttnFusion+AAT", "EvidFusion", "EvidFusion+AAT"]
    rows = []
    for model in order:
        for metric, column in (("mAP", "mAP"), ("MacroF1", "Macro-F1")):
            subset = summary.loc[(summary["model"] == model) & (summary["metric"] == metric)]
            if subset.empty:
                continue
            row = {"Model": display(model), "Metric": column}
            for condition in conditions:
                entry = subset.loc[subset["condition"] == condition]
                row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0]))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"Table_{number}_aamce_{slug}.csv", index=False)
    write_markdown(frame, output_dir / f"Table_{number}_aamce_{slug}.md", f"Table {number}. AA-MCE transferred to {dataset}: test mAP and Macro-F1 (mean ± std over five seeds) of MCE, its availability-aware variants and the controls with (+AA thresholds) and without availability-aware thresholds")
    rows = []
    subset = tests.loc[tests["proposed"] == "AA-MCE"]
    for baseline in [m for m in order if m != "AA-MCE"]:
        for metric in ("AP", "F1"):
            entries = subset.loc[(subset["baseline"] == baseline) & (subset["metric"] == metric)]
            if entries.empty:
                continue
            row = {"Baseline": display(baseline), "Metric": metric}
            for condition in conditions:
                entry = entries.loc[entries["condition"] == condition]
                row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"Table_{number}b_aamce_{slug}_significance.csv", index=False)
    write_markdown(frame, output_dir / f"Table_{number}b_aamce_{slug}_significance.md", f"Table {number}b. {dataset}: per-label AP and F1 deltas of AA-MCE against each variant and control (Holm within condition and metric)", SIGNIFICANCE_NOTE)


def table_aamce_transfer(project_root: Path, output_dir: Path) -> None:
    _aamce_tables(project_root, output_dir, "aamce_fma", "13", "FMA", PLOT_PARAMS["conditions_fma"], PLOT_PARAMS["conditions_fma_labels"])
    _aamce_tables(project_root, output_dir, "aamce_sixview", "13c", "six-view Onion", PLOT_PARAMS["conditions_sixview"], PLOT_PARAMS["conditions_sixview_labels"])
    root = run_path(project_root, "aamce_5c")
    path = root / "metrics_summary.csv"
    if path.exists():
        summary = pd.read_csv(path)
        tests = pd.read_csv(root / "paired_label_bootstrap_holm.csv")
        tests = tests.loc[(tests["proposed"] == "AA-MCE") & (tests["metric"] == "AP")]
        conditions = PLOT_PARAMS["conditions_onion"]
        condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
        rows = []
        for model in ["MCE", "AA-MCE"]:
            subset = summary.loc[(summary["model"] == model) & (summary["metric"] == "mAP")]
            row = {"Model": "AA-MCE, 5 candidates" if model == "AA-MCE" else display(model)}
            for condition in conditions:
                entry = subset.loc[subset["condition"] == condition]
                row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0]))
            rows.append(row)
        three = load_summary(run_path(project_root, "aamce"))
        subset = three.loc[(three["model"] == "AA-MCE") & (three["metric"] == "mAP")]
        row = {"Model": "AA-MCE, 3 candidates"}
        for condition in conditions:
            entry = subset.loc[subset["condition"] == condition]
            row[condition_labels[condition]] = fmt(float(entry["mean"].iloc[0]), float(entry["std"].iloc[0]))
        rows.append(row)
        row = {"Model": "Δ AP: 5 candidates − MCE"}
        for condition in conditions:
            entry = tests.loc[(tests["baseline"] == "MCE") & (tests["condition"] == condition)]
            row[condition_labels[condition]] = "—" if entry.empty else f"{float(entry['mean_delta'].iloc[0]):+.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
        rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "Table_13d_aamce_five_candidates.csv", index=False)
        write_markdown(frame, output_dir / "Table_13d_aamce_five_candidates.md", "Table 13d. Upper-bound ablation: AA-MCE with all five trained architectures as candidate components versus the three-candidate AA-MCE and MCE (Onion, test mAP; last row: per-tag AP delta with Holm-adjusted significance)", SIGNIFICANCE_NOTE)


# ---------------------------------------------------------------------------
# Table 14: post-hoc calibration
# ---------------------------------------------------------------------------
def table_calibration_posthoc(project_root: Path, output_dir: Path) -> None:
    root = run_path(project_root, "calibration")
    path = root / "calibration_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path)
    tests = pd.read_csv(root / "calibration_paired_holm.csv")
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = dict(zip(conditions, PLOT_PARAMS["conditions_onion_labels"], strict=True))
    for column, test_metric, suffix, name in (("MacroECE15", "ECE", "", "Macro-ECE (15 bins)"), ("Brier", "Brier", "b", "Brier score")):
        rows = []
        for model in PLOT_PARAMS["fig14_models"]:
            for variant in ("raw", "cal-obs", "cal-AA"):
                subset = metrics.loc[(metrics["model"] == model) & (metrics["variant"] == variant)]
                if subset.empty:
                    continue
                row = {"Model": display(model), "Variant": variant}
                for condition in conditions:
                    value = float(subset.loc[subset["condition"] == condition, column].iloc[0])
                    text = f"{value:.4f}"
                    if variant != "raw":
                        entry = tests.loc[(tests["family"] == f"{variant}_vs_raw") & (tests["proposed"] == f"{model}/{variant}") & (tests["condition"] == condition) & (tests["metric"] == test_metric)]
                        if not entry.empty:
                            text += f" ({significance_mark(float(entry['p_value_holm'].iloc[0]))})"
                    row[condition_labels[condition]] = text
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / f"Table_14{suffix}_posthoc_calibration.csv", index=False)
        write_markdown(frame, output_dir / f"Table_14{suffix}_posthoc_calibration.md", f"Table 14{suffix}. {name} before (raw) and after validation-fitted per-label Platt scaling, fitted on the observed pattern (cal-obs) or per availability pattern (cal-AA); parentheses give the Holm-adjusted significance of the improvement over raw", SIGNIFICANCE_NOTE)
    rows = []
    subset = tests.loc[tests["family"] == "AA-MCE_cal-AA_vs_others_cal-AA"]
    for model in [m for m in PLOT_PARAMS["fig14_models"] if m != "AA-MCE"]:
        for metric in ("ECE", "Brier"):
            entries = subset.loc[(subset["baseline"] == f"{model}/cal-AA") & (subset["metric"] == metric)]
            if entries.empty:
                continue
            row = {"Comparison": f"AA-MCE − {display(model)} (both AA-calibrated)", "Metric": metric}
            for condition in conditions:
                entry = entries.loc[entries["condition"] == condition]
                value = float(entry["improvement_positive_is_better"].iloc[0])
                row[condition_labels[condition]] = f"{'AA-MCE better' if value > 0 else 'baseline better'} {abs(value):.4f} {significance_mark(float(entry['p_value_holm'].iloc[0]))}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_14c_posthoc_calibration_comparison.csv", index=False)
    write_markdown(frame, output_dir / "Table_14c_posthoc_calibration_comparison.md", "Table 14c. Calibration of AA-MCE versus every baseline after both receive availability-aware Platt scaling (paired label-level differences of ECE and Brier)", SIGNIFICANCE_NOTE)


# ---------------------------------------------------------------------------
# Table 15/16: track-level bootstrap and selection-margin sensitivity
# ---------------------------------------------------------------------------
def table_robustness_checks(project_root: Path, output_dir: Path) -> None:
    root = run_path(project_root, "robustness")
    path = root / "track_level_bootstrap.csv"
    if not path.exists():
        return
    boot = pd.read_csv(path)
    rows = []
    for (dataset, proposed, baseline), block in boot.groupby(["dataset", "proposed", "baseline"], sort=False):
        conditions = PLOT_PARAMS["conditions_onion"] if dataset == "onion" else PLOT_PARAMS["conditions_fma"]
        labels = PLOT_PARAMS["conditions_onion_labels"] if dataset == "onion" else PLOT_PARAMS["conditions_fma_labels"]
        row = {"Dataset": "Onion" if dataset == "onion" else "FMA", "Comparison": f"{display(proposed)} − {display(baseline)}"}
        for condition, label in zip(conditions, labels, strict=True):
            entry = block.loc[block["condition"] == condition]
            row[label] = "—" if entry.empty else f"{float(entry['mean_delta_mAP'].iloc[0]):+.4f} [{float(entry['ci95_low'].iloc[0]):+.4f}, {float(entry['ci95_high'].iloc[0]):+.4f}] {significance_mark(float(entry['p_value'].iloc[0]))}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "Table_15_track_level_bootstrap.csv", index=False)
    write_markdown(frame, output_dir / "Table_15_track_level_bootstrap.md", "Table 15. Track-level bootstrap (1,000 resamples of test tracks) of the mAP difference: point estimate, 95% CI and two-sided bootstrap p-value (uncorrected); complements the label-level tests of Tables 3, 5 and 11", "*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.")
    sensitivity_path = root / "selection_margin_sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        rows = []
        for run, block in sensitivity.groupby("run", sort=False):
            row = {"Run": run, "Patterns": int(block["pattern"].nunique())}
            for margin in sorted(block["margin"].unique()):
                sub = block.loc[block["margin"] == margin]
                ramt = [c for c in sub.columns if c == "w_RAMT"]
                row[f"margin {margin:g}: deviating patterns"] = f"{int(sub['deviates'].sum())} (mean RAMT weight {float(sub[ramt[0]].mean()):.2f})" if ramt else str(int(sub["deviates"].sum()))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "Table_16_selection_margin_sensitivity.csv", index=False)
        write_markdown(frame, output_dir / "Table_16_selection_margin_sensitivity.md", "Table 16. Sensitivity of the AA-MCE weight selection to the predeclared validation margin (validation grids only; no test evaluation)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output_dir = args.project_root / "outputs" / "tables_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = (
        table_onion_conditions,
        table_onion_observed,
        table_significance,
        table_fma,
        table_efficiency,
        table_calibration,
        table_fma_strata,
        table_mechanism,
        table_supplementary_compact,
        table_tag_groups,
        table_aamce,
        table_sixview,
        table_aamce_transfer,
        table_calibration_posthoc,
        table_robustness_checks,
    )
    for builder in builders:
        try:
            builder(args.project_root, output_dir)
            print(f"{builder.__name__}: done", flush=True)
        except (FileNotFoundError, KeyError, IndexError) as error:
            print(f"{builder.__name__}: skipped ({error!r})", flush=True)
    print(json.dumps({"output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
