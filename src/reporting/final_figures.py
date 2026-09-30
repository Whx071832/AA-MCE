"""Final publication figure suite for the redesigned tagging study.

Every figure of the redesigned paper (Fig. 1-9 plus the supplementary
feature-representation figure S1) is produced by this module.  All tunable
choices are centralised in ``PLOT_PARAMS`` below and in the ``figures:``
section of ``configs/experiments.yaml`` (font, widths, colors, dpi).  Change
parameters there instead of editing the plotting logic.

Result sources.  The Onion model family is assembled from four frozen runs:
the full-feature mechanism-ablation family (E1-F), the locked MCE and its two
components (E1), the two recent strong baselines (E2) and, for the
supplementary feature-representation comparison only, the compact-feature
family (v3).  Calibration, robustness aggregates and the FMA natural
missingness strata come from the post-hoc ``paper_analysis`` run (E0/E5).

Usage::

    python -m src.reporting.final_figures --project-root .            # all
    python -m src.reporting.final_figures --project-root . --only 2 3 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .paper_style import apply_paper_style, figure_size, load_config, panel_label, polish_axis, save_figure


# ---------------------------------------------------------------------------
# Central figure parameters.  Everything the figures depend on is set here.
# ---------------------------------------------------------------------------
PLOT_PARAMS: dict[str, Any] = {
    # Result directories (relative to the project root).
    "runs": {
        "v3": "outputs/experiments/robust_multimodal_tagging_v3",
        "ablation": "outputs/experiments/full_feature_ablation_v1",
        "final": "outputs/experiments/optimized_multimodal_tagging_final_v1",
        "validation": "outputs/experiments/optimized_tagging_validation_v2",
        "advanced": "outputs/experiments/advanced_fusion_baselines_v1",
        "fma": "outputs/experiments/fma_cross_dataset_v1",
        "analysis": "outputs/experiments/paper_analysis_v1",
        "aamce": "outputs/experiments/availability_aware_ensemble_v1",
        "sixview": "outputs/experiments/extended_modalities_v1",
        "aamce_fma": "outputs/experiments/availability_aware_ensemble_fma_v1",
        "aamce_sixview": "outputs/experiments/availability_aware_ensemble_sixview_v1",
        "aamce_5c": "outputs/experiments/availability_aware_ensemble_5c_v1",
        "calibration": "outputs/experiments/posthoc_calibration_v1",
        "robustness": "outputs/experiments/robustness_checks_v1",
    },
    "output_dir": "outputs/figures_final",
    # Display names used in every figure.
    "display": {
        "AudioOnly": "Audio only",
        "LyricsOnly": "Lyrics only",
        "VisualOnly": "Visual only",
        "TextOnly": "Text only",
        "SocialOnly": "Social only",
        "UniformMean": "Uniform mean",
        "EarlyConcat": "Early concat",
        "ConcatModDrop": "Concat+MD",
        "ReliabilityGate": "Reliability gate",
        "RAMT": "RAMT",
        "FullConcatMD-BCE": "Concat H192/MD30",
        "FullConcatMD10-H384-BCE": "Concat H384/MD10",
        "ConcatMD10-H384": "Concat H384/MD10",
        "AttnFusion": "AttnFusion",
        "EvidFusion": "EvidFusion",
        "OptimizedEnsemble": "MCE",
        "MCE": "MCE",
        "MCE+AAT": "MCE + AA thresholds",
        "AA-MCE-w": "3-component fixed weights",
        "AA-MCE-2": "AA-MCE (2 components)",
        "AA-MCE": "AA-MCE (ours)",
        "RAMT+AAT": "RAMT + AA thresholds",
        "AttnFusion+AAT": "AttnFusion + AA thresholds",
        "EvidFusion+AAT": "EvidFusion + AA thresholds",
        "AudioEssentiaOnly": "Essentia only",
        "AudioIvecOnly": "i-vector only",
        "LyricsTfidfOnly": "TF-IDF only",
        "LyricsW2vOnly": "word2vec only",
        "VisualResnetOnly": "ResNet only",
        "VisualIncpOnly": "Inception only",
        "OptimizedEnsemble@3view": "MCE (3 views)",
    },
    # Figure 11: availability-aware ensemble.
    "fig11_components": ["FullConcatMD10-H384-BCE", "FullConcatMD-BCE", "RAMT"],
    "fig11_pattern_labels": {
        "audio+lyrics+visual": "A + L + V",
        "audio+lyrics": "A + L",
        "audio+visual": "A + V",
        "lyrics+visual": "L + V",
        "audio": "A only",
        "lyrics": "L only",
        "visual": "V only",
    },
    "fig11_line_models": ["MCE", "MCE+AAT", "AA-MCE", "RAMT", "RAMT+AAT", "AttnFusion+AAT"],
    # Figure 12: six-view study.
    "fig12_bar_models": ["AudioEssentiaOnly", "AudioIvecOnly", "LyricsTfidfOnly", "LyricsW2vOnly", "VisualResnetOnly", "VisualIncpOnly", "UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "AttnFusion", "EvidFusion", "ConcatMD10-H384", "MCE"],
    "fig12_line_models": ["AudioEssentiaOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "AttnFusion", "EvidFusion", "ConcatMD10-H384", "MCE"],
    "conditions_sixview": ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_view_missing", "random_half_views_missing", "keep_one_view"],
    "conditions_sixview_labels": ["Full", "−Audio (2 views)", "−Lyrics (2 views)", "−Visual (2 views)", "Rand drop 1 view", "Rand drop half", "Keep 1 view"],
    "fig12_shared_conditions": ["observed", "no_audio", "no_lyrics", "no_visual"],
    # Figure 13: AA-MCE on FMA and on six views.
    "fig13_models": ["MCE", "MCE+AAT", "AA-MCE", "RAMT+AAT", "AttnFusion+AAT", "EvidFusion+AAT"],
    # Figure 14: post-hoc calibration.
    "fig14_models": ["MCE", "AA-MCE", "RAMT", "AttnFusion", "EvidFusion"],
    # Condition axis labels per dataset.
    "conditions_onion": ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"],
    "conditions_onion_labels": ["Full", "−Audio", "−Lyrics", "−Visual", "Rand drop 1", "Keep only 1"],
    "conditions_fma": ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"],
    "conditions_fma_labels": ["Full", "−Audio", "−Text", "−Social", "Rand drop 1", "Keep only 1"],
    # Model families.
    "ablation_family": ["AudioOnly", "LyricsOnly", "VisualOnly", "UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT"],
    "full_extra_models": ["FullConcatMD-BCE", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion", "OptimizedEnsemble"],
    "fig2_metrics": ["mAP", "MacroF1"],
    # Figure 3: robustness line models and worst-case bar models.
    "fig3_models": ["AudioOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion", "OptimizedEnsemble"],
    "fig3_bar_models": ["UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "FullConcatMD-BCE", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion", "OptimizedEnsemble"],
    "fig3_metric": "mAP",
    # Figure 4 mechanism panels.
    "fig4_gate_model": "RAMT",
    "fig4_dropout_points": {  # validation mAP for H192 capacity across dropout rates
        "FullConcat-BCE": 0.0,
        "FullConcatMD10-BCE": 0.10,
        "FullConcatMD20-BCE": 0.20,
        "FullConcatMD-BCE": 0.30,
    },
    "fig4_capacity_points": {"FullConcatMD10-BCE": 192, "FullConcatMD10-H256-BCE": 256, "FullConcatMD10-H384-BCE": 384},
    # Figure 5 ensemble analysis.
    "fig5_pair": ("FullConcatMD10-H384-BCE", "FullConcatMD-BCE"),
    "fig5_locked_weight": 0.6,
    # Figure 6 per-label gains.
    "fig6_reference_a": "AudioOnly",
    "fig6_reference_b": "FullConcatMD10-H384-BCE",
    "fig6_proposed": "OptimizedEnsemble",
    "fig6_top": 15,
    # Figure 7 calibration.
    "fig7_models": ["EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "AttnFusion", "EvidFusion", "OptimizedEnsemble"],
    "fig7_metrics": [("MacroECE15", "Macro-ECE (15 bins)"), ("Brier", "Brier score")],
    # Figure 8 FMA panels.
    "fig8_bar_models": ["AudioOnly", "TextOnly", "SocialOnly", "UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "AttnFusion", "EvidFusion", "ConcatMD10-H384", "MCE"],
    "fig8_line_models": ["AudioOnly", "TextOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "AttnFusion", "EvidFusion", "MCE"],
    "fig8_strata_models": ["TextOnly", "EarlyConcat", "ConcatModDrop", "RAMT", "AttnFusion", "EvidFusion", "MCE"],
    "fig8_strata_labels": {"echonest_available": "Echonest social\ndescriptors present", "echonest_unavailable": "Echonest social\ndescriptors absent"},
    # Figure 9 efficiency.
    "fig9_models": ["UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "FullConcatMD-BCE", "FullConcatMD10-H384-BCE", "AttnFusion", "EvidFusion", "OptimizedEnsemble"],
    # Supplementary figure S1: compact vs full features per architecture.
    "figs1_conditions": [("observed", "Full"), ("random_two_missing", "Keep only 1")],
    # Figure 10: tag-group analysis.
    "fig10_models": ["AudioOnly", "EarlyConcat", "RAMT", "AttnFusion", "EvidFusion", "FullConcatMD10-H384-BCE", "OptimizedEnsemble"],
    "fig10_group_labels": {"genre_content": "Genre / content", "era_origin": "Era / origin", "mood_affect": "Mood / affect", "preference_context": "Preference / context"},
    # Height ratios (figure height = width * ratio).
    "height_ratio": {"fig1": 0.30, "fig2": 0.42, "fig3": 0.36, "fig4": 0.46, "fig5": 0.48, "fig6": 0.62, "fig7": 1.45, "fig8": 0.36, "fig9": 0.62, "figs1": 0.48, "fig10": 0.46, "fig11": 0.46, "fig12": 0.36, "fig13": 0.36, "fig14": 0.46},
    "bar_edge": "#FFFFFF",
    "annotate_best": True,
}

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "p"]
LINESTYLES = ["-", "--", "-.", ":"]


def display(name: str) -> str:
    return PLOT_PARAMS["display"].get(name, name)


def run_path(project_root: Path, key: str) -> Path:
    return project_root / PLOT_PARAMS["runs"][key]


def color_for(config: dict[str, Any], model: str) -> str:
    return config["figures"]["models"].get(model, config["figures"]["colors"]["charcoal"])


def load_summary(run_root: Path) -> pd.DataFrame:
    return pd.read_csv(run_root / "metrics_summary.csv")


def summary_value(summary: pd.DataFrame, model: str, condition: str, metric: str) -> tuple[float, float]:
    row = summary.loc[(summary["model"] == model) & (summary["condition"] == condition) & (summary["metric"] == metric)]
    if row.empty:
        raise KeyError(f"missing {model}/{condition}/{metric}")
    return float(row["mean"].iloc[0]), float(row["ci95_across_seeds"].iloc[0])


def has_full_family(project_root: Path) -> bool:
    return (run_path(project_root, "ablation") / "metrics_summary.csv").exists()


def onion_summary(project_root: Path, family: str = "full") -> pd.DataFrame:
    """Merged Onion metric summary.

    ``family="full"`` returns the feature-matched family: the retrained
    full-feature ablation models override the compact-feature models of the
    same name whenever the E1-F run exists.  ``family="compact"`` returns the
    compact-feature (v3) family only.
    """
    frames = []
    for key, tag in (("v3", "compact"), ("ablation", "full"), ("final", "full"), ("advanced", "full")):
        path = run_path(project_root, key) / "metrics_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source"] = key
            frame["family"] = tag
            frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if family == "compact":
        return merged.loc[merged["source"] == "v3"].reset_index(drop=True)
    return merged.drop_duplicates(subset=["model", "condition", "metric"], keep="last").reset_index(drop=True)


def onion_models(project_root: Path) -> tuple[list[str], list[str]]:
    """(ablation family, remaining full-feature models) present in the summary."""
    summary = onion_summary(project_root)
    family = [m for m in PLOT_PARAMS["ablation_family"] if not summary.loc[summary["model"] == m].empty]
    extra = [m for m in PLOT_PARAMS["full_extra_models"] if not summary.loc[summary["model"] == m].empty]
    return family, extra


def per_label_observed(project_root: Path) -> pd.DataFrame:
    """Per-label observed-condition metrics of the feature-matched family."""
    frames = []
    for key in ("final", "ablation", "advanced"):
        path = run_path(project_root, key) / "per_label_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source"] = key
            frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["model", "condition", "label"], keep="last")
    return merged.loc[merged["condition"] == "observed"]


# ---------------------------------------------------------------------------
# Figure 1: data and protocol overview (wide, 190 mm)
# ---------------------------------------------------------------------------
def figure_1(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    labels_payload = json.loads((run_path(project_root, "v3") / "labels.json").read_text(encoding="utf-8"))
    frequencies = labels_payload["train_frequencies"]
    onion_manifest = json.loads((run_path(project_root, "v3") / "run_manifest.json").read_text(encoding="utf-8"))
    fma_manifest_path = run_path(project_root, "fma") / "run_manifest.json"
    fma_manifest = json.loads(fma_manifest_path.read_text(encoding="utf-8")) if fma_manifest_path.exists() else None

    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig1"])
    figure, axes = plt.subplots(1, 4, figsize=(width, height), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05, 0.95]})
    colors = config["figures"]["colors"]
    small = config["figures"]["font_size_pt"] - 1.4

    axis = axes[0]
    values = sorted(frequencies.values(), reverse=True)
    axis.bar(np.arange(len(values)), values, width=0.85, color=colors["blue"], edgecolor="none")
    axis.set_yscale("log")
    axis.set_xlabel("Tag rank (50 training tags)")
    axis.set_ylabel("Training-track frequency")
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    splits = ["train", "validation", "test"]
    onion_cov = onion_manifest["coverage"]
    modality_names = ["audio", "lyrics", "visual"]
    modality_colors = [colors["blue"], colors["orange"], colors["green"]]
    positions = np.arange(len(splits))
    bar_width = 0.26
    for index, modality in enumerate(modality_names):
        heights = [100.0 * onion_cov[split][modality] for split in splits]
        axis.bar(positions + (index - 1) * bar_width, heights, width=bar_width, color=modality_colors[index], label=modality.capitalize(), edgecolor="none")
    axis.set_xticks(positions, ["Train", "Val.", "Test"])
    axis.set_ylim(0, 124)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_ylabel("Modality coverage (%)")
    axis.legend(frameon=False, ncol=3, loc="upper center", columnspacing=0.7, handlelength=1.0, handletextpad=0.4, fontsize=small)
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)

    axis = axes[2]
    datasets = [("Onion", onion_manifest["samples"], onion_manifest["artists"])]
    if fma_manifest is not None:
        datasets.append(("FMA", fma_manifest["samples"], fma_manifest["artists"]))
    group_positions = np.arange(len(splits))
    for index, (name, samples, artists) in enumerate(datasets):
        offset = (index - (len(datasets) - 1) / 2) * 0.38
        track_values = [samples[split] for split in splits]
        axis.bar(group_positions + offset, track_values, width=0.34, color=[colors["blue"], colors["sky"]][index], label=f"{name} tracks", edgecolor="none")
        for position, (track_count, artist_count) in enumerate(zip(track_values, [artists[split] for split in splits], strict=True)):
            axis.text(group_positions[position] + offset, track_count * 1.12, f"{artist_count:,}\nartists", ha="center", va="bottom", fontsize=small - 1.0, color="#333333")
    axis.set_yscale("log")
    axis.set_ylim(top=axis.get_ylim()[1] * 30)
    axis.set_xticks(group_positions, ["Train", "Val.", "Test"])
    axis.set_ylabel("Tracks (log scale)")
    axis.legend(frameon=False, loc="upper right", fontsize=small, handlelength=1.0)
    panel_label(axis, "(c)", config=config)
    polish_axis(axis)

    # (d) predeclared availability-condition matrix shared by both datasets.
    axis = axes[3]
    conditions = PLOT_PARAMS["conditions_onion_labels"]
    modality_labels = ["M1\naudio", "M2\nlyrics /\ntext", "M3\nvisual /\nsocial"]
    states = [
        [1, 1, 1],  # observed
        [0, 1, 1],  # no M1
        [1, 0, 1],  # no M2
        [1, 1, 0],  # no M3
        [2, 2, 2],  # random one missing
        [3, 3, 3],  # keep only one (random)
    ]
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(len(conditions) - 0.5, -0.5)
    for row, state_row in enumerate(states):
        for column, state in enumerate(state_row):
            if state == 1:
                patch = Rectangle((column - 0.42, row - 0.38), 0.84, 0.76, facecolor=colors["blue"], edgecolor="none")
            elif state == 0:
                patch = Rectangle((column - 0.42, row - 0.38), 0.84, 0.76, facecolor="white", edgecolor="#999999", linewidth=0.6)
            else:
                patch = Rectangle(
                    (column - 0.42, row - 0.38), 0.84, 0.76, facecolor=colors["sky"] if state == 2 else colors["light_gray"], edgecolor="#777777", linewidth=0.5, hatch="////" if state == 2 else "xxxx"
                )
            axis.add_patch(patch)
    axis.set_xticks(range(3), modality_labels, fontsize=small - 0.6)
    axis.set_yticks(range(len(conditions)), conditions, fontsize=small)
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors["blue"], edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor=colors["sky"], edgecolor="#777777", hatch="////"),
        Rectangle((0, 0), 1, 1, facecolor=colors["light_gray"], edgecolor="#777777", hatch="xxxx"),
    ]
    axis.legend(
        handles,
        ["kept", "removed", "one removed at random", "one kept at random"],
        frameon=False,
        fontsize=small - 0.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
    )
    panel_label(axis, "(d)", config=config)

    figure.tight_layout(w_pad=1.4)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_1_data_protocol_overview",
        metadata={"figure": 1, "content": "tag frequencies; modality coverage; artist-disjoint split sizes; availability-condition matrix"},
    )


# ---------------------------------------------------------------------------
# Figure 2: Onion observed-condition main results (wide)
# ---------------------------------------------------------------------------
def figure_2(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    full = has_full_family(project_root)
    summary = onion_summary(project_root, "full")
    family, extra = onion_models(project_root)
    models = family + extra
    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig2"])
    figure, axes = plt.subplots(1, len(PLOT_PARAMS["fig2_metrics"]), figsize=(width, height))
    for metric_index, metric in enumerate(PLOT_PARAMS["fig2_metrics"]):
        axis = axes[metric_index]
        means, cis, bar_colors = [], [], []
        for model in models:
            mean, ci = summary_value(summary, model, "observed", metric)
            means.append(mean)
            cis.append(ci)
            bar_colors.append(color_for(config, model))
        positions = np.arange(len(models))
        axis.bar(positions, means, yerr=cis, capsize=1.6, color=bar_colors, edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.4, error_kw={"elinewidth": 0.7})
        if not full:
            axis.axvspan(len(family) - 0.5, len(models) - 0.5, color="#F2F2F2", zorder=0)
            axis.text(len(family) / 2 - 0.5, max(means) * 1.10, "Compact features (256-d sketch)", ha="center", fontsize=config["figures"]["font_size_pt"] - 1.4, color="#555555")
            axis.text(len(family) + len(extra) / 2 - 0.5, max(means) * 1.10, "Full features", ha="center", fontsize=config["figures"]["font_size_pt"] - 1.4, color="#555555")
        else:
            chain_start = len(family) - 4  # EarlyConcat -> ConcatModDrop -> ReliabilityGate -> RAMT
            axis.axvspan(chain_start - 0.5, len(family) - 0.5, color="#F4F4F4", zorder=0)
            axis.text(chain_start + 1.5, max(means) * 1.12, "Mechanism\nablations", ha="center", va="bottom", fontsize=config["figures"]["font_size_pt"] - 1.6, color="#555555")
            axis.axvspan(len(family) - 0.5, len(models) - 0.5, color="#E9E9E9", zorder=0)
            axis.text(len(family) + len(extra) / 2 - 0.5, max(means) * 1.12, "Components,\nrecent baselines, MCE", ha="center", va="bottom", fontsize=config["figures"]["font_size_pt"] - 1.6, color="#555555")
        if PLOT_PARAMS["annotate_best"]:
            best = int(np.argmax(means))
            axis.text(best, means[best] + cis[best] + 0.004, f"{means[best]:.3f}", ha="center", va="bottom", fontsize=config["figures"]["font_size_pt"] - 1.2, fontweight="bold")
        axis.set_xticks(positions, [display(model) for model in models], rotation=42, ha="right")
        axis.set_ylabel({"mAP": "mAP", "MacroF1": "Macro-F1"}.get(metric, metric))
        axis.set_ylim(0, max(means) * (1.34 if full else 1.22))
        panel_label(axis, f"({'ab'[metric_index]})", config=config)
        polish_axis(axis)
    figure.tight_layout(w_pad=1.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_2_onion_main_results",
        metadata={"figure": 2, "models": models, "condition": "observed", "feature_matched": full},
    )


# ---------------------------------------------------------------------------
# Figure 3: missing-modality robustness on Onion (wide)
# ---------------------------------------------------------------------------
def figure_3(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    summary = onion_summary(project_root, "full")
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = PLOT_PARAMS["conditions_onion_labels"]
    metric = PLOT_PARAMS["fig3_metric"]
    models = [m for m in PLOT_PARAMS["fig3_models"] if not summary.loc[summary["model"] == m].empty]
    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig3"])
    figure, axes = plt.subplots(1, 3, figsize=(width, height), gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]})
    positions = np.arange(len(conditions))
    for model_index, model in enumerate(models):
        values = [summary_value(summary, model, condition, metric)[0] for condition in conditions]
        for axis_index, series in enumerate((values, [100.0 * value / values[0] for value in values])):
            axes[axis_index].plot(
                positions,
                series,
                marker=MARKERS[model_index % len(MARKERS)],
                markersize=config["figures"]["marker_size_pt"] - 0.6,
                linewidth=config["figures"]["line_width_pt"],
                linestyle=LINESTYLES[model_index % len(LINESTYLES)],
                color=color_for(config, model),
                label=display(model),
            )
    axes[0].set_ylabel(metric)
    axes[1].set_ylabel(f"{metric} retention vs. full (%)")
    axes[1].axhline(100.0, color="#AAAAAA", linewidth=0.6, linestyle=":")
    for axis_index in range(2):
        axes[axis_index].set_xticks(positions, condition_labels, rotation=28, ha="right")
        polish_axis(axes[axis_index])
    axes[0].legend(frameon=False, ncol=2, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="lower left", handlelength=1.6, columnspacing=0.8)

    # (c) mean over the five missing conditions and worst missing condition.
    axis = axes[2]
    bar_models = [m for m in PLOT_PARAMS["fig3_bar_models"] if not summary.loc[summary["model"] == m].empty]
    missing = conditions[1:]
    mean_missing = [np.mean([summary_value(summary, m, c, metric)[0] for c in missing]) for m in bar_models]
    worst = [np.min([summary_value(summary, m, c, metric)[0] for c in missing]) for m in bar_models]
    bar_positions = np.arange(len(bar_models))
    axis.bar(bar_positions - 0.19, mean_missing, width=0.36, color=[color_for(config, m) for m in bar_models], edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.3, label="Mean of 5 missing conditions")
    axis.bar(bar_positions + 0.19, worst, width=0.36, color=[color_for(config, m) for m in bar_models], alpha=0.45, edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.3, hatch="////", label="Worst missing condition")
    axis.set_xticks(bar_positions, [display(m) for m in bar_models], rotation=42, ha="right")
    axis.set_ylabel(f"{metric} under missingness")
    lower = min(worst) - 0.03
    axis.set_ylim(lower, max(mean_missing) + 0.03)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#777777", edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.45, hatch="////", edgecolor="#FFFFFF"),
    ]
    axis.legend(handles, ["Mean of 5 missing conditions", "Worst missing condition"], frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="upper left", handlelength=1.2)
    polish_axis(axis)
    for axis_index, axis in enumerate(axes):
        panel_label(axis, f"({'abc'[axis_index]})", config=config)
    figure.tight_layout(w_pad=1.6)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_3_onion_missing_modality",
        metadata={"figure": 3, "models": models, "bar_models": bar_models, "metric": metric},
    )


# ---------------------------------------------------------------------------
# Figure 4: mechanism analysis (medium)
# ---------------------------------------------------------------------------
def figure_4(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    gate_source = "ablation" if (run_path(project_root, "ablation") / "gate_weights.csv").exists() else "v3"
    gates = pd.read_csv(run_path(project_root, gate_source) / "gate_weights.csv")
    gates = gates.loc[gates["model"] == PLOT_PARAMS["fig4_gate_model"]]
    sweep = pd.read_csv(run_path(project_root, "validation") / "validation_sweep.csv")

    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig4"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    colors = config["figures"]["colors"]

    axis = axes[0]
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = PLOT_PARAMS["conditions_onion_labels"]
    modality_names = ["audio", "lyrics", "visual"]
    modality_colors = [colors["blue"], colors["orange"], colors["green"]]
    positions = np.arange(len(conditions))
    bar_width = 0.26
    for index, modality in enumerate(modality_names):
        means = [
            gates.loc[(gates["condition"] == condition) & (gates["modality"] == modality), "mean_gate"].mean()
            for condition in conditions
        ]
        axis.bar(positions + (index - 1) * bar_width, means, width=bar_width, color=modality_colors[index], label=modality.capitalize(), edgecolor="none")
    axis.set_xticks(positions, condition_labels, rotation=25, ha="right")
    axis.set_ylabel("Mean reliability-gate weight")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.2)
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    dropout_points = PLOT_PARAMS["fig4_dropout_points"]
    rates, values = [], []
    for model, rate in dropout_points.items():
        row = sweep.loc[sweep["model"] == model]
        if not row.empty:
            rates.append(rate)
            values.append(float(row["mAP"].iloc[0]))
    order = np.argsort(rates)
    axis.plot(np.asarray(rates)[order], np.asarray(values)[order], marker="o", color=colors["blue"], linewidth=config["figures"]["line_width_pt"], label="H192 capacity")
    for model, hidden in PLOT_PARAMS["fig4_capacity_points"].items():
        row = sweep.loc[sweep["model"] == model]
        if not row.empty and hidden != 192:
            axis.scatter([0.10], [float(row["mAP"].iloc[0])], marker={"256": "s", "384": "D"}[str(hidden)], color=colors["vermillion"], zorder=5)
            axis.annotate(f"H{hidden}", (0.10, float(row["mAP"].iloc[0])), textcoords="offset points", xytext=(6, -2), fontsize=config["figures"]["font_size_pt"] - 1.2)
    axis.set_xlabel("Training modality-dropout rate")
    axis.set_ylabel("Validation mAP")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.2, loc="lower left")
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)

    figure.tight_layout(w_pad=1.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_4_mechanism_analysis",
        metadata={"figure": 4, "gate_model": PLOT_PARAMS["fig4_gate_model"], "gate_source": gate_source},
    )


# ---------------------------------------------------------------------------
# Figure 5: multi-capacity ensemble analysis (medium)
# ---------------------------------------------------------------------------
def figure_5(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    search = pd.read_csv(run_path(project_root, "validation") / "validation_ensemble_search.csv")
    left, right = PLOT_PARAMS["fig5_pair"]
    pair = search.loc[
        (search["kind"] == "probability") & (search["left"] == left) & (search["right"] == right)
    ].sort_values("left_weight")
    sweep = pd.read_csv(run_path(project_root, "validation") / "validation_sweep.csv")
    observed = per_label_observed(project_root)
    left_ap = observed.loc[observed["model"] == left].set_index("label")["AP"]
    right_ap = observed.loc[observed["model"] == right].set_index("label")["AP"]
    labels = left_ap.index.intersection(right_ap.index)

    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig5"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    colors = config["figures"]["colors"]

    axis = axes[0]
    axis.plot(pair["left_weight"], pair["validation_mAP"], marker="o", color=colors["vermillion"], linewidth=config["figures"]["line_width_pt"], label="Probability ensemble")
    for model, style in ((left, "--"), (right, ":")):
        row = sweep.loc[sweep["model"] == model]
        if not row.empty:
            axis.axhline(float(row["mAP"].iloc[0]), linestyle=style, linewidth=0.8, color=color_for(config, model), label=display(model))
    locked = PLOT_PARAMS["fig5_locked_weight"]
    locked_row = pair.loc[np.isclose(pair["left_weight"], locked)]
    if not locked_row.empty:
        axis.scatter([locked], [float(locked_row["validation_mAP"].iloc[0])], s=44, facecolors="none", edgecolors="#333333", linewidths=1.0, zorder=6, label="Locked 0.6/0.4")
    axis.set_xlabel(f"Weight of {display(left)}")
    axis.set_ylabel("Validation mAP")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="lower center")
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    x = right_ap.loc[labels].to_numpy()
    y = left_ap.loc[labels].to_numpy()
    winner_left = y >= x
    axis.scatter(x[winner_left], y[winner_left], s=11, color=color_for(config, left), alpha=0.85, label=f"{display(left)} better")
    axis.scatter(x[~winner_left], y[~winner_left], s=11, color=color_for(config, right), alpha=0.85, label=f"{display(right)} better")
    limits = [min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03]
    axis.plot(limits, limits, color="#999999", linewidth=0.6, linestyle="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel(f"Per-tag AP: {display(right)}")
    axis.set_ylabel(f"Per-tag AP: {display(left)}")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="upper left")
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)

    figure.tight_layout(w_pad=1.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_5_ensemble_analysis",
        metadata={"figure": 5, "pair": [left, right]},
    )


# ---------------------------------------------------------------------------
# Figure 6: per-tag gains of the ensemble (medium)
# ---------------------------------------------------------------------------
def figure_6(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    observed = per_label_observed(project_root)
    proposed = observed.loc[observed["model"] == PLOT_PARAMS["fig6_proposed"]].set_index("label")["AP"]
    top = PLOT_PARAMS["fig6_top"]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig6"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    colors = config["figures"]["colors"]
    for axis_index, reference_name in enumerate((PLOT_PARAMS["fig6_reference_a"], PLOT_PARAMS["fig6_reference_b"])):
        axis = axes[axis_index]
        reference = observed.loc[observed["model"] == reference_name].set_index("label")["AP"]
        delta = (proposed - reference).dropna().sort_values(ascending=False)
        shown = pd.concat([delta.head(top), delta.tail(2)])
        bar_colors = [colors["green"] if value >= 0 else colors["vermillion"] for value in shown.to_numpy()]
        positions = np.arange(len(shown))[::-1]
        axis.barh(positions, shown.to_numpy(), color=bar_colors, edgecolor="none", height=0.72)
        axis.axvline(0.0, color="#555555", linewidth=0.7)
        axis.set_yticks(positions, shown.index.tolist(), fontsize=config["figures"]["font_size_pt"] - 1.4)
        axis.set_xlabel(f"ΔAP: {display(PLOT_PARAMS['fig6_proposed'])} − {display(reference_name)}")
        panel_label(axis, f"({'ab'[axis_index]})", config=config)
        polish_axis(axis)
    figure.tight_layout(w_pad=2.4)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_6_per_tag_gains",
        metadata={"figure": 6, "references": [PLOT_PARAMS["fig6_reference_a"], PLOT_PARAMS["fig6_reference_b"]], "feature_matched": has_full_family(project_root)},
    )


# ---------------------------------------------------------------------------
# Figure 7: calibration under missingness (small, two stacked panels)
# ---------------------------------------------------------------------------
def figure_7(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    analysis = run_path(project_root, "analysis") / "onion_calibration_metrics.csv"
    if analysis.exists():
        calibration = pd.read_csv(analysis)
        calibration = calibration.loc[calibration["family"] == "full"]
        source = "analysis"
    else:
        calibration = pd.read_csv(run_path(project_root, "v3") / "calibration_metrics_ensemble.csv")
        source = "v3"
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = PLOT_PARAMS["conditions_onion_labels"]
    models = [m for m in PLOT_PARAMS["fig7_models"] if m in set(calibration["model"])]
    width, height = figure_size(config, "small", height_ratio=PLOT_PARAMS["height_ratio"]["fig7"])
    figure, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    positions = np.arange(len(conditions))
    for axis_index, (metric, label) in enumerate(PLOT_PARAMS["fig7_metrics"]):
        axis = axes[axis_index]
        for model_index, model in enumerate(models):
            values = [
                float(calibration.loc[(calibration["model"] == model) & (calibration["condition"] == condition), metric].iloc[0])
                for condition in conditions
            ]
            axis.plot(
                positions,
                values,
                marker=MARKERS[model_index % len(MARKERS)],
                markersize=config["figures"]["marker_size_pt"] - 0.8,
                linewidth=config["figures"]["line_width_pt"] if model == "OptimizedEnsemble" else config["figures"]["line_width_pt"] - 0.3,
                linestyle=LINESTYLES[model_index % len(LINESTYLES)],
                color=color_for(config, model),
                label=display(model),
            )
        axis.set_ylabel(label)
        panel_label(axis, f"({'ab'[axis_index]})", config=config)
        polish_axis(axis)
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low, low + (high - low) * 1.30)
    axes[0].legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, ncol=2, loc="upper left", handlelength=1.6, columnspacing=0.8)
    axes[1].set_xticks(positions, condition_labels, rotation=30, ha="right")
    figure.tight_layout(h_pad=0.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_7_calibration",
        metadata={"figure": 7, "metrics": [m for m, _ in PLOT_PARAMS["fig7_metrics"]], "models": models, "source": source},
    )


# ---------------------------------------------------------------------------
# Figure 8: FMA cross-dataset transfer (wide)
# ---------------------------------------------------------------------------
def figure_8(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    summary = load_summary(run_path(project_root, "fma"))
    conditions = PLOT_PARAMS["conditions_fma"]
    condition_labels = PLOT_PARAMS["conditions_fma_labels"]
    strata_path = run_path(project_root, "analysis") / "fma_natural_missingness_strata.csv"
    panels = 3 if strata_path.exists() else 2
    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig8"] if panels == 3 else 0.42)
    figure, axes = plt.subplots(1, panels, figsize=(width, height), gridspec_kw={"width_ratios": [1.15, 1.0, 0.85][:panels]})

    axis = axes[0]
    models = [m for m in PLOT_PARAMS["fig8_bar_models"] if not summary.loc[summary["model"] == m].empty]
    means, cis, bar_colors = [], [], []
    for model in models:
        mean, ci = summary_value(summary, model, "observed", "mAP")
        means.append(mean)
        cis.append(ci)
        bar_colors.append(color_for(config, model))
    positions = np.arange(len(models))
    axis.bar(positions, means, yerr=cis, capsize=1.6, color=bar_colors, edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.4, error_kw={"elinewidth": 0.7})
    if PLOT_PARAMS["annotate_best"]:
        best = int(np.argmax(means))
        axis.text(best, means[best] + cis[best] + 0.004, f"{means[best]:.3f}", ha="center", va="bottom", fontsize=config["figures"]["font_size_pt"] - 1.2, fontweight="bold")
    axis.set_xticks(positions, [display(model) for model in models], rotation=42, ha="right")
    axis.set_ylabel("mAP (observed)")
    axis.set_ylim(0, max(means) * 1.18)
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    line_models = [m for m in PLOT_PARAMS["fig8_line_models"] if not summary.loc[summary["model"] == m].empty]
    positions = np.arange(len(conditions))
    for model_index, model in enumerate(line_models):
        values = [summary_value(summary, model, condition, "mAP")[0] for condition in conditions]
        axis.plot(
            positions,
            values,
            marker=MARKERS[model_index % len(MARKERS)],
            markersize=config["figures"]["marker_size_pt"] - 0.6,
            linewidth=config["figures"]["line_width_pt"],
            linestyle=LINESTYLES[model_index % len(LINESTYLES)],
            color=color_for(config, model),
            label=display(model),
        )
    axis.set_xticks(positions, condition_labels, rotation=28, ha="right")
    axis.set_ylabel("mAP")
    axis.legend(frameon=False, ncol=2, fontsize=config["figures"]["font_size_pt"] - 1.7, loc="lower left", handlelength=1.6, columnspacing=0.8)
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)

    if panels == 3:
        strata = pd.read_csv(strata_path)
        strata = strata.loc[strata["condition"] == "observed"]
        strata_models = [m for m in PLOT_PARAMS["fig8_strata_models"] if m in set(strata["model"])]
        stratum_keys = ["echonest_available", "echonest_unavailable"]
        axis = axes[2]
        group_positions = np.arange(len(stratum_keys))
        bar_width = 0.8 / len(strata_models)
        for index, model in enumerate(strata_models):
            values = [float(strata.loc[(strata["stratum"] == key) & (strata["model"] == model), "mAP"].iloc[0]) for key in stratum_keys]
            axis.bar(group_positions + (index - (len(strata_models) - 1) / 2) * bar_width, values, width=bar_width, color=color_for(config, model), edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.3, label=display(model))
        tick_labels = []
        for key in stratum_keys:
            count = int(strata.loc[strata["stratum"] == key, "tracks"].iloc[0])
            tick_labels.append(f"{PLOT_PARAMS['fig8_strata_labels'][key]}\n(n = {count:,})")
        axis.set_xticks(group_positions, tick_labels, fontsize=config["figures"]["font_size_pt"] - 1.0)
        axis.set_ylabel("mAP (observed, per stratum)")
        selected = strata.loc[strata["model"].isin(strata_models), "mAP"]
        axis.set_ylim(max(0.0, float(selected.min()) - 0.05), float(selected.max()) + 0.07)
        axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, ncol=2, loc="upper center", handlelength=1.0, columnspacing=0.7)
        panel_label(axis, "(c)", config=config)
        polish_axis(axis)

    figure.tight_layout(w_pad=1.6)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_8_fma_cross_dataset",
        metadata={"figure": 8, "models": models, "strata_panel": panels == 3},
    )


# ---------------------------------------------------------------------------
# Figure 9: accuracy vs. efficiency (medium, two panels)
# ---------------------------------------------------------------------------
def efficiency_table(project_root: Path) -> pd.DataFrame:
    frames = []
    for key in ("advanced", "ablation"):
        path = run_path(project_root, key) / "efficiency.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source"] = key
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model"], keep="first")


def figure_9(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    efficiency = efficiency_table(project_root).set_index("model")
    summary = onion_summary(project_root, "full")
    models = [m for m in PLOT_PARAMS["fig9_models"] if m in efficiency.index]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig9"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    for index, model in enumerate(models):
        mean, ci = summary_value(summary, model, "observed", "mAP")
        row = efficiency.loc[model]
        style = {
            "marker": MARKERS[index % len(MARKERS)],
            "s": 46 if model == "OptimizedEnsemble" else 30,
            "color": color_for(config, model),
            "edgecolor": "white",
            "linewidth": 0.5,
            "zorder": 5,
            "label": display(model),
        }
        axes[0].scatter(row["parameters"] / 1e6, mean, **style)
        axes[1].scatter(row["median_latency_ms"], mean, **style)
    axes[0].set_xlabel("Parameters (millions)")
    axes[1].set_xlabel("Median batch latency (ms / 1,024 tracks)")
    for axis_index, axis in enumerate(axes):
        axis.set_ylabel("mAP (observed)")
        axis.margins(x=0.15, y=0.18)
        panel_label(axis, f"({'ab'[axis_index]})", config=config)
        polish_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=5, fontsize=config["figures"]["font_size_pt"] - 1.6, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0)
    figure.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_9_efficiency",
        metadata={"figure": 9, "models": models, "source": "E2 and E1-F efficiency audits"},
    )


# ---------------------------------------------------------------------------
# Supplementary Figure S1: compact vs. full features per architecture (medium)
# ---------------------------------------------------------------------------
def figure_s1(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    if not has_full_family(project_root):
        raise FileNotFoundError("full-feature ablation run is required for Fig. S1")
    compact = onion_summary(project_root, "compact")
    full = pd.read_csv(run_path(project_root, "ablation") / "metrics_summary.csv")
    models = [m for m in PLOT_PARAMS["ablation_family"] if not full.loc[full["model"] == m].empty]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["figs1"])
    figure, axes = plt.subplots(1, len(PLOT_PARAMS["figs1_conditions"]), figsize=(width, height), sharey=False)
    colors = config["figures"]["colors"]
    positions = np.arange(len(models))
    for axis_index, (condition, label) in enumerate(PLOT_PARAMS["figs1_conditions"]):
        axis = axes[axis_index]
        compact_values = [summary_value(compact, m, condition, "mAP") for m in models]
        full_values = [summary_value(full, m, condition, "mAP") for m in models]
        axis.bar(positions - 0.19, [v for v, _ in compact_values], yerr=[c for _, c in compact_values], width=0.36, color=colors["gray"], capsize=1.2, error_kw={"elinewidth": 0.6}, label="Compact 256-d sketch, H128 (v3)")
        axis.bar(positions + 0.19, [v for v, _ in full_values], yerr=[c for _, c in full_values], width=0.36, color=colors["blue"], capsize=1.2, error_kw={"elinewidth": 0.6}, label="Full features, H192 (E1-F)")
        axis.set_xticks(positions, [display(m) for m in models], rotation=42, ha="right")
        axis.set_ylabel(f"mAP ({label})")
        lower = min(v for v, _ in compact_values + full_values) - 0.04
        axis.set_ylim(max(0.0, lower), max(v for v, _ in compact_values + full_values) + (0.07 if axis_index == 0 else 0.03))
        panel_label(axis, f"({'ab'[axis_index]})", config=config)
        polish_axis(axis)
    axes[0].legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="upper left")
    figure.tight_layout(w_pad=1.8)
    save_figure(
        figure,
        config=config,
        output_stem=output_dir / "Fig_S1_feature_representation",
        metadata={"figure": "S1", "models": models},
    )


# ---------------------------------------------------------------------------
# Figure 10: tag-group analysis (medium)
# ---------------------------------------------------------------------------
def figure_10(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    analysis = run_path(project_root, "analysis")
    metrics = pd.read_csv(analysis / "onion_tag_group_metrics.csv")
    groups = list(config["tag_groups"].keys())
    group_labels = PLOT_PARAMS["fig10_group_labels"]
    models = [m for m in PLOT_PARAMS["fig10_models"] if m in set(metrics["model"])]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig10"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1.15, 1.0]})

    axis = axes[0]
    observed = metrics.loc[metrics["condition"] == "observed"]
    positions = np.arange(len(groups))
    bar_width = 0.8 / len(models)
    for index, model in enumerate(models):
        values = [float(observed.loc[(observed["model"] == model) & (observed["group"] == group), "mean_AP"].iloc[0]) for group in groups]
        axis.bar(positions + (index - (len(models) - 1) / 2) * bar_width, values, width=bar_width, color=color_for(config, model), edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.3, label=display(model))
    counts = {group: int(observed.loc[observed["group"] == group, "labels"].iloc[0]) for group in groups}
    axis.set_xticks(positions, [f"{group_labels[group].replace(' / ', '/' + chr(10))}\n({counts[group]} tags)" for group in groups], fontsize=config["figures"]["font_size_pt"] - 1.2)
    axis.set_ylabel("Mean AP (observed)")
    axis.set_ylim(0, max(observed.loc[observed["model"].isin(models), "mean_AP"]) * 1.32)
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, ncol=2, loc="upper right", handlelength=1.0, columnspacing=0.7)
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = PLOT_PARAMS["conditions_onion_labels"]
    proposed = metrics.loc[metrics["model"] == PLOT_PARAMS["fig6_proposed"]]
    colors = config["figures"]["colors"]
    group_colors = [colors["blue"], colors["green"], colors["orange"], colors["vermillion"]]
    for index, group in enumerate(groups):
        values = [float(proposed.loc[(proposed["condition"] == condition) & (proposed["group"] == group), "mean_AP"].iloc[0]) for condition in conditions]
        retention = [100.0 * value / values[0] for value in values]
        axis.plot(np.arange(len(conditions)), retention, marker=MARKERS[index], markersize=config["figures"]["marker_size_pt"] - 0.6, linewidth=config["figures"]["line_width_pt"], linestyle=LINESTYLES[index % len(LINESTYLES)], color=group_colors[index % len(group_colors)], label=group_labels[group])
    axis.axhline(100.0, color="#AAAAAA", linewidth=0.6, linestyle=":")
    axis.set_xticks(np.arange(len(conditions)), condition_labels, rotation=28, ha="right")
    axis.set_ylabel(f"{display(PLOT_PARAMS['fig6_proposed'])} AP retention (%)")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="lower left")
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)
    figure.tight_layout(w_pad=1.8)
    save_figure(figure, config=config, output_stem=output_dir / "Fig_10_tag_groups", metadata={"figure": 10, "groups": groups, "models": models})


# ---------------------------------------------------------------------------
# Figure 11: availability-aware multi-capacity ensemble (medium)
# ---------------------------------------------------------------------------
def figure_11(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    root = run_path(project_root, "aamce")
    weights = pd.read_csv(root / "pattern_weights.csv")
    weights = weights.loc[weights["variant"] == "AA-MCE"]
    summary = load_summary(root)
    components = PLOT_PARAMS["fig11_components"]
    pattern_labels = PLOT_PARAMS["fig11_pattern_labels"]
    patterns = [p for p in pattern_labels if p in set(weights["pattern"])]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig11"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [0.9, 1.1]})

    axis = axes[0]
    positions = np.arange(len(patterns))[::-1]
    left = np.zeros(len(patterns))
    for component in components:
        values = np.asarray([float(weights.loc[weights["pattern"] == pattern, f"w_{component}"].iloc[0]) for pattern in patterns])
        axis.barh(positions, values, left=left, height=0.7, color=color_for(config, component), edgecolor="white", linewidth=0.4, label=display(component))
        for position, value, start in zip(positions, values, left, strict=True):
            if value >= 0.15:
                axis.text(start + value / 2, position, f"{value:.1f}", ha="center", va="center", fontsize=config["figures"]["font_size_pt"] - 2.0, color="white")
        left += values
    axis.set_yticks(positions, [pattern_labels[p] for p in patterns])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Selected mixing weight (validation only)")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.0, columnspacing=0.8)
    axis.grid(False)
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    conditions = PLOT_PARAMS["conditions_onion"]
    condition_labels = PLOT_PARAMS["conditions_onion_labels"]
    models = [m for m in PLOT_PARAMS["fig11_line_models"] if not summary.loc[summary["model"] == m].empty]
    positions = np.arange(len(conditions))
    for index, model in enumerate(models):
        values = [summary_value(summary, model, condition, "MacroF1")[0] for condition in conditions]
        axis.plot(positions, values, marker=MARKERS[index % len(MARKERS)], markersize=config["figures"]["marker_size_pt"] - 0.6, linewidth=config["figures"]["line_width_pt"] + (0.4 if model == "AA-MCE" else 0.0), linestyle=LINESTYLES[index % len(LINESTYLES)], color=color_for(config, model), label=display(model))
    axis.set_xticks(positions, condition_labels, rotation=28, ha="right")
    axis.set_ylabel("Macro-F1")
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, loc="lower left", handlelength=1.6)
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, config=config, output_stem=output_dir / "Fig_11_availability_aware_ensemble", metadata={"figure": 11, "components": components, "models": models})


# ---------------------------------------------------------------------------
# Figure 12: six-view extended-modality study (wide)
# ---------------------------------------------------------------------------
def figure_12(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    root = run_path(project_root, "sixview")
    summary = load_summary(root)
    three_view = onion_summary(project_root, "full")
    conditions = PLOT_PARAMS["conditions_sixview"]
    condition_labels = PLOT_PARAMS["conditions_sixview_labels"]
    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig12"])
    figure, axes = plt.subplots(1, 3, figsize=(width, height), gridspec_kw={"width_ratios": [1.3, 1.0, 0.8]})

    axis = axes[0]
    models = [m for m in PLOT_PARAMS["fig12_bar_models"] if not summary.loc[summary["model"] == m].empty]
    means, cis, bar_colors = [], [], []
    for model in models:
        mean, ci = summary_value(summary, model, "observed", "mAP")
        means.append(mean)
        cis.append(ci)
        bar_colors.append(color_for(config, model))
    positions = np.arange(len(models))
    axis.bar(positions, means, yerr=cis, capsize=1.4, color=bar_colors, edgecolor=PLOT_PARAMS["bar_edge"], linewidth=0.4, error_kw={"elinewidth": 0.7})
    reference, _ = summary_value(three_view, "OptimizedEnsemble", "observed", "mAP")
    axis.axhline(reference, color=color_for(config, "OptimizedEnsemble"), linestyle="--", linewidth=0.9, label=f"MCE, 3 views ({reference:.3f})")
    if PLOT_PARAMS["annotate_best"]:
        best = int(np.argmax(means))
        axis.text(best, means[best] + cis[best] + 0.004, f"{means[best]:.3f}", ha="center", va="bottom", fontsize=config["figures"]["font_size_pt"] - 1.2, fontweight="bold")
    axis.set_xticks(positions, [display(model) for model in models], rotation=42, ha="right")
    axis.set_ylabel("mAP (observed, 6 views)")
    axis.set_ylim(0, max(means) * 1.20)
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="upper left")
    panel_label(axis, "(a)", config=config)
    polish_axis(axis)

    axis = axes[1]
    line_models = [m for m in PLOT_PARAMS["fig12_line_models"] if not summary.loc[summary["model"] == m].empty]
    positions = np.arange(len(conditions))
    for index, model in enumerate(line_models):
        values = [summary_value(summary, model, condition, "mAP")[0] for condition in conditions]
        axis.plot(positions, values, marker=MARKERS[index % len(MARKERS)], markersize=config["figures"]["marker_size_pt"] - 0.6, linewidth=config["figures"]["line_width_pt"], linestyle=LINESTYLES[index % len(LINESTYLES)], color=color_for(config, model), label=display(model))
    axis.set_xticks(positions, condition_labels, rotation=28, ha="right", fontsize=config["figures"]["font_size_pt"] - 1.0)
    axis.set_ylabel("mAP")
    axis.legend(frameon=False, ncol=2, fontsize=config["figures"]["font_size_pt"] - 1.8, loc="lower left", handlelength=1.5, columnspacing=0.7)
    panel_label(axis, "(b)", config=config)
    polish_axis(axis)

    axis = axes[2]
    shared = PLOT_PARAMS["fig12_shared_conditions"]
    shared_labels = [PLOT_PARAMS["conditions_onion_labels"][PLOT_PARAMS["conditions_onion"].index(c)] for c in shared]
    positions = np.arange(len(shared))
    three = [summary_value(three_view, "OptimizedEnsemble", c, "mAP") for c in shared]
    six = [summary_value(summary, "MCE", c, "mAP") for c in shared]
    axis.bar(positions - 0.19, [v for v, _ in three], yerr=[c for _, c in three], width=0.36, color=color_for(config, "OptimizedEnsemble@3view"), capsize=1.2, error_kw={"elinewidth": 0.6}, label="MCE, 3 views")
    axis.bar(positions + 0.19, [v for v, _ in six], yerr=[c for _, c in six], width=0.36, color=color_for(config, "MCE"), capsize=1.2, error_kw={"elinewidth": 0.6}, label="MCE, 6 views")
    axis.set_xticks(positions, shared_labels, rotation=28, ha="right")
    axis.set_ylabel("mAP")
    lower = min(v for v, _ in three + six) - 0.04
    axis.set_ylim(max(0.0, lower), max(v for v, _ in three + six) + 0.05)
    axis.legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.6, loc="upper right")
    panel_label(axis, "(c)", config=config)
    polish_axis(axis)
    figure.tight_layout(w_pad=1.6)
    save_figure(figure, config=config, output_stem=output_dir / "Fig_12_six_view_study", metadata={"figure": 12, "models": models})


# ---------------------------------------------------------------------------
# Figure 13: AA-MCE on FMA and on six views (wide)
# ---------------------------------------------------------------------------
def _condition_lines(axis, summary: pd.DataFrame, models: list[str], conditions: list[str], labels: list[str], metric: str, config: dict[str, Any], ylabel: str) -> None:
    positions = np.arange(len(conditions))
    for index, model in enumerate(models):
        if summary.loc[summary["model"] == model].empty:
            continue
        values = [summary_value(summary, model, condition, metric)[0] for condition in conditions]
        axis.plot(positions, values, marker=MARKERS[index % len(MARKERS)], markersize=config["figures"]["marker_size_pt"] - 0.6, linewidth=config["figures"]["line_width_pt"] + (0.4 if model == "AA-MCE" else 0.0), linestyle=LINESTYLES[index % len(LINESTYLES)], color=color_for(config, model), label=display(model))
    axis.set_xticks(positions, labels, rotation=28, ha="right", fontsize=config["figures"]["font_size_pt"] - 1.0)
    axis.set_ylabel(ylabel)
    polish_axis(axis)


def figure_13(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    fma_summary = load_summary(run_path(project_root, "aamce_fma"))
    six_summary = load_summary(run_path(project_root, "aamce_sixview"))
    models = PLOT_PARAMS["fig13_models"]
    width, height = figure_size(config, "wide", height_ratio=PLOT_PARAMS["height_ratio"]["fig13"])
    figure, axes = plt.subplots(1, 3, figsize=(width, height))
    _condition_lines(axes[0], fma_summary, models, PLOT_PARAMS["conditions_fma"], PLOT_PARAMS["conditions_fma_labels"], "MacroF1", config, "Macro-F1 (FMA)")
    _condition_lines(axes[1], six_summary, models, PLOT_PARAMS["conditions_sixview"], PLOT_PARAMS["conditions_sixview_labels"], "mAP", config, "mAP (Onion, 6 views)")
    _condition_lines(axes[2], six_summary, models, PLOT_PARAMS["conditions_sixview"], PLOT_PARAMS["conditions_sixview_labels"], "MacroF1", config, "Macro-F1 (Onion, 6 views)")
    axes[0].legend(frameon=False, fontsize=config["figures"]["font_size_pt"] - 1.8, loc="lower left", handlelength=1.6)
    for index, axis in enumerate(axes):
        panel_label(axis, f"({'abc'[index]})", config=config)
    figure.tight_layout(w_pad=1.6)
    save_figure(figure, config=config, output_stem=output_dir / "Fig_13_aamce_transfer", metadata={"figure": 13, "models": models})


# ---------------------------------------------------------------------------
# Figure 14: post-hoc calibration (medium)
# ---------------------------------------------------------------------------
def figure_14(project_root: Path, config: dict[str, Any], output_dir: Path) -> None:
    metrics = pd.read_csv(run_path(project_root, "calibration") / "calibration_metrics.csv")
    conditions = PLOT_PARAMS["conditions_onion"]
    labels = PLOT_PARAMS["conditions_onion_labels"]
    models = [m for m in PLOT_PARAMS["fig14_models"] if m in set(metrics["model"])]
    width, height = figure_size(config, "medium", height_ratio=PLOT_PARAMS["height_ratio"]["fig14"])
    figure, axes = plt.subplots(1, 2, figsize=(width, height))
    positions = np.arange(len(conditions))
    for axis_index, (column, ylabel) in enumerate((("MacroECE15", "Macro-ECE (15 bins)"), ("Brier", "Brier score"))):
        axis = axes[axis_index]
        for index, model in enumerate(models):
            for variant, style, alpha in (("raw", ":", 0.55), ("cal-AA", "-", 1.0)):
                block = metrics.loc[(metrics["model"] == model) & (metrics["variant"] == variant)].set_index("condition")
                values = [float(block.loc[condition, column]) for condition in conditions]
                axis.plot(positions, values, marker=MARKERS[index % len(MARKERS)], markersize=config["figures"]["marker_size_pt"] - 0.8, linewidth=config["figures"]["line_width_pt"], linestyle=style, color=color_for(config, model), alpha=alpha, label=f"{display(model)} ({'raw' if variant == 'raw' else 'AA-calibrated'})" if axis_index == 0 else None)
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.set_ylabel(ylabel)
        panel_label(axis, f"({'ab'[axis_index]})", config=config)
        polish_axis(axis)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=5, fontsize=config["figures"]["font_size_pt"] - 2.0, bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=0.8)
    figure.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.8)
    save_figure(figure, config=config, output_stem=output_dir / "Fig_14_posthoc_calibration", metadata={"figure": 14, "models": models})


FIGURES = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5, 6: figure_6, 7: figure_7, 8: figure_8, 9: figure_9, 10: figure_s1, 11: figure_10, 12: figure_11, 13: figure_12, 14: figure_13, 15: figure_14}
FILE_STEMS = {
    1: "Fig_1_data_protocol_overview",
    2: "Fig_2_onion_main_results",
    3: "Fig_3_onion_missing_modality",
    4: "Fig_4_mechanism_analysis",
    5: "Fig_5_ensemble_analysis",
    6: "Fig_6_per_tag_gains",
    7: "Fig_7_calibration",
    8: "Fig_8_fma_cross_dataset",
    9: "Fig_9_efficiency",
    10: "Fig_S1_feature_representation",
    11: "Fig_10_tag_groups",
    12: "Fig_11_availability_aware_ensemble",
    13: "Fig_12_six_view_study",
    14: "Fig_13_aamce_transfer",
    15: "Fig_14_posthoc_calibration",
}

CAPTIONS = {
    1: "Fig. 1. Datasets and evaluation protocol. (a) Training-frequency distribution of the 50 user-generated tags on Music4All-Onion. (b) Modality coverage per artist-disjoint split. (c) Track and artist counts of the artist-disjoint splits of both datasets (log scale). (d) The six predeclared availability conditions applied identically to every model; random masks are shared across models. [190 mm]",
    2: "Fig. 2. Observed-condition results on Music4All-Onion for (a) mAP and (b) Macro-F1 (mean over five seeds; error bars: 95% CI). All models use the full 1,034/1,000/4,096-d features; the mechanism ablations, the two locked MCE components, the two recent strong baselines and MCE are grouped by shading. [190 mm]",
    3: "Fig. 3. Missing-modality robustness on Music4All-Onion. (a) Absolute mAP under the six predeclared availability conditions. (b) Retention relative to the fully observed condition. (c) Mean over the five missing conditions (solid) and the worst missing condition (hatched) for every fusion model. [190 mm]",
    4: "Fig. 4. Mechanism analysis. (a) Mean reliability-gate weights of the full-feature RAMT per condition. (b) Validation mAP as a function of the training modality-dropout rate and of encoder capacity at 10% dropout. [140 mm]",
    5: "Fig. 5. Multi-capacity ensemble. (a) Validation-only weight search of the probability ensemble against its two components; the circled point is the locked 0.6/0.4 configuration. (b) Complementarity of the components: per-tag test AP. [140 mm]",
    6: "Fig. 6. Per-tag AP gains of the locked ensemble over (a) the full-feature audio-only baseline and (b) the strongest single component (top gains plus the two largest losses). [140 mm]",
    7: "Fig. 7. Calibration under missingness: (a) macro expected calibration error (15 bins) and (b) Brier score of the five-seed mean predictions across availability conditions. [85 mm]",
    8: "Fig. 8. Cross-dataset transfer to FMA genre tagging with the locked recipe. (a) Observed-condition mAP (mean of five seeds; error bars: 95% CI). (b) mAP across availability conditions. (c) Observed-condition mAP within the natural-missingness strata defined by the presence of the Echonest social descriptors. [190 mm]",
    9: "Fig. 9. Accuracy versus efficiency of the full-feature model family: observed mAP against (a) parameter count and (b) median warmed batch inference latency on one RTX 3090. [140 mm]",
    10: "Fig. S1. Feature representation ablation: test mAP of the eight mechanism-ablation architectures trained on the compact 256-d CountSketch features (v3 protocol) versus all raw feature coordinates (E1-F) under (a) the fully observed condition and (b) the keep-only-one condition (mean over five seeds; error bars: 95% CI). [140 mm]",
    11: "Fig. 10. Tag-group analysis. (a) Mean observed-condition AP of the 50 Onion tags grouped into genre/content, era/origin, mood/affect and preference/context tags (group assignment declared in the configuration). (b) AP retention of MCE per tag group across availability conditions. [140 mm]",
    12: "Fig. 11. Availability-aware multi-capacity ensemble. (a) Mixing weights over the two capacity components and the gated RAMT selected on validation tracks masked to each availability pattern (A audio, L lyrics, V visual); the locked 0.6/0.4 mixture is retained unless the validation gain exceeds the predeclared margin. (b) Macro-F1 across availability conditions for MCE, its availability-aware variants and the strongest baselines with the same availability-aware thresholds. [140 mm]",
    13: "Fig. 12. Six-view extended-modality study on Music4All-Onion (Essentia and i-vector audio, TF-IDF and word2vec lyrics, ResNet and Inception visual views; recipe locked from the three-view study). (a) Observed-condition mAP of every model (mean of five seeds; error bars: 95% CI) with the three-view MCE as reference line. (b) mAP across the seven predeclared availability conditions. (c) MCE with three versus six views on the shared sense-level conditions. [190 mm]",
    14: "Fig. 13. Availability-aware ensemble transferred without re-tuning. (a) Macro-F1 on FMA across the six availability conditions. (b) mAP and (c) Macro-F1 on the six-view Onion study across its seven conditions (63 availability patterns). Baselines carry the same availability-aware thresholds as AA-MCE. [190 mm]",
    15: "Fig. 14. Post-hoc per-label Platt scaling fitted on validation only: (a) macro expected calibration error and (b) Brier score of the raw (dotted) and availability-aware calibrated (solid) five-seed mean predictions across conditions. [140 mm]",
}


def write_captions(output_dir: Path) -> None:
    """Captions for every figure whose file exists (robust to ``--only``)."""
    captions = [CAPTIONS[number] for number in sorted(FIGURES) if (output_dir / f"{FILE_STEMS[number]}.pdf").exists()]
    (output_dir / "figure_captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--only", nargs="*", type=int, default=None, help="figure numbers (10 = supplementary Fig. S1)")
    args = parser.parse_args()
    config = load_config(args.project_root, args.config)
    apply_paper_style(config)
    output_dir = args.project_root / PLOT_PARAMS["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(FIGURES)
    produced, skipped = [], []
    for number in selected:
        try:
            FIGURES[number](args.project_root, config, output_dir)
            produced.append(number)
            print(f"figure {number}: done", flush=True)
        except (FileNotFoundError, KeyError) as error:
            skipped.append((number, repr(error)))
            print(f"figure {number}: skipped ({error!r})", flush=True)
    write_captions(output_dir)
    print(json.dumps({"produced": produced, "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
