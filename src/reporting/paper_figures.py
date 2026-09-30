"""Composite figures of the manuscript (Chinese labels, PDF vector + 600 dpi PNG).

Fig. 1  AA-MCE framework and availability protocol
Fig. 2  Main results and missing-modality robustness on Music4All-Onion
Fig. 3  Mechanism analysis (gate weights, dropout/capacity, ensemble weights, pattern weights)
Fig. 4  Cross-dataset (FMA) and six-view transfer
Fig. 5  Probability calibration and tag-semantic analysis

All numbers are read from ``--data`` (default ``outputs/paper/data``, produced by ``paper_data.py``) and
from the frozen experiment directories.  Figure parameters are centralised in
``STYLE`` and ``COLORS``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

EXP = Path("outputs/experiments")
MM = 1 / 25.4
WIDTH = 170 * MM

STYLE = {
    "font.family": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

COLORS = {
    "AudioOnly": "#7A7A7A", "LyricsOnly": "#B0B0B0", "VisualOnly": "#9A9A9A",
    "TextOnly": "#B0B0B0", "SocialOnly": "#9A9A9A",
    "AudioEssentiaOnly": "#7A7A7A", "AudioIvecOnly": "#8E8E8E", "LyricsTfidfOnly": "#A8A8A8",
    "LyricsW2vOnly": "#B8B8B8", "VisualResnetOnly": "#9A9A9A", "VisualIncpOnly": "#C4C4C4",
    "UniformMean": "#56B4E9", "EarlyConcat": "#0072B2", "ConcatModDrop": "#E69F00",
    "ReliabilityGate": "#CC79A7", "RAMT": "#009E73", "H192": "#DDCC77", "H384": "#882255",
    "AttnFusion": "#7B68B5", "EvidFusion": "#937860", "MCE": "#F0A070", "AA-MCE": "#B2182B",
    "MCE+AAT": "#F4A582", "RAMT+AAT": "#009E73", "AttnFusion+AAT": "#7B68B5", "EvidFusion+AAT": "#937860",
    "AA-MCE-5": "#67001F",
}
NAMES = {
    "AudioOnly": "Audio only", "LyricsOnly": "Lyrics only", "VisualOnly": "Visual only",
    "TextOnly": "Text only", "SocialOnly": "Social only",
    "AudioEssentiaOnly": "Essentia only", "AudioIvecOnly": "i-vector only", "LyricsTfidfOnly": "TF-IDF only",
    "LyricsW2vOnly": "word2vec only", "VisualResnetOnly": "ResNet only", "VisualIncpOnly": "Inception only",
    "UniformMean": "Uniform mean", "EarlyConcat": "Early concat", "ConcatModDrop": "Concat+MD",
    "ReliabilityGate": "Reliability gate", "RAMT": "RAMT", "H192": "Concat-H192/MD30", "H384": "Concat-H384/MD10",
    "AttnFusion": "AttnFusion", "EvidFusion": "EvidFusion", "MCE": "MCE", "AA-MCE": "AA-MCE（本文）",
    "MCE+AAT": "MCE+模式阈值", "RAMT+AAT": "RAMT+模式阈值", "AttnFusion+AAT": "AttnFusion+模式阈值",
    "EvidFusion+AAT": "EvidFusion+模式阈值", "AA-MCE-5": "AA-MCE（五候选）",
}
ONION_CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_missing", "random_two_missing"]
ONION_LABELS = ["完整观测", "缺音频", "缺歌词", "缺视觉", "随机缺一", "仅留一"]
FMA_CONDITIONS = ["observed", "no_audio", "no_text", "no_social", "random_one_missing", "random_two_missing"]
FMA_LABELS = ["完整观测", "缺音频", "缺文本", "缺社交", "随机缺一", "仅留一"]
SIX_CONDITIONS = ["observed", "no_audio", "no_lyrics", "no_visual", "random_one_view_missing", "random_half_views_missing", "keep_one_view"]
SIX_LABELS = ["完整观测", "缺音频", "缺歌词", "缺视觉", "随机缺一视图", "随机缺半数", "仅留一视图"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "p"]
LINESTYLES = ["-", "--", "-.", ":"]


def panel(ax, text: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="left", family="Times New Roman")


def value(frame: pd.DataFrame, model: str, condition: str, metric: str, column: str = "mean") -> float:
    row = frame.loc[(frame["model"] == model) & (frame["condition"] == condition) & (frame["metric"] == metric)]
    if row.empty:
        raise KeyError(f"{model}/{condition}/{metric}")
    return float(row[column].iloc[0])


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=600)
    plt.close(fig)
    print("saved", stem, flush=True)


def line_panel(ax, frame, models, conditions, labels, metric, ylabel, emphasize="AA-MCE", legend=True, ncol=2, loc="lower left"):
    x = np.arange(len(conditions))
    for i, model in enumerate(models):
        values = [value(frame, model, c, metric) for c in conditions]
        dashed = model.endswith("+AAT")
        ax.plot(
            x, values,
            marker=MARKERS[i % len(MARKERS)],
            linestyle="--" if dashed else ("-" if model in (emphasize, "MCE") else LINESTYLES[i % len(LINESTYLES)]),
            color=COLORS.get(model, "#333333"),
            linewidth=1.6 if model == emphasize else 0.95,
            markersize=3.6 if model == emphasize else 3.0,
            zorder=5 if model == emphasize else 3,
            label=NAMES.get(model, model),
        )
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, ncol=ncol, loc=loc, handlelength=1.8, columnspacing=0.8, borderaxespad=0.2)


def bar_panel(ax, frame, models, condition, metric, ylabel, reference=None, reference_label=None):
    means = [value(frame, m, condition, metric) for m in models]
    cis = [value(frame, m, condition, metric, "ci95") for m in models]
    x = np.arange(len(models))
    colors = [COLORS.get(m, "#555555") for m in models]
    ax.bar(x, means, yerr=cis, color=colors, width=0.72, edgecolor="white", linewidth=0.4, capsize=1.3, error_kw={"elinewidth": 0.6, "capthick": 0.6})
    best = len(means) - 1 - int(np.argmax(means[::-1]))  # ties go to the rightmost (proposed) model
    ax.text(best, means[best] + cis[best] + 0.006, f"{means[best]:.3f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    if reference is not None:
        ax.axhline(reference, color=COLORS["AA-MCE"], linestyle="--", linewidth=0.8)
        ax.text(0.02, reference + 0.008, reference_label, transform=ax.get_yaxis_transform(), fontsize=6.3, color=COLORS["AA-MCE"])
    ax.set_xticks(x, [NAMES.get(m, m) for m in models], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(means) * 1.16)
    ax.set_xlim(-0.6, len(models) - 0.4)


# ---------------------------------------------------------------------------
# Figure 1: framework and protocol
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, face, edge="#555555", size=6.6, bold=False, lw=0.7):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, start, end, style="-|>", color="#444444", lw=0.7, dashed=False, connection="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=6.5, color=color, linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-", connectionstyle=connection, shrinkA=0, shrinkB=0)
    )


def box2(ax, x, y, w, h, lines, face, edge="#555555", lw=0.7, sizes=None):
    """Rounded box with separately drawn lines (Chinese lines never mix with mathtext)."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    sizes = sizes or [6.6] * len(lines)
    step = h / (len(lines) + 1)
    for i, (text, size) in enumerate(zip(lines, sizes, strict=True)):
        ax.text(x + w / 2, y + h - step * (i + 1), text, ha="center", va="center", fontsize=size, fontweight="bold" if i == 0 else "normal")


def figure_1(out: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, 80 * MM))
    ax = fig.add_axes([0.0, 0.0, 0.745, 1.0])
    ax.set_xlim(0, 131)
    ax.set_ylim(0, 80)
    ax.axis("off")
    for x, text in ((15, "多模态输入"), (55, "组件网络（训练艺术家上训练）"), (104, "推理：按可用模式查表融合")):
        ax.text(x, 77.5, text, ha="center", va="center", fontsize=7.2, fontweight="bold")
    inputs = [(("音频特征", "Essentia，1 034 维"), 61), (("歌词特征", "TF-IDF，1 000 维"), 46), (("视觉特征", "ResNet，4 096 维"), 31)]
    for lines, y in inputs:
        box2(ax, 2, y, 26, 11, list(lines), "#DCEAF7")
    comps = [
        (("C1 拼接网络", "隐层 384，模态丢弃率 0.1"), 61, "#FDE3CF"),
        (("C2 拼接网络", "隐层 192，模态丢弃率 0.3"), 46, "#FDE3CF"),
        (("C3 可靠性门控网络", "隐层 192，模态丢弃率 0.3"), 31, "#D5EFE3"),
    ]
    ax.plot([31, 31], [36.5, 66.5], color="#444444", linewidth=0.7)
    for _, y in inputs:
        ax.plot([28.3, 31], [y + 5.5, y + 5.5], color="#444444", linewidth=0.7)
    for lines, y, face in comps:
        arrow(ax, (31, y + 5.5), (37.7, y + 5.5))
        box2(ax, 38, y, 34, 11, list(lines), face)
    for k, (_, y, _) in enumerate(comps, start=1):
        arrow(ax, (72.3, y + 5.5), (79.7, y + 5.5))
        ax.text(76, y + 7.0, f"$p_{k}$", ha="center", fontsize=6.6)
    box2(ax, 80, 31, 21, 41, ["模式加权融合", "$p_{\\pi}=\\sum_{k} w_{\\pi,k}\\,p_{k}$"], "#F8D7D5", edge="#B2182B", lw=0.9, sizes=[6.8, 7.0])
    box2(ax, 106, 52, 21, 20, ["Platt 校准", "$\\tilde{p}=\\sigma(\\alpha_{\\pi}z+\\beta_{\\pi})$", "$z=\\mathrm{logit}\\,p_{\\pi}$"], "#F8D7D5", edge="#B2182B", lw=0.9, sizes=[6.6, 6.6, 6.2])
    box2(ax, 106, 31, 21, 16, ["标签决策", "$\\hat{y}_{l}=\\mathrm{I}(p_{\\pi,l}\\geq\\tau_{\\pi,l})$"], "#F8D7D5", edge="#B2182B", lw=0.9, sizes=[6.6, 6.4])
    arrow(ax, (101.3, 62), (105.7, 62))
    arrow(ax, (101.3, 39), (105.7, 39))
    box2(ax, 2, 10, 26, 13, ["可用性向量", "$\\mathbf{a}=(a_A,a_L,a_V)$"], "#EFEFEF")
    box2(ax, 38, 10, 20, 13, ["可用模式", "$\\pi(\\mathbf{a})$", "M=3 时共 7 种"], "#EFEFEF", sizes=[6.6, 6.8, 6.0])
    box2(ax, 64, 7, 42, 19, ["模式查找表（每种模式一行）", "$\\mathbf{w}_{\\pi},\\;\\tau_{\\pi,l},\\;(\\alpha_{\\pi,l},\\beta_{\\pi,l})$", "组件权重 · 逐标签阈值 · 校准参数"], "#FFF3C4", edge="#8A6D00", lw=0.9, sizes=[6.6, 6.8, 6.1])
    box2(ax, 110, 7, 17, 19, ["验证艺术家", "按模式掩码", "选择参数"], "#FFFBE6", edge="#8A6D00", sizes=[6.4, 6.2, 6.2])
    arrow(ax, (28.3, 16.5), (37.7, 16.5))
    arrow(ax, (58.3, 16.5), (63.7, 16.5))
    arrow(ax, (109.7, 16.5), (106.3, 16.5))
    arrow(ax, (90, 26.3), (90, 30.7), dashed=True, color="#8A6D00")
    ax.text(88.6, 28.6, "$\\mathbf{w}_{\\pi}$", ha="right", va="center", fontsize=6.4, color="#8A6D00")
    arrow(ax, (101, 26.3), (111, 30.7), dashed=True, color="#8A6D00", connection="arc3,rad=-0.25")
    ax.text(104.3, 28.0, "$\\tau_{\\pi}$", ha="left", va="center", fontsize=6.4, color="#8A6D00")
    ax.plot([127.6, 129.6, 129.6], [16.5, 16.5, 62], color="#8A6D00", linewidth=0.7, linestyle=(0, (3, 2)))
    arrow(ax, (129.6, 62), (127.4, 62), dashed=True, color="#8A6D00")
    ax.text(130.3, 41.5, "$(\\alpha_{\\pi},\\beta_{\\pi})$", ha="left", va="center", fontsize=6.2, color="#8A6D00", rotation=90)
    arrow(ax, (15, 23.3), (15, 30.7), dashed=True)
    ax.text(16.2, 27.0, "缺失模态置零", ha="left", va="center", fontsize=6.0, color="#444444")
    ax.text(1, 1.5, "注：训练时按丢弃率随机屏蔽模态（至少保留一个）；锁定权重 (0.6, 0.4, 0) 仅在验证增益超过 δ=0.0005 时被替换。", fontsize=6.0, color="#333333")
    ax.text(0.5, 79.2, "(a)", fontsize=8.5, fontweight="bold", va="top")

    # (b) availability condition matrix
    axb = fig.add_axes([0.852, 0.24, 0.145, 0.66])
    states = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 2, 2], [3, 3, 3]]
    axb.set_xlim(-0.5, 2.5)
    axb.set_ylim(len(states) - 0.5, -0.5)
    for r, row in enumerate(states):
        for c, s in enumerate(row):
            if s == 1:
                patch = Rectangle((c - 0.42, r - 0.36), 0.84, 0.72, facecolor="#0072B2", edgecolor="none")
            elif s == 0:
                patch = Rectangle((c - 0.42, r - 0.36), 0.84, 0.72, facecolor="white", edgecolor="#888888", linewidth=0.6)
            else:
                patch = Rectangle((c - 0.42, r - 0.36), 0.84, 0.72, facecolor="#56B4E9" if s == 2 else "#DDDDDD", edgecolor="#777777", linewidth=0.5, hatch="////" if s == 2 else "xxxx")
            axb.add_patch(patch)
    axb.set_xticks(range(3), ["音频", "歌词", "视觉"])
    axb.set_yticks(range(len(states)), ONION_LABELS)
    axb.tick_params(length=0)
    axb.grid(False)
    for spine in axb.spines.values():
        spine.set_visible(False)
    axb.xaxis.tick_top()
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#0072B2", edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#888888"),
        Rectangle((0, 0), 1, 1, facecolor="#56B4E9", edgecolor="#777777", hatch="////"),
        Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="#777777", hatch="xxxx"),
    ]
    axb.legend(handles, ["保留", "移除", "随机移除一个", "随机保留一个"], frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.3, -0.02), fontsize=6.0, handlelength=1.1, columnspacing=0.6, handletextpad=0.4)
    axb.text(-2.35, -1.25, "(b)", fontsize=8.5, fontweight="bold", va="top")
    save(fig, out, "Fig1_framework_protocol")


# ---------------------------------------------------------------------------
# Figure 2: main results and robustness (Onion, three views)
# ---------------------------------------------------------------------------
def figure_2(data: Path, out: Path) -> None:
    onion = pd.read_csv(data / "onion_summary.csv")
    ablation = pd.read_csv(data / "ablation_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 120 * MM), gridspec_kw={"hspace": 0.62, "wspace": 0.24})
    order = ["AudioOnly", "LyricsOnly", "VisualOnly", "UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "H192", "H384", "AttnFusion", "EvidFusion", "MCE", "AA-MCE"]
    ax = axes[0, 0]
    bar_panel(ax, onion, order, "observed", "mAP", "完整观测 mAP")
    ax.axvspan(11.5, 13.5, color="#F3F3F3", zorder=0)
    panel(ax, "(a)", -0.12)
    ax = axes[0, 1]
    line_models = ["EarlyConcat", "ConcatModDrop", "RAMT", "AttnFusion", "EvidFusion", "H384", "MCE", "AA-MCE"]
    line_panel(ax, onion, line_models, ONION_CONDITIONS, ONION_LABELS, "mAP", "mAP", loc="lower left")
    ax.set_ylim(0.24, 0.46)
    panel(ax, "(b)", -0.12)
    ax = axes[1, 0]
    f1_models = ["EarlyConcat", "RAMT+AAT", "AttnFusion+AAT", "EvidFusion+AAT", "MCE", "AA-MCE"]
    frame = pd.concat([ablation, onion], ignore_index=True).drop_duplicates(subset=["model", "condition", "metric"])
    line_panel(ax, frame, f1_models, ONION_CONDITIONS, ONION_LABELS, "MacroF1", "Macro-F1", loc="upper right")
    ax.set_ylim(0.355, 0.50)
    ax.set_yticks([0.36, 0.38, 0.40, 0.42, 0.44, 0.46])
    panel(ax, "(c)", -0.12, 1.1)
    ax = axes[1, 1]
    bars = ["UniformMean", "EarlyConcat", "ConcatModDrop", "ReliabilityGate", "RAMT", "H192", "H384", "AttnFusion", "EvidFusion", "MCE", "AA-MCE"]
    mean_missing = [np.mean([value(onion, m, c, "mAP") for c in ONION_CONDITIONS[1:]]) for m in bars]
    worst = [np.min([value(onion, m, c, "mAP") for c in ONION_CONDITIONS[1:]]) for m in bars]
    x = np.arange(len(bars))
    ax.bar(x - 0.19, mean_missing, width=0.36, color=[COLORS[m] for m in bars], edgecolor="white", linewidth=0.3, label="五种缺失条件均值")
    ax.bar(x + 0.19, worst, width=0.36, color=[COLORS[m] for m in bars], alpha=0.45, hatch="////", edgecolor="white", linewidth=0.3, label="最坏缺失条件（仅留一）")
    ax.set_xticks(x, [NAMES[m] for m in bars], rotation=45, ha="right")
    ax.set_ylim(0.28, 0.415)
    ax.set_ylabel("缺失条件下的 mAP")
    ax.text(x[-1] - 0.19, mean_missing[-1] + 0.002, f"{mean_missing[-1]:.3f}", ha="right", va="bottom", fontsize=6.3, fontweight="bold")
    ax.text(x[-1] + 0.19, worst[-1] + 0.002, f"{worst[-1]:.3f}", ha="left", va="bottom", fontsize=6.3, fontweight="bold")
    handles = [Rectangle((0, 0), 1, 1, facecolor="#777777"), Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.45, hatch="////", edgecolor="white")]
    ax.legend(handles, ["五种缺失条件均值", "最坏缺失条件（仅留一）"], frameon=False, loc="upper left", ncol=1)
    panel(ax, "(d)", -0.12, 1.1)
    save(fig, out, "Fig2_main_results_robustness")


# ---------------------------------------------------------------------------
# Figure 3: mechanism analysis
# ---------------------------------------------------------------------------
def figure_3(data: Path, out: Path) -> None:
    gates = pd.read_csv(EXP / "full_feature_ablation_v1" / "gate_weights.csv")
    gates = gates.loc[gates["model"] == "RAMT"]
    sweep = pd.read_csv(EXP / "optimized_tagging_validation_v2" / "validation_sweep.csv").set_index("model")["mAP"]
    search = pd.read_csv(EXP / "optimized_tagging_validation_v2" / "validation_ensemble_search.csv")
    pair = search.loc[(search["kind"] == "probability") & (search["left"] == "FullConcatMD10-H384-BCE") & (search["right"] == "FullConcatMD-BCE")].sort_values("left_weight")
    weights = pd.read_csv(EXP / "availability_aware_ensemble_v1" / "pattern_weights.csv")
    weights = weights.loc[weights["variant"] == "AA-MCE"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 108 * MM), gridspec_kw={"hspace": 0.55, "wspace": 0.42})
    # (a) gate weights
    ax = axes[0, 0]
    x = np.arange(len(ONION_CONDITIONS))
    for i, (modality, label, color) in enumerate((("audio", "音频", "#0072B2"), ("lyrics", "歌词", "#E69F00"), ("visual", "视觉", "#009E73"))):
        values = [gates.loc[(gates["condition"] == c) & (gates["modality"] == modality), "mean_gate"].mean() for c in ONION_CONDITIONS]
        ax.bar(x + (i - 1) * 0.26, np.nan_to_num(values), width=0.26, color=color, label=label, edgecolor="none")
    ax.set_xticks(x, ONION_LABELS, rotation=30, ha="right")
    ax.set_ylabel("RAMT 平均门控权重")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    panel(ax, "(a)", -0.12)
    # (b) dropout and capacity (validation)
    ax = axes[0, 1]
    rates = [0.0, 0.1, 0.2, 0.3]
    names = ["FullConcat-BCE", "FullConcatMD10-BCE", "FullConcatMD20-BCE", "FullConcatMD-BCE"]
    ax.plot(rates, [sweep[n] for n in names], marker="o", color="#0072B2", label="隐层 192")
    for name, hidden, marker in (("FullConcatMD10-H256-BCE", 256, "s"), ("FullConcatMD10-H384-BCE", 384, "D")):
        ax.scatter([0.1], [sweep[name]], marker=marker, color="#B2182B", s=16, zorder=5)
        ax.annotate(f"隐层 {hidden}", (0.1, sweep[name]), textcoords="offset points", xytext=(5, -2), fontsize=6.3)
    ax.set_xlabel("训练时模态丢弃率 ρ")
    ax.set_ylabel("验证集 mAP")
    ax.set_xticks(rates)
    ax.set_ylim(0.4100, 0.4190)
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "(b)", -0.16, 1.1)
    # (c) ensemble weight search
    ax = axes[1, 0]
    ax.plot(pair["left_weight"], pair["validation_mAP"], marker="o", color=COLORS["MCE"], linewidth=1.2, label="两组件加权")
    ax.axhline(sweep["FullConcatMD10-H384-BCE"], color=COLORS["H384"], linestyle="--", linewidth=0.8, label="H384/MD10 单模型")
    ax.axhline(sweep["FullConcatMD-BCE"], color="#B59F2E", linestyle=":", linewidth=0.9, label="H192/MD30 单模型")
    locked = pair.loc[np.isclose(pair["left_weight"], 0.6), "validation_mAP"].iloc[0]
    ax.scatter([0.6], [locked], s=40, facecolors="none", edgecolors="#222222", linewidths=0.9, zorder=6, label="锁定 0.6/0.4")
    ax.set_xlabel("Concat-H384/MD10 的权重")
    ax.set_ylabel("验证集 mAP")
    ax.set_ylim(0.4125, 0.4292)
    ax.legend(frameon=False, loc="upper center", fontsize=6.0, ncol=2, columnspacing=0.8)
    panel(ax, "(c)", -0.16, 1.1)
    # (d) AA-MCE pattern weights
    ax = axes[1, 1]
    pattern_names = {
        "audio+lyrics+visual": "音频+歌词+视觉", "audio+lyrics": "音频+歌词", "audio+visual": "音频+视觉",
        "lyrics+visual": "歌词+视觉", "audio": "仅音频", "lyrics": "仅歌词", "visual": "仅视觉",
    }
    order = list(pattern_names)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for column, label, color in (("w_FullConcatMD10-H384-BCE", "$C_1$ (H384/MD10)", COLORS["H384"]), ("w_FullConcatMD-BCE", "$C_2$ (H192/MD30)", "#DDCC77"), ("w_RAMT", "$C_3$ (RAMT)", COLORS["RAMT"])):
        values = np.asarray([float(weights.loc[weights["pattern"] == p, column].iloc[0]) for p in order])
        ax.barh(y, values, left=left, color=color, height=0.68, edgecolor="white", linewidth=0.4, label=label)
        for yy, v, l in zip(y, values, left, strict=True):
            if v >= 0.15:
                ax.text(l + v / 2, yy, f"{v:.1f}", ha="center", va="center", fontsize=6.0, color="white" if column != "w_FullConcatMD-BCE" else "#333333")
        left += values
    ax.set_yticks(y, [pattern_names[p] for p in order], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("验证集选出的组件权重")
    ax.grid(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.16), fontsize=6.0, handlelength=1.0, columnspacing=0.7)
    panel(ax, "(d)", -0.22, 1.16)
    save(fig, out, "Fig3_mechanism")


# ---------------------------------------------------------------------------
# Figure 4: FMA and six views
# ---------------------------------------------------------------------------
def figure_4(data: Path, out: Path) -> None:
    fma = pd.read_csv(data / "fma_summary.csv")
    six = pd.read_csv(data / "sixview_summary.csv")
    fma_f1 = pd.concat([fma, pd.read_csv(data / "fma_aa_variants_summary.csv").rename(columns={"ci95_across_seeds": "ci95"})], ignore_index=True).drop_duplicates(subset=["model", "condition", "metric"], keep="first")
    six_f1 = pd.concat([six, pd.read_csv(data / "sixview_aa_variants_summary.csv").rename(columns={"ci95_across_seeds": "ci95"})], ignore_index=True).drop_duplicates(subset=["model", "condition", "metric"], keep="first")
    map_models = ["EarlyConcat", "ConcatModDrop", "RAMT", "AttnFusion", "EvidFusion", "H384", "MCE", "AA-MCE"]
    f1_models = ["EarlyConcat", "RAMT+AAT", "AttnFusion+AAT", "EvidFusion+AAT", "MCE", "AA-MCE"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 112 * MM), gridspec_kw={"hspace": 0.62, "wspace": 0.24})
    ax = axes[0, 0]
    line_panel(ax, fma, map_models, FMA_CONDITIONS, FMA_LABELS, "mAP", "FMA mAP", loc="lower left")
    ax.set_ylim(0.16, 0.52)
    panel(ax, "(a)", -0.12, 1.12)
    ax = axes[0, 1]
    line_panel(ax, fma_f1, f1_models, FMA_CONDITIONS, FMA_LABELS, "MacroF1", "FMA Macro-F1", loc="lower left")
    ax.set_ylim(0.16, 0.50)
    panel(ax, "(b)", -0.12, 1.12)
    ax = axes[1, 0]
    line_panel(ax, six, map_models, SIX_CONDITIONS, SIX_LABELS, "mAP", "六视图 mAP", loc="lower left")
    ax.set_ylim(0.21, 0.47)
    panel(ax, "(c)", -0.12, 1.12)
    ax = axes[1, 1]
    line_panel(ax, six_f1, f1_models, SIX_CONDITIONS, SIX_LABELS, "MacroF1", "六视图 Macro-F1", loc="lower left")
    ax.set_ylim(0.24, 0.49)
    panel(ax, "(d)", -0.12, 1.12)
    save(fig, out, "Fig4_transfer_fma_sixview")


# ---------------------------------------------------------------------------
# Figure 5: calibration and cost
# ---------------------------------------------------------------------------
def figure_5(data: Path, out: Path) -> None:
    calibration = pd.read_csv(EXP / "posthoc_calibration_v1" / "calibration_metrics.csv")
    efficiency = pd.read_csv(EXP / "paper_efficiency_v1" / "efficiency.csv")
    onion = pd.read_csv(data / "onion_summary.csv")
    five = pd.read_csv(data / "ablation_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 62 * MM), gridspec_kw={"wspace": 0.4})
    models = ["MCE", "AA-MCE", "RAMT", "AttnFusion", "EvidFusion"]
    x = np.arange(len(ONION_CONDITIONS))
    for axis_index, (metric, label) in enumerate((("MacroECE15", "宏平均 ECE（15 分箱）"), ("Brier", "Brier 分数"))):
        ax = axes[axis_index]
        for i, model in enumerate(models):
            for variant, style, alpha in (("raw", ":", 0.8), ("cal-AA", "-", 1.0)):
                block = calibration.loc[(calibration["model"] == model) & (calibration["variant"] == variant)].set_index("condition")
                ax.plot(x, [block.loc[c, metric] for c in ONION_CONDITIONS], linestyle=style, marker=MARKERS[i], color=COLORS[model], alpha=alpha, linewidth=1.4 if model == "AA-MCE" else 0.9, markersize=2.8, label=NAMES[model] if variant == "cal-AA" else None)
        ax.set_xticks(x, ONION_LABELS, rotation=30, ha="right")
        ax.set_ylabel(label)
        panel(ax, f"({'ab'[axis_index]})", -0.2)
    axes[0].legend(frameon=False, loc="center", fontsize=6.0, ncol=2, bbox_to_anchor=(0.47, 0.40), columnspacing=0.7, handlelength=1.6)
    ax = axes[2]
    gpu = efficiency.loc[efficiency["device"] == "cuda"].set_index("model")
    cpu = efficiency.loc[efficiency["device"] == "cpu"].set_index("model")
    points = [
        ("Concat H192/MD30", "H192", value(onion, "H192", "observed", "mAP")),
        ("Concat H384/MD10", "H384", value(onion, "H384", "observed", "mAP")),
        ("RAMT", "RAMT", value(onion, "RAMT", "observed", "mAP")),
        ("AttnFusion", "AttnFusion", value(onion, "AttnFusion", "observed", "mAP")),
        ("EvidFusion", "EvidFusion", value(onion, "EvidFusion", "observed", "mAP")),
        ("MCE", "MCE", value(onion, "MCE", "observed", "mAP")),
        ("AA-MCE", "AA-MCE", value(onion, "AA-MCE", "observed", "mAP")),
        ("AA-MCE (5 candidates)", "AA-MCE-5", value(five, "AA-MCE-5", "observed", "mAP")),
    ]
    for key, model, mean in points:
        latency = float(cpu.loc[key, "median_latency_ms"])
        params = float(gpu.loc[key, "parameters"]) / 1e6
        ax.scatter(latency, mean, s=12 + 9 * params, color=COLORS[model], edgecolor="white", linewidth=0.5, zorder=5)
        offsets = {"H192": (3, -9), "H384": (4, 1), "RAMT": (4, -4), "AttnFusion": (4, 2), "EvidFusion": (4, -2), "MCE": (-4, -10), "AA-MCE": (4, -9), "AA-MCE-5": (-52, 6)}
        ax.annotate(NAMES[model].replace("（本文）", ""), (latency, mean), textcoords="offset points", xytext=offsets.get(model, (3, 3)), fontsize=6.0, ha="left")
    ax.set_xlabel("CPU 单线程耗时 / ms")
    ax.set_ylabel("完整观测 mAP")
    ax.set_xlim(0, 240)
    ax.set_ylim(0.418, 0.4535)
    ax.text(0.97, 0.05, "点面积∝参数量", transform=ax.transAxes, ha="right", fontsize=6.0, color="#555555")
    panel(ax, "(c)", -0.2)
    save(fig, out, "Fig5_calibration_cost")


# ---------------------------------------------------------------------------
# Figure 6: tag semantics
# ---------------------------------------------------------------------------
def figure_6(data: Path, out: Path) -> None:
    groups = pd.read_csv(data / "tag_groups.csv")
    points = pd.read_csv(data / "tag_points.csv")
    order = ["genre_content", "era_origin", "mood_affect", "preference_context"]
    labels = {"genre_content": "流派/内容", "era_origin": "年代/地域", "mood_affect": "情绪/氛围", "preference_context": "偏好/情境"}
    counts = {"genre_content": 26, "era_origin": 6, "mood_affect": 9, "preference_context": 9}
    group_colors = {"genre_content": "#0072B2", "era_origin": "#009E73", "mood_affect": "#E69F00", "preference_context": "#B2182B"}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 62 * MM), gridspec_kw={"wspace": 0.3, "width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    models = ["AudioOnly", "EarlyConcat", "RAMT", "AttnFusion", "EvidFusion", "MCE", "AA-MCE"]
    width = 0.8 / len(models)
    obs = groups.loc[groups["condition"] == "observed"]
    for i, model in enumerate(models):
        values = [float(obs.loc[(obs["model"] == model) & (obs["group"] == g), "AP"].iloc[0]) for g in order]
        ax.bar(np.arange(4) + (i - (len(models) - 1) / 2) * width, values, width=width, color=COLORS[model], edgecolor="white", linewidth=0.3, label=NAMES[model])
    ax.set_xticks(np.arange(4), [f"{labels[g].replace('/', '/' + chr(10))}\n({counts[g]})" for g in order])
    ax.set_ylabel("完整观测平均 AP")
    ax.set_ylim(0, 0.78)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=5.9, handlelength=1.0, columnspacing=0.6)
    panel(ax, "(a)", -0.16)
    ax = axes[1]
    for g in order:
        block = points.loc[points["group"] == g]
        ax.scatter(block["train_frequency"], block["AA-MCE"], s=12, color=group_colors[g], label=labels[g], edgecolor="white", linewidth=0.3, zorder=4)
    for tag, dx, dy, ha in (("pop", 4, -1, "left"), ("rock", 4, -2, "left"), ("female vocalists", 4, -8, "left"), ("loved", 4, -2, "left"), ("6 of 10 stars", 4, -4, "left"), ("epic", 4, -1, "left"), ("metal", 4, 1, "left"), ("favorites", 4, -2, "left")):
        row = points.loc[points["label"] == tag]
        if not row.empty:
            ax.annotate(tag, (float(row["train_frequency"].iloc[0]), float(row["AA-MCE"].iloc[0])), textcoords="offset points", xytext=(dx, dy), fontsize=5.8, color="#333333", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(4000, 60000)
    ax.set_xticks([5000, 10000, 20000, 40000], ["5 000", "10 000", "20 000", "40 000"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("训练集标签频次（对数坐标）")
    ax.set_ylabel("AA-MCE 逐标签 AP")
    ax.legend(frameon=False, loc="lower right", fontsize=6.0, handletextpad=0.2)
    panel(ax, "(b)", -0.16)
    ax = axes[2]
    aa = groups.loc[groups["model"] == "AA-MCE"]
    x = np.arange(len(ONION_CONDITIONS))
    for i, g in enumerate(order):
        base = float(aa.loc[(aa["condition"] == "observed") & (aa["group"] == g), "AP"].iloc[0])
        values = [100 * float(aa.loc[(aa["condition"] == c) & (aa["group"] == g), "AP"].iloc[0]) / base for c in ONION_CONDITIONS]
        ax.plot(x, values, marker=MARKERS[i], linestyle=LINESTYLES[i], color=group_colors[g], label=labels[g])
    ax.axhline(100, color="#AAAAAA", linewidth=0.6, linestyle=":")
    ax.set_xticks(x, ONION_LABELS, rotation=30, ha="right")
    ax.set_ylabel("AP 保持率 / %")
    ax.legend(frameon=False, loc="lower left", fontsize=6.0)
    panel(ax, "(c)", -0.2)
    save(fig, out, "Fig6_tag_semantics")


# ---------------------------------------------------------------------------
# Figure 5 (manuscript): calibration and tag semantics
# ---------------------------------------------------------------------------
def figure_5_combined(data: Path, out: Path) -> None:
    calibration = pd.read_csv(EXP / "posthoc_calibration_v1" / "calibration_metrics.csv")
    groups = pd.read_csv(data / "tag_groups.csv")
    points = pd.read_csv(data / "tag_points.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 116 * MM), gridspec_kw={"hspace": 0.55, "wspace": 0.26})
    models = ["MCE", "AA-MCE", "RAMT", "AttnFusion", "EvidFusion"]
    x = np.arange(len(ONION_CONDITIONS))
    for index, (metric, label) in enumerate((("MacroECE15", "宏平均 ECE（15 分箱）"), ("Brier", "Brier 分数"))):
        ax = axes[0, index]
        for i, model in enumerate(models):
            for variant, style, alpha in (("raw", ":", 0.8), ("cal-AA", "-", 1.0)):
                block = calibration.loc[(calibration["model"] == model) & (calibration["variant"] == variant)].set_index("condition")
                ax.plot(x, [block.loc[c, metric] for c in ONION_CONDITIONS], linestyle=style, marker=MARKERS[i], color=COLORS[model], alpha=alpha, linewidth=1.4 if model == "AA-MCE" else 0.9, markersize=2.8, label=f"{NAMES[model]}" if variant == "cal-AA" else None)
        ax.set_xticks(x, ONION_LABELS, rotation=30, ha="right")
        ax.set_ylabel(label)
        panel(ax, f"({'ab'[index]})", -0.14)
    axes[0, 0].legend(frameon=False, loc="center", fontsize=6.0, ncol=2, bbox_to_anchor=(0.5, 0.45), columnspacing=0.7, handlelength=1.6)
    axes[0, 0].text(0.5, 0.25, "点线：校准前；实线：按可用模式 Platt 校准后", transform=axes[0, 0].transAxes, ha="center", fontsize=6.0, color="#333333")
    order = ["genre_content", "era_origin", "mood_affect", "preference_context"]
    labels = {"genre_content": "流派/内容", "era_origin": "年代/地域", "mood_affect": "情绪/氛围", "preference_context": "偏好/情境"}
    counts = {"genre_content": 26, "era_origin": 6, "mood_affect": 9, "preference_context": 9}
    group_colors = {"genre_content": "#0072B2", "era_origin": "#009E73", "mood_affect": "#E69F00", "preference_context": "#B2182B"}
    ax = axes[1, 0]
    bar_models = ["AudioOnly", "EarlyConcat", "RAMT", "AttnFusion", "EvidFusion", "MCE", "AA-MCE"]
    width = 0.8 / len(bar_models)
    observed = groups.loc[groups["condition"] == "observed"]
    for i, model in enumerate(bar_models):
        values = [float(observed.loc[(observed["model"] == model) & (observed["group"] == g), "AP"].iloc[0]) for g in order]
        ax.bar(np.arange(4) + (i - (len(bar_models) - 1) / 2) * width, values, width=width, color=COLORS[model], edgecolor="white", linewidth=0.3, label=NAMES[model])
    ax.set_xticks(np.arange(4), [f"{labels[g]}\n（{counts[g]} 个）" for g in order])
    ax.set_ylabel("完整观测平均 AP")
    ax.set_ylim(0, 0.74)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=5.9, handlelength=1.0, columnspacing=0.6)
    panel(ax, "(c)", -0.14)
    ax = axes[1, 1]
    for g in order:
        block = points.loc[points["group"] == g]
        ax.scatter(block["train_frequency"], block["AA-MCE"], s=12, color=group_colors[g], label=labels[g], edgecolor="white", linewidth=0.3, zorder=4)
    for tag, dx, dy in (("pop", 4, -1), ("rock", 4, -2), ("female vocalists", 4, -8), ("loved", 4, 0), ("6 of 10 stars", 4, -2), ("epic", 4, -1), ("metal", 4, 1), ("favorites", 4, -2)):
        row = points.loc[points["label"] == tag]
        if not row.empty:
            ax.annotate(tag, (float(row["train_frequency"].iloc[0]), float(row["AA-MCE"].iloc[0])), textcoords="offset points", xytext=(dx, dy), fontsize=5.8, color="#333333")
    ax.set_xscale("log")
    ax.set_xlim(4000, 60000)
    ax.set_ylim(0.08, 0.92)
    ax.set_xticks([5000, 10000, 20000, 40000], ["5 000", "10 000", "20 000", "40 000"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("训练集标签频次（对数坐标）")
    ax.set_ylabel("AA-MCE 逐标签 AP")
    ax.legend(frameon=False, loc="lower right", fontsize=6.0, handletextpad=0.2)
    panel(ax, "(d)", -0.14)
    save(fig, out, "Fig5_calibration_tag_semantics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/paper/data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/paper/figures"))
    parser.add_argument("--only", nargs="*", type=int, default=None)
    args = parser.parse_args()
    mpl.rcParams.update(STYLE)
    args.out.mkdir(parents=True, exist_ok=True)
    builders = {1: lambda: figure_1(args.out), 2: lambda: figure_2(args.data, args.out), 3: lambda: figure_3(args.data, args.out), 4: lambda: figure_4(args.data, args.out), 5: lambda: figure_5_combined(args.data, args.out)}
    for number in args.only or sorted(builders):
        builders[number]()


if __name__ == "__main__":
    main()


