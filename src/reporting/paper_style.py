"""Centralized publication style for Information Fusion-ready figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import yaml


MM_PER_INCH = 25.4


def load_config(project_root: Path, config_path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mm_to_inches(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


def apply_paper_style(config: dict[str, Any]) -> None:
    style = config["figures"]
    font = style["font_family"]
    # Times New Roman is required by the target-journal artwork standard.
    # Fail loudly if the host lacks it instead of silently substituting.
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(f"Required figure font is unavailable: {font}") from error
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font],
            "mathtext.fontset": "stix",
            "font.size": style["font_size_pt"],
            "axes.labelsize": style["font_size_pt"],
            "axes.titlesize": style["font_size_pt"],
            "xtick.labelsize": style["font_size_pt"] - 0.5,
            "ytick.labelsize": style["font_size_pt"] - 0.5,
            "legend.fontsize": style["font_size_pt"] - 0.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": style["line_width_pt"],
            "lines.markersize": style["marker_size_pt"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_size(config: dict[str, Any], width: str, *, height_ratio: float | None = None) -> tuple[float, float]:
    style = config["figures"]
    width_inches = mm_to_inches(style["widths_mm"][width])
    return width_inches, width_inches * (height_ratio or style["height_ratio"])


def panel_label(axis: plt.Axes, label: str, *, config: dict[str, Any]) -> None:
    axis.text(
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=config["figures"]["panel_label_size_pt"],
        fontweight="bold",
        va="top",
        ha="left",
    )


def polish_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(color="#555555", labelcolor="#333333", length=3, width=0.65)


def save_figure(
    figure: plt.Figure,
    *,
    config: dict[str, Any],
    output_stem: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Export one publication-quality vector PDF and a 600 dpi PNG."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(config["figures"]["dpi"])
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if metadata is not None:
        output_stem.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(figure)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def model_color(config: dict[str, Any], model: str) -> str:
    return config["figures"]["models"].get(model, config["figures"]["colors"]["charcoal"])


def ordered_colors(config: dict[str, Any], names: Iterable[str]) -> list[str]:
    return [model_color(config, name) for name in names]
