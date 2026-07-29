#!/usr/bin/env python3
"""Build deterministic README plots from the canonical benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "benchmark_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "assets" / "benchmark_mr.png"
DEFAULT_ARCHITECTURE_OUTPUT = REPO_ROOT / "assets" / "architecture.png"
SELECTED_IDS = (
    "KAIST_B0_RGB",
    "KAIST_B1_LWIR",
    "KAIST_B2_EARLY6",
    "KAIST_M1_DCAF",
    "KAIST_M2_DCAF_CDR",
    "KAIST_M5_FORMAL",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["record_id"]: row for row in rows}
    missing = [record_id for record_id in SELECTED_IDS if record_id not in by_id]
    if missing:
        raise ValueError(f"Missing benchmark rows: {', '.join(missing)}")
    return [by_id[record_id] for record_id in SELECTED_IDS]


def build_chart(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    rows = load_rows(input_path)
    labels = [
        "RGB only",
        "LWIR only",
        "Early Fusion (6ch)",
        "DCAF",
        "DCAF + CDR",
        "IA-DASR",
    ]
    values = [float(row["mr_all_pct"]) for row in rows]
    colors = ["#A7B5C8", "#7D9CB8", "#E9A05B", "#6C8CD5", "#4C72C2", "#13A68A"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#D7DEE8",
            "axes.labelcolor": "#26354A",
            "xtick.color": "#52657D",
            "ytick.color": "#26354A",
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.1), dpi=180)
    fig.patch.set_facecolor("#F6F8FB")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(labels, values, color=colors, height=0.60, edgecolor="none")
    ax.invert_yaxis()

    ax.set_xlim(0, 43)
    ax.set_xlabel("Log-average miss rate, MR-all (%)  ·  lower is better", labelpad=12)
    ax.set_title(
        "IA-DASR cuts KAIST miss rate by 47.3% vs. early fusion",
        loc="left",
        fontsize=17,
        pad=20,
        color="#17253A",
    )
    ax.text(
        0,
        1.015,
        "Same-protocol repository re-evaluation · KAIST Reasonable · input 640 · seed 0",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#64748B",
    )

    ax.xaxis.grid(True, color="#E8EDF4", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D7DEE8")
    ax.tick_params(axis="y", length=0, pad=10)

    for index, (bar, value) in enumerate(zip(bars, values)):
        is_formal = index == len(values) - 1
        ax.text(
            value + 0.55,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold" if is_formal else "normal",
            color="#087F6B" if is_formal else "#34465E",
        )

    early = values[2]
    formal = values[-1]
    relative = 100 * (early - formal) / early
    card = FancyBboxPatch(
        (0.615, 0.075),
        0.335,
        0.205,
        transform=ax.transAxes,
        boxstyle="round,pad=0.016,rounding_size=0.02",
        linewidth=1,
        edgecolor="#B9E5DA",
        facecolor="#EEFAF7",
    )
    ax.add_patch(card)
    ax.text(
        0.64,
        0.225,
        "FORMAL SINGLE-MODEL GAIN",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="#087F6B",
    )
    ax.text(
        0.64,
        0.155,
        f"−{relative:.1f}% relative MR",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#0A6F60",
    )
    ax.text(
        0.64,
        0.095,
        f"{early:.2f}%  →  {formal:.2f}%",
        transform=ax.transAxes,
        fontsize=10,
        color="#42625D",
    )

    fig.text(
        0.105,
        0.022,
        "Source: results/benchmark_summary.csv · protocol-aware/system rows excluded from this chart",
        fontsize=8.5,
        color="#718096",
    )
    fig.subplots_adjust(left=0.22, right=0.94, top=0.83, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_architecture(output_path: Path) -> None:
    """Draw a clean, English-first overview of the dual-stream model."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    colors = {
        "ink": "#17253A",
        "muted": "#64748B",
        "rgb": "#2F6690",
        "rgb_fill": "#EDF4FA",
        "ir": "#D17A22",
        "ir_fill": "#FFF4E8",
        "fusion": "#3F8A5B",
        "fusion_fill": "#EDF8F0",
        "head": "#BF4B2E",
        "head_fill": "#FFF1EC",
        "neutral": "#5F6B7A",
        "neutral_fill": "#F3F5F8",
        "protocol": "#7B5BB5",
        "protocol_fill": "#F4F0FC",
    }

    fig, ax = plt.subplots(figsize=(14.2, 7.2), dpi=170)
    fig.patch.set_facecolor("#F6F8FB")
    ax.set_facecolor("#F6F8FB")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        xy: tuple[float, float],
        width: float,
        height: float,
        title: str,
        subtitle: str,
        edge: str,
        face: str,
        *,
        title_size: float = 12.5,
    ) -> None:
        x, y = xy
        shadow = FancyBboxPatch(
            (x + 0.004, y - 0.006),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0,
            facecolor="#CBD5E1",
            alpha=0.35,
            zorder=1,
        )
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(shadow)
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height * 0.61,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=colors["ink"],
            zorder=3,
        )
        ax.text(
            x + width / 2,
            y + height * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
            color=colors["muted"],
            zorder=3,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        *,
        width: float = 1.8,
        style: str = "-",
        connection: str = "arc3,rad=0",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=width,
                linestyle=style,
                color=color,
                connectionstyle=connection,
                shrinkA=1,
                shrinkB=1,
                zorder=4,
            )
        )

    ax.text(
        0.04,
        0.94,
        "IA-DASR · reliability-aware RGB–thermal detection",
        fontsize=22,
        fontweight="bold",
        color=colors["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.04,
        0.895,
        "Bidirectional deformable alignment at P3/P4/P5, followed by bounded reliability modulation",
        fontsize=10.5,
        color=colors["muted"],
        ha="left",
        va="center",
    )

    # Inputs and backbones.
    box((0.035, 0.59), 0.105, 0.15, "RGB input", "visible image", colors["rgb"], colors["rgb_fill"])
    box((0.035, 0.25), 0.105, 0.15, "LWIR input", "thermal image", colors["ir"], colors["ir_fill"])
    box((0.19, 0.56), 0.16, 0.21, "RGB backbone", "YOLOv5s-style stream", colors["rgb"], colors["rgb_fill"])
    box((0.19, 0.22), 0.16, 0.21, "LWIR backbone", "YOLOv5s-style stream", colors["ir"], colors["ir_fill"])
    arrow((0.14, 0.665), (0.19, 0.665), colors["rgb"])
    arrow((0.14, 0.325), (0.19, 0.325), colors["ir"])

    fusion_y = (0.665, 0.455, 0.245)
    fusion_titles = ("P3 · fine", "P4 · medium", "P5 · coarse")
    for center_y, scale_title in zip(fusion_y, fusion_titles):
        box(
            (0.45, center_y - 0.075),
            0.215,
            0.15,
            scale_title,
            "DCAF  →  CDR  →  DSRE",
            colors["fusion"],
            colors["fusion_fill"],
            title_size=12,
        )

    # Multi-scale paths from both backbones.
    rgb_starts = (0.70, 0.665, 0.625)
    ir_starts = (0.36, 0.325, 0.285)
    for index, center_y in enumerate(fusion_y):
        arrow(
            (0.35, rgb_starts[index]),
            (0.45, center_y + 0.022),
            colors["rgb"],
            width=1.55,
        )
        arrow(
            (0.35, ir_starts[index]),
            (0.45, center_y - 0.022),
            colors["ir"],
            width=1.55,
        )

    box(
        (0.75, 0.365),
        0.105,
        0.17,
        "PAN/FPN",
        "multi-scale neck",
        colors["neutral"],
        colors["neutral_fill"],
    )
    for center_y in fusion_y:
        arrow(
            (0.665, center_y),
            (0.75, 0.45),
            colors["fusion"],
            width=1.7,
        )

    box(
        (0.895, 0.365),
        0.075,
        0.17,
        "Detect",
        "box · obj · cls",
        colors["head"],
        colors["head_fill"],
        title_size=12,
    )
    arrow((0.855, 0.45), (0.895, 0.45), colors["neutral"])

    # Training-only supervision boundary.
    supervision = FancyBboxPatch(
        (0.19, 0.065),
        0.475,
        0.075,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.4,
        edgecolor=colors["fusion"],
        facecolor="#FFFFFF",
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(supervision)
    ax.text(
        0.4275,
        0.102,
        "Training only · ignore-aware objectness + reliability auxiliary supervision",
        ha="center",
        va="center",
        fontsize=10,
        color=colors["fusion"],
        fontweight="bold",
    )
    arrow(
        (0.665, 0.102),
        (0.925, 0.365),
        colors["fusion"],
        width=1.3,
        style="--",
        connection="arc3,rad=-0.14",
    )

    # Optional protocol-aware branch with a clear non-mainline boundary.
    protocol = FancyBboxPatch(
        (0.75, 0.08),
        0.22,
        0.095,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.4,
        edgecolor=colors["protocol"],
        facecolor=colors["protocol_fill"],
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(protocol)
    ax.text(
        0.86,
        0.138,
        "Round 2I+ only",
        ha="center",
        va="center",
        fontsize=9.5,
        color=colors["protocol"],
        fontweight="bold",
    )
    ax.text(
        0.86,
        0.103,
        "bounded protocol-semantic score factor",
        ha="center",
        va="center",
        fontsize=8.4,
        color=colors["muted"],
    )
    arrow(
        (0.86, 0.175),
        (0.93, 0.365),
        colors["protocol"],
        width=1.3,
        style="--",
        connection="arc3,rad=0.14",
    )

    ax.text(
        0.04,
        0.015,
        "Formal mainline: DCAF + CDR + DSRE + ignore-aware objectness · protocol-aware calibration is reported separately",
        fontsize=8.8,
        color=colors["muted"],
        ha="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--architecture-output",
        type=Path,
        default=DEFAULT_ARCHITECTURE_OUTPUT,
    )
    args = parser.parse_args()
    build_chart(args.input, args.output)
    build_architecture(args.architecture_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.architecture_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
