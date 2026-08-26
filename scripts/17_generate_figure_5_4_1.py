from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

WALKING_COLOR = "#7A8793"
TRANSIT_COLOR = "#F28E2B"
DRT_COLOR = "#7A5195"
LINE_COLOR = "#DCE3EA"
INK = "#263238"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.4.1: 15-minute accessibility of DRT, "
            "walking and GTFS fixed-route public transport."
        )
    )
    parser.add_argument(
        "--overall-file",
        default="outputs/metrics/multimodal_evaluation/paper_table_overall_modes.csv",
    )
    parser.add_argument(
        "--group-file",
        default="outputs/metrics/multimodal_evaluation/group_drt_vs_benchmarks.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_multimodal",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#AAB5BD",
            "axes.linewidth": 0.8,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "grid.color": "#D9E1E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def validate_inputs(overall: pd.DataFrame, group: pd.DataFrame) -> None:
    overall_required = {"mode", "within_threshold_rate"}
    group_required = {
        "group_variable",
        "group_value",
        "walking_access_rate",
        "transit_access_rate",
        "drt_expected_access_rate",
    }

    missing_overall = sorted(overall_required - set(overall.columns))
    missing_group = sorted(group_required - set(group.columns))

    if missing_overall:
        raise KeyError(
            "paper_table_overall_modes.csv is missing required columns: "
            f"{missing_overall}"
        )
    if missing_group:
        raise KeyError(
            "group_drt_vs_benchmarks.csv is missing required columns: "
            f"{missing_group}"
        )


def extract_values(
    overall: pd.DataFrame,
    group: pd.DataFrame,
) -> tuple[list[str], list[list[float]]]:
    mode = overall.copy()
    mode["mode"] = mode["mode"].astype(str).str.strip()
    mode = mode.set_index("mode")

    required_modes = ["Walking", "Public transit", "DRT proposed full"]
    missing_modes = [name for name in required_modes if name not in mode.index]
    if missing_modes:
        raise KeyError(
            "paper_table_overall_modes.csv does not contain required modes: "
            f"{missing_modes}. Available modes: {list(mode.index)}"
        )

    gap = group.loc[
        group["group_variable"].astype(str).eq("is_gap")
        & group["group_value"].astype(str).str.strip().isin({"1", "1.0", "True", "true"})
    ].copy()

    if gap.empty:
        raise ValueError(
            "Could not find the gap-origin row in group_drt_vs_benchmarks.csv "
            "(expected group_variable='is_gap' and group_value=1)."
        )

    gap_row = gap.iloc[0]

    populations = ["All requests", "Gap-origin requests"]
    values = [
        [
            float(mode.loc["Walking", "within_threshold_rate"]),
            float(mode.loc["Public transit", "within_threshold_rate"]),
            float(mode.loc["DRT proposed full", "within_threshold_rate"]),
        ],
        [
            float(gap_row["walking_access_rate"]),
            float(gap_row["transit_access_rate"]),
            float(gap_row["drt_expected_access_rate"]),
        ],
    ]
    return populations, values


def plot_figure(
    populations: list[str],
    values: list[list[float]],
    output_dir: Path,
    dpi: int,
) -> None:
    mode_names = ["Walking", "GTFS public transit", "Proposed DRT"]
    colors = [WALKING_COLOR, TRANSIT_COLOR, DRT_COLOR]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y_positions = [1.0, 0.0]

    flat = [v for row in values for v in row]
    minimum, maximum = min(flat), max(flat)
    span = max(maximum - minimum, 0.10)

    xmin = max(0.0, minimum - 0.12 * span)
    xmax = min(1.0, maximum + 0.12 * span)

    if xmax - xmin < 0.22:
        midpoint = (xmin + xmax) / 2
        xmin = max(0.0, midpoint - 0.11)
        xmax = min(1.0, midpoint + 0.11)

    for y, row in zip(y_positions, values):
        ax.plot(
            [min(row), max(row)],
            [y, y],
            color=LINE_COLOR,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=1,
        )

        for name, value, color in zip(mode_names, row, colors):
            ax.scatter(
                value,
                y,
                s=92,
                color=color,
                edgecolors=WHITE,
                linewidths=0.8,
                zorder=3,
            )
            if name == "GTFS public transit":
                dy, va = -0.21, "top"
            else:
                dy, va = 0.14, "bottom"

            ax.text(
                value,
                y + dy,
                f"{value:.1%}",
                ha="center",
                va=va,
                fontsize=9.0,
                color=INK,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(populations)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.48, 1.38)
    ax.set_xlabel("Within-15-minute accessibility rate")
    ax.set_ylabel("Request population")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=WHITE,
            markersize=8,
            label=name,
        )
        for name, color in zip(mode_names, colors)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "Figure_5_4_1_multimodal_accessibility.png"
    pdf_path = output_dir / "Figure_5_4_1_multimodal_accessibility.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    configure_style()
    args = parse_args()

    overall_path = resolve(args.overall_file)
    group_path = resolve(args.group_file)
    output_dir = resolve(args.output_dir)

    if not overall_path.exists():
        raise FileNotFoundError(f"Overall multimodal file not found: {overall_path}")
    if not group_path.exists():
        raise FileNotFoundError(f"Group multimodal file not found: {group_path}")

    overall = pd.read_csv(overall_path)
    group = pd.read_csv(group_path)

    validate_inputs(overall, group)
    populations, values = extract_values(overall, group)
    plot_figure(populations, values, output_dir, args.dpi)


if __name__ == "__main__":
    main()
