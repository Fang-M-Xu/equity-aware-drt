from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "1.1-publication"

from config import load_config, path_from_config


# ============================================================
# Figure configuration
# ============================================================

PROPOSED_METHOD = "proposed_full"

METRICS = [
    {
        "column": "service_rate",
        "label": "Overall service rate",
        "higher_is_better": True,
    },
    {
        "column": "gap_closure_rate",
        "label": "Gap-origin service rate",
        "higher_is_better": True,
    },
    {
        "column": "pickup_delay_mean_sec",
        "label": "Mean pickup delay",
        "higher_is_better": False,
    },
    {
        "column": "pickup_delay_p90_sec",
        "label": "Pickup-delay p90",
        "higher_is_better": False,
    },
    {
        "column": "door_to_door_mean_sec",
        "label": "Mean door-to-door",
        "higher_is_better": False,
    },
    {
        "column": "door_to_door_p90_sec",
        "label": "Door-to-door p90",
        "higher_is_better": False,
    },
    {
        "column": "ride_time_mean_sec",
        "label": "Mean in-vehicle time",
        "higher_is_better": False,
    },
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.2.2: operational effects of "
            "gap-and-vulnerability-weighted vehicle prepositioning."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Formal competitive-experiment configuration.",
    )

    parser.add_argument(
        "--scenario-metrics",
        default=None,
        help=(
            "Scenario-level metrics CSV. "
            "Default: <metrics_dir>/scenario_level_metrics.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_prepositioning",
        help="Output directory for Figure 5.2.2.",
    )

    parser.add_argument(
        "--proposed-method",
        default=PROPOSED_METHOD,
    )

    parser.add_argument(
        "--random-method",
        default="alns_random_preposition",
    )

    parser.add_argument(
        "--expected-method",
        default="alns_demand_preposition",
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    return parser.parse_args()


# ============================================================
# Statistical helpers
# ============================================================

def holm_adjust(
    p_values: list[float],
) -> np.ndarray:
    """
    Holm step-down adjusted p-values.
    NaN values remain NaN.
    """

    values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    valid_positions = np.flatnonzero(
        np.isfinite(values)
    )

    if len(valid_positions) == 0:
        return adjusted

    valid_values = values[
        valid_positions
    ]

    order = np.argsort(
        valid_values
    )

    ordered_values = valid_values[
        order
    ]

    m = len(
        ordered_values
    )

    ordered_adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, p_value in enumerate(
        ordered_values
    ):

        multiplier = (
            m - rank
        )

        corrected = min(
            1.0,
            multiplier * p_value,
        )

        running_max = max(
            running_max,
            corrected,
        )

        ordered_adjusted[
            rank
        ] = running_max

    reverse_order = np.empty_like(
        order
    )

    reverse_order[
        order
    ] = np.arange(
        m
    )

    valid_adjusted = (
        ordered_adjusted[
            reverse_order
        ]
    )

    adjusted[
        valid_positions
    ] = valid_adjusted

    return adjusted


def paired_wilcoxon_p(
    proposed: np.ndarray,
    baseline: np.ndarray,
) -> float:
    """
    Two-sided paired Wilcoxon signed-rank test.
    """

    proposed = np.asarray(
        proposed,
        dtype=float,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    valid = (
        np.isfinite(proposed)
        & np.isfinite(baseline)
    )

    proposed = proposed[
        valid
    ]

    baseline = baseline[
        valid
    ]

    if len(proposed) == 0:
        return np.nan

    difference = (
        proposed
        - baseline
    )

    if np.allclose(
        difference,
        0.0,
    ):
        return 1.0

    try:

        result = wilcoxon(
            proposed,
            baseline,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return float(
            result.pvalue
        )

    except ValueError:

        return np.nan


def favorable_relative_change(
    proposed: np.ndarray,
    baseline: np.ndarray,
    higher_is_better: bool,
) -> np.ndarray:
    """
    Scenario-level favorable percentage change.

    Benefit metric:
        100 * (Proposed - Baseline) / Baseline

    Cost/time metric:
        100 * (Baseline - Proposed) / Baseline

    Therefore:
        positive = Proposed is better
        negative = Proposed is worse
    """

    proposed = np.asarray(
        proposed,
        dtype=float,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    valid = (
        np.isfinite(proposed)
        & np.isfinite(baseline)
        & (
            np.abs(baseline)
            > 1e-12
        )
    )

    result = np.full(
        proposed.shape,
        np.nan,
        dtype=float,
    )

    if higher_is_better:

        result[
            valid
        ] = (
            100.0
            * (
                proposed[
                    valid
                ]
                -
                baseline[
                    valid
                ]
            )
            /
            baseline[
                valid
            ]
        )

    else:

        result[
            valid
        ] = (
            100.0
            * (
                baseline[
                    valid
                ]
                -
                proposed[
                    valid
                ]
            )
            /
            baseline[
                valid
            ]
        )

    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    alpha: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Bootstrap CI for the mean of paired scenario-level
    favorable percentage changes.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean_value = float(
        values.mean()
    )

    if len(values) == 1:

        return (
            mean_value,
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            repetitions,
            len(values),
        ),
    )

    bootstrap_means = (
        values[
            indices
        ]
        .mean(
            axis=1
        )
    )

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0
            - alpha / 2.0,
        )
    )

    return (
        mean_value,
        lower,
        upper,
    )


# ============================================================
# Data loading
# ============================================================

def load_scenario_metrics(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            "Cannot find scenario-level metrics:\n"
            f"{path}\n\n"
            "Run 16_evaluate_all.py first."
        )

    frame = pd.read_csv(
        path
    )

    required = [
        "method",
        "scenario_id",
    ]

    missing = [
        column
        for column
        in required
        if column not in frame.columns
    ]

    if missing:

        raise KeyError(
            "Missing required scenario-metric columns:\n"
            f"{missing}\n\n"
            "Available columns:\n"
            f"{frame.columns.tolist()}"
        )

    frame[
        "method"
    ] = (
        frame[
            "method"
        ]
        .astype(str)
    )

    frame[
        "scenario_id"
    ] = pd.to_numeric(
        frame[
            "scenario_id"
        ],
        errors="coerce",
    )

    frame = (
        frame
        .dropna(
            subset=[
                "scenario_id"
            ]
        )
        .copy()
    )

    frame[
        "scenario_id"
    ] = (
        frame[
            "scenario_id"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # Gap-origin service-rate compatibility
    #
    # Current evaluation output uses gap_closure_rate.
    # If a future version exports gap_service_rate,
    # use that instead.
    # --------------------------------------------------------

    if (
        "gap_service_rate"
        in frame.columns
    ):

        frame[
            "gap_origin_service_rate"
        ] = pd.to_numeric(
            frame[
                "gap_service_rate"
            ],
            errors="coerce",
        )

    elif (
        "gap_closure_rate"
        in frame.columns
    ):

        frame[
            "gap_origin_service_rate"
        ] = pd.to_numeric(
            frame[
                "gap_closure_rate"
            ],
            errors="coerce",
        )

    else:

        warnings.warn(
            "Neither gap_service_rate nor "
            "gap_closure_rate was found. "
            "Gap-origin service rate cannot be plotted."
        )

    return frame


def metric_specs_for_frame(
    frame: pd.DataFrame,
) -> list[dict]:

    specifications = []

    for item in METRICS:

        item = dict(
            item
        )

        if (
            item[
                "column"
            ]
            == "gap_closure_rate"
        ):

            item[
                "column"
            ] = (
                "gap_origin_service_rate"
            )

        if (
            item[
                "column"
            ]
            not in frame.columns
        ):

            warnings.warn(
                "Skipping unavailable metric: "
                f"{item['label']} "
                f"({item['column']})"
            )

            continue

        specifications.append(
            item
        )

    return specifications


# ============================================================
# Comparison calculation
# ============================================================

def calculate_comparison(
    frame: pd.DataFrame,
    proposed_method: str,
    baseline_method: str,
    baseline_label: str,
    metric_specs: list[dict],
    repetitions: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:

    rows = []

    for metric_index, spec in enumerate(
        metric_specs
    ):

        metric = (
            spec[
                "column"
            ]
        )

        pivot = (
            frame[
                frame[
                    "method"
                ].isin(
                    [
                        proposed_method,
                        baseline_method,
                    ]
                )
            ]
            .pivot_table(
                index=
                    "scenario_id",

                columns=
                    "method",

                values=
                    metric,

                aggfunc=
                    "mean",
            )
        )

        if (
            proposed_method
            not in pivot.columns
            or baseline_method
            not in pivot.columns
        ):

            continue

        pair = (
            pivot[
                [
                    proposed_method,
                    baseline_method,
                ]
            ]
            .dropna()
        )

        if pair.empty:

            continue

        proposed = (
            pair[
                proposed_method
            ]
            .to_numpy(
                dtype=float
            )
        )

        baseline = (
            pair[
                baseline_method
            ]
            .to_numpy(
                dtype=float
            )
        )

        favorable_change = (
            favorable_relative_change(
                proposed,
                baseline,
                higher_is_better=
                    bool(
                        spec[
                            "higher_is_better"
                        ]
                    ),
            )
        )

        valid_change = (
            favorable_change[
                np.isfinite(
                    favorable_change
                )
            ]
        )

        mean_change, lower, upper = (
            bootstrap_mean_ci(
                valid_change,
                repetitions=
                    repetitions,

                seed=
                    seed
                    + metric_index,

                alpha=
                    alpha,
            )
        )

        raw_p = paired_wilcoxon_p(
            proposed,
            baseline,
        )

        rows.append(
            {
                "comparison":
                    baseline_label,

                "baseline_method":
                    baseline_method,

                "proposed_method":
                    proposed_method,

                "metric":
                    metric,

                "metric_label":
                    spec[
                        "label"
                    ],

                "higher_is_better":
                    spec[
                        "higher_is_better"
                    ],

                "n_paired_scenarios":
                    int(
                        len(
                            pair
                        )
                    ),

                "proposed_mean":
                    float(
                        np.mean(
                            proposed
                        )
                    ),

                "baseline_mean":
                    float(
                        np.mean(
                            baseline
                        )
                    ),

                "favorable_change_pct":
                    mean_change,

                "ci95_lower_pct":
                    lower,

                "ci95_upper_pct":
                    upper,

                "p_value":
                    raw_p,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plotting
# ============================================================

def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    show_y_labels: bool,
    alpha: float,
    panel_color: str,
    x_limits: tuple[float, float],
) -> None:

    if data.empty:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    data = data.copy().reset_index(drop=True)
    y = np.arange(len(data))

    ax.axvline(
        0.0,
        color="0.45",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    for row_index, row in data.iterrows():
        mean = float(row["favorable_change_pct"])
        lower = float(row["ci95_lower_pct"])
        upper = float(row["ci95_upper_pct"])

        significant = bool(
            np.isfinite(row["p_value_holm"])
            and row["p_value_holm"] < alpha
        )

        if np.isfinite(lower) and np.isfinite(upper):
            ax.hlines(
                y=row_index,
                xmin=lower,
                xmax=upper,
                color=panel_color,
                linewidth=1.8,
                zorder=2,
            )
            ax.vlines(
                [lower, upper],
                ymin=row_index - 0.075,
                ymax=row_index + 0.075,
                color=panel_color,
                linewidth=1.2,
                zorder=2,
            )

        if significant:
            ax.scatter(
                mean,
                row_index,
                s=62,
                color=panel_color,
                edgecolor=panel_color,
                linewidth=1.1,
                zorder=4,
            )
        else:
            ax.scatter(
                mean,
                row_index,
                s=62,
                facecolor="white",
                edgecolor=panel_color,
                linewidth=1.6,
                zorder=4,
            )

        if np.isfinite(mean):
            # Place the numeric annotation just to the right of the
            # confidence-interval whisker. This keeps the label visually
            # associated with the horizontal CI while avoiding overlap
            # with the point estimate.
            label_x = (
                upper
                if np.isfinite(upper)
                else mean
            )

            ax.annotate(
                f"{mean:+.1f}%",
                xy=(label_x, row_index),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color="0.15",
                clip_on=False,
                zorder=5,
            )

    ax.set_yticks(y)

    if show_y_labels:
        ax.set_yticklabels(
            data["metric_label"],
            fontsize=10,
        )
    else:
        ax.set_yticklabels([])

    ax.set_ylim(len(data) - 0.55, -0.55)
    ax.set_xlim(x_limits)

    ax.set_xlabel(
        "Favorable change relative to baseline (%)",
        fontsize=10,
    )

    ax.grid(
        axis="x",
        color="0.90",
        linewidth=0.7,
        zorder=0,
    )

    ax.set_title(
        title,
        fontsize=11.5,
        fontweight="semibold",
        pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("0.35")

    ax.tick_params(
        axis="y",
        length=0,
        pad=8,
    )
    ax.tick_params(
        axis="x",
        labelsize=9,
    )


def plot_figure(
    comparison_data: pd.DataFrame,
    output_base: Path,
    dpi: int,
    alpha: float,
) -> None:

    random_data = (
        comparison_data[
            comparison_data["comparison"].eq("Proposed vs Random")
        ]
        .copy()
    )

    expected_data = (
        comparison_data[
            comparison_data["comparison"].eq(
                "Proposed vs Expected-demand"
            )
        ]
        .copy()
    )

    label_order = [item["label"] for item in METRICS]
    order_map = {
        label: index
        for index, label in enumerate(label_order)
    }

    for data in [random_data, expected_data]:
        data["_order"] = data["metric_label"].map(order_map)
        data.sort_values("_order", inplace=True)
        data.drop(columns=["_order"], inplace=True)
        data.reset_index(drop=True, inplace=True)

    all_values = pd.concat(
        [
            random_data["ci95_lower_pct"],
            random_data["ci95_upper_pct"],
            expected_data["ci95_lower_pct"],
            expected_data["ci95_upper_pct"],
        ],
        ignore_index=True,
    )
    all_values = pd.to_numeric(
        all_values,
        errors="coerce",
    ).dropna()

    if all_values.empty:
        x_min, x_max = -10.0, 10.0
    else:
        x_min = float(all_values.min())
        x_max = float(all_values.max())
        span = max(x_max - x_min, 10.0)
        padding = 0.10 * span
        x_min = min(x_min - padding, -5.0)
        # Slightly more right-side room for percentage labels placed
        # just beyond the right CI cap.
        x_max = max(x_max + 0.16 * span, 5.0)

    x_limits = (x_min, x_max)

    random_color = "#1f4e79"
    expected_color = "#d97706"

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.8),
        sharey=False,
    )

    draw_panel(
        ax=axes[0],
        data=random_data,
        title="(a) Proposed vs Random",
        show_y_labels=True,
        alpha=alpha,
        panel_color=random_color,
        x_limits=x_limits,
    )

    draw_panel(
        ax=axes[1],
        data=expected_data,
        title="(b) Proposed vs Expected-demand",
        show_y_labels=False,
        alpha=alpha,
        panel_color=expected_color,
        x_limits=x_limits,
    )

    figure.suptitle(
        "Operational effects of gap-and-vulnerability-weighted "
        "vehicle prepositioning",
        fontsize=14.5,
        fontweight="semibold",
        y=0.975,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.20",
            markeredgecolor="0.20",
            markersize=7,
            label="Holm-adjusted p < 0.05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.20",
            markeredgewidth=1.4,
            markersize=7,
            label="Not significant after Holm correction",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="0.35",
            linewidth=1.6,
            label="Paired bootstrap 95% CI",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.8,
    )

    figure.subplots_adjust(
        left=0.225,
        right=0.975,
        bottom=0.115,
        top=0.82,
        wspace=0.14,
    )

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    config = load_config(
        args.config
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    metrics_dir = path_from_config(
        config,
        "metrics_dir",
    )

    scenario_path = (
        Path(
            args.scenario_metrics
        )
        if args.scenario_metrics
        else (
            metrics_dir
            / "scenario_level_metrics.csv"
        )
    )

    if not scenario_path.is_absolute():

        scenario_path = (
            PROJECT_ROOT
            / scenario_path
        )

    # --------------------------------------------------------
    # Output
    #
    # Default:
    # outputs/paper_results/chapter5_prepositioning
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    if not output_dir.is_absolute():

        output_dir = (
            PROJECT_ROOT
            / output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load scenario-level results
    # --------------------------------------------------------

    frame = load_scenario_metrics(
        scenario_path
    )

    proposed_method = str(
        args.proposed_method
    )

    baseline_map = {
        str(
            args.random_method
        ):
            "Proposed vs Random",

        str(
            args.expected_method
        ):
            "Proposed vs Expected-demand",
    }

    available_methods = set(
        frame[
            "method"
        ]
        .dropna()
        .astype(str)
    )

    required_methods = {
        proposed_method,
        *baseline_map.keys(),
    }

    missing_methods = sorted(
        required_methods
        - available_methods
    )

    if missing_methods:

        raise KeyError(
            "Required prepositioning methods are absent from "
            "scenario_level_metrics.csv:\n"
            f"{missing_methods}\n\n"
            "Available methods:\n"
            f"{sorted(available_methods)}"
        )

    metric_specs = metric_specs_for_frame(
        frame
    )

    if not metric_specs:

        raise KeyError(
            "None of the Figure 5.2.2 metrics "
            "were found in scenario_level_metrics.csv."
        )

    # --------------------------------------------------------
    # Paired comparisons
    # --------------------------------------------------------

    comparison_frames = []

    for baseline_index, (
        baseline_method,
        comparison_label,
    ) in enumerate(
        baseline_map.items()
    ):

        comparison = calculate_comparison(
            frame=
                frame,

            proposed_method=
                proposed_method,

            baseline_method=
                baseline_method,

            baseline_label=
                comparison_label,

            metric_specs=
                metric_specs,

            repetitions=
                int(
                    args.bootstrap_repetitions
                ),

            seed=
                int(
                    args.seed
                )
                + baseline_index
                * 1000,

            alpha=
                float(
                    args.alpha
                ),
        )

        comparison_frames.append(
            comparison
        )

    results = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    if results.empty:

        raise RuntimeError(
            "No valid paired scenario comparisons "
            "were available."
        )

    # ========================================================
    # Holm correction
    #
    # Correction is applied within each baseline comparison
    # across all displayed metrics.
    # ========================================================

    results[
        "p_value_holm"
    ] = np.nan

    for comparison_name, indices in (
        results
        .groupby(
            "comparison"
        )
        .groups
        .items()
    ):

        adjusted = holm_adjust(
            results.loc[
                indices,
                "p_value",
            ].tolist()
        )

        results.loc[
            indices,
            "p_value_holm",
        ] = adjusted

    results[
        "significant_after_holm"
    ] = (
        results[
            "p_value_holm"
        ]
        < float(
            args.alpha
        )
    )

    # ========================================================
    # Output files
    # ========================================================

    data_path = (
        output_dir
        / "Figure_5_2_2_operational_effects_data.csv"
    )

    results.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    output_base = (
        output_dir
        / "Figure_5_2_2_operational_effects"
    )

    plot_figure(
        comparison_data=
            results,

        output_base=
            output_base,

        dpi=
            int(
                args.dpi
            ),

        alpha=
            float(
                args.alpha
            ),
    )

    # ========================================================
    # Console summary
    # ========================================================

    display_columns = [
        "comparison",
        "metric_label",
        "n_paired_scenarios",
        "proposed_mean",
        "baseline_mean",
        "favorable_change_pct",
        "ci95_lower_pct",
        "ci95_upper_pct",
        "p_value",
        "p_value_holm",
        "significant_after_holm",
    ]

    print()
    print("=" * 100)

    print(
        "Figure 5.2.2 — Operational effects of "
        "gap-and-vulnerability-weighted vehicle prepositioning"
    )

    print("=" * 100)

    print(
        f"Runner version : {RUNNER_VERSION}"
    )

    print(
        f"Input          : {scenario_path}"
    )

    print(
        f"Output dir     : {output_dir}"
    )

    print(
        f"Proposed       : {proposed_method}"
    )

    print(
        f"Random         : {args.random_method}"
    )

    print(
        f"Expected-demand: {args.expected_method}"
    )

    print()

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Figure PNG : "
        f"{output_base.with_suffix('.png')}"
    )

    print(
        f"Figure PDF : "
        f"{output_base.with_suffix('.pdf')}"
    )

    print(
        f"Plot data  : "
        f"{data_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()