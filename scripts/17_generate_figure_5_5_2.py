from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"
# ============================================================
# Core configurations used in Figure 5.5.2
#
# Keep a fixed paper-facing order.
# ============================================================

METHODS = [
    {
        "method": "alns_random_preposition",
        "label": "Random prepositioning",
    },
    {
        "method": "alns_demand_preposition",
        "label": "Expected-demand prepositioning",
    },
    {
        "method": "greedy_full",
        "label": "Greedy",
    },
    {
        "method": "proposed_full",
        "label": "Proposed-full",
    },
]


# ============================================================
# Bootstrap CI for descriptive audit only
#
# CI is saved in the summary CSV and printed to console,
# but is not drawn on top of the boxplot.
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(
        np.mean(values)
    )

    if len(values) == 1:
        return (
            mean_value,
            mean_value,
            mean_value,
        )

    bootstrap_means = np.empty(
        repetitions,
        dtype=float,
    )

    for i in range(repetitions):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    lower = float(
        np.quantile(
            bootstrap_means,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            0.975,
        )
    )

    return (
        mean_value,
        lower,
        upper,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.5.2 — "
            "Robustness across demand scenarios."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "outputs/metrics/"
            "scenario_level_metrics.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/paper_results/"
            "chapter5_sensitivity"
        ),
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--jitter-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find scenario-level metrics:\n"
            f"{input_path}"
        )

    # ========================================================
    # 1. Load formal scenario-level results
    # ========================================================

    data = pd.read_csv(
        input_path
    )

    required = [
        "method",
        "scenario_id",
        "service_rate",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "scenario_level_metrics.csv is missing columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )

    # ========================================================
    # 2. Keep only the four core configurations
    # ========================================================

    method_names = [
        item["method"]
        for item in METHODS
    ]

    selected = data[
        data[
            "method"
        ].isin(
            method_names
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "None of the requested methods were found."
        )

    selected[
        "service_rate"
    ] = pd.to_numeric(
        selected[
            "service_rate"
        ],
        errors="coerce",
    )

    selected = selected.dropna(
        subset=[
            "scenario_id",
            "service_rate",
        ]
    )

    # ========================================================
    # 3. Validate each method
    # ========================================================

    available_methods = set(
        selected[
            "method"
        ].unique()
    )

    for item in METHODS:

        method = item[
            "method"
        ]

        if method not in available_methods:
            raise ValueError(
                f"Required method '{method}' "
                "was not found in the source file."
            )

    # There should be one row per method × scenario.
    duplicated = selected.duplicated(
        subset=[
            "method",
            "scenario_id",
        ],
        keep=False,
    )

    if duplicated.any():

        problem = selected.loc[
            duplicated,
            [
                "method",
                "scenario_id",
                "service_rate",
            ],
        ]

        raise ValueError(
            "Duplicated method × scenario rows found:\n"
            f"{problem.head(20).to_string(index=False)}"
        )

    # ========================================================
    # 4. Keep common scenarios
    #
    # This ensures that all configurations are compared on
    # exactly the same realized-demand scenarios.
    # ========================================================

    scenario_sets = []

    for item in METHODS:

        method = item[
            "method"
        ]

        scenarios = set(
            selected.loc[
                selected[
                    "method"
                ].eq(method),
                "scenario_id",
            ].tolist()
        )

        scenario_sets.append(
            scenarios
        )

    common_scenarios = set.intersection(
        *scenario_sets
    )

    if not common_scenarios:
        raise ValueError(
            "No common scenarios exist across "
            "the selected configurations."
        )

    selected = selected[
        selected[
            "scenario_id"
        ].isin(
            common_scenarios
        )
    ].copy()

    # ========================================================
    # 5. Paper-facing method labels
    # ========================================================

    label_mapping = {
        item["method"]:
            item["label"]
        for item in METHODS
    }

    selected[
        "configuration"
    ] = (
        selected[
            "method"
        ]
        .map(
            label_mapping
        )
    )

    # ========================================================
    # 6. Descriptive summary
    # ========================================================

    bootstrap_rng = np.random.default_rng(
        args.bootstrap_seed
    )

    summary_rows = []

    for item in METHODS:

        method = item[
            "method"
        ]

        label = item[
            "label"
        ]

        values = (
            selected.loc[
                selected[
                    "method"
                ].eq(method),
                "service_rate",
            ]
            .to_numpy(
                dtype=float
            )
        )

        mean_value, ci_lower, ci_upper = (
            bootstrap_mean_ci(
                values,
                repetitions=(
                    args.bootstrap_repetitions
                ),
                rng=bootstrap_rng,
            )
        )

        summary_rows.append(
            {
                "method":
                    method,

                "configuration":
                    label,

                "n_scenarios":
                    len(values),

                "mean_service_rate":
                    float(
                        np.mean(values)
                    ),

                "median_service_rate":
                    float(
                        np.median(values)
                    ),

                "std_service_rate":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "minimum_service_rate":
                    float(
                        np.min(values)
                    ),

                "maximum_service_rate":
                    float(
                        np.max(values)
                    ),

                "q25_service_rate":
                    float(
                        np.quantile(
                            values,
                            0.25,
                        )
                    ),

                "q75_service_rate":
                    float(
                        np.quantile(
                            values,
                            0.75,
                        )
                    ),

                "ci95_lower":
                    ci_lower,

                "ci95_upper":
                    ci_upper,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # 7. Prepare boxplot data in fixed order
    # ========================================================

    plot_data = []

    labels = []

    for item in METHODS:

        method = item[
            "method"
        ]

        label = item[
            "label"
        ]

        values = (
            selected.loc[
                selected[
                    "method"
                ].eq(method),
                "service_rate",
            ]
            .sort_index()
            .to_numpy(
                dtype=float
            )
            * 100.0
        )

        plot_data.append(
            values
        )

        labels.append(
            label
        )

    # ========================================================
    # 8. Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8)
    )

    positions = np.arange(
        1,
        len(METHODS) + 1,
    )

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------

    ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.52,
        showfliers=False,
        showmeans=True,
        meanline=False,
    )

    # --------------------------------------------------------
    # Overlay all scenario observations
    #
    # Small horizontal jitter prevents points from completely
    # overlapping while retaining scenario distribution.
    # --------------------------------------------------------

    jitter_rng = np.random.default_rng(
        args.jitter_seed
    )

    for position, values in zip(
        positions,
        plot_data,
    ):

        jitter = jitter_rng.uniform(
            -0.10,
            0.10,
            size=len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                position,
                dtype=float,
            )
            + jitter,
            values,
            s=20,
            alpha=0.55,
            zorder=3,
        )

    # ========================================================
    # 9. Formatting
    # ========================================================

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=0,
        fontsize=9.3,
    )

    ax.set_ylabel(
        "Service rate (%)"
    )

    ax.set_xlabel(
        "Configuration"
    )

    # --------------------------------------------------------
    # Use a common informative y-axis range.
    # Do not automatically force 0–100 if all observations
    # lie within a narrower range.
    # --------------------------------------------------------

    all_values = np.concatenate(
        plot_data
    )

    minimum = float(
        np.nanmin(
            all_values
        )
    )

    maximum = float(
        np.nanmax(
            all_values
        )
    )

    ymin = max(
        0.0,
        np.floor(
            (minimum - 5.0)
            / 5.0
        )
        * 5.0,
    )

    ymax = min(
        100.0,
        np.ceil(
            (maximum + 5.0)
            / 5.0
        )
        * 5.0,
    )

    if ymax <= ymin:
        ymin = 0.0
        ymax = 100.0

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # ========================================================
    # 10. Add mean labels above boxes
    # ========================================================

    for position, item in zip(
        positions,
        METHODS,
    ):

        method = item[
            "method"
        ]

        mean_value = float(
            summary.loc[
                summary[
                    "method"
                ].eq(method),
                "mean_service_rate",
            ].iloc[0]
        ) * 100.0

        ax.text(
            position,
            ymax - 1.0,
            f"Mean {mean_value:.1f}%",
            ha="center",
            va="top",
            fontsize=8.5,
        )

    fig.tight_layout()

    # ========================================================
    # 11. Save
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "Figure_5_5_2_robustness_across_demand_scenarios.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_5_2_robustness_across_demand_scenarios.pdf"
    )

    plot_data_path = (
        output_dir
        / "Figure_5_5_2_scenario_distribution_data.csv"
    )

    summary_path = (
        output_dir
        / "Figure_5_5_2_robustness_summary.csv"
    )

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    selected[
        [
            "scenario_id",
            "method",
            "configuration",
            "service_rate",
        ]
    ].sort_values(
        [
            "configuration",
            "scenario_id",
        ]
    ).to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.close(fig)

    # ========================================================
    # 12. Console audit
    # ========================================================

    print()
    print("=" * 96)

    print(
        "Figure 5.5.2 — "
        "Robustness across demand scenarios"
    )

    print("=" * 96)

    print(
        f"Source file       : {input_path}"
    )

    print(
        f"Common scenarios  : {len(common_scenarios)}"
    )

    print(
        f"Scenario IDs      : "
        f"{sorted(common_scenarios)}"
    )

    print()

    display = summary.copy()

    for column in [
        "mean_service_rate",
        "median_service_rate",
        "std_service_rate",
        "minimum_service_rate",
        "maximum_service_rate",
        "q25_service_rate",
        "q75_service_rate",
        "ci95_lower",
        "ci95_upper",
    ]:
        display[
            column
        ] = (
            display[
                column
            ]
            * 100.0
        )

    print(
        display[
            [
                "configuration",
                "n_scenarios",
                "mean_service_rate",
                "median_service_rate",
                "std_service_rate",
                "minimum_service_rate",
                "maximum_service_rate",
                "ci95_lower",
                "ci95_upper",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"PNG saved         : {png_path}"
    )

    print(
        f"PDF saved         : {pdf_path}"
    )

    print(
        f"Scenario data     : {plot_data_path}"
    )

    print(
        f"Summary data      : {summary_path}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()