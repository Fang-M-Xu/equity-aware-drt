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
# Formal OFAT settings from sensitivity_ofat_all_scenarios.csv
# ============================================================

PARAMETERS = [
    {
        "name": "fleet_size",
        "title": "(a) Fleet size",
        "xlabel": "Fleet size",
        "baseline_column": "fleet_size",
    },
    {
        "name": "gap_service_benefit",
        "title": "(b) Service- gap weight",
        "xlabel": "Service- gap weight",
        "baseline_column": "gap_service_benefit",
    },
    {
        "name": "social_welfare_benefit_weight",
        "title": "(c) Social-equity weight",
        "xlabel": "Social-equity weight",
        "baseline_column": "social_welfare_benefit_weight",
    },
    {
        "name": "max_pickup_delay_sec",
        "title": "(d) Maximum pickup delay",
        "xlabel": "Maximum pickup delay (s)",
        "baseline_column": "max_pickup_delay_sec",
    },
    {
        "name": "max_door_to_door_sec",
        "title": "(e) Maximum door-to-door time",
        "xlabel": "Maximum door-to-door time (s)",
        "baseline_column": "max_door_to_door_sec",
    },
    {
        "name": "iterations",
        "title": "(f) ALNS iterations",
        "xlabel": "Maximum ALNS iterations",
        "baseline_column": "iterations",
    },
]


METRICS = [
    {
        "column": "service_rate_all",
        "label": "Overall service rate",
    },
    {
        "column": "gap_service_rate",
        "label": "Gap-origin service rate",
    },
]


# ============================================================
# Bootstrap across sensitivity scenarios
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
# Validate source
# ============================================================

def validate_input(
    data: pd.DataFrame,
) -> None:

    required = [
        "scenario_id",
        "run_id",
        "varied_parameter",
        "varied_value",
        "is_baseline",

        "fleet_size",
        "gap_service_benefit",
        "social_welfare_benefit_weight",
        "max_pickup_delay_sec",
        "max_door_to_door_sec",
        "iterations",

        "service_rate_all",
        "gap_service_rate",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Sensitivity file is missing columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )


# ============================================================
# Construct one OFAT parameter dataset
#
# Important:
# baseline is stored separately in the source CSV.
# It must be inserted into every parameter curve.
# ============================================================

def build_parameter_data(
    data: pd.DataFrame,
    parameter_definition: dict,
) -> pd.DataFrame:

    parameter_name = (
        parameter_definition["name"]
    )

    baseline_column = (
        parameter_definition[
            "baseline_column"
        ]
    )

    # --------------------------------------------------------
    # Perturbed OFAT settings
    # --------------------------------------------------------

    varied = data[
        data[
            "varied_parameter"
        ].eq(parameter_name)
    ].copy()

    varied[
        "plot_value"
    ] = pd.to_numeric(
        varied["varied_value"],
        errors="coerce",
    )

    varied[
        "setting_type"
    ] = "Sensitivity"

    # --------------------------------------------------------
    # Baseline rows
    # --------------------------------------------------------

    baseline = data[
        data["is_baseline"].astype(bool)
    ].copy()

    baseline[
        "plot_value"
    ] = pd.to_numeric(
        baseline[
            baseline_column
        ],
        errors="coerce",
    )

    baseline[
        "setting_type"
    ] = "Baseline"

    # --------------------------------------------------------
    # Combine baseline + varied settings
    # --------------------------------------------------------

    result = pd.concat(
        [
            varied,
            baseline,
        ],
        ignore_index=True,
        sort=False,
    )

    result = result.dropna(
        subset=[
            "scenario_id",
            "plot_value",
        ]
    )

    # Ensure each scenario has at most one result
    # for each tested parameter value.
    duplicate = result.duplicated(
        subset=[
            "scenario_id",
            "plot_value",
        ],
        keep=False,
    )

    if duplicate.any():

        problem = result.loc[
            duplicate,
            [
                "scenario_id",
                "run_id",
                "plot_value",
                "setting_type",
            ],
        ]

        raise ValueError(
            f"Duplicated scenario × parameter-value rows "
            f"for {parameter_name}:\n"
            f"{problem.to_string(index=False)}"
        )

    return result


# ============================================================
# Summarize one parameter
# ============================================================

def summarize_parameter(
    parameter_data: pd.DataFrame,
    parameter_name: str,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    rows = []

    for metric_definition in METRICS:

        metric_column = (
            metric_definition["column"]
        )

        for value, group in (
            parameter_data
            .groupby(
                "plot_value",
                sort=True,
            )
        ):

            values = pd.to_numeric(
                group[metric_column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            mean_value, lower, upper = (
                bootstrap_mean_ci(
                    values,
                    repetitions,
                    rng,
                )
            )

            baseline_rows = group[
                group[
                    "setting_type"
                ].eq("Baseline")
            ]

            is_baseline_value = (
                len(baseline_rows) > 0
            )

            rows.append(
                {
                    "parameter":
                        parameter_name,

                    "parameter_value":
                        float(value),

                    "metric":
                        metric_column,

                    "metric_label":
                        metric_definition[
                            "label"
                        ],

                    "n_scenarios":
                        int(
                            group[
                                "scenario_id"
                            ].nunique()
                        ),

                    "mean":
                        mean_value,

                    "ci95_lower":
                        lower,

                    "ci95_upper":
                        upper,

                    "is_baseline":
                        bool(
                            is_baseline_value
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.5.1 — "
            "OFAT sensitivity of key performance indicators."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "outputs/sensitivity/"
            "sensitivity_ofat_all_scenarios.csv"
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
            f"Cannot find:\n{input_path}"
        )

    # ========================================================
    # 1. Read actual sensitivity output
    # ========================================================

    data = pd.read_csv(
        input_path
    )

    validate_input(
        data
    )

    # Normalize bool field
    data[
        "is_baseline"
    ] = (
        data[
            "is_baseline"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    # ========================================================
    # 2. Audit experimental structure
    # ========================================================

    scenarios = sorted(
        data[
            "scenario_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print()
    print("=" * 100)

    print(
        "Figure 5.5.1 — "
        "OFAT sensitivity of key performance indicators"
    )

    print("=" * 100)

    print(
        f"Source file          : {input_path}"
    )

    print(
        f"Rows                 : {len(data):,}"
    )

    print(
        f"Sensitivity scenarios: {scenarios}"
    )

    print(
        f"Number of scenarios  : {len(scenarios)}"
    )

    print()

    print(
        "Runs per scenario:"
    )

    print(
        data
        .groupby(
            "scenario_id"
        )
        .size()
        .to_string()
    )

    # ========================================================
    # 3. Build summary for all six parameters
    # ========================================================

    rng = np.random.default_rng(
        args.bootstrap_seed
    )

    all_summaries = []

    parameter_datasets = {}

    for definition in PARAMETERS:

        parameter_name = (
            definition["name"]
        )

        parameter_data = (
            build_parameter_data(
                data,
                definition,
            )
        )

        parameter_datasets[
            parameter_name
        ] = parameter_data

        summary = (
            summarize_parameter(
                parameter_data,
                parameter_name,
                repetitions=(
                    args.bootstrap_repetitions
                ),
                rng=rng,
            )
        )

        all_summaries.append(
            summary
        )

    summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    # ========================================================
    # 4. Plot 2 × 3 panels
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.4, 8.2),
        sharey=True,
    )

    axes = axes.flatten()

    for ax, definition in zip(
        axes,
        PARAMETERS,
    ):

        parameter_name = (
            definition["name"]
        )

        panel = summary[
            summary[
                "parameter"
            ].eq(parameter_name)
        ].copy()

        # ----------------------------------------------------
        # Plot each KPI
        # ----------------------------------------------------

        for metric_definition in METRICS:

            metric_column = (
                metric_definition[
                    "column"
                ]
            )

            metric_label = (
                metric_definition[
                    "label"
                ]
            )

            metric_data = panel[
                panel[
                    "metric"
                ].eq(metric_column)
            ].copy()

            metric_data = (
                metric_data
                .sort_values(
                    "parameter_value"
                )
            )

            x = metric_data[
                "parameter_value"
            ].to_numpy(
                dtype=float
            )

            y = (
                metric_data[
                    "mean"
                ].to_numpy(
                    dtype=float
                )
                * 100.0
            )

            lower = (
                metric_data[
                    "ci95_lower"
                ].to_numpy(
                    dtype=float
                )
                * 100.0
            )

            upper = (
                metric_data[
                    "ci95_upper"
                ].to_numpy(
                    dtype=float
                )
                * 100.0
            )

            yerr = np.vstack(
                [
                    y - lower,
                    upper - y,
                ]
            )

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.4,
                markersize=5.5,
                capsize=3,
                label=metric_label,
            )

        # ----------------------------------------------------
        # Mark baseline parameter value
        # ----------------------------------------------------

        baseline_value = (
            panel.loc[
                panel[
                    "is_baseline"
                ],
                "parameter_value",
            ]
            .drop_duplicates()
        )

        if len(baseline_value) == 1:

            baseline_value = float(
                baseline_value.iloc[0]
            )

            ax.axvline(
                baseline_value,
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
            )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        ax.set_title(
            definition["title"],
            fontsize=11,
        )

        ax.set_xlabel(
            definition["xlabel"],
            fontsize=9.5,
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

        tested_values = sorted(
            panel[
                "parameter_value"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        ax.set_xticks(
            tested_values
        )

        xtick_labels = []

        for value in tested_values:

            if np.isclose(
                value,
                round(value),
            ):
                xtick_labels.append(
                    f"{int(round(value)):,}"
                )
            else:
                xtick_labels.append(
                    f"{value:g}"
                )

        ax.set_xticklabels(
            xtick_labels,
            fontsize=8.5,
        )

    # ========================================================
    # 5. Shared Y axis
    # ========================================================

    axes[0].set_ylabel(
        "Service rate (%)",
        fontsize=10,
    )

    axes[3].set_ylabel(
        "Service rate (%)",
        fontsize=10,
    )

    # Use common scale, but do NOT force 0–100 if
    # all results occupy a narrower useful range.
    valid_lower = (
        summary[
            "ci95_lower"
        ]
        .dropna()
        .min()
        * 100.0
    )

    valid_upper = (
        summary[
            "ci95_upper"
        ]
        .dropna()
        .max()
        * 100.0
    )

    ymin = max(
        0.0,
        np.floor(
            (valid_lower - 5.0)
            / 5.0
        )
        * 5.0,
    )

    ymax = min(
        100.0,
        np.ceil(
            (valid_upper + 5.0)
            / 5.0
        )
        * 5.0,
    )

    if ymax <= ymin:
        ymin = 0.0
        ymax = 100.0

    for ax in axes:
        ax.set_ylim(
            ymin,
            ymax,
        )

    # ========================================================
    # 6. Shared legend
    # ========================================================

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
    )

    fig.tight_layout(
        rect=[
            0,
            0.055,
            1,
            1,
        ]
    )

    # ========================================================
    # 7. Save
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
        / "Figure_5_5_1_OFAT_sensitivity_KPIs.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_5_1_OFAT_sensitivity_KPIs.pdf"
    )

    summary_path = (
        output_dir
        / "Figure_5_5_1_OFAT_sensitivity_plot_data.csv"
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

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.close(fig)

    # ========================================================
    # 8. Console audit
    # ========================================================

    print()
    print(
        "OFAT parameter values including baseline:"
    )

    for definition in PARAMETERS:

        name = definition["name"]

        values = sorted(
            summary.loc[
                summary[
                    "parameter"
                ].eq(name),
                "parameter_value",
            ]
            .drop_duplicates()
            .tolist()
        )

        print(
            f"  {name}: {values}"
        )

    print()

    display = summary.copy()

    display[
        "mean_percent"
    ] = (
        display[
            "mean"
        ]
        * 100.0
    )

    display[
        "ci95_lower_percent"
    ] = (
        display[
            "ci95_lower"
        ]
        * 100.0
    )

    display[
        "ci95_upper_percent"
    ] = (
        display[
            "ci95_upper"
        ]
        * 100.0
    )

    print(
        display[
            [
                "parameter",
                "parameter_value",
                "metric_label",
                "n_scenarios",
                "mean_percent",
                "ci95_lower_percent",
                "ci95_upper_percent",
                "is_baseline",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"PNG saved            : {png_path}"
    )

    print(
        f"PDF saved            : {pdf_path}"
    )

    print(
        f"Plot data            : {summary_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()