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
# Population groups used in Figure 5.3.3
# ============================================================

GROUPS = [
    {
        "label": "Gap-origin",
        "attribute": "is_gap",
        "group": "1",
    },
    {
        "label": "No-car",
        "attribute": "is_no_car",
        "group": "1",
    },
    {
        "label": "Low-income",
        "attribute": "is_low_income",
        "group": "1",
    },
    {
        "label": "Older",
        "attribute": "is_older",
        "group": "1",
    },
    {
        "label": "Student",
        "attribute": "is_student",
        "group": "1",
    },
    {
        "label": "Arab",
        "attribute": "sector_group",
        "group": "Arab",
    },
    {
        "label": "Ultra-Orthodox",
        "attribute": "sector_group",
        "group": "Ultra-Orthodox",
    },
    {
        "label": "Secular",
        "attribute": "sector_group",
        "group": "Secular",
    },
]

METHOD = "proposed_full"


# ============================================================
# Bootstrap CI across scenarios
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(values))

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
        bootstrap_means[i] = np.mean(sample)

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

    return mean_value, lower, upper


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.3.4 — "
            "DRT service rates across population groups."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "outputs/metrics/"
            "group_equity_metrics.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/paper_results/"
            "chapter5_routing"
        ),
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
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load group-level experiment results
    # ========================================================

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{input_path}"
        )

    data = pd.read_csv(input_path)

    required_columns = [
        "attribute",
        "group",
        "requests",
        "served",
        "service_rate",
        "method",
        "scenario_id",
        "solver_seed",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )

    # Make group matching robust
    data["group"] = (
        data["group"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # 2. Keep Proposed-full only
    # ========================================================

    frame = data[
        data["method"].eq(METHOD)
    ].copy()

    if frame.empty:
        raise ValueError(
            f"No rows found for method '{METHOD}'."
        )

    # ========================================================
    # 3. Calculate Proposed-full overall service rate
    #
    # We can recover overall SR from the is_gap partition:
    # is_gap=0 + is_gap=1 jointly cover all requests.
    #
    # First aggregate within scenario × solver seed,
    # then average solver seeds within each scenario,
    # then average scenarios.
    # ========================================================

    gap_partition = frame[
        frame["attribute"].eq("is_gap")
    ].copy()

    if gap_partition.empty:
        raise ValueError(
            "Cannot calculate overall service rate because "
            "the is_gap partition is missing."
        )

    overall_seed = (
        gap_partition
        .groupby(
            [
                "scenario_id",
                "solver_seed",
            ],
            as_index=False,
        )
        .agg(
            requests=("requests", "sum"),
            served=("served", "sum"),
        )
    )

    overall_seed[
        "service_rate"
    ] = (
        overall_seed["served"]
        / overall_seed["requests"]
    )

    overall_scenario = (
        overall_seed
        .groupby(
            "scenario_id",
            as_index=False,
        )
        .agg(
            service_rate=(
                "service_rate",
                "mean",
            )
        )
    )

    overall_service_rate = float(
        overall_scenario[
            "service_rate"
        ].mean()
    )

    # ========================================================
    # 4. Extract subgroup results
    #
    # Important:
    # First average repeated solver seeds WITHIN scenario.
    # Then scenarios become the statistical units.
    # ========================================================

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    for definition in GROUPS:

        label = definition["label"]
        attribute = definition["attribute"]
        group_value = definition["group"]

        subset = frame[
            frame["attribute"].eq(attribute)
            & frame["group"].eq(group_value)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No data found for group:\n"
                f"{label}\n"
                f"attribute={attribute}, "
                f"group={group_value}"
            )

        # -----------------------------------------------
        # First average the repeated solver seeds
        # within each realized scenario.
        # -----------------------------------------------

        scenario_values = (
            subset
            .groupby(
                "scenario_id",
                as_index=False,
            )
            .agg(
                service_rate=(
                    "service_rate",
                    "mean",
                ),
                mean_requests=(
                    "requests",
                    "mean",
                ),
            )
        )

        mean_sr, ci_lower, ci_upper = (
            bootstrap_mean_ci(
                scenario_values[
                    "service_rate"
                ].to_numpy(dtype=float),
                repetitions=(
                    args.bootstrap_repetitions
                ),
                rng=rng,
            )
        )

        rows.append(
            {
                "Group": label,
                "attribute": attribute,
                "group_value": group_value,
                "n_scenarios": int(
                    scenario_values[
                        "scenario_id"
                    ].nunique()
                ),
                "mean_requests_per_scenario":
                    float(
                        scenario_values[
                            "mean_requests"
                        ].mean()
                    ),
                "service_rate":
                    mean_sr,
                "ci95_lower":
                    ci_lower,
                "ci95_upper":
                    ci_upper,
                "difference_from_overall":
                    mean_sr
                    - overall_service_rate,
            }
        )

    results = pd.DataFrame(rows)

    # ========================================================
    # 5. Preserve requested paper order
    # ========================================================

    group_order = [
        definition["label"]
        for definition in GROUPS
    ]

    results["Group"] = pd.Categorical(
        results["Group"],
        categories=group_order,
        ordered=True,
    )

    results = (
        results
        .sort_values("Group")
        .reset_index(drop=True)
    )

    # ========================================================
    # 6. Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.3, 5.8)
    )

    # Reverse so first group appears at the top
    plot_data = (
        results
        .iloc[::-1]
        .reset_index(drop=True)
    )

    y = np.arange(
        len(plot_data)
    )

    x = plot_data[
        "service_rate"
    ].to_numpy(dtype=float)

    lower = plot_data[
        "ci95_lower"
    ].to_numpy(dtype=float)

    upper = plot_data[
        "ci95_upper"
    ].to_numpy(dtype=float)

    xerr = np.vstack(
        [
            x - lower,
            upper - x,
        ]
    )

    # --------------------------------------------------------
    # Overall Proposed-full reference line
    # --------------------------------------------------------

    ax.axvline(
        overall_service_rate,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Overall service rate "
            f"({overall_service_rate:.3f})"
        ),
        zorder=1,
    )

    # --------------------------------------------------------
    # Group estimates + bootstrap 95% CI
    # --------------------------------------------------------

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7,
        capsize=3,
        linewidth=1.1,
        zorder=3,
    )

    # ========================================================
    # 7. Labels
    # ========================================================

    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_data[
            "Group"
        ].astype(str)
    )

    ax.set_xlabel(
        "Service rate"
    )

    ax.set_ylabel(
        "Population group"
    )

    # ========================================================
    # 8. Add numeric labels
    # ========================================================

    for yi, value in zip(
        y,
        x,
    ):

        ax.annotate(
            f"{value:.3f}",
            xy=(value, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )

    # ========================================================
    # 9. Axis range
    # ========================================================

    minimum = min(
        float(lower.min()),
        overall_service_rate,
    )

    maximum = max(
        float(upper.max()),
        overall_service_rate,
    )

    margin = max(
        (maximum - minimum) * 0.18,
        0.04,
    )

    ax.set_xlim(
        max(
            0.0,
            minimum - margin,
        ),
        min(
            1.0,
            maximum + margin,
        ),
    )

    # ========================================================
    # 10. Publication formatting
    # ========================================================

    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=9,
    )

    fig.tight_layout()

    # ========================================================
    # 11. Save outputs
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
        / "Figure_5_3_4_group_service_rates.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_3_4_group_service_rates.pdf"
    )

    data_path = (
        output_dir
        / "Figure_5_3_4_group_service_rates_data.csv"
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

    results.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.close(fig)

    # ========================================================
    # 12. Console audit
    # ========================================================

    print()
    print("=" * 82)

    print(
        "Figure 5.3.4 — DRT service rates "
        "across population groups"
    )

    print("=" * 82)

    print(
        f"Source              : {input_path}"
    )

    print(
        f"Method              : {METHOD}"
    )

    print(
        f"Overall service rate: "
        f"{overall_service_rate:.4f}"
    )

    print()

    display = results[
        [
            "Group",
            "n_scenarios",
            "mean_requests_per_scenario",
            "service_rate",
            "ci95_lower",
            "ci95_upper",
            "difference_from_overall",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print()

    print(
        f"PNG saved           : {png_path}"
    )

    print(
        f"PDF saved           : {pdf_path}"
    )

    print(
        f"Plot data saved     : {data_path}"
    )

    print("=" * 82)


if __name__ == "__main__":
    main()