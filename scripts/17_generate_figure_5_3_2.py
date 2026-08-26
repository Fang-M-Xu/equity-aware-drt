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

from config import load_config, path_from_config


# ============================================================
# Ablation variants used in Section 5.3.2
# ============================================================

METHOD_LABELS = {
    "alns_efficiency_only": "Efficiency-only",
    "alns_gap_only": "Gap-only",
    "proposed_full": "Proposed-full",
}

METHOD_ORDER = [
    "Efficiency-only",
    "Gap-only",
    "Proposed-full",
]


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.3.2 — Operational–equity "
            "trade-offs across routing objectives."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_routing",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load formal scenario-level results
    # ========================================================

    config = load_config(args.config)

    metrics_dir = path_from_config(
        config,
        "metrics_dir",
    )

    input_path = (
        metrics_dir
        / "scenario_level_metrics.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{input_path}"
        )

    data = pd.read_csv(input_path)

    # ========================================================
    # 2. Required fields
    # ========================================================

    required_columns = [
        "method",
        "scenario_id",
        "service_rate",
        "gap_requests",
        "gap_served",
        "gap_closure_rate",
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

    # ========================================================
    # 3. Keep only the three routing-objective variants
    # ========================================================

    frame = data[
        data["method"].isin(
            METHOD_LABELS.keys()
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No objective-ablation methods were found."
        )

    available_methods = set(
        frame["method"].unique()
    )

    missing_methods = [
        method
        for method in METHOD_LABELS
        if method not in available_methods
    ]

    if missing_methods:
        raise ValueError(
            "Missing required ablation methods:\n"
            f"{missing_methods}"
        )

    frame["Method"] = (
        frame["method"]
        .map(METHOD_LABELS)
    )

    # ========================================================
    # 4. Validate gap-origin service-rate definition
    #
    # In the current evaluation output:
    #
    # gap_closure_rate = gap_served / gap_requests
    #
    # We therefore display it in the paper as
    # "Gap-origin service rate".
    # ========================================================

    calculated_gap_rate = (
        frame["gap_served"]
        / frame["gap_requests"]
    )

    difference = (
        calculated_gap_rate
        - frame["gap_closure_rate"]
    ).abs()

    valid_difference = (
        difference
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if (
        len(valid_difference) > 0
        and valid_difference.max() > 1e-8
    ):
        raise ValueError(
            "gap_closure_rate is not consistent with "
            "gap_served / gap_requests."
        )

    # ========================================================
    # 5. Check that all methods use common scenarios
    # ========================================================

    scenario_sets = {
        method: set(
            frame.loc[
                frame["method"].eq(method),
                "scenario_id",
            ]
        )
        for method in METHOD_LABELS
    }

    common_scenarios = set.intersection(
        *scenario_sets.values()
    )

    if not common_scenarios:
        raise ValueError(
            "The three ablation methods have no common scenarios."
        )

    frame = frame[
        frame["scenario_id"].isin(
            common_scenarios
        )
    ].copy()

    # ========================================================
    # 6. Aggregate across matched scenarios
    # ========================================================

    summary = (
        frame
        .groupby(
            "Method",
            as_index=False,
        )
        .agg(
            n_scenarios=(
                "scenario_id",
                "nunique",
            ),

            overall_service_rate=(
                "service_rate",
                "mean",
            ),

            gap_origin_service_rate=(
                "gap_closure_rate",
                "mean",
            ),
        )
    )

    # ========================================================
    # 7. Order methods
    # ========================================================

    summary["Method"] = pd.Categorical(
        summary["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("Method")
        .reset_index(drop=True)
    )

    # ========================================================
    # 8. Optional scenario-level variability
    #
    # SD is calculated only to provide visual context.
    # The main plotted points remain scenario-level means.
    # ========================================================

    variability = (
        frame
        .groupby("Method")
        .agg(
            service_rate_sd=(
                "service_rate",
                "std",
            ),
            gap_rate_sd=(
                "gap_closure_rate",
                "std",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        variability,
        on="Method",
        how="left",
    )

    # ========================================================
    # 9. Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.4, 6.2)
    )

    # --------------------------------------------------------
    # Plot the three mean points
    # --------------------------------------------------------

    for _, row in summary.iterrows():

        x = float(
            row["overall_service_rate"]
        )

        y = float(
            row["gap_origin_service_rate"]
        )

        method = str(
            row["Method"]
        )

        ax.scatter(
            x,
            y,
            s=90,
            marker="o",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        # Slightly offset labels so they do not cover markers
        ax.annotate(
            method,
            xy=(x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
            ha="left",
            va="bottom",
        )

    # ========================================================
    # 10. Connect the ablation progression
    #
    # Efficiency-only → Gap-only → Proposed-full
    #
    # This is not a time series; the thin line simply helps
    # readers see how objective components shift the solution.
    # ========================================================

    ax.plot(
        summary[
            "overall_service_rate"
        ].to_numpy(),
        summary[
            "gap_origin_service_rate"
        ].to_numpy(),
        linewidth=0.9,
        linestyle="--",
        alpha=0.55,
        zorder=2,
    )

    # ========================================================
    # 11. Axis labels
    # ========================================================

    ax.set_xlabel(
        "Overall service rate",
        fontsize=11,
    )

    ax.set_ylabel(
        "Gap-origin service rate",
        fontsize=11,
    )

    # ========================================================
    # 12. Add "better" direction
    # ========================================================

    ax.annotate(
        "Higher operational performance",
        xy=(0.98, 0.035),
        xycoords="axes fraction",
        xytext=(0.63, 0.035),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
        ),
        ha="center",
        va="center",
        fontsize=8.5,
    )

    ax.annotate(
        "Higher equity performance",
        xy=(0.035, 0.98),
        xycoords="axes fraction",
        xytext=(0.035, 0.63),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
        ),
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.5,
    )

    # ========================================================
    # 13. Axis range with small margins
    # ========================================================

    x_values = summary[
        "overall_service_rate"
    ].to_numpy(dtype=float)

    y_values = summary[
        "gap_origin_service_rate"
    ].to_numpy(dtype=float)

    x_range = max(
        float(np.max(x_values) - np.min(x_values)),
        0.02,
    )

    y_range = max(
        float(np.max(y_values) - np.min(y_values)),
        0.02,
    )

    ax.set_xlim(
        max(
            0.0,
            float(np.min(x_values))
            - x_range * 0.30,
        ),
        min(
            1.0,
            float(np.max(x_values))
            + x_range * 0.45,
        ),
    )

    ax.set_ylim(
        max(
            0.0,
            float(np.min(y_values))
            - y_range * 0.30,
        ),
        min(
            1.0,
            float(np.max(y_values))
            + y_range * 0.45,
        ),
    )

    # ========================================================
    # 14. Publication formatting
    # ========================================================

    ax.grid(
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    # ========================================================
    # 15. Save
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
        / "Figure_5_3_2_operational_equity_tradeoff.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_3_2_operational_equity_tradeoff.pdf"
    )

    data_path = (
        output_dir
        / "Figure_5_3_2_operational_equity_tradeoff_data.csv"
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
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.close(fig)

    # ========================================================
    # 16. Console audit
    # ========================================================

    print()
    print("=" * 82)

    print(
        "Figure 5.3.2 — Operational–equity "
        "trade-offs across routing objectives"
    )

    print("=" * 82)

    print(
        f"Source             : {input_path}"
    )

    print(
        f"Matched scenarios  : {len(common_scenarios)}"
    )

    print()

    print(
        summary[
            [
                "Method",
                "n_scenarios",
                "overall_service_rate",
                "gap_origin_service_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"PNG saved          : {png_path}"
    )

    print(
        f"PDF saved          : {pdf_path}"
    )

    print(
        f"Plot data saved    : {data_path}"
    )

    print("=" * 82)


if __name__ == "__main__":
    main()