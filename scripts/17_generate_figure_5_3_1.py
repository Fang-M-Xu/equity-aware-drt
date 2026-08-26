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


GREEDY_METHOD = "greedy_full"
ALNS_METHOD = "proposed_full"


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 5.3.1 — "
            "Greedy versus ALNS performance across scenarios."
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
    # 1. Load formal scenario-level metrics
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

    required_columns = [
        "method",
        "scenario_id",
        "service_rate",
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
    # 2. Keep Greedy and ALNS only
    # ========================================================

    frame = data[
        data["method"].isin(
            [
                GREEDY_METHOD,
                ALNS_METHOD,
            ]
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No greedy_full / proposed_full records found."
        )

    frame["service_rate"] = pd.to_numeric(
        frame["service_rate"],
        errors="coerce",
    )

    # ========================================================
    # 3. Build paired scenario table
    # ========================================================

    paired = (
        frame
        .pivot_table(
            index="scenario_id",
            columns="method",
            values="service_rate",
            aggfunc="mean",
        )
        .dropna(
            subset=[
                GREEDY_METHOD,
                ALNS_METHOD,
            ]
        )
        .reset_index()
    )

    if paired.empty:
        raise ValueError(
            "No matched Greedy–ALNS scenarios were found."
        )

    paired = paired.sort_values(
        "scenario_id"
    ).reset_index(drop=True)

    paired = paired.rename(
        columns={
            GREEDY_METHOD: "Greedy",
            ALNS_METHOD: "ALNS",
        }
    )

    # ========================================================
    # 4. Paired differences
    # ========================================================

    paired["difference"] = (
        paired["ALNS"]
        - paired["Greedy"]
    )

    greedy_mean = float(
        paired["Greedy"].mean()
    )

    alns_mean = float(
        paired["ALNS"].mean()
    )

    difference_mean = float(
        paired["difference"].mean()
    )

    improved_count = int(
        (paired["difference"] > 0).sum()
    )

    equal_count = int(
        np.isclose(
            paired["difference"],
            0.0,
        ).sum()
    )

    worse_count = int(
        (paired["difference"] < 0).sum()
    )

    # ========================================================
    # 5. Create Figure 5.3.1
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.3),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.55, 1.0],
        },
    )

    scenario_x = np.arange(
        len(paired)
    )

    # ========================================================
    # Panel (a): scenario-level service rates
    # ========================================================

    ax = axes[0]

    # Thin paired connectors
    for x, greedy, alns in zip(
        scenario_x,
        paired["Greedy"],
        paired["ALNS"],
    ):
        ax.plot(
            [x, x],
            [greedy, alns],
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )

    # Greedy observations
    ax.scatter(
        scenario_x,
        paired["Greedy"],
        marker="o",
        s=38,
        label="Greedy",
        zorder=3,
    )

    # ALNS observations
    ax.scatter(
        scenario_x,
        paired["ALNS"],
        marker="s",
        s=38,
        label="ALNS",
        zorder=3,
    )

    # Mean reference lines
    ax.axhline(
        greedy_mean,
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.axhline(
        alns_mean,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.set_ylabel(
        "Service rate"
    )

    ax.set_title(
        "(a) Scenario-level paired service rates",
        fontsize=11,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========================================================
    # Panel (b): ALNS − Greedy
    # ========================================================

    ax = axes[1]

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    # vertical line from zero to difference
    ax.vlines(
        scenario_x,
        0,
        paired["difference"],
        linewidth=1.0,
        alpha=0.65,
    )

    ax.scatter(
        scenario_x,
        paired["difference"],
        s=38,
        zorder=3,
    )

    # Mean paired difference
    ax.axhline(
        difference_mean,
        linestyle=":",
        linewidth=1.2,
        label=(
            f"Mean difference = "
            f"{difference_mean:+.3f}"
        ),
    )

    ax.set_ylabel(
        "Δ Service rate\n(ALNS − Greedy)"
    )

    ax.set_xlabel(
        "Scenario"
    )

    ax.set_title(
        "(b) Paired service-rate difference",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========================================================
    # 6. Scenario labels
    # ========================================================

    ax.set_xticks(
        scenario_x
    )

    ax.set_xticklabels(
        paired["scenario_id"].astype(str),
        rotation=0,
        fontsize=8,
    )

    # ========================================================
    # 7. Summary note
    # ========================================================

    fig.text(
        0.5,
        0.01,
        (
            f"Matched scenarios: {len(paired)}; "
            f"ALNS higher: {improved_count}; "
            f"equal: {equal_count}; "
            f"lower: {worse_count}."
        ),
        ha="center",
        fontsize=8.5,
    )

    fig.tight_layout(
        rect=[
            0,
            0.035,
            1,
            1,
        ]
    )

    # ========================================================
    # 8. Save
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
        / "Figure_5_3_1_greedy_vs_alns_across_scenarios.png"
    )

    pdf_path = (
        output_dir
        / "Figure_5_3_1_greedy_vs_alns_across_scenarios.pdf"
    )

    data_path = (
        output_dir
        / "Figure_5_3_1_greedy_vs_alns_across_scenarios_data.csv"
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

    paired.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.close(fig)

    # ========================================================
    # 9. Console audit
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Figure 5.3.1 — Greedy versus ALNS "
        "performance across scenarios"
    )

    print("=" * 80)

    print(
        f"Source             : {input_path}"
    )

    print(
        f"Matched scenarios  : {len(paired)}"
    )

    print(
        f"Greedy mean SR     : {greedy_mean:.4f}"
    )

    print(
        f"ALNS mean SR       : {alns_mean:.4f}"
    )

    print(
        f"Mean paired Δ      : {difference_mean:+.4f}"
    )

    print(
        f"ALNS higher        : {improved_count}"
    )

    print(
        f"Equal              : {equal_count}"
    )

    print(
        f"ALNS lower         : {worse_count}"
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

    print("=" * 80)


if __name__ == "__main__":
    main()