from __future__ import annotations

import argparse

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
# Methods used specifically for the prepositioning comparison
# ============================================================

METHOD_LABELS = {
    "alns_random_preposition": "Random",
    "alns_demand_preposition": "Expected-demand",
    "proposed_full": "Equity-weighted",
}

METHOD_ORDER = [
    "Random",
    "Expected-demand",
    "Equity-weighted",
]


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.2.1 — Operational performance "
            "by prepositioning strategy."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base_competitive.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_prepositioning",
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load experiment configuration
    # ========================================================

    config = load_config(args.config)

    metrics_dir = path_from_config(
        config,
        "metrics_dir",
    )

    metrics_path = (
        metrics_dir
        / "scenario_level_metrics.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{metrics_path}"
        )

    # ========================================================
    # 2. Load formal scenario-level evaluation results
    # ========================================================

    data = pd.read_csv(
        metrics_path
    )

    required_columns = [
        "method",
        "scenario_id",
        "requests",
        "served",
        "service_rate",
        "gap_requests",
        "gap_served",
        "gap_closure_rate",
        "pickup_delay_mean_sec",
        "door_to_door_mean_sec",
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
    # 3. Keep only the three prepositioning strategies
    # ========================================================

    frame = data[
        data["method"].isin(
            METHOD_LABELS.keys()
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No prepositioning strategies were found."
        )

    frame["Prepositioning strategy"] = (
        frame["method"]
        .map(METHOD_LABELS)
    )

    # ========================================================
    # 4. Validate gap-origin service-rate definition
    #
    # In the current scenario_level_metrics.csv:
    #
    # gap_closure_rate = gap_served / gap_requests
    #
    # Therefore it represents the service rate among
    # requests originating from service-gap TAZs.
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
        difference.dropna()
    )

    if (
        len(valid_difference) > 0
        and valid_difference.max() > 1e-8
    ):
        raise ValueError(
            "gap_closure_rate is inconsistent with "
            "gap_served / gap_requests."
        )

    # ========================================================
    # 5. Aggregate across the 30 realized scenarios
    #
    # Each scenario contributes equally to the reported mean.
    # ========================================================

    summary = (
        frame
        .groupby(
            "Prepositioning strategy",
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

            mean_pickup_delay_sec=(
                "pickup_delay_mean_sec",
                "mean",
            ),

            mean_door_to_door_sec=(
                "door_to_door_mean_sec",
                "mean",
            ),
        )
    )

    # ========================================================
    # 6. Convert door-to-door seconds to minutes
    # ========================================================

    summary[
        "mean_door_to_door_min"
    ] = (
        summary[
            "mean_door_to_door_sec"
        ]
        / 60.0
    )

    # ========================================================
    # 7. Build publication table
    # ========================================================

    table = summary[
        [
            "Prepositioning strategy",
            "overall_service_rate",
            "gap_origin_service_rate",
            "mean_pickup_delay_sec",
            "mean_door_to_door_min",
        ]
    ].copy()

    table = table.rename(
        columns={
            "overall_service_rate":
                "Overall service rate",

            "gap_origin_service_rate":
                "Gap-origin service rate",

            "mean_pickup_delay_sec":
                "Mean pickup delay (s)",

            "mean_door_to_door_min":
                "Mean door-to-door time (min)",
        }
    )

    # ========================================================
    # 8. Method order
    # ========================================================

    table[
        "Prepositioning strategy"
    ] = pd.Categorical(
        table[
            "Prepositioning strategy"
        ],
        categories=METHOD_ORDER,
        ordered=True,
    )

    table = (
        table
        .sort_values(
            "Prepositioning strategy"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 9. Publication rounding
    # ========================================================

    table[
        "Overall service rate"
    ] = table[
        "Overall service rate"
    ].round(4)

    table[
        "Gap-origin service rate"
    ] = table[
        "Gap-origin service rate"
    ].round(4)

    table[
        "Mean pickup delay (s)"
    ] = table[
        "Mean pickup delay (s)"
    ].round(1)

    table[
        "Mean door-to-door time (min)"
    ] = table[
        "Mean door-to-door time (min)"
    ].round(2)

    # ========================================================
    # 10. Save
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "Table_5_2_1_operational_performance_by_prepositioning.csv"
    )

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 11. Console report
    # ========================================================

    print()
    print("=" * 85)

    print(
        "Table 5.2.1 — Operational performance "
        "by prepositioning strategy"
    )

    print("=" * 85)

    print(
        f"Source: {metrics_path}"
    )

    print(
        f"Scenario rows used: {len(frame)}"
    )

    print()

    print(
        table.to_string(index=False)
    )

    print()

    print(
        f"Saved: {output_path}"
    )

    print("=" * 85)


if __name__ == "__main__":
    main()