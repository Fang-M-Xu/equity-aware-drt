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


METHOD_LABELS = {
    "greedy_full": "Greedy",
    "proposed_full": "ALNS",
}

METHOD_ORDER = [
    "Greedy",
    "ALNS",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.3.1 — Greedy and ALNS routing performance."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/base_competitive.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_routing",
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load configuration
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
    # 2. Load scenario-level results
    # ========================================================

    data = pd.read_csv(
        metrics_path
    )

    required_columns = [
        "method",
        "scenario_id",
        "served",
        "service_rate",
        "pickup_delay_mean_sec",
        "ride_time_mean_sec",
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
    # 3. Keep only Greedy and ALNS
    # ========================================================

    frame = data[
        data["method"].isin(
            METHOD_LABELS.keys()
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "Neither greedy_full nor proposed_full "
            "was found in scenario_level_metrics.csv."
        )

    frame["Method"] = (
        frame["method"]
        .map(METHOD_LABELS)
    )

    # ========================================================
    # 4. Check scenario pairing
    # ========================================================

    scenario_counts = (
        frame
        .groupby("Method")[
            "scenario_id"
        ]
        .nunique()
    )

    print()
    print("Scenario counts:")
    print(scenario_counts)

    greedy_scenarios = set(
        frame.loc[
            frame["Method"].eq("Greedy"),
            "scenario_id",
        ]
    )

    alns_scenarios = set(
        frame.loc[
            frame["Method"].eq("ALNS"),
            "scenario_id",
        ]
    )

    common_scenarios = (
        greedy_scenarios
        & alns_scenarios
    )

    if not common_scenarios:
        raise ValueError(
            "Greedy and ALNS have no common scenarios."
        )

    # Use exactly the common realized scenarios
    frame = frame[
        frame["scenario_id"].isin(
            common_scenarios
        )
    ].copy()

    # ========================================================
    # 5. Aggregate across matched scenarios
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
            service_rate=(
                "service_rate",
                "mean",
            ),
            served_requests=(
                "served",
                "mean",
            ),
            pickup_delay_sec=(
                "pickup_delay_mean_sec",
                "mean",
            ),
            ride_time_sec=(
                "ride_time_mean_sec",
                "mean",
            ),
            door_to_door_sec=(
                "door_to_door_mean_sec",
                "mean",
            ),
        )
    )

    # ========================================================
    # 6. Convert time metrics to publication units
    # ========================================================

    summary[
        "ride_time_min"
    ] = (
        summary[
            "ride_time_sec"
        ] / 60.0
    )

    summary[
        "door_to_door_min"
    ] = (
        summary[
            "door_to_door_sec"
        ] / 60.0
    )

    # ========================================================
    # 7. Reorganize into paper table
    # ========================================================

    wide = (
        summary
        .set_index("Method")
    )

    table = pd.DataFrame(
        {
            "Metric": [
                "Service rate",
                "Served requests",
                "Mean pickup delay (s)",
                "Mean ride time (min)",
                "Mean door-to-door time (min)",
            ],

            "Greedy": [
                wide.loc[
                    "Greedy",
                    "service_rate",
                ],
                wide.loc[
                    "Greedy",
                    "served_requests",
                ],
                wide.loc[
                    "Greedy",
                    "pickup_delay_sec",
                ],
                wide.loc[
                    "Greedy",
                    "ride_time_min",
                ],
                wide.loc[
                    "Greedy",
                    "door_to_door_min",
                ],
            ],

            "ALNS": [
                wide.loc[
                    "ALNS",
                    "service_rate",
                ],
                wide.loc[
                    "ALNS",
                    "served_requests",
                ],
                wide.loc[
                    "ALNS",
                    "pickup_delay_sec",
                ],
                wide.loc[
                    "ALNS",
                    "ride_time_min",
                ],
                wide.loc[
                    "ALNS",
                    "door_to_door_min",
                ],
            ],
        }
    )

    # ========================================================
    # 8. Publication formatting
    # ========================================================

    table.loc[
        table["Metric"].eq(
            "Service rate"
        ),
        ["Greedy", "ALNS"],
    ] = (
        table.loc[
            table["Metric"].eq(
                "Service rate"
            ),
            ["Greedy", "ALNS"],
        ]
        .round(4)
    )

    table.loc[
        table["Metric"].eq(
            "Served requests"
        ),
        ["Greedy", "ALNS"],
    ] = (
        table.loc[
            table["Metric"].eq(
                "Served requests"
            ),
            ["Greedy", "ALNS"],
        ]
        .round(2)
    )

    time_rows = ~table[
        "Metric"
    ].isin(
        [
            "Service rate",
            "Served requests",
        ]
    )

    table.loc[
        time_rows,
        ["Greedy", "ALNS"],
    ] = (
        table.loc[
            time_rows,
            ["Greedy", "ALNS"],
        ]
        .round(2)
    )

    # ========================================================
    # 9. Save
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
        / "Table_5_3_1_greedy_vs_alns_routing_performance.csv"
    )

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. Console audit
    # ========================================================

    print()
    print("=" * 78)

    print(
        "Table 5.3.1 — Greedy and ALNS "
        "routing performance"
    )

    print("=" * 78)

    print(
        f"Source file      : {metrics_path}"
    )

    print(
        f"Matched scenarios: {len(common_scenarios)}"
    )

    print()

    print(
        table.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {output_path}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()