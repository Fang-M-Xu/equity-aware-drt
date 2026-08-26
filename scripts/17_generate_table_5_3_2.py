from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "alns_efficiency_only": "Efficiency-only ALNS",
    "alns_gap_only": "Gap-only ALNS",
    "proposed_full": "Proposed-full ALNS",
}

METRICS = [
    ("gap_service_rate", "Gap-origin SR"),
    ("worst_group_service_rate", "Worst-group SR"),
    ("max_min_service_gap", "Max–min gap"),
    ("jain_service_index", "Jain index"),
    ("theil_service_index", "Theil index"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table 5.3.2: equity and disparity metrics under different ALNS route objective configurations."
    )
    parser.add_argument(
        "--scenario-metrics",
        default="outputs/paper_results/chapter5_routing/routing_scenario_metrics.csv",
        help="Scenario-level routing/equity metrics produced by 20_evaluate_routing_equity.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_routing",
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    input_path = resolve(args.scenario_metrics)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Scenario-level routing/equity file not found: {input_path}\n"
            "Run 20_evaluate_routing_equity.py first."
        )

    frame = pd.read_csv(input_path)
    required = {"method", "scenario_id"} | {metric for metric, _ in METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            "The scenario-level file is missing required columns:\n"
            f"{missing}\nAvailable columns:\n{frame.columns.tolist()}"
        )

    frame["method"] = frame["method"].astype(str)
    frame["scenario_id"] = pd.to_numeric(frame["scenario_id"], errors="coerce")
    frame = frame.dropna(subset=["scenario_id"]).copy()

    rows = []
    for method, label in METHOD_LABELS.items():
        subset = frame.loc[frame["method"].eq(method)].copy()
        if subset.empty:
            raise ValueError(f"Required method {method!r} is absent from {input_path}.")

        row = {
            "Configuration": label,
            "N scenarios": int(subset["scenario_id"].nunique()),
        }
        for metric, display_name in METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[display_name] = float(values.mean()) if not values.empty else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)[
        [
            "Configuration",
            "Gap-origin SR",
            "Worst-group SR",
            "Max–min gap",
            "Jain index",
            "Theil index",
            "N scenarios",
        ]
    ]

    output_path = output_dir / "Table_5_3_2_equity_disparity_metrics.csv"
    table.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"Saved: {output_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
