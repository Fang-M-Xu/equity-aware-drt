from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.4.2: DRT accessibility conversion for "
            "walking- and transit-disadvantaged requests."
        )
    )
    parser.add_argument(
        "--input-file",
        default=(
            "outputs/metrics/multimodal_evaluation/"
            "request_consensus_across_seeds.csv"
        ),
        help=(
            "Request-level consensus output produced by "
            "17_5_analyze_drt_vs_benchmarks.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_multimodal",
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "t", "yes", "y"})
    )


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def validate(frame: pd.DataFrame) -> None:
    required = {
        "walk_within_threshold",
        "transit_within_threshold",
        "drt_access_probability",
        "mean_walk_time_saving_sec",
        "mean_transit_time_saving_sec",
        "is_gap",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            "request_consensus_across_seeds.csv is missing required columns: "
            f"{missing}\nAvailable columns: {frame.columns.tolist()}"
        )


def build_row(
    frame: pd.DataFrame,
    population: str,
    disadvantage: str,
    eligible_mask: pd.Series,
    saving_column: str,
) -> dict[str, object]:
    eligible = frame.loc[eligible_mask].copy()
    n_eligible = int(len(eligible))

    conversion_rate = safe_rate(
        pd.to_numeric(
            eligible["drt_access_probability"],
            errors="coerce",
        ).fillna(0.0).sum(),
        n_eligible,
    )

    saving_values = pd.to_numeric(
        eligible[saving_column],
        errors="coerce",
    )
    paired_mask = np.isfinite(saving_values)

    mean_saving_sec = finite_mean(saving_values.loc[paired_mask])
    mean_saving_min = (
        mean_saving_sec / 60.0
        if np.isfinite(mean_saving_sec)
        else float("nan")
    )

    return {
        "Population": population,
        "Benchmark disadvantage": disadvantage,
        "Eligible requests": n_eligible,
        "DRT conversion rate": conversion_rate,
        "Mean paired DRT time saving (min)": mean_saving_min,
        "Paired time observations": int(paired_mask.sum()),
    }


def main() -> None:
    args = parse_args()

    input_path = resolve(args.input_file)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run 17_5_analyze_drt_vs_benchmarks.py first."
        )

    frame = pd.read_csv(input_path)
    validate(frame)

    frame = frame.copy()
    frame["walk_within_threshold"] = to_bool(
        frame["walk_within_threshold"]
    )
    frame["transit_within_threshold"] = to_bool(
        frame["transit_within_threshold"]
    )
    frame["is_gap"] = to_bool(frame["is_gap"])

    walk_disadvantaged = ~frame["walk_within_threshold"]
    transit_disadvantaged = ~frame["transit_within_threshold"]
    gap_origin = frame["is_gap"]

    rows = [
        build_row(
            frame,
            population="All requests",
            disadvantage="Walking > 15 min",
            eligible_mask=walk_disadvantaged,
            saving_column="mean_walk_time_saving_sec",
        ),
        build_row(
            frame,
            population="All requests",
            disadvantage="GTFS > 15 min / unavailable",
            eligible_mask=transit_disadvantaged,
            saving_column="mean_transit_time_saving_sec",
        ),
        build_row(
            frame,
            population="Gap-origin requests",
            disadvantage="Walking > 15 min",
            eligible_mask=gap_origin & walk_disadvantaged,
            saving_column="mean_walk_time_saving_sec",
        ),
        build_row(
            frame,
            population="Gap-origin requests",
            disadvantage="GTFS > 15 min / unavailable",
            eligible_mask=gap_origin & transit_disadvantaged,
            saving_column="mean_transit_time_saving_sec",
        ),
    ]

    table = pd.DataFrame(rows)

    output_path = (
        output_dir / "Table_5_4_2_DRT_accessibility_conversion.csv"
    )
    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print(f"Saved: {output_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
