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

GROUP_SPECS = [
    ("is_gap", "1", "Gap-origin"),
    ("is_no_car", "1", "No car"),
    ("is_low_income", "1", "Low income"),
    ("is_older", "1", "Older person"),
    ("is_student", "1", "Student"),
    ("sector_group", "Ultra-Orthodox", "Ultra-Orthodox sector"),
    ("sector_group", "Arab", "Arab sector"),
    ("sector_group", "Secular", "Secular sector"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.4.3: Group-level within-15-minute "
            "accessibility under DRT, direct walking, and GTFS transit."
        )
    )
    parser.add_argument(
        "--input-file",
        default=(
            "outputs/metrics/multimodal_evaluation/"
            "group_drt_vs_benchmarks.csv"
        ),
        help=(
            "Group-level multimodal evaluation file produced by "
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


def normalize_group_value(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass

    return text


def validate(frame: pd.DataFrame) -> None:
    required = {
        "group_variable",
        "group_value",
        "walking_access_rate",
        "transit_access_rate",
        "drt_expected_access_rate",
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise KeyError(
            "group_drt_vs_benchmarks.csv is missing required columns: "
            f"{missing}\nAvailable columns: {frame.columns.tolist()}"
        )


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    data["group_variable"] = data["group_variable"].astype(str).str.strip()
    data["group_value_norm"] = data["group_value"].map(normalize_group_value)

    rows = []

    for variable, value, label in GROUP_SPECS:
        subset = data.loc[
            data["group_variable"].eq(variable)
            & data["group_value_norm"].eq(value)
        ].copy()

        if subset.empty:
            print(
                f"Warning: group not found and skipped: "
                f"{variable}={value} ({label})"
            )
            continue

        if len(subset) > 1:
            print(
                f"Warning: multiple rows found for {variable}={value}; "
                "using the first row."
            )

        row = subset.iloc[0]

        rows.append(
            {
                "Population group": label,
                "Proposed DRT": float(row["drt_expected_access_rate"]),
                "Direct walking": float(row["walking_access_rate"]),
                "GTFS transit": float(row["transit_access_rate"]),
            }
        )

    if not rows:
        raise RuntimeError(
            "None of the requested population groups were found in "
            "group_drt_vs_benchmarks.csv."
        )

    return pd.DataFrame(rows)


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

    table = build_table(frame)

    output_path = (
        output_dir
        / "Table_5_4_3_group_level_multimodal_accessibility.csv"
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
