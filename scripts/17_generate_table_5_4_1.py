from __future__ import annotations

import argparse
import re

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
# Constants
# ============================================================

METHOD_DIR = "proposed_full"

POPULATIONS = [
    "All requests",
    "Gap-origin",
]

MODES = [
    "Walking",
    "Public transit",
    "DRT",
]


# ============================================================
# Helpers
# ============================================================

def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(series, errors="coerce")
            .fillna(0)
            .ne(0)
        )

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "true",
                "yes",
                "y",
            }
        )
    )


def extract_scenario_id(path: Path) -> int:
    match = re.search(
        r"scenario[_-]?0*(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Could not extract scenario_id from:\n{path}"
        )

    return int(match.group(1))


def extract_solver_seed(path: Path) -> int:
    match = re.search(
        r"seed[_-]?(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Could not extract solver seed from:\n{path}"
        )

    return int(match.group(1))


# ============================================================
# Benchmark loader
# ============================================================

def load_benchmarks(
    benchmark_dir: Path,
) -> pd.DataFrame:

    files = sorted(
        benchmark_dir.glob(
            "scenario_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No benchmark scenario files found in:\n"
            f"{benchmark_dir}"
        )

    frames = []

    for path in files:

        frame = pd.read_parquet(
            path
        ).copy()

        scenario_id = extract_scenario_id(
            path
        )

        frame["scenario_id"] = (
            scenario_id
        )

        frame["_benchmark_file"] = (
            path.name
        )

        frames.append(
            frame
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = [
        "scenario_id",
        "request_id",
        "walking_time_sec",
        "walking_within_15min",
        "transit_available",
        "transit_time_sec",
        "transit_within_15min",
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            "Benchmark data are missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{result.columns.tolist()}"
        )

    # Ensure unique request inside scenario
    duplicates = result.duplicated(
        subset=[
            "scenario_id",
            "request_id",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Benchmark files contain duplicated "
            "scenario_id + request_id pairs."
        )

    result[
        "walking_within_15min"
    ] = to_bool(
        result[
            "walking_within_15min"
        ]
    )

    result[
        "transit_within_15min"
    ] = to_bool(
        result[
            "transit_within_15min"
        ]
    )

    result[
        "transit_available"
    ] = to_bool(
        result[
            "transit_available"
        ]
    )

    return result


# ============================================================
# Proposed-full DRT loader
# ============================================================

def load_drt_results(
    alns_dir: Path,
) -> pd.DataFrame:

    method_dir = (
        alns_dir
        / METHOD_DIR
    )

    if not method_dir.exists():
        raise FileNotFoundError(
            f"Cannot find proposed-full directory:\n"
            f"{method_dir}"
        )

    files = sorted(
        method_dir.glob(
            "scenario_*/seed_*/request_results.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No proposed-full request_results.parquet "
            f"files found under:\n{method_dir}"
        )

    frames = []

    for path in files:

        frame = pd.read_parquet(
            path
        ).copy()

        scenario_id = extract_scenario_id(
            path
        )

        solver_seed = extract_solver_seed(
            path
        )

        # Validate scenario_id if file already contains it
        if "scenario_id" in frame.columns:

            scenario_values = (
                pd.to_numeric(
                    frame["scenario_id"],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if (
                len(scenario_values) > 0
                and not all(
                    int(value) == scenario_id
                    for value in scenario_values
                )
            ):
                raise ValueError(
                    f"scenario_id mismatch in:\n{path}"
                )

        frame["scenario_id"] = (
            scenario_id
        )

        frame["solver_seed"] = (
            solver_seed
        )

        frame["_request_result_file"] = (
            str(path)
        )

        frames.append(
            frame
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = [
        "scenario_id",
        "solver_seed",
        "request_id",
        "served",
        "door_to_door_sec",
        "within_15min",
        "is_gap",
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            "DRT request results are missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{result.columns.tolist()}"
        )

    result["served"] = to_bool(
        result["served"]
    )

    result["within_15min"] = to_bool(
        result["within_15min"]
    )

    result["is_gap"] = to_bool(
        result["is_gap"]
    )

    # DRT is accessible only if actually served
    result[
        "drt_within_15min"
    ] = (
        result["served"]
        & result["within_15min"]
    )

    # Validate uniqueness within scenario × seed
    duplicates = result.duplicated(
        subset=[
            "scenario_id",
            "solver_seed",
            "request_id",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "DRT results contain duplicated "
            "scenario_id + solver_seed + request_id rows."
        )

    return result


# ============================================================
# Accessibility aggregation
# ============================================================

def calculate_fixed_mode_scenario_rates(
    frame: pd.DataFrame,
    reachable_col: str,
    population: str,
) -> pd.DataFrame:

    data = frame.copy()

    if population == "Gap-origin":
        data = data[
            data["is_gap"]
        ].copy()

    result = (
        data
        .groupby(
            "scenario_id",
            as_index=False,
        )
        .agg(
            total_requests=(
                "request_id",
                "size",
            ),
            accessible_requests=(
                reachable_col,
                "sum",
            ),
        )
    )

    result[
        "accessibility_rate"
    ] = (
        result[
            "accessible_requests"
        ]
        / result[
            "total_requests"
        ]
    )

    return result


def calculate_drt_scenario_rates(
    frame: pd.DataFrame,
    population: str,
) -> pd.DataFrame:

    data = frame.copy()

    if population == "Gap-origin":
        data = data[
            data["is_gap"]
        ].copy()

    # --------------------------------------------------------
    # Step 1:
    # Accessibility rate for each scenario × solver seed
    # --------------------------------------------------------

    seed_level = (
        data
        .groupby(
            [
                "scenario_id",
                "solver_seed",
            ],
            as_index=False,
        )
        .agg(
            total_requests=(
                "request_id",
                "size",
            ),
            accessible_requests=(
                "drt_within_15min",
                "sum",
            ),
        )
    )

    seed_level[
        "accessibility_rate"
    ] = (
        seed_level[
            "accessible_requests"
        ]
        / seed_level[
            "total_requests"
        ]
    )

    # --------------------------------------------------------
    # Step 2:
    # Average repeated solver seeds WITHIN scenario
    # --------------------------------------------------------

    scenario_level = (
        seed_level
        .groupby(
            "scenario_id",
            as_index=False,
        )
        .agg(
            accessibility_rate=(
                "accessibility_rate",
                "mean",
            ),
            mean_accessible_requests=(
                "accessible_requests",
                "mean",
            ),
            mean_total_requests=(
                "total_requests",
                "mean",
            ),
            n_solver_seeds=(
                "solver_seed",
                "nunique",
            ),
        )
    )

    return scenario_level


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 5.4.1 — "
            "Fifteen-minute accessibility by travel mode."
        )
    )

    parser.add_argument(
        "--benchmark-dir",
        default=(
            "outputs/benchmarks_competitive"
        ),
    )

    parser.add_argument(
        "--alns-dir",
        default=(
            "outputs/alns_competitive"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/paper_results/"
            "chapter5_multimodal"
        ),
    )

    args = parser.parse_args()

    benchmark_dir = Path(
        args.benchmark_dir
    )

    alns_dir = Path(
        args.alns_dir
    )

    output_dir = Path(
        args.output_dir
    )

    # ========================================================
    # 1. Load data
    # ========================================================

    benchmark = load_benchmarks(
        benchmark_dir
    )

    drt = load_drt_results(
        alns_dir
    )

    # ========================================================
    # 2. Recover request-level gap label from Proposed-full
    #
    # Benchmark files do not contain is_gap.
    # A request's gap status does not depend on solver_seed,
    # so take one unique label per scenario + request.
    # ========================================================

    gap_labels = (
        drt[
            [
                "scenario_id",
                "request_id",
                "is_gap",
            ]
        ]
        .drop_duplicates()
    )

    gap_consistency = (
        gap_labels
        .groupby(
            [
                "scenario_id",
                "request_id",
            ]
        )[
            "is_gap"
        ]
        .nunique()
    )

    if (
        gap_consistency
        .gt(1)
        .any()
    ):
        raise ValueError(
            "is_gap is inconsistent across solver seeds "
            "for the same request."
        )

    gap_labels = (
        gap_labels
        .drop_duplicates(
            subset=[
                "scenario_id",
                "request_id",
            ]
        )
    )

    benchmark = benchmark.merge(
        gap_labels,
        on=[
            "scenario_id",
            "request_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    # ========================================================
    # 3. Validate request matching
    # ========================================================

    expected_benchmark_rows = len(
        load_benchmarks(
            benchmark_dir
        )
    )

    matched_rows = len(
        benchmark
    )

    if matched_rows != expected_benchmark_rows:
        print(
            "WARNING: not every benchmark request matched "
            "a proposed-full request."
        )

        print(
            f"Benchmark rows: {expected_benchmark_rows:,}"
        )

        print(
            f"Matched rows  : {matched_rows:,}"
        )

    # ========================================================
    # 4. Build Table 5.4.1
    # ========================================================

    rows = []

    for population in POPULATIONS:

        # ----------------------------------------------------
        # Walking
        # ----------------------------------------------------

        walking_scenario = (
            calculate_fixed_mode_scenario_rates(
                benchmark,
                reachable_col=(
                    "walking_within_15min"
                ),
                population=population,
            )
        )

        rows.append(
            {
                "Population": population,
                "Mode": "Walking",
                "Scenarios": int(
                    walking_scenario[
                        "scenario_id"
                    ].nunique()
                ),
                "Mean accessible requests":
                    float(
                        walking_scenario[
                            "accessible_requests"
                        ].mean()
                    ),
                "Mean total requests":
                    float(
                        walking_scenario[
                            "total_requests"
                        ].mean()
                    ),
                "Accessibility rate":
                    float(
                        walking_scenario[
                            "accessibility_rate"
                        ].mean()
                    ),
            }
        )

        # ----------------------------------------------------
        # Public transit
        # ----------------------------------------------------

        transit_scenario = (
            calculate_fixed_mode_scenario_rates(
                benchmark,
                reachable_col=(
                    "transit_within_15min"
                ),
                population=population,
            )
        )

        rows.append(
            {
                "Population": population,
                "Mode": "Public transit",
                "Scenarios": int(
                    transit_scenario[
                        "scenario_id"
                    ].nunique()
                ),
                "Mean accessible requests":
                    float(
                        transit_scenario[
                            "accessible_requests"
                        ].mean()
                    ),
                "Mean total requests":
                    float(
                        transit_scenario[
                            "total_requests"
                        ].mean()
                    ),
                "Accessibility rate":
                    float(
                        transit_scenario[
                            "accessibility_rate"
                        ].mean()
                    ),
            }
        )

        # ----------------------------------------------------
        # DRT
        # ----------------------------------------------------

        drt_scenario = (
            calculate_drt_scenario_rates(
                drt,
                population=population,
            )
        )

        rows.append(
            {
                "Population": population,
                "Mode": "DRT",
                "Scenarios": int(
                    drt_scenario[
                        "scenario_id"
                    ].nunique()
                ),
                "Mean accessible requests":
                    float(
                        drt_scenario[
                            "mean_accessible_requests"
                        ].mean()
                    ),
                "Mean total requests":
                    float(
                        drt_scenario[
                            "mean_total_requests"
                        ].mean()
                    ),
                "Accessibility rate":
                    float(
                        drt_scenario[
                            "accessibility_rate"
                        ].mean()
                    ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    # ========================================================
    # 5. Paper order
    # ========================================================

    table[
        "Population"
    ] = pd.Categorical(
        table[
            "Population"
        ],
        categories=POPULATIONS,
        ordered=True,
    )

    table[
        "Mode"
    ] = pd.Categorical(
        table[
            "Mode"
        ],
        categories=MODES,
        ordered=True,
    )

    table = (
        table
        .sort_values(
            [
                "Population",
                "Mode",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 6. Publication formatting
    # ========================================================

    table[
        "Mean accessible requests"
    ] = table[
        "Mean accessible requests"
    ].round(2)

    table[
        "Mean total requests"
    ] = table[
        "Mean total requests"
    ].round(2)

    table[
        "Accessibility rate"
    ] = table[
        "Accessibility rate"
    ].round(4)

    # ========================================================
    # 7. Save publication table
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        output_dir
        / "Table_5_4_1_fifteen_minute_accessibility_by_mode.csv"
    )

    table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. Also save merged request-level source data
    #
    # This is useful for later Figure 5.4.x,
    # McNemar tests, conversion analysis, etc.
    # ========================================================

    request_level_path = (
        output_dir
        / "multimodal_request_level_base.csv"
    )

    benchmark[
        [
            "scenario_id",
            "request_id",
            "is_gap",
            "walking_time_sec",
            "walking_distance_m",
            "walking_within_15min",
            "transit_available",
            "transit_time_sec",
            "transit_within_15min",
            "transit_transfers",
        ]
    ].to_csv(
        request_level_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 9. Save scenario-level plotting/statistical data
    # ========================================================

    scenario_rows = []

    for population in POPULATIONS:

        walking = (
            calculate_fixed_mode_scenario_rates(
                benchmark,
                "walking_within_15min",
                population,
            )
        )

        for _, row in walking.iterrows():

            scenario_rows.append(
                {
                    "population": population,
                    "mode": "Walking",
                    "scenario_id":
                        int(
                            row[
                                "scenario_id"
                            ]
                        ),
                    "accessibility_rate":
                        float(
                            row[
                                "accessibility_rate"
                            ]
                        ),
                }
            )

        transit = (
            calculate_fixed_mode_scenario_rates(
                benchmark,
                "transit_within_15min",
                population,
            )
        )

        for _, row in transit.iterrows():

            scenario_rows.append(
                {
                    "population": population,
                    "mode":
                        "Public transit",
                    "scenario_id":
                        int(
                            row[
                                "scenario_id"
                            ]
                        ),
                    "accessibility_rate":
                        float(
                            row[
                                "accessibility_rate"
                            ]
                        ),
                }
            )

        drt_mode = (
            calculate_drt_scenario_rates(
                drt,
                population,
            )
        )

        for _, row in drt_mode.iterrows():

            scenario_rows.append(
                {
                    "population": population,
                    "mode": "DRT",
                    "scenario_id":
                        int(
                            row[
                                "scenario_id"
                            ]
                        ),
                    "accessibility_rate":
                        float(
                            row[
                                "accessibility_rate"
                            ]
                        ),
                }
            )

    scenario_output = pd.DataFrame(
        scenario_rows
    )

    scenario_path = (
        output_dir
        / "Table_5_4_1_scenario_level_accessibility.csv"
    )

    scenario_output.to_csv(
        scenario_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. Console audit
    # ========================================================

    print()
    print("=" * 90)

    print(
        "Table 5.4.1 — "
        "Fifteen-minute accessibility by travel mode"
    )

    print("=" * 90)

    print(
        f"Benchmark directory : {benchmark_dir}"
    )

    print(
        f"DRT directory       : "
        f"{alns_dir / METHOD_DIR}"
    )

    print(
        f"Matched requests    : "
        f"{matched_rows:,}"
    )

    print(
        f"DRT result rows     : "
        f"{len(drt):,}"
    )

    print(
        f"Scenarios           : "
        f"{benchmark['scenario_id'].nunique()}"
    )

    print(
        f"Solver seeds        : "
        f"{drt['solver_seed'].nunique()}"
    )

    print()

    print(
        table.to_string(
            index=False
        )
    )

    print()

    print(
        f"Table saved         : {table_path}"
    )

    print(
        f"Request-level base  : {request_level_path}"
    )

    print(
        f"Scenario-level data : {scenario_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()