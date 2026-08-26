from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import wilcoxon
except ImportError as exc:
    raise SystemExit(
        "scipy is required. Install it with: pip install scipy"
    ) from exc


SCRIPT_VERSION = "1.0.0"

DEFAULT_GROUP_COLUMNS = [
    "is_gap",
    "is_low_income",
    "is_no_car",
    "is_older",
    "is_80plus",
    "is_disabled",
    "is_student",
    "sector_group",
    "gender_group",
    "age_group",
    "income_group",
    "vehicle_group",
]

ALIASES: dict[str, list[str]] = {
    "request_id": [
        "request_id", "req_id", "request", "trip_id", "id",
        "request_index", "request_idx",
    ],
    "scenario_id": [
        "scenario_id", "scenario", "scenario_index", "scenario_idx",
    ],
    "seed": [
        "seed", "solver_seed", "random_seed", "run_seed",
    ],
    "replicate_index": [
        "replicate_index", "replicate", "rep", "solver_rep",
    ],
    "method_name": [
        "method_name", "method", "experiment_name", "algorithm",
    ],
    "served": [
        "served", "is_served", "accepted", "assigned",
    ],
    "within_15min": [
        "within_15min", "within_15_min", "served_within_15min",
        "door_to_door_within_15min", "within_threshold",
    ],
    "drt_door_sec": [
        "door_to_door_sec", "door_time_sec", "door_to_door_time_sec",
        "total_door_time_sec", "total_travel_time_sec",
        "passenger_total_time_sec", "drt_door_to_door_sec",
        "access_time_sec",
    ],
    "walking_sec": [
        "walking_time_sec", "walk_time_sec", "walking_duration_sec",
        "walk_duration_sec", "pedestrian_time_sec", "walking_total_sec",
        "walk_total_sec", "benchmark_walk_sec", "walking_sec",
        "walk_time", "walking_time",
    ],
    "transit_sec": [
        "transit_time_sec", "public_transit_time_sec",
        "public_transport_time_sec", "otp_total_time_sec",
        "otp_time_sec", "transit_duration_sec", "pt_time_sec",
        "transit_total_sec", "transit_door_to_door_sec",
        "public_transit_duration_sec", "transit_time",
    ],
    "transit_available": [
        "transit_available", "public_transit_available",
        "public_transport_available", "otp_available", "otp_found",
        "transit_feasible", "has_transit", "itinerary_found",
        "transit_itinerary_found",
    ],
    "direct_drive_sec": [
        "direct_drive_sec", "direct_driving_time_sec", "drive_time_sec",
        "driving_time_sec", "car_time_sec", "direct_time_sec",
    ],
}


@dataclass
class ColumnMapping:
    request_id: str
    walking_sec: str
    transit_sec: str
    transit_available: str | None
    direct_drive_sec: str | None
    served: str
    within_15min: str | None
    drt_door_sec: str | None


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(
    frame: pd.DataFrame,
    key: str,
    explicit: str | None = None,
    required: bool = True,
) -> str | None:
    normalized_to_actual = {normalize_name(c): str(c) for c in frame.columns}

    if explicit:
        normalized = normalize_name(explicit)
        if normalized in normalized_to_actual:
            return normalized_to_actual[normalized]
        if required:
            raise ValueError(
                f"Requested column '{explicit}' was not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return None

    for alias in ALIASES[key]:
        normalized = normalize_name(alias)
        if normalized in normalized_to_actual:
            return normalized_to_actual[normalized]

    if required:
        raise ValueError(
            f"Could not detect column for '{key}'. "
            f"Available columns: {list(frame.columns)}. "
            f"Use the relevant --*-column option."
        )
    return None


def parse_scenario_from_path(path: Path) -> int | None:
    text = str(path)
    patterns = [
        r"scenario[_-]?0*(\d+)",
        r"scenario=(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_seed_from_path(path: Path) -> int | None:
    text = str(path)
    patterns = [
        r"seed[_=-]?(\d+)",
        r"rep_\d+_seed_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_replicate_from_path(path: Path) -> int | None:
    match = re.search(r"rep[_=-]?0*(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def canonical_request_id(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        if re.fullmatch(r"-?\d+\.0+", text):
            return text.split(".", 1)[0]
        return text

    return series.map(convert)


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)

    true_values = {
        "1", "true", "t", "yes", "y", "served", "available", "feasible",
    }
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(true_values)
    )


def to_seconds(series: pd.Series, column_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    normalized = normalize_name(column_name)

    minute_markers = (
        normalized.endswith("_min")
        or normalized.endswith("_mins")
        or normalized.endswith("_minute")
        or normalized.endswith("_minutes")
        or "_time_min" in normalized
        or "_duration_min" in normalized
    )
    hour_markers = (
        normalized.endswith("_hour")
        or normalized.endswith("_hours")
        or normalized.endswith("_hr")
    )

    if minute_markers:
        values = values * 60.0
    elif hour_markers:
        values = values * 3600.0
    return values.astype(float)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def finite_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else float("nan")


def finite_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if len(numeric) else float("nan")


def finite_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else float("nan")


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_indices = np.where(np.isfinite(values))[0]
    if not len(valid_indices):
        return adjusted.tolist()

    ordered = valid_indices[np.argsort(values[valid_indices])]
    m = len(ordered)
    running_max = 0.0
    for rank, index in enumerate(ordered):
        candidate = (m - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def exact_mcnemar(b: int, c: int) -> dict[str, float | int]:
    discordant = b + c
    if discordant == 0:
        return {
            "b_drt_only": b,
            "c_benchmark_only": c,
            "discordant": 0,
            "exact_two_sided_p": 1.0,
            "odds_ratio_b_over_c": float("nan"),
        }

    k = min(b, c)
    tail = sum(math.comb(discordant, i) for i in range(k + 1))
    p_value = min(1.0, 2.0 * tail / (2.0 ** discordant))
    odds_ratio = float("inf") if c == 0 and b > 0 else safe_rate(b, c)
    return {
        "b_drt_only": b,
        "c_benchmark_only": c,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
        "odds_ratio_b_over_c": odds_ratio,
    }


def bootstrap_ci(
    values: pd.Series,
    repetitions: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0

    estimate = float(np.mean(array))
    if n == 1:
        return estimate, estimate, estimate, 1

    sampled_means = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sampled = rng.choice(array, size=n, replace=True)
        sampled_means[i] = sampled.mean()

    lower = float(np.quantile(sampled_means, alpha / 2))
    upper = float(np.quantile(sampled_means, 1 - alpha / 2))
    return estimate, lower, upper, n


def load_benchmarks(
    benchmark_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, ColumnMapping, dict[str, Any]]:
    files = sorted(benchmark_dir.glob("scenario_*.parquet"))
    if not files:
        files = sorted(benchmark_dir.glob("scenario_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No scenario_*.parquet or scenario_*.csv files found in "
            f"{benchmark_dir}"
        )

    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for path in files:
        frame = read_table(path)
        scenario_id = parse_scenario_from_path(path)
        if scenario_id is None:
            raise ValueError(f"Could not parse scenario id from {path}")
        frame = frame.copy()
        frame["_scenario_from_file"] = scenario_id
        frame["_benchmark_source"] = str(path)
        frames.append(frame)
        source_rows.append(
            {
                "scenario_id": scenario_id,
                "path": str(path),
                "rows": len(frame),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)

    request_id = find_column(
        combined, "request_id", args.request_id_column, required=False
    )
    if request_id is None:
        warnings.warn(
            "Benchmark request ID was not found. Falling back to row order "
            "within each scenario. Verify that DRT result rows use the same "
            "order."
        )
        combined["_request_row_index"] = combined.groupby(
            "_scenario_from_file"
        ).cumcount()
        request_id = "_request_row_index"

    walking_sec = find_column(
        combined, "walking_sec", args.walking_column, required=True
    )
    transit_sec = find_column(
        combined, "transit_sec", args.transit_column, required=True
    )
    transit_available = find_column(
        combined,
        "transit_available",
        args.transit_available_column,
        required=False,
    )
    direct_drive_sec = find_column(
        combined,
        "direct_drive_sec",
        args.direct_drive_column,
        required=False,
    )

    mapping = ColumnMapping(
        request_id=request_id,
        walking_sec=walking_sec,
        transit_sec=transit_sec,
        transit_available=transit_available,
        direct_drive_sec=direct_drive_sec,
        served="",
        within_15min=None,
        drt_door_sec=None,
    )

    benchmark = pd.DataFrame(
        {
            "scenario_id": combined["_scenario_from_file"].astype(int),
            "request_id": canonical_request_id(combined[request_id]),
            "walking_sec": to_seconds(combined[walking_sec], walking_sec),
            "transit_sec": to_seconds(combined[transit_sec], transit_sec),
        }
    )

    if transit_available is not None:
        benchmark["transit_available"] = to_bool(combined[transit_available])
        benchmark["transit_available"] &= benchmark["transit_sec"].notna()
    else:
        benchmark["transit_available"] = benchmark["transit_sec"].notna()

    if direct_drive_sec is not None:
        benchmark["direct_drive_sec"] = to_seconds(
            combined[direct_drive_sec], direct_drive_sec
        )

    reserved = {
        request_id,
        walking_sec,
        transit_sec,
        transit_available,
        direct_drive_sec,
        "_scenario_from_file",
        "_benchmark_source",
    }
    for column in combined.columns:
        if column not in reserved and (
            column in DEFAULT_GROUP_COLUMNS
            or normalize_name(column) in {normalize_name(c) for c in DEFAULT_GROUP_COLUMNS}
        ):
            benchmark[column] = combined[column].values

    duplicate_keys = benchmark.duplicated(
        ["scenario_id", "request_id"], keep=False
    )
    if duplicate_keys.any():
        sample = benchmark.loc[
            duplicate_keys, ["scenario_id", "request_id"]
        ].head(10)
        raise ValueError(
            "Benchmark contains duplicate scenario/request keys. Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    quality = {
        "benchmark_files": len(files),
        "benchmark_rows": len(benchmark),
        "benchmark_scenarios": int(benchmark["scenario_id"].nunique()),
        "detected_columns": {
            "request_id": request_id,
            "walking_sec": walking_sec,
            "transit_sec": transit_sec,
            "transit_available": transit_available,
            "direct_drive_sec": direct_drive_sec,
        },
        "source_files": source_rows,
    }
    return benchmark, mapping, quality


def find_status_csv(alns_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Status CSV not found: {path}")
        return path.resolve()

    candidates = list(
        alns_dir.glob("_comparative_progress/**/run_status.csv")
    )
    if not candidates:
        return None

    formal = [p for p in candidates if "formal" in str(p).lower()]
    pool = formal or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def load_valid_runs(
    status_path: Path | None,
    method_name: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if status_path is None:
        return None, {
            "status_csv": None,
            "status_filter_applied": False,
            "warning": (
                "No run_status.csv found. Old result files may be included."
            ),
        }

    status = pd.read_csv(status_path)
    required = {"method_name", "scenario_id", "seed", "status"}
    missing = required - set(status.columns)
    if missing:
        raise ValueError(
            f"run_status.csv is missing columns: {sorted(missing)}"
        )

    status = status[
        status["method_name"].astype(str).eq(method_name)
        & status["status"].astype(str).str.lower().isin(
            ["completed", "skipped"]
        )
    ].copy()

    if "replicate_index" not in status.columns:
        status["replicate_index"] = (
            status.sort_values(["scenario_id", "seed"])
            .groupby("scenario_id")
            .cumcount()
        )

    status["scenario_id"] = pd.to_numeric(
        status["scenario_id"], errors="raise"
    ).astype(int)
    status["seed"] = pd.to_numeric(
        status["seed"], errors="raise"
    ).astype(int)
    status["replicate_index"] = pd.to_numeric(
        status["replicate_index"], errors="coerce"
    ).fillna(0).astype(int)

    status = status[
        [
            "method_name",
            "scenario_id",
            "replicate_index",
            "seed",
            "status",
        ]
    ].drop_duplicates()

    return status, {
        "status_csv": str(status_path),
        "status_filter_applied": True,
        "valid_status_rows": len(status),
    }


def load_drt_results(
    alns_dir: Path,
    method_name: str,
    valid_runs: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, ColumnMapping, dict[str, Any]]:
    candidates = sorted(alns_dir.rglob("request_results.parquet"))
    candidates += sorted(alns_dir.rglob("request_results.csv"))

    candidates = [
        path
        for path in candidates
        if "_comparative_progress" not in str(path)
        and method_name.lower() in str(path).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No request_results.parquet/csv containing method name "
            f"'{method_name}' found under {alns_dir}"
        )

    loaded: list[pd.DataFrame] = []
    file_records: list[dict[str, Any]] = []

    for path in candidates:
        frame = read_table(path)
        if frame.empty:
            continue

        method_col = find_column(
            frame, "method_name", required=False
        )
        if method_col is not None:
            frame = frame[
                frame[method_col].astype(str).eq(method_name)
            ].copy()
            if frame.empty:
                continue

        scenario_col = find_column(
            frame, "scenario_id", required=False
        )
        seed_col = find_column(frame, "seed", required=False)
        replicate_col = find_column(
            frame, "replicate_index", required=False
        )

        scenario_path = parse_scenario_from_path(path)
        seed_path = parse_seed_from_path(path)
        replicate_path = parse_replicate_from_path(path)

        frame = frame.copy()
        if scenario_col is not None:
            frame["_scenario_id"] = pd.to_numeric(
                frame[scenario_col], errors="coerce"
            )
        else:
            frame["_scenario_id"] = scenario_path

        if seed_col is not None:
            frame["_seed"] = pd.to_numeric(
                frame[seed_col], errors="coerce"
            )
        else:
            frame["_seed"] = seed_path

        if replicate_col is not None:
            frame["_replicate_index"] = pd.to_numeric(
                frame[replicate_col], errors="coerce"
            )
        else:
            frame["_replicate_index"] = replicate_path

        if frame["_scenario_id"].isna().all():
            warnings.warn(f"Skipping file with unknown scenario: {path}")
            continue
        if frame["_seed"].isna().all():
            warnings.warn(f"Skipping file with unknown seed: {path}")
            continue

        frame["_scenario_id"] = frame["_scenario_id"].astype(int)
        frame["_seed"] = frame["_seed"].astype(int)
        frame["_source_file"] = str(path)
        frame["_source_mtime"] = path.stat().st_mtime
        loaded.append(frame)

        file_records.append(
            {
                "path": str(path),
                "rows": len(frame),
                "scenario_id": int(frame["_scenario_id"].iloc[0]),
                "seed": int(frame["_seed"].iloc[0]),
            }
        )

    if not loaded:
        raise ValueError("No readable DRT request result files remained.")

    combined = pd.concat(loaded, ignore_index=True, sort=False)

    if valid_runs is not None:
        valid_key = valid_runs[
            ["scenario_id", "seed", "replicate_index"]
        ].copy()
        valid_key = valid_key.rename(
            columns={
                "scenario_id": "_scenario_id",
                "seed": "_seed",
                "replicate_index": "_valid_replicate_index",
            }
        )
        combined = combined.merge(
            valid_key,
            on=["_scenario_id", "_seed"],
            how="inner",
        )
        combined["_replicate_index"] = combined[
            "_replicate_index"
        ].fillna(combined["_valid_replicate_index"])
        combined = combined.drop(columns=["_valid_replicate_index"])

    if combined.empty:
        raise ValueError(
            "DRT files were found, but none matched the completed/skipped "
            "formal run tuples in run_status.csv."
        )

    request_id = find_column(
        combined, "request_id", args.request_id_column, required=False
    )
    if request_id is None:
        warnings.warn(
            "DRT request ID was not found. Falling back to row order within "
            "each scenario and seed."
        )
        combined["_request_row_index"] = combined.groupby(
            ["_scenario_id", "_seed"]
        ).cumcount()
        request_id = "_request_row_index"

    served = find_column(
        combined, "served", args.served_column, required=True
    )
    within = find_column(
        combined,
        "within_15min",
        args.within_column,
        required=False,
    )
    door = find_column(
        combined,
        "drt_door_sec",
        args.drt_door_column,
        required=False,
    )

    mapping = ColumnMapping(
        request_id=request_id,
        walking_sec="",
        transit_sec="",
        transit_available=None,
        direct_drive_sec=None,
        served=served,
        within_15min=within,
        drt_door_sec=door,
    )

    drt = pd.DataFrame(
        {
            "scenario_id": combined["_scenario_id"].astype(int),
            "seed": combined["_seed"].astype(int),
            "replicate_index": pd.to_numeric(
                combined["_replicate_index"], errors="coerce"
            ).fillna(0).astype(int),
            "request_id": canonical_request_id(combined[request_id]),
            "served": to_bool(combined[served]),
            "_source_file": combined["_source_file"].astype(str),
            "_source_mtime": combined["_source_mtime"].astype(float),
        }
    )

    if door is not None:
        drt["drt_door_sec"] = to_seconds(combined[door], door)
    else:
        drt["drt_door_sec"] = np.nan

    if within is not None:
        drt["drt_within_threshold_raw"] = to_bool(combined[within])
    else:
        drt["drt_within_threshold_raw"] = False

    reserved_normalized = {
        normalize_name(request_id),
        normalize_name(served),
        normalize_name(within or ""),
        normalize_name(door or ""),
    }

    for requested_group in args.group_columns:
        detected = None
        for column in combined.columns:
            if normalize_name(column) == normalize_name(requested_group):
                detected = column
                break
        if detected is not None and normalize_name(detected) not in reserved_normalized:
            drt[requested_group] = combined[detected].values

    # If duplicate copies of the same formal run exist, retain the newest file.
    drt = drt.sort_values("_source_mtime")
    duplicate_run_request = drt.duplicated(
        ["scenario_id", "seed", "request_id"], keep="last"
    )
    duplicate_count = int(duplicate_run_request.sum())
    drt = drt.loc[~duplicate_run_request].copy()

    quality = {
        "candidate_result_files": len(candidates),
        "loaded_result_files": len(file_records),
        "drt_rows_after_filtering": len(drt),
        "drt_scenarios": int(drt["scenario_id"].nunique()),
        "drt_runs": int(
            drt[["scenario_id", "seed"]].drop_duplicates().shape[0]
        ),
        "duplicate_run_request_rows_removed": duplicate_count,
        "detected_columns": {
            "request_id": request_id,
            "served": served,
            "within_threshold": within,
            "drt_door_sec": door,
        },
        "source_files": file_records,
    }
    return drt, mapping, quality


def merge_run_with_benchmark(
    benchmark: pd.DataFrame,
    drt: pd.DataFrame,
    threshold_sec: float,
    served_implies_threshold: bool,
    group_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmark_keys = set(
        zip(benchmark["scenario_id"], benchmark["request_id"])
    )
    drt_keys = set(zip(drt["scenario_id"], drt["request_id"]))

    missing_benchmark_keys = len(drt_keys - benchmark_keys)
    missing_drt_keys = len(benchmark_keys - drt_keys)

    merged_frames: list[pd.DataFrame] = []
    for (scenario_id, seed, replicate), run in drt.groupby(
        ["scenario_id", "seed", "replicate_index"],
        sort=True,
    ):
        scenario_benchmark = benchmark[
            benchmark["scenario_id"].eq(scenario_id)
        ].copy()
        if scenario_benchmark.empty:
            warnings.warn(
                f"No benchmark rows for scenario {scenario_id}; skipping run."
            )
            continue

        run_columns = [
            "scenario_id",
            "request_id",
            "served",
            "drt_door_sec",
            "drt_within_threshold_raw",
        ] + [c for c in group_columns if c in run.columns]

        run_unique = run[run_columns].drop_duplicates(
            ["scenario_id", "request_id"], keep="last"
        )

        merged = scenario_benchmark.merge(
            run_unique,
            on=["scenario_id", "request_id"],
            how="left",
            suffixes=("_benchmark", "_drt"),
            validate="one_to_one",
        )
        merged["seed"] = int(seed)
        merged["replicate_index"] = int(replicate)
        merged["served"] = merged["served"].fillna(False).astype(bool)

        raw_within = merged[
            "drt_within_threshold_raw"
        ].fillna(False).astype(bool)

        if merged["drt_door_sec"].notna().any():
            derived_within = (
                merged["served"]
                & merged["drt_door_sec"].le(threshold_sec)
            )
            merged["drt_within_threshold"] = raw_within | derived_within
        elif served_implies_threshold:
            merged["drt_within_threshold"] = raw_within | merged["served"]
        else:
            merged["drt_within_threshold"] = raw_within

        merged["walk_within_threshold"] = (
            merged["walking_sec"].notna()
            & merged["walking_sec"].le(threshold_sec)
        )
        merged["transit_within_threshold"] = (
            merged["transit_available"].fillna(False).astype(bool)
            & merged["transit_sec"].notna()
            & merged["transit_sec"].le(threshold_sec)
        )

        merged["walk_unreachable"] = ~merged["walk_within_threshold"]
        merged["transit_unreachable"] = ~merged["transit_within_threshold"]

        merged["walk_closure"] = (
            merged["drt_within_threshold"] & merged["walk_unreachable"]
        )
        merged["transit_closure"] = (
            merged["drt_within_threshold"] & merged["transit_unreachable"]
        )
        merged["walk_access_loss"] = (
            ~merged["drt_within_threshold"]
            & merged["walk_within_threshold"]
        )
        merged["transit_access_loss"] = (
            ~merged["drt_within_threshold"]
            & merged["transit_within_threshold"]
        )

        merged["walk_time_saving_sec"] = np.where(
            merged["served"]
            & merged["drt_door_sec"].notna()
            & merged["walking_sec"].notna(),
            merged["walking_sec"] - merged["drt_door_sec"],
            np.nan,
        )
        merged["transit_time_saving_sec"] = np.where(
            merged["served"]
            & merged["drt_door_sec"].notna()
            & merged["transit_available"].fillna(False).astype(bool)
            & merged["transit_sec"].notna(),
            merged["transit_sec"] - merged["drt_door_sec"],
            np.nan,
        )
        merged["drt_faster_than_walk"] = (
            merged["walk_time_saving_sec"].gt(0)
        )
        merged["drt_faster_than_transit"] = (
            merged["transit_time_saving_sec"].gt(0)
        )

        # Coalesce group fields when both benchmark and DRT carry them.
        for group in group_columns:
            benchmark_name = f"{group}_benchmark"
            drt_name = f"{group}_drt"
            if benchmark_name in merged.columns and drt_name in merged.columns:
                merged[group] = merged[drt_name].combine_first(
                    merged[benchmark_name]
                )
            elif benchmark_name in merged.columns:
                merged[group] = merged[benchmark_name]
            elif drt_name in merged.columns:
                merged[group] = merged[drt_name]

        merged_frames.append(merged)

    if not merged_frames:
        raise ValueError("No DRT run could be merged with benchmark data.")

    comparison = pd.concat(merged_frames, ignore_index=True, sort=False)
    quality = {
        "drt_keys_missing_from_benchmark": missing_benchmark_keys,
        "benchmark_keys_missing_from_all_drt_results": missing_drt_keys,
        "comparison_rows": len(comparison),
        "comparison_runs": int(
            comparison[["scenario_id", "seed"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    return comparison, quality


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    n = len(frame)
    drt_access = frame["drt_within_threshold"].astype(bool)
    walk_access = frame["walk_within_threshold"].astype(bool)
    transit_access = frame["transit_within_threshold"].astype(bool)
    transit_available = frame["transit_available"].fillna(False).astype(bool)

    walk_unreachable_n = int((~walk_access).sum())
    transit_unreachable_n = int((~transit_access).sum())
    walk_reachable_n = int(walk_access.sum())
    transit_reachable_n = int(transit_access.sum())

    valid_walk_saving = frame["walk_time_saving_sec"].notna()
    valid_transit_saving = frame["transit_time_saving_sec"].notna()

    return {
        "n_requests": n,
        "served_count": int(frame["served"].sum()),
        "drt_within_threshold_count": int(drt_access.sum()),
        "drt_within_threshold_rate": safe_rate(drt_access.sum(), n),
        "walking_within_threshold_count": int(walk_access.sum()),
        "walking_within_threshold_rate": safe_rate(walk_access.sum(), n),
        "transit_available_count": int(transit_available.sum()),
        "transit_available_rate": safe_rate(transit_available.sum(), n),
        "transit_within_threshold_count": int(transit_access.sum()),
        "transit_within_threshold_rate": safe_rate(transit_access.sum(), n),
        "drt_minus_walking_access_rate": (
            safe_rate(drt_access.sum(), n) - safe_rate(walk_access.sum(), n)
        ),
        "drt_minus_transit_access_rate": (
            safe_rate(drt_access.sum(), n) - safe_rate(transit_access.sum(), n)
        ),
        "walk_unreachable_count": walk_unreachable_n,
        "walk_closure_count": int(frame["walk_closure"].sum()),
        "walk_closure_rate_among_unreachable": safe_rate(
            frame["walk_closure"].sum(), walk_unreachable_n
        ),
        "walk_access_loss_count": int(frame["walk_access_loss"].sum()),
        "walk_access_loss_rate_among_reachable": safe_rate(
            frame["walk_access_loss"].sum(), walk_reachable_n
        ),
        "walk_net_access_gain_rate_all": safe_rate(
            frame["walk_closure"].sum() - frame["walk_access_loss"].sum(),
            n,
        ),
        "transit_unreachable_count": transit_unreachable_n,
        "transit_closure_count": int(frame["transit_closure"].sum()),
        "transit_closure_rate_among_unreachable": safe_rate(
            frame["transit_closure"].sum(), transit_unreachable_n
        ),
        "transit_access_loss_count": int(frame["transit_access_loss"].sum()),
        "transit_access_loss_rate_among_reachable": safe_rate(
            frame["transit_access_loss"].sum(), transit_reachable_n
        ),
        "transit_net_access_gain_rate_all": safe_rate(
            frame["transit_closure"].sum()
            - frame["transit_access_loss"].sum(),
            n,
        ),
        "mean_drt_door_sec_among_served": finite_mean(
            frame.loc[frame["served"], "drt_door_sec"]
        ),
        "median_drt_door_sec_among_served": finite_median(
            frame.loc[frame["served"], "drt_door_sec"]
        ),
        "mean_walking_sec": finite_mean(frame["walking_sec"]),
        "mean_transit_sec_among_available": finite_mean(
            frame.loc[transit_available, "transit_sec"]
        ),
        "mean_walk_time_saving_sec": finite_mean(
            frame["walk_time_saving_sec"]
        ),
        "median_walk_time_saving_sec": finite_median(
            frame["walk_time_saving_sec"]
        ),
        "drt_faster_than_walk_rate_among_comparable": safe_rate(
            frame.loc[valid_walk_saving, "drt_faster_than_walk"].sum(),
            valid_walk_saving.sum(),
        ),
        "mean_transit_time_saving_sec": finite_mean(
            frame["transit_time_saving_sec"]
        ),
        "median_transit_time_saving_sec": finite_median(
            frame["transit_time_saving_sec"]
        ),
        "drt_faster_than_transit_rate_among_comparable": safe_rate(
            frame.loc[
                valid_transit_saving, "drt_faster_than_transit"
            ].sum(),
            valid_transit_saving.sum(),
        ),
    }


def build_run_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario_id, seed, replicate), frame in comparison.groupby(
        ["scenario_id", "seed", "replicate_index"], sort=True
    ):
        row = {
            "scenario_id": int(scenario_id),
            "seed": int(seed),
            "replicate_index": int(replicate),
        }
        row.update(calculate_metrics(frame))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["scenario_id", "replicate_index", "seed"]
    )


def build_scenario_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"scenario_id", "seed", "replicate_index"}
    metric_columns = [
        column
        for column in run_metrics.columns
        if column not in id_columns
    ]

    scenario = (
        run_metrics.groupby("scenario_id", as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    counts = (
        run_metrics.groupby("scenario_id")
        .size()
        .rename("n_solver_runs")
        .reset_index()
    )
    return scenario.merge(counts, on="scenario_id", how="left")


def build_request_consensus(
    comparison: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    static_columns = [
        "walking_sec",
        "transit_sec",
        "transit_available",
        "walk_within_threshold",
        "transit_within_threshold",
    ]
    if "direct_drive_sec" in comparison.columns:
        static_columns.append("direct_drive_sec")
    static_columns += [
        column for column in group_columns if column in comparison.columns
    ]

    aggregation: dict[str, str] = {
        column: "first" for column in static_columns
    }
    aggregation.update(
        {
            "served": "mean",
            "drt_within_threshold": "mean",
            "drt_door_sec": "mean",
            "walk_closure": "mean",
            "transit_closure": "mean",
            "walk_access_loss": "mean",
            "transit_access_loss": "mean",
            "walk_time_saving_sec": "mean",
            "transit_time_saving_sec": "mean",
        }
    )

    consensus = (
        comparison.groupby(
            ["scenario_id", "request_id"], as_index=False
        )
        .agg(aggregation)
        .rename(
            columns={
                "served": "drt_service_probability",
                "drt_within_threshold": "drt_access_probability",
                "drt_door_sec": "mean_drt_door_sec_when_observed",
                "walk_closure": "walk_closure_probability",
                "transit_closure": "transit_closure_probability",
                "walk_access_loss": "walk_access_loss_probability",
                "transit_access_loss": "transit_access_loss_probability",
                "walk_time_saving_sec": "mean_walk_time_saving_sec",
                "transit_time_saving_sec": "mean_transit_time_saving_sec",
            }
        )
    )
    return consensus


def build_group_metrics(
    request_consensus: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group in group_columns:
        if group not in request_consensus.columns:
            continue

        values = request_consensus[group]
        non_missing = values.notna()
        if not non_missing.any():
            continue

        for group_value, frame in request_consensus[non_missing].groupby(
            group, dropna=False
        ):
            n = len(frame)
            walk_unreachable = ~frame[
                "walk_within_threshold"
            ].astype(bool)
            transit_unreachable = ~frame[
                "transit_within_threshold"
            ].astype(bool)

            rows.append(
                {
                    "group_variable": group,
                    "group_value": group_value,
                    "n_requests": n,
                    "n_scenarios": int(frame["scenario_id"].nunique()),
                    "drt_expected_access_rate": finite_mean(
                        frame["drt_access_probability"]
                    ),
                    "walking_access_rate": safe_rate(
                        frame["walk_within_threshold"].sum(), n
                    ),
                    "transit_access_rate": safe_rate(
                        frame["transit_within_threshold"].sum(), n
                    ),
                    "drt_minus_walking_access_rate": (
                        finite_mean(frame["drt_access_probability"])
                        - safe_rate(
                            frame["walk_within_threshold"].sum(), n
                        )
                    ),
                    "drt_minus_transit_access_rate": (
                        finite_mean(frame["drt_access_probability"])
                        - safe_rate(
                            frame["transit_within_threshold"].sum(), n
                        )
                    ),
                    "walk_closure_rate_among_unreachable": safe_rate(
                        frame.loc[
                            walk_unreachable,
                            "drt_access_probability",
                        ].sum(),
                        walk_unreachable.sum(),
                    ),
                    "transit_closure_rate_among_unreachable": safe_rate(
                        frame.loc[
                            transit_unreachable,
                            "drt_access_probability",
                        ].sum(),
                        transit_unreachable.sum(),
                    ),
                    "mean_walk_time_saving_sec": finite_mean(
                        frame["mean_walk_time_saving_sec"]
                    ),
                    "mean_transit_time_saving_sec": finite_mean(
                        frame["mean_transit_time_saving_sec"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_overall_table(
    request_consensus: pd.DataFrame,
    threshold_min: float,
) -> pd.DataFrame:
    n = len(request_consensus)
    drt_access_rate = finite_mean(
        request_consensus["drt_access_probability"]
    )
    walk_access_rate = safe_rate(
        request_consensus["walk_within_threshold"].sum(), n
    )
    transit_access_rate = safe_rate(
        request_consensus["transit_within_threshold"].sum(), n
    )
    transit_available_rate = safe_rate(
        request_consensus["transit_available"].sum(), n
    )

    walk_unreachable = ~request_consensus[
        "walk_within_threshold"
    ].astype(bool)
    transit_unreachable = ~request_consensus[
        "transit_within_threshold"
    ].astype(bool)

    rows = [
        {
            "mode": "DRT proposed full",
            "threshold_min": threshold_min,
            "n_requests": n,
            "availability_or_access_rate": drt_access_rate,
            "within_threshold_rate": drt_access_rate,
            "mean_time_sec": finite_mean(
                request_consensus["mean_drt_door_sec_when_observed"]
            ),
            "notes": (
                "DRT rate is averaged across solver seeds for each unique "
                "scenario-request."
            ),
        },
        {
            "mode": "Walking",
            "threshold_min": threshold_min,
            "n_requests": n,
            "availability_or_access_rate": 1.0,
            "within_threshold_rate": walk_access_rate,
            "mean_time_sec": finite_mean(
                request_consensus["walking_sec"]
            ),
            "notes": "Walking benchmark from Valhalla.",
        },
        {
            "mode": "Public transit",
            "threshold_min": threshold_min,
            "n_requests": n,
            "availability_or_access_rate": transit_available_rate,
            "within_threshold_rate": transit_access_rate,
            "mean_time_sec": finite_mean(
                request_consensus.loc[
                    request_consensus["transit_available"].astype(bool),
                    "transit_sec",
                ]
            ),
            "notes": "OTP public-transit benchmark.",
        },
        {
            "mode": "DRT closure of walking-unreachable requests",
            "threshold_min": threshold_min,
            "n_requests": int(walk_unreachable.sum()),
            "availability_or_access_rate": safe_rate(
                request_consensus.loc[
                    walk_unreachable, "drt_access_probability"
                ].sum(),
                walk_unreachable.sum(),
            ),
            "within_threshold_rate": safe_rate(
                request_consensus.loc[
                    walk_unreachable, "drt_access_probability"
                ].sum(),
                walk_unreachable.sum(),
            ),
            "mean_time_sec": float("nan"),
            "notes": (
                "Expected DRT access probability among requests not "
                f"walkable within {threshold_min:g} minutes."
            ),
        },
        {
            "mode": "DRT closure of transit-unreachable requests",
            "threshold_min": threshold_min,
            "n_requests": int(transit_unreachable.sum()),
            "availability_or_access_rate": safe_rate(
                request_consensus.loc[
                    transit_unreachable, "drt_access_probability"
                ].sum(),
                transit_unreachable.sum(),
            ),
            "within_threshold_rate": safe_rate(
                request_consensus.loc[
                    transit_unreachable, "drt_access_probability"
                ].sum(),
                transit_unreachable.sum(),
            ),
            "mean_time_sec": float("nan"),
            "notes": (
                "Expected DRT access probability among requests not "
                f"reachable by public transit within {threshold_min:g} "
                "minutes, including unavailable itineraries."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_statistical_tests(
    scenario_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = [
        (
            "DRT minus walking within-threshold rate",
            "drt_minus_walking_access_rate",
            "greater",
        ),
        (
            "DRT minus transit within-threshold rate",
            "drt_minus_transit_access_rate",
            "greater",
        ),
        (
            "Walking time minus DRT time",
            "mean_walk_time_saving_sec",
            "greater",
        ),
        (
            "Transit time minus DRT time",
            "mean_transit_time_saving_sec",
            "greater",
        ),
        (
            "Walking gap-closure rate",
            "walk_closure_rate_among_unreachable",
            "greater",
        ),
        (
            "Transit gap-closure rate",
            "transit_closure_rate_among_unreachable",
            "greater",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for label, column, alternative in specifications:
        values = pd.to_numeric(
            scenario_metrics[column], errors="coerce"
        )
        values = values[np.isfinite(values)]
        nonzero = values[~np.isclose(values, 0.0)]

        if len(nonzero) == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                nonzero,
                alternative=alternative,
                zero_method="wilcox",
                correction=False,
                mode="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "test": label,
                "metric_column": column,
                "alternative": alternative,
                "n_scenarios": len(values),
                "n_nonzero_scenarios": len(nonzero),
                "mean_difference": float(values.mean())
                if len(values)
                else float("nan"),
                "median_difference": float(values.median())
                if len(values)
                else float("nan"),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm_adjust(tests["p_value"])

    primary = (
        comparison.sort_values(
            ["scenario_id", "replicate_index", "seed"]
        )
        .groupby(["scenario_id", "request_id"], as_index=False)
        .first()
    )

    mcnemar_rows: list[dict[str, Any]] = []
    for label, benchmark_column in [
        ("DRT vs walking accessibility", "walk_within_threshold"),
        ("DRT vs transit accessibility", "transit_within_threshold"),
    ]:
        drt_access = primary["drt_within_threshold"].astype(bool)
        benchmark_access = primary[benchmark_column].astype(bool)
        b = int((drt_access & ~benchmark_access).sum())
        c = int((~drt_access & benchmark_access).sum())
        result = exact_mcnemar(b, c)
        result["comparison"] = label
        result["n_unique_requests"] = len(primary)
        mcnemar_rows.append(result)

    mcnemar = pd.DataFrame(mcnemar_rows)
    return tests, mcnemar


def build_bootstrap_table(
    scenario_metrics: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = [
        "drt_within_threshold_rate",
        "walking_within_threshold_rate",
        "transit_within_threshold_rate",
        "drt_minus_walking_access_rate",
        "drt_minus_transit_access_rate",
        "walk_closure_rate_among_unreachable",
        "transit_closure_rate_among_unreachable",
        "mean_walk_time_saving_sec",
        "mean_transit_time_saving_sec",
    ]

    rows: list[dict[str, Any]] = []
    for column in columns:
        estimate, lower, upper, n = bootstrap_ci(
            scenario_metrics[column],
            repetitions=repetitions,
            rng=rng,
        )
        rows.append(
            {
                "metric": column,
                "estimate": estimate,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "n_scenarios": n,
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    overall: pd.DataFrame,
    scenario: pd.DataFrame,
    tests: pd.DataFrame,
    group: pd.DataFrame,
    quality: dict[str, Any],
    threshold_min: float,
) -> None:
    def pct(value: float) -> str:
        return "NA" if not np.isfinite(value) else f"{100 * value:.2f}%"

    mode_rows = {
        row["mode"]: row
        for row in overall.to_dict(orient="records")
    }

    drt = mode_rows["DRT proposed full"]
    walk = mode_rows["Walking"]
    transit = mode_rows["Public transit"]
    walk_closure = mode_rows[
        "DRT closure of walking-unreachable requests"
    ]
    transit_closure = mode_rows[
        "DRT closure of transit-unreachable requests"
    ]

    test_lookup = {
        row["test"]: row
        for row in tests.to_dict(orient="records")
    }

    gap_notes = ""
    if not group.empty and "is_gap" in set(group["group_variable"]):
        gap_subset = group[
            group["group_variable"].eq("is_gap")
        ].copy()
        gap_notes = "\n### Service-gap subgroup\n\n"
        for _, row in gap_subset.iterrows():
            gap_notes += (
                f"- `is_gap={row['group_value']}`: "
                f"n={int(row['n_requests'])}, "
                f"DRT={pct(row['drt_expected_access_rate'])}, "
                f"walking={pct(row['walking_access_rate'])}, "
                f"transit={pct(row['transit_access_rate'])}.\n"
            )

    markdown = f"""# DRT, Walking, and Public Transit Comparison

Generator script version: `{SCRIPT_VERSION}`

## Data Integrity

- Benchmark scenarios: {quality['benchmark']['benchmark_scenarios']}
- Unique benchmark requests: {quality['benchmark']['benchmark_rows']}
- Official DRT runs: {quality['drt']['drt_runs']}
- Request-level comparison records (including solver seeds): {quality['merge']['comparison_rows']}
- DRT keys not found in benchmark data: {quality['merge']['drt_keys_missing_from_benchmark']}
- Benchmark keys with no DRT results: {quality['merge']['benchmark_keys_missing_from_all_drt_results']}

## Overall Results

At the {threshold_min:g}-minute threshold:

- Expected Proposed DRT access rate: **{pct(drt['within_threshold_rate'])}**
- Walking access rate: **{pct(walk['within_threshold_rate'])}**
- Public transit route availability rate: **{pct(transit['availability_or_access_rate'])}**
- Public transit access rate: **{pct(transit['within_threshold_rate'])}**
- DRT closure rate for requests inaccessible by walking: **{pct(walk_closure['within_threshold_rate'])}**
- DRT closure rate for requests inaccessible by transit: **{pct(transit_closure['within_threshold_rate'])}**

## Scenario-Level Paired Tests

- Difference in access rate between DRT and walking:
  mean={test_lookup['DRT minus walking within-threshold rate']['mean_difference']:.4f},
  Holm-adjusted p={test_lookup['DRT minus walking within-threshold rate']['p_holm']:.6g}
- Difference in access rate between DRT and public transit:
  mean={test_lookup['DRT minus transit within-threshold rate']['mean_difference']:.4f},
  Holm-adjusted p={test_lookup['DRT minus transit within-threshold rate']['p_holm']:.6g}
- Walking time minus DRT time:
  mean={test_lookup['Walking time minus DRT time']['mean_difference']:.2f} seconds,
  Holm-adjusted p={test_lookup['Walking time minus DRT time']['p_holm']:.6g}
- Public transit time minus DRT time:
  mean={test_lookup['Transit time minus DRT time']['mean_difference']:.2f} seconds,
  Holm-adjusted p={test_lookup['Transit time minus DRT time']['p_holm']:.6g}

{gap_notes}

## Notes for Interpretation in the Paper

1. The DRT access rate is calculated as the average success probability for the same request across multiple ALNS seeds, avoiding repeated treatment of the same request as independent observations.
2. Wilcoxon tests use scenarios as the paired units. The primary independent sample size is the number of scenarios, not the number of requests multiplied by the number of seeds.
3. “Inaccessible by transit” includes requests for which OTP returned no feasible itinerary and requests whose public transit door-to-door time exceeded the threshold.
4. Time savings are calculated only for requests served by DRT that also have a valid corresponding benchmark time.
5. DRT closure should not be reported alone; access loss (benchmark-accessible requests not served by DRT) and net gain should also be reported.
"""
    output_path.write_text(markdown, encoding="utf-8")


def write_excel(
    path: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, frame in tables.items():
                safe_sheet = sheet_name[:31]
                frame.to_excel(
                    writer,
                    sheet_name=safe_sheet,
                    index=False,
                )
    except ImportError:
        warnings.warn(
            "openpyxl is not installed; paper_tables.xlsx was not created. "
            "Install with: pip install openpyxl"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Proposed DRT request-level results against walking and "
            "OTP public-transit benchmarks."
        )
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method-name", default="proposed_full")
    parser.add_argument("--benchmark-dir", default="outputs/benchmarks")
    parser.add_argument("--alns-dir", default="outputs/alns")
    parser.add_argument(
        "--status-csv",
        default=None,
        help=(
            "Formal run_status.csv. When omitted, the newest formal status "
            "file under outputs/alns/_comparative_progress is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/multimodal_evaluation",
    )
    parser.add_argument("--threshold-min", type=float, default=15.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--served-implies-within-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Used only when request_results lacks both a within-threshold "
            "field and door-to-door time. This project normally enforces "
            "15 minutes as a hard constraint."
        ),
    )

    parser.add_argument("--request-id-column")
    parser.add_argument("--walking-column")
    parser.add_argument("--transit-column")
    parser.add_argument("--transit-available-column")
    parser.add_argument("--direct-drive-column")
    parser.add_argument("--served-column")
    parser.add_argument("--within-column")
    parser.add_argument("--drt-door-column")
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=DEFAULT_GROUP_COLUMNS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (project_root / path).resolve()
        )

    config_path = resolve(args.config)
    benchmark_dir = resolve(args.benchmark_dir)
    alns_dir = resolve(args.alns_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load only for provenance. Paths are controlled explicitly by CLI.
    base_config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                base_config = loaded

    threshold_sec = args.threshold_min * 60.0

    print("=" * 78)
    print("SIDRE-DRT multimodal evaluation")
    print(f"Version:          {SCRIPT_VERSION}")
    print(f"Project root:     {project_root}")
    print(f"Method:           {args.method_name}")
    print(f"Benchmark dir:    {benchmark_dir}")
    print(f"ALNS dir:         {alns_dir}")
    print(f"Threshold:        {args.threshold_min:g} minutes")
    print(f"Output dir:       {output_dir}")
    print("=" * 78)

    benchmark, benchmark_mapping, benchmark_quality = load_benchmarks(
        benchmark_dir,
        args,
    )

    status_path = find_status_csv(alns_dir, args.status_csv)
    valid_runs, status_quality = load_valid_runs(
        status_path,
        args.method_name,
    )

    drt, drt_mapping, drt_quality = load_drt_results(
        alns_dir,
        args.method_name,
        valid_runs,
        args,
    )

    comparison, merge_quality = merge_run_with_benchmark(
        benchmark,
        drt,
        threshold_sec=threshold_sec,
        served_implies_threshold=args.served_implies_within_threshold,
        group_columns=args.group_columns,
    )

    run_metrics = build_run_metrics(comparison)
    scenario_metrics = build_scenario_metrics(run_metrics)
    request_consensus = build_request_consensus(
        comparison,
        args.group_columns,
    )
    group_metrics = build_group_metrics(
        request_consensus,
        args.group_columns,
    )
    overall_table = build_overall_table(
        request_consensus,
        args.threshold_min,
    )
    statistical_tests, mcnemar_tests = build_statistical_tests(
        scenario_metrics,
        comparison,
    )
    bootstrap_table = build_bootstrap_table(
        scenario_metrics,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )

    quality = {
        "script_version": SCRIPT_VERSION,
        "base_config_path": str(config_path),
        "base_config_loaded": bool(base_config),
        "method_name": args.method_name,
        "threshold_min": args.threshold_min,
        "benchmark": benchmark_quality,
        "status": status_quality,
        "drt": drt_quality,
        "merge": merge_quality,
        "output_rows": {
            "request_level_comparison": len(comparison),
            "request_consensus": len(request_consensus),
            "run_level_metrics": len(run_metrics),
            "scenario_level_metrics": len(scenario_metrics),
            "group_metrics": len(group_metrics),
        },
    }

    comparison.to_parquet(
        output_dir / "request_level_comparison.parquet",
        index=False,
    )
    request_consensus.to_csv(
        output_dir / "request_consensus_across_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    run_metrics.to_csv(
        output_dir / "run_level_drt_vs_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_metrics.to_csv(
        output_dir / "scenario_level_drt_vs_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_metrics.to_csv(
        output_dir / "group_drt_vs_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_table.to_csv(
        output_dir / "paper_table_overall_modes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    statistical_tests.to_csv(
        output_dir / "paired_wilcoxon_drt_vs_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mcnemar_tests.to_csv(
        output_dir / "mcnemar_accessibility_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_table.to_csv(
        output_dir / "bootstrap_ci_drt_vs_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_summary(
        output_dir / "paper_results_summary.md",
        overall_table,
        scenario_metrics,
        statistical_tests,
        group_metrics,
        quality,
        args.threshold_min,
    )

    write_excel(
        output_dir / "paper_tables.xlsx",
        {
            "Overall Modes": overall_table,
            "Scenario Metrics": scenario_metrics,
            "Group Comparison": group_metrics,
            "Wilcoxon Tests": statistical_tests,
            "McNemar Tests": mcnemar_tests,
            "Bootstrap CI": bootstrap_table,
            "Run Metrics": run_metrics,
        },
    )

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name:<45} {path.stat().st_size:>10} bytes")

    failed_quality = (
        merge_quality["drt_keys_missing_from_benchmark"] > 0
        or merge_quality["benchmark_keys_missing_from_all_drt_results"] > 0
    )
    if failed_quality:
        print(
            "\nWARNING: Key mismatches were detected. Review "
            "data_quality_report.json before using the results in the paper."
        )

    print("\nKey outputs:")
    print(f"  {output_dir / 'paper_results_summary.md'}")
    print(f"  {output_dir / 'paper_tables.xlsx'}")
    print(f"  {output_dir / 'data_quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
