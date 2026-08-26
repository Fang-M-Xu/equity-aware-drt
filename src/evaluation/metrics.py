from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BINARY_EQUITY_FIELDS = [
    "is_gap",
    "is_older",
    "is_80plus",
    "is_disabled",
    "is_student",
    "is_low_income",
    "is_no_car",
    "is_unlicensed_adult",
    "has_children",
    "has_older_household_member",
]

CATEGORICAL_EQUITY_FIELDS = [
    "sector_group",
    "gender_group",
    "age_group",
    "income_group",
    "vehicle_group",
]


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def summarize_request_results(frame: pd.DataFrame) -> dict[str, float]:
    requests = len(frame)
    served = int(frame["served"].fillna(False).sum())
    within = int(frame["within_15min"].fillna(False).sum())
    served_frame = frame[frame["served"].fillna(False)]
    gap_frame = frame[pd.to_numeric(frame.get("is_gap", 0), errors="coerce").eq(1)]
    gap_served = int(gap_frame["served"].fillna(False).sum()) if not gap_frame.empty else 0
    return {
        "requests": requests,
        "served": served,
        "unserved": requests - served,
        "service_rate": served / requests if requests else np.nan,
        "rejection_rate": (requests - served) / requests if requests else np.nan,
        "within_15min_served": within,
        "within_15min_rate_among_served": within / served if served else np.nan,
        "within_15min_rate_all_requests": within / requests if requests else np.nan,
        "gap_requests": len(gap_frame),
        "gap_served": gap_served,
        "gap_closure_rate": gap_served / len(gap_frame) if len(gap_frame) else np.nan,
        "pickup_delay_mean_sec": float(pd.to_numeric(served_frame.get("pickup_delay_sec"), errors="coerce").mean()),
        "pickup_delay_p50_sec": percentile(served_frame.get("pickup_delay_sec", pd.Series(dtype=float)), 0.50),
        "pickup_delay_p90_sec": percentile(served_frame.get("pickup_delay_sec", pd.Series(dtype=float)), 0.90),
        "ride_time_mean_sec": float(pd.to_numeric(served_frame.get("ride_time_sec"), errors="coerce").mean()),
        "ride_time_p50_sec": percentile(served_frame.get("ride_time_sec", pd.Series(dtype=float)), 0.50),
        "ride_time_p90_sec": percentile(served_frame.get("ride_time_sec", pd.Series(dtype=float)), 0.90),
        "door_to_door_mean_sec": float(pd.to_numeric(served_frame.get("door_to_door_sec"), errors="coerce").mean()),
        "door_to_door_p90_sec": percentile(served_frame.get("door_to_door_sec", pd.Series(dtype=float)), 0.90),
    }


def group_equity_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for field in BINARY_EQUITY_FIELDS:
        if field not in frame.columns:
            continue
        for value in [0, 1]:
            group = frame[pd.to_numeric(frame[field], errors="coerce").eq(value)]
            if group.empty:
                continue
            metrics = summarize_request_results(group)
            rows.append({"attribute": field, "group": str(value), **metrics})
    for field in CATEGORICAL_EQUITY_FIELDS:
        if field not in frame.columns:
            continue
        for value, group in frame.groupby(field, dropna=False):
            metrics = summarize_request_results(group)
            rows.append({"attribute": field, "group": str(value), **metrics})
    return pd.DataFrame(rows)


def collect_result_files(alns_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in alns_dir.glob("*/scenario_*/seed_*/request_results.parquet"):
        frame = pd.read_parquet(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
