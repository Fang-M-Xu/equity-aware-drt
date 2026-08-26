from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EQUITY_SHARE_COLUMNS = [
    "is_gap",
    "is_low_income",
    "is_no_car",
    "is_older",
    "is_disabled",
    "is_student",
]


def build_zone_static_features(
    buildings: pd.DataFrame,
    train_trips: pd.DataFrame,
) -> pd.DataFrame:
    """Build zone-level features using only buildings and the training split.

    No validation/test outcomes are used here. This makes the resulting feature
    table safe to reuse when constructing validation, test, and forecast panels.
    """
    building_aggregations: dict[str, tuple[str, str]] = {
        "zone_building_count": ("building_id", "nunique"),
        "zone_gap": ("gap_i", "max"),
        "zone_ai_mean": ("ai", "mean") if "ai" in buildings.columns else ("gap_i", "mean"),
    }
    if "floorspace" in buildings.columns:
        building_aggregations["zone_floorspace_sum"] = ("floorspace", "sum")
    if "building_height" in buildings.columns:
        building_aggregations["zone_building_height_mean"] = ("building_height", "mean")

    static = buildings.groupby("zone_id", as_index=False).agg(**building_aggregations)

    activity_column = "activity_type" if "activity_type" in buildings.columns else None
    if activity_column is not None:
        activity_counts = pd.crosstab(buildings["zone_id"], buildings[activity_column])
        activity_counts.columns = [
            f"building_activity_count__{str(column)}" for column in activity_counts.columns
        ]
        static = static.merge(activity_counts.reset_index(), on="zone_id", how="left")

    poi_column = "primary_poi_type" if "primary_poi_type" in buildings.columns else None
    if poi_column is not None:
        poi_counts = pd.crosstab(buildings["zone_id"], buildings[poi_column])
        poi_counts.columns = [f"poi_count__{str(column)}" for column in poi_counts.columns]
        static = static.merge(poi_counts.reset_index(), on="zone_id", how="left")

    train_profile_aggs: dict[str, tuple[str, str]] = {
        "train_origin_trip_count": ("trip_id", "size"),
        "train_origin_households": ("household_id", "nunique"),
    }
    for column in EQUITY_SHARE_COLUMNS:
        if column in train_trips.columns:
            train_profile_aggs[f"train_{column}_share"] = (column, "mean")

    profile = train_trips.groupby("origin_zone_id", as_index=False).agg(**train_profile_aggs)
    profile = profile.rename(columns={"origin_zone_id": "zone_id"})
    static = static.merge(profile, on="zone_id", how="left")

    numeric_columns = static.select_dtypes(include=["number"]).columns
    static[numeric_columns] = static[numeric_columns].fillna(0)
    return static


def _add_cyclical_features(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    minute = pd.to_numeric(output["time_bin"], errors="coerce").fillna(0)
    day = pd.to_numeric(output["survey_day_of_week_code"], errors="coerce").fillna(1)
    output["time_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    output["time_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    output["day_sin"] = np.sin(2 * np.pi * (day - 1) / 7.0)
    output["day_cos"] = np.cos(2 * np.pi * (day - 1) / 7.0)
    return output


def _clean_day_codes(values: Iterable[object]) -> list[int]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype(int)
    return sorted(value for value in numeric.unique().tolist() if 1 <= int(value) <= 7)


def build_zone_time_panel(
    trips: pd.DataFrame,
    zone_static: pd.DataFrame,
    bin_minutes: int,
    include_all_time_bins: bool = True,
    all_day_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Construct a complete day-zone-time demand panel.

    The target is normalized by the number of surveyed households observed on
    the corresponding day-of-week in the split. This is preferable to dividing
    every weekday by the total number of households in the entire split.
    """
    required = [
        "trip_id",
        "household_id",
        "origin_zone_id",
        "origin_building_id",
        "estimated_departure_bin",
        "survey_day_of_week_code",
        "estimated_direct_drive_sec",
        "estimated_direct_drive_m",
    ]
    missing = [column for column in required if column not in trips.columns]
    if missing:
        raise ValueError(f"Routed trip data is missing: {missing}")

    if bin_minutes <= 0 or 1440 % int(bin_minutes) != 0:
        raise ValueError("bin_minutes must be a positive divisor of 1440.")

    valid = trips.dropna(
        subset=[
            "origin_zone_id",
            "estimated_departure_bin",
            "survey_day_of_week_code",
            "household_id",
        ]
    ).copy()
    if valid.empty:
        raise ValueError("No valid routed trips remain for the zone-time panel.")

    valid["origin_zone_id"] = pd.to_numeric(
        valid["origin_zone_id"], errors="coerce"
    ).astype("Int64")
    valid["time_bin"] = pd.to_numeric(
        valid["estimated_departure_bin"], errors="coerce"
    ).astype("Int64")
    valid["survey_day_of_week_code"] = pd.to_numeric(
        valid["survey_day_of_week_code"], errors="coerce"
    ).astype("Int64")
    valid = valid.dropna(subset=["origin_zone_id", "time_bin", "survey_day_of_week_code"])

    invalid_bins = ~valid["time_bin"].between(0, 1439)
    if invalid_bins.any():
        raise ValueError(
            f"Found {int(invalid_bins.sum())} estimated departure bins outside [0, 1439]."
        )

    aggregations: dict[str, tuple[str, str]] = {
        "request_count": ("trip_id", "size"),
        "request_households": ("household_id", "nunique"),
        "origin_building_diversity": ("origin_building_id", "nunique"),
        "mean_direct_time_sec": ("estimated_direct_drive_sec", "mean"),
        "mean_direct_distance_m": ("estimated_direct_drive_m", "mean"),
    }
    for column in EQUITY_SHARE_COLUMNS:
        if column in valid.columns:
            aggregations[f"observed_{column}_share"] = (column, "mean")

    observed = valid.groupby(
        ["survey_day_of_week_code", "origin_zone_id", "time_bin"],
        as_index=False,
    ).agg(**aggregations)

    # Exposure must be computed separately for each day-of-week.
    day_exposure = (
        valid.groupby("survey_day_of_week_code", as_index=False)
        .agg(day_household_count=("household_id", "nunique"))
        .astype({"survey_day_of_week_code": "Int64"})
    )

    days = all_day_codes if all_day_codes is not None else _clean_day_codes(
        valid["survey_day_of_week_code"]
    )
    if not days:
        days = list(range(1, 8))

    zones = sorted(
        int(value)
        for value in pd.to_numeric(zone_static["zone_id"], errors="coerce")
        .dropna()
        .unique()
    )
    if not zones:
        raise ValueError("zone_static contains no valid zone_id values.")

    if include_all_time_bins:
        time_bins = list(range(0, 1440, int(bin_minutes)))
    else:
        time_bins = sorted(int(value) for value in valid["time_bin"].dropna().unique())

    full = pd.DataFrame(
        itertools.product(days, zones, time_bins),
        columns=["survey_day_of_week_code", "origin_zone_id", "time_bin"],
    )
    panel = full.merge(
        observed,
        on=["survey_day_of_week_code", "origin_zone_id", "time_bin"],
        how="left",
        validate="1:1",
    )

    count_columns = [
        "request_count",
        "request_households",
        "origin_building_diversity",
    ]
    panel[count_columns] = panel[count_columns].fillna(0)

    panel = panel.merge(
        day_exposure,
        on="survey_day_of_week_code",
        how="left",
        validate="m:1",
    )
    missing_exposure = panel["day_household_count"].isna()
    if missing_exposure.any():
        missing_days = sorted(panel.loc[missing_exposure, "survey_day_of_week_code"].unique())
        raise ValueError(
            "No household exposure is available for day codes "
            f"{missing_days}. Remove them from all_day_codes or supply observations."
        )

    panel["day_household_count"] = pd.to_numeric(
        panel["day_household_count"], errors="coerce"
    ).clip(lower=1)
    panel["split_household_count"] = int(valid["household_id"].nunique())
    panel["demand_rate_per_1000_households"] = (
        panel["request_count"] / panel["day_household_count"] * 1000.0
    )

    zone_static_for_merge = zone_static.rename(columns={"zone_id": "origin_zone_id"})
    panel = panel.merge(
        zone_static_for_merge,
        on="origin_zone_id",
        how="left",
        validate="m:1",
    )

    numeric_columns = panel.select_dtypes(include=["number"]).columns
    # Keep observed means absent for zero-demand cells as zero for audit only;
    # all observed_* variables are excluded from LightGBM features downstream.
    panel[numeric_columns] = panel[numeric_columns].fillna(0)
    panel = _add_cyclical_features(panel)

    return panel.sort_values(
        ["survey_day_of_week_code", "origin_zone_id", "time_bin"]
    ).reset_index(drop=True)


def summarize_panel(panel: pd.DataFrame, target_column: str) -> dict[str, float | int]:
    target = pd.to_numeric(panel[target_column], errors="coerce").fillna(0.0)
    positive = target > 0
    return {
        "rows": int(len(panel)),
        "positive_rows": int(positive.sum()),
        "zero_rows": int((~positive).sum()),
        "zero_share": float((~positive).mean()) if len(panel) else float("nan"),
        "target_mean": float(target.mean()) if len(panel) else float("nan"),
        "positive_target_mean": float(target.loc[positive].mean()) if positive.any() else float("nan"),
        "day_household_count_min": float(panel["day_household_count"].min()),
        "day_household_count_max": float(panel["day_household_count"].max()),
        "day_household_count_mean": float(panel["day_household_count"].mean()),
    }


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)