from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer


TRIP_REQUIRED_COLUMNS = [
    "trip_id",
    "person_id",
    "household_id",
    "survey_day_id",
    "stop_seq",
    "destination_arrival_time",
    "origin_x",
    "origin_y",
    "destination_x",
    "destination_y",
    "origin_zone_id",
    "destination_zone_id",
    "origin_gap",
    "destination_gap",
    "is_gap",
    "social_welfare_score",
]

BUILDING_REQUIRED_COLUMNS = [
    "building_id",
    "building_x",
    "building_y",
    "zone_id",
    "gap_i",
]


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _transform_xy(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    source_crs: str,
    target_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(lon), np.asarray(lat)


def add_trip_wgs84(df: pd.DataFrame, source_crs: str, target_crs: str = "EPSG:4326") -> pd.DataFrame:
    output = df.copy()
    origin_lon, origin_lat = _transform_xy(
        output["origin_x"], output["origin_y"], source_crs, target_crs
    )
    destination_lon, destination_lat = _transform_xy(
        output["destination_x"], output["destination_y"], source_crs, target_crs
    )
    output["origin_lon"] = origin_lon
    output["origin_lat"] = origin_lat
    output["destination_lon"] = destination_lon
    output["destination_lat"] = destination_lat
    return output


def load_trip_split(
    path: str | Path,
    source_crs: str = "EPSG:2039",
    target_crs: str = "EPSG:4326",
    add_compatibility_fields: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(df, TRIP_REQUIRED_COLUMNS, str(path))
    df = add_trip_wgs84(df, source_crs, target_crs)
    df["destination_arrival_time"] = pd.to_datetime(
        df["destination_arrival_time"], errors="coerce"
    )
    if add_compatibility_fields:
        df["service_gap"] = pd.to_numeric(df["is_gap"], errors="coerce").astype("Int64")
        if "survey_weight" not in df.columns:
            df["survey_weight"] = 1.0
    return df


def load_dwell_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(
        df,
        ["event_id", "survey_day_id", "person_id", "household_id", "stop_seq", "arrival_time"],
        str(path),
    )
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    return df


def load_building_master(
    path: str | Path,
    source_crs: str = "EPSG:2039",
    target_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    require_columns(df, BUILDING_REQUIRED_COLUMNS, str(path))
    lon, lat = _transform_xy(df["building_x"], df["building_y"], source_crs, target_crs)
    df = df.copy()
    df["longitude"] = lon
    df["latitude"] = lat
    return df


def add_day_context_from_dwell(trips: pd.DataFrame, dwell: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        column
        for column in [
            "survey_day_id",
            "survey_day_of_week_code",
            "day_of_week",
            "day_type",
        ]
        if column in dwell.columns
    ]
    if len(context_columns) == 1:
        result = trips.copy()
        result["survey_day_of_week_code"] = result["destination_arrival_time"].dt.dayofweek + 1
        result["day_of_week"] = result["destination_arrival_time"].dt.day_name().str.lower()
        result["day_type"] = np.where(
            result["destination_arrival_time"].dt.dayofweek.isin([4, 5]), "weekend", "weekday"
        )
        return result

    aggregations = {column: "first" for column in context_columns if column != "survey_day_id"}
    context = dwell.groupby("survey_day_id", as_index=False).agg(aggregations)
    result = trips.merge(context, on="survey_day_id", how="left", validate="m:1")
    if "survey_day_of_week_code" not in result.columns:
        result["survey_day_of_week_code"] = result["destination_arrival_time"].dt.dayofweek + 1
    if "day_of_week" not in result.columns:
        result["day_of_week"] = result["destination_arrival_time"].dt.day_name().str.lower()
    if "day_type" not in result.columns:
        result["day_type"] = np.where(
            result["destination_arrival_time"].dt.dayofweek.isin([4, 5]), "weekend", "weekday"
        )
    return result


def validate_household_splits(split_paths: dict[str, Path]) -> dict[str, int]:
    household_sets: dict[str, set[int]] = {}
    row_counts: dict[str, int] = {}
    for name, path in split_paths.items():
        df = pd.read_parquet(path, columns=["household_id"])
        household_sets[name] = set(pd.to_numeric(df["household_id"], errors="coerce").dropna().astype(int))
        row_counts[name] = len(df)
    names = list(household_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = household_sets[left] & household_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Household leakage between {left} and {right}: {len(overlap)} households."
                )
    return row_counts
