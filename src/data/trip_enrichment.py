from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from routing.valhalla import ValhallaClient

LOGGER = logging.getLogger(__name__)


def enrich_direct_times(
    trips: pd.DataFrame,
    client: ValhallaClient,
    costing: str,
    max_targets: int = 80,
) -> pd.DataFrame:
    output = trips.copy()
    output["estimated_direct_drive_sec"] = np.nan
    output["estimated_direct_drive_m"] = np.nan

    # Repeated origins are handled with one-to-many matrix requests.
    rounded_columns = ["origin_lon", "origin_lat", "destination_lon", "destination_lat"]
    for column in rounded_columns:
        output[f"_{column}"] = output[column].round(6)

    group_columns = ["_origin_lon", "_origin_lat"]
    for (origin_lon, origin_lat), group in output.groupby(group_columns, sort=False):
        unique_targets = (
            group[["_destination_lon", "_destination_lat"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        source = [(float(origin_lon), float(origin_lat))]
        target_times: dict[tuple[float, float], tuple[float, float]] = {}
        for start in range(0, len(unique_targets), max_targets):
            target_chunk = unique_targets.iloc[start : start + max_targets]
            targets = list(
                zip(
                    target_chunk["_destination_lon"].astype(float),
                    target_chunk["_destination_lat"].astype(float),
                )
            )
            times, distances = client.matrix(source, targets, costing)
            for index, target in enumerate(targets):
                target_times[target] = (float(times[0, index]), float(distances[0, index]))

        for row_index in group.index:
            key = (
                float(output.at[row_index, "_destination_lon"]),
                float(output.at[row_index, "_destination_lat"]),
            )
            time_sec, distance_m = target_times[key]
            output.at[row_index, "estimated_direct_drive_sec"] = time_sec
            output.at[row_index, "estimated_direct_drive_m"] = distance_m

    unavailable_columns = ["estimated_direct_drive_sec", "estimated_direct_drive_m"]
    output[unavailable_columns] = output[unavailable_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    output = output.drop(columns=[f"_{column}" for column in rounded_columns])
    return output


def add_estimated_departure_fields(trips: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    output = trips.copy()
    direct_seconds = pd.to_numeric(
        output["estimated_direct_drive_sec"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    output["estimated_direct_drive_sec"] = direct_seconds
    output["destination_arrival_time"] = pd.to_datetime(
        output["destination_arrival_time"], errors="coerce"
    )
    output["estimated_departure_time"] = output["destination_arrival_time"] - pd.to_timedelta(
        direct_seconds, unit="s"
    )
    output["estimated_departure_minute"] = (
        output["estimated_departure_time"].dt.hour * 60
        + output["estimated_departure_time"].dt.minute
    ).astype("Int64")
    output["estimated_departure_bin"] = (
        output["estimated_departure_minute"] // bin_minutes * bin_minutes
    ).astype("Int64")
    output["estimated_duration_min"] = output["estimated_direct_drive_sec"] / 60.0
    output["direct_time_available"] = direct_seconds.notna().astype("Int64")
    return output


def save_routed_split(trips: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(output_path, index=False)
    LOGGER.info("Saved %d routed survey trips to %s", len(trips), output_path)
