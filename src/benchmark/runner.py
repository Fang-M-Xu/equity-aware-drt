from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from routing.valhalla import ValhallaClient

from .otp import OTPClient


def run_benchmarks(
    requests: pd.DataFrame,
    valhalla: ValhallaClient,
    walk_costing: str,
    auto_costing: str,
    otp_client: OTPClient | None,
    otp_departure_datetime: str,
    otp_first_itineraries: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for request in requests.itertuples(index=False):
        walking = valhalla.route(
            request.origin_lon,
            request.origin_lat,
            request.destination_lon,
            request.destination_lat,
            walk_costing,
        )
        driving = valhalla.route(
            request.origin_lon,
            request.origin_lat,
            request.destination_lon,
            request.destination_lat,
            auto_costing,
        )
        transit_available = False
        transit_time = np.nan
        transfers = np.nan
        if otp_client is not None:
            result = otp_client.plan(
                request.origin_lon,
                request.origin_lat,
                request.destination_lon,
                request.destination_lat,
                otp_departure_datetime,
                otp_first_itineraries,
            )
            transit_available = result.available
            transit_time = result.total_time_sec if result.total_time_sec is not None else np.nan
            transfers = result.transfers if result.transfers is not None else np.nan
        rows.append(
            {
                "request_id": request.request_id,
                "walking_time_sec": walking.time_sec,
                "walking_distance_m": walking.distance_m,
                "walking_within_15min": walking.time_sec <= 900,
                "direct_driving_time_sec": driving.time_sec,
                "direct_driving_distance_m": driving.distance_m,
                "transit_available": transit_available,
                "transit_time_sec": transit_time,
                "transit_within_15min": bool(transit_available and transit_time <= 900),
                "transit_transfers": transfers,
            }
        )
    return pd.DataFrame(rows)


def save_benchmarks(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
