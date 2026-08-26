from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from routing.valhalla import ValhallaClient

from .weighted_kmedoids import allocate_vehicles_to_selected_points, greedy_weighted_facility_selection


def load_zone_centroids(
    zones_path: Path,
    zone_id_column: str,
    metric_crs: str,
    target_crs: str,
) -> pd.DataFrame:
    zones = gpd.read_file(zones_path)
    if zone_id_column not in zones.columns:
        raise ValueError(f"Zone file does not contain {zone_id_column}")
    metric = zones.to_crs(metric_crs).copy()
    metric["centroid_geometry"] = metric.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame(
        metric[[zone_id_column]].copy(), geometry=metric["centroid_geometry"], crs=metric_crs
    ).to_crs(target_crs)
    return pd.DataFrame(
        {
            "zone_id": centroid_gdf[zone_id_column].values,
            "longitude": centroid_gdf.geometry.x,
            "latitude": centroid_gdf.geometry.y,
        }
    )


def compute_demand_weights(
    zone_prediction: pd.DataFrame,
    gap_weight: float,
    vulnerability_weight: float,
) -> pd.DataFrame:
    output = zone_prediction.copy()
    vulnerability_columns = [
        column
        for column in [
            "train_is_low_income_share",
            "train_is_no_car_share",
            "train_is_older_share",
            "train_is_disabled_share",
        ]
        if column in output.columns
    ]
    output["vulnerability_index"] = (
        output[vulnerability_columns].mean(axis=1) if vulnerability_columns else 0.0
    )
    output["demand_weight"] = np.clip(output["predicted_intensity"].astype(float), 0, None) * (
        1.0
        + gap_weight * pd.to_numeric(output.get("zone_gap", 0), errors="coerce").fillna(0)
        + vulnerability_weight * output["vulnerability_index"].fillna(0)
    )
    return output


def preposition_vehicles(
    candidates: pd.DataFrame,
    zone_prediction: pd.DataFrame,
    zone_centroids: pd.DataFrame,
    client: ValhallaClient,
    costing: str,
    fleet_size: int,
    unique_points: int,
    capacity: int,
    max_sources: int,
    max_targets: int,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    zones = zone_prediction.merge(zone_centroids, left_on="origin_zone_id", right_on="zone_id", how="inner")
    weights = zones["demand_weight"].to_numpy(dtype=float)
    method_weights: np.ndarray | None = None
    candidate_coordinates = list(zip(candidates["longitude"], candidates["latitude"]))
    zone_coordinates = list(zip(zones["longitude"], zones["latitude"]))

    if method == "random":
        selected = rng.choice(
            len(candidates),
            size=min(unique_points, len(candidates)),
            replace=False,
        ).tolist()
        zone_to_candidate_time = np.zeros(
            (len(zones), len(candidates)),
            dtype=float,
        )
    else:
        candidate_to_zone_time, _ = client.matrix_chunked(
            candidate_coordinates,
            zone_coordinates,
            costing,
            max_sources=max_sources,
            max_targets=max_targets,
        )
        zone_to_candidate_time = candidate_to_zone_time.T

        if method == "demand_only":
            method_weights = np.clip(
                zones["predicted_intensity"].to_numpy(dtype=float),
                0,
                None,
            )
        else:
            method_weights = weights

        selected = greedy_weighted_facility_selection(
            zone_to_candidate_time,
            method_weights,
            min(unique_points, fleet_size),
        )

    if method == "random":
        counts = np.full(
            len(selected),
            fleet_size // len(selected),
            dtype=int,
        )
        counts[: fleet_size % len(selected)] += 1
    else:
        assert method_weights is not None

        counts = allocate_vehicles_to_selected_points(
            zone_to_candidate_time,
            method_weights,
            selected,
            fleet_size,
        )

    rows: list[dict] = []
    vehicle_id = 0
    for selected_position, candidate_index in enumerate(selected):
        candidate = candidates.iloc[candidate_index]
        for _ in range(int(counts[selected_position])):
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "candidate_id": candidate["candidate_id"],
                    "start_x": candidate.get("x", np.nan),
                    "start_y": candidate.get("y", np.nan),
                    "start_lon": candidate["longitude"],
                    "start_lat": candidate["latitude"],
                    "capacity": capacity,
                    "preposition_method": method,
                }
            )
            vehicle_id += 1
    vehicles = pd.DataFrame(rows).head(fleet_size).reset_index(drop=True)
    return vehicles, zone_to_candidate_time
