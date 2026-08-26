from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .snapping import load_roads_metric, snap_xy_to_roads
from .valhalla import ValhallaClient


def _walk_result(
    client: ValhallaClient,
    origin_lon: float,
    origin_lat: float,
    destination_lon: float,
    destination_lat: float,
    costing: str,
) -> tuple[float, float]:
    from .valhalla import haversine_m
    straight = haversine_m(origin_lon, origin_lat, destination_lon, destination_lat)
    if straight < 1.0:
        return 0.0, straight
    result = client.route(origin_lon, origin_lat, destination_lon, destination_lat, costing)
    return result.time_sec, result.distance_m


def prepare_operating_scenario(
    scenario: pd.DataFrame,
    vehicles: pd.DataFrame,
    roads_path: Path,
    metric_crs: str,
    target_crs: str,
    snap_max_distance_m: float,
    client: ValhallaClient,
    auto_costing: str,
    walk_costing: str,
    constraints: dict,
    max_sources: int,
    max_targets: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    roads = load_roads_metric(roads_path, metric_crs)
    requests = scenario.copy()
    requests["request_id"] = pd.to_numeric(requests["request_id"], errors="raise").astype(int)

    origin_points = requests[["request_id", "origin_x", "origin_y"]].copy()
    destination_points = requests[["request_id", "destination_x", "destination_y"]].copy()
    pickup_snap = snap_xy_to_roads(
        origin_points,
        "origin_x",
        "origin_y",
        "request_id",
        roads,
        metric_crs,
        target_crs,
        snap_max_distance_m,
    ).rename(
        columns={
            "snapped_x": "pickup_x",
            "snapped_y": "pickup_y",
            "snapped_lon": "pickup_lon",
            "snapped_lat": "pickup_lat",
            "snap_distance_m": "pickup_snap_distance_m",
            "snap_valid": "pickup_snap_valid",
        }
    )
    dropoff_snap = snap_xy_to_roads(
        destination_points,
        "destination_x",
        "destination_y",
        "request_id",
        roads,
        metric_crs,
        target_crs,
        snap_max_distance_m,
    ).rename(
        columns={
            "snapped_x": "dropoff_x",
            "snapped_y": "dropoff_y",
            "snapped_lon": "dropoff_lon",
            "snapped_lat": "dropoff_lat",
            "snap_distance_m": "dropoff_snap_distance_m",
            "snap_valid": "dropoff_snap_valid",
        }
    )
    requests = requests.merge(pickup_snap, on="request_id", how="left", validate="1:1")
    requests = requests.merge(dropoff_snap, on="request_id", how="left", validate="1:1")

    walk_records = []
    for row in requests.itertuples(index=False):
        if not bool(row.pickup_snap_valid) or not bool(row.dropoff_snap_valid):
            walk_records.append((np.inf, np.inf, np.inf, np.inf))
            continue
        access_time, access_distance = _walk_result(
            client,
            row.origin_lon,
            row.origin_lat,
            row.pickup_lon,
            row.pickup_lat,
            walk_costing,
        )
        egress_time, egress_distance = _walk_result(
            client,
            row.dropoff_lon,
            row.dropoff_lat,
            row.destination_lon,
            row.destination_lat,
            walk_costing,
        )
        walk_records.append((access_time, access_distance, egress_time, egress_distance))
    walk_array = np.asarray(walk_records, dtype=float)
    requests[["access_walk_sec", "access_walk_m", "egress_walk_sec", "egress_walk_m"]] = walk_array

    requests["direct_drive_sec"] = np.inf
    requests["direct_drive_m"] = np.inf
    valid_snap_mask = requests["pickup_snap_valid"].fillna(False) & requests["dropoff_snap_valid"].fillna(False)
    valid_indices = requests.index[valid_snap_mask].tolist()
    if valid_indices:
        pickup_coordinates = list(
            zip(requests.loc[valid_indices, "pickup_lon"], requests.loc[valid_indices, "pickup_lat"])
        )
        dropoff_coordinates = list(
            zip(requests.loc[valid_indices, "dropoff_lon"], requests.loc[valid_indices, "dropoff_lat"])
        )
        direct_time, direct_distance = client.matrix_chunked(
            pickup_coordinates,
            dropoff_coordinates,
            auto_costing,
            max_sources=max_sources,
            max_targets=max_targets,
        )
        diagonal = np.arange(len(valid_indices))
        requests.loc[valid_indices, "direct_drive_sec"] = direct_time[diagonal, diagonal]
        requests.loc[valid_indices, "direct_drive_m"] = direct_distance[diagonal, diagonal]
    requests["minimum_door_time_sec"] = (
        requests["access_walk_sec"] + requests["direct_drive_sec"] + requests["egress_walk_sec"]
    )
    requests["direct_feasible"] = (
        requests["pickup_snap_valid"].fillna(False)
        & requests["dropoff_snap_valid"].fillna(False)
        & requests["access_walk_m"].le(float(constraints["max_access_walk_m"]))
        & requests["egress_walk_m"].le(float(constraints["max_egress_walk_m"]))
        & requests["minimum_door_time_sec"].le(float(constraints["max_door_to_door_sec"]))
        & np.isfinite(requests["direct_drive_sec"])
    )

    feasible = requests[requests["direct_feasible"]].copy().reset_index(drop=True)
    feasible["request_index"] = np.arange(len(feasible), dtype=int)

    vehicle_nodes = vehicles.copy().reset_index(drop=True)
    vehicle_nodes["node_type"] = "vehicle_start"
    vehicle_nodes["node_id"] = np.arange(len(vehicle_nodes), dtype=int)
    vehicle_nodes["longitude"] = vehicle_nodes["start_lon"]
    vehicle_nodes["latitude"] = vehicle_nodes["start_lat"]

    pickup_nodes = pd.DataFrame(
        {
            "node_type": "pickup",
            "request_id": feasible["request_id"],
            "longitude": feasible["pickup_lon"],
            "latitude": feasible["pickup_lat"],
        }
    )
    pickup_nodes["node_id"] = np.arange(
        len(vehicle_nodes), len(vehicle_nodes) + len(pickup_nodes), dtype=int
    )
    dropoff_nodes = pd.DataFrame(
        {
            "node_type": "dropoff",
            "request_id": feasible["request_id"],
            "longitude": feasible["dropoff_lon"],
            "latitude": feasible["dropoff_lat"],
        }
    )
    dropoff_nodes["node_id"] = np.arange(
        len(vehicle_nodes) + len(pickup_nodes),
        len(vehicle_nodes) + len(pickup_nodes) + len(dropoff_nodes),
        dtype=int,
    )
    nodes = pd.concat(
        [
            vehicle_nodes[["node_id", "node_type", "vehicle_id", "longitude", "latitude"]],
            pickup_nodes[["node_id", "node_type", "request_id", "longitude", "latitude"]],
            dropoff_nodes[["node_id", "node_type", "request_id", "longitude", "latitude"]],
        ],
        ignore_index=True,
        sort=False,
    )
    pickup_map = pickup_nodes.set_index("request_id")["node_id"]
    dropoff_map = dropoff_nodes.set_index("request_id")["node_id"]
    feasible["pickup_node_id"] = feasible["request_id"].map(pickup_map).astype(int)
    feasible["dropoff_node_id"] = feasible["request_id"].map(dropoff_map).astype(int)
    vehicle_node_map = vehicle_nodes.set_index("vehicle_id")["node_id"]
    vehicles_output = vehicles.copy()
    vehicles_output["start_node_id"] = vehicles_output["vehicle_id"].map(vehicle_node_map).astype(int)

    coordinates = list(zip(nodes["longitude"].astype(float), nodes["latitude"].astype(float)))
    time_matrix, distance_matrix = client.matrix_chunked(
        coordinates,
        coordinates,
        auto_costing,
        max_sources=max_sources,
        max_targets=max_targets,
    )
    return requests, feasible, vehicles_output, time_matrix, distance_matrix, nodes


def save_operating_scenario(
    output_dir: Path,
    all_requests: pd.DataFrame,
    feasible_requests: pd.DataFrame,
    vehicles: pd.DataFrame,
    nodes: pd.DataFrame,
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_requests.to_parquet(output_dir / "requests_all.parquet", index=False)
    feasible_requests.to_parquet(output_dir / "requests_feasible.parquet", index=False)
    vehicles.to_parquet(output_dir / "vehicles.parquet", index=False)
    nodes.to_parquet(output_dir / "nodes.parquet", index=False)
    np.savez_compressed(
        output_dir / "travel_matrices.npz",
        time_sec=time_matrix,
        distance_m=distance_matrix,
    )
    summary = {
        "requests_all": len(all_requests),
        "requests_feasible": len(feasible_requests),
        "requests_infeasible": int((~all_requests["direct_feasible"]).sum()),
        "vehicles": len(vehicles),
        "nodes": len(nodes),
    }
    (output_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
