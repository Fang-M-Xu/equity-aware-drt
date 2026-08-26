from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .entities import Request, Vehicle
from .solver import SolverResult


def load_problem(scenario_dir: Path, fleet_size: int | None = None):
    all_requests_frame = pd.read_parquet(scenario_dir / "requests_all.parquet")
    requests_frame = pd.read_parquet(scenario_dir / "requests_feasible.parquet")
    vehicles_frame = pd.read_parquet(scenario_dir / "vehicles.parquet")
    if fleet_size is not None:
        vehicles_frame = vehicles_frame.sort_values("vehicle_id").head(fleet_size).copy()
    matrix = np.load(scenario_dir / "travel_matrices.npz")
    time_matrix = matrix["time_sec"]
    distance_matrix = matrix["distance_m"]
    requests = {
        int(row.request_id): Request(
            request_id=int(row.request_id),
            pickup_node=int(row.pickup_node_id),
            dropoff_node=int(row.dropoff_node_id),
            access_walk_sec=float(row.access_walk_sec),
            egress_walk_sec=float(row.egress_walk_sec),
            direct_drive_sec=float(row.direct_drive_sec),
            service_gap=int(row.is_gap),
            social_welfare_score=float(row.social_welfare_score),
            reveal_time_sec=float(row.reveal_time_sec),
        )
        for row in requests_frame.itertuples(index=False)
    }
    vehicles = {
        int(row.vehicle_id): Vehicle(
            vehicle_id=int(row.vehicle_id),
            start_node=int(row.start_node_id),
            capacity=int(row.capacity),
        )
        for row in vehicles_frame.itertuples(index=False)
    }
    return (
        all_requests_frame,
        requests_frame,
        vehicles_frame,
        requests,
        vehicles,
        time_matrix,
        distance_matrix,
    )


def save_solver_result(
    result: SolverResult,
    all_requests_frame: pd.DataFrame,
    feasible_requests_frame: pd.DataFrame,
    output_dir: Path,
    method: str,
    scenario_id: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_rows: list[dict] = []
    request_metrics: dict[int, dict] = {}
    vehicle_rows: list[dict] = []
    for vehicle_id, route in result.solution.routes.items():
        route_evaluation = result.evaluation.route_evaluations[vehicle_id]
        vehicle_rows.append(
            {
                "vehicle_id": vehicle_id,
                "stops": len(route.stops),
                "served_requests": len(route.request_ids()),
                "drive_time_sec": route_evaluation.drive_time_sec,
                "drive_distance_m": route_evaluation.drive_distance_m,
                "empty_time_sec": route_evaluation.empty_time_sec,
                "empty_distance_m": route_evaluation.empty_distance_m,
                "route_end_time_sec": route_evaluation.end_time_sec,
            }
        )
        request_metrics.update(route_evaluation.request_metrics)
        for sequence, stop in enumerate(route.stops):
            route_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "sequence": sequence,
                    "request_id": stop.request_id,
                    "stop_type": stop.kind,
                    "node_id": stop.node_id,
                }
            )
    feasible_columns = [
        column
        for column in [
            "request_id",
            "pickup_node_id",
            "dropoff_node_id",
            "access_walk_sec",
            "access_walk_m",
            "egress_walk_sec",
            "egress_walk_m",
            "direct_drive_sec",
            "direct_drive_m",
            "minimum_door_time_sec",
        ]
        if column in feasible_requests_frame.columns
    ]
    request_results = all_requests_frame.copy()
    for column in feasible_columns:
        if column == "request_id" or column in request_results.columns:
            continue
        request_results = request_results.merge(
            feasible_requests_frame[["request_id", column]],
            on="request_id",
            how="left",
            validate="1:1",
        )
    request_results["served"] = request_results["request_id"].isin(result.evaluation.served_requests)
    request_results["alns_eligible"] = request_results["request_id"].isin(feasible_requests_frame["request_id"])
    for column in [
        "pickup_time_sec",
        "pickup_delay_sec",
        "dropoff_time_sec",
        "ride_time_sec",
        "door_to_door_sec",
    ]:
        request_results[column] = request_results["request_id"].map(
            lambda request_id: request_metrics.get(int(request_id), {}).get(column, np.nan)
        )
    request_results["within_15min"] = (
        request_results["served"] & request_results["door_to_door_sec"].le(900.0 + 1e-9)
    )
    request_results["method"] = method
    request_results["scenario_id"] = scenario_id
    request_results["solver_seed"] = seed

    pd.DataFrame(route_rows).to_parquet(output_dir / "routes.parquet", index=False)
    pd.DataFrame(vehicle_rows).to_parquet(output_dir / "vehicle_results.parquet", index=False)
    request_results.to_parquet(output_dir / "request_results.parquet", index=False)
    pd.DataFrame(result.history).to_csv(output_dir / "solver_history.csv", index=False)
    total_requests = len(all_requests_frame)
    served = len(result.evaluation.served_requests)
    summary = {
        "method": method,
        "scenario_id": scenario_id,
        "solver_seed": seed,
        "objective": result.evaluation.objective,
        "requests_total": total_requests,
        "requests_alns_eligible": len(feasible_requests_frame),
        "served": served,
        "unserved_total": total_requests - served,
        "service_rate_all": served / total_requests if total_requests else None,
        "elapsed_sec": result.elapsed_sec,
        **result.evaluation.totals,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
