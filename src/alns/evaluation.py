from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .entities import Request, Stop, Vehicle
from .solution import Route, Solution


@dataclass
class RouteEvaluation:
    feasible: bool
    drive_time_sec: float = 0.0
    drive_distance_m: float = 0.0
    empty_time_sec: float = 0.0
    empty_distance_m: float = 0.0
    passenger_wait_sec: float = 0.0
    passenger_ride_sec: float = 0.0
    end_time_sec: float = 0.0
    request_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class SolutionEvaluation:
    feasible: bool
    objective: float
    route_evaluations: dict[int, RouteEvaluation]
    served_requests: set[int]
    unserved_requests: set[int]
    totals: dict[str, float]


def evaluate_route(
    route: Route,
    vehicle: Vehicle,
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict[str, float],
) -> RouteEvaluation:
    current_node = vehicle.start_node
    current_time = 0.0
    onboard: set[int] = set()
    picked_up: set[int] = set()
    pickup_times: dict[int, float] = {}
    metrics: dict[int, dict[str, float]] = {}
    evaluation = RouteEvaluation(feasible=True)

    for stop in route.stops:
        if stop.request_id not in requests:
            return RouteEvaluation(False, reason=f"Unknown request {stop.request_id}")
        travel_time = float(time_matrix[current_node, stop.node_id])
        travel_distance = float(distance_matrix[current_node, stop.node_id])
        if not np.isfinite(travel_time) or not np.isfinite(travel_distance):
            return RouteEvaluation(False, reason="Unreachable route segment")
        if len(onboard) == 0:
            evaluation.empty_time_sec += travel_time
            evaluation.empty_distance_m += travel_distance
        evaluation.drive_time_sec += travel_time
        evaluation.drive_distance_m += travel_distance
        arrival_time = current_time + travel_time
        request = requests[stop.request_id]

        if stop.kind == "pickup":
            if stop.request_id in picked_up or stop.request_id in onboard:
                return RouteEvaluation(False, reason="Duplicate pickup")
            passenger_ready_time = request.reveal_time_sec + request.access_walk_sec
            pickup_time = max(arrival_time, passenger_ready_time)
            pickup_delay = pickup_time - passenger_ready_time
            if pickup_delay > float(constraints["max_pickup_delay_sec"]) + 1e-9:
                return RouteEvaluation(False, reason="Pickup delay constraint")
            onboard.add(stop.request_id)
            picked_up.add(stop.request_id)
            if len(onboard) > vehicle.capacity:
                return RouteEvaluation(False, reason="Vehicle capacity constraint")
            pickup_times[stop.request_id] = pickup_time
            metrics[stop.request_id] = {
                "pickup_time_sec": pickup_time,
                "pickup_delay_sec": pickup_delay,
            }
            evaluation.passenger_wait_sec += pickup_delay
            current_time = pickup_time

        elif stop.kind == "dropoff":
            if stop.request_id not in onboard:
                return RouteEvaluation(False, reason="Dropoff before pickup")
            dropoff_time = arrival_time
            pickup_time = pickup_times[stop.request_id]
            ride_time = dropoff_time - pickup_time
            maximum_ride_time = (
                float(constraints["max_ride_time_ratio"]) * request.direct_drive_sec
                + float(constraints["max_ride_time_additive_sec"])
            )
            if ride_time > maximum_ride_time + 1e-9:
                return RouteEvaluation(False, reason="Maximum ride time constraint")
            door_to_door = dropoff_time + request.egress_walk_sec - request.reveal_time_sec
            if door_to_door > float(constraints["max_door_to_door_sec"]) + 1e-9:
                return RouteEvaluation(False, reason="Door-to-door constraint")
            onboard.remove(stop.request_id)
            metrics[stop.request_id].update(
                {
                    "dropoff_time_sec": dropoff_time,
                    "ride_time_sec": ride_time,
                    "door_to_door_sec": door_to_door,
                }
            )
            evaluation.passenger_ride_sec += ride_time
            current_time = dropoff_time
        else:
            return RouteEvaluation(False, reason=f"Unknown stop type {stop.kind}")
        current_node = stop.node_id

    if onboard:
        return RouteEvaluation(False, reason="Route ends with onboard passengers")
    evaluation.end_time_sec = current_time
    evaluation.request_metrics = metrics
    return evaluation


def evaluate_solution(
    solution: Solution,
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict[str, float],
    objective_weights: dict[str, float],
) -> SolutionEvaluation:
    route_evaluations: dict[int, RouteEvaluation] = {}
    served: set[int] = set()
    totals = {
        "drive_time_sec": 0.0,
        "drive_distance_m": 0.0,
        "empty_time_sec": 0.0,
        "empty_distance_m": 0.0,
        "passenger_wait_sec": 0.0,
        "passenger_ride_sec": 0.0,
    }
    for vehicle_id, route in solution.routes.items():
        route_evaluation = evaluate_route(
            route,
            vehicles[vehicle_id],
            requests,
            time_matrix,
            distance_matrix,
            constraints,
        )
        route_evaluations[vehicle_id] = route_evaluation
        if not route_evaluation.feasible:
            return SolutionEvaluation(
                False,
                float("inf"),
                route_evaluations,
                served,
                set(requests) - served,
                totals,
            )
        request_ids = route.request_ids()
        if served & request_ids:
            return SolutionEvaluation(
                False,
                float("inf"),
                route_evaluations,
                served,
                set(requests) - served,
                totals,
            )
        served |= request_ids
        for key in totals:
            totals[key] += float(getattr(route_evaluation, key))

    unserved = set(requests) - served
    gap_benefit = sum(requests[request_id].service_gap for request_id in served)
    welfare_benefit = sum(requests[request_id].social_welfare_score for request_id in served)
    objective = (
        float(objective_weights["travel_time_weight"]) * totals["drive_time_sec"]
        + float(objective_weights["empty_time_weight"]) * totals["empty_time_sec"]
        + float(objective_weights["wait_time_weight"]) * totals["passenger_wait_sec"]
        + float(objective_weights["ride_time_weight"]) * totals["passenger_ride_sec"]
        + float(objective_weights["unserved_penalty"]) * len(unserved)
        - float(objective_weights["gap_service_benefit"]) * gap_benefit
        - float(objective_weights["social_welfare_benefit_weight"]) * welfare_benefit
    )
    totals["gap_served"] = float(gap_benefit)
    totals["welfare_served_sum"] = float(welfare_benefit)
    totals["served"] = float(len(served))
    totals["unserved"] = float(len(unserved))
    return SolutionEvaluation(True, objective, route_evaluations, served, unserved, totals)
