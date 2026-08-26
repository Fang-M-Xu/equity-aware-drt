from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entities import Request, Stop, Vehicle
from .evaluation import evaluate_route
from .solution import Route, Solution


@dataclass
class InsertionOption:
    vehicle_id: int
    pickup_position: int
    dropoff_position: int
    route: Route
    route_cost_proxy: float


def route_cost_proxy(evaluation) -> float:
    return (
        evaluation.drive_time_sec
        + evaluation.empty_time_sec
        + 2.0 * evaluation.passenger_wait_sec
        + evaluation.passenger_ride_sec
    )


def enumerate_feasible_insertions(
    solution: Solution,
    request: Request,
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict[str, float],
) -> list[InsertionOption]:
    options: list[InsertionOption] = []
    for vehicle_id, route in solution.routes.items():
        base_stops = route.stops
        for pickup_position in range(len(base_stops) + 1):
            with_pickup = list(base_stops)
            with_pickup.insert(pickup_position, Stop.pickup(request))
            for dropoff_position in range(pickup_position + 1, len(with_pickup) + 1):
                candidate_stops = list(with_pickup)
                candidate_stops.insert(dropoff_position, Stop.dropoff(request))
                candidate_route = Route(vehicle_id, candidate_stops)
                evaluation = evaluate_route(
                    candidate_route,
                    vehicles[vehicle_id],
                    requests,
                    time_matrix,
                    distance_matrix,
                    constraints,
                )
                if evaluation.feasible:
                    options.append(
                        InsertionOption(
                            vehicle_id,
                            pickup_position,
                            dropoff_position,
                            candidate_route,
                            route_cost_proxy(evaluation),
                        )
                    )
    return sorted(options, key=lambda option: option.route_cost_proxy)


def apply_insertion(solution: Solution, request_id: int, option: InsertionOption) -> None:
    solution.routes[option.vehicle_id] = option.route
    solution.unserved.discard(request_id)
