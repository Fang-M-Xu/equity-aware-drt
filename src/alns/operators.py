from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .entities import Request, Vehicle
from .evaluation import evaluate_solution
from .insertion import apply_insertion, enumerate_feasible_insertions
from .solution import Solution

DestroyOperator = Callable[..., Solution]
RepairOperator = Callable[..., Solution]


def _number_to_remove(solution: Solution, fraction: float) -> int:
    return max(1, int(round(len(solution.served_request_ids()) * fraction)))


def random_removal(solution: Solution, rng: np.random.Generator, fraction: float, **_) -> Solution:
    candidate = solution.clone()
    served = sorted(candidate.served_request_ids())
    if not served:
        return candidate
    count = min(len(served), _number_to_remove(candidate, fraction))
    remove = set(rng.choice(served, size=count, replace=False).tolist())
    candidate.remove_requests(remove)
    return candidate


def route_removal(solution: Solution, rng: np.random.Generator, fraction: float, **_) -> Solution:
    candidate = solution.clone()
    nonempty = [route for route in candidate.routes.values() if route.stops]
    if not nonempty:
        return candidate
    route = nonempty[int(rng.integers(0, len(nonempty)))]
    candidate.remove_requests(route.request_ids())
    return candidate


def related_removal(
    solution: Solution,
    rng: np.random.Generator,
    fraction: float,
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    **_,
) -> Solution:
    candidate = solution.clone()
    served = sorted(candidate.served_request_ids())
    if not served:
        return candidate
    seed_request = int(rng.choice(served))
    seed = requests[seed_request]
    related = sorted(
        served,
        key=lambda request_id: (
            float(time_matrix[seed.pickup_node, requests[request_id].pickup_node])
            + 300.0 * abs(seed.service_gap - requests[request_id].service_gap)
            + 100.0 * abs(seed.social_welfare_score - requests[request_id].social_welfare_score)
        ),
    )
    count = min(len(related), _number_to_remove(candidate, fraction))
    candidate.remove_requests(set(related[:count]))
    return candidate


def gap_focus_removal(
    solution: Solution,
    rng: np.random.Generator,
    fraction: float,
    requests: dict[int, Request],
    **_,
) -> Solution:
    candidate = solution.clone()
    served = sorted(candidate.served_request_ids())
    if not served:
        return candidate
    non_gap = [request_id for request_id in served if requests[request_id].service_gap == 0]
    gap = [request_id for request_id in served if requests[request_id].service_gap == 1]
    rng.shuffle(non_gap)
    rng.shuffle(gap)
    ordered = non_gap + gap
    count = min(len(ordered), _number_to_remove(candidate, fraction))
    candidate.remove_requests(set(ordered[:count]))
    return candidate


def worst_removal(
    solution: Solution,
    rng: np.random.Generator,
    fraction: float,
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict,
    objective_weights: dict,
    **_,
) -> Solution:
    candidate = solution.clone()
    base = evaluate_solution(
        candidate, vehicles, requests, time_matrix, distance_matrix, constraints, objective_weights
    )
    marginal: list[tuple[float, int]] = []
    for request_id in candidate.served_request_ids():
        removed = candidate.clone()
        removed.remove_requests({request_id})
        evaluation = evaluate_solution(
            removed, vehicles, requests, time_matrix, distance_matrix, constraints, objective_weights
        )
        marginal.append((base.objective - evaluation.objective, request_id))
    marginal.sort(reverse=True)
    count = min(len(marginal), _number_to_remove(candidate, fraction))
    candidate.remove_requests({request_id for _, request_id in marginal[:count]})
    return candidate


def _greedy_repair_ordered(
    solution: Solution,
    ordered_request_ids: list[int],
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict,
    **_,
) -> Solution:
    candidate = solution.clone()
    for request_id in ordered_request_ids:
        if request_id not in candidate.unserved:
            continue
        options = enumerate_feasible_insertions(
            candidate,
            requests[request_id],
            vehicles,
            requests,
            time_matrix,
            distance_matrix,
            constraints,
        )
        if options:
            apply_insertion(candidate, request_id, options[0])
    return candidate


def greedy_repair(solution: Solution, rng: np.random.Generator, requests: dict[int, Request], **kwargs) -> Solution:
    order = list(solution.unserved)
    rng.shuffle(order)
    return _greedy_repair_ordered(solution, order, requests=requests, **kwargs)


def gap_priority_repair(solution: Solution, rng: np.random.Generator, requests: dict[int, Request], **kwargs) -> Solution:
    order = sorted(
        solution.unserved,
        key=lambda request_id: (
            requests[request_id].service_gap,
            requests[request_id].social_welfare_score,
            rng.random(),
        ),
        reverse=True,
    )
    return _greedy_repair_ordered(solution, order, requests=requests, **kwargs)


def equity_priority_repair(solution: Solution, rng: np.random.Generator, requests: dict[int, Request], **kwargs) -> Solution:
    order = sorted(
        solution.unserved,
        key=lambda request_id: (
            requests[request_id].social_welfare_score,
            requests[request_id].service_gap,
            rng.random(),
        ),
        reverse=True,
    )
    return _greedy_repair_ordered(solution, order, requests=requests, **kwargs)


def regret2_repair(
    solution: Solution,
    rng: np.random.Generator,
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict,
    **_,
) -> Solution:
    candidate = solution.clone()
    while candidate.unserved:
        best_request = None
        best_option = None
        best_regret = -np.inf
        for request_id in list(candidate.unserved):
            options = enumerate_feasible_insertions(
                candidate,
                requests[request_id],
                vehicles,
                requests,
                time_matrix,
                distance_matrix,
                constraints,
            )
            if not options:
                continue
            regret = (
                options[1].route_cost_proxy - options[0].route_cost_proxy
                if len(options) > 1
                else 1e9
            )
            regret += 100.0 * requests[request_id].service_gap
            if regret > best_regret:
                best_regret = regret
                best_request = request_id
                best_option = options[0]
        if best_request is None or best_option is None:
            break
        apply_insertion(candidate, best_request, best_option)
    return candidate


DESTROY_OPERATORS = {
    "random": random_removal,
    "worst": worst_removal,
    "related": related_removal,
    "route": route_removal,
    "gap_focus": gap_focus_removal,
}

REPAIR_OPERATORS = {
    "greedy": greedy_repair,
    "regret2": regret2_repair,
    "gap_priority": gap_priority_repair,
    "equity_priority": equity_priority_repair,
}
