from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .entities import Request, Vehicle
from .evaluation import SolutionEvaluation, evaluate_solution
from .insertion import apply_insertion, enumerate_feasible_insertions
from .operators import DESTROY_OPERATORS, REPAIR_OPERATORS
from .solution import Route, Solution


@dataclass
class SolverResult:
    solution: Solution
    evaluation: SolutionEvaluation
    history: list[dict]
    elapsed_sec: float


def build_empty_solution(vehicles: dict[int, Vehicle], requests: dict[int, Request]) -> Solution:
    return Solution(
        routes={vehicle_id: Route(vehicle_id) for vehicle_id in vehicles},
        unserved=set(requests),
    )


def build_greedy_initial_solution(
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict,
) -> Solution:
    solution = build_empty_solution(vehicles, requests)
    ordered = sorted(
        requests,
        key=lambda request_id: (
            requests[request_id].service_gap,
            requests[request_id].social_welfare_score,
            -requests[request_id].direct_drive_sec,
        ),
        reverse=True,
    )
    for request_id in ordered:
        options = enumerate_feasible_insertions(
            solution,
            requests[request_id],
            vehicles,
            requests,
            time_matrix,
            distance_matrix,
            constraints,
        )
        if options:
            apply_insertion(solution, request_id, options[0])
    return solution


def solve_alns(
    vehicles: dict[int, Vehicle],
    requests: dict[int, Request],
    time_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    constraints: dict,
    objective_weights: dict,
    alns_config: dict,
    seed: int,
    greedy_only: bool = False,
) -> SolverResult:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    current = build_greedy_initial_solution(
        vehicles, requests, time_matrix, distance_matrix, constraints
    )
    current_evaluation = evaluate_solution(
        current, vehicles, requests, time_matrix, distance_matrix, constraints, objective_weights
    )
    best = current.clone()
    best_evaluation = current_evaluation
    history = [{"iteration": 0, "objective": current_evaluation.objective, "served": len(current_evaluation.served_requests)}]
    if greedy_only:
        return SolverResult(best, best_evaluation, history, time.perf_counter() - start)

    destroy_names = list(DESTROY_OPERATORS)
    repair_names = list(REPAIR_OPERATORS)
    destroy_weights = np.ones(len(destroy_names), dtype=float)
    repair_weights = np.ones(len(repair_names), dtype=float)
    destroy_scores = np.zeros_like(destroy_weights)
    repair_scores = np.zeros_like(repair_weights)
    destroy_uses = np.zeros_like(destroy_weights)
    repair_uses = np.zeros_like(repair_weights)
    temperature = max(
        1.0,
        abs(current_evaluation.objective) * float(alns_config["initial_temperature_fraction"]),
    )
    iterations = int(alns_config["iterations"])
    time_limit = float(alns_config["time_limit_sec"])
    segment_length = int(alns_config["segment_length"])
    reaction = float(alns_config["reaction_factor"])

    common_kwargs = {
        "vehicles": vehicles,
        "requests": requests,
        "time_matrix": time_matrix,
        "distance_matrix": distance_matrix,
        "constraints": constraints,
        "objective_weights": objective_weights,
    }

    for iteration in range(1, iterations + 1):
        if time.perf_counter() - start >= time_limit:
            break
        destroy_index = int(rng.choice(len(destroy_names), p=destroy_weights / destroy_weights.sum()))
        repair_index = int(rng.choice(len(repair_names), p=repair_weights / repair_weights.sum()))
        fraction = float(
            rng.uniform(
                float(alns_config["destroy_fraction_min"]),
                float(alns_config["destroy_fraction_max"]),
            )
        )
        partial = DESTROY_OPERATORS[destroy_names[destroy_index]](
            current,
            rng=rng,
            fraction=fraction,
            **common_kwargs,
        )
        candidate = REPAIR_OPERATORS[repair_names[repair_index]](
            partial,
            rng=rng,
            **common_kwargs,
        )
        candidate_evaluation = evaluate_solution(
            candidate,
            vehicles,
            requests,
            time_matrix,
            distance_matrix,
            constraints,
            objective_weights,
        )
        delta = candidate_evaluation.objective - current_evaluation.objective
        accepted = candidate_evaluation.feasible and (
            delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
        )
        score = 0.0
        if candidate_evaluation.objective < best_evaluation.objective:
            best = candidate.clone()
            best_evaluation = candidate_evaluation
            score = float(alns_config["score_global_best"])
        elif candidate_evaluation.objective < current_evaluation.objective:
            score = float(alns_config["score_improved"])
        elif accepted:
            score = float(alns_config["score_accepted"])
        if accepted:
            current = candidate
            current_evaluation = candidate_evaluation

        destroy_scores[destroy_index] += score
        repair_scores[repair_index] += score
        destroy_uses[destroy_index] += 1
        repair_uses[repair_index] += 1
        temperature *= float(alns_config["cooling_rate"])

        if iteration % segment_length == 0:
            for index in range(len(destroy_weights)):
                if destroy_uses[index] > 0:
                    destroy_weights[index] = (1 - reaction) * destroy_weights[index] + reaction * (
                        destroy_scores[index] / destroy_uses[index]
                    )
            for index in range(len(repair_weights)):
                if repair_uses[index] > 0:
                    repair_weights[index] = (1 - reaction) * repair_weights[index] + reaction * (
                        repair_scores[index] / repair_uses[index]
                    )
            destroy_scores.fill(0)
            repair_scores.fill(0)
            destroy_uses.fill(0)
            repair_uses.fill(0)

        if iteration % 25 == 0 or iteration == 1:
            history.append(
                {
                    "iteration": iteration,
                    "objective": current_evaluation.objective,
                    "best_objective": best_evaluation.objective,
                    "served": len(current_evaluation.served_requests),
                    "best_served": len(best_evaluation.served_requests),
                    "temperature": temperature,
                    "destroy": destroy_names[destroy_index],
                    "repair": repair_names[repair_index],
                }
            )

    return SolverResult(best, best_evaluation, history, time.perf_counter() - start)
