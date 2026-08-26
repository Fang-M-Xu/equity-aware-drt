from __future__ import annotations

import numpy as np


def greedy_weighted_facility_selection(
    travel_time_sec: np.ndarray,
    demand_weights: np.ndarray,
    number_of_facilities: int,
) -> list[int]:
    if travel_time_sec.ndim != 2:
        raise ValueError("travel_time_sec must be a two-dimensional matrix")
    if len(demand_weights) != travel_time_sec.shape[0]:
        raise ValueError("demand_weights length must match demand rows")
    number_of_facilities = min(number_of_facilities, travel_time_sec.shape[1])
    selected: list[int] = []
    current_best = np.full(travel_time_sec.shape[0], np.inf, dtype=float)
    for _ in range(number_of_facilities):
        best_candidate = None
        best_objective = np.inf
        for candidate in range(travel_time_sec.shape[1]):
            if candidate in selected:
                continue
            updated = np.minimum(current_best, travel_time_sec[:, candidate])
            objective = float(np.sum(demand_weights * updated))
            if objective < best_objective:
                best_objective = objective
                best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        current_best = np.minimum(current_best, travel_time_sec[:, best_candidate])
    return selected


def allocate_vehicles_to_selected_points(
    travel_time_sec: np.ndarray,
    demand_weights: np.ndarray,
    selected_candidate_indices: list[int],
    fleet_size: int,
) -> np.ndarray:
    if not selected_candidate_indices:
        raise ValueError("At least one selected candidate is required")
    selected_times = travel_time_sec[:, selected_candidate_indices]
    assignment = np.argmin(selected_times, axis=1)
    covered_weight = np.zeros(len(selected_candidate_indices), dtype=float)
    for demand_index, facility_index in enumerate(assignment):
        covered_weight[facility_index] += demand_weights[demand_index]
    if covered_weight.sum() <= 0:
        shares = np.full(len(covered_weight), 1.0 / len(covered_weight))
    else:
        shares = covered_weight / covered_weight.sum()
    counts = np.floor(shares * fleet_size).astype(int)
    counts[counts == 0] = 1
    while counts.sum() > fleet_size:
        removable = np.where(counts > 1)[0]
        if len(removable) == 0:
            break
        index = removable[np.argmin(shares[removable])]
        counts[index] -= 1
    while counts.sum() < fleet_size:
        index = int(np.argmax(shares - counts / max(fleet_size, 1)))
        counts[index] += 1
    return counts
