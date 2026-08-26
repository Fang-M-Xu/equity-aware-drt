from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .lightgbm_model import load_model, predict
from .transformer_data import VocabularyBundle, build_profile_matrix
from .transformer_train import load_transformer_checkpoint


def build_target_zone_prediction(
    zone_static: pd.DataFrame,
    lightgbm_artifact: dict[str, Any],
    day_of_week_code: int,
    departure_bin: int,
) -> pd.DataFrame:
    frame = zone_static.rename(columns={"zone_id": "origin_zone_id"}).copy()
    frame["survey_day_of_week_code"] = int(day_of_week_code)
    frame["time_bin"] = int(departure_bin)
    frame["split_household_count"] = 1000
    frame["time_sin"] = np.sin(2 * np.pi * departure_bin / 1440.0)
    frame["time_cos"] = np.cos(2 * np.pi * departure_bin / 1440.0)
    frame["day_sin"] = np.sin(2 * np.pi * (day_of_week_code - 1) / 7.0)
    frame["day_cos"] = np.cos(2 * np.pi * (day_of_week_code - 1) / 7.0)
    for feature in lightgbm_artifact["features"]:
        if feature not in frame.columns:
            frame[feature] = 0.0
    frame["predicted_intensity"] = predict(lightgbm_artifact, frame)
    return frame.sort_values("origin_zone_id").reset_index(drop=True)


def allocate_zone_counts(
    prediction: pd.DataFrame,
    total_requests: int,
    rng: np.random.Generator,
    gap_share_override: float | None = None,
) -> pd.DataFrame:
    output = prediction.copy()
    intensity = np.clip(output["predicted_intensity"].to_numpy(dtype=float), 0, None)
    if gap_share_override is None:
        probabilities = intensity / intensity.sum() if intensity.sum() > 0 else np.full(len(output), 1 / len(output))
        counts = rng.multinomial(total_requests, probabilities)
    else:
        gap_mask = pd.to_numeric(output.get("zone_gap", 0), errors="coerce").fillna(0).eq(1).to_numpy()
        gap_total = int(round(total_requests * float(gap_share_override)))
        non_gap_total = total_requests - gap_total
        counts = np.zeros(len(output), dtype=int)
        for mask, group_total in [(gap_mask, gap_total), (~gap_mask, non_gap_total)]:
            indices = np.flatnonzero(mask)
            if len(indices) == 0 or group_total == 0:
                continue
            group_intensity = intensity[indices]
            probabilities = (
                group_intensity / group_intensity.sum()
                if group_intensity.sum() > 0
                else np.full(len(indices), 1 / len(indices))
            )
            counts[indices] = rng.multinomial(group_total, probabilities)
    output["generated_count"] = counts
    return output


def _sample_logits(
    logits: torch.Tensor,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
    forbidden_indices: set[int] | None = None,
) -> int:
    values = logits.detach().cpu().numpy().astype(float)
    values[0] = -np.inf
    if forbidden_indices:
        for index in forbidden_indices:
            if 0 <= index < len(values):
                values[index] = -np.inf
    finite = np.isfinite(values)
    if not finite.any():
        return 1
    values = values / max(temperature, 1e-6)
    if top_k > 0 and top_k < finite.sum():
        valid_indices = np.flatnonzero(finite)
        best = valid_indices[np.argsort(values[valid_indices])[-top_k:]]
        mask = np.ones(len(values), dtype=bool)
        mask[best] = False
        values[mask] = -np.inf
    maximum = np.nanmax(values[np.isfinite(values)])
    probabilities = np.zeros_like(values)
    probabilities[np.isfinite(values)] = np.exp(values[np.isfinite(values)] - maximum)
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(len(values), p=probabilities))


def _prepare_profile_row(row: pd.Series, profile_columns: list[str]) -> np.ndarray:
    frame = pd.DataFrame([row.to_dict()])
    profile, _ = build_profile_matrix(frame, profile_columns)
    return profile.iloc[0].to_numpy(dtype=np.float32)


def generate_scenario(
    zone_counts: pd.DataFrame,
    buildings: pd.DataFrame,
    train_trips: pd.DataFrame,
    transformer_model_path: Path,
    vocabulary_path: Path,
    departure_bin: int,
    departure_time: str,
    total_requests: int,
    minimum_od_distance_m: float,
    maximum_sampling_attempts: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_transformer_checkpoint(transformer_model_path, device=device)
    vocabularies = VocabularyBundle.load(vocabulary_path)
    profile_columns = list(checkpoint["profile_columns"])

    valid_buildings = buildings.dropna(
        subset=["building_id", "zone_id", "longitude", "latitude", "gap_i", "building_x", "building_y"]
    ).copy()
    valid_buildings["zone_id_token"] = valid_buildings["zone_id"].astype(str)
    if "activity_type" not in valid_buildings.columns:
        valid_buildings["activity_type"] = "unknown"
    valid_buildings["activity_type"] = valid_buildings["activity_type"].fillna("unknown").astype(str)

    origin_frequency = train_trips["origin_building_id"].astype(str).value_counts(normalize=True)
    profile_pool_by_zone = {
        str(zone): group.reset_index(drop=True)
        for zone, group in train_trips.groupby(train_trips["origin_zone_id"].astype(str))
    }
    global_profile_pool = train_trips.reset_index(drop=True)
    if global_profile_pool.empty:
        raise RuntimeError("Cannot generate scenarios because train_trips is empty.")

    # Pre-compute feasible origin-building pools.  A zone allocated demand by
    # LightGBM may occasionally have no valid building after coordinate/gap
    # filtering.  Such a zone cannot contribute an origin request and its
    # quota therefore needs to be redistributed rather than silently lost.
    zone_building_pools: dict[str, pd.DataFrame] = {}
    zone_origin_weights: dict[str, np.ndarray] = {}
    for zone in zone_counts["origin_zone_id"].astype(str).unique():
        pool = valid_buildings[valid_buildings["zone_id_token"] == zone].reset_index(drop=True)
        if pool.empty:
            continue
        weights = pool["building_id"].astype(str).map(origin_frequency).fillna(1e-6).to_numpy(dtype=float)
        weights = weights / weights.sum()
        zone_building_pools[zone] = pool
        zone_origin_weights[zone] = weights

    if not zone_building_pools:
        raise RuntimeError("No feasible origin zones remain after building validation.")

    # Refill probabilities follow the original LightGBM intensity whenever
    # available, so the fallback remains demand-informed rather than uniform.
    feasible_zone_rows = zone_counts[
        zone_counts["origin_zone_id"].astype(str).isin(zone_building_pools.keys())
    ].copy()
    feasible_zone_tokens = feasible_zone_rows["origin_zone_id"].astype(str).to_numpy()
    if "predicted_intensity" in feasible_zone_rows.columns:
        refill_weight = np.clip(
            pd.to_numeric(feasible_zone_rows["predicted_intensity"], errors="coerce").fillna(0).to_numpy(dtype=float),
            0,
            None,
        )
    else:
        refill_weight = np.clip(
            pd.to_numeric(feasible_zone_rows["generated_count"], errors="coerce").fillna(0).to_numpy(dtype=float),
            0,
            None,
        )
    refill_probabilities = (
        refill_weight / refill_weight.sum()
        if refill_weight.sum() > 0
        else np.full(len(feasible_zone_tokens), 1.0 / len(feasible_zone_tokens))
    )

    rows: list[dict[str, Any]] = []
    request_id = 0

    def _generate_one(origin_zone: str) -> dict[str, Any] | None:
        nonlocal request_id
        zone_buildings = zone_building_pools.get(origin_zone)
        if zone_buildings is None or zone_buildings.empty:
            return None

        weights = zone_origin_weights[origin_zone]
        profile_pool = profile_pool_by_zone.get(origin_zone, global_profile_pool)
        if profile_pool.empty:
            profile_pool = global_profile_pool

        profile_row = profile_pool.iloc[int(rng.integers(0, len(profile_pool)))]
        origin = zone_buildings.iloc[int(rng.choice(len(zone_buildings), p=weights))]
        origin_activity = str(origin.get("activity_type", "unknown"))
        origin_zone_index = vocabularies.zone.encode(origin_zone)
        origin_activity_index = vocabularies.activity.encode(origin_activity)
        departure_index = vocabularies.departure.encode(int(departure_bin))
        profile_vector = _prepare_profile_row(profile_row, profile_columns)

        with torch.no_grad():
            output = model(
                torch.tensor([[origin_zone_index]], dtype=torch.long, device=device),
                torch.tensor([[origin_activity_index]], dtype=torch.long, device=device),
                torch.tensor([[departure_index]], dtype=torch.long, device=device),
                torch.from_numpy(profile_vector).unsqueeze(0).to(device=device, dtype=torch.float32),
                torch.tensor([[False]], dtype=torch.bool, device=device),
            )

        destination_zone_index = _sample_logits(
            output["destination_zone_logits"][0], rng, temperature, top_k
        )
        destination_activity_index = _sample_logits(
            output["destination_activity_logits"][0], rng, temperature, top_k
        )
        duration_index = _sample_logits(output["duration_logits"][0], rng, temperature, top_k)
        destination_zone = vocabularies.zone.decode(destination_zone_index)
        destination_activity = vocabularies.activity.decode(destination_activity_index)
        duration_token = vocabularies.duration.decode(duration_index)

        destination_candidates = valid_buildings[
            valid_buildings["zone_id_token"].eq(destination_zone)
            & valid_buildings["activity_type"].eq(destination_activity)
        ]
        if destination_candidates.empty:
            destination_candidates = valid_buildings[
                valid_buildings["zone_id_token"].eq(destination_zone)
            ]
        if destination_candidates.empty:
            destination_candidates = valid_buildings

        selected_destination = None
        for _attempt in range(maximum_sampling_attempts):
            candidate = destination_candidates.iloc[int(rng.integers(0, len(destination_candidates)))]
            distance = float(
                np.hypot(
                    float(candidate["building_x"]) - float(origin["building_x"]),
                    float(candidate["building_y"]) - float(origin["building_y"]),
                )
            )
            if (
                str(candidate["building_id"]) != str(origin["building_id"])
                and distance >= minimum_od_distance_m
            ):
                selected_destination = candidate
                break

        if selected_destination is None:
            fallback = valid_buildings.copy()
            fallback["_distance"] = np.hypot(
                fallback["building_x"].astype(float) - float(origin["building_x"]),
                fallback["building_y"].astype(float) - float(origin["building_y"]),
            )
            fallback = fallback[
                fallback["building_id"].astype(str).ne(str(origin["building_id"]))
                & fallback["_distance"].ge(minimum_od_distance_m)
            ]
            if fallback.empty:
                return None
            selected_destination = fallback.iloc[int(rng.integers(0, len(fallback)))]

        destination = selected_destination
        result: dict[str, Any] = {
            "request_id": request_id,
            "scenario_seed": seed,
            "departure_time": departure_time,
            "departure_bin": departure_bin,
            "reveal_time_sec": 0.0,
            "origin_building_id": str(origin["building_id"]),
            "destination_building_id": str(destination["building_id"]),
            "origin_x": float(origin["building_x"]),
            "origin_y": float(origin["building_y"]),
            "destination_x": float(destination["building_x"]),
            "destination_y": float(destination["building_y"]),
            "origin_lon": float(origin["longitude"]),
            "origin_lat": float(origin["latitude"]),
            "destination_lon": float(destination["longitude"]),
            "destination_lat": float(destination["latitude"]),
            "origin_zone_id": origin["zone_id"],
            "destination_zone_id": destination["zone_id"],
            "origin_gap": int(origin["gap_i"]),
            "destination_gap": int(destination["gap_i"]),
            "is_gap": int(origin["gap_i"]),
            "service_gap": int(origin["gap_i"]),
            "touches_gap": int(int(origin["gap_i"]) == 1 or int(destination["gap_i"]) == 1),
            "origin_activity_type": origin_activity,
            "destination_activity_type": str(destination.get("activity_type", destination_activity)),
            "transformer_duration_bin": duration_token,
        }
        for column in [
            "sector_group",
            "gender_group",
            "age_final",
            "age_group",
            "is_older",
            "is_80plus",
            "is_disabled",
            "is_student",
            "is_low_income",
            "is_no_car",
            "household_size",
            "has_children",
            "has_older_household_member",
            "household_dependency_count",
            "social_welfare_score",
            "equity_core_complete",
            "equity_missing_component_count",
        ]:
            if column in profile_row.index:
                result[column] = profile_row[column]

        request_id += 1
        return result

    # Primary pass: honour the original zone-level integer allocation wherever
    # the allocated zone has a feasible origin-building pool.
    skipped_zone_quota = 0
    failed_request_count = 0
    for zone_row in zone_counts.itertuples(index=False):
        origin_zone = str(getattr(zone_row, "origin_zone_id"))
        requested_count = int(getattr(zone_row, "generated_count"))
        if requested_count <= 0:
            continue
        if origin_zone not in zone_building_pools:
            skipped_zone_quota += requested_count
            continue
        for _ in range(requested_count):
            result = _generate_one(origin_zone)
            if result is None:
                failed_request_count += 1
            else:
                rows.append(result)

    # Demand-weighted refill: replace any quota lost because an allocated zone
    # had no valid origin building or a request could not satisfy OD-distance
    # feasibility.  This guarantees a fixed candidate-pool size while keeping
    # the fallback distribution tied to LightGBM demand intensity.
    missing = total_requests - len(rows)
    refill_attempts = 0
    max_refill_attempts = max(total_requests * 20, maximum_sampling_attempts)
    while missing > 0 and refill_attempts < max_refill_attempts:
        refill_attempts += 1
        origin_zone = str(rng.choice(feasible_zone_tokens, p=refill_probabilities))
        result = _generate_one(origin_zone)
        if result is not None:
            rows.append(result)
            missing -= 1

    if len(rows) != total_requests:
        raise RuntimeError(
            "Scenario generation incomplete after demand-weighted refill: "
            f"expected {total_requests}, got {len(rows)}; "
            f"skipped_zone_quota={skipped_zone_quota}, "
            f"failed_requests={failed_request_count}, "
            f"refill_attempts={refill_attempts}."
        )

    scenario = pd.DataFrame(rows)
    scenario["request_id"] = np.arange(len(scenario), dtype=int)
    return scenario.reset_index(drop=True)


def save_zone_prediction(prediction: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(path, index=False)
