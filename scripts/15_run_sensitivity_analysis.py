from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import yaml

from alns.io import load_problem
from alns.solver import solve_alns
from config import load_config, project_path, path_from_config
from logging_utils import configure_logging


SUPPORTED_PARAMETERS = (
    "fleet_size",
    "gap_service_benefit",
    "social_welfare_benefit_weight",
    "max_pickup_delay_sec",
    "max_door_to_door_sec",
    "iterations",
)


def _baseline_from_config(config: dict[str, Any]) -> dict[str, float | int]:
    """Return the frozen baseline represented by the selected experiment config."""
    return {
        "fleet_size": int(config["vehicle"]["fleet_size"]),
        "gap_service_benefit": float(config["objective"]["gap_service_benefit"]),
        "social_welfare_benefit_weight": float(
            config["objective"]["social_welfare_benefit_weight"]
        ),
        "max_pickup_delay_sec": float(config["constraints"]["max_pickup_delay_sec"]),
        "max_door_to_door_sec": float(config["constraints"]["max_door_to_door_sec"]),
        "iterations": int(config["alns"]["iterations"]),
    }


def _run_id(parameter: str, value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{parameter}__{text.replace('.', 'p').replace('-', 'm')}"


def _build_ofat_plan(
    config: dict[str, Any],
    sensitivity_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the baseline plus one-factor-at-a-time alternatives."""
    mode = str(sensitivity_config.get("mode", "")).strip().lower()
    if mode not in {"ofat", "one_factor_at_a_time"}:
        raise ValueError(
            "sensitivity YAML must set mode: ofat (or one_factor_at_a_time)"
        )

    alternatives = sensitivity_config.get("alternatives")
    if not isinstance(alternatives, dict) or not alternatives:
        raise ValueError("sensitivity YAML must contain a non-empty alternatives map")

    unknown = set(alternatives) - set(SUPPORTED_PARAMETERS)
    if unknown:
        raise ValueError(f"Unsupported sensitivity parameters: {sorted(unknown)}")

    baseline = _baseline_from_config(config)
    plan: list[dict[str, Any]] = [
        {
            "run_id": "baseline",
            "varied_parameter": "baseline",
            "varied_value": None,
            "is_baseline": True,
            **baseline,
        }
    ]

    for parameter in SUPPORTED_PARAMETERS:
        if parameter not in alternatives:
            continue
        values = alternatives[parameter]
        if not isinstance(values, list):
            raise TypeError(f"alternatives.{parameter} must be a YAML list")

        for raw_value in values:
            value: float | int
            if parameter in {"fleet_size", "iterations"}:
                value = int(raw_value)
            else:
                value = float(raw_value)

            if value == baseline[parameter]:
                continue

            # requests_feasible.parquet was prepared under the baseline threshold.
            # A stricter limit can reject additional requests during evaluation, but a
            # looser limit cannot recover requests already filtered from that file.
            if (
                parameter == "max_door_to_door_sec"
                and float(value) > float(baseline[parameter])
            ):
                raise ValueError(
                    "max_door_to_door_sec cannot exceed the base value when "
                    "requests_feasible.parquet was prepared under the base constraint. "
                    "Regenerate operating scenarios/matrices before testing a looser limit."
                )

            parameters = dict(baseline)
            parameters[parameter] = value
            plan.append(
                {
                    "run_id": _run_id(parameter, value),
                    "varied_parameter": parameter,
                    "varied_value": value,
                    "is_baseline": False,
                    **parameters,
                }
            )

    run_ids = [row["run_id"] for row in plan]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate OFAT run IDs were generated")
    return plan


def _atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def _combine_scenario_outputs(output_dir: Path) -> Path | None:
    frames: list[pd.DataFrame] = []
    for path in sorted(output_dir.glob("scenario_*.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return None

    combined_path = output_dir / "sensitivity_ofat_all_scenarios.csv"
    combined = pd.concat(frames, ignore_index=True)
    sort_columns = [
        column
        for column in ["scenario_id", "plan_index", "run_id"]
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    _atomic_write_csv(combined, combined_path)
    return combined_path


def _finalize_outputs(
    output_dir: Path,
    expected_scenarios: list[int],
    plan: list[dict[str, Any]],
) -> Path:
    """Validate all checkpoint CSVs, build the final combined CSV, then clean them."""
    if not expected_scenarios:
        raise ValueError("--finalize requires --expected-scenarios")

    expected_run_ids = {str(row["run_id"]) for row in plan}
    problems: list[str] = []

    for scenario_id in expected_scenarios:
        path = output_dir / f"scenario_{scenario_id:03d}.csv"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            problems.append(f"empty {path.name}")
            continue
        if "run_id" not in frame.columns:
            problems.append(f"{path.name} has no run_id column")
            continue
        completed = set(frame["run_id"].astype(str))
        missing = sorted(expected_run_ids - completed)
        if missing:
            problems.append(
                f"{path.name} missing {len(missing)} OFAT run(s): {', '.join(missing)}"
            )

    if problems:
        raise RuntimeError(
            "Sensitivity finalization aborted because checkpoints are incomplete:\n- "
            + "\n- ".join(problems)
        )

    combined_path = _combine_scenario_outputs(output_dir)
    if combined_path is None or not combined_path.exists():
        raise RuntimeError("Could not create sensitivity_ofat_all_scenarios.csv")

    combined = pd.read_csv(combined_path)
    expected_rows = len(expected_scenarios) * len(plan)
    if len(combined) != expected_rows:
        raise RuntimeError(
            f"Combined sensitivity file has {len(combined)} rows; expected {expected_rows}. "
            "Checkpoint files were not deleted."
        )

    for scenario_id in expected_scenarios:
        path = output_dir / f"scenario_{scenario_id:03d}.csv"
        if path.exists():
            path.unlink()

    log_path = output_dir / "15_sensitivity_ofat.log"
    if log_path.exists():
        try:
            log_path.unlink()
        except PermissionError:
            # On some Windows logging configurations the handle can remain open until
            # interpreter shutdown. The PowerShell wrapper also removes this file.
            pass

    print(f"Sensitivity outputs finalized: {combined_path}")
    print(f"Rows: {len(combined)} ({len(expected_scenarios)} scenarios × {len(plan)} OFAT runs)")
    return combined_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run checkpointed one-factor-at-a-time ALNS sensitivity analysis."
    )
    parser.add_argument("--config", default="configs/base_competitive.yaml")
    parser.add_argument("--sensitivity", default="configs/sensitivity_ofat.yaml")
    parser.add_argument("--scenario-id", type=int, default=None)
    parser.add_argument("--preposition-method", default="proposed")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--output-subdir",
        default="sensitivity_ofat",
        help="Subdirectory created under metrics_dir.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore the scenario checkpoint and rerun every OFAT configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the OFAT plan without running ALNS.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Validate all expected scenario checkpoints, rebuild the combined CSV, "
            "and delete per-scenario checkpoints/logs so only the final CSV remains."
        ),
    )
    parser.add_argument(
        "--expected-scenarios",
        type=int,
        nargs="*",
        default=None,
        help="Scenario IDs that must be complete before --finalize can clean checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.finalize and args.scenario_id is None:
        raise ValueError("--scenario-id is required unless --finalize is used")

    config = load_config(args.config)
    sensitivity_path = project_path(config, args.sensitivity)
    sensitivity_config = yaml.safe_load(sensitivity_path.read_text(encoding="utf-8-sig"))
    plan = _build_ofat_plan(config, sensitivity_config)

    output_dir = path_from_config(config, "metrics_dir") / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        _finalize_outputs(output_dir, args.expected_scenarios or [], plan)
        return

    configure_logging(output_dir / "15_sensitivity_ofat.log")

    scenario_output = output_dir / f"scenario_{args.scenario_id:03d}.csv"
    if args.restart or not scenario_output.exists():
        existing = pd.DataFrame()
    else:
        try:
            existing = pd.read_csv(scenario_output)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

    completed_run_ids = (
        set(existing["run_id"].astype(str))
        if not existing.empty and "run_id" in existing.columns
        else set()
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "scenario_id": args.scenario_id,
                    "runs_per_scenario": len(plan),
                    "completed": len(completed_run_ids),
                    "remaining": sum(
                        row["run_id"] not in completed_run_ids for row in plan
                    ),
                    "run_ids": [row["run_id"] for row in plan],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    scenario_dir = (
        path_from_config(config, "operating_dir")
        / args.preposition_method
        / f"scenario_{args.scenario_id:03d}"
    )

    (
        all_requests_frame,
        requests_frame,
        vehicles_frame,
        requests,
        all_vehicles,
        time_matrix,
        distance_matrix,
    ) = load_problem(scenario_dir)

    available_vehicle_ids = sorted(all_vehicles)
    available_fleet_size = len(available_vehicle_ids)
    gap_total = (
        int(all_requests_frame["is_gap"].fillna(0).astype(int).sum())
        if "is_gap" in all_requests_frame.columns
        else None
    )

    rows = existing.to_dict("records") if not existing.empty else []
    session_elapsed: list[float] = []
    executed_this_session = 0
    base_seed = int(config["project"]["seed"])
    solver_seed = base_seed + int(args.scenario_id)

    for plan_index, parameters in enumerate(plan):
        run_id = str(parameters["run_id"])
        if run_id in completed_run_ids:
            continue
        if args.max_runs is not None and executed_this_session >= args.max_runs:
            break

        fleet_size = int(parameters["fleet_size"])
        if fleet_size > available_fleet_size:
            raise ValueError(
                f"Requested fleet_size={fleet_size}, but vehicles.parquet contains "
                f"only {available_fleet_size} vehicles."
            )
        vehicle_ids = available_vehicle_ids[:fleet_size]
        vehicles = {vehicle_id: all_vehicles[vehicle_id] for vehicle_id in vehicle_ids}

        objective = copy.deepcopy(config["objective"])
        objective["gap_service_benefit"] = float(parameters["gap_service_benefit"])
        objective["social_welfare_benefit_weight"] = float(
            parameters["social_welfare_benefit_weight"]
        )

        constraints = copy.deepcopy(config["constraints"])
        constraints["max_pickup_delay_sec"] = float(parameters["max_pickup_delay_sec"])
        constraints["max_door_to_door_sec"] = float(parameters["max_door_to_door_sec"])

        alns_config = copy.deepcopy(config["alns"])
        alns_config["iterations"] = int(parameters["iterations"])

        started = time.perf_counter()
        result = solve_alns(
            vehicles,
            requests,
            time_matrix,
            distance_matrix,
            constraints,
            objective,
            alns_config,
            seed=solver_seed,
        )
        wall_elapsed = time.perf_counter() - started
        session_elapsed.append(wall_elapsed)
        executed_this_session += 1

        served = len(result.evaluation.served_requests)
        total_requests = len(all_requests_frame)
        eligible_requests = len(requests)
        gap_served = float(result.evaluation.totals.get("gap_served", 0.0))

        row = {
            "scenario_id": int(args.scenario_id),
            "plan_index": plan_index,
            "run_id": run_id,
            "varied_parameter": parameters["varied_parameter"],
            "varied_value": parameters["varied_value"],
            "is_baseline": bool(parameters["is_baseline"]),
            "solver_seed": solver_seed,
            "preposition_method": args.preposition_method,
            "fleet_size": fleet_size,
            "gap_service_benefit": float(parameters["gap_service_benefit"]),
            "social_welfare_benefit_weight": float(
                parameters["social_welfare_benefit_weight"]
            ),
            "max_pickup_delay_sec": float(parameters["max_pickup_delay_sec"]),
            "max_door_to_door_sec": float(parameters["max_door_to_door_sec"]),
            "iterations": int(parameters["iterations"]),
            "time_limit_sec": float(alns_config["time_limit_sec"]),
            "objective": float(result.evaluation.objective),
            "requests_total": total_requests,
            "requests_alns_eligible": eligible_requests,
            "served": served,
            "unserved_alns_eligible": len(result.evaluation.unserved_requests),
            "unserved_total": total_requests - served,
            "service_rate_all": served / total_requests if total_requests else math.nan,
            "gap_requests_total": gap_total,
            "gap_service_rate": (
                gap_served / gap_total
                if gap_total is not None and gap_total > 0
                else math.nan
            ),
            "elapsed_sec": float(result.elapsed_sec),
            "wall_elapsed_sec": wall_elapsed,
            **result.evaluation.totals,
        }
        rows.append(row)
        completed_run_ids.add(run_id)

        # Per-scenario CSV is intentionally retained as a checkpoint until the wrapper
        # confirms that every selected sensitivity scenario completed successfully.
        scenario_frame = pd.DataFrame(rows)
        if "plan_index" in scenario_frame.columns:
            scenario_frame = scenario_frame.sort_values("plan_index")
        _atomic_write_csv(scenario_frame, scenario_output)
        combined_path = _combine_scenario_outputs(output_dir)

        remaining = sum(item["run_id"] not in completed_run_ids for item in plan)
        mean_elapsed = sum(session_elapsed) / len(session_elapsed)
        eta_sec = remaining * mean_elapsed

        message = {
            **row,
            "progress": f"{len(completed_run_ids)}/{len(plan)}",
            "remaining_runs": remaining,
            "estimated_remaining_sec": eta_sec,
            "scenario_output": str(scenario_output),
            "combined_output": str(combined_path) if combined_path else None,
        }
        print(json.dumps(message, ensure_ascii=False), flush=True)

    _combine_scenario_outputs(output_dir)


if __name__ == "__main__":
    main()
