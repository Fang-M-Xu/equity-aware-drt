from __future__ import annotations

"""
Generate Table 5.5.2 — Computational performance.

The script reads the formal experiment outputs under outputs/alns (or a user-
specified directory) and summarizes computational performance by method.

Primary source:
    summary.json

Optional fallback sources:
    request_results.parquet
    routes.parquet
    vehicle_results.parquet

The script does NOT require solver_history.csv.

Aggregation rule
----------------
1. Read each completed method × scenario × solver-seed run.
2. Extract computational fields from summary.json when available.
3. Infer requests/scenario from request_results.parquet if needed.
4. Average repeated solver seeds within each method × scenario.
5. Average across scenarios for the final paper table.

Outputs
-------
Table_5_5_2_computational_performance.csv
Table_5_5_2_computational_performance.tex
Table_5_5_2_computational_performance_audit.csv
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, path_from_config

RUNNER_VERSION = "1.0-natural-demand-computational-performance"


# ============================================================
# Method labels/order
# ============================================================

METHOD_LABELS = {
    "alns_random_preposition": "Random + ALNS",
    "alns_demand_preposition": "Expected-demand + ALNS",
    "greedy_full": "Proposed + greedy",
    "alns_efficiency_only": "Efficiency-only ALNS",
    "alns_gap_only": "Gap-only ALNS",
    "alns_vulnerability_only": "Vulnerability-only ALNS",
    "proposed_full": "Proposed full",
}

METHOD_ORDER = [
    "alns_random_preposition",
    "alns_demand_preposition",
    "greedy_full",
    "alns_efficiency_only",
    "alns_gap_only",
    "alns_vulnerability_only",
    "proposed_full",
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table 5.5.2 — Computational performance."
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Formal experiment configuration.",
    )

    parser.add_argument(
        "--alns-dir",
        default=None,
        help=(
            "Optional ALNS result directory. If omitted, alns_dir from "
            "the config is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_5",
        help="Directory for Table 5.5.2 outputs.",
    )

    return parser.parse_args()


# ============================================================
# JSON helpers
# ============================================================

def _flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested JSON using dotted keys.
    """
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_json(value, full_key))
    else:
        out[prefix] = obj

    return out


def _find_numeric(flat: dict[str, Any], candidates: list[str]) -> float:
    """
    Find the first numeric value whose flattened key exactly matches
    or ends with one of the supplied candidate names.
    """
    lower_map = {str(k).lower(): v for k, v in flat.items()}

    # Exact match first.
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            try:
                return float(lower_map[key])
            except (TypeError, ValueError):
                pass

    # Then suffix match for nested JSON keys.
    for candidate in candidates:
        suffix = "." + candidate.lower()
        for key, value in lower_map.items():
            if key.endswith(suffix):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return np.nan


def _find_text(flat: dict[str, Any], candidates: list[str]) -> str | None:
    lower_map = {str(k).lower(): v for k, v in flat.items()}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map and lower_map[key] is not None:
            return str(lower_map[key])

    for candidate in candidates:
        suffix = "." + candidate.lower()
        for key, value in lower_map.items():
            if key.endswith(suffix) and value is not None:
                return str(value)

    return None


# ============================================================
# Run discovery
# ============================================================

def discover_runs(alns_dir: Path) -> list[dict[str, Any]]:
    """
    Discover completed runs by locating summary.json files.

    Expected directory structure:
      alns_dir / method / scenario_XXX / seed_YYYY / summary.json
    """
    runs: list[dict[str, Any]] = []

    for summary_path in sorted(alns_dir.glob("*/scenario_*/seed_*/summary.json")):
        try:
            method = summary_path.parents[2].name
            scenario_name = summary_path.parents[1].name
            seed_name = summary_path.parent.name

            scenario_id = int(scenario_name.split("_")[-1])
            solver_seed = int(seed_name.split("_")[-1])
        except Exception:
            continue

        runs.append(
            {
                "method": method,
                "scenario_id": scenario_id,
                "solver_seed": solver_seed,
                "run_dir": summary_path.parent,
                "summary_path": summary_path,
            }
        )

    return runs


# ============================================================
# Per-run extraction
# ============================================================

def extract_run_metrics(
    run: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary_path: Path = run["summary_path"]
    run_dir: Path = run["run_dir"]

    with summary_path.open("r", encoding="utf-8") as f:
        summary_json = json.load(f)

    flat = _flatten_json(summary_json)

    # Runtime
    runtime_sec = _find_numeric(
        flat,
        [
            "runtime_sec",
            "runtime_seconds",
            "runtime_s",
            "elapsed_sec",
            "elapsed_seconds",
            "wall_time_sec",
            "wall_time_seconds",
            "solve_time_sec",
            "solver_runtime_sec",
            "total_runtime_sec",
        ],
    )

    # Completed iterations
    completed_iterations = _find_numeric(
        flat,
        [
            "completed_iterations",
            "iterations_completed",
            "n_iterations",
            "num_iterations",
            "iteration_count",
            "iterations",
            "final_iteration",
        ],
    )

    # Objective
    objective = _find_numeric(
        flat,
        [
            "objective",
            "objective_value",
            "best_objective",
            "final_objective",
            "best_cost",
            "final_cost",
            "cost",
        ],
    )

    # Convergence/status text
    convergence = _find_text(
        flat,
        [
            "convergence",
            "termination_reason",
            "stop_reason",
            "status",
            "solver_status",
        ],
    )

    # Requests per scenario
    requests = _find_numeric(
        flat,
        [
            "requests",
            "request_count",
            "n_requests",
            "num_requests",
            "total_requests",
        ],
    )

    request_results_path = run_dir / "request_results.parquet"
    if pd.isna(requests) and request_results_path.exists():
        try:
            requests = float(len(pd.read_parquet(request_results_path)))
        except Exception:
            pass

    # Fleet size
    fleet_size = _find_numeric(
        flat,
        [
            "fleet_size",
            "vehicles",
            "vehicle_count",
            "n_vehicles",
            "num_vehicles",
        ],
    )

    if pd.isna(fleet_size):
        try:
            fleet_size = float(config["vehicle"]["fleet_size"])
        except Exception:
            pass

    # Maximum ALNS iterations from run summary; fallback to config.
    max_iterations = _find_numeric(
        flat,
        [
            "max_iterations",
            "maximum_iterations",
            "iteration_limit",
        ],
    )

    if pd.isna(max_iterations):
        try:
            max_iterations = float(config["alns"]["iterations"])
        except Exception:
            try:
                max_iterations = float(config["solver"]["iterations"])
            except Exception:
                pass

    return {
        "method": run["method"],
        "scenario_id": run["scenario_id"],
        "solver_seed": run["solver_seed"],
        "requests_per_scenario": requests,
        "fleet_size": fleet_size,
        "max_iterations": max_iterations,
        "runtime_sec": runtime_sec,
        "completed_iterations": completed_iterations,
        "objective": objective,
        "convergence": convergence,
        "summary_path": str(summary_path),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.alns_dir:
        alns_dir = Path(args.alns_dir)
        if not alns_dir.is_absolute():
            alns_dir = PROJECT_ROOT / alns_dir
    else:
        alns_dir = path_from_config(config, "alns_dir")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(alns_dir)

    if not runs:
        raise FileNotFoundError(
            "No completed runs with summary.json were found under:\n"
            f"{alns_dir}"
        )

    rows = [extract_run_metrics(run, config) for run in runs]
    audit = pd.DataFrame(rows)

    # Keep formal Chapter 5.5 methods when present.
    available_methods = set(audit["method"].dropna().astype(str).unique())
    selected_methods = [m for m in METHOD_ORDER if m in available_methods]

    if selected_methods:
        audit = audit.loc[audit["method"].isin(selected_methods)].copy()
    else:
        selected_methods = sorted(available_methods)

    # --------------------------------------------------------
    # Average seeds within scenario first
    # --------------------------------------------------------

    numeric_columns = [
        "requests_per_scenario",
        "fleet_size",
        "max_iterations",
        "runtime_sec",
        "completed_iterations",
        "objective",
    ]

    scenario_summary = (
        audit
        .groupby(["method", "scenario_id"], as_index=False)[numeric_columns]
        .mean()
    )

    # --------------------------------------------------------
    # Average across scenarios
    # --------------------------------------------------------

    method_summary = (
        scenario_summary
        .groupby("method", as_index=False)[numeric_columns]
        .mean()
    )

    method_summary["Configuration"] = method_summary["method"].map(
        METHOD_LABELS
    ).fillna(method_summary["method"])

    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    method_summary["_order"] = method_summary["method"].map(order_map)
    method_summary["_order"] = method_summary["_order"].fillna(9999)

    # --------------------------------------------------------
    # Choose columns that are actually supported by files
    # --------------------------------------------------------

    table_columns = [
        "Configuration",
        "Requests/scenario",
        "Fleet size",
        "Max iterations",
        "Mean runtime (s)",
    ]

    rename_map = {
        "requests_per_scenario": "Requests/scenario",
        "fleet_size": "Fleet size",
        "max_iterations": "Max iterations",
        "runtime_sec": "Mean runtime (s)",
        "completed_iterations": "Completed iterations",
        "objective": "Mean objective",
    }

    table = (
        method_summary
        .sort_values("_order")
        .rename(columns=rename_map)
    )

    # Only include completed iterations/objective if summary.json actually
    # contains them for at least one run.
    if audit["completed_iterations"].notna().any():
        table_columns.append("Completed iterations")

    if audit["objective"].notna().any():
        table_columns.append("Mean objective")

    table = table[table_columns].reset_index(drop=True)

    # --------------------------------------------------------
    # Save audit file
    # --------------------------------------------------------

    audit_path = (
        output_dir
        / "Table_5_5_2_computational_performance_audit.csv"
    )
    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Save final CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "Table_5_5_2_computational_performance.csv"
    )

    table.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.3f",
    )

    # --------------------------------------------------------
    # Save LaTeX
    # --------------------------------------------------------

    formatted = table.copy()

    for c in ["Requests/scenario", "Fleet size", "Max iterations", "Completed iterations"]:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(
                lambda x: "" if pd.isna(x) else f"{x:.0f}"
            )

    if "Mean runtime (s)" in formatted.columns:
        formatted["Mean runtime (s)"] = formatted["Mean runtime (s)"].map(
            lambda x: "" if pd.isna(x) else f"{x:.1f}"
        )

    if "Mean objective" in formatted.columns:
        formatted["Mean objective"] = formatted["Mean objective"].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )

    tex_path = (
        output_dir
        / "Table_5_5_2_computational_performance.tex"
    )

    column_format = "l" + "r" * (len(formatted.columns) - 1)

    latex = formatted.to_latex(
        index=False,
        escape=True,
        column_format=column_format,
        caption="Computational performance of the evaluated configurations.",
        label="tab:computational_performance",
    )

    tex_path.write_text(latex, encoding="utf-8")

    # --------------------------------------------------------
    # Console audit
    # --------------------------------------------------------

    print()
    print("=" * 96)
    print("Table 5.5.2 — Computational performance")
    print("=" * 96)
    print(f"Runner version : {RUNNER_VERSION}")
    print(f"ALNS directory : {alns_dir}")
    print(f"Methods        : {selected_methods}")
    print(f"Completed runs : {len(audit)}")
    print(f"Scenario rows  : {len(scenario_summary)}")
    print()
    print(table.to_string(index=False))
    print()
    print("Availability audit:")
    print(
        f"  runtime available              : {audit['runtime_sec'].notna().sum()} / {len(audit)}"
    )
    print(
        f"  completed iterations available : {audit['completed_iterations'].notna().sum()} / {len(audit)}"
    )
    print(
        f"  objective available            : {audit['objective'].notna().sum()} / {len(audit)}"
    )
    print()
    print(f"CSV saved   : {csv_path}")
    print(f"LaTeX saved : {tex_path}")
    print(f"Audit saved : {audit_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
