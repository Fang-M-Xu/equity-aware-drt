from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

from analysis_common import (
    bootstrap_mean_ci,
    canonical_id,
    cliffs_delta,
    find_column,
    friedman_test,
    holm_adjust,
    jain_index,
    load_yaml,
    paired_bootstrap_difference_ci,
    paired_wilcoxon,
    parse_scenario_from_path,
    parse_seed_from_path,
    read_table,
    resolve_path,
    safe_divide,
    theil_index,
    to_bool,
    write_excel,
    write_json,
)


DEFAULT_GROUP_COLUMNS = [
    "is_gap",
    "is_low_income",
    "is_no_car",
    "is_older",
    "is_80plus",
    "is_disabled",
    "is_student",
    "sector_group",
    "gender_group",
    "age_group",
    "income_group",
    "vehicle_group",
]

ALIASES = {
    "request_id": ["request_id", "req_id", "trip_id", "id"],
    "served": ["served", "is_served", "accepted", "assigned"],
    "within": ["within_15min", "within_15_min", "within_threshold"],
    "pickup_delay": ["pickup_delay_sec", "pickup_delay", "wait_time_sec", "waiting_time_sec"],
    "ride_time": ["ride_time_sec", "in_vehicle_time_sec", "drt_ride_time_sec"],
    "door_time": ["door_to_door_sec", "door_to_door_time_sec", "total_travel_time_sec"],
    "distance": ["route_distance_m", "vehicle_distance_m", "distance_m", "total_distance_m"],
    "empty_distance": ["empty_distance_m", "deadhead_distance_m", "first_leg_empty_distance_m"],
    "objective": ["objective", "objective_value", "total_cost", "cost"],
    "is_gap": ["is_gap", "service_gap", "gap_i", "origin_gap"],
    "need_score": ["need_score", "social_welfare_score", "vulnerability_score", "welfare_score"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute routing, subgroup equity, price-of-fairness and statistical tests.")
    parser.add_argument("--analysis-config", default="configs/results_analysis.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--alns-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _discover_method(path: Path, known_methods: list[str]) -> str:
    text = str(path).lower()
    matches = [method for method in known_methods if method.lower() in text]
    if matches:
        return max(matches, key=len)
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "alns" and index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.parent.parent.name


def _load_request_results(alns_dir: Path, methods: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(alns_dir.rglob("request_results.parquet")) + sorted(alns_dir.rglob("request_results.csv"))
    paths = [path for path in paths if "_comparative_progress" not in str(path)]
    frames = []
    records = []
    for path in paths:
        frame = read_table(path)
        if frame.empty:
            continue
        method = _discover_method(path, methods)
        scenario = parse_scenario_from_path(path)
        seed = parse_seed_from_path(path)
        method_col = find_column(frame, ["method", "method_name"])
        scenario_col = find_column(frame, ["scenario_id", "scenario"])
        seed_col = find_column(frame, ["seed", "solver_seed"])
        frame = frame.copy()
        frame["_method"] = frame[method_col].astype(str) if method_col else method
        frame["_scenario_id"] = pd.to_numeric(frame[scenario_col], errors="coerce") if scenario_col else scenario
        frame["_seed"] = pd.to_numeric(frame[seed_col], errors="coerce") if seed_col else seed
        if frame["_scenario_id"].isna().all() or frame["_seed"].isna().all():
            continue
        frame["_source_file"] = str(path)
        frames.append(frame)
        records.append({"path": str(path), "method": method, "scenario_id": scenario, "seed": seed, "rows": len(frame)})
    if not frames:
        raise FileNotFoundError(f"No request_results files found under {alns_dir}")
    return pd.concat(frames, ignore_index=True, sort=False), records


def _group_columns(frame: pd.DataFrame, configured: list[str]) -> list[str]:
    result = []
    lookup = {column.lower(): column for column in frame.columns}
    for requested in configured:
        if requested in frame.columns:
            result.append(requested)
        elif requested.lower() in lookup:
            result.append(lookup[requested.lower()])
    return list(dict.fromkeys(result))


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    served_col = find_column(frame, ALIASES["served"], required=True)
    request_col = find_column(frame, ALIASES["request_id"])
    mapping = {
        "served": served_col,
        "request_id": request_col,
        "within": find_column(frame, ALIASES["within"]),
        "pickup_delay": find_column(frame, ALIASES["pickup_delay"]),
        "ride_time": find_column(frame, ALIASES["ride_time"]),
        "door_time": find_column(frame, ALIASES["door_time"]),
        "distance": find_column(frame, ALIASES["distance"]),
        "empty_distance": find_column(frame, ALIASES["empty_distance"]),
        "objective": find_column(frame, ALIASES["objective"]),
        "is_gap": find_column(frame, ALIASES["is_gap"]),
        "need_score": find_column(frame, ALIASES["need_score"]),
    }
    output = pd.DataFrame({
        "method": frame["_method"].astype(str),
        "scenario_id": pd.to_numeric(frame["_scenario_id"], errors="coerce"),
        "seed": pd.to_numeric(frame["_seed"], errors="coerce"),
        "request_id": canonical_id(frame[request_col]) if request_col else frame.groupby(["_method", "_scenario_id", "_seed"]).cumcount().astype(str),
        "served": to_bool(frame[served_col]),
        "_source_file": frame["_source_file"].astype(str),
    })
    for target, source in mapping.items():
        if target in {"served", "request_id"} or not source:
            continue
        if target in {"within", "is_gap"}:
            output[target] = to_bool(frame[source])
        else:
            output[target] = pd.to_numeric(frame[source], errors="coerce")
    for column in frame.columns:
        if column.startswith("is_") or column.endswith("_group") or column in {"sector_group", "gender_group", "age_group", "income_group", "vehicle_group"}:
            if column not in output:
                output[column] = frame[column].values
    output = output.dropna(subset=["scenario_id", "seed"]).copy()
    output["scenario_id"] = output["scenario_id"].astype(int)
    output["seed"] = output["seed"].astype(int)
    output = output.sort_values("_source_file").drop_duplicates(["method", "scenario_id", "seed", "request_id"], keep="last")
    return output, mapping


def _run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["method", "scenario_id", "seed"], sort=True):
        method, scenario, seed = keys
        served = group["served"]
        served_group = group[served]
        row = {
            "method": method,
            "scenario_id": scenario,
            "seed": seed,
            "requests": len(group),
            "served": int(served.sum()),
            "service_rate": float(served.mean()),
            "rejection_rate": float((~served).mean()),
        }
        if "within" in group:
            row["within_15min_rate_all_requests"] = float(group["within"].fillna(False).mean())
        if "is_gap" in group:
            gap = group["is_gap"].fillna(False).astype(bool)
            row["gap_requests"] = int(gap.sum())
            row["gap_service_rate"] = float(group.loc[gap, "served"].mean()) if gap.any() else float("nan")
            row["nongap_service_rate"] = float(group.loc[~gap, "served"].mean()) if (~gap).any() else float("nan")
            row["gap_minus_nongap_service_rate"] = row["gap_service_rate"] - row["nongap_service_rate"]
        for metric in ["pickup_delay", "ride_time", "door_time", "distance", "empty_distance", "objective"]:
            if metric in served_group:
                values = pd.to_numeric(served_group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
                row[f"{metric}_p90"] = float(values.quantile(0.90)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _group_metrics(frame: pd.DataFrame, group_columns: list[str], min_cell: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fairness_rows = []
    for (method, scenario, seed), run in frame.groupby(["method", "scenario_id", "seed"], sort=True):
        run_group_rates = []
        for attribute in group_columns:
            if attribute not in run:
                continue
            raw = run[attribute]
            if attribute.startswith("is_"):
                values = to_bool(raw).map({True: "1", False: "0"})
            else:
                values = raw.fillna("__MISSING__").astype(str)
            local_rates = []
            for value, group in run.assign(_group_value=values).groupby("_group_value", dropna=False):
                n = len(group)
                served_n = int(group["served"].sum())
                row = {
                    "method": method,
                    "scenario_id": scenario,
                    "seed": seed,
                    "group_attribute": attribute,
                    "group_value": str(value),
                    "requests": n,
                    "served": served_n,
                    "service_rate": served_n / n if n else float("nan"),
                    "eligible_for_fairness": n >= min_cell,
                }
                for metric in ["pickup_delay", "door_time", "ride_time"]:
                    if metric in group:
                        served_values = pd.to_numeric(group.loc[group["served"], metric], errors="coerce").dropna()
                        row[f"{metric}_mean"] = float(served_values.mean()) if len(served_values) else float("nan")
                rows.append(row)
                if n >= min_cell:
                    local_rates.append(row["service_rate"])
                    run_group_rates.append(row["service_rate"])
            if len(local_rates) >= 2:
                fairness_rows.append({
                    "method": method,
                    "scenario_id": scenario,
                    "seed": seed,
                    "group_attribute": attribute,
                    "groups_included": len(local_rates),
                    "worst_group_service_rate": float(np.min(local_rates)),
                    "best_group_service_rate": float(np.max(local_rates)),
                    "max_min_service_gap": float(np.max(local_rates) - np.min(local_rates)),
                    "jain_service_index": jain_index(local_rates),
                    "theil_service_index": theil_index(local_rates),
                    "service_rate_cv": float(np.std(local_rates, ddof=1) / np.mean(local_rates)) if len(local_rates) > 1 and np.mean(local_rates) > 0 else float("nan"),
                })
        if run_group_rates:
            fairness_rows.append({
                "method": method,
                "scenario_id": scenario,
                "seed": seed,
                "group_attribute": "__all_protected_cells__",
                "groups_included": len(run_group_rates),
                "worst_group_service_rate": float(np.min(run_group_rates)),
                "best_group_service_rate": float(np.max(run_group_rates)),
                "max_min_service_gap": float(np.max(run_group_rates) - np.min(run_group_rates)),
                "jain_service_index": jain_index(run_group_rates),
                "theil_service_index": theil_index(run_group_rates),
                "service_rate_cv": float(np.std(run_group_rates, ddof=1) / np.mean(run_group_rates)) if len(run_group_rates) > 1 and np.mean(run_group_rates) > 0 else float("nan"),
            })
    return pd.DataFrame(rows), pd.DataFrame(fairness_rows)


def _scenario_average(run_metrics: pd.DataFrame, fairness: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in run_metrics.select_dtypes(include="number").columns
        if column not in {"scenario_id", "seed"}
    ]
    scenario = run_metrics.groupby(["method", "scenario_id"], as_index=False)[numeric].mean()
    overall_fairness = fairness[fairness["group_attribute"].eq("__all_protected_cells__")]
    if not overall_fairness.empty:
        fairness_numeric = [
            "worst_group_service_rate", "best_group_service_rate", "max_min_service_gap",
            "jain_service_index", "theil_service_index", "service_rate_cv"
        ]
        fair_scenario = overall_fairness.groupby(["method", "scenario_id"], as_index=False)[fairness_numeric].mean()
        scenario = scenario.merge(fair_scenario, on=["method", "scenario_id"], how="left")
    return scenario


def _summary_table(scenario: pd.DataFrame, metrics: list[str], repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    for method, group in scenario.groupby("method"):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean, lower, upper, n = bootstrap_mean_ci(values, repetitions=repetitions, seed=seed)
            rows.append({
                "method": method,
                "metric": metric,
                "n_scenarios": n,
                "mean": mean,
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "ci95_lower": lower,
                "ci95_upper": upper,
            })
    return pd.DataFrame(rows)


def _statistical_tests(scenario: pd.DataFrame, metrics: list[str], reference: str, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    friedman_rows = []
    methods = sorted(scenario["method"].unique())
    for metric in metrics:
        if metric not in scenario:
            continue
        friedman_rows.append({"metric": metric, **friedman_test(scenario, subject_column="scenario_id", method_column="method", value_column=metric)})
        pivot = scenario.pivot_table(index="scenario_id", columns="method", values=metric, aggfunc="mean")
        if reference not in pivot:
            continue
        for baseline in methods:
            if baseline == reference or baseline not in pivot:
                continue
            pair = pivot[[reference, baseline]].dropna()
            test = paired_wilcoxon(pair[reference], pair[baseline])
            difference, lower, upper, n = paired_bootstrap_difference_ci(pair[reference], pair[baseline], repetitions=repetitions, seed=seed)
            rows.append({
                "metric": metric,
                "reference": reference,
                "baseline": baseline,
                "pairs": n,
                "mean_difference_reference_minus_baseline": difference,
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
                "wilcoxon_statistic": test["statistic"],
                "p_value": test["p_value"],
                "cliffs_delta_unpaired_effect": cliffs_delta(pair[reference], pair[baseline]),
            })
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["p_value_holm"] = holm_adjust(tests["p_value"])
    return tests, pd.DataFrame(friedman_rows)


def _price_of_fairness(scenario: pd.DataFrame, efficiency_method: str, equity_methods: list[str]) -> pd.DataFrame:
    rows = []
    metrics = [
        "service_rate", "gap_service_rate", "worst_group_service_rate",
        "pickup_delay_mean", "door_time_mean", "distance_mean", "objective_mean"
    ]
    base = scenario[scenario["method"].eq(efficiency_method)]
    for method in equity_methods:
        current = scenario[scenario["method"].eq(method)]
        merged = current.merge(base, on="scenario_id", suffixes=("_equity", "_efficiency"))
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            output = {"method": method, "efficiency_baseline": efficiency_method, "scenario_id": int(row["scenario_id"])}
            for metric in metrics:
                left = f"{metric}_equity"
                right = f"{metric}_efficiency"
                if left in merged.columns and right in merged.columns:
                    output[f"delta_{metric}"] = row[left] - row[right]
            gain = output.get("delta_worst_group_service_rate")
            cost = output.get("delta_pickup_delay_mean")
            output["pickup_seconds_per_1pct_worst_group_gain"] = (
                cost / (100 * gain) if gain is not None and np.isfinite(gain) and gain > 0 and cost is not None and np.isfinite(cost) else float("nan")
            )
            service_cost = -output.get("delta_service_rate", float("nan"))
            output["service_rate_loss_per_1pct_worst_group_gain"] = (
                service_cost / gain if np.isfinite(service_cost) and gain is not None and np.isfinite(gain) and gain > 0 else float("nan")
            )
            rows.append(output)
    return pd.DataFrame(rows)


def _pareto_frontier(summary_wide: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "service_rate", "worst_group_service_rate"}
    if not required.issubset(summary_wide.columns):
        return pd.DataFrame()
    rows = []
    for _, candidate in summary_wide.iterrows():
        dominated = False
        for _, other in summary_wide.iterrows():
            if other["method"] == candidate["method"]:
                continue
            no_worse = other["service_rate"] >= candidate["service_rate"] and other["worst_group_service_rate"] >= candidate["worst_group_service_rate"]
            strictly = other["service_rate"] > candidate["service_rate"] or other["worst_group_service_rate"] > candidate["worst_group_service_rate"]
            if no_worse and strictly:
                dominated = True
                break
        rows.append({**candidate.to_dict(), "pareto_efficient": not dominated})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    analysis_path = resolve_path(project_root, args.analysis_config)
    analysis = load_yaml(analysis_path)
    base_path = resolve_path(project_root, args.base_config or analysis.get("base_config", "configs/base.yaml"))
    base = load_yaml(base_path)
    settings = analysis.get("routing_equity", {})

    alns_dir = resolve_path(project_root, args.alns_dir or settings.get("alns_dir") or base.get("paths", {}).get("alns_dir", "outputs/alns"))
    output_dir = resolve_path(project_root, args.output_dir or settings.get("output_dir", "outputs/paper_results/routing_equity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    experiments_path = resolve_path(project_root, settings.get("experiments_config", "configs/experiments_nature.yaml"))
    methods = []
    if experiments_path.exists():
        experiments = load_yaml(experiments_path)
        methods = [str(item["name"]) for item in experiments.get("methods", []) if isinstance(item, dict) and item.get("name")]
    methods.extend([str(value) for value in settings.get("additional_methods", [])])
    methods = list(dict.fromkeys(methods))

    raw, source_records = _load_request_results(alns_dir, methods)
    prepared, detected = _prepare(raw)
    available_methods = sorted(prepared["method"].unique())
    configured_groups = [str(value) for value in settings.get("group_columns", DEFAULT_GROUP_COLUMNS)]
    groups = _group_columns(prepared, configured_groups)
    min_cell = int(settings.get("minimum_group_cell", 10))

    run_metrics = _run_metrics(prepared)
    group_metrics, fairness = _group_metrics(prepared, groups, min_cell)
    scenario = _scenario_average(run_metrics, fairness)

    metrics = [
        metric for metric in settings.get("summary_metrics", [
            "service_rate", "gap_service_rate", "pickup_delay_mean", "door_time_mean",
            "distance_mean", "worst_group_service_rate", "max_min_service_gap",
            "jain_service_index", "theil_service_index"
        ]) if metric in scenario.columns
    ]
    repetitions = int(analysis.get("statistics", {}).get("bootstrap_repetitions", 5000))
    seed = int(analysis.get("statistics", {}).get("seed", 2026))
    summary = _summary_table(scenario, metrics, repetitions, seed)

    reference = str(settings.get("reference_method", "proposed_full"))
    tests, friedman = _statistical_tests(scenario, metrics, reference, repetitions, seed)
    efficiency = str(settings.get("efficiency_baseline", "alns_efficiency_only"))
    equity_methods = [str(value) for value in settings.get("equity_methods", ["alns_gap_only", "proposed_full"])]
    price = _price_of_fairness(scenario, efficiency, equity_methods)

    wide = summary.pivot(index="method", columns="metric", values="mean").reset_index()
    pareto = _pareto_frontier(wide)

    run_metrics.to_csv(output_dir / "routing_run_metrics.csv", index=False, encoding="utf-8-sig")
    scenario.to_csv(output_dir / "routing_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(output_dir / "group_service_metrics.csv", index=False, encoding="utf-8-sig")
    fairness.to_csv(output_dir / "fairness_indices_by_run.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "paper_table_routing_equity.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(output_dir / "paired_routing_tests.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(output_dir / "friedman_routing_tests.csv", index=False, encoding="utf-8-sig")
    price.to_csv(output_dir / "price_of_fairness.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(output_dir / "efficiency_equity_pareto.csv", index=False, encoding="utf-8-sig")

    # Figure: service vs worst-group performance.
    if {"service_rate", "worst_group_service_rate"}.issubset(wide.columns):
        plt.figure(figsize=(7.2, 5.6))
        plt.scatter(wide["service_rate"], wide["worst_group_service_rate"], s=55)
        for _, row in wide.iterrows():
            plt.annotate(row["method"], (row["service_rate"], row["worst_group_service_rate"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        plt.xlabel("Overall service rate")
        plt.ylabel("Worst-group service rate")
        plt.title("Efficiency–equity comparison")
        plt.tight_layout()
        plt.savefig(figure_dir / "fig_5_3_efficiency_equity_frontier.png", dpi=300)
        plt.close()

    # Figure: group rates for the reference method.
    reference_group = group_metrics[
        group_metrics["method"].eq(reference) & group_metrics["eligible_for_fairness"]
    ]
    if not reference_group.empty:
        display = reference_group.groupby(["group_attribute", "group_value"], as_index=False)["service_rate"].mean()
        display["label"] = display["group_attribute"] + "=" + display["group_value"]
        display = display.sort_values("service_rate")
        plt.figure(figsize=(9.5, max(5.5, 0.28 * len(display))))
        y = np.arange(len(display))
        plt.barh(y, display["service_rate"])
        plt.yticks(y, display["label"], fontsize=8)
        plt.xlabel("Service rate")
        plt.title(f"Group-specific service rates: {reference}")
        plt.tight_layout()
        plt.savefig(figure_dir / "fig_5_3_group_service_rates.png", dpi=300)
        plt.close()

    write_excel(output_dir / "routing_equity_tables.xlsx", {
        "Paper table": summary,
        "Scenario metrics": scenario,
        "Group service": group_metrics,
        "Fairness indices": fairness,
        "Price of fairness": price,
        "Pareto": pareto,
        "Paired tests": tests,
        "Friedman": friedman,
    })
    write_json(output_dir / "routing_equity_report.json", {
        "script_version": "1.0.0",
        "alns_dir": str(alns_dir),
        "available_methods": available_methods,
        "source_files": source_records,
        "detected_columns": detected,
        "group_columns_found": groups,
        "minimum_group_cell": min_cell,
        "reference_method": reference,
        "efficiency_baseline": efficiency,
        "warnings": [
            "Worst-group results exclude cells below the configured minimum size.",
            "Sensitive attributes are used for ex-post auditing unless the paper documents a legal/policy basis for direct optimization.",
        ],
    })
    print(f"Routing/equity evaluation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
