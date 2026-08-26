from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, path_from_config
from evaluation.metrics import collect_result_files, group_equity_metrics, summarize_request_results

RUNNER_VERSION = "5.0-natural-demand-ablation"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Table 5.5.1 comparative ablation results."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--alns-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_results/chapter5_5",
    )
    return parser.parse_args()


def first_value(mapping, names):
    for name in names:
        if name in mapping and pd.notna(mapping[name]):
            return float(mapping[name])
    return np.nan


def worst_group_service_rate(group):
    equity = group_equity_metrics(group)
    if equity is None or len(equity) == 0:
        return np.nan

    equity = equity.copy()

    rate_col = None
    for c in ["service_rate", "served_rate", "group_service_rate", "sr"]:
        if c in equity.columns:
            rate_col = c
            break

    if rate_col is None:
        candidates = [
            c for c in equity.select_dtypes(include=["number"]).columns
            if "service" in c.lower() and "rate" in c.lower()
        ]
        if candidates:
            rate_col = candidates[0]

    if rate_col is None:
        raise KeyError(
            "Cannot identify service-rate column in group_equity_metrics(). "
            f"Columns: {equity.columns.tolist()}"
        )

    for label_col in ["group", "group_name", "population_group", "subgroup", "category"]:
        if label_col in equity.columns:
            labels = equity[label_col].astype(str).str.strip().str.lower()
            equity = equity.loc[
                ~labels.isin({"overall", "all", "all requests", "total", "reference"})
            ].copy()
            break

    values = pd.to_numeric(equity[rate_col], errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def summarize_run(group):
    s = summarize_request_results(group)
    return {
        "service_rate": first_value(s, ["service_rate"]),
        "gap_sr": first_value(
            s,
            ["gap_service_rate", "gap_origin_service_rate", "gap_closure_rate"],
        ),
        "pickup_mean_sec": first_value(
            s,
            ["pickup_delay_mean_sec", "pickup_mean_sec"],
        ),
        "door_mean_sec": first_value(
            s,
            ["door_to_door_mean_sec", "door_mean_sec"],
        ),
        "worst_group_sr": worst_group_service_rate(group),
    }


def main():
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

    all_results = collect_result_files(alns_dir)
    if all_results.empty:
        raise FileNotFoundError(f"No request_results.parquet found in {alns_dir}")

    required = {"method", "scenario_id", "solver_seed"}
    missing = required - set(all_results.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available: {all_results.columns.tolist()}"
        )

    available_methods = set(all_results["method"].dropna().astype(str).unique())
    selected_methods = [m for m in METHOD_ORDER if m in available_methods]

    if not selected_methods:
        raise ValueError(
            f"No expected methods found. Available methods: {sorted(available_methods)}"
        )

    filtered = all_results.loc[
        all_results["method"].isin(selected_methods)
    ].copy()

    run_rows = []
    for (method, scenario_id, solver_seed), group in filtered.groupby(
        ["method", "scenario_id", "solver_seed"], dropna=False
    ):
        run_rows.append({
            "method": method,
            "scenario_id": scenario_id,
            "solver_seed": solver_seed,
            **summarize_run(group),
        })

    run_summary = pd.DataFrame(run_rows)

    metric_cols = [
        "service_rate",
        "gap_sr",
        "pickup_mean_sec",
        "door_mean_sec",
        "worst_group_sr",
    ]

    scenario_summary = (
        run_summary
        .groupby(["method", "scenario_id"], as_index=False)[metric_cols]
        .mean()
    )

    method_summary = (
        scenario_summary
        .groupby("method", as_index=False)[metric_cols]
        .mean()
    )

    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    method_summary["Configuration"] = method_summary["method"].map(METHOD_LABELS)
    method_summary["_order"] = method_summary["method"].map(order_map)

    table = (
        method_summary
        .sort_values("_order")
        .rename(columns={
            "service_rate": "Service rate",
            "gap_sr": "Gap-origin SR",
            "pickup_mean_sec": "Pickup mean (s)",
            "door_mean_sec": "Door mean (s)",
            "worst_group_sr": "Worst-group SR",
        })
        [[
            "Configuration",
            "Service rate",
            "Gap-origin SR",
            "Pickup mean (s)",
            "Door mean (s)",
            "Worst-group SR",
        ]]
        .reset_index(drop=True)
    )

    csv_path = output_dir / "Table_5_5_1_comparative_ablation_results.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    formatted = table.copy()
    for c in ["Service rate", "Gap-origin SR", "Worst-group SR"]:
        formatted[c] = formatted[c].map(
            lambda x: "" if pd.isna(x) else f"{x:.3f}"
        )
    for c in ["Pickup mean (s)", "Door mean (s)"]:
        formatted[c] = formatted[c].map(
            lambda x: "" if pd.isna(x) else f"{x:.1f}"
        )

    tex_path = output_dir / "Table_5_5_1_comparative_ablation_results.tex"
    tex_path.write_text(
        formatted.to_latex(
            index=False,
            escape=True,
            column_format="lrrrrr",
            caption="Comparative ablation results for methodological components.",
            label="tab:comparative_ablation",
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Table 5.5.1 — Comparative ablation results for methodological components")
    print("=" * 88)
    print(f"Runner version : {RUNNER_VERSION}")
    print(f"ALNS directory : {alns_dir}")
    print(f"Methods        : {selected_methods}")
    print(f"Run rows       : {len(run_summary)}")
    print(f"Scenario rows  : {len(scenario_summary)}")
    print()
    print(table.to_string(index=False))
    print()
    print(f"CSV saved      : {csv_path}")
    print(f"LaTeX saved    : {tex_path}")


if __name__ == "__main__":
    main()
