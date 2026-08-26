from __future__ import annotations

import argparse
import json

import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

from config import load_config, path_from_config
from evaluation.metrics import collect_result_files, group_equity_metrics, summarize_request_results
from evaluation.statistics import bootstrap_mean_ci, paired_wilcoxon_table
from logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--proposed-method", default="proposed_full")
    args = parser.parse_args()
    config = load_config(args.config)
    log_path = path_from_config(config, "metrics_dir") / "16_evaluate_all.log"
    configure_logging(log_path)
    all_results = collect_result_files(path_from_config(config, "alns_dir"))
    if all_results.empty:
        raise FileNotFoundError("No ALNS request_results.parquet files were found.")

    run_columns = ["method", "scenario_id", "solver_seed"]
    run_summaries = []
    equity_tables = []
    for keys, group in all_results.groupby(run_columns, dropna=False):
        method, scenario_id, solver_seed = keys
        summary = summarize_request_results(group)
        run_summaries.append(
            {"method": method, "scenario_id": scenario_id, "solver_seed": solver_seed, **summary}
        )
        equity = group_equity_metrics(group)
        equity["method"] = method
        equity["scenario_id"] = scenario_id
        equity["solver_seed"] = solver_seed
        equity_tables.append(equity)
    run_summary = pd.DataFrame(run_summaries)
    equity_summary = pd.concat(equity_tables, ignore_index=True)

    # Average repeated solver seeds first, then conduct paired tests across scenarios.
    numeric_columns = [column for column in run_summary.select_dtypes(include=["number"]).columns if column not in {"scenario_id", "solver_seed"}]
    scenario_summary = (
        run_summary.groupby(["method", "scenario_id"], as_index=False)[numeric_columns]
        .mean()
        .drop(columns=["solver_seed"], errors="ignore")
    )
    methods = sorted(run_summary["method"].dropna().unique())
    baselines = [method for method in methods if method != args.proposed_method]
    test_metrics = [
        metric
        for metric in ["service_rate", "gap_closure_rate", "pickup_delay_mean_sec", "door_to_door_mean_sec"]
        if metric in scenario_summary.columns
    ]
    tests = paired_wilcoxon_table(scenario_summary, args.proposed_method, baselines, test_metrics)

    ci_rows = []
    for (method, metric), group in scenario_summary.melt(
        id_vars=["method", "scenario_id"], value_vars=test_metrics, var_name="metric", value_name="value"
    ).groupby(["method", "metric"]):
        lower, upper = bootstrap_mean_ci(group["value"], int(config["project"]["seed"]))
        ci_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": float(group["value"].mean()),
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )

    metrics_dir = path_from_config(config, "metrics_dir")
    statistics_dir = path_from_config(config, "statistics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    scenario_summary.to_csv(metrics_dir / "scenario_level_metrics.csv", index=False)
    equity_summary.to_csv(metrics_dir / "group_equity_metrics.csv", index=False)
    tests.to_csv(statistics_dir / "paired_wilcoxon_tests.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(statistics_dir / "bootstrap_confidence_intervals.csv", index=False)
    print(
        json.dumps(
            {
                "request_result_rows": len(all_results),
                "runs": len(run_summary),
                "scenario_method_rows": len(scenario_summary),
                "equity_rows": len(equity_summary),
            },
            indent=2,
        )
    )

    # The log is useful during execution, but is not part of the retained
    # scientific outputs. Remove it after a successful evaluation run.
    try:
        if log_path.exists():
            log_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
