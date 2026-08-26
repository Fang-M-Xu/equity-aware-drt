from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from config import load_config, path_from_config
from demand.lightgbm_model import (
    KEY_COLUMNS,
    build_baseline_predictions,
    evaluate_methods,
    predict,
    train_model,
)
from logging_utils import configure_logging
from random_utils import set_global_seed


def _subset_tables(
    table: pd.DataFrame,
    scenario_day: int,
    scenario_time_bin: int,
) -> dict[str, pd.DataFrame]:
    return {
        "all_cells": table,
        "target_time_all_days": table[table["time_bin"].eq(scenario_time_bin)].copy(),
        "scenario_day_time": table[
            table["time_bin"].eq(scenario_time_bin)
            & table["survey_day_of_week_code"].eq(scenario_day)
        ].copy(),
    }


def _evaluate_split(
    split_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    artifact: dict[str, Any],
    metrics_dir: Path,
    scenario_day: int,
    scenario_time_bin: int,
    top_k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = artifact["target_column"]
    prediction_table = build_baseline_predictions(train, evaluation, target)
    prediction_table["lightgbm"] = predict(artifact, evaluation)

    ordered_columns = KEY_COLUMNS + [target, "lightgbm", "zero", "global_mean", "zone_mean", "day_time_mean", "historical_cell"]
    prediction_table = prediction_table[ordered_columns]
    prediction_table.to_parquet(
        metrics_dir / f"lightgbm_{split_name}_predictions.parquet",
        index=False,
    )

    method_columns = [
        "lightgbm",
        "zero",
        "global_mean",
        "zone_mean",
        "day_time_mean",
        "historical_cell",
    ]
    metric_frames: list[pd.DataFrame] = []
    for subset_name, subset in _subset_tables(
        prediction_table,
        scenario_day=scenario_day,
        scenario_time_bin=scenario_time_bin,
    ).items():
        if subset.empty:
            continue
        metric_frames.append(
            evaluate_methods(
                subset,
                target_column=target,
                method_columns=method_columns,
                top_k_values=top_k_values,
                split=split_name,
                subset=subset_name,
            )
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(
        metrics_dir / f"lightgbm_{split_name}_method_comparison.csv",
        index=False,
    )
    return prediction_table, metrics


def _print_primary_summary(metrics: pd.DataFrame) -> None:
    preferred = metrics[
        metrics["subset"].eq("scenario_day_time")
        & metrics["method"].isin(["lightgbm", "historical_cell", "zero"])
    ]
    if preferred.empty:
        preferred = metrics[
            metrics["subset"].eq("all_cells")
            & metrics["method"].isin(["lightgbm", "historical_cell", "zero"])
        ]
    columns = [
        column
        for column in [
            "split",
            "subset",
            "method",
            "rows",
            "mae",
            "rmse",
            "poisson_deviance",
            "spearman",
            "top_5_capture",
            "top_10_capture",
        ]
        if column in preferred.columns
    ]
    print(preferred[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the locked model on the held-out test split. Use only after model choices are final.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_dir = path_from_config(config, "metrics_dir")
    configure_logging(metrics_dir / "06_train_lightgbm.log")
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    train = pd.read_parquet(path_from_config(config, "zone_panel_train"))
    validation = pd.read_parquet(path_from_config(config, "zone_panel_validation"))
    target = str(config["zone_panel"]["target_column"])

    artifact = train_model(
        train,
        validation,
        config["lightgbm"],
        target_column=target,
        output_path=path_from_config(config, "lightgbm_model"),
        seed=seed,
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = [int(value) for value in config["lightgbm"].get("top_k_values", [5, 10])]
    scenario_day = int(config["scenario"]["day_of_week_code"])
    scenario_time_bin = int(config["scenario"]["departure_bin_minute"])

    _, validation_metrics = _evaluate_split(
        "validation",
        train,
        validation,
        artifact,
        metrics_dir,
        scenario_day,
        scenario_time_bin,
        top_k_values,
    )

    all_metrics = [validation_metrics]
    evaluate_test = bool(args.evaluate_test or config["lightgbm"].get("evaluate_test_by_default", False))
    if evaluate_test:
        test = pd.read_parquet(path_from_config(config, "zone_panel_test"))
        validation_prediction = predict(artifact, validation)
        test_prediction = predict(artifact, test)
        _, test_metrics = _evaluate_split(
            "test",
            train,
            test,
            artifact,
            metrics_dir,
            scenario_day,
            scenario_time_bin,
            top_k_values,
        )
        all_metrics.append(test_metrics)

        feature_columns = artifact["features"]
        same_shape = validation[feature_columns].shape == test[feature_columns].shape
        identical_raw_features = bool(
            same_shape
            and validation[feature_columns].reset_index(drop=True).equals(
                test[feature_columns].reset_index(drop=True)
            )
        )
        prediction_diagnostic = {
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "raw_feature_matrices_identical": identical_raw_features,
            "predictions_identical": bool(
                len(validation_prediction) == len(test_prediction)
                and np.allclose(validation_prediction, test_prediction, rtol=0, atol=1e-12)
            ),
        }
        (metrics_dir / "lightgbm_validation_test_diagnostic.json").write_text(
            json.dumps(prediction_diagnostic, indent=2),
            encoding="utf-8",
        )

    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison.to_csv(metrics_dir / "lightgbm_all_method_comparison.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": artifact["features"],
            "importance_gain": artifact["model"].booster_.feature_importance(importance_type="gain"),
            "importance_split": artifact["model"].booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    feature_importance.to_csv(metrics_dir / "lightgbm_feature_importance.csv", index=False)

    metadata = {
        "best_iteration": artifact["best_iteration"],
        "target_column": target,
        "feature_count": len(artifact["features"]),
        "categorical_features": artifact["categorical_features"],
        "use_exposure_weights": artifact["use_exposure_weights"],
        "exposure_column": artifact["exposure_column"],
        "test_evaluated": evaluate_test,
        "scenario_day_of_week_code": scenario_day,
        "scenario_time_bin": scenario_time_bin,
    }
    (metrics_dir / "lightgbm_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    _print_primary_summary(comparison)
    print(json.dumps(metadata, indent=2))
    if not evaluate_test:
        print(
            "Test split was not evaluated. After all model choices are locked, run the same command with --evaluate-test."
        )


if __name__ == "__main__":
    main()