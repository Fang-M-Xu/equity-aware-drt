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

RUNNER_VERSION = "3.1-add-pearson-r2"

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
    r2_score,
)
from scipy.stats import pearsonr

from config import load_config, path_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Table 5.1.1 from held-out LightGBM test predictions."
    )
    parser.add_argument(
        "--config",
        default="configs/base_competitive.yaml",
        help="Experiment configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ---------------------------------------------------------
    # 1. Load formal held-out test predictions
    # ---------------------------------------------------------
    metrics_dir = path_from_config(config, "metrics_dir")
    prediction_path = metrics_dir / "lightgbm_test_predictions.parquet"

    paper_results_dir = path_from_config(config, "paper_results_dir")

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{prediction_path}\n\n"
            "Run LightGBM test evaluation first."
        )

    df = pd.read_parquet(prediction_path)

    # ---------------------------------------------------------
    # 2. Experimental columns
    # ---------------------------------------------------------
    target_col = str(config["zone_panel"]["target_column"])
    prediction_col = "lightgbm"

    required_columns = [target_col, prediction_col]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # 3. Observed and predicted values
    # ---------------------------------------------------------
    y_true = pd.to_numeric(
        df[target_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    y_pred = pd.to_numeric(
        df[prediction_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError(
            "No valid held-out prediction pairs were found."
        )

    y_true = np.clip(y_true, 0.0, None)
    y_pred = np.clip(y_pred, 0.0, None)

    # ---------------------------------------------------------
    # 4. Metrics
    # ---------------------------------------------------------
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    poisson_deviance = mean_poisson_deviance(
        y_true,
        np.clip(y_pred, 1e-9, None),
    )

    # Pearson correlation
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(
            y_true,
            y_pred,
        )
    else:
        pearson_r = np.nan
        pearson_p = np.nan

    # Coefficient of determination
    r2 = r2_score(
        y_true,
        y_pred,
    )

    observed_mean = np.mean(y_true)
    predicted_mean = np.mean(y_pred)

    # ---------------------------------------------------------
    # 5. Table 5.1.1
    # ---------------------------------------------------------
    table = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "Poisson deviance",
                "Pearson r",
                "R²",
                "Observed mean",
                "Predicted mean",
            ],
            "Value": [
                mae,
                rmse,
                poisson_deviance,
                pearson_r,
                r2,
                observed_mean,
                predicted_mean,
            ],
        }
    )

    # Round only for publication output
    table["Value"] = table["Value"].round(3)

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------
    output_dir = paper_results_dir / "chapter5_1_model"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "Table_5_1_1_zone_level_demand_prediction_performance.csv"
    )

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 7. Console output
    # ---------------------------------------------------------
    print()
    print("=" * 65)
    print("Table 5.1.1 — Zone-level demand prediction performance")
    print("=" * 65)

    print(f"Source file     : {prediction_path}")
    print(f"Evaluation rows : {len(y_true):,}")
    print()

    print(table.to_string(index=False))

    print()
    print(
        f"Pearson p-value : "
        f"{pearson_p:.6g}"
        if np.isfinite(pearson_p)
        else "Pearson p-value : N/A"
    )

    print()
    print(f"Saved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()