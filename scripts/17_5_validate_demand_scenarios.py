from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

from demand.lightgbm_model import (
    load_model as load_lightgbm_artifact,
    predict as predict_lightgbm,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)

from analysis_common import (
    aligned_probabilities,
    categorical_distribution,
    find_column,
    hellinger_distance,
    js_divergence,
    load_yaml,
    read_table,
    resolve_path,
    total_variation,
    write_excel,
    write_json,
)


SCRIPT_VERSION = "1.0.2"

ZONE_ALIASES = ["zone_id", "origin_zone_id", "origin_zone", "origin_taz", "o_zone", "taz", "TAZ_1270"]
DEST_ZONE_ALIASES = ["destination_zone_id", "destination_zone", "dest_zone", "d_zone", "destination_taz"]
TARGET_ALIASES = ["demand_rate_per_1000_households", "target", "observed", "actual", "demand", "count"]
PREDICTION_ALIASES = ["prediction", "predicted", "y_pred", "demand_prediction", "predicted_demand", "lambda_hat"]
ACTIVITY_ALIASES = ["destination_activity", "activity_type", "destination_activity_type", "activity"]
DURATION_ALIASES = ["direct_duration_min", "duration_min", "trip_duration_min", "direct_time_min", "duration_minutes"]
TIME_BIN_ALIASES = ["time_bin", "departure_bin_minute", "departure_minute", "estimated_departure_bin"]
DAY_ALIASES = ["survey_day_of_week_code", "day_of_week_code", "day_code", "weekday"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LightGBM demand predictions and generated scenario fidelity.")
    parser.add_argument("--analysis-config", default="configs/results_analysis.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail instead of recording unavailable optional analyses.")
    return parser.parse_args()


def _resolve_configured_column(
    frame: pd.DataFrame,
    configured_column: str | None,
    aliases: list[str],
    label: str,
) -> str:
    """Resolve a configured column safely, falling back to known aliases.

    A stale analysis configuration may name a generic column such as
    ``zone_id`` even when the locked panel stores ``origin_zone_id``.
    The function preserves a valid configured name, otherwise detects the
    first available alias and emits a warning describing the fallback.
    """
    if configured_column:
        configured_column = str(configured_column)

        if configured_column in frame.columns:
            return configured_column

        detected = find_column(frame, aliases)

        if detected:
            warnings.warn(
                f"Configured {label} column {configured_column!r} was not found. "
                f"Using detected column {detected!r} instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return detected

        raise KeyError(
            f"Configured {label} column {configured_column!r} was not found, "
            f"and none of the aliases {aliases} exist. "
            f"Available columns: {frame.columns.tolist()}"
        )

    detected = find_column(frame, aliases, required=True)
    return str(detected)


def _load_or_predict(
    panel: pd.DataFrame,
    target_column: str,
    prediction_path: Path | None,
    model_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prediction_path and prediction_path.exists():
        predictions = read_table(prediction_path)
        prediction_column = find_column(predictions, PREDICTION_ALIASES, required=True)
        if len(predictions) != len(panel):
            join_columns = [
                column for column in [
                    find_column(panel, ZONE_ALIASES),
                    find_column(panel, TIME_BIN_ALIASES),
                    find_column(panel, DAY_ALIASES),
                ] if column and column in predictions.columns
            ]
            if not join_columns:
                raise ValueError("Prediction file row count differs from test panel and no common keys were detected.")
            result = panel.merge(predictions[join_columns + [prediction_column]], on=join_columns, how="left", validate="one_to_one")
        else:
            result = panel.copy()
            result["_prediction"] = pd.to_numeric(predictions[prediction_column], errors="coerce").to_numpy()
            prediction_column = "_prediction"
        result["prediction"] = pd.to_numeric(result[prediction_column], errors="coerce").clip(lower=1e-9)
        return result, {"source": str(prediction_path), "prediction_column": prediction_column}

    if not model_path.exists():
        raise FileNotFoundError(
            f"Neither prediction file nor model was found. Missing model: {model_path}"
        )
    # Load the complete training artifact so that the exact feature order,
    # categorical-feature list, and saved category levels are reused.
    # Calling the raw estimator directly would discard this metadata and can
    # trigger:
    #   ValueError: train and valid dataset categorical_feature do not match.
    artifact = load_lightgbm_artifact(model_path)

    prediction = np.asarray(
        predict_lightgbm(artifact, panel),
        dtype=float,
    )

    result = panel.copy()
    result["prediction"] = np.maximum(prediction, 1e-9)

    features = [str(column) for column in artifact["features"]]
    categorical_features = [
        str(column) for column in artifact["categorical_features"]
    ]

    return result, {
        "source": str(model_path),
        "prediction_source": "saved_lightgbm_artifact",
        "feature_count": len(features),
        "features": features,
        "categorical_features": categorical_features,
    }


def _point_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y) & np.isfinite(prediction)
    y = y[mask]
    prediction = np.maximum(prediction[mask], 1e-9)
    if len(y) == 0:
        raise ValueError("No finite prediction/target pairs.")
    mae = mean_absolute_error(y, prediction)
    rmse = math.sqrt(mean_squared_error(y, prediction))
    denominator = np.abs(y) + np.abs(prediction)
    smape = float(np.mean(np.where(denominator > 0, 2 * np.abs(prediction - y) / denominator, 0.0)))
    poisson = mean_poisson_deviance(np.maximum(y, 0), prediction)
    pearson = pearsonr(y, prediction).statistic if len(y) > 1 and np.std(y) > 0 and np.std(prediction) > 0 else float("nan")
    spearman = spearmanr(y, prediction).statistic if len(y) > 1 else float("nan")
    return {
        "n": len(y),
        "mae": float(mae),
        "rmse": float(rmse),
        "poisson_deviance": float(poisson),
        "smape": smape,
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else float("nan"),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "observed_mean": float(np.mean(y)),
        "predicted_mean": float(np.mean(prediction)),
        "mean_bias": float(np.mean(prediction - y)),
    }


def _high_demand_metrics(frame: pd.DataFrame, target: str, zone: str, quantile: float) -> tuple[dict[str, Any], pd.DataFrame]:
    zone_table = frame.groupby(zone, as_index=False).agg(
        observed=(target, "sum"),
        predicted=("prediction", "sum"),
    )
    threshold = float(zone_table["observed"].quantile(quantile))
    zone_table["observed_high"] = zone_table["observed"].ge(threshold)
    count_high = max(1, int(zone_table["observed_high"].sum()))
    predicted_top = set(zone_table.nlargest(count_high, "predicted")[zone].astype(str))
    observed_top = set(zone_table.loc[zone_table["observed_high"], zone].astype(str))
    intersection = len(predicted_top & observed_top)
    precision = intersection / len(predicted_top) if predicted_top else float("nan")
    recall = intersection / len(observed_top) if observed_top else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    zone_table["predicted_high"] = zone_table[zone].astype(str).isin(predicted_top)
    return {
        "quantile": quantile,
        "threshold": threshold,
        "observed_high_zones": len(observed_top),
        "predicted_high_zones": len(predicted_top),
        "intersection": intersection,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": intersection / len(predicted_top | observed_top) if predicted_top | observed_top else float("nan"),
    }, zone_table


def _categorical_fidelity(
    observed: pd.Series,
    generated: pd.Series,
    variable: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    p_series = categorical_distribution(observed)
    q_series = categorical_distribution(generated)
    p, q, categories = aligned_probabilities(p_series, q_series)
    detail = pd.DataFrame({"variable": variable, "category": categories, "heldout_share": p, "generated_share": q})
    detail["absolute_difference"] = np.abs(detail["generated_share"] - detail["heldout_share"])
    return {
        "variable": variable,
        "heldout_n": int(observed.notna().sum()),
        "generated_n": int(generated.notna().sum()),
        "categories": len(categories),
        "js_divergence": js_divergence(p, q),
        "total_variation": total_variation(p, q),
        "hellinger_distance": hellinger_distance(p, q),
        "top10_mass_heldout": float(np.sort(p)[-10:].sum()),
        "top10_mass_generated": float(np.sort(q)[-10:].sum()),
    }, detail


def _load_scenarios(forecast_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(forecast_dir.glob("forecast_scenario_*.parquet")) + sorted(forecast_dir.glob("scenario_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scenario files found under {forecast_dir}")
    frames = []
    records = []
    for index, path in enumerate(paths):
        frame = read_table(path).copy()
        frame["_scenario_id_file"] = index
        frames.append(frame)
        records.append({"path": str(path), "rows": len(frame)})
    return pd.concat(frames, ignore_index=True, sort=False), records


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    analysis_path = resolve_path(project_root, args.analysis_config)
    analysis = load_yaml(analysis_path)
    base_path = resolve_path(project_root, args.base_config or analysis.get("base_config", "configs/base.yaml"))
    base = load_yaml(base_path)
    paths = base.get("paths", {})
    settings = analysis.get("demand_validation", {})

    output_dir = resolve_path(project_root, args.output_dir or settings.get("output_dir", "outputs/paper_results/demand_validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    panel_path = resolve_path(project_root, settings.get("zone_panel_test") or paths["zone_panel_test"])
    trips_path = resolve_path(project_root, settings.get("trips_test") or paths.get("trips_test_routed", paths.get("trips_test")))
    forecast_dir = resolve_path(project_root, settings.get("forecast_dir") or paths["forecast_dir"])
    model_path = resolve_path(project_root, settings.get("lightgbm_model") or paths["lightgbm_model"])

    panel = read_table(panel_path)
    target = _resolve_configured_column(
        panel,
        settings.get("target_column"),
        TARGET_ALIASES,
        "target",
    )
    zone = _resolve_configured_column(
        panel,
        settings.get("zone_column"),
        ZONE_ALIASES,
        "zone",
    )
    prediction_path = resolve_path(project_root, args.prediction_file) if args.prediction_file else None
    evaluated, prediction_provenance = _load_or_predict(panel, target, prediction_path, model_path)

    y = pd.to_numeric(evaluated[target], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(evaluated["prediction"], errors="coerce").to_numpy(dtype=float)
    point_metrics = _point_metrics(y, prediction)
    high_metrics, zone_table = _high_demand_metrics(
        evaluated, target, zone, float(settings.get("high_demand_quantile", 0.80))
    )

    prediction_output = evaluated[
        [
            column
            for column in [
                zone,
                target,
                "prediction",
                find_column(evaluated, TIME_BIN_ALIASES),
                find_column(evaluated, DAY_ALIASES),
            ]
            if column
        ]
    ]
    try:
        prediction_output.to_parquet(
            output_dir / "lightgbm_test_predictions.parquet",
            index=False,
        )
    except (ImportError, ModuleNotFoundError):
        prediction_output.to_csv(
            output_dir / "lightgbm_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    zone_table.to_csv(output_dir / "zone_prediction_comparison.csv", index=False, encoding="utf-8-sig")

    generated, scenario_sources = _load_scenarios(forecast_dir)
    heldout = read_table(trips_path)
    target_time_bin = settings.get("target_time_bin", base.get("scenario", {}).get("departure_bin_minute"))
    target_day = settings.get("target_day_code", base.get("scenario", {}).get("day_of_week_code"))
    heldout_filtered = heldout.copy()
    heldout_time_col = find_column(heldout_filtered, TIME_BIN_ALIASES)
    heldout_day_col = find_column(heldout_filtered, DAY_ALIASES)
    if target_time_bin is not None and heldout_time_col:
        numeric = pd.to_numeric(heldout_filtered[heldout_time_col], errors="coerce")
        exact = heldout_filtered[numeric.eq(float(target_time_bin))]
        if not exact.empty:
            heldout_filtered = exact
    if target_day is not None and heldout_day_col:
        numeric = pd.to_numeric(heldout_filtered[heldout_day_col], errors="coerce")
        exact = heldout_filtered[numeric.eq(float(target_day))]
        if not exact.empty:
            heldout_filtered = exact

    fidelity_rows: list[dict[str, Any]] = []
    distribution_details: list[pd.DataFrame] = []

    variable_pairs = [
        ("origin_zone", ZONE_ALIASES, ZONE_ALIASES),
        ("destination_zone", DEST_ZONE_ALIASES, DEST_ZONE_ALIASES),
        ("destination_activity", ACTIVITY_ALIASES, ACTIVITY_ALIASES),
    ]
    origin_generated_col = find_column(generated, ZONE_ALIASES)
    heldout_origin_col = find_column(heldout_filtered, ZONE_ALIASES)
    if origin_generated_col and heldout_origin_col:
        row, detail = _categorical_fidelity(
            heldout_filtered[heldout_origin_col],
            generated[origin_generated_col],
            "origin_zone",
        )
        fidelity_rows.append(row)
        distribution_details.append(detail)
    elif origin_generated_col:
        # Fallback only when held-out trip origins are unavailable. The panel
        # target may be a normalized rate, so this fallback is descriptive.
        origin_observed = zone_table.set_index(zone)["observed"]
        expanded_observed = pd.Series(
            np.repeat(
                origin_observed.index.astype(str),
                np.maximum(np.rint(origin_observed.to_numpy()), 0).astype(int),
            )
        )
        if len(expanded_observed):
            row, detail = _categorical_fidelity(
                expanded_observed,
                generated[origin_generated_col],
                "origin_zone",
            )
            fidelity_rows.append(row)
            distribution_details.append(detail)

    for variable, observed_aliases, generated_aliases in variable_pairs[1:]:
        observed_col = find_column(heldout_filtered, observed_aliases)
        generated_col = find_column(generated, generated_aliases)
        if observed_col and generated_col:
            row, detail = _categorical_fidelity(heldout_filtered[observed_col], generated[generated_col], variable)
            fidelity_rows.append(row)
            distribution_details.append(detail)

    observed_origin = find_column(heldout_filtered, ZONE_ALIASES)
    observed_dest = find_column(heldout_filtered, DEST_ZONE_ALIASES)
    generated_origin = find_column(generated, ZONE_ALIASES)
    generated_dest = find_column(generated, DEST_ZONE_ALIASES)
    if observed_origin and observed_dest and generated_origin and generated_dest:
        observed_od = heldout_filtered[observed_origin].astype(str) + "→" + heldout_filtered[observed_dest].astype(str)
        generated_od = generated[generated_origin].astype(str) + "→" + generated[generated_dest].astype(str)
        row, detail = _categorical_fidelity(observed_od, generated_od, "od_pair")
        fidelity_rows.append(row)
        distribution_details.append(detail)

    numeric_rows: list[dict[str, Any]] = []
    observed_duration = find_column(heldout_filtered, DURATION_ALIASES)
    generated_duration = find_column(generated, DURATION_ALIASES)
    if observed_duration and generated_duration:
        observed_values = pd.to_numeric(heldout_filtered[observed_duration], errors="coerce").dropna()
        generated_values = pd.to_numeric(generated[generated_duration], errors="coerce").dropna()
        if len(observed_values) and len(generated_values):
            numeric_rows.append({
                "variable": "duration_min",
                "heldout_n": len(observed_values),
                "generated_n": len(generated_values),
                "heldout_mean": float(observed_values.mean()),
                "generated_mean": float(generated_values.mean()),
                "heldout_p90": float(observed_values.quantile(0.90)),
                "generated_p90": float(generated_values.quantile(0.90)),
                "wasserstein_distance": float(wasserstein_distance(observed_values, generated_values)),
                "ks_statistic": float(ks_2samp(observed_values, generated_values).statistic),
                "ks_p_value": float(ks_2samp(observed_values, generated_values).pvalue),
            })

    fidelity = pd.DataFrame(fidelity_rows)
    distribution_detail = pd.concat(distribution_details, ignore_index=True) if distribution_details else pd.DataFrame()
    numeric_fidelity = pd.DataFrame(numeric_rows)
    fidelity.to_csv(output_dir / "scenario_categorical_fidelity.csv", index=False, encoding="utf-8-sig")
    distribution_detail.to_csv(output_dir / "scenario_distribution_details.csv", index=False, encoding="utf-8-sig")
    numeric_fidelity.to_csv(output_dir / "scenario_numeric_fidelity.csv", index=False, encoding="utf-8-sig")

    # Figures
    finite = np.isfinite(y) & np.isfinite(prediction)
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(y[finite], prediction[finite], alpha=0.55)
    lower = float(min(y[finite].min(), prediction[finite].min()))
    upper = float(max(y[finite].max(), prediction[finite].max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Observed demand")
    plt.ylabel("Predicted demand")
    plt.title("Held-out zone-time demand calibration")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_5_1_observed_vs_predicted.png", dpi=300)
    plt.close()

    zone_plot = zone_table.sort_values("observed", ascending=False).head(int(settings.get("top_zone_plot_count", 30)))
    x = np.arange(len(zone_plot))
    width = 0.4
    plt.figure(figsize=(11, 5.8))
    plt.bar(x - width / 2, zone_plot["observed"], width=width, label="Observed")
    plt.bar(x + width / 2, zone_plot["predicted"], width=width, label="Predicted")
    plt.xticks(x, zone_plot[zone].astype(str), rotation=75)
    plt.ylabel("Aggregated demand")
    plt.xlabel("Zone")
    plt.title("Observed and predicted demand in highest-demand zones")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_5_1_zone_demand_comparison.png", dpi=300)
    plt.close()

    if observed_duration and generated_duration and numeric_rows:
        observed_values = pd.to_numeric(heldout_filtered[observed_duration], errors="coerce").dropna().sort_values()
        generated_values = pd.to_numeric(generated[generated_duration], errors="coerce").dropna().sort_values()
        plt.figure(figsize=(6.5, 5.5))
        plt.plot(observed_values, np.linspace(0, 1, len(observed_values), endpoint=True), label="Held-out")
        plt.plot(generated_values, np.linspace(0, 1, len(generated_values), endpoint=True), label="Generated")
        plt.xlabel("Duration (minutes)")
        plt.ylabel("Empirical cumulative probability")
        plt.title("Held-out and generated duration distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "fig_5_1_duration_cdf.png", dpi=300)
        plt.close()

    metric_table = pd.DataFrame([{"model": "LightGBM", **point_metrics, **{f"high_zone_{k}": v for k, v in high_metrics.items()}}])
    metric_table.to_csv(output_dir / "paper_table_demand_model.csv", index=False, encoding="utf-8-sig")
    write_excel(output_dir / "demand_validation_tables.xlsx", {
        "Demand model": metric_table,
        "Zone comparison": zone_table,
        "Categorical fidelity": fidelity,
        "Distribution detail": distribution_detail,
        "Numeric fidelity": numeric_fidelity,
    })

    report = {
        "script_version": SCRIPT_VERSION,
        "base_config": str(base_path),
        "analysis_config": str(analysis_path),
        "input": {
            "zone_panel_test": str(panel_path),
            "trips_test": str(trips_path),
            "forecast_dir": str(forecast_dir),
            "scenario_sources": scenario_sources,
        },
        "prediction_provenance": prediction_provenance,
        "point_metrics": point_metrics,
        "high_demand_zone_metrics": high_metrics,
        "heldout_rows_before_filter": len(heldout),
        "heldout_rows_after_filter": len(heldout_filtered),
        "warnings": [],
    }
    if len(heldout_filtered) < 30:
        report["warnings"].append(
            "The exact held-out 10:30/day subset is small. Treat distributional tests as descriptive and also report an adjacent-time-bin sensitivity."
        )
    write_json(output_dir / "demand_validation_report.json", report)
    print(f"Demand/scenario validation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())