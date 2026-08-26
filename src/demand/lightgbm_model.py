from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    ndcg_score,
)


DEFAULT_CATEGORICAL_FEATURES = [
    "origin_zone_id",
    "survey_day_of_week_code",
]

EXCLUDED_FEATURES = {
    "request_count",
    "request_households",
    "demand_rate_per_1000_households",
    "origin_building_diversity",
    "mean_direct_time_sec",
    "mean_direct_distance_m",
    "day_household_count",
    "split_household_count",
}

KEY_COLUMNS = [
    "survey_day_of_week_code",
    "origin_zone_id",
    "time_bin",
]


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def choose_features(
    panel: pd.DataFrame,
    target_column: str,
    categorical_features: Iterable[str] | None = None,
) -> list[str]:
    categorical = set(categorical_features or DEFAULT_CATEGORICAL_FEATURES)
    excluded = EXCLUDED_FEATURES | {target_column}
    features: list[str] = []
    for column in panel.columns:
        if column in excluded or column.startswith("observed_"):
            continue
        if column in categorical or pd.api.types.is_numeric_dtype(panel[column]):
            features.append(column)
    if not features:
        raise ValueError("No usable LightGBM features were found.")
    return features


def infer_category_levels(
    train: pd.DataFrame,
    categorical_features: Iterable[str],
) -> dict[str, list[Any]]:
    levels: dict[str, list[Any]] = {}
    for column in categorical_features:
        if column not in train.columns:
            continue
        values = train[column].dropna().unique().tolist()
        try:
            values = sorted(values)
        except TypeError:
            values = sorted(values, key=lambda value: str(value))
        levels[column] = [_as_python_scalar(value) for value in values]
    return levels


def prepare_features(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: Iterable[str],
    category_levels: dict[str, list[Any]],
) -> pd.DataFrame:
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing LightGBM features: {missing}")

    categorical = set(categorical_features)
    output = frame[features].copy()
    for column in features:
        if column in categorical:
            if column not in category_levels:
                raise ValueError(f"No saved category levels are available for {column!r}.")
            output[column] = pd.Categorical(
                output[column],
                categories=category_levels[column],
            )
            if output[column].isna().any() and frame[column].notna().any():
                unknown_values = sorted(
                    {
                        _as_python_scalar(value)
                        for value in frame.loc[output[column].isna() & frame[column].notna(), column]
                    },
                    key=lambda value: str(value),
                )
                raise ValueError(
                    f"Feature {column!r} contains unseen category values: {unknown_values[:10]}"
                )
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    return output


def _safe_spearman(actual: np.ndarray, prediction: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(actual, prediction, nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def top_k_demand_capture(actual: np.ndarray, prediction: np.ndarray, k: int) -> float:
    if len(actual) == 0 or float(np.sum(actual)) <= 0:
        return float("nan")
    effective_k = min(int(k), len(actual))
    top_indices = np.argsort(prediction)[-effective_k:]
    return float(np.sum(actual[top_indices]) / np.sum(actual))


def _safe_ndcg(actual: np.ndarray, prediction: np.ndarray, k: int) -> float:
    if len(actual) < 2 or float(np.sum(actual)) <= 0:
        return float("nan")
    effective_k = min(int(k), len(actual))
    return float(
        ndcg_score(
            np.asarray(actual, dtype=float).reshape(1, -1),
            np.asarray(prediction, dtype=float).reshape(1, -1),
            k=effective_k,
        )
    )


def evaluate_predictions(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    top_k_values: Iterable[int] = (5, 10),
) -> dict[str, float | int]:
    y_true = np.asarray(pd.to_numeric(pd.Series(actual), errors="coerce").fillna(0.0), dtype=float)
    y_pred = np.asarray(pd.to_numeric(pd.Series(prediction), errors="coerce").fillna(0.0), dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and prediction arrays have different lengths.")
    if len(y_true) == 0:
        return {"rows": 0}

    positive_mask = y_true > 0
    denominator = float(np.sum(np.abs(y_true)))
    poisson_prediction = np.clip(y_pred, 1e-9, None)

    metrics: dict[str, float | int] = {
        "rows": int(len(y_true)),
        "positive_rows": int(positive_mask.sum()),
        "zero_share": float((~positive_mask).mean()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "poisson_deviance": float(mean_poisson_deviance(y_true, poisson_prediction)),
        "wape": float(np.sum(np.abs(y_true - y_pred)) / denominator) if denominator > 0 else float("nan"),
        "mean_actual": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
        "positive_only_mae": (
            float(mean_absolute_error(y_true[positive_mask], y_pred[positive_mask]))
            if positive_mask.any()
            else float("nan")
        ),
    }
    for k in top_k_values:
        metrics[f"top_{int(k)}_capture"] = top_k_demand_capture(y_true, y_pred, int(k))
        metrics[f"ndcg_at_{int(k)}"] = _safe_ndcg(y_true, y_pred, int(k))
    return metrics


def build_baseline_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    required = KEY_COLUMNS + [target_column]
    for label, frame in [("train", train), ("evaluation", evaluation)]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} panel is missing baseline columns: {missing}")

    result = evaluation[KEY_COLUMNS + [target_column]].copy()
    train_target = pd.to_numeric(train[target_column], errors="coerce").fillna(0.0)
    global_mean = float(train_target.mean())

    result["zero"] = 0.0
    result["global_mean"] = global_mean

    zone_mean = (
        train.assign(_target=train_target)
        .groupby("origin_zone_id", as_index=False)["_target"]
        .mean()
        .rename(columns={"_target": "zone_mean"})
    )
    result = result.merge(zone_mean, on="origin_zone_id", how="left", validate="m:1")
    result["zone_mean"] = result["zone_mean"].fillna(global_mean)

    day_time_mean = (
        train.assign(_target=train_target)
        .groupby(["survey_day_of_week_code", "time_bin"], as_index=False)["_target"]
        .mean()
        .rename(columns={"_target": "day_time_mean"})
    )
    result = result.merge(
        day_time_mean,
        on=["survey_day_of_week_code", "time_bin"],
        how="left",
        validate="m:1",
    )
    result["day_time_mean"] = result["day_time_mean"].fillna(global_mean)

    historical_cell = train[KEY_COLUMNS].copy()
    historical_cell["historical_cell"] = train_target.to_numpy()
    if historical_cell.duplicated(KEY_COLUMNS).any():
        raise ValueError("Training panel contains duplicate day-zone-time cells.")
    result = result.merge(historical_cell, on=KEY_COLUMNS, how="left", validate="m:1")
    result["historical_cell"] = result["historical_cell"].fillna(global_mean)
    return result


def evaluate_methods(
    prediction_table: pd.DataFrame,
    target_column: str,
    method_columns: Iterable[str],
    top_k_values: Iterable[int],
    split: str,
    subset: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in method_columns:
        metrics = evaluate_predictions(
            prediction_table[target_column],
            prediction_table[method],
            top_k_values=top_k_values,
        )
        rows.append({"split": split, "subset": subset, "method": method, **metrics})
    return pd.DataFrame(rows)


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, Any],
    target_column: str,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    categorical_features = [
        column
        for column in config.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES)
        if column in train.columns
    ]
    features = choose_features(train, target_column, categorical_features)
    category_levels = infer_category_levels(train, categorical_features)

    X_train = prepare_features(train, features, categorical_features, category_levels)
    X_validation = prepare_features(validation, features, categorical_features, category_levels)
    y_train = pd.to_numeric(train[target_column], errors="coerce").fillna(0).astype(float)
    y_validation = pd.to_numeric(validation[target_column], errors="coerce").fillna(0).astype(float)

    sample_weight = None
    validation_weight = None
    if bool(config.get("use_exposure_weights", True)):
        exposure_column = str(config.get("exposure_column", "day_household_count"))
        if exposure_column not in train.columns or exposure_column not in validation.columns:
            raise ValueError(
                f"Exposure weighting is enabled but {exposure_column!r} is absent from a panel."
            )
        sample_weight = pd.to_numeric(train[exposure_column], errors="coerce").fillna(1.0)
        validation_weight = pd.to_numeric(validation[exposure_column], errors="coerce").fillna(1.0)
        sample_weight = np.clip(sample_weight.to_numpy(dtype=float), 1.0, None)
        validation_weight = np.clip(validation_weight.to_numpy(dtype=float), 1.0, None)
        sample_weight = sample_weight / np.mean(sample_weight)
        validation_weight = validation_weight / np.mean(validation_weight)

    model = LGBMRegressor(
        objective=config["objective"],
        num_leaves=int(config["num_leaves"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    eval_metrics = list(config.get("eval_metrics", ["poisson", "l1"]))
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_validation, y_validation)],
        eval_sample_weight=[validation_weight] if validation_weight is not None else None,
        eval_metric=eval_metrics,
        categorical_feature=categorical_features,
        callbacks=[
            early_stopping(int(config["early_stopping_rounds"]), verbose=True),
            log_evaluation(period=int(config.get("log_period", 50))),
        ],
    )

    prediction = np.clip(model.predict(X_validation), 0, None)
    top_k_values = [int(value) for value in config.get("top_k_values", [5, 10])]
    metrics = evaluate_predictions(y_validation, prediction, top_k_values=top_k_values)

    artifact = {
        "model": model,
        "features": features,
        "categorical_features": categorical_features,
        "category_levels": category_levels,
        "target_column": target_column,
        "metrics": metrics,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "use_exposure_weights": bool(config.get("use_exposure_weights", True)),
        "exposure_column": str(config.get("exposure_column", "day_household_count")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def load_model(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {
        "model",
        "features",
        "categorical_features",
        "category_levels",
        "target_column",
    }
    if not required.issubset(artifact):
        raise ValueError(f"Invalid LightGBM artifact: missing {required - set(artifact)}")
    return artifact


def predict(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    X = prepare_features(
        frame,
        artifact["features"],
        artifact["categorical_features"],
        artifact["category_levels"],
    )
    return np.clip(artifact["model"].predict(X), 0, None)