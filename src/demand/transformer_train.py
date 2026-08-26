from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .transformer_data import (
    PAD_TOKEN,
    UNK_TOKEN,
    TripSequenceDataset,
    Vocabulary,
    VocabularyBundle,
    collate_trip_sequences,
)
from .transformer_model import ActivityTripTransformer, TransformerDimensions


_METADATA_KEYS = {
    "sample_id",
    "survey_day_id",
    "person_id",
    "household_id",
}


def _move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _top_k_accuracy_from_rankings(
    targets: np.ndarray,
    rankings: np.ndarray,
    k: int,
) -> float:
    if len(targets) == 0:
        return float("nan")
    if rankings.ndim != 2:
        raise ValueError("rankings must be a two-dimensional array.")
    effective_k = min(max(int(k), 1), rankings.shape[1])
    return float(
        np.mean(
            [target in row[:effective_k] for target, row in zip(targets, rankings)]
        )
    )


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    rankings: np.ndarray | None,
    top_k_values: Iterable[int],
) -> dict[str, float | int]:
    if len(targets) == 0:
        return {
            "samples": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
        }

    metrics: dict[str, float | int] = {
        "samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "observed_target_classes": int(len(np.unique(targets))),
        "predicted_classes": int(len(np.unique(predictions))),
    }
    if rankings is not None:
        for k in sorted({int(value) for value in top_k_values if int(value) > 0}):
            metrics[f"top_{k}_accuracy"] = _top_k_accuracy_from_rankings(
                targets, rankings, k
            )
    return metrics


def _duration_minutes(vocabulary: Vocabulary, indices: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for index in indices:
        token = vocabulary.decode(int(index))
        try:
            values.append(float(token))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _duration_ordinal_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    vocabulary: Vocabulary,
    duration_bin_minutes: int,
) -> dict[str, float]:
    target_minutes = _duration_minutes(vocabulary, targets)
    prediction_minutes = _duration_minutes(vocabulary, predictions)
    valid = np.isfinite(target_minutes) & np.isfinite(prediction_minutes)
    if not valid.any():
        return {
            "duration_mae_minutes": float("nan"),
            "duration_within_one_bin_accuracy": float("nan"),
            "duration_within_two_bins_accuracy": float("nan"),
        }
    absolute_error = np.abs(target_minutes[valid] - prediction_minutes[valid])
    return {
        "duration_mae_minutes": float(absolute_error.mean()),
        "duration_within_one_bin_accuracy": float(
            np.mean(absolute_error <= float(duration_bin_minutes) + 1e-9)
        ),
        "duration_within_two_bins_accuracy": float(
            np.mean(absolute_error <= 2.0 * float(duration_bin_minutes) + 1e-9)
        ),
    }


def _loss_weights(config: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(config.get("zone_loss_weight", 1.0)),
        float(config.get("activity_loss_weight", 0.5)),
        float(config.get("duration_loss_weight", 0.5)),
    )


def _run_epoch(
    model: ActivityTripTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    vocabularies: VocabularyBundle | None = None,
    collect_predictions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    zone_weight, activity_weight, duration_weight = _loss_weights(config)
    top_k_values = config.get("metric_top_k_values", [3])

    loss_total = 0.0
    zone_loss_total = 0.0
    activity_loss_total = 0.0
    duration_loss_total = 0.0
    sample_count = 0

    zone_targets: list[np.ndarray] = []
    zone_predictions: list[np.ndarray] = []
    zone_rankings: list[np.ndarray] = []
    zone_confidences: list[np.ndarray] = []

    activity_targets: list[np.ndarray] = []
    activity_predictions: list[np.ndarray] = []
    activity_rankings: list[np.ndarray] = []
    activity_confidences: list[np.ndarray] = []

    duration_targets: list[np.ndarray] = []
    duration_predictions: list[np.ndarray] = []
    duration_rankings: list[np.ndarray] = []
    duration_confidences: list[np.ndarray] = []

    metadata_rows: list[dict[str, Any]] = []

    for batch in loader:
        batch = _move_tensors_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        output = model(
            batch["origin_zones"],
            batch["origin_activities"],
            batch["departures"],
            batch["profiles"],
            batch["padding_mask"],
        )
        zone_loss = criterion(
            output["destination_zone_logits"], batch["destination_zone"]
        )
        activity_loss = criterion(
            output["destination_activity_logits"], batch["destination_activity"]
        )
        duration_loss = criterion(output["duration_logits"], batch["duration"])
        loss = (
            zone_weight * zone_loss
            + activity_weight * activity_loss
            + duration_weight * duration_loss
        )

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()

        batch_count = int(batch["destination_zone"].shape[0])
        sample_count += batch_count
        loss_total += float(loss.detach().cpu()) * batch_count
        zone_loss_total += float(zone_loss.detach().cpu()) * batch_count
        activity_loss_total += float(activity_loss.detach().cpu()) * batch_count
        duration_loss_total += float(duration_loss.detach().cpu()) * batch_count

        zone_probabilities = torch.softmax(output["destination_zone_logits"], dim=1)
        activity_probabilities = torch.softmax(
            output["destination_activity_logits"], dim=1
        )
        duration_probabilities = torch.softmax(output["duration_logits"], dim=1)

        zone_order = torch.argsort(zone_probabilities, dim=1, descending=True)
        activity_order = torch.argsort(
            activity_probabilities, dim=1, descending=True
        )
        duration_order = torch.argsort(duration_probabilities, dim=1, descending=True)

        zone_targets.append(batch["destination_zone"].detach().cpu().numpy())
        zone_predictions.append(zone_order[:, 0].detach().cpu().numpy())
        zone_rankings.append(zone_order.detach().cpu().numpy())
        zone_confidences.append(
            zone_probabilities.max(dim=1).values.detach().cpu().numpy()
        )

        activity_targets.append(batch["destination_activity"].detach().cpu().numpy())
        activity_predictions.append(activity_order[:, 0].detach().cpu().numpy())
        activity_rankings.append(activity_order.detach().cpu().numpy())
        activity_confidences.append(
            activity_probabilities.max(dim=1).values.detach().cpu().numpy()
        )

        duration_targets.append(batch["duration"].detach().cpu().numpy())
        duration_predictions.append(duration_order[:, 0].detach().cpu().numpy())
        duration_rankings.append(duration_order.detach().cpu().numpy())
        duration_confidences.append(
            duration_probabilities.max(dim=1).values.detach().cpu().numpy()
        )

        if collect_predictions:
            for row_index in range(batch_count):
                metadata_rows.append(
                    {
                        "sample_id": batch["sample_id"][row_index],
                        "survey_day_id": batch["survey_day_id"][row_index],
                        "person_id": batch["person_id"][row_index],
                        "household_id": batch["household_id"][row_index],
                        "stop_seq": int(batch["stop_seq"][row_index].detach().cpu()),
                        "sequence_length": int(
                            batch["sequence_length"][row_index].detach().cpu()
                        ),
                        "last_origin_zone_index": int(
                            batch["last_origin_zone"][row_index].detach().cpu()
                        ),
                        "last_origin_activity_index": int(
                            batch["last_origin_activity"][row_index].detach().cpu()
                        ),
                        "last_departure_index": int(
                            batch["last_departure"][row_index].detach().cpu()
                        ),
                    }
                )

    denominator = max(sample_count, 1)
    zone_target_array = np.concatenate(zone_targets) if zone_targets else np.array([])
    zone_prediction_array = (
        np.concatenate(zone_predictions) if zone_predictions else np.array([])
    )
    zone_ranking_array = (
        np.concatenate(zone_rankings) if zone_rankings else np.empty((0, 0), dtype=int)
    )
    zone_confidence_array = (
        np.concatenate(zone_confidences) if zone_confidences else np.array([])
    )

    activity_target_array = (
        np.concatenate(activity_targets) if activity_targets else np.array([])
    )
    activity_prediction_array = (
        np.concatenate(activity_predictions) if activity_predictions else np.array([])
    )
    activity_ranking_array = (
        np.concatenate(activity_rankings)
        if activity_rankings
        else np.empty((0, 0), dtype=int)
    )
    activity_confidence_array = (
        np.concatenate(activity_confidences) if activity_confidences else np.array([])
    )

    duration_target_array = (
        np.concatenate(duration_targets) if duration_targets else np.array([])
    )
    duration_prediction_array = (
        np.concatenate(duration_predictions) if duration_predictions else np.array([])
    )
    duration_ranking_array = (
        np.concatenate(duration_rankings)
        if duration_rankings
        else np.empty((0, 0), dtype=int)
    )
    duration_confidence_array = (
        np.concatenate(duration_confidences) if duration_confidences else np.array([])
    )

    metrics: dict[str, Any] = {
        "loss": loss_total / denominator,
        "zone_loss": zone_loss_total / denominator,
        "activity_loss": activity_loss_total / denominator,
        "duration_loss": duration_loss_total / denominator,
        "zone": _classification_metrics(
            zone_target_array,
            zone_prediction_array,
            zone_ranking_array,
            top_k_values,
        ),
        "activity": _classification_metrics(
            activity_target_array,
            activity_prediction_array,
            activity_ranking_array,
            top_k_values,
        ),
        "duration": _classification_metrics(
            duration_target_array,
            duration_prediction_array,
            duration_ranking_array,
            top_k_values,
        ),
    }
    if vocabularies is not None:
        metrics["duration"].update(
            _duration_ordinal_metrics(
                duration_target_array,
                duration_prediction_array,
                vocabularies.duration,
                int(config["duration_bin_minutes"]),
            )
        )

    predictions_frame: pd.DataFrame | None = None
    if collect_predictions:
        predictions_frame = pd.DataFrame(metadata_rows)
        if len(predictions_frame) != len(zone_target_array):
            raise RuntimeError("Prediction metadata and target arrays have different lengths.")

        predictions_frame["zone_target_index"] = zone_target_array
        predictions_frame["zone_prediction_index"] = zone_prediction_array
        predictions_frame["zone_prediction_confidence"] = zone_confidence_array
        predictions_frame["activity_target_index"] = activity_target_array
        predictions_frame["activity_prediction_index"] = activity_prediction_array
        predictions_frame["activity_prediction_confidence"] = activity_confidence_array
        predictions_frame["duration_target_index"] = duration_target_array
        predictions_frame["duration_prediction_index"] = duration_prediction_array
        predictions_frame["duration_prediction_confidence"] = duration_confidence_array

        if vocabularies is not None:
            predictions_frame["last_origin_zone"] = predictions_frame[
                "last_origin_zone_index"
            ].map(vocabularies.zone.decode)
            predictions_frame["last_origin_activity"] = predictions_frame[
                "last_origin_activity_index"
            ].map(vocabularies.activity.decode)
            predictions_frame["last_departure_bin"] = predictions_frame[
                "last_departure_index"
            ].map(vocabularies.departure.decode)
            predictions_frame["zone_target"] = predictions_frame[
                "zone_target_index"
            ].map(vocabularies.zone.decode)
            predictions_frame["zone_prediction"] = predictions_frame[
                "zone_prediction_index"
            ].map(vocabularies.zone.decode)
            predictions_frame["activity_target"] = predictions_frame[
                "activity_target_index"
            ].map(vocabularies.activity.decode)
            predictions_frame["activity_prediction"] = predictions_frame[
                "activity_prediction_index"
            ].map(vocabularies.activity.decode)
            predictions_frame["duration_target"] = predictions_frame[
                "duration_target_index"
            ].map(vocabularies.duration.decode)
            predictions_frame["duration_prediction"] = predictions_frame[
                "duration_prediction_index"
            ].map(vocabularies.duration.decode)

            maximum_k = max([int(value) for value in top_k_values] + [3])
            predictions_frame["zone_top_predictions"] = [
                "|".join(
                    vocabularies.zone.decode(int(value))
                    for value in row[: min(maximum_k, len(row))]
                )
                for row in zone_ranking_array
            ]
            predictions_frame["activity_top_predictions"] = [
                "|".join(
                    vocabularies.activity.decode(int(value))
                    for value in row[: min(maximum_k, len(row))]
                )
                for row in activity_ranking_array
            ]
            predictions_frame["duration_top_predictions"] = [
                "|".join(
                    vocabularies.duration.decode(int(value))
                    for value in row[: min(maximum_k, len(row))]
                )
                for row in duration_ranking_array
            ]

    return metrics, predictions_frame


def _counter_ranking(counter: Counter[int], fallback: list[int]) -> list[int]:
    ranking = [
        int(label)
        for label, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    ranking.extend(label for label in fallback if label not in ranking)
    return ranking


def _global_ranking(samples: list[dict[str, Any]], target: str) -> list[int]:
    return _counter_ranking(Counter(int(sample[target]) for sample in samples), [])


def fit_baseline_artifact(train_dataset: TripSequenceDataset) -> dict[str, Any]:
    samples = train_dataset.samples
    if not samples:
        raise ValueError("Cannot fit Transformer baselines on an empty training dataset.")

    global_zone = _global_ranking(samples, "destination_zone")
    global_activity = _global_ranking(samples, "destination_activity")
    global_duration = _global_ranking(samples, "duration")

    zone_by_origin: dict[int, Counter[int]] = defaultdict(Counter)
    zone_by_origin_time: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    activity_by_origin_activity: dict[int, Counter[int]] = defaultdict(Counter)
    duration_by_origin_time: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    for sample in samples:
        origin_zone = int(sample["origin_zones"][-1])
        origin_activity = int(sample["origin_activities"][-1])
        departure = int(sample["departures"][-1])
        destination_zone = int(sample["destination_zone"])
        destination_activity = int(sample["destination_activity"])
        duration = int(sample["duration"])

        zone_by_origin[origin_zone][destination_zone] += 1
        zone_by_origin_time[(origin_zone, departure)][destination_zone] += 1
        activity_by_origin_activity[origin_activity][destination_activity] += 1
        duration_by_origin_time[(origin_zone, departure)][duration] += 1

    return {
        "global_zone": global_zone,
        "global_activity": global_activity,
        "global_duration": global_duration,
        "zone_by_origin": {
            str(key): _counter_ranking(value, global_zone)
            for key, value in zone_by_origin.items()
        },
        "zone_by_origin_time": {
            f"{key[0]}|{key[1]}": _counter_ranking(value, global_zone)
            for key, value in zone_by_origin_time.items()
        },
        "activity_by_origin_activity": {
            str(key): _counter_ranking(value, global_activity)
            for key, value in activity_by_origin_activity.items()
        },
        "duration_by_origin_time": {
            f"{key[0]}|{key[1]}": _counter_ranking(value, global_duration)
            for key, value in duration_by_origin_time.items()
        },
    }


def _baseline_rankings(
    dataset: TripSequenceDataset,
    artifact: dict[str, Any],
    target: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    rankings: list[list[int]] = []
    targets: list[int] = []

    if target == "zone":
        global_ranking = [int(value) for value in artifact["global_zone"]]
    elif target == "activity":
        global_ranking = [int(value) for value in artifact["global_activity"]]
    elif target == "duration":
        global_ranking = [int(value) for value in artifact["global_duration"]]
    else:
        raise ValueError(f"Unknown target: {target}")

    for sample in dataset.samples:
        origin_zone = int(sample["origin_zones"][-1])
        origin_activity = int(sample["origin_activities"][-1])
        departure = int(sample["departures"][-1])

        if target == "zone":
            targets.append(int(sample["destination_zone"]))
            if method == "global_majority":
                ranking = global_ranking
            elif method == "origin_zone_majority":
                ranking = artifact["zone_by_origin"].get(
                    str(origin_zone), global_ranking
                )
            elif method == "historical_transition":
                ranking = artifact["zone_by_origin_time"].get(
                    f"{origin_zone}|{departure}",
                    artifact["zone_by_origin"].get(str(origin_zone), global_ranking),
                )
            else:
                raise ValueError(f"Unknown zone baseline: {method}")
        elif target == "activity":
            targets.append(int(sample["destination_activity"]))
            if method == "global_majority":
                ranking = global_ranking
            elif method == "origin_activity_majority":
                ranking = artifact["activity_by_origin_activity"].get(
                    str(origin_activity), global_ranking
                )
            else:
                raise ValueError(f"Unknown activity baseline: {method}")
        else:
            targets.append(int(sample["duration"]))
            if method == "global_majority":
                ranking = global_ranking
            elif method == "origin_zone_time_majority":
                ranking = artifact["duration_by_origin_time"].get(
                    f"{origin_zone}|{departure}", global_ranking
                )
            else:
                raise ValueError(f"Unknown duration baseline: {method}")

        rankings.append([int(value) for value in ranking])

    maximum_length = max((len(row) for row in rankings), default=0)
    padded = np.full((len(rankings), maximum_length), -1, dtype=int)
    for index, row in enumerate(rankings):
        padded[index, : len(row)] = row
    return np.asarray(targets, dtype=int), padded


def evaluate_baselines(
    dataset: TripSequenceDataset,
    artifact: dict[str, Any],
    vocabularies: VocabularyBundle,
    config: dict[str, Any],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_k_values = config.get("metric_top_k_values", [3])

    baseline_definitions = {
        "zone": ["global_majority", "origin_zone_majority", "historical_transition"],
        "activity": ["global_majority", "origin_activity_majority"],
        "duration": ["global_majority", "origin_zone_time_majority"],
    }

    for target, methods in baseline_definitions.items():
        for method in methods:
            targets, rankings = _baseline_rankings(dataset, artifact, target, method)
            predictions = rankings[:, 0]
            metrics = _classification_metrics(
                targets, predictions, rankings, top_k_values
            )
            row: dict[str, Any] = {
                "split": split,
                "target": target,
                "method": method,
                **metrics,
            }
            if target == "duration":
                row.update(
                    _duration_ordinal_metrics(
                        targets,
                        predictions,
                        vocabularies.duration,
                        int(config["duration_bin_minutes"]),
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def transformer_method_comparison(
    metrics: dict[str, Any],
    baseline_table: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in ["zone", "activity", "duration"]:
        row = {
            "split": split,
            "target": target,
            "method": "transformer",
            **metrics[target],
        }
        rows.append(row)
    transformer_table = pd.DataFrame(rows)
    all_columns = sorted(set(transformer_table.columns) | set(baseline_table.columns))
    return pd.concat(
        [
            transformer_table.reindex(columns=all_columns),
            baseline_table.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )


def dataset_class_distribution(
    dataset: TripSequenceDataset,
    vocabularies: VocabularyBundle,
    split: str,
) -> pd.DataFrame:
    definitions = {
        "zone": ("destination_zone", vocabularies.zone),
        "activity": ("destination_activity", vocabularies.activity),
        "duration": ("duration", vocabularies.duration),
    }
    rows: list[dict[str, Any]] = []
    for target_name, (sample_key, vocabulary) in definitions.items():
        counts = Counter(int(sample[sample_key]) for sample in dataset.samples)
        total = max(sum(counts.values()), 1)
        for class_index, count in sorted(counts.items()):
            rows.append(
                {
                    "split": split,
                    "target": target_name,
                    "class_index": int(class_index),
                    "class_label": vocabulary.decode(class_index),
                    "count": int(count),
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def dataset_diagnostics(dataset: TripSequenceDataset, split: str) -> dict[str, Any]:
    lengths = np.asarray(
        [len(sample["origin_zones"]) for sample in dataset.samples], dtype=float
    )
    return {
        "split": split,
        "samples": int(len(dataset)),
        "survey_days": int(len({sample["survey_day_id"] for sample in dataset.samples})),
        "households": int(len({sample["household_id"] for sample in dataset.samples})),
        "sequence_length_mean": float(lengths.mean()) if len(lengths) else float("nan"),
        "sequence_length_p50": float(np.median(lengths)) if len(lengths) else float("nan"),
        "sequence_length_p90": float(np.quantile(lengths, 0.9)) if len(lengths) else float("nan"),
        "sequence_length_max": int(lengths.max()) if len(lengths) else 0,
    }


def confusion_long_form(
    predictions: pd.DataFrame,
    target: str,
    vocabulary: Vocabulary,
    split: str,
) -> pd.DataFrame:
    target_column = f"{target}_target_index"
    prediction_column = f"{target}_prediction_index"
    labels = sorted(
        set(predictions[target_column].astype(int))
        | set(predictions[prediction_column].astype(int))
    )
    matrix = confusion_matrix(
        predictions[target_column].astype(int),
        predictions[prediction_column].astype(int),
        labels=labels,
    )
    rows: list[dict[str, Any]] = []
    for actual_position, actual_index in enumerate(labels):
        for prediction_position, prediction_index in enumerate(labels):
            count = int(matrix[actual_position, prediction_position])
            if count == 0:
                continue
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "actual_index": int(actual_index),
                    "actual_label": vocabulary.decode(actual_index),
                    "predicted_index": int(prediction_index),
                    "predicted_label": vocabulary.decode(prediction_index),
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def train_transformer(
    train_dataset: TripSequenceDataset,
    validation_dataset: TripSequenceDataset,
    vocabularies: VocabularyBundle,
    profile_columns: list[str],
    config: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimensions = TransformerDimensions(
        num_zones=len(vocabularies.zone),
        num_activities=len(vocabularies.activity),
        num_departure_bins=len(vocabularies.departure),
        num_duration_bins=len(vocabularies.duration),
        profile_dim=len(profile_columns),
    )
    model = ActivityTripTransformer(
        dimensions=dimensions,
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_sequence_length=int(config["max_sequence_length"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "collate_fn": collate_trip_sequences,
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": bool(config.get("pin_memory", torch.cuda.is_available())),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics, _ = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            config,
            vocabularies=vocabularies,
            collect_predictions=False,
        )
        with torch.no_grad():
            validation_metrics, _ = _run_epoch(
                model,
                validation_loader,
                device,
                None,
                config,
                vocabularies=vocabularies,
                collect_predictions=False,
            )

        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_zone_acc={validation_metrics['zone']['accuracy']:.4f} "
            f"val_activity_macro_f1={validation_metrics['activity']['macro_f1']:.4f} "
            f"val_duration_within1={validation_metrics['duration'].get('duration_within_one_bin_accuracy', float('nan')):.4f}"
        )

        if validation_metrics["loss"] < best_loss - float(
            config.get("minimum_improvement", 1e-6)
        ):
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dimensions": dimensions.__dict__,
                    "model_config": {
                        key: config[key]
                        for key in [
                            "d_model",
                            "n_heads",
                            "n_layers",
                            "dim_feedforward",
                            "dropout",
                            "max_sequence_length",
                        ]
                    },
                    "training_config": {
                        "zone_loss_weight": _loss_weights(config)[0],
                        "activity_loss_weight": _loss_weights(config)[1],
                        "duration_loss_weight": _loss_weights(config)[2],
                        "label_smoothing": float(config.get("label_smoothing", 0.0)),
                        "duration_bin_minutes": int(config["duration_bin_minutes"]),
                    },
                    "profile_columns": profile_columns,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                },
                model_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break

    result = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "loss_weights": {
            "zone": _loss_weights(config)[0],
            "activity": _loss_weights(config)[1],
            "duration": _loss_weights(config)[2],
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_transformer_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[ActivityTripTransformer, dict[str, Any]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    dimensions = TransformerDimensions(**checkpoint["dimensions"])
    model = ActivityTripTransformer(
        dimensions=dimensions,
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def evaluate_transformer_detailed(
    model_path: Path,
    dataset: TripSequenceDataset,
    batch_size: int,
    config: dict[str, Any],
    vocabularies: VocabularyBundle,
) -> tuple[dict[str, Any], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_transformer_checkpoint(model_path, device=device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trip_sequences,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", torch.cuda.is_available())),
    )
    with torch.no_grad():
        metrics, predictions = _run_epoch(
            model,
            loader,
            device,
            None,
            config,
            vocabularies=vocabularies,
            collect_predictions=True,
        )
    if predictions is None:
        raise RuntimeError("Detailed Transformer evaluation did not produce predictions.")
    return metrics, predictions


def evaluate_transformer_dataset(
    model_path: Path,
    dataset: TripSequenceDataset,
    batch_size: int,
    config: dict[str, Any] | None = None,
    vocabularies: VocabularyBundle | None = None,
) -> dict[str, Any]:
    """Backward-compatible metrics-only evaluation wrapper."""
    effective_config = dict(config or {})
    effective_config.setdefault("duration_bin_minutes", dataset.duration_bin_minutes)
    effective_config.setdefault("metric_top_k_values", [3])
    effective_config.setdefault("zone_loss_weight", 1.0)
    effective_config.setdefault("activity_loss_weight", 0.5)
    effective_config.setdefault("duration_loss_weight", 0.5)
    effective_vocabularies = vocabularies or dataset.vocabularies
    metrics, _ = evaluate_transformer_detailed(
        model_path,
        dataset,
        batch_size,
        effective_config,
        effective_vocabularies,
    )
    return metrics