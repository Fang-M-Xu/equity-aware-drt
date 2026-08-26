from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from config import load_config, path_from_config
from demand.transformer_data import (
    TripSequenceDataset,
    VocabularyBundle,
    build_profile_matrix,
    build_vocabularies,
)
from demand.transformer_train import (
    dataset_class_distribution,
    evaluate_baselines,
    evaluate_transformer_detailed,
    fit_baseline_artifact,
    train_transformer,
    transformer_method_comparison,
)
from logging_utils import configure_logging
from random_utils import set_global_seed


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_split_outputs(
    split: str,
    dataset: TripSequenceDataset,
    model_path: Path,
    vocabularies: VocabularyBundle,
    baseline_artifact: dict[str, Any],
    transformer_config: dict[str, Any],
    metrics_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics, predictions = evaluate_transformer_detailed(
        model_path=model_path,
        dataset=dataset,
        batch_size=int(transformer_config["batch_size"]),
        config=transformer_config,
        vocabularies=vocabularies,
    )
    baseline_table = evaluate_baselines(
        dataset=dataset,
        artifact=baseline_artifact,
        vocabularies=vocabularies,
        config=transformer_config,
        split=split,
    )
    comparison = transformer_method_comparison(metrics, baseline_table, split)

    _write_json(metrics, metrics_dir / f"transformer_{split}_metrics.json")
    predictions.to_parquet(
        metrics_dir / f"transformer_{split}_predictions.parquet", index=False
    )
    # Keep only detailed predictions and metrics for each evaluated split.
    # Split-level comparison CSVs and confusion tables are intentionally omitted
    # because they can be reconstructed from the retained predictions and the
    # consolidated transformer_all_method_comparison.csv file.
    return metrics, comparison


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics["loss"],
        "zone_accuracy": metrics["zone"]["accuracy"],
        "zone_macro_f1": metrics["zone"]["macro_f1"],
        "zone_top_3_accuracy": metrics["zone"].get("top_3_accuracy"),
        "activity_accuracy": metrics["activity"]["accuracy"],
        "activity_macro_f1": metrics["activity"]["macro_f1"],
        "activity_top_3_accuracy": metrics["activity"].get("top_3_accuracy"),
        "duration_accuracy": metrics["duration"]["accuracy"],
        "duration_macro_f1": metrics["duration"]["macro_f1"],
        "duration_mae_minutes": metrics["duration"].get("duration_mae_minutes"),
        "duration_within_one_bin_accuracy": metrics["duration"].get(
            "duration_within_one_bin_accuracy"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the untouched household-level test split after model choices are fixed.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_dir = path_from_config(config, "metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(metrics_dir / "07_train_transformer.log")

    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    transformer_config = dict(config["transformer"])

    # Defaults make the replacement files work without forcing a base.yaml overwrite.
    transformer_config.setdefault("zone_loss_weight", 1.0)
    transformer_config.setdefault("activity_loss_weight", 0.5)
    transformer_config.setdefault("duration_loss_weight", 0.5)
    transformer_config.setdefault("label_smoothing", 0.0)
    transformer_config.setdefault("gradient_clip_norm", 1.0)
    transformer_config.setdefault("metric_top_k_values", [3])
    transformer_config.setdefault("num_workers", 0)
    transformer_config.setdefault("pin_memory", True)
    transformer_config.setdefault("evaluate_test_by_default", False)

    evaluate_test = bool(args.evaluate_test) or bool(
        transformer_config["evaluate_test_by_default"]
    )

    train = pd.read_parquet(path_from_config(config, "trips_train_routed"))
    validation = pd.read_parquet(
        path_from_config(config, "trips_validation_routed")
    )

    _, profile_columns = build_profile_matrix(train)
    profile_schema_path = path_from_config(config, "transformer_profile_schema")
    _write_json(profile_columns, profile_schema_path)

    vocabularies = build_vocabularies(
        train,
        duration_bin_minutes=int(transformer_config["duration_bin_minutes"]),
        max_duration_minutes=int(transformer_config["max_duration_minutes"]),
    )
    vocabulary_path = path_from_config(config, "transformer_vocabulary")
    vocabularies.save(vocabulary_path)

    common_arguments = {
        "vocabularies": vocabularies,
        "profile_columns": profile_columns,
        "max_sequence_length": int(transformer_config["max_sequence_length"]),
        "duration_bin_minutes": int(transformer_config["duration_bin_minutes"]),
        "max_duration_minutes": int(transformer_config["max_duration_minutes"]),
    }
    train_dataset = TripSequenceDataset(train, **common_arguments)
    validation_dataset = TripSequenceDataset(validation, **common_arguments)

    # Baseline statistics are needed only during this evaluation run.
    # They are intentionally not persisted because they can be rebuilt from
    # the retained training dataset whenever needed.
    baseline_artifact = fit_baseline_artifact(train_dataset)

    training_result = train_transformer(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        vocabularies=vocabularies,
        profile_columns=profile_columns,
        config=transformer_config,
        model_path=path_from_config(config, "transformer_model"),
        metrics_path=metrics_dir / "transformer_training_metrics.json",
        seed=seed,
    )

    validation_metrics, validation_comparison = _save_split_outputs(
        split="validation",
        dataset=validation_dataset,
        model_path=path_from_config(config, "transformer_model"),
        vocabularies=vocabularies,
        baseline_artifact=baseline_artifact,
        transformer_config=transformer_config,
        metrics_dir=metrics_dir,
    )

    distributions = [
        dataset_class_distribution(train_dataset, vocabularies, "train"),
        dataset_class_distribution(validation_dataset, vocabularies, "validation"),
    ]
    test_metrics: dict[str, Any] | None = None
    test_comparison: pd.DataFrame | None = None
    if evaluate_test:
        test = pd.read_parquet(path_from_config(config, "trips_test_routed"))
        test_dataset = TripSequenceDataset(test, **common_arguments)
        test_metrics, test_comparison = _save_split_outputs(
            split="test",
            dataset=test_dataset,
            model_path=path_from_config(config, "transformer_model"),
            vocabularies=vocabularies,
            baseline_artifact=baseline_artifact,
            transformer_config=transformer_config,
            metrics_dir=metrics_dir,
        )
        distributions.append(
            dataset_class_distribution(test_dataset, vocabularies, "test")
        )
    pd.concat(distributions, ignore_index=True).to_csv(
        metrics_dir / "transformer_class_distribution.csv", index=False
    )
    comparisons = [validation_comparison]
    if test_comparison is not None:
        comparisons.append(test_comparison)
    pd.concat(comparisons, ignore_index=True).to_csv(
        metrics_dir / "transformer_all_method_comparison.csv", index=False
    )

    summary = {
        "training": {
            key: value for key, value in training_result.items() if key != "history"
        },
        "validation": _compact_metrics(validation_metrics),
        "test_evaluated": evaluate_test,
    }
    if test_metrics is not None:
        summary["test"] = _compact_metrics(test_metrics)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nValidation baseline comparison:")
    display_columns = [
        "target",
        "method",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top_3_accuracy",
        "duration_mae_minutes",
        "duration_within_one_bin_accuracy",
    ]
    available_columns = [
        column for column in display_columns if column in validation_comparison.columns
    ]
    print(validation_comparison[available_columns].to_string(index=False))

    if not evaluate_test:
        print(
            "\nTest split was not evaluated. After the Transformer design and loss weights "
            "are fixed, rerun with --evaluate-test."
        )


if __name__ == "__main__":
    main()