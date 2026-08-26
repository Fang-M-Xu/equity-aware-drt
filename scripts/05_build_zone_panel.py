from __future__ import annotations

import argparse
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from config import load_config, path_from_config
from data.experiment_data import load_building_master
from demand.zone_panel import (
    build_zone_static_features,
    build_zone_time_panel,
    save_panel,
    summarize_panel,
)
from logging_utils import configure_logging


def main() -> None:
    """
    Build the zone-level spatiotemporal panel.

    Load the building master data and routed trip data according to the
    configuration, generate static zone features and a time panel for each
    data split (train/validation/test), and output panel summary metrics.

    Command-line arguments:
        --config: Path to the configuration file (default: configs/base.yaml)

    Workflow:
        1. Load the configuration file and initialize logging.
        2. Load the building master data.
        3. Load the routed trip data for each split.
        4. Build static zone features from the training set.
        5. Generate and save a zone time panel for each split.
        6. Output panel sparsity and exposure summary metrics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics_dir = path_from_config(config, "metrics_dir")
    configure_logging(metrics_dir / "05_build_zone_panel.log")

    buildings = load_building_master(
        path_from_config(config, "building_master"),
        source_crs=config["project"]["source_crs"],
    )
    routed = {
        split: pd.read_parquet(path_from_config(config, f"trips_{split}_routed"))
        for split in ["train", "validation", "test"]
    }

    zone_static = build_zone_static_features(buildings, routed["train"])
    zone_static_path = path_from_config(config, "zone_static_features")
    zone_static_path.parent.mkdir(parents=True, exist_ok=True)
    zone_static.to_parquet(zone_static_path, index=False)

    target_column = str(config["zone_panel"]["target_column"])
    audit_rows: list[dict[str, object]] = []
    for split, trips in routed.items():
        panel = build_zone_time_panel(
            trips,
            zone_static,
            bin_minutes=int(config["survey_time"]["bin_minutes"]),
            include_all_time_bins=bool(config["zone_panel"]["include_all_time_bins"]),
            all_day_codes=config["zone_panel"].get("all_day_codes"),
        )
        save_panel(panel, path_from_config(config, f"zone_panel_{split}"))
        summary = {"split": split, **summarize_panel(panel, target_column)}
        audit_rows.append(summary)
        print(
            f"{split}: rows={len(panel):,} "
            f"zero_share={summary['zero_share']:.3f} "
            f"target_mean={summary['target_mean']:.6f} "
            f"day_households={summary['day_household_count_min']:.0f}–"
            f"{summary['day_household_count_max']:.0f}"
        )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        metrics_dir / "zone_panel_sparsity_and_exposure.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
