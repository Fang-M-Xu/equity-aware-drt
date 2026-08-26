from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, path_from_config
from logging_utils import configure_logging
from prepositioning.candidates import (
    generate_road_candidates,
    save_candidates,
)


def _load_active_forecast_zones(config: dict, enabled: bool) -> list[object] | None:
    if not enabled:
        return None

    forecast_path = (
        path_from_config(config, "forecast_dir")
        / "zone_demand_1030.parquet"
    )
    if not forecast_path.exists():
        print(
            "Warning: restrict_to_forecast_zones=true, but the 10:30 forecast "
            f"does not exist: {forecast_path}. All TAZs will be retained."
        )
        return None

    forecast = pd.read_parquet(forecast_path)
    zone_column = next(
        (
            column
            for column in ["origin_zone_id", "zone_id", "TAZ_1270"]
            if column in forecast.columns
        ),
        None,
    )
    if zone_column is None:
        raise ValueError(
            "The 10:30 forecast has no origin_zone_id/zone_id column."
        )

    return forecast[zone_column].dropna().drop_duplicates().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_dir = path_from_config(config, "metrics_dir")
    configure_logging(metrics_dir / "09_build_waiting_candidates.log")

    settings = config["prepositioning"]
    active_zone_ids = _load_active_forecast_zones(
        config,
        enabled=bool(settings.get("restrict_to_forecast_zones", True)),
    )

    candidates, audit = generate_road_candidates(
        roads_path=path_from_config(config, "roads"),
        zones_path=path_from_config(config, "zones"),
        zone_id_column=config["zone_panel"]["zone_id_column_in_shapefile"],
        metric_crs=config["project"]["metric_crs"],
        target_crs=config["project"]["target_crs"],
        spacing_m=float(settings.get("candidate_spacing_m", 300.0)),
        dedup_grid_m=float(settings.get("candidate_dedup_grid_m", 100.0)),
        max_candidates_per_zone=int(settings.get("max_candidates_per_zone", 10)),
        active_zone_ids=active_zone_ids,
        allowed_road_classes=settings.get("allowed_road_classes") or [],
        excluded_road_classes=settings.get("excluded_road_classes") or [],
        road_class_column=settings.get("road_class_column"),
    )

    output_path = path_from_config(config, "candidate_points")
    save_candidates(candidates, output_path)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    audit_path = metrics_dir / "waiting_candidate_generation_summary.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Generated {len(candidates):,} final road-based waiting candidates "
        f"across {audit['zones_with_candidates']:,} TAZs"
    )
    print(
        "Candidate count per TAZ: "
        f"min={audit['minimum_candidates_per_zone']}, "
        f"mean={audit['mean_candidates_per_zone']:.2f}, "
        f"max={audit['maximum_candidates_per_zone']}"
    )
    print(f"Saved candidate audit to: {audit_path}")


if __name__ == "__main__":
    main()