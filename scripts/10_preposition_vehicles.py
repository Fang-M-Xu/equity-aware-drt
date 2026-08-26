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
from logging_utils import configure_logging
from prepositioning.service import (
    compute_demand_weights,
    load_zone_centroids,
    preposition_vehicles,
)
from routing.factory import build_valhalla_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method", choices=["random", "demand_only", "proposed"], default="proposed")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(path_from_config(config, "metrics_dir") / f"10_preposition_{args.method}.log")

    candidates = pd.read_parquet(path_from_config(config, "candidate_points"))
    prediction = pd.read_parquet(path_from_config(config, "forecast_dir") / "zone_demand_1030.parquet")
    prediction = compute_demand_weights(
        prediction,
        gap_weight=float(config["prepositioning"]["gap_weight"]),
        vulnerability_weight=float(config["prepositioning"]["vulnerability_weight"]),
    )
    centroids = load_zone_centroids(
        path_from_config(config, "zones"),
        config["zone_panel"]["zone_id_column_in_shapefile"],
        config["project"]["metric_crs"],
        config["project"]["target_crs"],
    )
    vehicles, matrix = preposition_vehicles(
        candidates,
        prediction,
        centroids,
        build_valhalla_client(config),
        costing=config["routing"]["costing_auto"],
        fleet_size=int(config["vehicle"]["fleet_size"]),
        unique_points=int(config["vehicle"]["unique_preposition_points"]),
        capacity=int(config["vehicle"]["capacity"]),
        max_sources=int(config["routing"]["matrix_max_sources"]),
        max_targets=int(config["routing"]["matrix_max_targets"]),
        method=args.method,
        seed=int(config["project"]["seed"]),
    )
    output_dir = path_from_config(config, "preposition_dir") / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicles.to_parquet(output_dir / "vehicle_initial_positions.parquet", index=False)
    __import__("numpy").save(output_dir / "zone_to_candidate_time_sec.npy", matrix)
    print(f"Saved {len(vehicles)} vehicles using {args.method} prepositioning")


if __name__ == "__main__":
    main()
