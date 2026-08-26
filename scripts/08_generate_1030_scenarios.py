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
from demand.lightgbm_model import load_model
from demand.scenario_generator import (
    allocate_zone_counts,
    build_target_zone_prediction,
    generate_scenario,
    save_zone_prediction,
)
from logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--scenario-count", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(path_from_config(config, "metrics_dir") / "08_generate_scenarios.log")
    scenario_config = config["scenario"]
    transformer_config = config["transformer"]
    seed_base = int(config["project"]["seed"])

    buildings = load_building_master(
        path_from_config(config, "building_master"),
        source_crs=config["project"]["source_crs"],
    )
    train_trips = pd.read_parquet(path_from_config(config, "trips_train_routed"))
    zone_static = pd.read_parquet(path_from_config(config, "zone_static_features"))
    lightgbm_artifact = load_model(path_from_config(config, "lightgbm_model"))
    prediction = build_target_zone_prediction(
        zone_static,
        lightgbm_artifact,
        day_of_week_code=int(scenario_config["day_of_week_code"]),
        departure_bin=int(scenario_config["departure_bin_minute"]),
    )
    forecast_dir = path_from_config(config, "forecast_dir")
    forecast_dir.mkdir(parents=True, exist_ok=True)
    save_zone_prediction(prediction, forecast_dir / "zone_demand_1030.parquet")

    scenario_count = args.scenario_count or int(scenario_config["number_of_scenarios"])
    for scenario_id in range(scenario_count):
        seed = seed_base + scenario_id
        counts = allocate_zone_counts(
            prediction,
            total_requests=int(scenario_config["total_requests"]),
            rng=__import__("numpy").random.default_rng(seed),
            gap_share_override=scenario_config.get("gap_share_override"),
        )
        scenario = generate_scenario(
            counts,
            buildings,
            train_trips,
            path_from_config(config, "transformer_model"),
            path_from_config(config, "transformer_vocabulary"),
            departure_bin=int(scenario_config["departure_bin_minute"]),
            departure_time=str(scenario_config["departure_time"]),
            total_requests=int(scenario_config["total_requests"]),
            minimum_od_distance_m=float(scenario_config["minimum_od_distance_m"]),
            maximum_sampling_attempts=int(scenario_config["maximum_sampling_attempts"]),
            temperature=float(transformer_config["sampling_temperature"]),
            top_k=int(transformer_config["top_k"]),
            seed=seed,
        )
        output_path = forecast_dir / f"forecast_scenario_{scenario_id:03d}.parquet"
        scenario.to_parquet(output_path, index=False)
        print(
            f"scenario={scenario_id:03d} requests={len(scenario)} "
            f"gap_share={scenario['is_gap'].mean() if len(scenario) else float('nan'):.3f}"
        )


if __name__ == "__main__":
    main()
