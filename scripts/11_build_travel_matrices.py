from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from config import load_config, path_from_config
from logging_utils import configure_logging
from routing.factory import build_valhalla_client
from routing.operating_scenario import prepare_operating_scenario, save_operating_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--preposition-method", default="proposed", choices=["random", "demand_only", "proposed"])
    parser.add_argument("--scenario-id", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(path_from_config(config, "metrics_dir") / "11_build_travel_matrices.log")
    client = build_valhalla_client(config)
    forecast_dir = path_from_config(config, "forecast_dir")
    vehicle_path = (
        path_from_config(config, "preposition_dir")
        / args.preposition_method
        / "vehicle_initial_positions.parquet"
    )
    vehicles = pd.read_parquet(vehicle_path)
    scenario_paths = sorted(forecast_dir.glob("forecast_scenario_*.parquet"))
    if args.scenario_id is not None:
        scenario_paths = [forecast_dir / f"forecast_scenario_{args.scenario_id:03d}.parquet"]

    for scenario_path in scenario_paths:
        scenario_id = int(scenario_path.stem.split("_")[-1])
        scenario = pd.read_parquet(scenario_path)
        all_requests, feasible, vehicles_output, time_matrix, distance_matrix, nodes = prepare_operating_scenario(
            scenario,
            vehicles,
            path_from_config(config, "roads"),
            metric_crs=config["project"]["metric_crs"],
            target_crs=config["project"]["target_crs"],
            snap_max_distance_m=float(config["routing"]["snap_max_distance_m"]),
            client=client,
            auto_costing=config["routing"]["costing_auto"],
            walk_costing=config["routing"]["costing_walk"],
            constraints=config["constraints"],
            max_sources=int(config["routing"]["matrix_max_sources"]),
            max_targets=int(config["routing"]["matrix_max_targets"]),
        )
        output_dir = (
            path_from_config(config, "operating_dir")
            / args.preposition_method
            / f"scenario_{scenario_id:03d}"
        )
        save_operating_scenario(
            output_dir,
            all_requests,
            feasible,
            vehicles_output,
            nodes,
            time_matrix,
            distance_matrix,
        )
        print(
            f"scenario={scenario_id:03d} all={len(all_requests)} feasible={len(feasible)} "
            f"nodes={len(nodes)}"
        )


if __name__ == "__main__":
    main()
