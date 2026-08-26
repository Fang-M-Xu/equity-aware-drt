from __future__ import annotations

import argparse

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from benchmark.otp import OTPClient
from benchmark.runner import run_benchmarks, save_benchmarks
from config import load_config, path_from_config
from logging_utils import configure_logging
from routing.factory import build_valhalla_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_competitive.yaml")
    parser.add_argument("--scenario-id", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(path_from_config(config, "metrics_dir") / "13_multimodal_benchmarks.log")
    valhalla = build_valhalla_client(config)
    otp_config = config["otp"]
    otp_client = (
        OTPClient(otp_config["graphql_endpoint"], int(otp_config["timeout_sec"]))
        if bool(otp_config["enabled"])
        else None
    )
    forecast_dir = path_from_config(config, "forecast_dir")
    scenario_paths = sorted(forecast_dir.glob("forecast_scenario_*.parquet"))
    if args.scenario_id is not None:
        scenario_paths = [forecast_dir / f"forecast_scenario_{args.scenario_id:03d}.parquet"]
    for scenario_path in scenario_paths:
        scenario_id = int(scenario_path.stem.split("_")[-1])
        requests = pd.read_parquet(scenario_path)
        results = run_benchmarks(
            requests,
            valhalla,
            config["routing"]["costing_walk"],
            config["routing"]["costing_auto"],
            otp_client,
            otp_config["departure_datetime"],
            int(otp_config["first_itineraries"]),
        )
        output = path_from_config(config, "benchmark_dir") / f"scenario_{scenario_id:03d}.parquet"
        save_benchmarks(results, output)
        print(f"scenario={scenario_id:03d} benchmarked={len(results)}")


if __name__ == "__main__":
    main()
