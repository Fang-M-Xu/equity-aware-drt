from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, path_from_config
from data.experiment_data import add_day_context_from_dwell, load_dwell_split, load_trip_split
from data.trip_enrichment import add_estimated_departure_fields, enrich_direct_times, save_routed_split
from logging_utils import configure_logging
from routing.factory import build_valhalla_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(path_from_config(config, "metrics_dir") / "04_enrich_survey_trip_times.log")
    client = build_valhalla_client(config)
    source_crs = config["project"]["source_crs"]
    bin_minutes = int(config["survey_time"]["bin_minutes"])

    for split in args.splits:
        trip_key = f"trips_{split}"
        dwell_key = f"dwell_{split}"
        output_key = f"trips_{split}_routed"
        trips = load_trip_split(path_from_config(config, trip_key), source_crs=source_crs)
        dwell = load_dwell_split(path_from_config(config, dwell_key))
        trips = add_day_context_from_dwell(trips, dwell)
        trips = enrich_direct_times(
            trips,
            client,
            costing=config["routing"]["costing_auto"],
            max_targets=int(config["routing"]["matrix_max_targets"]),
        )
        trips = add_estimated_departure_fields(trips, bin_minutes)
        save_routed_split(trips, path_from_config(config, output_key))
        logging.info(
            "%s: direct times available for %d/%d trips",
            split,
            int(trips["direct_time_available"].sum()),
            len(trips),
        )


if __name__ == "__main__":
    main()
