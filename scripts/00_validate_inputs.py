from __future__ import annotations

import argparse
import json

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, path_from_config
from data.experiment_data import validate_household_splits
from routing.factory import build_valhalla_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-valhalla", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    required_keys = [
        "building_master",
        "dwell_train",
        "dwell_validation",
        "dwell_test",
        "trips_train",
        "trips_validation",
        "trips_test",
        "zones",
        "roads",
    ]
    missing = [str(path_from_config(config, key)) for key in required_keys if not path_from_config(config, key).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    counts = validate_household_splits(
        {
            "train": path_from_config(config, "trips_train"),
            "validation": path_from_config(config, "trips_validation"),
            "test": path_from_config(config, "trips_test"),
        }
    )
    status = None
    if not args.skip_valhalla:
        status = build_valhalla_client(config).status()
    print(json.dumps({"split_rows": counts, "valhalla_status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
