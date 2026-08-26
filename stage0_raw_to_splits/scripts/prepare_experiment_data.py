from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build building, dwell-event and OD-trip master datasets from raw "
            "survey files, then split them by household."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    run_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
