from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


VERSION = "1.0.0"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            data,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated SIDRE-DRT configuration for generating a "
            "large synthetic request candidate pool."
        )
    )
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument(
        "--output-config",
        default="configs/base_candidate_pool.yaml",
    )
    parser.add_argument(
        "--forecast-dir",
        default="outputs/forecast_candidate_pool",
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics_candidate_pool",
    )
    parser.add_argument("--requests-per-scenario", type=int, default=200)
    parser.add_argument("--number-of-scenarios", type=int, default=30)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=102026,
        help=(
            "A seed distinct from the natural-demand experiment. "
            "This changes sampling only; trained model files are reused."
        ),
    )
    parser.add_argument(
        "--maximum-sampling-attempts",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--gap-share-override",
        type=float,
        default=None,
        help=(
            "Leave unset to preserve the model's natural gap share. "
            "The later competition sampler creates the balanced 50/50 design."
        ),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    base_path = resolve(root, args.base_config)
    output_path = resolve(root, args.output_config)

    base = load_yaml(base_path)
    candidate = copy.deepcopy(base)

    project = candidate.setdefault("project", {})
    paths = candidate.setdefault("paths", {})
    scenario = candidate.setdefault("scenario", {})
    experiments = candidate.setdefault("experiments", {})

    project["seed"] = int(args.generation_seed)
    paths["forecast_dir"] = str(Path(args.forecast_dir))
    paths["metrics_dir"] = str(Path(args.metrics_dir))

    scenario["total_requests"] = int(args.requests_per_scenario)
    scenario["number_of_scenarios"] = int(args.number_of_scenarios)
    scenario["maximum_sampling_attempts"] = int(
        args.maximum_sampling_attempts
    )
    scenario["gap_share_override"] = args.gap_share_override

    # These values are not used by script 08, but keeping them consistent
    # avoids confusion when the configuration is inspected.
    experiments["scenario_seeds"] = int(args.number_of_scenarios)

    dump_yaml(candidate, output_path)

    print("=" * 72)
    print("Large candidate-pool configuration created")
    print(f"Version:               {VERSION}")
    print(f"Base config:           {base_path}")
    print(f"Output config:         {output_path}")
    print(f"Forecast output:       {args.forecast_dir}")
    print(f"Requests/scenario:     {args.requests_per_scenario}")
    print(f"Number of scenarios:   {args.number_of_scenarios}")
    print(
        "Total candidate rows:  "
        f"{args.requests_per_scenario * args.number_of_scenarios}"
    )
    print(f"Generation seed:       {args.generation_seed}")
    print(f"Gap share override:    {args.gap_share_override}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
