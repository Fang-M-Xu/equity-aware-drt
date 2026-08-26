from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from alns.io import load_problem, save_solver_result
from alns.solver import solve_alns
from config import load_config, path_from_config
from logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--preposition-method", default="proposed")
    parser.add_argument("--scenario-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--method-name", default="proposed_full")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    configure_logging(path_from_config(config, "metrics_dir") / "12_run_alns.log")
    scenario_dir = (
        path_from_config(config, "operating_dir")
        / args.preposition_method
        / f"scenario_{args.scenario_id:03d}"
    )
    all_requests_frame, requests_frame, _, requests, vehicles, time_matrix, distance_matrix = load_problem(scenario_dir)
    result = solve_alns(
        vehicles,
        requests,
        time_matrix,
        distance_matrix,
        constraints=config["constraints"],
        objective_weights=config["objective"],
        alns_config=config["alns"],
        seed=seed,
        greedy_only=args.greedy_only,
    )
    output_dir = (
        path_from_config(config, "alns_dir")
        / args.method_name
        / f"scenario_{args.scenario_id:03d}"
        / f"seed_{seed}"
    )
    save_solver_result(
        result,
        all_requests_frame,
        requests_frame,
        output_dir,
        args.method_name,
        args.scenario_id,
        seed,
    )

    # Keep the formal experiment output compact. The iteration-level ALNS
    # history is useful for convergence diagnostics, but is not required by
    # the downstream evaluation pipeline. Remove it after the standard saver
    # has written the core result files.
    history_path = output_dir / "solver_history.csv"
    if history_path.exists():
        history_path.unlink()

    print(
        f"served={len(result.evaluation.served_requests)}/{len(requests)} "
        f"objective={result.evaluation.objective:.2f} elapsed={result.elapsed_sec:.2f}s"
    )


if __name__ == "__main__":
    main()
