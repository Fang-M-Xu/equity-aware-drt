from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNNER_VERSION = "3.0-simplified-output"

_ACTIVE_PROCESSES: dict[str, subprocess.Popen[Any]] = {}
_ACTIVE_PROCESSES_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExperimentTask:
    method_name: str
    solver: str
    preposition_method: str
    objective_overrides: dict[str, float]
    scenario_id: int
    replicate_index: int
    seed: int


@dataclass
class TaskResult:
    method_name: str
    solver: str
    preposition_method: str
    scenario_id: int
    replicate_index: int
    seed: int
    status: str
    return_code: int | None
    duration_sec: float
    attempts: int
    signature: str
    result_dir: str
    message: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unnamed"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            data,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    os.replace(temporary, path)


def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "method_name",
        "solver",
        "preposition_method",
        "scenario_id",
        "replicate_index",
        "seed",
        "status",
        "return_code",
        "duration_sec",
        "attempts",
        "signature",
        "result_dir",
        "message",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(temporary, path)


def _read_status_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_key(task: ExperimentTask) -> tuple[str, int, int, int]:
    return (
        task.method_name,
        task.scenario_id,
        task.replicate_index,
        task.seed,
    )


def _status_key(row: dict[str, Any]) -> tuple[str, int, int, int] | None:
    try:
        return (
            str(row["method_name"]),
            int(row["scenario_id"]),
            int(row["replicate_index"]),
            int(row["seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _sha256_json(data: Any) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _build_run_config(
    base_config: dict[str, Any],
    task: ExperimentTask,
    execution: dict[str, Any],
    task_metrics_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    objective = config.setdefault("objective", {})
    if not isinstance(objective, dict):
        raise ValueError("Base config section 'objective' must be a mapping.")
    objective.update(task.objective_overrides)

    alns_overrides = execution.get("alns_overrides") or {}
    if not isinstance(alns_overrides, dict):
        raise ValueError("execution.alns_overrides must be a mapping.")
    alns = config.setdefault("alns", {})
    if not isinstance(alns, dict):
        raise ValueError("Base config section 'alns' must be a mapping.")
    alns.update(alns_overrides)

    # Child-process metrics/logs are temporary. This keeps the permanent
    # comparative directory limited to execution_plan.json, run_status.csv,
    # and summary.json.
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Base config section 'paths' must be a mapping.")
    paths["metrics_dir"] = str(task_metrics_dir)
    return config


def _signature_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return a stable config copy for resume signatures.

    Temporary metrics paths change every run and must not invalidate resume.
    """
    config = copy.deepcopy(run_config)
    paths = config.get("paths")
    if isinstance(paths, dict):
        paths.pop("metrics_dir", None)
    return config


def _task_signature(
    task: ExperimentTask,
    run_config: dict[str, Any],
    script12: Path,
) -> str:
    script_metadata: dict[str, Any] = {"path": str(script12)}
    if script12.exists():
        stat = script12.stat()
        script_metadata.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return _sha256_json(
        {
            "runner_version": RUNNER_VERSION,
            "task": asdict(task),
            "run_config": _signature_config(run_config),
            "script12": script_metadata,
        }
    )


def _expected_result_dir(
    project_root: Path,
    base_config: dict[str, Any],
    task: ExperimentTask,
) -> Path:
    paths = base_config.get("paths") or {}
    alns_dir = _resolve_path(project_root, paths.get("alns_dir", "outputs/alns"))
    return (
        alns_dir
        / task.method_name
        / f"scenario_{task.scenario_id:03d}"
        / f"seed_{task.seed}"
    )


def _formal_results_complete(result_dir: Path) -> bool:
    required = [
        "routes.parquet",
        "vehicle_results.parquet",
        "request_results.parquet",
        "summary.json",
    ]
    return all((result_dir / name).exists() for name in required)


def _register_process(key: str, process: subprocess.Popen[Any]) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES[key] = process


def _unregister_process(key: str) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.pop(key, None)


def _terminate_active_processes() -> None:
    with _ACTIVE_PROCESSES_LOCK:
        processes = list(_ACTIVE_PROCESSES.items())
    if not processes:
        return

    print(f"\nStopping {len(processes)} active child process(es)...")
    for _, process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for _, process in processes):
            return
        time.sleep(0.1)

    for _, process in processes:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass


def _build_command(
    python_executable: str,
    script12: Path,
    run_config_path: Path,
    task: ExperimentTask,
) -> list[str]:
    command = [
        python_executable,
        str(script12),
        "--config",
        str(run_config_path),
        "--preposition-method",
        task.preposition_method,
        "--scenario-id",
        str(task.scenario_id),
        "--seed",
        str(task.seed),
        "--method-name",
        task.method_name,
    ]
    if task.solver == "greedy":
        command.append("--greedy-only")
    return command


def _run_task(
    *,
    task: ExperimentTask,
    project_root: Path,
    script12: Path,
    base_config: dict[str, Any],
    execution: dict[str, Any],
    run_label: str,
    resume: bool,
    force: bool,
    retries: int,
    thread_limit: int,
    previous_status: dict[tuple[str, int, int, int], dict[str, Any]],
) -> TaskResult:
    # Temporary files are isolated per task and deleted in finally.
    temp_root = Path(tempfile.mkdtemp(prefix="sidre_comparative_"))
    task_metrics_dir = temp_root / "metrics"
    child_log_path = temp_root / "child.log"

    generated_name = (
        f"_comparative_tmp_{_safe_name(run_label)}_"
        f"{_safe_name(task.method_name)}_scenario_{task.scenario_id:03d}_"
        f"rep_{task.replicate_index:02d}_seed_{task.seed}.yaml"
    )
    config_path = project_root / "configs" / generated_name

    run_config = _build_run_config(
        base_config,
        task,
        execution,
        task_metrics_dir,
    )
    signature = _task_signature(task, run_config, script12)
    result_dir = _expected_result_dir(project_root, base_config, task)

    old = previous_status.get(_task_key(task))
    if resume and not force and old is not None:
        if (
            str(old.get("status", "")) in {"completed", "skipped"}
            and str(old.get("signature", "")) == signature
            and _formal_results_complete(result_dir)
        ):
            shutil.rmtree(temp_root, ignore_errors=True)
            return TaskResult(
                method_name=task.method_name,
                solver=task.solver,
                preposition_method=task.preposition_method,
                scenario_id=task.scenario_id,
                replicate_index=task.replicate_index,
                seed=task.seed,
                status="skipped",
                return_code=0,
                duration_sec=float(old.get("duration_sec", 0.0) or 0.0),
                attempts=int(old.get("attempts", 1) or 1),
                signature=signature,
                result_dir=str(result_dir),
                message="Matching completed task found in run_status.csv.",
            )

    _dump_yaml(run_config, config_path)
    command = _build_command(sys.executable, script12, config_path, task)

    environment = os.environ.copy()
    thread_value = str(max(1, thread_limit))
    for variable in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        environment[variable] = thread_value
    environment["PYTHONUNBUFFERED"] = "1"

    start = time.monotonic()
    return_code: int | None = None
    attempts = 0
    message = ""

    try:
        for attempt in range(1, retries + 2):
            attempts = attempt
            key = (
                f"{task.method_name}|{task.scenario_id}|"
                f"{task.replicate_index}|{task.seed}"
            )
            try:
                child_log_path.parent.mkdir(parents=True, exist_ok=True)
                with child_log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write("\n" + "=" * 100 + "\n")
                    log.write(f"Started: {_utc_now()}\n")
                    log.write(f"Task: {json.dumps(asdict(task), ensure_ascii=False)}\n")
                    log.write(f"Command: {subprocess.list2cmdline(command)}\n")
                    if attempt > 1:
                        log.write(f"Retry {attempt - 1}/{retries}\n")
                    process = subprocess.Popen(
                        command,
                        cwd=project_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        text=True,
                    )
                    _register_process(key, process)
                    return_code = process.wait()
            except Exception as exc:  # noqa: BLE001
                return_code = -1
                message = f"{type(exc).__name__}: {exc}"
            finally:
                _unregister_process(key)

            if return_code == 0:
                break

        if return_code != 0 and child_log_path.exists():
            try:
                text = child_log_path.read_text(encoding="utf-8", errors="replace")
                tail = text[-4000:].strip()
                if tail:
                    message = (message + "\n" + tail).strip()
            except OSError:
                pass

        duration = time.monotonic() - start
        status = "completed" if return_code == 0 else "failed"
        return TaskResult(
            method_name=task.method_name,
            solver=task.solver,
            preposition_method=task.preposition_method,
            scenario_id=task.scenario_id,
            replicate_index=task.replicate_index,
            seed=task.seed,
            status=status,
            return_code=return_code,
            duration_sec=duration,
            attempts=attempts,
            signature=signature,
            result_dir=str(result_dir),
            message=message,
        )
    finally:
        try:
            config_path.unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(temp_root, ignore_errors=True)


def _validate_methods(methods: Any) -> list[dict[str, Any]]:
    if not isinstance(methods, list) or not methods:
        raise ValueError("Experiment YAML must contain a non-empty 'methods' list.")

    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(methods):
        if not isinstance(raw, dict):
            raise ValueError(f"methods[{index}] must be a mapping.")

        name = str(raw.get("name", "")).strip()
        solver = str(raw.get("solver", "")).strip().lower()
        preposition = str(raw.get("preposition_method", "")).strip()
        overrides = raw.get("objective_overrides") or {}

        if not name:
            raise ValueError(f"methods[{index}].name is required.")
        if name in names:
            raise ValueError(f"Duplicate method name: {name}")
        if solver not in {"alns", "greedy"}:
            raise ValueError(
                f"methods[{index}].solver must be 'alns' or 'greedy': {solver}"
            )
        if not preposition:
            raise ValueError(f"methods[{index}].preposition_method is required.")
        if not isinstance(overrides, dict):
            raise ValueError(
                f"methods[{index}].objective_overrides must be a mapping."
            )

        names.add(name)
        validated.append(
            {
                "name": name,
                "solver": solver,
                "preposition_method": preposition,
                "objective_overrides": {
                    str(key): float(value) for key, value in overrides.items()
                },
            }
        )
    return validated


def _build_tasks(
    *,
    methods: list[dict[str, Any]],
    selected_method_names: set[str] | None,
    scenario_count: int,
    alns_repeats: int,
    greedy_repeats: int,
    base_seed: int,
) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    selected = [
        method
        for method in methods
        if selected_method_names is None or method["name"] in selected_method_names
    ]

    if selected_method_names is not None:
        available = {method["name"] for method in methods}
        missing = selected_method_names - available
        if missing:
            raise ValueError("Unknown method name(s): " + ", ".join(sorted(missing)))

    for scenario_id in range(scenario_count):
        for method in selected:
            repeats = greedy_repeats if method["solver"] == "greedy" else alns_repeats
            for replicate_index in range(repeats):
                seed = base_seed + scenario_id * 10_000 + replicate_index
                tasks.append(
                    ExperimentTask(
                        method_name=method["name"],
                        solver=method["solver"],
                        preposition_method=method["preposition_method"],
                        objective_overrides=method["objective_overrides"],
                        scenario_id=scenario_id,
                        replicate_index=replicate_index,
                        seed=seed,
                    )
                )
    return tasks


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not (seconds < float("inf")):
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run comparative experiments with compact persistent output: "
            "execution_plan.json, run_status.csv, summary.json, plus solver results."
        )
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiments", default="configs/experiments_optimized.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-solver-seeds", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume tasks recorded as completed in run_status.csv when signatures match.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore run_status.csv completion state and rerun selected tasks.",
    )
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write execution_plan.json and print the plan without running solvers.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()
    config_path = _resolve_path(project_root, args.config)
    experiments_path = _resolve_path(project_root, args.experiments)
    script12 = project_root / "scripts" / "12_run_alns.py"

    if not script12.exists():
        raise FileNotFoundError(f"Required solver entry point not found: {script12}")

    base_config = _load_yaml(config_path)
    experiment_spec = _load_yaml(experiments_path)
    methods = _validate_methods(experiment_spec.get("methods"))

    execution = experiment_spec.get("execution") or {}
    if not isinstance(execution, dict):
        raise ValueError("Experiment YAML section 'execution' must be a mapping.")

    base_experiments = base_config.get("experiments") or {}
    base_scenario = base_config.get("scenario") or {}
    base_project = base_config.get("project") or {}
    base_paths = base_config.get("paths") or {}

    generated_scenarios = int(base_scenario.get("number_of_scenarios", 1))
    configured_scenarios = int(
        execution.get(
            "scenario_count",
            base_experiments.get("scenario_seeds", generated_scenarios),
        )
    )
    scenario_count = min(configured_scenarios, generated_scenarios)
    if args.max_scenarios is not None:
        scenario_count = min(scenario_count, args.max_scenarios)

    alns_repeats = int(
        execution.get(
            "solver_seeds_per_scenario",
            base_experiments.get("solver_seeds_per_scenario", 1),
        )
    )
    if args.max_solver_seeds is not None:
        alns_repeats = min(alns_repeats, args.max_solver_seeds)

    greedy_repeats = int(execution.get("greedy_repeats_per_scenario", 1))
    base_seed = int(base_project.get("seed", 2026))

    if scenario_count < 1:
        raise ValueError("Scenario count must be at least 1.")
    if alns_repeats < 1:
        raise ValueError("ALNS solver seeds per scenario must be at least 1.")
    if greedy_repeats < 1:
        raise ValueError("Greedy repeats per scenario must be at least 1.")

    auto_workers = max(1, min(4, max(1, (os.cpu_count() or 2) // 2)))
    workers = int(
        args.workers if args.workers is not None else execution.get("workers", auto_workers)
    )
    workers = max(1, workers)

    resume = bool(execution.get("resume", True))
    if args.resume is not None:
        resume = bool(args.resume)

    retries = int(args.retries if args.retries is not None else execution.get("retries", 0))
    retries = max(0, retries)
    thread_limit = max(1, int(execution.get("thread_limit_per_process", 1)))
    run_label = _safe_name(str(execution.get("run_label", "comparative")))

    alns_dir = _resolve_path(project_root, base_paths.get("alns_dir", "outputs/alns"))
    progress_root = alns_dir / "_comparative_progress" / run_label
    progress_root.mkdir(parents=True, exist_ok=True)

    # Remove legacy runner directories if present, so the final persistent
    # comparative directory contains only the three recommended files.
    for legacy_name in ["logs", "markers", "internal_metrics"]:
        shutil.rmtree(progress_root / legacy_name, ignore_errors=True)

    selected_method_names = set(args.methods) if args.methods else None
    tasks = _build_tasks(
        methods=methods,
        selected_method_names=selected_method_names,
        scenario_count=scenario_count,
        alns_repeats=alns_repeats,
        greedy_repeats=greedy_repeats,
        base_seed=base_seed,
    )
    workers = min(workers, max(1, len(tasks)))

    alns_task_count = sum(task.solver == "alns" for task in tasks)
    greedy_task_count = len(tasks) - alns_task_count
    alns_budget_sec = float(
        (execution.get("alns_overrides") or {}).get(
            "time_limit_sec",
            (base_config.get("alns") or {}).get("time_limit_sec", 0.0),
        )
    )
    serial_upper_bound = alns_task_count * alns_budget_sec
    parallel_upper_bound = serial_upper_bound / workers if workers else 0.0

    print("=" * 78)
    print(f"Comparative experiment runner ({RUNNER_VERSION})")
    print(f"Project root:            {project_root}")
    print(f"Base config:             {config_path}")
    print(f"Experiment config:       {experiments_path}")
    print(f"Run label:               {run_label}")
    print(f"Scenarios:               {scenario_count}")
    print(f"ALNS repeats/scenario:   {alns_repeats}")
    print(f"Greedy repeats/scenario: {greedy_repeats}")
    print(f"ALNS tasks:              {alns_task_count}")
    print(f"Greedy tasks:            {greedy_task_count}")
    print(f"Parallel workers:        {workers}")
    print(f"Resume:                  {resume and not args.force}")
    print(f"Retries:                 {retries}")
    if alns_budget_sec > 0:
        print(
            "ALNS time-budget upper bound: "
            f"serial {_format_eta(serial_upper_bound)}, "
            f"parallel {_format_eta(parallel_upper_bound)}"
        )
    print(f"Progress directory:      {progress_root}")
    print("=" * 78)

    plan_payload = {
        "runner_version": RUNNER_VERSION,
        "created_at": _utc_now(),
        "project_root": str(project_root),
        "config_path": str(config_path),
        "experiments_path": str(experiments_path),
        "execution": execution,
        "scenario_count": scenario_count,
        "alns_repeats": alns_repeats,
        "greedy_repeats": greedy_repeats,
        "workers": workers,
        "resume": resume,
        "force": args.force,
        "retries": retries,
        "task_count": len(tasks),
        "alns_task_count": alns_task_count,
        "greedy_task_count": greedy_task_count,
        "tasks": [asdict(task) for task in tasks],
    }
    _atomic_write_json(plan_payload, progress_root / "execution_plan.json")

    if args.dry_run:
        print("Dry run complete. No child process was started.")
        return 0

    status_csv = progress_root / "run_status.csv"
    old_rows = _read_status_csv(status_csv)
    previous_status = {
        key: row
        for row in old_rows
        if (key := _status_key(row)) is not None
    }

    # Keep one latest row per task key. This enables compact persistent resume state.
    result_map: dict[tuple[str, int, int, int], dict[str, Any]] = {
        key: row for key, row in previous_status.items()
    }

    start_all = time.monotonic()
    completed_counter = 0
    failed_counter = 0
    skipped_counter = 0
    interrupted = False

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="sidre-comparative",
    )
    futures: dict[Future[TaskResult], ExperimentTask] = {}

    try:
        for task in tasks:
            future = executor.submit(
                _run_task,
                task=task,
                project_root=project_root,
                script12=script12,
                base_config=base_config,
                execution=execution,
                run_label=run_label,
                resume=resume,
                force=args.force,
                retries=retries,
                thread_limit=thread_limit,
                previous_status=previous_status,
            )
            futures[future] = task

        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = TaskResult(
                    method_name=task.method_name,
                    solver=task.solver,
                    preposition_method=task.preposition_method,
                    scenario_id=task.scenario_id,
                    replicate_index=task.replicate_index,
                    seed=task.seed,
                    status="failed",
                    return_code=-1,
                    duration_sec=0.0,
                    attempts=0,
                    signature="",
                    result_dir=str(_expected_result_dir(project_root, base_config, task)),
                    message=f"{type(exc).__name__}: {exc}",
                )

            row = asdict(result)
            row["duration_sec"] = round(float(row["duration_sec"]), 6)
            result_map[_task_key(task)] = row
            ordered_rows = [
                result_map[key]
                for key in sorted(
                    result_map,
                    key=lambda item: (item[1], item[0], item[2], item[3]),
                )
            ]
            _atomic_write_csv(ordered_rows, status_csv)

            if result.status == "completed":
                completed_counter += 1
            elif result.status == "skipped":
                skipped_counter += 1
            else:
                failed_counter += 1

            elapsed = time.monotonic() - start_all
            finished = completed_counter + skipped_counter + failed_counter
            measured = [
                float(item["duration_sec"])
                for item in ordered_rows
                if item.get("status") == "completed"
                and float(item.get("duration_sec", 0) or 0) > 0
            ]
            mean_duration = sum(measured) / len(measured) if measured else 0.0
            remaining = len(tasks) - finished
            eta = remaining * mean_duration / workers if mean_duration > 0 else 0.0

            print(
                f"[{index:>4}/{len(tasks)}] "
                f"{result.status.upper():<9} "
                f"{result.method_name} "
                f"scenario={result.scenario_id} "
                f"rep={result.replicate_index} "
                f"seed={result.seed} "
                f"time={result.duration_sec:.1f}s "
                f"elapsed={_format_eta(elapsed)} "
                f"ETA={_format_eta(eta)}"
            )
            if result.status == "failed" and result.message:
                print("    Error tail saved in run_status.csv message field.")

    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboard interrupt received.")
        for future in futures:
            future.cancel()
        _terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)

    total_duration = time.monotonic() - start_all
    summary = {
        "runner_version": RUNNER_VERSION,
        "finished_at": _utc_now(),
        "interrupted": interrupted,
        "total_tasks": len(tasks),
        "completed": completed_counter,
        "skipped": skipped_counter,
        "failed": failed_counter,
        "total_duration_sec": round(total_duration, 6),
        "workers": workers,
        "status_csv": str(status_csv),
        "progress_root": str(progress_root),
    }
    _atomic_write_json(summary, progress_root / "summary.json")

    # Final cleanup for old generated comparative YAML files from earlier runner versions.
    for old_config in (project_root / "configs").glob("_comparative_*.yaml"):
        try:
            old_config.unlink()
        except OSError:
            pass

    print("=" * 78)
    print(
        f"Completed={completed_counter}, skipped={skipped_counter}, "
        f"failed={failed_counter}, duration={_format_eta(total_duration)}"
    )
    print(f"Status:  {status_csv}")
    print(f"Summary: {progress_root / 'summary.json'}")
    print("Persistent comparative control files:")
    print(f"  {progress_root / 'execution_plan.json'}")
    print(f"  {progress_root / 'run_status.csv'}")
    print(f"  {progress_root / 'summary.json'}")
    print("=" * 78)

    if interrupted:
        return 130
    return 1 if failed_counter else 0


if __name__ == "__main__":
    raise SystemExit(main())
