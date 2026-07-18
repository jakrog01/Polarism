"""Local multi-GPU scenario pool for prepared pipeline runs."""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

from pipeline.config.loader import load_config
from pipeline.manifest.io import load_scenario_index, resolve_scenario_name


def _parse_gpu_ids(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parallel_config(cfg: dict[str, Any]) -> tuple[list[str], int | None]:
    parallel = cfg.get("parallel", {})
    gpu_ids_raw = parallel.get("gpu_ids", [])
    if isinstance(gpu_ids_raw, str):
        gpu_ids = _parse_gpu_ids(gpu_ids_raw)
    elif isinstance(gpu_ids_raw, list):
        gpu_ids = [str(item) for item in gpu_ids_raw]
    else:
        gpu_ids = []

    scenarios_per_node_raw = parallel.get("scenarios_per_node")
    scenarios_per_node = (
        int(scenarios_per_node_raw)
        if scenarios_per_node_raw is not None
        else None
    )
    return gpu_ids, scenarios_per_node


def _default_gpu_ids() -> list[str]:
    visible = _parse_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        return visible
    return ["0"]


def _scenario_indices(run_dir: str, selected: list[int] | None) -> list[int]:
    scenario_names = load_scenario_index(run_dir)
    if selected is None:
        return list(range(len(scenario_names)))
    for index in selected:
        resolve_scenario_name(run_dir, index)
    return selected


def _run_worker(run_dir: str, scenario_index: int, gpu_id: str) -> tuple[int, str, int]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["SLURM_ARRAY_TASK_ID"] = str(scenario_index)
    os.environ["POLARITON_SCRATCH_ID"] = (
        f"local_{os.getpid()}_gpu{gpu_id}_scenario{scenario_index}"
    )

    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"local_scenario_{scenario_index}_gpu{gpu_id}.out")
    err_path = os.path.join(logs_dir, f"local_scenario_{scenario_index}_gpu{gpu_id}.err")

    scenario_name = resolve_scenario_name(run_dir, scenario_index)
    argv_orig = sys.argv[:]
    sys.argv = [
        "run_scenario",
        "--run-dir",
        run_dir,
        "--scenario-index",
        str(scenario_index),
    ]

    try:
        with open(out_path, "w") as out, open(err_path, "w") as err:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                from pipeline.stages.gpu.run_scenario import main

                try:
                    main()
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
                    return scenario_index, scenario_name, code
                except BaseException:
                    traceback.print_exc()
                    return scenario_index, scenario_name, 1
        return scenario_index, scenario_name, 0
    finally:
        sys.argv = argv_orig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prepared scenarios locally across GPUs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids. Defaults to config parallel.gpu_ids, then CUDA_VISIBLE_DEVICES, then 0.",
    )
    parser.add_argument(
        "--scenarios-per-node",
        type=int,
        default=None,
        help="Maximum concurrent scenarios. Defaults to config parallel.scenarios_per_node, then number of GPU ids.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        action="append",
        default=None,
        help="0-based scenario index to run. May be repeated. Defaults to all scenarios.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    cfg_gpu_ids, cfg_scenarios_per_node = _parallel_config(cfg)

    gpu_ids = _parse_gpu_ids(args.gpu_ids) or cfg_gpu_ids or _default_gpu_ids()
    scenarios_per_node = (
        args.scenarios_per_node
        if args.scenarios_per_node is not None
        else cfg_scenarios_per_node
    )
    max_workers = scenarios_per_node or len(gpu_ids)
    max_workers = max(1, min(max_workers, len(gpu_ids)))

    indices = _scenario_indices(run_dir, args.scenario_index)
    if not indices:
        raise SystemExit("No scenarios to run.")

    print("=" * 60)
    print(" Local GPU scenario pool")
    print("=" * 60)
    print(f"  Run dir   : {run_dir}")
    print(f"  GPU ids   : {', '.join(gpu_ids)}")
    print(f"  Workers   : {max_workers}")
    print(f"  Scenarios : {len(indices)}")
    print("")

    failures: list[tuple[int, str, int]] = []
    mp_context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
        futures = {
            pool.submit(_run_worker, run_dir, index, gpu_ids[pos % len(gpu_ids)]): index
            for pos, index in enumerate(indices)
        }
        for future in as_completed(futures):
            index, name, code = future.result()
            if code == 0:
                print(f"  OK     [{index}] {name}")
            else:
                print(f"  FAILED [{index}] {name} code={code}", file=sys.stderr)
                failures.append((index, name, code))

    if failures:
        failed = ", ".join(f"{name}[{index}]={code}" for index, name, code in failures)
        raise SystemExit(f"Scenario failures: {failed}")


if __name__ == "__main__":
    main()
