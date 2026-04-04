"""GPU stage: scenario simulation.

Runs one scenario identified by ``$SLURM_ARRAY_TASK_ID`` (or
``--scenario-index``).  Reads scenario name from
``<run_dir>/scenario_index.json``.  Writes HDF5 output and a metadata
sidecar to the run directory.  Exits nonzero on any failure.

Temporary data is written to a job-unique subdirectory of ``$SCRATCH``
(preferred for large HDF5 files) or ``$TMPDIR``, then atomically copied back.
Never shares a temp directory with another array task.

Invoked by Slurm as:
    python -m pipeline.stages.gpu.run_scenario --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

from pipeline.config.builder import build_scenario_config, build_scenario_lasers
from pipeline.config.loader import get_scenario, load_config
from pipeline.manifest.io import (
    atomic_write_json,
    resolve_scenario_name,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.simulation.core import RNG_SEED, compute_batch_size, run_simulation_from_config
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.grid.create_grid import create_grid


def _make_scratch_dir(run_dir: str, scenario_name: str) -> str:
    """Return a job-unique temp directory.

    Layout: ``<base>/polariton/<run_name>/<job_id>_<task_id>/<scenario>/``

    Prefers ``$SCRATCH`` over ``$TMPDIR`` because HDF5 outputs can be large.
    Returns *run_dir* itself only as a last resort (no scratch configured).
    """
    run_name = os.path.basename(run_dir.rstrip("/"))
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    unique_id = f"{job_id}_{task_id}"

    for env_var in ("SCRATCH", "TMPDIR"):
        base = os.environ.get(env_var)
        if base and os.path.isdir(base):
            scratch = os.path.join(base, "polariton", run_name, unique_id, scenario_name)
            os.makedirs(scratch, exist_ok=False)
            return scratch

    # No external scratch: write directly to the run directory.
    return run_dir


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU scenario simulation (array task)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scenario-index", type=int, default=None,
        help="0-based index into scenario_index.json.  "
             "Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve array task index.
    if args.scenario_index is not None:
        scenario_index = args.scenario_index
    else:
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task is None:
            print(
                "ERROR: --scenario-index not given and SLURM_ARRAY_TASK_ID not set.",
                file=sys.stderr,
            )
            sys.exit(1)
        scenario_index = int(slurm_task)

    try:
        scenario_name = resolve_scenario_name(run_dir, scenario_index)
    except (FileNotFoundError, IndexError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Load config and threshold from the run directory (canonical snapshot).
    config_path = os.path.join(run_dir, "config.yaml")
    threshold_path = os.path.join(run_dir, "threshold_result.json")
    for path in (config_path, threshold_path):
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(config_path)
    with open(threshold_path) as f:
        threshold: dict = json.load(f)

    if not threshold.get("search_completed"):
        print("ERROR: threshold search did not complete successfully.", file=sys.stderr)
        sys.exit(1)

    global_cfg = cfg["global"]
    scenario = get_scenario(cfg, scenario_name)

    print("=" * 60)
    print(f" GPU Scenario: {scenario_name}  (index {scenario_index})")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    print(f"  P_threshold : {threshold['P_threshold']:.1f}")
    print(
        f"  SLURM job   : "
        f"{os.environ.get('SLURM_JOB_ID', 'N/A')} / "
        f"task {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}"
    )

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    sim_cfg = build_scenario_config(global_cfg, threshold, scenario)
    grid = create_grid(sim_cfg.grid)

    rng = np.random.default_rng(RNG_SEED)
    lasers, phases = build_scenario_lasers(scenario, global_cfg, threshold, grid, rng)

    print(f"  Lasers: {len(lasers)}")
    for i, laser in enumerate(lasers):
        print(
            f"    [{i}] P={laser.P0:.2f}  "
            f"pos=({laser.x0:.1f}, {laser.y0:.1f})  "
            f"delay={laser.delay:.3f} ps"
        )

    batch_size = compute_batch_size(grid.ny, grid.nx)
    n_steps = int(sim_cfg.solver.total_time / sim_cfg.solver.dt)
    print(f"  Batch: {batch_size},  Steps: {n_steps:,}")

    scratch_dir = _make_scratch_dir(run_dir, scenario_name)
    using_scratch = scratch_dir != run_dir
    if using_scratch:
        print(f"  Scratch: {scratch_dir}")

    h5_final = os.path.join(run_dir, f"{scenario_name}.h5")

    try:
        print("\n  Running simulation ...")
        t_cond = run_simulation_from_config(
            scenario_name, lasers, sim_cfg, batch_size, scratch_dir,
        )
    except Exception:
        if using_scratch and os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)
        sys.exit(1)

    if using_scratch:
        src = os.path.join(scratch_dir, f"{scenario_name}.h5")
        if not os.path.isfile(src):
            print(f"ERROR: expected HDF5 not found at {src}", file=sys.stderr)
            shutil.rmtree(scratch_dir, ignore_errors=True)
            sys.exit(1)
        tmp_dst = h5_final + ".tmp"
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, h5_final)
        print(f"  Copied {src}\n       → {h5_final}")
        shutil.rmtree(scratch_dir, ignore_errors=True)

    meta: dict = {
        "scenario": scenario_name,
        "scenario_index": scenario_index,
        "P_threshold": threshold["P_threshold"],
        "t_cond": t_cond,
        "h5_file": f"{scenario_name}.h5",
        "n_lasers": len(lasers),
        "phase_offsets": phases,
        "lasers": [
            {
                "x0": float(laser.x0), "y0": float(laser.y0),
                "P0": float(laser.P0), "sigma_time": float(laser.sigma_time),
                "pulse_separation": float(laser.pulse_separation),
                "delay": float(laser.delay),
            }
            for laser in lasers
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_write_json(scenario_meta_path(run_dir, scenario_name), meta)
    print(f"  Metadata: {scenario_meta_path(run_dir, scenario_name)}")
    print(f"\n  Scenario '{scenario_name}' complete. t_cond={t_cond}")


if __name__ == "__main__":
    main()
