"""GPU stage: scenario simulation + inline rendering.

Runs one scenario (identified by ``$SLURM_ARRAY_TASK_ID`` or
``--scenario-index``), renders PNGs and an MP4 animation while the HDF5 is
still on local scratch, then persists only lightweight artifacts to the run
directory on ``/lu/tetyda``.  Raw HDF5 is copied to the run directory only
when ``archive_raw_hdf5: true`` is set in the config ``output`` section.

Invoked by Slurm as:
    python -m pipeline.stages.gpu.run_scenario --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

from pipeline.config.builder import build_scenario_config, build_scenario_lasers
from pipeline.config.loader import get_scenario, load_config
from pipeline.config.output_policy import output_policy_from_config
from pipeline.manifest.io import (
    atomic_write_json,
    resolve_scenario_name,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.render.nvenc_stream import generate_animation
from pipeline.simulation.core import RNG_SEED, run_simulation_from_config
from pipeline.stages.cpu.viz_engine import (
    FIELD_SPECS,
    generate_field_png,
    generate_scalar_traces,
)
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.grid.create_grid import create_grid


def _make_scratch_dir(run_dir: str, scenario_name: str) -> str:
    """Return a job-unique temp directory.

    Prefers ``$SCRATCH`` over Slurm/node-local temp directories because HDF5
    outputs can be large.
    """
    run_name = os.path.basename(run_dir.rstrip("/"))
    job_id = (
        os.environ.get("POLARITON_SCRATCH_ID")
        or os.environ.get("SLURM_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or "local"
    )
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    unique_id = f"{job_id}_{task_id}"

    for env_var in ("SCRATCH", "SLURM_TMPDIR", "TMPDIR"):
        base = os.environ.get(env_var)
        if base and os.path.isdir(base):
            if os.path.realpath(base) == "/tmp":
                continue
            scratch = os.path.join(base, "polariton", run_name, unique_id, scenario_name)
            os.makedirs(scratch, exist_ok=False)
            return scratch

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError(
            "No suitable scratch directory found for the GPU scenario job. "
            "Set and export SCRATCH in slurm.env or provide SLURM_TMPDIR/TMPDIR "
            "that points to real job scratch, not /tmp."
        )

    return run_dir


def _render_artifacts(
    scenario_name: str,
    data_dir: str,
    scratch_results_dir: str,
    extent: list[float],
    render_snapshots: bool,
    render_animation: bool,
) -> None:
    """Generate PNGs and animation from scratch-local HDF5."""
    if render_snapshots:
        for field_key in FIELD_SPECS:
            print(f"  PNG {field_key} ...", end="", flush=True)
            try:
                generate_field_png(
                    scenario_name, field_key, extent, data_dir, scratch_results_dir
                )
                print(" done")
            except Exception as e:
                print(f" WARNING: {e}", file=sys.stderr)

    if render_animation:
        print("  animation ...", flush=True)
        generate_animation(
            scenario_name, FIELD_SPECS, extent, data_dir, scratch_results_dir
        )


def _copyback_artifacts(
    scratch_dir: str,
    run_dir: str,
    scenario_name: str,
    archive_raw_hdf5: bool,
) -> float:
    """Copy lightweight artifacts from scratch to run_dir; return seconds elapsed."""
    t0 = time.monotonic()

    src_scenario_results = os.path.join(scratch_dir, "results", scenario_name)
    dst_scenario_results = os.path.join(run_dir, "results", scenario_name)
    if os.path.isdir(src_scenario_results):
        os.makedirs(os.path.dirname(dst_scenario_results), exist_ok=True)
        if os.path.isdir(dst_scenario_results):
            shutil.rmtree(dst_scenario_results)
        shutil.copytree(src_scenario_results, dst_scenario_results)

    src_sidecar = os.path.join(scratch_dir, f"{scenario_name}_scalars.npz")
    if os.path.isfile(src_sidecar):
        shutil.copy2(src_sidecar, os.path.join(run_dir, f"{scenario_name}_scalars.npz"))

    if archive_raw_hdf5:
        src_h5 = os.path.join(scratch_dir, f"{scenario_name}.h5")
        if os.path.isfile(src_h5):
            dst_h5 = os.path.join(run_dir, f"{scenario_name}.h5")
            tmp_dst = dst_h5 + ".tmp"
            try:
                shutil.copy2(src_h5, tmp_dst)
                os.replace(tmp_dst, dst_h5)
            except Exception:
                traceback.print_exc()
                if os.path.isfile(tmp_dst):
                    try:
                        os.remove(tmp_dst)
                    except OSError:
                        pass
                raise RuntimeError(
                    f"HDF5 archive copy failed: {src_h5} -> {dst_h5}"
                )

    return time.monotonic() - t0


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU scenario simulation + render (array task)")
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

    output_policy = output_policy_from_config(cfg)
    lx: float = threshold["lx"]
    ly: float = threshold["ly"]
    extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

    global_cfg = cfg["global"]
    scenario = get_scenario(cfg, scenario_name)

    print("=" * 60)
    print(f" GPU Scenario: {scenario_name}  (index {scenario_index})")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    print(f"  P_threshold : {threshold['P_threshold']:.1f}")
    print(f"  archive_h5  : {output_policy.archive_raw_hdf5}")
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

    required_time = max(
        (
            laser.delay
            + (max(laser.n_pulses, 1) - 1) * laser.pulse_separation
            + 2.0 * laser.cutoff_sigma * laser.sigma_time
            if getattr(laser, "n_pulses", 0) > 0
            else laser.delay + 2.0 * laser.cutoff_sigma * laser.sigma_time
        )
        for laser in lasers
    )
    if required_time > sim_cfg.solver.total_time:
        print(
            "ERROR: scenario timing does not fit inside the configured simulation "
            f"window. Need at least {required_time:.1f} ps, but total_time="
            f"{sim_cfg.solver.total_time:.1f} ps.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_steps = int(sim_cfg.solver.total_time / sim_cfg.solver.dt)
    print(f"  Steps: {n_steps:,}")

    try:
        scratch_dir = _make_scratch_dir(run_dir, scenario_name)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    using_scratch = scratch_dir != run_dir
    if using_scratch:
        print(f"  Scratch: {scratch_dir}")

    try:
        print("\n  Running simulation ...")
        t_sim_start = time.monotonic()
        t_cond, sidecar_path = run_simulation_from_config(
            scenario_name, lasers, sim_cfg, scratch_dir, output_policy,
        )
        elapsed_sim = time.monotonic() - t_sim_start
    except Exception:
        traceback.print_exc()
        if using_scratch and os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)
        sys.exit(1)

    print("\n  Rendering artifacts from scratch ...")
    scratch_results_dir = os.path.join(scratch_dir, "results")
    try:
        print("  scalar traces ...", end="", flush=True)
        generate_scalar_traces(scenario_name, scratch_dir, scratch_results_dir)
        print(" done")
    except Exception as e:
        print(f" WARNING: {e}", file=sys.stderr)
    _render_artifacts(
        scenario_name, scratch_dir, scratch_results_dir,
        extent,
        output_policy.render_snapshots,
        output_policy.render_animation,
    )

    copyback_seconds: float = 0.0
    if using_scratch:
        print(f"\n  Copying artifacts scratch → {run_dir} ...")
        try:
            copyback_seconds = _copyback_artifacts(
                scratch_dir, run_dir, scenario_name, output_policy.archive_raw_hdf5
            )
        except Exception:
            traceback.print_exc()
            print(
                "ERROR: artifact copy-back failed; preserving scratch for recovery: "
                f"{scratch_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Copy-back done  ({copyback_seconds:.1f}s)")
        shutil.rmtree(scratch_dir, ignore_errors=True)

    h5_final = os.path.join(run_dir, f"{scenario_name}.h5")
    h5_bytes = os.path.getsize(h5_final) if os.path.isfile(h5_final) else 0
    meta: dict = {
        "scenario": scenario_name,
        "scenario_index": scenario_index,
        "P_threshold": threshold["P_threshold"],
        "t_cond": t_cond,
        "h5_file": f"{scenario_name}.h5" if output_policy.archive_raw_hdf5 else None,
        "sidecar_file": f"{scenario_name}_scalars.npz",
        "n_lasers": len(lasers),
        "phase_offsets": phases,
        "sweep": scenario.get("sweep"),
        "lasers": [
            {
                "x0": float(laser.x0), "y0": float(laser.y0),
                "P0": float(laser.P0), "sigma_time": float(laser.sigma_time),
                "sigma_space": float(laser.sigma_space),
                "pulse_separation": float(laser.pulse_separation),
                "n_pulses": int(getattr(laser, "n_pulses", 0)),
                "delay": float(laser.delay),
            }
            for laser in lasers
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "telemetry": {
            "elapsed_sim_seconds": round(elapsed_sim, 2),
            "n_steps": n_steps,
            "h5_bytes": h5_bytes,
            "copyback_seconds": round(copyback_seconds, 2),
        },
    }
    atomic_write_json(scenario_meta_path(run_dir, scenario_name), meta)
    print(f"  Metadata: {scenario_meta_path(run_dir, scenario_name)}")
    print(f"\n  Scenario '{scenario_name}' complete. t_cond={t_cond}")


if __name__ == "__main__":
    main()
