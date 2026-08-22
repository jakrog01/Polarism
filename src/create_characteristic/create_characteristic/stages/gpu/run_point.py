"""GPU Slurm array task: simulate one 2D grid point and write scalar results.

Invoked by Slurm as:
    python -m create_characteristic.stages.gpu.run_point \\
        --run-dir <run_dir> [--point-index <idx>]

When ``--point-index`` is omitted the task index is read from
``$SLURM_ARRAY_TASK_ID``.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters

from create_characteristic.config.loader import get_sweep_config, load_config
from create_characteristic.manifest.io import (
    atomic_write_json,
    load_point_index,
    point_result_path,
)
from create_characteristic.simulation.core import run_grid_point


def _point_trace_path(run_dir: str, point_index: int) -> str:
    return os.path.join(run_dir, "points", f"point_{point_index:06d}_trace.npz")


def _result_to_dict(result: Any) -> dict:
    return {
        "point_index": result.point_index,
        "energy_index": result.energy_index,
        "sep_index": result.sep_index,
        "pulse_energy": result.pulse_energy,
        "pulse_separation": result.pulse_separation,
        "total_time": result.total_time,
        "status": result.status,
        "psi_sq_max": result.psi_sq_max,
        "t_psi_sq_max": result.t_psi_sq_max,
        "n_steps_total": result.n_steps_total,
        "last_step_completed": result.last_step_completed,
        "diverged_at_step": result.diverged_at_step,
        "diverged_at_t": result.diverged_at_t,
        "wall_time_seconds": round(result.wall_time_seconds, 2),
        "n_gain_crossings": result.n_gain_crossings,
        "first_crossing_ps": result.first_crossing_ps,
        "nR_center_max": result.nR_center_max,
        "ratio_to_critical": result.ratio_to_critical,
        "n_active_max_domain": result.n_active_max_domain,
        "gain_check_every": result.gain_check_every,
        "klass": result.klass,
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU characteristic map task")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--point-index",
        type=int,
        default=None,
        help="Point index; falls back to $SLURM_ARRAY_TASK_ID when omitted.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    if args.point_index is not None:
        point_index = args.point_index
    else:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            print(
                "ERROR: --point-index not provided and $SLURM_ARRAY_TASK_ID is unset.",
                file=sys.stderr,
            )
            sys.exit(1)
        point_index = int(env_val)

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    sweep = get_sweep_config(cfg)

    grid_points = load_point_index(run_dir)
    if point_index >= len(grid_points):
        print(
            f"ERROR: point_index={point_index} out of range (have {len(grid_points)} points).",
            file=sys.stderr,
        )
        sys.exit(1)

    point = grid_points[point_index]
    pulse_energy = float(point["pulse_energy"])
    pulse_separation = float(point["pulse_separation"])

    if "total_time" in point:
        total_time = float(point["total_time"])
    else:
        total_time = float(cfg["global"]["solver"].get("total_time", 220.0))

    save_trace = cfg.get("output", {}).get("save_per_point_trace", False)

    print("=" * 60)
    print(" Characteristic Map GPU Task")
    print("=" * 60)
    print(f"  Run dir      : {run_dir}")
    print(f"  Point index  : {point_index}  /  {len(grid_points) - 1}")
    print(f"  Energy index : {point['energy_index']}  Sep index : {point['sep_index']}")
    print(f"  Pulse energy : {pulse_energy}")
    print(f"  Separation   : {pulse_separation} ps")
    print(f"  Total time   : {total_time} ps")
    print(f"  Save trace   : {save_trace}")
    print()

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    result = run_grid_point(
        point_index=point_index,
        energy_index=int(point["energy_index"]),
        sep_index=int(point["sep_index"]),
        pulse_energy=pulse_energy,
        pulse_separation=pulse_separation,
        total_time=total_time,
        cfg=cfg,
        scalar_check_every=int(sweep.get("scalar_check_every", 100)),
        gain_check_every=int(sweep.get("gain_check_every", 10)),
        early_stop_on_divergence=bool(sweep.get("early_stop_on_divergence", True)),
        save_trace=save_trace,
    )

    points_dir = os.path.join(run_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    out_path = point_result_path(run_dir, point_index)
    atomic_write_json(out_path, _result_to_dict(result))
    print(f"\n  Result written: {out_path}")

    if save_trace and result.scalar_times:
        trace_path = _point_trace_path(run_dir, point_index)
        np.savez_compressed(
            trace_path,
            t_ps=np.array(result.scalar_times, dtype=np.float64),
            psi_sq_max=np.array(result.scalar_psi_sq_max, dtype=np.float64),
            n_active_center=np.array(result.scalar_n_active_center, dtype=np.float64),
            gain_loss=np.array(result.scalar_gain_loss, dtype=np.float64),
        )
        print(f"  Trace written:  {trace_path}")

    print("\n  Task complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
