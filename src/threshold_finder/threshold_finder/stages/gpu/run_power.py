"""GPU Slurm array task: simulate one power point and write scalar results.

Invoked by Slurm as:
    python -m threshold_finder.stages.gpu.run_power \\
        --run-dir <run_dir> [--power-index <idx>]

When ``--power-index`` is omitted the task index is read from
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

from threshold_finder.config.loader import get_sweep_config, load_config
from threshold_finder.manifest.io import (
    atomic_write_json,
    load_power_index,
    power_result_path,
    power_trace_path,
)
from threshold_finder.simulation.sweep_core import run_power_point


def _result_to_dict(result: Any) -> dict:
    return {
        "power_index": result.power_index,
        "P": result.P,
        "status": result.status,
        "psi_sq_max": result.psi_sq_max,
        "t_psi_sq_max": result.t_psi_sq_max,
        "n_steps_total": result.n_steps_total,
        "last_step_completed": result.last_step_completed,
        "diverged_at_step": result.diverged_at_step,
        "diverged_at_t": result.diverged_at_t,
        "wall_time_seconds": round(result.wall_time_seconds, 2),
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU power-sweep task")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--power-index",
        type=int,
        default=None,
        help="Power index; falls back to $SLURM_ARRAY_TASK_ID when omitted.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    if args.power_index is not None:
        power_index = args.power_index
    else:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            print(
                "ERROR: --power-index not provided and $SLURM_ARRAY_TASK_ID is unset.",
                file=sys.stderr,
            )
            sys.exit(1)
        power_index = int(env_val)

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    sweep = get_sweep_config(cfg)

    power_values = load_power_index(run_dir)
    if power_index >= len(power_values):
        print(
            f"ERROR: power_index={power_index} out of range (have {len(power_values)} powers).",
            file=sys.stderr,
        )
        sys.exit(1)

    P = power_values[power_index]
    save_trace = cfg.get("output", {}).get("save_per_power_trace", True)

    print("=" * 60)
    print(" Power-Sweep GPU Task")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    print(f"  Power index : {power_index}  /  {len(power_values) - 1}")
    print(f"  P           : {P}")
    print(f"  Save trace  : {save_trace}")
    print()

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    result = run_power_point(
        power_index=power_index,
        P=P,
        cfg=cfg,
        scalar_check_every=int(sweep.get("scalar_check_every", 50)),
        early_stop_on_divergence=bool(sweep.get("early_stop_on_divergence", True)),
        save_trace=save_trace,
    )

    powers_dir = os.path.join(run_dir, "powers")
    os.makedirs(powers_dir, exist_ok=True)

    out_path = power_result_path(run_dir, power_index)
    atomic_write_json(out_path, _result_to_dict(result))
    print(f"\n  Result written: {out_path}")

    if save_trace and result.scalar_times:
        trace_path = power_trace_path(run_dir, power_index)
        np.savez_compressed(
            trace_path,
            time=np.array(result.scalar_times, dtype=np.float64),
            psi_sq_max=np.array(result.scalar_psi_sq_max, dtype=np.float64),
        )
        print(f"  Trace written:  {trace_path}")

    print("\n  Task complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
