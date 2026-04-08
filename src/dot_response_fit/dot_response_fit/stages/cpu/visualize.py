"""CPU stage: per-scenario visualisation (optional post-hoc use).

Reads ``<run_dir>/<scenario_name>.h5`` and writes plots to
``<run_dir>/results/<scenario_name>/``.  Requires ``archive_raw_hdf5=True``
in the output config — the inline GPU render covers the normal case.

Invoked as::

    python -m dot_response_fit.stages.cpu.visualize \\
        --run-dir <run_dir> [--scenario-index N] [--no-animation]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dot_response_fit.config.loader import load_config
from dot_response_fit.manifest.io import load_scenario_index
from dot_response_fit.stages.cpu.viz_engine import FIELD_SPECS, generate_field_png
from pipeline.render.nvenc_stream import generate_animation


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="per-scenario visualisation (post-hoc)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scenario-index", type=int, default=None)
    parser.add_argument("--no-animation", action="store_true")
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

    scenarios = load_scenario_index(run_dir)
    if scenario_index >= len(scenarios):
        print(
            f"ERROR: scenario_index={scenario_index} out of range ({len(scenarios)} scenarios)",
            file=sys.stderr,
        )
        sys.exit(1)
    scenario_name = scenarios[scenario_index]

    h5_path = os.path.join(run_dir, f"{scenario_name}.h5")
    if not os.path.isfile(h5_path):
        print(f"ERROR: HDF5 not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    cfg = load_config(config_path)
    global_cfg = cfg["global"]
    grid_cfg = global_cfg.get("grid", {})
    lx = float(grid_cfg.get("lx", 250.0))
    ly = float(grid_cfg.get("ly", 250.0))
    extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f" CPU Visualize: {scenario_name}")
    print("=" * 60)

    for field_key in FIELD_SPECS:
        try:
            generate_field_png(scenario_name, field_key, extent, run_dir, results_dir)
        except Exception as exc:
            print(f"  WARNING: {field_key} PNG failed: {exc}", file=sys.stderr)

    if not args.no_animation:
        generate_animation(scenario_name, FIELD_SPECS, extent, run_dir, results_dir)

    print(f"\n  Visualisation complete for '{scenario_name}'.")


if __name__ == "__main__":
    main()
