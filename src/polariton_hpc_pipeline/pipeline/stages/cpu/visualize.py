"""CPU stage: per-scenario visualization (optional post-hoc use).

Reads ``<run_dir>/<scenario_name>.h5`` and writes plots to
``<run_dir>/results/<scenario_name>/``.  Requires ``archive_raw_hdf5=True``
in the output config — the inline GPU render covers the normal case.

Invoked as:
    python -m pipeline.stages.cpu.visualize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline.manifest.io import resolve_scenario_name, scenario_meta_path
from pipeline.render.nvenc_stream import generate_animation
from pipeline.stages.cpu.viz_engine import FIELD_SPECS, generate_field_png


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU visualization for one scenario")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scenario-index", type=int, default=None,
        help="0-based index into scenario_index.json.  "
             "Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--no-animation", action="store_true", default=False,
        help="Skip animation generation.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

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

    h5_path = os.path.join(run_dir, f"{scenario_name}.h5")
    if not os.path.isfile(h5_path):
        print(
            f"ERROR: HDF5 not found: {h5_path}\n"
            "  GPU simulation job may have failed or not yet completed.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(scenario_meta_path(run_dir, scenario_name)):
        print(
            f"ERROR: scenario metadata not found: {scenario_meta_path(run_dir, scenario_name)}",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold_path = os.path.join(run_dir, "threshold_result.json")
    with open(threshold_path) as f:
        threshold = json.load(f)
    lx: float = threshold["lx"]
    ly: float = threshold["ly"]
    extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

    results_dir = os.path.join(run_dir, "results")

    print("=" * 60)
    print(f" CPU Visualization: {scenario_name}  (index {scenario_index})")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    print(f"  Results dir : {results_dir}")
    print(f"  Extent      : {extent}")

    for field_key in FIELD_SPECS:
        print(f"  {field_key} ...", end="", flush=True)
        try:
            generate_field_png(scenario_name, field_key, extent,
                               data_dir=run_dir, results_dir=results_dir)
            print(" done")
        except Exception as e:
            print(f" WARNING: {e}", file=sys.stderr)

    if args.no_animation:
        print("  animation ... skipped (--no-animation)")
    else:
        print("  animation ...", flush=True)
        generate_animation(scenario_name, FIELD_SPECS, extent, data_dir=run_dir, results_dir=results_dir)

    print(f"\n  Visualization for '{scenario_name}' complete.")


if __name__ == "__main__":
    main()
