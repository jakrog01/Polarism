"""CPU stage: cross-scenario finalize.

Validates that all expected scenario artifacts are present, aggregates
per-scenario metadata into ``results_summary.json``, and generates the
cross-scenario comparison plot.

Depends (via Slurm ``afterok``) on ALL scenario GPU jobs.  If any scenario
artifact is missing the stage exits nonzero with an explicit error rather
than silently producing a partial summary.

Invoked by Slurm as:
    python -m pipeline.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline.manifest.io import (
    atomic_write_json,
    load_scenario_index,
    load_scenario_meta,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.stages.cpu.viz_engine import generate_summary


def _check_artifacts(run_dir: str, scenarios: list[str]) -> list[str]:
    """Return error strings for any missing required per-scenario artifact."""
    errors: list[str] = []
    for name in scenarios:
        h5 = os.path.join(run_dir, f"{name}.h5")
        meta = scenario_meta_path(run_dir, name)
        if not os.path.isfile(h5):
            errors.append(f"Missing HDF5 for scenario '{name}': {h5}")
        if not os.path.isfile(meta):
            errors.append(f"Missing metadata for scenario '{name}': {meta}")
    return errors


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU finalize / cross-scenario summary")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    try:
        scenarios = load_scenario_index(run_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" CPU Finalize")
    print("=" * 60)
    print(f"  Run dir   : {run_dir}")
    print(f"  Scenarios : {scenarios}")

    errors = _check_artifacts(run_dir, scenarios)
    if errors:
        print("\nFINALIZE ABORTED — missing artifacts:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(
            "\n  One or more scenario GPU jobs likely failed.  "
            "Inspect per-scenario logs and the run manifest.",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold_path = os.path.join(run_dir, "threshold_result.json")
    with open(threshold_path) as f:
        threshold = json.load(f)
    lx: float = threshold["lx"]
    ly: float = threshold["ly"]
    extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

    routines_summary: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        t_cond = meta.get("t_cond")
        routines_summary[name] = {
            "t_cond": t_cond,
            "h5_file": meta.get("h5_file"),
            "n_lasers": meta.get("n_lasers"),
            "phase_offsets": meta.get("phase_offsets"),
        }
        print(
            f"  {name}: t_cond="
            f"{'%.1f ps' % t_cond if t_cond is not None else 'NO CONDENSATION'}"
        )

    summary = {
        "run_dir": run_dir,
        "P_threshold": threshold["P_threshold"],
        "routines": routines_summary,
    }
    summary_path = os.path.join(run_dir, "results_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    print("  Generating cross-scenario comparison plot ...")
    try:
        generate_summary(extent, routines=scenarios, data_dir=run_dir, results_dir=results_dir)
    except Exception as e:
        print(f"  WARNING: summary plot failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "summary_path", summary_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")


if __name__ == "__main__":
    main()
