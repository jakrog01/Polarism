"""CPU Stage 4: cross-scenario finalise.

Aggregates per-scenario metadata into ``results_summary.json`` and generates
the cross-scenario comparison plot from scalar sidecars.  Does not require
per-scenario HDF5 files — reads only metadata JSON and ``*_scalars.npz``.

Invoked as::

    python -m dot_response_fit.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dot_response_fit.config.loader import load_config
from dot_response_fit.manifest.io import (
    atomic_write_json,
    load_scenario_index,
    load_scenario_meta,
    scenario_meta_path,
    set_manifest_field,
)
from dot_response_fit.stages.cpu.viz_engine import generate_summary


def _check_artifacts(run_dir: str, scenarios: list[str]) -> list[str]:
    """Return error strings for any missing per-scenario artifact."""
    errors: list[str] = []
    for name in scenarios:
        meta = scenario_meta_path(run_dir, name)
        sidecar = os.path.join(run_dir, f"{name}_scalars.npz")
        if not os.path.isfile(meta):
            errors.append(f"Missing metadata for scenario '{name}': {meta}")
        if not os.path.isfile(sidecar):
            errors.append(f"Missing scalar sidecar for scenario '{name}': {sidecar}")
    return errors


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU Stage 4: finalise and summarise")
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
        sys.exit(1)

    fit_path = os.path.join(run_dir, "fit_result.json")
    fit_result: dict = {}
    if os.path.isfile(fit_path):
        with open(fit_path) as f:
            fit_result = json.load(f)

    routines_summary: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        t_cond = meta.get("t_cond")
        routines_summary[name] = {
            "t_cond": t_cond,
            "n_lasers": meta.get("n_lasers"),
            "sigma_space": meta.get("sigma_space"),
            "fit_score": meta.get("fit_score"),
        }
        print(
            f"  {name}: t_cond="
            f"{'%.1f ps' % t_cond if t_cond is not None else 'NO CONDENSATION'}"
            f"  sigma_space={meta.get('sigma_space')}"
        )

    summary = {
        "run_dir": run_dir,
        "fit_result": fit_result,
        "routines": routines_summary,
    }
    summary_path = os.path.join(run_dir, "results_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    print("  Generating cross-scenario comparison plot ...")
    try:
        generate_summary(scenarios, run_dir, results_dir)
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
