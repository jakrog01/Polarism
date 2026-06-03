"""CPU stage: cross-scenario finalize.

Aggregates per-scenario metadata into ``results_summary.json`` and generates
the cross-scenario comparison plot from scalar sidecars.  Does not require
per-scenario HDF5 files — reads only metadata JSON and ``*_scalars.npz``.

Invoked by Slurm as:
    python -m pipeline.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline.experiments.probe5_trap_gate import Probe5TrapGate
from pipeline.experiments.square4_fringe import Square4FringeExperiment
from pipeline.manifest.io import (
    atomic_write_json,
    load_scenario_index,
    load_scenario_meta,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.stages.cpu.viz_engine import (
    generate_summary,
    generate_sweep_diagnostics,
    generate_sweep_heatmaps,
)


def _check_artifacts(run_dir: str, scenarios: list[str]) -> list[str]:
    """Return error strings for any missing required per-scenario artifact."""
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
        sys.exit(1)

    threshold_path = os.path.join(run_dir, "threshold_result.json")
    with open(threshold_path) as f:
        threshold = json.load(f)

    is_sweep = threshold.get("mode") == "parameter_sweep"
    routines_summary: dict = {}
    sweep_metas: list[dict] = []
    effective_powers: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        t_cond = meta.get("t_cond")
        routines_summary[name] = {
            "t_cond": t_cond,
            "n_lasers": meta.get("n_lasers"),
            "phase_offsets": meta.get("phase_offsets"),
        }
        if meta.get("sweep"):
            sweep_metas.append(meta)
        if is_sweep and meta.get("effective_power") is not None:
            effective_powers[name] = meta["effective_power"]
        print(
            f"  {name}: t_cond="
            f"{'%.1f ps' % t_cond if t_cond is not None else 'NO CONDENSATION'}"
        )

    power_definitions: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        pd = meta.get("effective_power_definition")
        if pd:
            power_definitions[name] = pd

    summary = {
        "run_dir": run_dir,
        "mode": threshold.get("mode", "threshold_search"),
        "power_definition": threshold.get("power_definition", "peak_amplitude"),
        "P_threshold": threshold["P_threshold"],
        **({"effective_powers": effective_powers} if is_sweep else {}),
        **({"power_definitions": power_definitions} if power_definitions else {}),
        "routines": routines_summary,
    }
    summary_path = os.path.join(run_dir, "results_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    probe5_scenarios = [
        meta.get("scenario")
        for meta in sweep_metas
        if (meta.get("sweep") or {}).get("mode") == "probe5"
    ]
    probe5_scenarios = [name for name in probe5_scenarios if name]
    all_probe5 = len(probe5_scenarios) == len(scenarios)

    if all_probe5:
        print("  Skipping generic comparison plot for probe5 sweep.")
    else:
        print("  Generating cross-scenario comparison plot ...")
        try:
            generate_summary(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: summary plot failed: {e}", file=sys.stderr)

    if sweep_metas and not all_probe5:
        print("  Generating parameter-sweep heatmaps ...")
        try:
            generate_sweep_heatmaps(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: sweep heatmaps failed: {e}", file=sys.stderr)
        print("  Generating parameter-sweep diagnostics ...")
        try:
            generate_sweep_diagnostics(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: sweep diagnostics failed: {e}", file=sys.stderr)

    fringe_sidecars = [
        name for name in scenarios
        if os.path.isfile(os.path.join(run_dir, f"{name}_fringe.json"))
    ]
    if fringe_sidecars:
        print("  Generating fringe summary ...")
        try:
            Square4FringeExperiment().summarize(fringe_sidecars, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: fringe summary failed: {e}", file=sys.stderr)

    if probe5_scenarios:
        print("  Generating probe5 threshold summary ...")
        try:
            Probe5TrapGate().summarize(probe5_scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: probe5 summary failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "summary_path", summary_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")


if __name__ == "__main__":
    main()
