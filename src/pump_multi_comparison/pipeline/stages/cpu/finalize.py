"""CPU stage: cross-scenario finalize.

Aggregates per-scenario metadata into ``results_summary.json`` and generates
the cross-scenario comparison plot from scalar sidecars.  Does not require
per-scenario HDF5 files — reads only metadata JSON and ``*_scalars.npz``.

Invoked by Slurm as:
    python -m pipeline.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

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


_FRINGE_CSV_COLUMNS = [
    "scenario", "square_side_um", "energy", "sigma_space",
    "fringe_contrast_max", "t_fringe_contrast_max_ps",
    "fringe_spacing_at_max_contrast_um", "fringe_fft_peak_k_at_max_contrast",
    "fringe_cv_max", "h_contrast_max", "v_contrast_max",
    "fringe_window_psi_sq_max", "t_fringe_window_psi_sq_max_ps",
    "central_roi_peak_psi_sq", "central_roi_peak_emission",
    "crossed_threshold",
]


def _load_fringe_json(run_dir: str, scenario_name: str) -> dict | None:
    path = os.path.join(run_dir, f"{scenario_name}_fringe.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_sidecar_roi_peaks(
    run_dir: str, scenario_name: str
) -> dict[str, float]:
    path = os.path.join(run_dir, f"{scenario_name}_scalars.npz")
    if not os.path.isfile(path):
        return {}
    data = np.load(path)
    out: dict[str, float] = {}
    prefix = "roi_center_D_circle"
    for suffix, key in (
        ("_mean_psi_sq", "central_roi_peak_psi_sq"),
        ("_integral_emission", "central_roi_peak_emission"),
    ):
        arr_key = f"{prefix}{suffix}"
        if arr_key in data:
            out[key] = float(data[arr_key].max())
    return out


def _generate_fringe_summary(
    scenarios: list[str], run_dir: str, results_dir: str
) -> None:
    rows: list[dict] = []
    for name in scenarios:
        fringe = _load_fringe_json(run_dir, name)
        if fringe is None or "aggregated" not in fringe:
            continue
        meta = load_scenario_meta(run_dir, name)
        sweep = meta.get("sweep") or {}
        agg = fringe["aggregated"]
        roi = _load_sidecar_roi_peaks(run_dir, name)
        row: dict = {
            "scenario": name,
            "square_side_um": sweep.get("square_side", float("nan")),
            "energy": sweep.get("power", float("nan")),
            "sigma_space": sweep.get("sigma_space", float("nan")),
            "fringe_contrast_max": agg.get("fringe_contrast_max", float("nan")),
            "t_fringe_contrast_max_ps": agg.get("t_fringe_contrast_max", float("nan")),
            "fringe_spacing_at_max_contrast_um": agg.get("fringe_spacing_at_max_contrast", float("nan")),
            "fringe_fft_peak_k_at_max_contrast": agg.get("fringe_fft_peak_k_at_max_contrast", float("nan")),
            "fringe_cv_max": agg.get("fringe_cv_max", float("nan")),
            "h_contrast_max": agg.get("h_contrast_max", float("nan")),
            "v_contrast_max": agg.get("v_contrast_max", float("nan")),
            "fringe_window_psi_sq_max": agg.get("fringe_window_psi_sq_max", float("nan")),
            "t_fringe_window_psi_sq_max_ps": agg.get("t_fringe_window_psi_sq_max", float("nan")),
            "central_roi_peak_psi_sq": roi.get("central_roi_peak_psi_sq", float("nan")),
            "central_roi_peak_emission": roi.get("central_roi_peak_emission", float("nan")),
            "crossed_threshold": agg.get("crossed_threshold", False),
        }
        rows.append(row)

    if not rows:
        return

    csv_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FRINGE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.json")
    atomic_write_json(json_path, rows)

    _generate_selected_extremes(rows, results_dir)
    print(f"  Fringe summary: {csv_path}  ({len(rows)} scenarios)")


def _generate_selected_extremes(rows: list[dict], results_dir: str) -> None:
    if not rows:
        return

    def _pick(key: str, best_fn) -> dict | None:
        valid = [r for r in rows if not _is_nan(r.get(key))]
        return best_fn(valid, key=lambda r: r[key]) if valid else None

    def _is_nan(v) -> bool:
        try:
            return v is None or float(v) != float(v)
        except (TypeError, ValueError):
            return True

    above = [r for r in rows if r.get("crossed_threshold")]
    below = [r for r in rows if not r.get("crossed_threshold")]

    extremes: dict = {}
    r = _pick("central_roi_peak_psi_sq", max)
    if r:
        extremes["max_central_psi_sq"] = r
    r = _pick("central_roi_peak_psi_sq", min)
    if r:
        extremes["min_central_psi_sq"] = r
    r = _pick("fringe_contrast_max", max)
    if r:
        extremes["max_fringe_contrast"] = r
    if above:
        extremes["first_above_threshold"] = min(
            above, key=lambda r: r.get("energy", float("inf"))
        )
    if below:
        valid_below = [r for r in below if not _is_nan(r.get("central_roi_peak_psi_sq"))]
        if valid_below:
            extremes["closest_below_threshold"] = max(
                valid_below, key=lambda r: r["central_roi_peak_psi_sq"]
            )

    atomic_write_json(os.path.join(results_dir, "selected_extremes.json"), extremes)


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

    import json
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
    print("  Generating cross-scenario comparison plot ...")
    try:
        generate_summary(scenarios, run_dir, results_dir)
    except Exception as e:
        print(f"  WARNING: summary plot failed: {e}", file=sys.stderr)

    if sweep_metas:
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
            _generate_fringe_summary(fringe_sidecars, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: fringe summary failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "summary_path", summary_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")


if __name__ == "__main__":
    main()
