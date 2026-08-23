"""CPU stage: aggregate per-point results into 2D characteristic map artifacts.

Reads all ``run_dir/points/point_*.json`` files and writes:
  - ``characteristic_map.csv``
  - ``characteristic_map.json``
  - ``crossing_summary.json``
  - ``results/psi_threshold_crossings_heatmap.png``

Invoked as:
    python -m create_characteristic.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np

from polarism.analysis.condensation import CONDENSATION_PSI_SQ_FLOOR
from create_characteristic.manifest.io import atomic_write_json, set_manifest_field

DEFAULT_THRESHOLD_CRITERION = CONDENSATION_PSI_SQ_FLOOR


def _load_all_results(run_dir: str) -> list[dict[str, Any]]:
    pattern = os.path.join(run_dir, "points", "point_*.json")
    paths = sorted(glob.glob(pattern))
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: (int(r["energy_index"]), int(r["sep_index"])))


def _build_axes(results: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    energies = sorted({float(r["pulse_energy"]) for r in results})
    separations = sorted({float(r["pulse_separation"]) for r in results})
    return np.array(energies), np.array(separations)


def _build_threshold_crossing_array(
    results: list[dict[str, Any]],
    energies: np.ndarray,
    separations: np.ndarray,
) -> np.ndarray:
    psi_threshold_map = np.full((len(energies), len(separations)), np.nan)
    energy_index = {value: index for index, value in enumerate(energies)}
    separation_index = {value: index for index, value in enumerate(separations)}
    for result in results:
        if result["status"] != "ok":
            continue
        ei = energy_index[float(result["pulse_energy"])]
        si = separation_index[float(result["pulse_separation"])]
        psi_threshold_map[ei, si] = float(result.get("n_psi_sq_threshold_crossings", 0))
    return psi_threshold_map


def _write_csv(results: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "point_index", "energy_index", "sep_index",
        "pulse_energy", "pulse_separation", "total_time",
        "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
        "n_gain_crossings", "first_crossing_ps",
        "n_psi_sq_threshold_crossings", "first_psi_sq_threshold_crossing_ps",
        "nR_center_max",
        "ratio_to_critical", "n_active_max_domain", "gain_check_every", "klass",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_crossings_heatmap(
    energies: np.ndarray,
    separations: np.ndarray,
    crossing_map: np.ndarray,
    out_path: str,
    label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    finite_crossings = crossing_map[np.isfinite(crossing_map)]
    maximum = max(1, int(finite_crossings.max())) if finite_crossings.size else 1
    boundaries = np.arange(-0.5, maximum + 1.5, 1.0)
    mesh = ax.pcolormesh(
        *np.meshgrid(separations, energies),
        crossing_map,
        shading="nearest",
        cmap="viridis",
        norm=BoundaryNorm(boundaries, ncolors=256),
    )
    fig.colorbar(mesh, ax=ax, label=label)
    try:
        ax.contour(*np.meshgrid(separations, energies), crossing_map >= 1.0, levels=[0.5], colors="white")
    except ValueError:
        pass
    ax.set(xlabel="Pulse separation (ps)", ylabel="Pulse energy", title=title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_crossing_summary(
    energies: np.ndarray,
    separations: np.ndarray,
    threshold_criterion: float,
    n_ok: int,
    n_div: int,
    psi_threshold_map: np.ndarray,
) -> dict[str, Any]:
    finite_crossings = psi_threshold_map[np.isfinite(psi_threshold_map)]
    return {
        "criterion": "all upward and downward crossings of threshold_criterion by max_abs_psi_sq",
        "threshold_criterion": threshold_criterion,
        "n_total": n_ok + n_div,
        "n_ok": n_ok,
        "n_diverged": n_div,
        "energy_axis": energies.tolist(),
        "separation_axis": separations.tolist(),
        "psi_threshold_crossings_total": int(np.nansum(psi_threshold_map)),
        "psi_threshold_crossings_max": int(finite_crossings.max()) if finite_crossings.size else None,
        "psi_threshold_crossings": psi_threshold_map.tolist(),
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU finalize: characteristic map aggregation")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    print("=" * 60)
    print(" Characteristic Map Finalize")
    print("=" * 60)
    print(f"  Run dir : {run_dir}")

    results = _load_all_results(run_dir)
    if not results:
        print("ERROR: no per-point result files found in run_dir/points/.", file=sys.stderr)
        sys.exit(1)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_div = sum(1 for r in results if r["status"] == "diverged")
    print(f"  Loaded  : {len(results)} results  ({n_ok} ok, {n_div} diverged)")

    cfg_path = os.path.join(run_dir, "config.yaml")
    threshold_criterion = DEFAULT_THRESHOLD_CRITERION
    if os.path.isfile(cfg_path):
        import yaml
        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f)
        threshold_criterion = float(
            raw_cfg.get("output", {}).get("threshold_criterion", DEFAULT_THRESHOLD_CRITERION)
        )

    csv_path = os.path.join(run_dir, "characteristic_map.csv")
    _write_csv(results, csv_path)
    print(f"  CSV     : {csv_path}")

    json_path = os.path.join(run_dir, "characteristic_map.json")
    atomic_write_json(json_path, results)
    print(f"  JSON    : {json_path}")

    energies, separations = _build_axes(results)
    psi_threshold_map = _build_threshold_crossing_array(results, energies, separations)

    summary = _build_crossing_summary(
        energies,
        separations,
        threshold_criterion,
        n_ok,
        n_div,
        psi_threshold_map,
    )
    crossing_path = os.path.join(run_dir, "crossing_summary.json")
    atomic_write_json(crossing_path, summary)
    print(f"  Summary : {crossing_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    psi_threshold_heatmap_path = os.path.join(results_dir, "psi_threshold_crossings_heatmap.png")
    try:
        _write_crossings_heatmap(
            energies,
            separations,
            psi_threshold_map,
            psi_threshold_heatmap_path,
            r"$|\psi|^2$ threshold crossings",
            r"$|\psi|^2$ threshold-crossing map",
        )
    except Exception as e:
        print(f"  WARNING: crossing heatmap generation failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "n_ok", n_ok)
        set_manifest_field(run_dir, "n_diverged", n_div)
        set_manifest_field(run_dir, "threshold_criterion", threshold_criterion)
        set_manifest_field(run_dir, "csv_path", csv_path)
        set_manifest_field(run_dir, "json_path", json_path)
        set_manifest_field(run_dir, "crossing_summary_path", crossing_path)
        set_manifest_field(run_dir, "crossing_heatmap_path", psi_threshold_heatmap_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
