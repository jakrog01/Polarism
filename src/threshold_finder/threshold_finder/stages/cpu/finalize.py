"""CPU stage: aggregate per-power results into summary artifacts.

Reads all ``run_dir/powers/power_*.json`` files, sorts by P, and writes:
  - ``threshold_curve.csv``
  - ``threshold_curve.json``
  - ``results/psi_max_vs_power.png``

Invoked by Slurm as:
    python -m threshold_finder.stages.cpu.finalize --run-dir <run_dir>
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
from matplotlib.ticker import MaxNLocator
import numpy as np

from threshold_finder.manifest.io import atomic_write_json, set_manifest_field

CONDENSATION_PSI_SQ_THRESHOLD = 5e-2


def _load_all_results(run_dir: str) -> list[dict[str, Any]]:
    pattern = os.path.join(run_dir, "powers", "power_*.json")
    paths = sorted(glob.glob(pattern))
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: float(r.get("sweep_value", r["P"])))


def _sweep_axis(results: list[dict[str, Any]]) -> tuple[str, str, str]:
    variable = str(results[0].get("sweep_variable", "P"))
    if variable == "pulse_separation":
        return "pulse_separation", "Pulse separation (ps)", "psi_max_vs_pulse_separation.png"
    return "P", "Pump power P", "psi_max_vs_power.png"


def _write_csv(results: list[dict[str, Any]], path: str, seeded: bool) -> None:
    fieldnames = [
        "sweep_variable", "sweep_value", "P", "pulse_separation",
        "total_time", "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
    ]
    if seeded:
        fieldnames.append("psi_sq_max_std")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _ensemble_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[float, str, float], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            float(result["P"]),
            str(result.get("sweep_variable", "P")),
            float(result.get("sweep_value", result["P"])),
        )
        groups.setdefault(key, []).append(result)
    ensemble: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    for (_, variable, value), rows in groups.items():
        values = [float(row["psi_sq_max"]) for row in rows]
        seeds = [int(row["seed"]) for row in rows if row.get("seed") is not None]
        summary = {
            "P": float(rows[0]["P"]),
            "sweep_variable": variable,
            "sweep_value": value,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "psi_sq_max_mean": float(np.mean(values)),
            "psi_sq_max_std": float(np.std(values)),
            "psi_sq_max_min": float(np.min(values)),
            "psi_sq_max_max": float(np.max(values)),
            "psi_sq_max_values": values,
        }
        ensemble.append(summary)
        canonical.append({
            **rows[0],
            "psi_sq_max": summary["psi_sq_max_mean"],
            "psi_sq_max_std": summary["psi_sq_max_std"],
        })
    return canonical, ensemble


def _write_plot(results: list[dict[str, Any]], out_path: str) -> None:
    axis_key, axis_label, _ = _sweep_axis(results)
    ok_P = [r.get(axis_key, r["P"]) for r in results if r["status"] == "ok"]
    ok_psi = [r["psi_sq_max"] for r in results if r["status"] == "ok"]
    div_P = [r.get(axis_key, r["P"]) for r in results if r["status"] == "diverged"]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F5F5F5")
    ax.set_facecolor("#F5F5F5")

    if ok_P:
        ax.plot(ok_P, ok_psi, "b-o", markersize=4, linewidth=1.5)

    if div_P:
        y_div = max(ok_psi) if ok_psi else 1.0
        ax.scatter(
            div_P,
            [y_div * 1.05] * len(div_P),
            marker="x",
            color="red",
            s=60,
            zorder=5,
        )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="major", labelsize=18, length=7, width=1.2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU finalize: power-sweep aggregation")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    print("=" * 60)
    print(" Power-Sweep Finalize")
    print("=" * 60)
    print(f"  Run dir : {run_dir}")

    results = _load_all_results(run_dir)
    if not results:
        print("ERROR: no per-power result files found in run_dir/powers/.", file=sys.stderr)
        sys.exit(1)

    seeded = any(result.get("seed") is not None for result in results)
    canonical_results, ensemble = _ensemble_rows(results) if seeded else (results, [])
    n_ok = sum(1 for r in canonical_results if r["status"] == "ok")
    n_div = sum(1 for r in canonical_results if r["status"] == "diverged")
    threshold_row = next(
        (
            r for r in canonical_results
            if r["status"] == "ok"
            and float(r.get("psi_sq_max", 0.0)) >= CONDENSATION_PSI_SQ_THRESHOLD
        ),
        None,
    )
    axis_key, axis_label, plot_name = _sweep_axis(canonical_results)
    threshold_estimate = (
        threshold_row.get(axis_key, threshold_row["P"])
        if threshold_row is not None else None
    )
    print(f"  Loaded  : {len(results)} results  ({n_ok} ok, {n_div} diverged)")
    print(f"  Threshold estimate: {threshold_estimate}")

    csv_path = os.path.join(run_dir, "threshold_curve.csv")
    _write_csv(canonical_results, csv_path, seeded)
    print(f"  CSV     : {csv_path}")

    json_path = os.path.join(run_dir, "threshold_curve.json")
    atomic_write_json(json_path, canonical_results)
    print(f"  JSON    : {json_path}")

    if seeded:
        ensemble_path = os.path.join(run_dir, "threshold_ensemble.json")
        atomic_write_json(ensemble_path, ensemble)
        print(f"  Ensemble: {ensemble_path}")

    threshold_path = os.path.join(run_dir, "threshold_estimate.json")
    atomic_write_json(
        threshold_path,
        {
            "criterion": f"psi_sq_max >= {CONDENSATION_PSI_SQ_THRESHOLD}",
            "threshold_axis": axis_key,
            "threshold_axis_label": axis_label,
            "threshold_estimate": threshold_estimate,
            "P_threshold_estimate": threshold_estimate if axis_key == "P" else None,
            "threshold_row": threshold_row,
        },
    )
    print(f"  Estimate: {threshold_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    png_path = os.path.join(results_dir, plot_name)
    try:
        _write_plot(canonical_results, png_path)
        print(f"  Plot    : {png_path}")
    except Exception as e:
        print(f"  WARNING: plot generation failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "n_ok", n_ok)
        set_manifest_field(run_dir, "n_diverged", n_div)
        set_manifest_field(run_dir, "threshold_axis", axis_key)
        set_manifest_field(run_dir, "threshold_estimate", threshold_estimate)
        if axis_key == "P":
            set_manifest_field(run_dir, "P_threshold_estimate", threshold_estimate)
        set_manifest_field(run_dir, "csv_path", csv_path)
        set_manifest_field(run_dir, "json_path", json_path)
        set_manifest_field(run_dir, "png_path", png_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
