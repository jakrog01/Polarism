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


def _write_csv(results: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "sweep_variable", "sweep_value", "P", "pulse_separation",
        "total_time", "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_plot(results: list[dict[str, Any]], out_path: str) -> None:
    axis_key, axis_label, _ = _sweep_axis(results)
    ok_P = [r.get(axis_key, r["P"]) for r in results if r["status"] == "ok"]
    ok_psi = [r["psi_sq_max"] for r in results if r["status"] == "ok"]
    div_P = [r.get(axis_key, r["P"]) for r in results if r["status"] == "diverged"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if ok_P:
        ax.plot(ok_P, ok_psi, "b-o", markersize=4, linewidth=1.5, label="ok")
        positive = [v for v in ok_psi if v > 0.0]
        if positive:
            ax.set_yscale("log")

    if div_P:
        y_div = max(ok_psi) if ok_psi else 1.0
        ax.scatter(
            div_P,
            [y_div * 1.05] * len(div_P),
            marker="x",
            color="red",
            s=60,
            zorder=5,
            label=f"diverged ({len(div_P)})",
        )

    ax.set_xlabel(axis_label)
    ax.set_ylabel(r"$\max |\psi|^2$")
    ax.set_title(rf"$\psi_\mathrm{{max}}^2$ vs {axis_label}")
    ax.axhline(
        CONDENSATION_PSI_SQ_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label=r"$|\psi|^2_{max}=5\times10^{-2}$",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_div = sum(1 for r in results if r["status"] == "diverged")
    threshold_row = next(
        (
            r for r in results
            if r["status"] == "ok"
            and float(r.get("psi_sq_max", 0.0)) >= CONDENSATION_PSI_SQ_THRESHOLD
        ),
        None,
    )
    axis_key, axis_label, plot_name = _sweep_axis(results)
    threshold_estimate = (
        threshold_row.get(axis_key, threshold_row["P"])
        if threshold_row is not None else None
    )
    print(f"  Loaded  : {len(results)} results  ({n_ok} ok, {n_div} diverged)")
    print(f"  Threshold estimate: {threshold_estimate}")

    csv_path = os.path.join(run_dir, "threshold_curve.csv")
    _write_csv(results, csv_path)
    print(f"  CSV     : {csv_path}")

    json_path = os.path.join(run_dir, "threshold_curve.json")
    atomic_write_json(json_path, results)
    print(f"  JSON    : {json_path}")

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
        _write_plot(results, png_path)
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
