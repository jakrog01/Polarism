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


def _load_all_results(run_dir: str) -> list[dict[str, Any]]:
    pattern = os.path.join(run_dir, "powers", "power_*.json")
    paths = sorted(glob.glob(pattern))
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: r["P"])


def _write_csv(results: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "P", "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_plot(results: list[dict[str, Any]], out_path: str) -> None:
    ok_P = [r["P"] for r in results if r["status"] == "ok"]
    ok_psi = [r["psi_sq_max"] for r in results if r["status"] == "ok"]
    div_P = [r["P"] for r in results if r["status"] == "diverged"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if ok_P:
        ax.plot(ok_P, ok_psi, "b-o", markersize=4, linewidth=1.5, label="ok")

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

    ax.set_xlabel("Pump power P")
    ax.set_ylabel(r"$\max |\psi|^2$")
    ax.set_title(r"$\psi_\mathrm{max}^2$ vs pump power")
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
    print(f"  Loaded  : {len(results)} results  ({n_ok} ok, {n_div} diverged)")

    csv_path = os.path.join(run_dir, "threshold_curve.csv")
    _write_csv(results, csv_path)
    print(f"  CSV     : {csv_path}")

    json_path = os.path.join(run_dir, "threshold_curve.json")
    atomic_write_json(json_path, results)
    print(f"  JSON    : {json_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    png_path = os.path.join(results_dir, "psi_max_vs_power.png")
    try:
        _write_plot(results, png_path)
        print(f"  Plot    : {png_path}")
    except Exception as e:
        print(f"  WARNING: plot generation failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "n_ok", n_ok)
        set_manifest_field(run_dir, "n_diverged", n_div)
        set_manifest_field(run_dir, "csv_path", csv_path)
        set_manifest_field(run_dir, "json_path", json_path)
        set_manifest_field(run_dir, "png_path", png_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
