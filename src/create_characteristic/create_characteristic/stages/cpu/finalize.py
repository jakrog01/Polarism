"""CPU stage: aggregate per-point results into 2D characteristic map artifacts.

Reads all ``run_dir/points/point_*.json`` files and writes:
  - ``characteristic_map.csv``
  - ``characteristic_map.json``
  - ``threshold_summary.json``
  - ``results/psi_max_heatmap.png``
  - ``results/psi_max_heatmap_log.png``
  - ``results/threshold_map.png``

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
import numpy as np

from create_characteristic.manifest.io import atomic_write_json, set_manifest_field

DEFAULT_THRESHOLD_CRITERION = 5e-2


def _load_all_results(run_dir: str) -> list[dict[str, Any]]:
    pattern = os.path.join(run_dir, "points", "point_*.json")
    paths = sorted(glob.glob(pattern))
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: (int(r["energy_index"]), int(r["sep_index"])))


def _build_2d_arrays(
    results: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energies = sorted({float(r["pulse_energy"]) for r in results})
    separations = sorted({float(r["pulse_separation"]) for r in results})

    ne = len(energies)
    ns = len(separations)

    energy_idx = {e: i for i, e in enumerate(energies)}
    sep_idx = {s: i for i, s in enumerate(separations)}

    psi_map = np.full((ne, ns), np.nan)
    diverged_mask = np.zeros((ne, ns), dtype=bool)

    for r in results:
        ei = energy_idx[float(r["pulse_energy"])]
        si = sep_idx[float(r["pulse_separation"])]
        if r["status"] == "ok":
            psi_map[ei, si] = float(r["psi_sq_max"])
        else:
            diverged_mask[ei, si] = True

    return np.array(energies), np.array(separations), psi_map, diverged_mask


def _write_csv(results: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "point_index", "energy_index", "sep_index",
        "pulse_energy", "pulse_separation", "total_time",
        "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_heatmap(
    energies: np.ndarray,
    separations: np.ndarray,
    psi_map: np.ndarray,
    diverged_mask: np.ndarray,
    out_path: str,
    log_scale: bool,
    threshold_criterion: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_data = psi_map.copy()
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_data = np.where(plot_data > 0, np.log10(plot_data), np.nan)
        label = r"$\log_{10}(\max |\psi|^2)$"
    else:
        label = r"$\max |\psi|^2$"

    sep_grid, e_grid = np.meshgrid(separations, energies)
    im = ax.pcolormesh(
        sep_grid,
        e_grid,
        plot_data,
        shading="nearest",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label=label)

    div_e = e_grid[diverged_mask]
    div_s = sep_grid[diverged_mask]
    if div_e.size > 0:
        ax.scatter(
            div_s, div_e,
            marker="x", color="red", s=40, linewidths=1.5,
            zorder=5, label=f"diverged ({div_e.size})",
        )

    if not log_scale:
        threshold_val = threshold_criterion
    else:
        threshold_val = np.log10(threshold_criterion) if threshold_criterion > 0 else np.nan

    try:
        cs = ax.contour(
            sep_grid, e_grid, plot_data,
            levels=[threshold_val],
            colors=["crimson"],
            linewidths=1.5,
        )
        ax.clabel(cs, fmt=f"threshold={threshold_criterion:.1e}", fontsize=8)
    except Exception:
        pass

    ax.set_xlabel("Pulse separation (ps)")
    ax.set_ylabel("Pulse energy")
    title = r"$\max |\psi|^2$ characteristic map"
    if log_scale:
        title = r"$\log_{10}(\max |\psi|^2)$ characteristic map"
    ax.set_title(title)
    if div_e.size > 0:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_threshold_map(
    energies: np.ndarray,
    separations: np.ndarray,
    psi_map: np.ndarray,
    diverged_mask: np.ndarray,
    threshold_criterion: float,
    out_path: str,
) -> None:
    above = psi_map >= threshold_criterion
    below = (~above) & (~diverged_mask) & (~np.isnan(psi_map))

    fig, ax = plt.subplots(figsize=(10, 7))

    sep_grid, e_grid = np.meshgrid(separations, energies)

    threshold_map = np.full_like(psi_map, np.nan)
    threshold_map[above] = 1.0
    threshold_map[below] = 0.0

    im = ax.pcolormesh(
        sep_grid, e_grid, threshold_map,
        shading="nearest",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label=f"above threshold (criterion: {threshold_criterion:.1e})")

    div_e = e_grid[diverged_mask]
    div_s = sep_grid[diverged_mask]
    if div_e.size > 0:
        ax.scatter(
            div_s, div_e,
            marker="x", color="black", s=40, linewidths=1.5,
            zorder=5, label=f"diverged ({div_e.size})",
        )

    ax.set_xlabel("Pulse separation (ps)")
    ax.set_ylabel("Pulse energy")
    ax.set_title(
        rf"Threshold map: $\max|\psi|^2 \geq {threshold_criterion:.1e}$"
    )
    if div_e.size > 0:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_threshold_summary(
    energies: np.ndarray,
    separations: np.ndarray,
    psi_map: np.ndarray,
    diverged_mask: np.ndarray,
    threshold_criterion: float,
    n_ok: int,
    n_div: int,
) -> dict[str, Any]:
    above = psi_map >= threshold_criterion
    n_above = int(np.sum(above))
    n_below = int(np.sum((~above) & (~diverged_mask) & (~np.isnan(psi_map))))

    threshold_energies: list[float] = []
    for si in range(len(separations)):
        col = psi_map[:, si]
        above_col = np.where(col >= threshold_criterion)[0]
        if above_col.size > 0:
            threshold_energies.append(float(energies[above_col[0]]))
        else:
            threshold_energies.append(float("nan"))

    return {
        "criterion": f"psi_sq_max >= {threshold_criterion}",
        "threshold_criterion": threshold_criterion,
        "n_total": n_ok + n_div,
        "n_ok": n_ok,
        "n_diverged": n_div,
        "n_above_threshold": n_above,
        "n_below_threshold": n_below,
        "energy_axis": energies.tolist(),
        "separation_axis": separations.tolist(),
        "threshold_energy_per_separation": threshold_energies,
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

    energies, separations, psi_map, diverged_mask = _build_2d_arrays(results)

    summary = _build_threshold_summary(
        energies, separations, psi_map, diverged_mask, threshold_criterion, n_ok, n_div
    )
    threshold_path = os.path.join(run_dir, "threshold_summary.json")
    atomic_write_json(threshold_path, summary)
    print(f"  Summary : {threshold_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    heatmap_path = os.path.join(results_dir, "psi_max_heatmap.png")
    try:
        _write_heatmap(
            energies, separations, psi_map, diverged_mask,
            heatmap_path, log_scale=False, threshold_criterion=threshold_criterion,
        )
        print(f"  Heatmap : {heatmap_path}")
    except Exception as e:
        print(f"  WARNING: heatmap generation failed: {e}", file=sys.stderr)

    if np.any(psi_map[~np.isnan(psi_map)] > 0):
        heatmap_log_path = os.path.join(results_dir, "psi_max_heatmap_log.png")
        try:
            _write_heatmap(
                energies, separations, psi_map, diverged_mask,
                heatmap_log_path, log_scale=True, threshold_criterion=threshold_criterion,
            )
            print(f"  Log map : {heatmap_log_path}")
        except Exception as e:
            print(f"  WARNING: log heatmap generation failed: {e}", file=sys.stderr)

    threshold_map_path = os.path.join(results_dir, "threshold_map.png")
    try:
        _write_threshold_map(
            energies, separations, psi_map, diverged_mask,
            threshold_criterion, threshold_map_path,
        )
        print(f"  Thr map : {threshold_map_path}")
    except Exception as e:
        print(f"  WARNING: threshold map generation failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "n_ok", n_ok)
        set_manifest_field(run_dir, "n_diverged", n_div)
        set_manifest_field(run_dir, "threshold_criterion", threshold_criterion)
        set_manifest_field(run_dir, "csv_path", csv_path)
        set_manifest_field(run_dir, "json_path", json_path)
        set_manifest_field(run_dir, "heatmap_path", heatmap_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
