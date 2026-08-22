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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
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


def _build_gain_arrays(
    results: list[dict[str, Any]],
    energies: np.ndarray,
    separations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain_map = np.full((len(energies), len(separations)), np.nan)
    class_map = np.full((len(energies), len(separations)), np.nan)
    energy_index = {value: index for index, value in enumerate(energies)}
    separation_index = {value: index for index, value in enumerate(separations)}
    class_values = {"dark": 0.0, "gain_only": 1.0, "latched": 2.0, "spiking": 3.0}
    for result in results:
        if result["status"] != "ok":
            continue
        ei = energy_index[float(result["pulse_energy"])]
        si = separation_index[float(result["pulse_separation"])]
        gain_map[ei, si] = float(result.get("n_gain_crossings", 0))
        class_map[ei, si] = class_values.get(str(result.get("klass", "dark")), np.nan)
    return gain_map, class_map


def _write_csv(results: list[dict[str, Any]], path: str) -> None:
    fieldnames = [
        "point_index", "energy_index", "sep_index",
        "pulse_energy", "pulse_separation", "total_time",
        "psi_sq_max", "status", "t_psi_sq_max",
        "diverged_at_t", "wall_time_seconds",
        "n_gain_crossings", "first_crossing_ps", "nR_center_max",
        "ratio_to_critical", "n_active_max_domain", "gain_check_every", "klass",
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


def _write_gain_crossings_heatmap(
    energies: np.ndarray,
    separations: np.ndarray,
    gain_map: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    maximum = max(1, int(np.nanmax(gain_map)))
    boundaries = np.arange(-0.5, maximum + 1.5, 1.0)
    mesh = ax.pcolormesh(
        *np.meshgrid(separations, energies),
        gain_map,
        shading="nearest",
        cmap="viridis",
        norm=BoundaryNorm(boundaries, ncolors=256),
    )
    fig.colorbar(mesh, ax=ax, label="Gain crossings")
    try:
        ax.contour(*np.meshgrid(separations, energies), gain_map >= 1.0, levels=[0.5], colors="white")
    except ValueError:
        pass
    ax.set(xlabel="Pulse separation (ps)", ylabel="Pulse energy", title="Gain-crossing map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_class_map(
    energies: np.ndarray,
    separations: np.ndarray,
    class_map: np.ndarray,
    out_path: str,
) -> None:
    labels = ("dark", "gain_only", "latched", "spiking")
    colors = ("#293241", "#f4a261", "#2a9d8f", "#e63946")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pcolormesh(
        *np.meshgrid(separations, energies),
        class_map,
        shading="nearest",
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(-0.5, 4.5, 1.0), ncolors=4),
    )
    ax.legend([Patch(color=color) for color in colors], labels, loc="upper right")
    ax.set(xlabel="Pulse separation (ps)", ylabel="Pulse energy", title="Condensation class map")
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
    gain_map: np.ndarray,
    class_map: np.ndarray,
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

    gain_thresholds = [
        float(energies[indices[0]]) if (indices := np.where(gain_map[:, si] >= 1.0)[0]).size else None
        for si in range(len(separations))
    ]
    spiking_bands = []
    for si in range(len(separations)):
        indices = np.where(class_map[:, si] == 3.0)[0]
        spiking_bands.append(None if not indices.size else [float(energies[indices[0]]), float(energies[indices[-1]])])
    classes = ("dark", "gain_only", "latched", "spiking")
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
        "gain_threshold_energy_per_separation": gain_thresholds,
        "spiking_band_per_separation": spiking_bands,
        "class_counts": {name: int(np.sum(class_map == index)) for index, name in enumerate(classes)},
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
    gain_map, class_map = _build_gain_arrays(results, energies, separations)

    summary = _build_threshold_summary(
        energies,
        separations,
        psi_map,
        diverged_mask,
        threshold_criterion,
        n_ok,
        n_div,
        gain_map,
        class_map,
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

    gain_heatmap_path = os.path.join(results_dir, "gain_crossings_heatmap.png")
    class_map_path = os.path.join(results_dir, "condensation_class_map.png")
    try:
        _write_gain_crossings_heatmap(energies, separations, gain_map, gain_heatmap_path)
        _write_class_map(energies, separations, class_map, class_map_path)
    except Exception as e:
        print(f"  WARNING: gain/class map generation failed: {e}", file=sys.stderr)
    gain_only_count = int(np.sum(class_map == 1.0))
    if gain_only_count > 0.05 * len(results):
        print(
            "WARNING: more than 5% of points are gain_only; gain crossing alone is not "
            "sufficient for observable condensation.",
            file=sys.stderr,
        )

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
