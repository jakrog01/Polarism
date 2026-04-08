"""Rendering helpers for scenario outputs.

All public functions take explicit data and result directories. The module
does not keep pipeline state in globals.

``generate_animation`` has been replaced by the NVENC streaming renderer in
``pipeline.render.nvenc_stream``.  This module handles static PNGs and the
scalar-sidecar-based summary plot only.
"""
from __future__ import annotations

import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec

FIELD_SPECS = {
    "psi_sq": {"source": "psi", "label": r"$|\psi|^2$", "cmap": "magma", "transform": "abs2"},
    "nI": {"source": "nI", "label": r"$n_I$", "cmap": "plasma", "transform": None},
    "nA": {"source": "nA", "label": r"$n_A$", "cmap": "viridis", "transform": None},
    "Pump": {"source": "Pump", "label": "Pump", "cmap": "inferno", "transform": None, "norm": "power"},
}
SCALAR_MAP = {"psi_sq": "psi_sq_max", "nI": "nI_max", "nA": "nA_max", "Pump": "P_max"}
COMPARISON_SCALARS = [
    ("psi_sq_max", r"$|\psi|^2_{max}$"),
    ("nI_max", r"$n_I^{max}$"),
    ("nA_max", r"$n_A^{max}$"),
]

SNAPSHOT_COUNT = 5
PLOT_DPI = 200
PUMP_NORM_GAMMA = 0.3


def routine_dir(routine: str, results_dir: str) -> str:
    """Return the output directory for *routine* within *results_dir*."""
    return os.path.join(results_dir, routine)


def open_h5(routine: str, data_dir: str) -> h5py.File:
    """Open the HDF5 file for *routine* in *data_dir*."""
    return h5py.File(os.path.join(data_dir, f"{routine}.h5"), "r")


def load_sorted_order(h5: h5py.File) -> np.ndarray:
    """Load the frame order stored in the file."""
    return np.argsort(h5["time"][:], kind="stable")


def _read_field_frame(h5: h5py.File, spec: dict, idx: int) -> np.ndarray:
    """Read one field frame from the file."""
    raw = h5[f"fields/{spec['source']}"][idx]
    if spec["transform"] == "abs2":
        return np.abs(raw) ** 2
    return raw


def pick_indices(n: int, count: int = SNAPSHOT_COUNT) -> list[int]:
    """Pick evenly spaced snapshot indices."""
    if n <= count:
        return list(range(n))
    return [int(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_norm(spec: dict, vmin: float, vmax: float):
    """Build the color normalization for a plot."""
    if spec.get("norm") == "power":
        return PowerNorm(gamma=PUMP_NORM_GAMMA, vmin=max(vmin, 1e-12), vmax=vmax)
    return None


def generate_field_png(
    routine: str,
    field_key: str,
    extent: list[float],
    data_dir: str,
    results_dir: str,
) -> None:
    """Generate a snapshot grid PNG with a scalar trace for *field_key*.

    Reads from ``{data_dir}/{routine}.h5`` — call while HDF5 is local on scratch.
    """
    spec = FIELD_SPECS[field_key]

    with open_h5(routine, data_dir) as h5:
        sort_order = load_sorted_order(h5)
        time_sorted = h5["time"][:][sort_order]
        n_frames = time_sorted.shape[0]

        snap_logical = pick_indices(n_frames)
        snap_physical = [int(sort_order[i]) for i in snap_logical]

        scalar_key = SCALAR_MAP.get(field_key)
        has_scalar = scalar_key is not None and f"scalars/{scalar_key}" in h5

        per_laser_keys = []
        if field_key == "Pump":
            i = 0
            while f"scalars/P_max_{i}" in h5:
                per_laser_keys.append(f"P_max_{i}")
                i += 1
        elif field_key == "psi_sq":
            i = 0
            while f"scalars/psi_sq_max_w{i}" in h5:
                per_laser_keys.append(f"psi_sq_max_w{i}")
                i += 1

        snapshots = [_read_field_frame(h5, spec, pi) for pi in snap_physical]
        scalar_data = h5[f"scalars/{scalar_key}"][:][sort_order] if has_scalar else None
        laser_data_map = {lk: h5[f"scalars/{lk}"][:][sort_order] for lk in per_laser_keys}

    n_cols = len(snap_logical)
    show_scalar_row = has_scalar or per_laser_keys
    n_rows = 2 if show_scalar_row else 1
    height_ratios = [3, 1] if n_rows == 2 else [1]

    fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=height_ratios)

    vmin = min(s.min() for s in snapshots)
    vmax = max(s.max() for s in snapshots)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = _make_norm(spec, vmin, vmax)

    for col, (li, snapshot) in enumerate(zip(snap_logical, snapshots)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            snapshot, origin="lower", extent=extent, cmap=spec["cmap"],
            norm=norm, **({"vmin": vmin, "vmax": vmax} if norm is None else {}),
            aspect="equal",
        )
        ax.set_title(f"t = {time_sorted[li]:.1f} ps", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"y ($\mu$m)")
        ax.set_xlabel(r"x ($\mu$m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_scalar_row:
        ax_sc = fig.add_subplot(gs[1, :])
        if has_scalar and scalar_data is not None:
            label = "Total" if per_laser_keys else scalar_key
            ax_sc.plot(time_sorted[: len(scalar_data)], scalar_data,
                       linewidth=1.2, color="black", label=label)
        for lk in per_laser_keys:
            ld = laser_data_map[lk]
            laser_label = lk.replace("P_max_", "Laser ") if lk.startswith("P_max_") else lk.replace("psi_sq_max_w", "Spot ")
            ax_sc.plot(time_sorted[: len(ld)], ld, linewidth=0.9, alpha=0.8, label=laser_label)
        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(scalar_key if has_scalar else "P_max")
        ax_sc.legend(fontsize=8, loc="best")
        ax_sc.grid(True, alpha=0.3)
        for li in snap_logical:
            ax_sc.axvline(time_sorted[li], color="gray", linestyle="--", alpha=0.4, linewidth=0.6)

    fig.suptitle(f"{routine.upper()} \u2014 {spec['label']}", fontsize=14, fontweight="bold")

    out_dir = routine_dir(routine, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{field_key}.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")


def generate_summary(
    routines: list[str],
    data_dir: str,
    results_dir: str,
) -> None:
    """Generate a cross-scenario comparison plot from scalar sidecars.

    Reads ``{data_dir}/{routine}_scalars.npz`` for each routine — no HDF5 required.
    """
    routine_data: dict[str, dict] = {}
    for routine in routines:
        sidecar = os.path.join(data_dir, f"{routine}_scalars.npz")
        if not os.path.isfile(sidecar):
            print(f"  WARNING: scalar sidecar not found, skipping {routine}: {sidecar}")
            continue
        npz = np.load(sidecar)
        scalars = {
            sc_key: npz[sc_key]
            for sc_key, _ in COMPARISON_SCALARS
            if sc_key in npz
        }
        routine_data[routine] = {"time": npz["time"], "scalars": scalars}

    if not routine_data:
        print("  No scalar sidecars found; skipping summary plot.")
        return

    n_sc = len(COMPARISON_SCALARS)
    fig = plt.figure(
        figsize=(max(10.0, 1.5 * len(routine_data)), max(3.5 * n_sc, 4.0)),
        constrained_layout=True,
    )
    gs = GridSpec(n_sc, 1, figure=fig)

    for row, (sc_key, sc_label) in enumerate(COMPARISON_SCALARS):
        ax_sc = fig.add_subplot(gs[row, 0])
        for routine, rd in routine_data.items():
            if sc_key in rd["scalars"]:
                scalar = rd["scalars"][sc_key]
                ax_sc.plot(rd["time"][: len(scalar)], scalar, label=routine)
        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(sc_label)
        ax_sc.set_title(f"{sc_label} \u2014 comparison")
        ax_sc.legend(fontsize=10, framealpha=0.9)
        ax_sc.grid(True, alpha=0.3)

    fig.suptitle("Multi-Pump Comparison Summary", fontsize=14, fontweight="bold")
    out_path = os.path.join(results_dir, "summary.png")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")
