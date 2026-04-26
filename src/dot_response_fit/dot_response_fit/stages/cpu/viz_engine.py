"""Rendering helpers for dot-response-fit pipeline outputs.

All public functions take explicit data and result directories.

``generate_animation`` has been replaced by the NVENC streaming renderer in
``pipeline.render.nvenc_stream``.  This module handles static PNGs and the
scalar-sidecar-based summary plot only.
"""
from __future__ import annotations

import math
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec

FIELD_SPECS = {
    "psi_sq": {
        "source": "psi",
        "label": r"$|\psi|^2$",
        "cmap": "magma",
        "transform": "abs2",
    },
    "nA": {"source": "nA", "label": r"$n_A$", "cmap": "viridis", "transform": None},
    "nI": {"source": "nI", "label": r"$n_I$", "cmap": "plasma", "transform": None},
    "Pump": {
        "source": "Pump",
        "label": "Pump",
        "cmap": "inferno",
        "transform": None,
        "norm": "power",
    },
}
SCALAR_MAP = {
    "psi_sq": "psi_sq_max",
    "nA": "nA_max",
    "nI": "nI_max",
    "Pump": "P_max",
}
COMPARISON_SCALARS = [
    ("psi_sq_max", r"$|\psi|^2_{\max}$"),
    ("nA_max", r"$n_A^{\max}$"),
    ("nI_max", r"$n_I^{\max}$"),
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
    """Return index array that sorts the time axis."""
    return np.argsort(h5["time"][:], kind="stable")


def _read_field_frame(h5: h5py.File, spec: dict, idx: int) -> np.ndarray:
    """Read one field frame, applying the transform if specified."""
    raw = h5[f"fields/{spec['source']}"][idx]
    if spec["transform"] == "abs2":
        return np.abs(raw) ** 2
    return raw


def pick_indices(n: int, count: int = SNAPSHOT_COUNT) -> list[int]:
    """Pick evenly spaced snapshot indices from 0..*n*-1."""
    if n <= count:
        return list(range(n))
    return [int(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_norm(spec: dict, vmin: float, vmax: float):
    """Build the matplotlib colour normalisation."""
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
    """Generate a snapshot-grid PNG with a scalar trace.

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

        snapshots = [_read_field_frame(h5, spec, pi) for pi in snap_physical]
        scalar_data = h5[f"scalars/{scalar_key}"][:][sort_order] if has_scalar else None

    n_cols = len(snap_logical)
    n_rows = 2 if has_scalar else 1
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
            snapshot,
            origin="lower",
            extent=extent,
            cmap=spec["cmap"],
            norm=norm,
            **({"vmin": vmin, "vmax": vmax} if norm is None else {}),
            aspect="equal",
        )
        ax.set_title(f"t = {time_sorted[li]:.1f} ps", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"y ($\mu$m)")
        ax.set_xlabel(r"x ($\mu$m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if has_scalar and scalar_data is not None:
        ax_sc = fig.add_subplot(gs[1, :])
        ax_sc.plot(
            time_sorted[: len(scalar_data)],
            scalar_data,
            linewidth=1.2,
            color="black",
        )
        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(scalar_key)
        ax_sc.grid(True, alpha=0.3)
        for li in snap_logical:
            ax_sc.axvline(time_sorted[li], color="gray", linestyle="--", alpha=0.4, linewidth=0.6)

    fig.suptitle(
        f"{routine.upper()} \u2014 {spec['label']}", fontsize=14, fontweight="bold"
    )

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

    fig.suptitle("Dot-Response Fit Summary", fontsize=14, fontweight="bold")
    out_path = os.path.join(results_dir, "summary.png")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")


def generate_trace_comparison(
    image_id: str,
    t_ref: np.ndarray,
    nC_ref: np.ndarray,
    pump_ref: np.ndarray | None,
    t_sim: np.ndarray,
    obs_sim: np.ndarray,
    sigma_space: float,
    rmse: float,
    dataset_index: int,
    digit_class: int,
    observable_label: str,
    out_dir: str,
) -> str:
    """Generate a trace comparison PNG for one image.

    Uses :func:`dot_response_fit.physics.trace_compare.score_traces` as the
    single source of resampling and normalisation, so the plotted curves are
    exactly what the score metric sees.

    Returns the path to the saved PNG.
    """
    from dot_response_fit.physics.trace_compare import score_traces

    _rmse_check, t_common, ref_norm, sim_norm = score_traces(t_ref, nC_ref, t_sim, obs_sim)
    comparison_valid = t_common.size > 0 and math.isfinite(rmse)

    n_rows = 3 if pump_ref is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), constrained_layout=True)
    ax_img, ax_trace = axes[0], axes[1]
    ax_pump = axes[2] if pump_ref is not None else None

    rmse_str = f"{rmse:.6f}" if math.isfinite(rmse) else "inf (invalid)"
    ax_img.axis("off")
    ax_img.text(
        0.5, 0.5,
        f"{image_id}  |  dataset_index={dataset_index}  |  digit={digit_class}\n"
        f"sigma_space={sigma_space:.2f} µm  |  RMSE={rmse_str}",
        ha="center", va="center", fontsize=11, transform=ax_img.transAxes,
    )

    if comparison_valid:
        ax_trace.plot(t_common, ref_norm, label=r"ODE $n_C(t)$ (normalised)", color="steelblue", lw=1.8)
        ax_trace.plot(t_common, sim_norm, label=f"2D {observable_label} (normalised)", color="darkorange", lw=1.8, ls="--")
        ax_trace.legend(fontsize=9, framealpha=0.9)
    else:
        msg = "no time overlap" if t_common.size == 0 else "near-zero trace / invalid comparison"
        ax_trace.text(0.5, 0.5, f"Invalid trace comparison:\n{msg}",
                      ha="center", va="center", fontsize=12, color="crimson",
                      transform=ax_trace.transAxes)
    ax_trace.set_xlabel("t (ps)")
    ax_trace.set_ylabel("Normalised trace")
    ax_trace.grid(True, alpha=0.3)

    if ax_pump is not None and pump_ref is not None:
        ax_pump.plot(t_ref, pump_ref, color="firebrick", lw=1.2)
        ax_pump.set_xlabel("t (ps)")
        ax_pump.set_ylabel("Pump P(t)")
        ax_pump.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "trace_comparison.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def generate_diagnostic_overlay_png(
    h5_path: str,
    extent: list[float],
    events: dict,
    image_norm: np.ndarray,
    out_dir: str,
    laser_position: tuple[float, float] | None = None,
) -> str:
    """Generate a single-frame diagnostic PNG with the active pixel marked.

    Uses the last HDF5 frame and marks the last active MNIST pulse on the
    psi_sq and Pump panels.  The active-pixel marker is intentionally absent
    from the animation — it appears only here.

    Returns the path to the saved PNG.
    """
    import h5py
    from dot_response_fit.stages.gpu.render_overlay import (
        _find_last_active_pixel,
        _phys_to_panel,
        _draw_ring,
    )

    centers_ode: list[float] = events["centers_ode"]
    pixel_flat_indices: list[int] = events["pixel_flat_indices"]
    sigma_time = float(events["sigma_time"])
    cutoff_sigma = float(events["cutoff_sigma"])
    if laser_position is None:
        if "laser_x0" in events and "laser_y0" in events:
            laser_position = (float(events["laser_x0"]), float(events["laser_y0"]))
        else:
            x_positions: list[float] = events["x_positions"]
            y_positions: list[float] = events["y_positions"]
            active_for_position = _find_last_active_pixel(
                centers_ode, sigma_time, cutoff_sigma
            )
            if active_for_position is not None:
                laser_position = (
                    float(x_positions[active_for_position]),
                    float(y_positions[active_for_position]),
                )

    active_idx = _find_last_active_pixel(centers_ode, sigma_time, cutoff_sigma)

    h, w_img = image_norm.shape
    grey = np.clip(image_norm * 255, 0, 255).astype(np.uint8)

    with h5py.File(h5_path, "r") as h5:
        sort_order = np.argsort(h5["time"][:], kind="stable")
        last_physical = int(sort_order[-1])
        psi_raw = h5["fields/psi"][last_physical]
        psi_sq = np.abs(psi_raw) ** 2
        try:
            pump_raw = h5["fields/Pump"][last_physical]
        except KeyError:
            pump_raw = None
        t_last = float(h5["time"][:][sort_order[-1]])

    n_panels = 2 + (1 if pump_raw is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), constrained_layout=True)
    ax_psi = axes[0]
    ax_thumb = axes[-1]
    ax_pump = axes[1] if pump_raw is not None else None

    ax_psi.imshow(
        psi_sq, origin="lower", extent=extent, cmap="magma", aspect="equal"
    )
    ax_psi.set_title(rf"$|\psi|^2$  t={t_last:.1f} ps")
    ax_psi.set_xlabel(r"x (µm)")
    ax_psi.set_ylabel(r"y (µm)")

    if laser_position is not None:
        x_um, y_um = laser_position
        ax_psi.plot(x_um, y_um, "o", ms=10, mec="orangered", mfc="none", mew=2)

    if ax_pump is not None and pump_raw is not None:
        ax_pump.imshow(
            pump_raw, origin="lower", extent=extent, cmap="inferno", aspect="equal"
        )
        ax_pump.set_title(f"Pump  t={t_last:.1f} ps")
        ax_pump.set_xlabel(r"x (µm)")
        if laser_position is not None:
            ax_pump.plot(x_um, y_um, "o", ms=10, mec="orangered", mfc="none", mew=2)

    ax_thumb.imshow(grey, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="equal")
    ax_thumb.set_title("MNIST input")
    ax_thumb.axis("off")
    if active_idx is not None:
        flat_idx = pixel_flat_indices[active_idx]
        src_row = flat_idx // w_img
        src_col = flat_idx % w_img
        circle = plt.Circle((src_col, src_row), radius=1.5, color="red", fill=False, lw=1.5)
        ax_thumb.add_patch(circle)

    fig.suptitle(
        f"Diagnostic overlay — last active pixel: {active_idx}",
        fontsize=11,
    )
    out_path = os.path.join(out_dir, "diagnostic_overlay.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path
