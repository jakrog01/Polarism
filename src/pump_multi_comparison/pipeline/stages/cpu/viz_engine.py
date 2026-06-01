"""Rendering helpers for scenario outputs.

All public functions take explicit data and result directories. The module
does not keep pipeline state in globals.

``generate_animation`` has been replaced by the NVENC streaming renderer in
``pipeline.render.nvenc_stream``.  This module handles static PNGs and the
scalar-sidecar-based summary plot only.
"""
from __future__ import annotations

import os
import json

import h5py
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec

FIELD_SPECS = {
    "psi_sq": {"source": "psi", "label": r"$|\psi|^2$", "cmap": "magma", "transform": "abs2", "norm": "power", "norm_gamma": 0.5},
    "psi_k":  {"source": "psi", "label": r"$\log_{10}|\psi_k|^2$ (norm.)", "cmap": "inferno", "transform": "kspace_log"},
    "nI": {"source": "nI", "label": r"$n_I$", "cmap": "magma", "transform": None},
    "nA": {"source": "nA", "label": r"$n_A$", "cmap": "inferno", "transform": None},
    "Pump": {"source": "Pump", "label": "Pump", "cmap": "inferno", "transform": None, "norm": "power"},
}
SCALAR_MAP = {"psi_sq": "psi_sq_max", "nI": "nI_max", "nA": "nA_max", "Pump": "P_max"}
_PEAK_FIELDS: frozenset[str] = frozenset({"psi_sq", "nI", "nA"})
COMPARISON_SCALARS = [
    ("P_max", r"$P_{max}$ (local peak)"),
    ("P_area_integral", r"$\int P\,dA$ (pump area integral)"),
    ("P_cumulative_area_time_integral", r"$\int\!\!\int P\,dA\,dt$ (cumulative dose)"),
    ("psi_sq_max", r"$|\psi|^2_{max}$"),
    ("nI_max", r"$n_I^{max}$"),
    ("nA_max", r"$n_{active}^{max}$"),
    ("k0_frac", r"$k=0$ fraction"),
]

SNAPSHOT_COUNT = 5
PLOT_DPI = 200
PUMP_NORM_GAMMA = 0.3
PSI_DIAGNOSTIC_IGNORE_PS = 5.0

ROI_LABEL_SUFFIXES = {
    "mean_psi_sq": r"ROI mean $|\psi|^2$",
    "integral_psi_sq": r"ROI $\int |\psi|^2 dA$",
    "mean_nR": r"ROI mean $n_R$",
    "mean_nI": r"ROI mean $n_I$",
    "integral_emission": r"ROI emission proxy",
}


def _sweep_axis(rows: list[dict]) -> tuple[str, str]:
    """Return the horizontal sweep axis key and label for scalar heatmaps."""
    if any(row.get("square_side") is not None for row in rows):
        return "square_side", "square side a (um)"
    return "pulse_separation", "pulse separation (ps)"


def _sweep_axis_values(rows: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        vals.append(float(value))
    return sorted(set(vals))


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
    if spec["transform"] == "kspace_log":
        psi_k = np.fft.fftshift(np.fft.fft2(raw))
        power = np.abs(psi_k) ** 2
        peak = float(power.max())
        if peak > 0:
            return np.log10(power / peak + 1e-10)
        return np.zeros_like(power, dtype=np.float64)
    return raw


def pick_indices(n: int, count: int = SNAPSHOT_COUNT) -> list[int]:
    """Pick evenly spaced snapshot indices."""
    if n <= count:
        return list(range(n))
    return [int(i * (n - 1) / (count - 1)) for i in range(count)]


def _load_sidecar_scalars(routine: str, data_dir: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Load dense scalar traces from the sidecar if it is available."""
    sidecar_path = os.path.join(data_dir, f"{routine}_scalars.npz")
    if not os.path.exists(sidecar_path):
        return {}

    wanted = ["time", *keys]
    arrays: dict[str, np.ndarray] = {}
    with np.load(sidecar_path) as npz:
        for key in wanted:
            if key in npz:
                arrays[key] = np.asarray(npz[key]).copy()
    return arrays


def _make_norm(spec: dict, vmin: float, vmax: float):
    """Build the color normalization for a plot."""
    if spec.get("norm") == "power":
        gamma = spec.get("norm_gamma", PUMP_NORM_GAMMA)
        return PowerNorm(gamma=gamma, vmin=max(vmin, 1e-12), vmax=vmax)
    return None


def generate_field_png(
    routine: str,
    field_key: str,
    extent: list[float],
    data_dir: str,
    results_dir: str,
    clim: tuple[float, float] | None = None,
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

        scalar_data = h5[f"scalars/{scalar_key}"][:][sort_order] if has_scalar else None
        laser_data_map = {lk: h5[f"scalars/{lk}"][:][sort_order] for lk in per_laser_keys}

        trace_scalar_time = time_sorted
        trace_scalar_data = scalar_data
        trace_laser_data_map = laser_data_map
        trace_laser_time_map = {lk: time_sorted for lk in per_laser_keys}
        sidecar_keys = ([scalar_key] if scalar_key else []) + per_laser_keys
        sidecar_scalars = _load_sidecar_scalars(routine, data_dir, sidecar_keys)
        if "time" in sidecar_scalars:
            sidecar_time = sidecar_scalars["time"]
            if scalar_key in sidecar_scalars:
                trace_scalar_data = sidecar_scalars[scalar_key]
                trace_scalar_time = sidecar_time
            for lk in per_laser_keys:
                if lk in sidecar_scalars:
                    trace_laser_data_map[lk] = sidecar_scalars[lk]
                    trace_laser_time_map[lk] = sidecar_time

        peak_logical: int | None = None
        if field_key in _PEAK_FIELDS and scalar_data is not None:
            peak_logical = int(np.argmax(scalar_data))
            if peak_logical not in snap_logical:
                snap_logical = sorted(set(snap_logical) | {peak_logical})

        snap_physical = [int(sort_order[i]) for i in snap_logical]
        snapshots = [_read_field_frame(h5, spec, pi) for pi in snap_physical]

    n_cols = len(snap_logical)
    show_scalar_row = has_scalar or per_laser_keys
    n_rows = 2 if show_scalar_row else 1
    height_ratios = [3, 1] if n_rows == 2 else [1]

    if field_key == "psi_k":
        lx = abs(extent[1] - extent[0])
        ly = abs(extent[3] - extent[2])
        ny_grid, nx_grid = snapshots[0].shape
        kmax_x = np.pi * nx_grid / lx
        kmax_y = np.pi * ny_grid / ly
        plot_extent = [-kmax_x, kmax_x, -kmax_y, kmax_y]
        x_label = r"$k_x$ ($\mu$m$^{-1}$)"
        y_label = r"$k_y$ ($\mu$m$^{-1}$)"
    else:
        plot_extent = extent
        x_label = r"x ($\mu$m)"
        y_label = r"y ($\mu$m)"

    fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=height_ratios)

    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin = min(s.min() for s in snapshots)
        vmax = max(s.max() for s in snapshots)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = _make_norm(spec, vmin, vmax)

    for col, (li, snapshot) in enumerate(zip(snap_logical, snapshots)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            snapshot, origin="lower", extent=plot_extent, cmap=spec["cmap"],
            norm=norm, **({"vmin": vmin, "vmax": vmax} if norm is None else {}),
            aspect="equal",
        )
        is_peak = li == peak_logical
        t_label = f"t = {time_sorted[li]:.1f} ps"
        ax.set_title(f"{t_label} [PEAK]" if is_peak else t_label, fontsize=10,
                     color="crimson" if is_peak else "black", fontweight="bold" if is_peak else "normal")
        if col == 0:
            ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_scalar_row:
        ax_sc = fig.add_subplot(gs[1, :])
        if has_scalar and trace_scalar_data is not None:
            label = "Total" if per_laser_keys else scalar_key
            ax_sc.plot(trace_scalar_time[: len(trace_scalar_data)], trace_scalar_data,
                       linewidth=1.2, color="black", label=label)
        for lk in per_laser_keys:
            ld = trace_laser_data_map[lk]
            lt = trace_laser_time_map[lk]
            laser_label = lk.replace("P_max_", "Laser ") if lk.startswith("P_max_") else lk.replace("psi_sq_max_w", "Spot ")
            ax_sc.plot(lt[: len(ld)], ld, linewidth=0.9, alpha=0.8, label=laser_label)
        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(scalar_key if has_scalar else "P_max")
        ax_sc.legend(fontsize=8, loc="best")
        ax_sc.grid(True, alpha=0.3)
        for li in snap_logical:
            is_peak = li == peak_logical
            ax_sc.axvline(time_sorted[li],
                          color="crimson" if is_peak else "gray",
                          linestyle="--" if is_peak else ":",
                          alpha=0.8 if is_peak else 0.4,
                          linewidth=1.2 if is_peak else 0.6)

    fig.suptitle(f"{routine.upper()} \u2014 {spec['label']}", fontsize=14, fontweight="bold")

    out_dir = routine_dir(routine, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{field_key}.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")


def generate_scalar_traces(
    routine: str,
    data_dir: str,
    results_dir: str,
) -> None:
    """Generate per-scenario scalar traces from the lightweight sidecar."""
    sidecar = os.path.join(data_dir, f"{routine}_scalars.npz")
    if not os.path.isfile(sidecar):
        print(f"  WARNING: scalar sidecar not found, skipping traces: {sidecar}")
        return

    npz = np.load(sidecar)
    time = npz["time"]
    panels = [
        ("P_max", r"$P_{max}$ (local peak)"),
        ("P_area_integral", r"$\int P\,dA$"),
        ("P_cumulative_area_time_integral", r"$\int\!\!\int P\,dA\,dt$ (dose)"),
        ("psi_sq_max", r"$|\psi|^2_{max}$"),
        ("nI_max", r"$n_I^{max}$"),
        ("nA_max", r"$n_{active}^{max}$"),
        ("k0_frac", r"$k=0$ fraction"),
    ]

    fig = plt.figure(figsize=(11.0, 12.0), constrained_layout=True)
    gs = GridSpec(len(panels), 1, figure=fig)
    for row, (key, label) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        if key in npz:
            y = npz[key]
            ax.plot(time[: len(y)], y, color="black", linewidth=1.1, label="total")
        if key == "P_max":
            i = 0
            while f"P_max_{i}" in npz:
                y = npz[f"P_max_{i}"]
                ax.plot(time[: len(y)], y, linewidth=0.8, alpha=0.75, label=f"laser {i}")
                i += 1
            if i > 0:
                ax.legend(fontsize=8, ncols=min(i + 1, 4), loc="best")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    ax.set_xlabel("t (ps)")
    fig.suptitle(f"{routine.upper()} scalar traces", fontsize=14, fontweight="bold")

    out_dir = routine_dir(routine, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scalar_traces.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")

    _generate_roi_scalar_traces(routine, npz, time, results_dir)


def _roi_label(key: str) -> str:
    """Return a readable label for one ROI scalar key."""
    for suffix, label in ROI_LABEL_SUFFIXES.items():
        marker = f"_circle_{suffix}"
        if key.endswith(marker):
            roi_id = key[: -len(marker)].replace("roi_", "", 1)
            return f"{roi_id}: {label}"
    return key


def _generate_roi_scalar_traces(
    routine: str,
    npz: np.lib.npyio.NpzFile,
    time: np.ndarray,
    results_dir: str,
) -> None:
    """Generate ROI-specific trace panels when ROI metrics are present."""
    roi_keys = sorted(k for k in npz.files if k.startswith("roi_"))
    if not roi_keys:
        return

    suffix_order = list(ROI_LABEL_SUFFIXES.keys())
    grouped: dict[str, list[str]] = {suffix: [] for suffix in suffix_order}
    for key in roi_keys:
        for suffix in suffix_order:
            if key.endswith(f"_circle_{suffix}"):
                grouped[suffix].append(key)
                break

    panels = [(suffix, keys) for suffix, keys in grouped.items() if keys]
    fig = plt.figure(figsize=(11.0, max(3.0 * len(panels), 4.0)), constrained_layout=True)
    gs = GridSpec(len(panels), 1, figure=fig)
    for row, (suffix, keys) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        for key in keys:
            y = npz[key]
            ax.plot(time[: len(y)], y, linewidth=1.0, label=_roi_label(key))
        ax.set_ylabel(ROI_LABEL_SUFFIXES.get(suffix, suffix))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("t (ps)")
    fig.suptitle(f"{routine.upper()} ROI traces", fontsize=14, fontweight="bold")

    out_dir = routine_dir(routine, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roi_traces.png")
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


def generate_sweep_heatmaps(
    routines: list[str],
    data_dir: str,
    results_dir: str,
) -> None:
    """Generate power/separation heatmaps for parameter-sweep runs.

    Produces one figure per (base_scenario, sigma_time) pair so the sigma_time
    dimension is never silently collapsed.
    """
    rows: list[dict] = []
    for routine in routines:
        meta_path = os.path.join(data_dir, f"{routine}_meta.json")
        sidecar = os.path.join(data_dir, f"{routine}_scalars.npz")
        if not (os.path.isfile(meta_path) and os.path.isfile(sidecar)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        sweep = meta.get("sweep")
        if not sweep or sweep.get("mode") == "probe5":
            continue
        npz = np.load(sidecar)
        metrics = {}
        for key, _ in COMPARISON_SCALARS:
            if key in npz:
                metrics[key] = float(np.nanmax(npz[key]))
        for key in npz.files:
            if key.startswith("roi_"):
                metrics[key] = float(np.nanmax(npz[key]))
        rows.append({
            "sigma_space": 0.0,
            **sweep,
            "effective_power_definition": meta.get(
                "effective_power_definition",
                sweep.get("power_definition", "peak_amplitude"),
            ),
            "routine": routine,
            "metrics": metrics,
        })

    if not rows:
        return

    metric_keys = [key for key, _ in COMPARISON_SCALARS if any(key in row["metrics"] for row in rows)]
    metric_keys.extend(
        sorted(
            key
            for key in {k for row in rows for k in row["metrics"]}
            if key.startswith("roi_")
        )
    )
    metric_labels = dict(COMPARISON_SCALARS)
    bases = sorted({str(row["base_scenario"]) for row in rows})

    for base in bases:
        base_rows = [row for row in rows if row["base_scenario"] == base]
        sigma_times = sorted({float(row.get("sigma_time", 0.0)) for row in base_rows})

        for sigma_time in sigma_times:
            st_rows = [row for row in base_rows if float(row.get("sigma_time", 0.0)) == sigma_time]
            sigma_spaces = sorted({float(row.get("sigma_space", 0.0)) for row in st_rows})

            for sigma_space in sigma_spaces:
                sigma_rows = [
                    row for row in st_rows
                    if float(row.get("sigma_space", 0.0)) == sigma_space
                ]
                powers = sorted({float(row["power"]) for row in sigma_rows})
                x_key, x_label = _sweep_axis(sigma_rows)
                x_values = _sweep_axis_values(sigma_rows, x_key)
                if not powers or not x_values:
                    continue

                power_def = next(
                    (str(row.get("effective_power_definition", "peak_amplitude")) for row in sigma_rows),
                    "peak_amplitude",
                )
                power_axis_label = (
                    "pulse energy (model units)"
                    if power_def == "pulse_energy"
                    else "peak amplitude"
                )

                fig = plt.figure(
                    figsize=(max(5.0, 3.0 * len(metric_keys)), 4.8),
                    constrained_layout=True,
                )
                gs = GridSpec(1, len(metric_keys), figure=fig)
                for col, key in enumerate(metric_keys):
                    arr = np.full((len(powers), len(x_values)), np.nan, dtype=float)
                    for row in sigma_rows:
                        if key not in row["metrics"] or row.get(x_key) is None:
                            continue
                        pi = powers.index(float(row["power"]))
                        xi = x_values.index(float(row[x_key]))
                        arr[pi, xi] = row["metrics"][key]
                    ax = fig.add_subplot(gs[0, col])
                    im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
                    ax.set_title(metric_labels.get(key, _roi_label(key)), fontsize=9)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(power_axis_label if col == 0 else "")
                    ax.set_xticks(range(len(x_values)), [f"{v:g}" for v in x_values], rotation=45)
                    ax.set_yticks(range(len(powers)), [f"{v:g}" for v in powers])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                sigma_tag = f"{sigma_time:g}".replace(".", "p")
                sp_tag = f"{sigma_space:g}".replace(".", "p")
                fig.suptitle(
                    f"{base}  σ_t={sigma_time:g} ps  σ_s={sigma_space:g} μm — parameter sweep peaks",
                    fontsize=14,
                    fontweight="bold",
                )
                out_path = os.path.join(results_dir, f"sweep_heatmaps_{base}_sig{sigma_tag}_sp{sp_tag}.png")
                os.makedirs(results_dir, exist_ok=True)
                fig.savefig(out_path, dpi=PLOT_DPI)
                plt.close(fig)
                print(f"    {out_path}")


def _load_physics_for_diagnostics(data_dir: str) -> tuple[float, float]:
    """Return (R, gamma_C) from the run-local config if available."""
    config_path = os.path.join(data_dir, "config.yaml")
    if not os.path.isfile(config_path):
        return 0.02, 0.083
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    physics = cfg.get("global", {}).get("physics", {})
    return float(physics.get("R", 0.02)), float(physics.get("gamma_C", 0.083))


def _load_sweep_diagnostic_rows(routines: list[str], data_dir: str) -> list[dict]:
    """Load scalar sidecars and sweep metadata into row dictionaries."""
    rows: list[dict] = []
    for routine in routines:
        meta_path = os.path.join(data_dir, f"{routine}_meta.json")
        sidecar = os.path.join(data_dir, f"{routine}_scalars.npz")
        if not (os.path.isfile(meta_path) and os.path.isfile(sidecar)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        sweep = meta.get("sweep")
        if not sweep or sweep.get("mode") == "probe5":
            continue
        npz = np.load(sidecar)
        rows.append({
            "sigma_space": 0.0,
            **sweep,
            "effective_power_definition": meta.get(
                "effective_power_definition",
                sweep.get("power_definition", "peak_amplitude"),
            ),
            "routine": routine,
            "time": np.array(npz["time"], dtype=float),
            "psi_sq_max": np.array(npz["psi_sq_max"], dtype=float) if "psi_sq_max" in npz else None,
            "nA_max": np.array(npz["nA_max"], dtype=float) if "nA_max" in npz else None,
            "nI_max": np.array(npz["nI_max"], dtype=float) if "nI_max" in npz else None,
            "k0_frac": np.array(npz["k0_frac"], dtype=float) if "k0_frac" in npz else None,
        })
    return rows


def _positive_duration(time: np.ndarray, mask: np.ndarray) -> float:
    """Approximate total duration where mask is true on a sampled time grid."""
    if time.size < 2 or mask.size == 0:
        return 0.0
    dt = np.diff(time, prepend=time[0])
    if dt.size > 1:
        dt[0] = dt[1]
    return float(np.sum(dt[: len(mask)][mask]))


def generate_sweep_diagnostics(
    routines: list[str],
    data_dir: str,
    results_dir: str,
) -> None:
    """Generate diagnostic sweep maps focused on threshold-search usefulness."""
    rows = _load_sweep_diagnostic_rows(routines, data_dir)
    if not rows:
        return

    R, gamma_C = _load_physics_for_diagnostics(data_dir)
    bases = sorted({str(row["base_scenario"]) for row in rows})

    for base in bases:
        base_rows = [row for row in rows if row["base_scenario"] == base]
        sigma_times = sorted({float(row.get("sigma_time", 0.0)) for row in base_rows})

        for sigma_time in sigma_times:
            st_rows = [
                row for row in base_rows
                if float(row.get("sigma_time", 0.0)) == sigma_time
            ]
            sigma_spaces = sorted({float(row.get("sigma_space", 0.0)) for row in st_rows})

            for sigma_space in sigma_spaces:
                sigma_rows = [
                    row for row in st_rows
                    if float(row.get("sigma_space", 0.0)) == sigma_space
                ]
                powers = sorted({float(row["power"]) for row in sigma_rows})
                x_key, x_label = _sweep_axis(sigma_rows)
                x_values = _sweep_axis_values(sigma_rows, x_key)
                if not powers or not x_values:
                    continue

                power_def = next(
                    (str(row.get("effective_power_definition", "peak_amplitude")) for row in sigma_rows),
                    "peak_amplitude",
                )
                power_axis_label = (
                    "pulse energy (model units)"
                    if power_def == "pulse_energy"
                    else "peak amplitude"
                )

                maps = {
                    "log10 psi peak / psi0": np.full((len(powers), len(x_values)), np.nan),
                    "log10 psi final / psi0": np.full((len(powers), len(x_values)), np.nan),
                    "n_active peak": np.full((len(powers), len(x_values)), np.nan),
                    "max(R n_active - gamma_C)": np.full((len(powers), len(x_values)), np.nan),
                    "time gain > 0 (ps)": np.full((len(powers), len(x_values)), np.nan),
                    "nI peak": np.full((len(powers), len(x_values)), np.nan),
                }

                for row in sigma_rows:
                    if row.get(x_key) is None:
                        continue
                    pi = powers.index(float(row["power"]))
                    xi = x_values.index(float(row[x_key]))
                    time = row["time"]
                    psi = row["psi_sq_max"]
                    nA = row["nA_max"]
                    nI = row["nI_max"]

                    if psi is not None and psi.size:
                        psi0 = max(float(psi[0]), 1e-300)
                        late = psi[time[: len(psi)] >= PSI_DIAGNOSTIC_IGNORE_PS]
                        if late.size:
                            maps["log10 psi peak / psi0"][pi, xi] = float(
                                np.log10(max(float(np.nanmax(late)), 1e-300) / psi0)
                            )
                        maps["log10 psi final / psi0"][pi, xi] = float(
                            np.log10(max(float(psi[-1]), 1e-300) / psi0)
                        )
                    if nA is not None and nA.size:
                        gain_margin = R * nA - gamma_C
                        maps["n_active peak"][pi, xi] = float(np.nanmax(nA))
                        maps["max(R n_active - gamma_C)"][pi, xi] = float(np.nanmax(gain_margin))
                        maps["time gain > 0 (ps)"][pi, xi] = _positive_duration(
                            time[: len(gain_margin)], gain_margin > 0.0
                        )
                    if nI is not None and nI.size:
                        maps["nI peak"][pi, xi] = float(np.nanmax(nI))

                fig = plt.figure(figsize=(13.0, 7.2), constrained_layout=True)
                gs = GridSpec(2, 3, figure=fig)
                for idx, (title, arr) in enumerate(maps.items()):
                    ax = fig.add_subplot(gs[idx // 3, idx % 3])
                    im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(power_axis_label if idx % 3 == 0 else "")
                    ax.set_xticks(range(len(x_values)), [f"{v:g}" for v in x_values])
                    ax.set_yticks(range(len(powers)), [f"{v:g}" for v in powers])
                    for pi, _ in enumerate(powers):
                        for xi, _ in enumerate(x_values):
                            val = arr[pi, xi]
                            if np.isfinite(val):
                                ax.text(xi, pi, f"{val:.2g}", ha="center", va="center",
                                        color="white", fontsize=7)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                sigma_tag = f"{sigma_time:g}".replace(".", "p")
                sp_tag = f"{sigma_space:g}".replace(".", "p")
                fig.suptitle(
                    f"{base}  sigma_t={sigma_time:g} ps  sigma_s={sigma_space:g} μm — sweep diagnostics",
                    fontsize=14,
                    fontweight="bold",
                )
                out_path = os.path.join(results_dir, f"sweep_diagnostics_{base}_sig{sigma_tag}_sp{sp_tag}.png")
                os.makedirs(results_dir, exist_ok=True)
                fig.savefig(out_path, dpi=PLOT_DPI)
                plt.close(fig)
                print(f"    {out_path}")
