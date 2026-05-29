"""Quantitative interference fringe analysis for polariton field snapshots.

All public functions operate on NumPy arrays only; no GPU/CuPy dependency.
Intended to be called from the GPU stage while the HDF5 is still on scratch.
"""
from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np


_EPS = 1e-30
_DC_EXCLUDE_FRACTION = 0.02


def fringe_contrast_p95_p05(psi_sq_flat: np.ndarray) -> float:
    """Robust Michelson-like contrast from the 5th and 95th percentile.

    Parameters
    ----------
    psi_sq_flat : np.ndarray
        1-D array of |ψ|² values inside the analysis window.

    Returns
    -------
    float
        ``(p95 - p05) / (p95 + p05 + eps)`` in ``[0, 1)``.
    """
    p95 = float(np.percentile(psi_sq_flat, 95))
    p05 = float(np.percentile(psi_sq_flat, 5))
    return (p95 - p05) / (p95 + p05 + _EPS)


def fringe_cv(psi_sq_flat: np.ndarray) -> float:
    """Coefficient of variation of |ψ|² inside the analysis window.

    Parameters
    ----------
    psi_sq_flat : np.ndarray
        1-D array of |ψ|² values.

    Returns
    -------
    float
        ``std / (mean + eps)``.
    """
    return float(np.std(psi_sq_flat)) / (float(np.mean(psi_sq_flat)) + _EPS)


def fringe_fft_peak_k(
    psi_sq_crop: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Dominant spatial frequency of fringes via 2-D FFT.

    Applies a 2-D Hann taper to the crop, subtracts the mean, then locates
    the peak in the power spectrum outside the DC/very-low-k region.

    Parameters
    ----------
    psi_sq_crop : np.ndarray
        2-D crop of |ψ|² centred on the square centre.
    dx, dy : float
        Grid spacings in μm.

    Returns
    -------
    (k_peak_um, fringe_spacing_um)
        Dominant radial wavenumber in μm⁻¹ and corresponding fringe spacing
        ``2π / k_peak`` in μm.  Returns ``(0.0, 0.0)`` when no significant
        peak is found.
    """
    ny_c, nx_c = psi_sq_crop.shape
    if ny_c < 4 or nx_c < 4:
        return 0.0, 0.0

    win_y = np.hanning(ny_c)
    win_x = np.hanning(nx_c)
    window = np.outer(win_y, win_x)
    field = (psi_sq_crop - psi_sq_crop.mean()) * window

    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(nx_c, d=dx)) * 2.0 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny_c, d=dy)) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)

    k_nyq = np.pi / min(dx, dy)
    dc_cutoff = _DC_EXCLUDE_FRACTION * k_nyq
    mask = kr > dc_cutoff
    if not np.any(mask):
        return 0.0, 0.0

    power_masked = power * mask
    if power_masked.max() < _EPS:
        return 0.0, 0.0

    peak_idx = int(np.argmax(power_masked))
    k_peak = float(kr.ravel()[peak_idx])
    if k_peak < _EPS:
        return 0.0, 0.0
    return k_peak, 2.0 * np.pi / k_peak


def _line_contrast_and_spacing(line: np.ndarray, dx: float) -> dict[str, float]:
    """Return 1-D fringe contrast and dominant spacing for a single line scan.

    Parameters
    ----------
    line : np.ndarray
        1-D |ψ|² profile.
    dx : float
        Pixel spacing in μm.

    Returns
    -------
    dict with keys ``contrast``, ``dominant_spacing_um``, ``dominant_k_um``.
    """
    n = len(line)
    if n < 4:
        return {"contrast": 0.0, "dominant_spacing_um": 0.0, "dominant_k_um": 0.0}

    peak = float(line.max())
    valley = float(line.min())
    denom = peak + valley + _EPS
    contrast = (peak - valley) / denom

    window = np.hanning(n)
    power = np.abs(np.fft.rfft((line - line.mean()) * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=dx)
    if len(freqs) < 2:
        return {"contrast": contrast, "dominant_spacing_um": 0.0, "dominant_k_um": 0.0}

    power[0] = 0.0
    peak_idx = int(np.argmax(power))
    freq_peak = float(freqs[peak_idx])
    k_peak = 2.0 * np.pi * freq_peak
    spacing = 1.0 / freq_peak if freq_peak > _EPS else 0.0

    return {
        "contrast": contrast,
        "dominant_spacing_um": spacing,
        "dominant_k_um": k_peak,
    }


def directional_line_scans(
    psi_sq_2d: np.ndarray,
    cy_idx: int,
    cx_idx: int,
    half_n: int,
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Compute 1-D fringe metrics along H, V, and diagonal directions.

    Parameters
    ----------
    psi_sq_2d : np.ndarray
        Full 2-D |ψ|² field.
    cy_idx, cx_idx : int
        Centre pixel indices (row, col).
    half_n : int
        Half-length of each line scan in pixels.
    dx, dy : float
        Grid spacings in μm.

    Returns
    -------
    dict
        Keys ``horizontal``, ``vertical``, ``diag_main``, ``diag_anti``,
        each containing a sub-dict from :func:`_line_contrast_and_spacing`.
    """
    ny, nx = psi_sq_2d.shape
    r0, r1 = max(0, cy_idx - half_n), min(ny, cy_idx + half_n + 1)
    c0, c1 = max(0, cx_idx - half_n), min(nx, cx_idx + half_n + 1)

    h_line = psi_sq_2d[cy_idx, c0:c1]
    v_line = psi_sq_2d[r0:r1, cx_idx]
    crop = psi_sq_2d[r0:r1, c0:c1]
    diag_main = np.diag(crop)
    diag_anti = np.diag(np.fliplr(crop))

    d_diag = float(np.sqrt(dx ** 2 + dy ** 2))
    return {
        "horizontal": _line_contrast_and_spacing(h_line, dx),
        "vertical": _line_contrast_and_spacing(v_line, dy),
        "diag_main": _line_contrast_and_spacing(diag_main, d_diag),
        "diag_anti": _line_contrast_and_spacing(diag_anti, d_diag),
    }


def _build_circular_mask(ny: int, nx: int, cy: int, cx: int, radius_px: float) -> np.ndarray:
    """Return a boolean mask for pixels within *radius_px* of *(cy, cx)*."""
    rows = np.arange(ny) - cy
    cols = np.arange(nx) - cx
    R, C = np.meshgrid(rows, cols, indexing="ij")
    return (R ** 2 + C ** 2) <= radius_px ** 2


def _infill_pump_cores(
    psi_sq: np.ndarray,
    valid_mask: np.ndarray,
    pump_cores_px: list[tuple[float, float]],
    excl_px: float,
) -> np.ndarray:
    """Return a copy of *psi_sq* with pump-core pixels replaced by the median of valid pixels.

    The infill prevents sharp pump-core edges from injecting spurious high-k power
    into the FFT and from biasing line-scan contrast estimates.

    Parameters
    ----------
    psi_sq : np.ndarray
        Full 2-D |ψ|² field.
    valid_mask : np.ndarray
        Boolean mask of pixels considered valid (fringe window minus cores).
        Used to compute the fill value.
    pump_cores_px : list of (row, col) floats in full-grid pixel coordinates.
    excl_px : float
        Exclusion radius in pixels.
    """
    if not pump_cores_px:
        return psi_sq
    ny, nx = psi_sq.shape
    rows = np.arange(ny, dtype=np.float64)[:, None]
    cols = np.arange(nx, dtype=np.float64)[None, :]
    core_union = np.zeros((ny, nx), dtype=bool)
    for cy, cx in pump_cores_px:
        core_union |= ((rows - cy) ** 2 + (cols - cx) ** 2) <= excl_px ** 2
    valid_vals = psi_sq[valid_mask]
    fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
    out = psi_sq.copy()
    out[core_union] = fill_val
    return out


def _build_fringe_mask(
    ny: int,
    nx: int,
    cy_idx: int,
    cx_idx: int,
    fringe_radius_px: float,
    dx: float,
    dy: float,
    pump_cores: list[tuple[float, float]],
    pump_core_exclusion_radius: float,
) -> np.ndarray:
    """Fringe window mask with pump-core holes punched out.

    Returns a boolean array where True = pixel belongs to the analysis window
    (inside fringe circle and outside all pump-core exclusion circles).
    """
    mask = _build_circular_mask(ny, nx, cy_idx, cx_idx, fringe_radius_px)
    excl_px = pump_core_exclusion_radius / min(dx, dy)
    for x0, y0 in pump_cores:
        core_cx = cx_idx + round(x0 / dx)
        core_cy = cy_idx + round(y0 / dy)
        mask &= ~_build_circular_mask(ny, nx, core_cy, core_cx, excl_px)
    return mask


def compute_fringe_time_series(
    h5_path: str,
    fringe_window_radius: float,
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    pump_cores: list[tuple[float, float]] | None = None,
    pump_core_exclusion_radius: float = 6.0,
    t_min_ps: float = 0.0,
) -> dict[str, Any]:
    """Read HDF5 frame by frame and compute per-frame fringe metrics.

    Parameters
    ----------
    h5_path : str
        Path to the scenario HDF5 file.
    fringe_window_radius : float
        Radius of the circular fringe analysis window in μm, centred at (0, 0).
    lx, ly : float
        Simulation box full widths in μm.
    nx, ny : int
        Grid dimensions.
    pump_cores : list of (x0, y0) tuples, optional
        Pump spot centre positions in μm.  Pixels within
        *pump_core_exclusion_radius* of each core are excluded from the
        fringe window.
    pump_core_exclusion_radius : float
        Exclusion radius per pump core in μm.  Defaults to
        ``cutoff_sigma × sigma_space = 3 × 2 = 6 μm``.
    t_min_ps : float
        Skip frames recorded before this time (ps).  Use to gate out frames
        where the pump is still firing.

    Returns
    -------
    dict
        Keys: ``times`` (list of float), ``metrics`` (list of per-frame dicts),
        ``grid`` (dict with dx, dy, cx_idx, cy_idx, radius_px, n_mask_pixels).
    """
    dx = lx / nx
    dy = ly / ny
    cx_idx = nx // 2
    cy_idx = ny // 2
    radius_px = fringe_window_radius / min(dx, dy)

    mask_2d = _build_fringe_mask(
        ny, nx, cy_idx, cx_idx, radius_px, dx, dy,
        pump_cores or [], pump_core_exclusion_radius,
    )
    n_mask = int(mask_2d.sum())
    r_px = int(np.ceil(radius_px))
    excl_px = pump_core_exclusion_radius / min(dx, dy)
    pump_cores_px = [
        (cy_idx + round(y0 / dy), cx_idx + round(x0 / dx))
        for x0, y0 in (pump_cores or [])
    ]

    with h5py.File(h5_path, "r") as h5:
        raw_times = h5["time"][:]
        order = np.argsort(raw_times, kind="stable")
        times_sorted = raw_times[order]

        per_frame: list[dict[str, Any]] = []
        for seq_i, idx in enumerate(order):
            t_frame = float(times_sorted[seq_i])
            if t_frame < t_min_ps:
                continue

            psi_frame = np.asarray(h5["fields/psi"][idx])
            psi_sq = np.abs(psi_frame) ** 2

            flat = psi_sq[mask_2d]
            if len(flat) < 4:
                continue
            contrast = fringe_contrast_p95_p05(flat)
            cv = fringe_cv(flat)

            psi_sq_filled = _infill_pump_cores(psi_sq, mask_2d, pump_cores_px, excl_px)

            r0 = max(0, cy_idx - r_px)
            r1 = min(ny, cy_idx + r_px + 1)
            c0 = max(0, cx_idx - r_px)
            c1 = min(nx, cx_idx + r_px + 1)
            crop = psi_sq_filled[r0:r1, c0:c1]
            k_peak, fringe_spacing = fringe_fft_peak_k(crop, dx, dy)

            scans = directional_line_scans(psi_sq_filled, cy_idx, cx_idx, r_px, dx, dy)

            per_frame.append({
                "t": t_frame,
                "fringe_contrast_p95_p05": contrast,
                "fringe_cv": cv,
                "fringe_fft_peak_k_um": k_peak,
                "fringe_spacing_um": fringe_spacing,
                "line_scans": scans,
                "psi_sq_window_max": float(flat.max()),
                "psi_sq_window_mean": float(flat.mean()),
            })

    return {
        "times": [m["t"] for m in per_frame],
        "metrics": per_frame,
        "grid": {
            "dx": dx,
            "dy": dy,
            "cx_idx": cx_idx,
            "cy_idx": cy_idx,
            "radius_px": float(radius_px),
            "fringe_window_radius_um": fringe_window_radius,
            "n_mask_pixels": n_mask,
            "pump_core_exclusion_radius_um": pump_core_exclusion_radius,
            "t_min_ps": t_min_ps,
        },
    }


def aggregate_fringe_scalars(
    time_series: dict[str, Any],
    threshold_criterion: float = 5e-2,
) -> dict[str, Any]:
    """Aggregate per-frame fringe metrics into scenario-level scalars.

    Parameters
    ----------
    time_series : dict
        Output of :func:`compute_fringe_time_series`.
    threshold_criterion : float
        |ψ|² threshold for the condensation check.

    Returns
    -------
    dict
        Scalar summary for one scenario.
    """
    times = time_series["times"]
    metrics = time_series["metrics"]
    if not metrics:
        return {}

    contrasts = np.array([m["fringe_contrast_p95_p05"] for m in metrics])
    cvs = np.array([m["fringe_cv"] for m in metrics])
    spacings = np.array([m["fringe_spacing_um"] for m in metrics])
    window_maxes = np.array([m["psi_sq_window_max"] for m in metrics])

    best_idx = int(np.argmax(contrasts))
    t_arr = np.array(times)

    h_contrasts = np.array([m["line_scans"]["horizontal"]["contrast"] for m in metrics])
    v_contrasts = np.array([m["line_scans"]["vertical"]["contrast"] for m in metrics])

    return {
        "fringe_contrast_max": float(contrasts.max()),
        "t_fringe_contrast_max": float(t_arr[best_idx]),
        "fringe_spacing_at_max_contrast": float(spacings[best_idx]),
        "fringe_fft_peak_k_at_max_contrast": float(metrics[best_idx]["fringe_fft_peak_k_um"]),
        "fringe_cv_max": float(cvs.max()),
        "fringe_window_psi_sq_max": float(window_maxes.max()),
        "t_fringe_window_psi_sq_max": float(t_arr[int(np.argmax(window_maxes))]),
        "h_contrast_max": float(h_contrasts.max()),
        "v_contrast_max": float(v_contrasts.max()),
        "h_spacing_at_max": float(metrics[int(np.argmax(h_contrasts))]["line_scans"]["horizontal"]["dominant_spacing_um"]),
        "v_spacing_at_max": float(metrics[int(np.argmax(v_contrasts))]["line_scans"]["vertical"]["dominant_spacing_um"]),
        "crossed_threshold": bool(window_maxes.max() >= threshold_criterion),
    }


def compute_fringe_sidecar(
    scenario_name: str,
    data_dir: str,
    fringe_window_radius: float,
    sim_cfg: Any,
    threshold_criterion: float = 5e-2,
    pump_cores: list[tuple[float, float]] | None = None,
    pump_core_exclusion_radius: float = 6.0,
    t_min_ps: float = 0.0,
) -> dict[str, Any]:
    """Compute and return the full fringe sidecar dict for one scenario.

    Parameters
    ----------
    scenario_name : str
        Stem used for the HDF5 filename.
    data_dir : str
        Directory containing ``{scenario_name}.h5``.
    fringe_window_radius : float
        Radius of the fringe analysis window in μm.
    sim_cfg : Config
        Simulation config (used for grid dimensions).
    threshold_criterion : float
        Condensation threshold for ``crossed_threshold`` flag.
    pump_cores : list of (x0, y0) tuples, optional
        Pump spot centres in μm.  Their neighbourhoods are excluded from
        the fringe window.
    pump_core_exclusion_radius : float
        Exclusion radius per pump core in μm.
    t_min_ps : float
        Minimum frame time in ps.  Frames before this are skipped so that
        direct pump emission does not inflate fringe metrics.

    Returns
    -------
    dict
        Ready to serialise as JSON; contains ``aggregated`` and ``grid`` keys.
    """
    h5_path = os.path.join(data_dir, f"{scenario_name}.h5")
    if not os.path.isfile(h5_path):
        return {"error": f"HDF5 not found: {h5_path}"}

    lx = float(sim_cfg.grid.lx)
    ly = float(sim_cfg.grid.ly)
    nx = int(sim_cfg.grid.nx)
    ny = int(sim_cfg.grid.ny)

    ts = compute_fringe_time_series(
        h5_path, fringe_window_radius, lx, ly, nx, ny,
        pump_cores=pump_cores,
        pump_core_exclusion_radius=pump_core_exclusion_radius,
        t_min_ps=t_min_ps,
    )
    aggregated = aggregate_fringe_scalars(ts, threshold_criterion)

    return {
        "scenario": scenario_name,
        "fringe_window_radius_um": fringe_window_radius,
        "pump_core_exclusion_radius_um": pump_core_exclusion_radius,
        "t_min_ps": t_min_ps,
        "n_frames_analysed": len(ts["times"]),
        "grid": ts["grid"],
        "aggregated": aggregated,
    }
