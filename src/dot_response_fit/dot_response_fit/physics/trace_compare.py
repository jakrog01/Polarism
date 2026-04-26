"""Trace scoring and normalization shared by fit_dot_size and run_scenario.

ODE state canonical order: [nR (active reservoir), nI (inactive reservoir), nC (condensate)].
nA in HDF5 corresponds to nR in ODE equations (polarism naming convention).

Both stages import from here so that fit scores and per-image RMSE plots use
identical normalization and resampling.
"""
from __future__ import annotations

import numpy as np

_RESAMPLE_POINTS = 500


def normalize_trace(trace: np.ndarray) -> np.ndarray:
    """Divide by maximum value; return zeros if max < 1e-15."""
    mx = float(trace.max())
    if mx < 1.0e-15:
        return np.zeros_like(trace)
    return trace / mx


def score_traces(
    t_ref: np.ndarray,
    nC_ref: np.ndarray,
    t_sim: np.ndarray,
    obs_sim: np.ndarray,
    n_resample: int = _RESAMPLE_POINTS,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RMSE on shape-normalised traces over the overlapping time window.

    Parameters
    ----------
    t_ref : np.ndarray
        ODE reference time axis (ps).
    nC_ref : np.ndarray
        ODE condensate trace nC(t).
    t_sim : np.ndarray
        Simulation time axis (ps).
    obs_sim : np.ndarray
        Simulation observable (e.g. psi_sq_max) trace.
    n_resample : int
        Number of points in the common resampling axis.

    Returns
    -------
    rmse : float
        inf when either trace max < 1e-15 or windows do not overlap.
    t_common : np.ndarray
    ref_norm : np.ndarray
    sim_norm : np.ndarray
    """
    t_lo = max(float(t_ref[0]), float(t_sim[0]))
    t_hi = min(float(t_ref[-1]), float(t_sim[-1]))
    if t_hi <= t_lo:
        empty = np.empty(0)
        return float("inf"), empty, empty, empty

    t_common = np.linspace(t_lo, t_hi, n_resample)
    ref_r = np.interp(t_common, t_ref, nC_ref)
    sim_r = np.interp(t_common, t_sim, obs_sim)

    ref_norm = normalize_trace(ref_r)
    sim_norm = normalize_trace(sim_r)

    if ref_norm.max() < 1.0e-15 or sim_norm.max() < 1.0e-15:
        return float("inf"), t_common, ref_norm, sim_norm

    rmse = float(np.sqrt(np.mean((ref_norm - sim_norm) ** 2)))
    return rmse, t_common, ref_norm, sim_norm
