"""Amplitude sweep over the time-only polariton ODE.

Runs the scalar ODE model from :mod:`polarism.response.time_only` for a
sequence of pump amplitudes and accumulates:

* the full ``nC(t)`` trajectory per amplitude,
* the integral ``∫ nC(t) dt`` (trapezoid rule),
* the pulse profile ``P(t)`` per amplitude.

The result dict is serialisable to JSON so Stage 1 of the fitting pipeline
can write ``target_response.json`` and later stages can reload it without
re-running the ODE.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from polarism.response.time_only import fwhm_to_sigma, gaussian_pulse, simulate_single_amplitude


def run_amplitude_sweep(
    amplitudes: np.ndarray,
    center: float,
    width_fwhm: float,
    t_eval: np.ndarray,
    params: dict[str, float],
) -> dict[str, Any]:
    """Run the ODE for every amplitude in *amplitudes*.

    Parameters
    ----------
    amplitudes : np.ndarray, shape (M,)
        Pump peak amplitudes to sweep.
    center : float
        Pulse centre time (ps).
    width_fwhm : float
        Pulse FWHM (ps).
    t_eval : np.ndarray, shape (T,)
        Time evaluation grid (ps).
    params : dict
        Physics parameters (see :func:`polarism.response.time_only.ode_rhs`).

    Returns
    -------
    dict with keys:

    ``"t_eval"``
        Time grid as a Python list (for JSON serialisation).
    ``"amplitudes"``
        Amplitude values as a Python list.
    ``"nc_integrals"``
        ``∫ nC(t) dt`` for each amplitude, as a list.
    ``"nc_trajectories"``
        2-D list ``(M, T)`` of ``nC(t)`` values.
    ``"pulse_trajectories"``
        2-D list ``(M, T)`` of pump pulse values.
    ``"params"``
        Copy of *params* for provenance.
    ``"center"``
        Pulse centre (ps).
    ``"width_fwhm"``
        Pulse FWHM (ps).
    """
    nc_integrals: list[float] = []
    nc_trajectories: list[list[float]] = []
    pulse_trajectories: list[list[float]] = []

    for amp in amplitudes:
        _, _, nC_t = simulate_single_amplitude(
            float(amp), center, width_fwhm, t_eval, params
        )
        pulse_t = gaussian_pulse(t_eval, float(amp), center, fwhm_to_sigma(width_fwhm))
        nc_integrals.append(float(np.trapezoid(nC_t, t_eval)))
        nc_trajectories.append(nC_t.tolist())
        pulse_trajectories.append(pulse_t.tolist())

    return {
        "t_eval": t_eval.tolist(),
        "amplitudes": amplitudes.tolist(),
        "nc_integrals": nc_integrals,
        "nc_trajectories": nc_trajectories,
        "pulse_trajectories": pulse_trajectories,
        "params": dict(params),
        "center": center,
        "width_fwhm": width_fwhm,
    }


def normalized_integral_curve(nc_integrals: list[float]) -> np.ndarray:
    """Return *nc_integrals* normalised to ``[0, 1]``.

    If the range is zero (all values identical) the array is left unchanged.

    Parameters
    ----------
    nc_integrals : list[float]
        Raw integral values, one per amplitude.

    Returns
    -------
    np.ndarray
        Normalised curve.
    """
    arr = np.asarray(nc_integrals, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-30:
        return arr
    return (arr - lo) / (hi - lo)


def rmse_score(
    target_integrals: list[float],
    candidate_integrals: list[float],
) -> float:
    """Compute the RMSE between two normalised amplitude-response curves.

    Both curves are normalised to ``[0, 1]`` independently before comparison
    so that absolute scale differences do not dominate the score.

    Parameters
    ----------
    target_integrals : list[float]
        Reference ``∫ nC(t) dt`` values from the 1-D ODE.
    candidate_integrals : list[float]
        Candidate observable values from the 2-D simulation or another ODE run.

    Returns
    -------
    float
        Root-mean-square error of the normalised curves.
    """
    target_norm = normalized_integral_curve(target_integrals)
    cand_norm = normalized_integral_curve(candidate_integrals)
    n = min(len(target_norm), len(cand_norm))
    return float(np.sqrt(np.mean((target_norm[:n] - cand_norm[:n]) ** 2)))
