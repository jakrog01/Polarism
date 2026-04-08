"""Scalar (time-only) polariton ODE model.

State order: ``(nR, nI, nC)``
------------------------------
* ``nR`` — active reservoir density
* ``nI`` — inactive reservoir density (directly driven by pump)
* ``nC`` — condensate density (scalar analogue of ``|ψ|²``)

Equations of motion
-------------------
``dnR/dt = kappa * nI² - gammaR * nR - R * nR * nC``
``dnI/dt = P(t)  - kappa * nI² - gammaI * nI``
``dnC/dt = R * nR * nC - gammaC * nC``

Condensate growth is seeded by the non-zero initial condition in
``INITIAL_STATE``, consistent with the 2-D GPE model which seeds
``|ψ₀|²`` via ``init_eps``-scale noise.

Physics parameters
------------------
The ``params`` dict used throughout this module expects the keys:

* ``"kappa"``  — quadratic inactive→active transfer coefficient
* ``"gammaR"`` — active reservoir decay rate
* ``"gammaI"`` — inactive reservoir decay rate
* ``"gammaC"`` — condensate decay rate
* ``"R"``      — stimulated scattering coefficient
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

INITIAL_STATE: tuple[float, float, float] = (0.0, 0.0, 1e-6)
SOLVER_PARAMS: dict[str, float] = {"rtol": 1e-6, "atol": 1e-7}


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert a FWHM width to a Gaussian sigma.

    Parameters
    ----------
    fwhm : float
        Full-width at half-maximum (same units as the time axis).

    Returns
    -------
    float
        Gaussian standard deviation ``sigma``.
    """
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_pulse(
    t: np.ndarray,
    amp: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a single Gaussian pulse on a time grid.

    Parameters
    ----------
    t : np.ndarray
        Time evaluation points (ps).
    amp : float
        Peak amplitude.
    center : float
        Pulse centre time (ps).
    sigma : float
        Gaussian standard deviation (ps).

    Returns
    -------
    np.ndarray
        Pulse values at each point in *t*.
    """
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def generate_pump(
    amps: np.ndarray,
    centers: np.ndarray,
    width_fwhm: float,
    t_eval: np.ndarray,
) -> interp1d:
    """Build a callable pump interpolant from a set of Gaussian pulses.

    Parameters
    ----------
    amps : np.ndarray, shape (N,)
        Peak amplitudes of the pulses.
    centers : np.ndarray, shape (N,)
        Centre times of the pulses (ps).
    width_fwhm : float
        FWHM pulse width (ps).
    t_eval : np.ndarray
        Time grid used for the interpolant.

    Returns
    -------
    scipy.interpolate.interp1d
        Callable ``P(t)`` returning the total pump at time *t*.
        Returns 0 outside the *t_eval* range.
    """
    sigma = fwhm_to_sigma(width_fwhm)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma ** 2)
    pump = np.zeros_like(t_eval)
    for a, c in zip(amps, centers):
        if a > 0:
            pump += a * np.exp(-(t_eval - c) ** 2 * inv_two_sigma_sq)
    return interp1d(t_eval, pump, bounds_error=False, fill_value=0.0)


def ode_rhs(
    t: float,
    y: list[float],
    amp: float,
    center: float,
    sigma: float,
    params: dict[str, float],
) -> list[float]:
    """ODE right-hand side for the time-only polariton model.

    State order: ``(nR, nI, nC)``.

    Parameters
    ----------
    t : float
        Current time (ps).
    y : list[float]
        Current state ``[nR, nI, nC]``.
    amp : float
        Pulse peak amplitude.
    center : float
        Pulse centre time (ps).
    sigma : float
        Pulse Gaussian sigma (ps).
    params : dict
        Physics parameters with keys ``"kappa"``, ``"gammaR"``,
        ``"gammaI"``, ``"gammaC"``, ``"R"``.

    Returns
    -------
    list[float]
        Derivatives ``[dnR/dt, dnI/dt, dnC/dt]``.
    """
    nR, nI, nC = y
    pulse = amp * math.exp(-0.5 * ((t - center) / sigma) ** 2)
    transfer = params["kappa"] * nI ** 2
    dnR = transfer - params["gammaR"] * nR - params["R"] * nR * nC
    dnI = pulse - transfer - params["gammaI"] * nI
    dnC = params["R"] * nR * nC - params["gammaC"] * nC
    return [dnR, dnI, dnC]


def simulate_single_amplitude(
    amp: float,
    center: float,
    width_fwhm: float,
    t_eval: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the ODE for a single pump amplitude.

    Parameters
    ----------
    amp : float
        Peak pump amplitude.
    center : float
        Pulse centre time (ps).
    width_fwhm : float
        Pulse FWHM (ps).
    t_eval : np.ndarray
        Time points at which to record the solution.
    params : dict
        Physics parameters (see :func:`ode_rhs`).

    Returns
    -------
    nR_t : np.ndarray
        Active reservoir trajectory.
    nI_t : np.ndarray
        Inactive reservoir trajectory.
    nC_t : np.ndarray
        Condensate density trajectory.
    """
    sigma = fwhm_to_sigma(width_fwhm)
    sol = solve_ivp(
        ode_rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        list(INITIAL_STATE),
        args=(amp, center, sigma, params),
        t_eval=t_eval,
        rtol=SOLVER_PARAMS["rtol"],
        atol=SOLVER_PARAMS["atol"],
    )
    return sol.y[0], sol.y[1], sol.y[2]


def params_from_physics(physics: Any) -> dict[str, float]:
    """Extract time-only ODE parameters from a ``PhysicsConstants`` object.

    Parameters
    ----------
    physics : PhysicsConstants
        Polarism physics constants dataclass.

    Returns
    -------
    dict[str, float]
        Parameter dict ready for :func:`ode_rhs` and related functions.
    """
    return {
        "kappa": physics.kappa,
        "gammaR": physics.gamma_R,
        "gammaI": physics.gamma_I,
        "gammaC": physics.gamma_C,
        "R": physics.R,
    }
