"""Quadratic-double-reservoir ODE for condensate reference traces.

Shared by Stage 1 (prepare_reference) and Stage 2 (fit_dot_size) so both
use identical physics when producing and comparing nC(t) reference traces.

ODE:
    dnR = kappa * nI^2 - gammaR * nR - R * nR * nC
    dnI = P(t) - kappa * nI^2 - gammaI * nI
    dnC = R * nR * nC - gammaC * nC + nc_source

where P(t) is a sum of Gaussian pulses defined by ``amps`` and ``centers``.
"""
from __future__ import annotations

import numpy as np


def params_from_physics(physics) -> dict[str, float]:
    """Extract ODE parameter dict from a PhysicsConstants dataclass."""
    return {
        "kappa": physics.kappa,
        "gammaR": physics.gamma_R,
        "gammaI": physics.gamma_I,
        "gammaC": physics.gamma_C,
        "R": physics.R,
    }


def _multipulse_pump(
    t: float,
    amps: np.ndarray,
    centers: np.ndarray,
    sigma_t: float,
) -> float:
    return float(np.sum(amps * np.exp(-0.5 * ((t - centers) / sigma_t) ** 2)))


def _ode_rhs(
    t: float,
    y: list[float],
    amps: np.ndarray,
    centers: np.ndarray,
    sigma_t: float,
    params: dict[str, float],
) -> list[float]:
    nR, nI, nC = y
    P = _multipulse_pump(t, amps, centers, sigma_t)
    dnR = params["kappa"] * nI ** 2 - params["gammaR"] * nR - params["R"] * nR * nC
    dnI = P - params["kappa"] * nI ** 2 - params["gammaI"] * nI
    dnC = params["R"] * nR * nC - params["gammaC"] * nC + params.get("nc_source", 1.0e-6)
    return np.array([dnR, dnI, dnC], dtype=np.float64)


def _rk4_step(
    t: float,
    y: np.ndarray,
    h: float,
    amps: np.ndarray,
    centers: np.ndarray,
    sigma_t: float,
    params: dict[str, float],
) -> np.ndarray:
    k1 = _ode_rhs(t, y, amps, centers, sigma_t, params)
    k2 = _ode_rhs(t + 0.5 * h, y + 0.5 * h * k1, amps, centers, sigma_t, params)
    k3 = _ode_rhs(t + 0.5 * h, y + 0.5 * h * k2, amps, centers, sigma_t, params)
    k4 = _ode_rhs(t + h, y + h * k3, amps, centers, sigma_t, params)
    y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.maximum(y_next, 0.0)


def run_reference_ode(
    amps: np.ndarray,
    centers_ode: np.ndarray,
    sigma_t: float,
    params: dict[str, float],
    t_end: float,
    n_points: int,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the condensate ODE and return (time, nC, pump) arrays.

    Parameters
    ----------
    amps : np.ndarray
        Gaussian pulse amplitudes.
    centers_ode : np.ndarray
        Pulse centre times (ps).
    sigma_t : float
        Temporal sigma shared by all pulses (ps).
    params : dict[str, float]
        Must contain kappa, gammaR, gammaI, gammaC, R. Optional nc_source
        (default 1e-6) sets both the ODE source term and the initial nC.
    t_end : float
        Integration end time (ps).
    n_points : int
        Number of uniformly-spaced evaluation points on [0, t_end].
    rtol, atol : float
        Accepted for config compatibility.  The internal solver is fixed-step
        RK4 with a conservative substep based on the pulse width.

    Returns
    -------
    t : np.ndarray
    nC : np.ndarray
    pump : np.ndarray
    """
    nc0 = params.get("nc_source", 1.0e-6)
    t_eval = np.linspace(0.0, t_end, n_points)
    y = np.array([0.0, 0.0, nc0], dtype=np.float64)
    nC = np.empty_like(t_eval)
    nC[0] = y[2]

    output_dt = t_eval[1] - t_eval[0] if n_points > 1 else t_end
    max_step = max(min(float(sigma_t) / 20.0, float(output_dt) / 4.0, 0.05), 1.0e-4)

    t = 0.0
    for i in range(1, n_points):
        t_target = float(t_eval[i])
        n_substeps = max(1, int(np.ceil((t_target - t) / max_step)))
        h = (t_target - t) / n_substeps
        for _ in range(n_substeps):
            y = _rk4_step(t, y, h, amps, centers_ode, sigma_t, params)
            t += h
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(
                    f"non-finite ODE state at t={t:.6g}: {y.tolist()}"
                )
        nC[i] = y[2]

    pump_trace = np.array([
        _multipulse_pump(t_i, amps, centers_ode, sigma_t) for t_i in t_eval
    ])
    return t_eval, nC, pump_trace
