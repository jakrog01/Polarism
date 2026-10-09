"""Shared gain and observable condensation criteria.

The reservoir gain criterion is ``R n_R - gamma_C > 0``.  It is necessary,
but an observable condensate additionally requires ``max(|psi|^2)`` to exceed
the selected floor.  The two conditions are deliberately reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


CONDENSATION_PSI_SQ_FLOOR = 5.0e-2


class _Physics(Protocol):
    R: float
    gamma_C: float
    gamma_R: float
    gamma_I: float
    kappa: float
    init_eps: float


@dataclass(frozen=True, slots=True)
class CrossingResult:
    """Diagnostics from Schmitt-triggered gain crossings."""

    n_crossings: int
    crossing_times_ps: tuple[float, ...]
    nR_max: float
    duty_above: float
    first_crossing_ps: float | None
    ratio_to_critical: float = float("nan")


@dataclass(frozen=True, slots=True)
class CondensationVerdict:
    """Combined necessary-gain and observable-condensate verdict."""

    gain_crossings: int
    first_crossing_ps: float | None
    nR_max: float
    ratio_to_critical: float
    psi_sq_max: float
    psi_sq_floor: float
    observed: bool
    klass: Literal["dark", "gain_only", "latched", "spiking"]


def critical_reservoir_density(physics: _Physics) -> float:
    """Return the active-reservoir density where gain equals loss."""
    scattering = float(physics.R)
    if scattering <= 0.0:
        raise ValueError(f"physics.R must be positive, got {scattering}")
    return float(physics.gamma_C) / scattering


def gain_loss_signal(n_active: np.ndarray, physics: _Physics) -> np.ndarray:
    """Return ``R*n_active - gamma_C`` as float64 values."""
    return float(physics.R) * np.asarray(n_active, dtype=np.float64) - float(
        physics.gamma_C
    )


def psi_sq_floor(
    physics: _Physics,
    *,
    mode: Literal["absolute", "seed_relative"] = "absolute",
    decades: float = 4.0,
) -> float:
    """Return the observable condensate-density floor."""
    if mode == "absolute":
        return CONDENSATION_PSI_SQ_FLOOR
    if mode == "seed_relative":
        return float(physics.init_eps) ** 2 * 10.0**float(decades)
    raise ValueError(f"Unsupported psi_sq_floor mode: {mode!r}")


def classify(
    crossings: CrossingResult,
    psi_sq_max: float,
    floor: float,
) -> CondensationVerdict:
    """Classify a point using both gain and observable criteria."""
    if floor <= 0.0:
        raise ValueError(f"psi_sq_floor must be positive, got {floor}")
    observed = float(psi_sq_max) >= float(floor)
    if crossings.n_crossings == 0:
        klass: Literal["dark", "gain_only", "latched", "spiking"] = "dark"
    elif not observed:
        klass = "gain_only"
    elif crossings.n_crossings == 1:
        klass = "latched"
    else:
        klass = "spiking"
    return CondensationVerdict(
        gain_crossings=crossings.n_crossings,
        first_crossing_ps=crossings.first_crossing_ps,
        nR_max=crossings.nR_max,
        ratio_to_critical=crossings.ratio_to_critical,
        psi_sq_max=float(psi_sq_max),
        psi_sq_floor=float(floor),
        observed=observed,
        klass=klass,
    )


def validate_sampling(dt_sample_ps: float, sigma_time_ps: float) -> None:
    """Validate temporal resolution used for gain-crossing detection."""
    if dt_sample_ps <= 0.0 or sigma_time_ps <= 0.0:
        raise ValueError("dt_sample_ps and sigma_time_ps must be positive")
    if dt_sample_ps > sigma_time_ps / 10.0:
        raise ValueError(
            "gain-crossing sampling is too sparse: dt_sample_ps must be at most "
            "sigma_time_ps / 10"
        )


def integrate_zero_dim(
    t: np.ndarray,
    pump: np.ndarray,
    physics: _Physics,
    *,
    model: Literal["pump_only", "coupled"] = "pump_only",
    spontaneous_source: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the zero-dimensional quadratic-double reservoir with RK4."""
    times = np.asarray(t, dtype=np.float64)
    source = np.asarray(pump, dtype=np.float64)
    if times.ndim != 1 or source.shape != times.shape or times.size < 2:
        raise ValueError("t and pump must be matching one-dimensional arrays of length >= 2")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(source)):
        raise ValueError("t and pump must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t must be strictly increasing")
    if model not in {"pump_only", "coupled"}:
        raise ValueError(f"Unsupported threshold model: {model!r}")
    n_r = np.zeros_like(times)
    n_i = np.zeros_like(times)
    n_c = np.full_like(times, float(physics.init_eps) ** 2) if model == "coupled" else None
    kappa = float(physics.kappa)
    gamma_r = float(physics.gamma_R)
    gamma_i = float(physics.gamma_I)
    scattering = float(physics.R)
    gamma_c = float(physics.gamma_C)
    active = 0.0
    inactive = 0.0
    condensate = float(physics.init_eps) ** 2
    for index, step in enumerate(np.diff(times)):
        p0 = float(source[index])
        p1 = float(source[index + 1])
        pm = 0.5 * (p0 + p1)
        if model == "pump_only":
            a1, i1 = _pump_only_rhs(active, inactive, p0, kappa, gamma_r, gamma_i)
            a2, i2 = _pump_only_rhs(
                active + 0.5 * step * a1,
                inactive + 0.5 * step * i1,
                pm,
                kappa,
                gamma_r,
                gamma_i,
            )
            a3, i3 = _pump_only_rhs(
                active + 0.5 * step * a2,
                inactive + 0.5 * step * i2,
                pm,
                kappa,
                gamma_r,
                gamma_i,
            )
            a4, i4 = _pump_only_rhs(
                active + step * a3,
                inactive + step * i3,
                p1,
                kappa,
                gamma_r,
                gamma_i,
            )
            active = max(0.0, active + step * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0)
            inactive = max(0.0, inactive + step * (i1 + 2.0 * i2 + 2.0 * i3 + i4) / 6.0)
        else:
            state = (active, inactive, condensate)
            k1 = _coupled_rhs(*state, p0, kappa, gamma_r, gamma_i, scattering, gamma_c, spontaneous_source)
            k2 = _coupled_rhs(*_rk_state(state, k1, 0.5 * step), pm, kappa, gamma_r, gamma_i, scattering, gamma_c, spontaneous_source)
            k3 = _coupled_rhs(*_rk_state(state, k2, 0.5 * step), pm, kappa, gamma_r, gamma_i, scattering, gamma_c, spontaneous_source)
            k4 = _coupled_rhs(*_rk_state(state, k3, step), p1, kappa, gamma_r, gamma_i, scattering, gamma_c, spontaneous_source)
            active, inactive, condensate = (
                max(0.0, value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0)
                for value, a, b, c, d in zip(state, k1, k2, k3, k4)
            )
        n_r[index + 1] = active
        n_i[index + 1] = inactive
        if n_c is not None:
            n_c[index + 1] = condensate
    return (times, n_r) if n_c is None else (times, n_r, n_c)


def count_upward_crossings(
    t: np.ndarray,
    s: np.ndarray,
    *,
    hysteresis: float,
    min_above_ps: float = 0.0,
) -> CrossingResult:
    """Count Schmitt-triggered upward gain crossings with linear timing."""
    times = np.asarray(t, dtype=np.float64)
    signal = np.asarray(s, dtype=np.float64)
    if times.ndim != 1 or signal.shape != times.shape or times.size == 0:
        raise ValueError("t and s must be matching nonempty one-dimensional arrays")
    if hysteresis < 0.0 or min_above_ps < 0.0:
        raise ValueError("hysteresis and min_above_ps must be nonnegative")
    above_duration = _positive_duration(times, signal)
    if times.size == 1:
        crossings = (float(times[0]),) if signal[0] >= hysteresis else ()
        return CrossingResult(
            len(crossings),
            crossings,
            float("nan"),
            1.0 if signal[0] > 0.0 else 0.0,
            crossings[0] if crossings else None,
        )
    armed = True
    candidates: list[float] = []
    if signal[0] >= hysteresis:
        candidates.append(float(times[0]))
        armed = False
    for index in range(1, times.size):
        left = signal[index - 1]
        right = signal[index]
        if armed and left < hysteresis <= right:
            candidates.append(
                _interpolate_time(times[index - 1], times[index], left, right, hysteresis)
            )
            armed = False
        elif not armed and right <= -hysteresis:
            armed = True
    crossings = tuple(
        candidate
        for candidate in candidates
        if _above_run_duration(times, signal, candidate) >= min_above_ps
    )
    duration = max(float(times[-1] - times[0]), 0.0)
    duty = above_duration / duration if duration else float(signal[0] > 0.0)
    return CrossingResult(
        len(crossings),
        crossings,
        float("nan"),
        duty,
        crossings[0] if crossings else None,
    )


def _pump_only_rhs(
    active: float,
    inactive: float,
    pump: float,
    kappa: float,
    gamma_r: float,
    gamma_i: float,
) -> tuple[float, float]:
    transfer = kappa * inactive * inactive
    return transfer - gamma_r * active, pump - transfer - gamma_i * inactive


def _coupled_rhs(
    active: float,
    inactive: float,
    condensate: float,
    pump: float,
    kappa: float,
    gamma_r: float,
    gamma_i: float,
    scattering: float,
    gamma_c: float,
    spontaneous_source: float,
) -> tuple[float, float, float]:
    transfer = kappa * inactive * inactive
    return (
        transfer - gamma_r * active - scattering * active * condensate,
        pump - transfer - gamma_i * inactive,
        (scattering * active - gamma_c) * condensate + spontaneous_source,
    )


def _rk_state(
    state: tuple[float, float, float],
    derivative: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    return tuple(value + scale * delta for value, delta in zip(state, derivative))


def _interpolate_time(t0: float, t1: float, y0: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def _positive_duration(t: np.ndarray, s: np.ndarray) -> float:
    total = 0.0
    for t0, t1, s0, s1 in zip(t[:-1], t[1:], s[:-1], s[1:]):
        step = float(t1 - t0)
        if s0 > 0.0 and s1 > 0.0:
            total += step
        elif s0 > 0.0 >= s1:
            total += _interpolate_time(float(t0), float(t1), float(s0), float(s1), 0.0) - float(t0)
        elif s0 <= 0.0 < s1:
            total += float(t1) - _interpolate_time(float(t0), float(t1), float(s0), float(s1), 0.0)
    return total


def _above_run_duration(t: np.ndarray, s: np.ndarray, crossing: float) -> float:
    index = int(np.searchsorted(t, crossing, side="right"))
    end = float(t[-1])
    for current in range(max(index, 1), t.size):
        if s[current - 1] > 0.0 >= s[current]:
            end = _interpolate_time(
                float(t[current - 1]),
                float(t[current]),
                float(s[current - 1]),
                float(s[current]),
                0.0,
            )
            break
    return max(0.0, end - crossing)
