"""Build typed ``Config`` objects for pipeline runs.

This module turns pipeline config data and threshold-search results into
the dataclasses used by the simulation code.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from pipeline.config.loader import (
    build_timing_namespace,
    get_laser_defaults,
    make_dataclass,
    resolve_delay,
    resolve_power,
)
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    ComputeEngineParameters,
    Config,
    GridParameters,
    LaserParameters,
    PhysicsConstants,
    PotentialParameters,
    ReservoirParameters,
    ResultParameters,
    SolverParameters,
)
from polarism.laser.pulse_gaussian import PulseGaussian


def build_scenario_config(
    global_cfg: dict[str, Any],
    threshold: dict[str, Any],
    scenario: dict[str, Any],
) -> Config:
    """Build a ``Config`` for a scenario simulation run.

    Parameters
    ----------
    global_cfg : dict
        The ``global`` section of config.yaml.
    threshold : dict
        Content of ``threshold_result.json`` from a completed threshold search.
    scenario : dict
        Single entry from config.yaml ``scenarios`` list.

    Returns
    -------
    Config
        Fully constructed simulation config ready for the kernel.
    """
    g = global_cfg
    defaults = g.get("laser_defaults", {})
    solver_cfg = g.get("solver", {})
    potential_cfg = scenario.get("potential", {"potential_type": "zero"})

    sigma_space: float = defaults.get("sigma_space", threshold.get("sigma_space", 5.0))
    sigma_time: float = threshold.get("sigma_time", 1.0)
    pulse_sep: float = threshold.get("pulse_separation", 10.0)
    cutoff_sigma: float = defaults.get(
        "cutoff_sigma", threshold.get("cutoff_sigma", 3.0)
    )

    return Config(
        grid=make_dataclass(GridParameters, g.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, g.get("boundary_condition", {})
        ),
        potential=make_dataclass(PotentialParameters, potential_cfg),
        physics=make_dataclass(PhysicsConstants, g.get("physics", {})),
        laser=LaserParameters(
            mode="single",
            laser_type=defaults.get("laser_type", "pulse-gaussian"),
            P0=threshold["P_threshold"],
            Pmax=threshold["P_threshold"],
            x0=0.0,
            y0=0.0,
            sigma_space=sigma_space,
            sigma_time=sigma_time,
            pulse_separation=pulse_sep,
            cutoff_sigma=cutoff_sigma,
        ),
        reservoir=make_dataclass(ReservoirParameters, g.get("reservoir", {})),
        solver=SolverParameters(
            total_time=solver_cfg.get("total_time", 500.0),
            dt=solver_cfg.get("dt", 0.001),
            method=solver_cfg.get("method", "rk4-cuda"),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def _assign_laser_ids(laser_defs: list[dict[str, Any]]) -> list[str]:
    return [str(ldef.get("id", f"laser_{i}")) for i, ldef in enumerate(laser_defs)]


def _apply_power_modifiers(
    base_power: float,
    laser_id: str,
    tags: list[str],
    modifiers: list[dict[str, Any]],
    p_th: float,
) -> float:
    for mod in modifiers:
        if laser_id in mod.get("ids", []) or any(t in tags for t in mod.get("tags", [])):
            return resolve_power(mod.get("power"), p_th)
    return base_power


def build_scenario_lasers(
    scenario: dict[str, Any],
    global_cfg: dict[str, Any],
    threshold: dict[str, Any],
    grid: Any,
    rng: np.random.Generator,
) -> tuple[list[Any], list[float]]:
    """Construct laser instances for a scenario.

    Parameters
    ----------
    scenario : dict
        Scenario definition from config.yaml.
    global_cfg : dict
        The ``global`` section of config.yaml.
    threshold : dict
        Threshold search result (``threshold_result.json``).
    grid : SimulationGrid2D
        Pre-built grid (caller constructs it once via ``create_grid``).
    rng : np.random.Generator
        Random-number generator for multi-pump phase offsets (legacy path
        only, used when no laser in the scenario has a ``delay`` field and
        no ``timing_vars`` block is present).

    Returns
    -------
    (lasers, phases)
        List of ``PulseGaussian`` instances and phase offsets (radians).
        When explicit delays are used, phases are all zero.

    Notes
    -----
    **Explicit delay path** (new default):
    If the scenario contains a ``timing_vars`` block or any laser defines a
    ``delay`` field, all per-laser delays are resolved from those expressions.
    A laser without a ``delay`` field defaults to ``delay = 0.0``.

    **Legacy random-phase path** (backward compatible):
    If neither ``timing_vars`` nor any ``delay`` field is present, each laser
    in a multi-laser scenario receives a random phase offset drawn from
    ``Uniform(0, 2π)`` and converted to a time offset within one pulse period.

    **Delay expressions** are evaluated against a per-laser namespace.
    ``timing_vars`` are resolved once for the scenario (using threshold/global
    defaults) and remain fixed.  The three base names — ``sigma_time``,
    ``pulse_separation``, ``cutoff_sigma`` — are then overridden with the
    laser's own merged values, so per-laser overrides of those parameters are
    reflected in the delay expression.
    """
    defaults = get_laser_defaults({"global": global_cfg})
    p_th: float = threshold["P_threshold"]
    th_sigma_time: float = threshold.get("sigma_time", 1.0)
    th_pulse_sep: float = threshold.get("pulse_separation", 10.0)

    laser_defs: list[dict[str, Any]] = scenario["lasers"]
    n_lasers = len(laser_defs)
    power_modifiers: list[dict[str, Any]] = scenario.get("power_modifiers", [])
    ids = _assign_laser_ids(laser_defs)

    has_explicit_delays = scenario.get("timing_vars") is not None or any(
        "delay" in ldef for ldef in laser_defs
    )

    if has_explicit_delays:
        timing_ns = build_timing_namespace(
            threshold, defaults, scenario.get("timing_vars")
        )
        phases_out: list[float] = [0.0] * n_lasers
    else:
        raw_phases: np.ndarray = (
            rng.uniform(0, 2 * np.pi, size=n_lasers) if n_lasers > 1 else np.zeros(1)
        )
        phases_out = raw_phases.tolist()
        timing_ns = None

    lasers: list[Any] = []
    for i, ldef in enumerate(laser_defs):
        merged = {**defaults, **ldef}
        base_power = resolve_power(merged.get("power"), p_th)
        tags: list[str] = ldef.get("tags") or []
        power = _apply_power_modifiers(base_power, ids[i], tags, power_modifiers, p_th)
        sigma_time = float(merged.get("sigma_time", th_sigma_time))
        raw_sep = merged.get("pulse_separation", None)
        if raw_sep is None:
            pulse_sep = th_pulse_sep
        elif isinstance(raw_sep, str) and timing_ns is not None:
            pulse_sep = resolve_delay(raw_sep, timing_ns)
        else:
            pulse_sep = float(raw_sep)

        if timing_ns is not None:
            per_laser_ns = {
                **timing_ns,
                "sigma_time": sigma_time,
                "pulse_separation": pulse_sep,
                "cutoff_sigma": float(merged.get("cutoff_sigma", 3.0)),
            }
            delay = resolve_delay(ldef.get("delay"), per_laser_ns)
            if not math.isfinite(delay) or delay < 0:
                raise ValueError(
                    f"Laser '{ids[i]}' resolved delay={delay!r} must be "
                    "finite and non-negative"
                )
        else:
            delay = (
                float(phases_out[i] / (2 * np.pi) * pulse_sep) if n_lasers > 1 else 0.0
            )

        laser_cfg = LaserParameters(
            mode="single",
            laser_type=merged.get("laser_type", "pulse-gaussian"),
            P0=power,
            Pmax=power,
            x0=float(merged.get("x0", 0.0)),
            y0=float(merged.get("y0", 0.0)),
            sigma_space=float(merged.get("sigma_space", 5.0)),
            sigma_time=sigma_time,
            pulse_separation=pulse_sep,
            cutoff_sigma=float(merged.get("cutoff_sigma", 3.0)),
            delay=delay,
            n_pulses=int(merged.get("n_pulses", 0)),
        )
        lasers.append(PulseGaussian(laser_cfg, grid.X, grid.Y))

    return lasers, phases_out
