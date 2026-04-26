"""Build typed ``Config`` objects for the dot-response-fit pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from dot_response_fit.config.loader import make_dataclass
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


def build_mnist_lasers(
    encoded_events: dict[str, Any],
    sigma_space: float,
    global_cfg: dict[str, Any],
    grid: Any,
    pixel_limit: int | None = None,
) -> list[Any]:
    """Construct one :class:`PulseGaussian` per encoded MNIST pixel.

    Pixels are encoded only in pulse amplitude and delay.  Every pulse is
    applied at the same physical dot position from ``global.laser_defaults``;
    the MNIST pixel coordinates stored in ``encoded_events`` are metadata for
    input inspection, not pump positions.

    Parameters
    ----------
    encoded_events : dict
        Content of ``reference/encoded_events.json``.
    sigma_space : float
        Gaussian dot spatial size (µm) — the fitted parameter.
    global_cfg : dict
        The ``global`` section of config.yaml.
    grid : SimulationGrid2D
        Pre-built simulation grid.
    pixel_limit : int or None
        Use only the first *pixel_limit* encoded pixels (fast fit mode).
        ``None`` means use all.

    Returns
    -------
    list of PulseGaussian
    """
    defaults = global_cfg.get("laser_defaults", {})
    cutoff_sigma = float(defaults.get("cutoff_sigma", 3.0))
    x0 = float(defaults.get("x0", 0.0))
    y0 = float(defaults.get("y0", 0.0))

    sigma_time = float(encoded_events["sigma_time"])
    amplitudes: list[float] = encoded_events["amplitudes"]
    delays: list[float] = encoded_events["delays"]
    separation = float(encoded_events["separation"])

    n = len(amplitudes)
    if pixel_limit is not None:
        n = min(n, pixel_limit)

    lasers: list[Any] = []
    for i in range(n):
        laser_cfg = LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=float(amplitudes[i]),
            Pmax=float(amplitudes[i]),
            x0=x0,
            y0=y0,
            sigma_space=sigma_space,
            sigma_time=sigma_time,
            pulse_separation=separation,
            cutoff_sigma=cutoff_sigma,
            delay=float(delays[i]),
            n_pulses=1,
        )
        lasers.append(PulseGaussian(laser_cfg, grid.X, grid.Y))
    return lasers


def build_scenario_config(
    global_cfg: dict[str, Any],
    scenario: dict[str, Any],
    sigma_space: float,
    sigma_time: float,
    pulse_separation: float,
) -> Config:
    """Build a ``Config`` for a scenario simulation run.

    Parameters
    ----------
    global_cfg : dict
        The ``global`` section of config.yaml.
    scenario : dict
        Single entry from the ``scenarios`` list.
    sigma_space : float
        Gaussian dot spatial size (μm) — the fitted parameter.
    sigma_time : float
        Temporal pulse width (ps).
    pulse_separation : float
        Inter-pulse interval (ps).

    Returns
    -------
    Config
        Fully constructed simulation config.
    """
    g = global_cfg
    defaults = g.get("laser_defaults", {})
    solver_cfg = g.get("solver", {})
    potential_cfg = scenario.get("potential", {"potential_type": "zero"})
    cutoff_sigma: float = float(defaults.get("cutoff_sigma", 3.0))

    laser_defs: list[dict[str, Any]] = scenario["lasers"]
    first_laser = {**defaults, **laser_defs[0]} if laser_defs else {}
    P0: float = float(first_laser.get("P0", 1.0))

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
            P0=P0,
            Pmax=P0,
            x0=float(first_laser.get("x0", 0.0)),
            y0=float(first_laser.get("y0", 0.0)),
            sigma_space=sigma_space,
            sigma_time=sigma_time,
            pulse_separation=pulse_separation,
            cutoff_sigma=cutoff_sigma,
        ),
        reservoir=make_dataclass(ReservoirParameters, g.get("reservoir", {})),
        solver=SolverParameters(
            total_time=float(solver_cfg.get("total_time", 400.0)),
            dt=float(solver_cfg.get("dt", 0.001)),
            method=solver_cfg.get("method", "rk4-cuda"),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def build_scenario_lasers(
    scenario: dict[str, Any],
    global_cfg: dict[str, Any],
    sigma_space: float,
    sigma_time: float,
    pulse_separation: float,
    grid: Any,
) -> list[Any]:
    """Construct laser instances for a scenario.

    Parameters
    ----------
    scenario : dict
        Scenario definition from config.yaml.
    global_cfg : dict
        The ``global`` section.
    sigma_space : float
        Fitted Gaussian dot spatial size (μm).
    sigma_time : float
        Temporal pulse width (ps).
    pulse_separation : float
        Inter-pulse separation (ps).
    grid : SimulationGrid2D
        Pre-built grid.

    Returns
    -------
    list
        List of :class:`polarism.laser.pulse_gaussian.PulseGaussian` instances.
    """
    defaults = global_cfg.get("laser_defaults", {})
    cutoff_sigma: float = float(defaults.get("cutoff_sigma", 3.0))

    lasers: list[Any] = []
    for ldef in scenario["lasers"]:
        merged = {**defaults, **ldef}
        P0 = float(merged.get("P0", merged.get("power", 1.0)))
        laser_cfg = LaserParameters(
            mode="single",
            laser_type=merged.get("laser_type", "pulse-gaussian"),
            P0=P0,
            Pmax=P0,
            x0=float(merged.get("x0", 0.0)),
            y0=float(merged.get("y0", 0.0)),
            sigma_space=sigma_space,
            sigma_time=sigma_time,
            pulse_separation=pulse_separation,
            cutoff_sigma=float(merged.get("cutoff_sigma", cutoff_sigma)),
            delay=float(merged.get("delay", 0.0)),
            n_pulses=int(merged.get("n_pulses", 0)),
        )
        lasers.append(PulseGaussian(laser_cfg, grid.X, grid.Y))

    return lasers
