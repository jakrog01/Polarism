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
