"""Build typed ``Config`` objects for pipeline runs.

This module turns pipeline config data and threshold-search results into
the dataclasses used by the simulation code.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.config.loader import get_laser_defaults, make_dataclass, resolve_power
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
from polarism.grid.create_grid import create_grid
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
        Random-number generator for multi-pump phase offsets.

    Returns
    -------
    (lasers, phases)
        List of ``PulseGaussian`` instances and phase offsets (radians).
    """
    defaults = get_laser_defaults({"global": global_cfg})
    p_th: float = threshold["P_threshold"]
    th_sigma_time: float = threshold.get("sigma_time", 1.0)
    th_pulse_sep: float = threshold.get("pulse_separation", 10.0)

    laser_defs: list[dict[str, Any]] = scenario["lasers"]
    n_lasers = len(laser_defs)
    phases: np.ndarray = (
        rng.uniform(0, 2 * np.pi, size=n_lasers) if n_lasers > 1 else np.zeros(1)
    )

    lasers: list[Any] = []
    for i, ldef in enumerate(laser_defs):
        merged = {**defaults, **ldef}
        power = resolve_power(merged.get("power"), p_th)
        sigma_time: float = merged.get("sigma_time", th_sigma_time)
        pulse_sep: float = merged.get("pulse_separation", th_pulse_sep)
        t_offset = (
            float(phases[i] / (2 * np.pi) * pulse_sep) if n_lasers > 1 else 0.0
        )
        laser_cfg = LaserParameters(
            mode="single",
            laser_type=merged.get("laser_type", "pulse-gaussian"),
            P0=power,
            Pmax=power,
            x0=merged.get("x0", 0.0),
            y0=merged.get("y0", 0.0),
            sigma_space=merged.get("sigma_space", 5.0),
            sigma_time=sigma_time,
            pulse_separation=pulse_sep,
            cutoff_sigma=merged.get("cutoff_sigma", 3.0),
            delay=t_offset,
        )
        lasers.append(PulseGaussian(laser_cfg, grid.X, grid.Y))

    return lasers, phases.tolist()
