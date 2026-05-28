"""Build typed ``Config`` objects for 2D characteristic map simulations."""
from __future__ import annotations

from typing import Any

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

from create_characteristic.config.loader import make_dataclass


def build_point_config(
    cfg: dict[str, Any],
    pulse_energy: float,
    pulse_separation: float,
    total_time: float,
) -> Config:
    """Build a ``Config`` for one 2D grid point.

    Parameters
    ----------
    cfg : dict
        Full parsed config dict.
    pulse_energy : float
        Pulse energy for this grid point.
    pulse_separation : float
        Pulse separation in ps for this grid point.
    total_time : float
        Simulation duration in ps (pre-computed adaptive value).

    Returns
    -------
    Config
        Fully constructed simulation config ready for the kernel.
    """
    g = cfg["global"]
    laser_raw = cfg.get("laser", {})
    solver_cfg = g.get("solver", {})

    return Config(
        grid=make_dataclass(GridParameters, g.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, g.get("boundary_condition", {})
        ),
        potential=make_dataclass(PotentialParameters, g.get("potential", {})),
        physics=make_dataclass(PhysicsConstants, g.get("physics", {})),
        laser=LaserParameters(
            mode="single",
            laser_type=laser_raw.get("laser_type", "pulse-gaussian"),
            P0=pulse_energy,
            Pmax=pulse_energy,
            x0=float(laser_raw.get("x0", 0.0)),
            y0=float(laser_raw.get("y0", 0.0)),
            sigma_space=float(laser_raw.get("sigma_space", 5.0)),
            sigma_time=float(laser_raw.get("sigma_time", 1.0)),
            pulse_separation=pulse_separation,
            cutoff_sigma=float(laser_raw.get("cutoff_sigma", 3.0)),
            delay=float(laser_raw.get("delay", 0.0)),
            n_pulses=int(laser_raw.get("n_pulses", 0)),
            power_definition=str(laser_raw.get("power_definition", "peak_amplitude")),
        ),
        reservoir=make_dataclass(ReservoirParameters, g.get("reservoir", {})),
        solver=SolverParameters(
            total_time=total_time,
            dt=float(solver_cfg.get("dt", 0.01)),
            method=str(solver_cfg.get("method", "ifrk4-fft-cuda")),
            precision=str(solver_cfg.get("precision", "double")),
            laplacian=str(solver_cfg.get("laplacian", "five-point")),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
