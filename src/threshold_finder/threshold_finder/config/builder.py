"""Build typed ``Config`` objects for power-sweep simulations."""
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

from threshold_finder.config.loader import make_dataclass


def build_power_config(cfg: dict[str, Any], power: float) -> Config:
    """Build a ``Config`` for one power-sweep point.

    Parameters
    ----------
    cfg : dict
        Full parsed config (output of :func:`~threshold_finder.config.loader.load_config`).
    power : float
        Pump power P for this sweep point; set as both P0 and Pmax on the laser.

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
            P0=power,
            Pmax=power,
            x0=float(laser_raw.get("x0", 0.0)),
            y0=float(laser_raw.get("y0", 0.0)),
            sigma_space=float(laser_raw.get("sigma_space", 5.0)),
            sigma_time=float(laser_raw.get("sigma_time", 1.0)),
            pulse_separation=float(laser_raw.get("pulse_separation", 10.0)),
            cutoff_sigma=float(laser_raw.get("cutoff_sigma", 3.0)),
            delay=float(laser_raw.get("delay", 0.0)),
            n_pulses=int(laser_raw.get("n_pulses", 0)),
            power_definition=str(laser_raw.get("power_definition", "peak_amplitude")),
        ),
        reservoir=make_dataclass(ReservoirParameters, g.get("reservoir", {})),
        solver=SolverParameters(
            total_time=float(solver_cfg.get("total_time", 750.0)),
            dt=float(solver_cfg.get("dt", 0.001)),
            method=str(solver_cfg.get("method", "rk4-cuda")),
            precision=str(solver_cfg.get("precision", "double")),
            laplacian=str(solver_cfg.get("laplacian", "five-point")),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
