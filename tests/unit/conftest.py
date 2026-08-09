"""Shared unit-tier constructors."""
from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config, GridParameters
from polarism.grid.create_grid import create_grid


def small_config(**overrides) -> Config:
    """Return a real, inexpensive simulation configuration."""
    cfg = Config()
    cfg.grid = GridParameters(nx=16, ny=16, lx=20.0, ly=20.0)
    cfg.physics.init_mode = "complex_gaussian_zero_mean"
    cfg.physics.init_seed = 0
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.reservoir.reservoir_type = "single"
    cfg.solver.dt = 0.01
    cfg.solver.total_time = 0.1
    for path, value in overrides.items():
        target, attr = path.rsplit("__", 1) if "__" in path else ("", path)
        obj = cfg
        for name in filter(None, target.split("__")):
            obj = getattr(obj, name)
        setattr(obj, attr, value)
    return cfg


def grid(nx: int = 16, ny: int = 16, grid_type: str = "periodic"):
    """Build a NumPy-backed grid."""
    compute_engine.xp = np
    return create_grid(GridParameters(nx=nx, ny=ny, lx=20.0, ly=20.0, grid_type=grid_type))
