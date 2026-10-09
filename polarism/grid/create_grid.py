"""Grid factory helpers."""
from __future__ import annotations

from polarism.grid.closed_interval_simulation_grid_2d import (
    ClosedIntervalSimulationGrid2D,
)
from polarism.grid.periodic_simulation_grid_2d import PeriodicSimulationGrid2D


def create_grid(cfg):
    """Create grid."""
    if cfg.grid_type == "periodic":
        return PeriodicSimulationGrid2D(cfg)

    if cfg.grid_type == "closed-interval":
        return ClosedIntervalSimulationGrid2D(cfg)

    raise ValueError(f"Unknown grid_type: {cfg.grid_type}")
