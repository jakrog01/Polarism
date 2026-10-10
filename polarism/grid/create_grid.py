"""Grid factory helpers."""
from __future__ import annotations

from polarism.grid.closed_interval_simulation_grid_2d import (
    ClosedIntervalSimulationGrid2D,
)
from polarism.grid.grid_registry import available_grids
from polarism.grid.periodic_simulation_grid_2d import PeriodicSimulationGrid2D


def create_grid(cfg):
    """Create grid."""
    if cfg.grid_type not in available_grids:
        raise ValueError(
            f"Unknown grid_type: '{cfg.grid_type}'. "
            f"Available: {list(available_grids.keys())}"
        )
    return available_grids[cfg.grid_type](cfg)


def grid_boundary(grid_type: str) -> str:
    """Return the boundary semantics of a registered grid type.

    Unregistered names are returned unchanged so callers can report them.
    """
    grid_cls = available_grids.get(grid_type)
    return grid_cls.boundary if grid_cls is not None else grid_type
