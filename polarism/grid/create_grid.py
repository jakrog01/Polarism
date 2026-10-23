"""Grid factory helpers."""
from __future__ import annotations

from polarism.grid.closed_interval_grid import (
    ClosedIntervalGrid,
)
from polarism.grid.grid_registry import available_grids
from polarism.grid.periodic_grid import PeriodicGrid


def create_grid(cfg):
    """Create grid."""
    return available_grids.lookup(cfg.grid_type)(cfg)


def grid_boundary(grid_type: str) -> str:
    """Return the boundary semantics of a registered grid type.

    Unregistered names are returned unchanged so callers can report them.
    """
    grid_cls = available_grids.get(grid_type)
    return grid_cls.boundary if grid_cls is not None else grid_type
