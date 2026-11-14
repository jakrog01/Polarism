"""Grid registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.grid.simulation_grid import SimulationGrid

GRID_BOUNDARIES = {"periodic", "closed-interval"}

available_grids: Registry[type[SimulationGrid]] = Registry("grid")


def register_grid(name: str) -> Callable[[type[SimulationGrid]], type[SimulationGrid]]:
    """Return a decorator that registers a grid under ``name``."""
    def decorator(cls: type[SimulationGrid]) -> type[SimulationGrid]:
        boundary = getattr(cls, "boundary", None)
        if boundary not in GRID_BOUNDARIES:
            raise ValueError(
                f"Grid '{name}' must define a class attribute 'boundary' set to one of "
                f"{sorted(GRID_BOUNDARIES)}, got {boundary!r}."
            )
        return available_grids.register(name)(cls)

    return decorator
