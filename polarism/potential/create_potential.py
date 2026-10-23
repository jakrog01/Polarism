"""Potential factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registry import available_potentials
from polarism.grid.simulation_grid import SimulationGrid

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


def create_potential(
    potential_config: PotentialParameters, grid: SimulationGrid
) -> Union[np.ndarray, cp.ndarray]:
    """Create potential."""
    from polarism.compute_engine import compute_engine
    Y = compute_engine.xp.zeros_like(grid.X) if grid.ndim == 1 else grid.Y
    return available_potentials.lookup(potential_config.potential_type)(grid.X, Y, potential_config)
