"""Potential factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registry import available_potentials
from polarism.grid.simulation_grid_2d import SimulationGrid2D

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


def create_potential(
    potential_config: PotentialParameters, grid: SimulationGrid2D
) -> Union[np.ndarray, cp.ndarray]:
    """Create potential."""
    return available_potentials.lookup(potential_config.potential_type)(grid.X, grid.Y, potential_config)
