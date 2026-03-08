from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registy import available_potentials
from polarism.grid.simulation_grid_2d import SimulationGrid2D

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


def create_potential(
    potential_config: PotentialParameters, grid: SimulationGrid2D
) -> Union[np.ndarray, cp.ndarray]:
    if potential_config.potential_type not in available_potentials:
        raise ValueError(
            f"Unknown potential: '{potential_config.potential_type}'. "
            f"Available: {list(available_potentials.keys())}"
        )

    return available_potentials[potential_config.potential_type](grid.X, grid.Y, potential_config)
