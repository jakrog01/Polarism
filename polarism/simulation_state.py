from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from polarism.simulation_grid_2D import SimulationGrid2D

rng = np.random.default_rng()


class SimulationState:
    psi: np.ndarray
    t: float

    def __init__(self, grid: SimulationGrid2D):
        self.psi = 1e-3 * (
            rng.random((grid.nx, grid.ny)) + 1j * rng.random((grid.nx, grid.ny))
        )
        self.t = 0.0

    def get_simulation_state(self) -> tuple[np.ndarray, float]:
        return self.psi, self.t
