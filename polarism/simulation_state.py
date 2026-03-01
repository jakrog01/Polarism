from __future__ import annotations

from typing import TYPE_CHECKING, Union
from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    from polarism.simulation_grid_2D import SimulationGrid2D
    import cupy as cp
    import numpy as np

class SimulationState:
    psi: Union[np.ndarray, cp.ndarray]
    t: float

    def __init__(self, grid: SimulationGrid2D):
        xp = compute_engine.xp
        rng = xp.random.default_rng()
        self.psi = 1e-7 * (
            rng.random((grid.ny, grid.nx)) + 1j * rng.random((grid.ny, grid.nx))
        )
        self.t = 0.0

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        return self.psi, self.t
