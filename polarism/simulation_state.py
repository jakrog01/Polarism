from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.grid.simulation_grid_2d import SimulationGrid2D


class SimulationState:
    psi: Union[np.ndarray, cp.ndarray]
    t: float

    def __init__(self, grid: SimulationGrid2D, eps: float = 1e-3):
        xp = compute_engine.xp
        rng = xp.random.default_rng()
        self.psi = (
            eps
            * (
                rng.random((grid.ny, grid.nx), dtype=xp.float64)
                + 1j * rng.random((grid.ny, grid.nx), dtype=xp.float64)
            )
        ).astype(xp.complex128)
        self.t = 0.0

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        return self.psi, self.t
