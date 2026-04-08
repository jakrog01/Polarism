"""Simulation state storage."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.grid.simulation_grid_2d import SimulationGrid2D


class SimulationState:
    """Store the wavefunction and current time."""
    psi: Union[np.ndarray, cp.ndarray]
    t: float

    def __init__(self, grid: SimulationGrid2D, eps: float = 1e-3, precision: str = "double"):
        """Set up the simulation state."""
        xp = compute_engine.xp
        rdtype = real_dtype(xp, precision)
        cdtype = complex_dtype(xp, precision)
        rng = xp.random.default_rng()
        self.psi = (
            eps
            * (
                rng.random((grid.ny, grid.nx), dtype=rdtype)
                + 1j * rng.random((grid.ny, grid.nx), dtype=rdtype)
            )
        ).astype(cdtype)
        self.t = 0.0

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        """Return simulation state."""
        return self.psi, self.t
