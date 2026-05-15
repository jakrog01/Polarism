"""Simulation state storage."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.init_condition import make_initial_psi

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.grid.simulation_grid_2d import SimulationGrid2D


class SimulationState:
    """Store the wavefunction and current time."""
    psi: Union[np.ndarray, cp.ndarray]
    t: float

    def __init__(
        self,
        grid: SimulationGrid2D,
        eps: float = 1e-3,
        precision: str = "double",
        *,
        mode: str = "legacy_positive_uniform",
        k_cutoff_um: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Set up the simulation state."""
        xp = compute_engine.xp
        rdtype = real_dtype(xp, precision)
        cdtype = complex_dtype(xp, precision)
        self.psi = make_initial_psi(
            xp, grid.ny, grid.nx, eps,
            mode=mode,
            dx=grid.dx, dy=grid.dy,
            k_cutoff_um=k_cutoff_um,
            seed=seed,
            cdtype=cdtype,
            rdtype=rdtype,
        )
        self.t = 0.0

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        """Return simulation state."""
        return self.psi, self.t
