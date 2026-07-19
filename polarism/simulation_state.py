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
    """Store the wavefunction and current time.

    At most one of ``_psi`` and ``psi_k`` is authoritative between step
    boundaries. Both may coexist after materialization and until the next
    mutation.
    """
    _psi: Union[np.ndarray, cp.ndarray] | None
    psi_k: Union[np.ndarray, cp.ndarray] | None
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
        self._psi = make_initial_psi(
            xp, grid.ny, grid.nx, eps,
            mode=mode,
            dx=grid.dx, dy=grid.dy,
            k_cutoff_um=k_cutoff_um,
            seed=seed,
            cdtype=cdtype,
            rdtype=rdtype,
        )
        self.psi_k = None
        self.t = 0.0

    @property
    def psi(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the real-space wavefunction."""
        if self._psi is None:
            self.materialize_psi(compute_engine.xp)
        if self._psi is None:
            raise ValueError("SimulationState.psi is unavailable")
        return self._psi

    @psi.setter
    def psi(self, value: Union[np.ndarray, cp.ndarray] | None) -> None:
        self._psi = value

    def materialize_psi(self, xp) -> None:
        """Materialize real-space psi from cached spectral state."""
        if self._psi is None:
            psi_k = getattr(self, "psi_k", None)
            if psi_k is None:
                raise ValueError("Cannot materialize psi without psi_k")
            self._psi = xp.fft.ifft2(psi_k, axes=(-2, -1))

    def invalidate_psi_k(self) -> None:
        """Clear cached spectral state after external real-space mutation."""
        self.psi_k = None

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        """Return simulation state."""
        return self.psi, self.t
