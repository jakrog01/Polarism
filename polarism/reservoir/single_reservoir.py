"""Single-reservoir model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.batch_physics import BatchPhysics
from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.grid.simulation_grid import SimulationGrid
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.laplacian import real_laplacian
from polarism.reservoir.reservoir_registry import register_reservoir
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


@register_reservoir("single")
class SingleReservoir(AbstractReservoir, ResultProvider):
    """Model a single active reservoir."""
    config: ReservoirParameters
    R: float
    gamma_r: float
    reservoir_diffusion_R: float
    dx: float
    dy: float
    grid_type: str
    nR: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid,
        precision: str = "double",
        batch_physics=None,
    ):
        """Set up the single reservoir."""
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.batch_physics = BatchPhysics(physics, batch_physics, self.xp, precision, ndim=grid.ndim)
        self.R = self.batch_physics.get("R")
        self.gamma_r = self.batch_physics.get("gamma_R")
        self.reservoir_diffusion_R = self.batch_physics.get("reservoir_diffusion_R")
        self.dx = grid.dx
        self.dy = grid.dy if grid.ndim == 2 else None
        self.grid_type = grid.boundary
        self._spatial_shape = grid.shape
        self._batched = False
        self.nR = self.xp.zeros((1, 1, *grid.shape), dtype=real_dtype(self.xp, precision))

    def reset(self, batch_size: int | None = None) -> None:
        """Reset zero density fields, optionally with a leading batch axis."""
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self._batched = batch_size is not None
        shape = (1 if batch_size is None else batch_size, 1, *self._spatial_shape)
        self.nR = self.xp.zeros(shape, dtype=self.nR.dtype)

    def get_state(self) -> tuple[Union[np.ndarray, cp.ndarray]]:
        """Return the current state."""
        return (self.nR,)

    def set_state(self, state: tuple[Union[np.ndarray, cp.ndarray]]) -> None:
        """Set the current state."""
        self._check_state_shape(state, self.nR.shape)
        self.nR = self.xp.maximum(state[0], 0).astype(self.nR.dtype, copy=False)

    def get_active_density(
        self, state: tuple[Union[np.ndarray, cp.ndarray]]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the active reservoir density."""
        return state[0]

    def get_derivatives(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
        state: tuple[Union[np.ndarray, cp.ndarray]],
        rho: Union[np.ndarray, cp.ndarray] | None = None,
    ) -> tuple[Union[np.ndarray, cp.ndarray]]:
        """Return the reservoir time derivatives."""
        self.batch_physics.check(psi)
        self._check_state_shape(state, psi.shape)
        nR = state[0]
        abs_psi2 = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag
        dnR = Pxy - (self.gamma_r + self.R * abs_psi2) * nR
        if self.batch_physics.any_nonzero("reservoir_diffusion_R"):
            dnR = dnR + self.reservoir_diffusion_R * real_laplacian(
                nR, self.xp, self.dx, self.dy, self.grid_type
            )
        return (dnR,)

    def step(
        self,
        dt: float,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        """Advance the reservoir by one time step."""
        s0 = self.get_state()
        (k1,) = self.get_derivatives(psi, Pxy, s0)
        s_mid = (s0[0] + 0.5 * dt * k1,)
        (k2,) = self.get_derivatives(psi, Pxy, s_mid)
        self.set_state((s0[0] + dt * k2,))

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the total reservoir density."""
        return self._public_density(self.nR)

    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        nodes = []
        if self.config.expose_results:
            nodes.append(
                ResultNode(
                    "N_R",
                    compute_fn=lambda **ctx: self._public_density(self.nR),
                    reduce_dim_fn=lambda f: float(f.max()),
                    cmap="viridis",
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=True,
                )
            )
            nodes.append(
                ResultNode(
                    "N_R_max",
                    compute_fn=lambda **ctx: self.nR.max(),
                    reduce_dim_fn=lambda v: float(v),
                    cmap=None,
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=False,
                )
            )
        return nodes
