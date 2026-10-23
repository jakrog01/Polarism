"""Two-reservoir model."""
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


@register_reservoir("double")
class DoubleReservoir(AbstractReservoir, ResultProvider):
    """Model coupled active and inactive reservoirs."""
    config: ReservoirParameters
    R: float
    gamma_I: float
    gamma_A: float
    R_IA: float
    R_AI: float
    reservoir_diffusion_I: float
    reservoir_diffusion_A: float
    dx: float
    dy: float
    grid_type: str
    nI: Union[np.ndarray, cp.ndarray]
    nA: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid,
        precision: str = "double",
        batch_physics=None,
    ):
        """Set up the double reservoir."""
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.batch_physics = BatchPhysics(physics, batch_physics, self.xp, precision, ndim=grid.ndim)
        self.R = self.batch_physics.get("R")
        self.gamma_I = self.batch_physics.get("gamma_I")
        self.gamma_A = self.batch_physics.get("gamma_A")
        self.R_IA = self.batch_physics.get("R_IA")
        self.R_AI = self.batch_physics.get("R_AI")
        self.reservoir_diffusion_I = self.batch_physics.get("reservoir_diffusion_I")
        self.reservoir_diffusion_A = self.batch_physics.get("reservoir_diffusion_A")
        self.dx = grid.dx
        self.dy = grid.dy if grid.ndim == 2 else None
        self.grid_type = grid.boundary
        self._spatial_shape = grid.shape
        self._batched = False
        rdtype = real_dtype(self.xp, precision)
        self.nI = self.xp.zeros((1, 1, *grid.shape), dtype=rdtype)
        self.nA = self.xp.zeros((1, 1, *grid.shape), dtype=rdtype)

    def reset(self, batch_size: int | None = None) -> None:
        """Reset zero density fields, optionally with a leading batch axis."""
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self._batched = batch_size is not None
        shape = (1 if batch_size is None else batch_size, 1, *self._spatial_shape)
        self.nA = self.xp.zeros(shape, dtype=self.nA.dtype)
        self.nI = self.xp.zeros(shape, dtype=self.nI.dtype)

    def get_state(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the current state."""
        return (self.nA, self.nI)

    def set_state(
        self, state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]
    ) -> None:
        """Set the current state."""
        self._check_state_shape(state, self.nA.shape)
        self.nA = self.xp.maximum(state[0], 0).astype(self.nA.dtype, copy=False)
        self.nI = self.xp.maximum(state[1], 0).astype(self.nI.dtype, copy=False)

    def get_active_density(
        self,
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the active reservoir density."""
        return state[0]

    def get_inactive_density(
        self,
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the inactive reservoir density."""
        return state[1]

    def get_derivatives(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
        rho: Union[np.ndarray, cp.ndarray] | None = None,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the reservoir time derivatives."""
        self.batch_physics.check(psi)
        self._check_state_shape(state, psi.shape)
        nA, nI = state
        abs_psi2 = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag

        dnI = Pxy - (self.gamma_I + self.R_IA) * nI + self.R_AI * nA
        dnA = self.R_IA * nI - (self.gamma_A + self.R_AI + self.R * abs_psi2) * nA
        if self.batch_physics.any_nonzero("reservoir_diffusion_I"):
            dnI = dnI + self.reservoir_diffusion_I * real_laplacian(
                nI, self.xp, self.dx, self.dy, self.grid_type
            )
        if self.batch_physics.any_nonzero("reservoir_diffusion_A"):
            dnA = dnA + self.reservoir_diffusion_A * real_laplacian(
                nA, self.xp, self.dx, self.dy, self.grid_type
            )

        return (dnA, dnI)

    def step(
        self,
        dt: float,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        """Advance the reservoir by one time step."""
        s0 = self.get_state()
        k1 = self.get_derivatives(psi, Pxy, s0)
        s_mid = (s0[0] + 0.5 * dt * k1[0], s0[1] + 0.5 * dt * k1[1])
        k2 = self.get_derivatives(psi, Pxy, s_mid)
        self.set_state((s0[0] + dt * k2[0], s0[1] + dt * k2[1]))

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the total reservoir density."""
        return self._public_density(self.nA)

    def get_reservoir_densities(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the active and inactive reservoir densities."""
        return self._public_density(self.nA), self._public_density(self.nI)

    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        nodes = []
        if self.config.expose_results:
            nodes.append(
                ResultNode(
                    "nA",
                    compute_fn=lambda **ctx: self._public_density(self.nA),
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
                    "nI",
                    compute_fn=lambda **ctx: self._public_density(self.nI),
                    reduce_dim_fn=lambda f: float(f.max()),
                    cmap="plasma",
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
                    "nA_max",
                    compute_fn=lambda **ctx: self.nA.max(),
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
            nodes.append(
                ResultNode(
                    "nI_max",
                    compute_fn=lambda **ctx: self.nI.max(),
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
