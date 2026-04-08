"""Two-reservoir model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class DoubleReservoir(AbstractReservoir, ResultProvider):
    """Model coupled active and inactive reservoirs."""
    config: ReservoirParameters
    R: float
    gamma_I: float
    gamma_A: float
    R_IA: float
    R_AI: float
    nI: Union[np.ndarray, cp.ndarray]
    nA: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
        precision: str = "double",
    ):
        """Set up the double reservoir."""
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_I = physics.gamma_I
        self.gamma_A = physics.gamma_A
        self.R_IA = physics.R_IA
        self.R_AI = physics.R_AI
        rdtype = real_dtype(self.xp, precision)
        self.nI = self.xp.zeros((grid.ny, grid.nx), dtype=rdtype)
        self.nA = self.xp.zeros((grid.ny, grid.nx), dtype=rdtype)

    def get_state(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the current state."""
        return (self.nA, self.nI)

    def set_state(
        self, state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]
    ) -> None:
        """Set the current state."""
        self.nA = self.xp.maximum(state[0], 0).astype(self.nA.dtype, copy=False)
        self.nI = self.xp.maximum(state[1], 0).astype(self.nI.dtype, copy=False)

    def get_active_density(
        self,
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Return the active reservoir density."""
        return state[0]

    def get_derivatives(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the reservoir time derivatives."""
        nA, nI = state
        abs_psi2 = self.xp.abs(psi) ** 2

        dnI = Pxy - (self.gamma_I + self.R_IA) * nI + self.R_AI * nA
        dnA = self.R_IA * nI - (self.gamma_A + self.R_AI + self.R * abs_psi2) * nA

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
        return self.nA

    def get_reservoir_densities(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        """Return the active and inactive reservoir densities."""
        return self.nA, self.nI

    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        nodes = []
        if self.config.expose_results:
            nodes.append(
                ResultNode(
                    "nA",
                    compute_fn=lambda **ctx: self.nA,
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
                    compute_fn=lambda **ctx: self.nI,
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
                    compute_fn=lambda **ctx: float(self.nA.max()),
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
                    compute_fn=lambda **ctx: float(self.nI.max()),
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
