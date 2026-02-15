from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider
from polarism.simulation_grid_2D import SimulationGrid2D

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class SingleReservoir(AbstractReservoir, ResultProvider):
    config: ReservoirParameters
    R: float
    gamma_r: float
    D: float
    dx: float
    dy: float
    nR: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_r = physics.gamma_R
        self.D = physics.D 
        self.dx = grid.dx
        self.dy = grid.dy
        self.nR = self.xp.zeros((grid.nx, grid.ny), dtype=self.xp.float32)

    def get_state(self) -> tuple[Union[np.ndarray, cp.ndarray]]:
        return (self.nR,)

    def set_state(self, state: tuple[Union[np.ndarray, cp.ndarray]]) -> None:
        self.nR = self.xp.maximum(state[0], 0)

    def get_active_density(
        self, state: tuple[Union[np.ndarray, cp.ndarray]]
    ) -> Union[np.ndarray, cp.ndarray]:
        return state[0]

    def get_derivatives(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
        state: tuple[Union[np.ndarray, cp.ndarray]],
    ) -> tuple[Union[np.ndarray, cp.ndarray]]:
        nR = state[0]
        abs_psi2 = self.xp.abs(psi) ** 2
        dnR = Pxy - (self.gamma_r + self.R * abs_psi2) * nR
        if self.D != 0.0:
            lap = (
                self.xp.roll(nR, -1, axis=0) - 2 * nR + self.xp.roll(nR, 1, axis=0)
            ) / (self.dx**2) + (
                self.xp.roll(nR, -1, axis=1) - 2 * nR + self.xp.roll(nR, 1, axis=1)
            ) / (
                self.dy**2
            )
            dnR = dnR + self.D * lap
        return (dnR,)

    def step(
        self,
        dt: float,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        s0 = self.get_state()
        (k1,) = self.get_derivatives(psi, Pxy, s0)
        s_mid = (s0[0] + 0.5 * dt * k1,)
        (k2,) = self.get_derivatives(psi, Pxy, s_mid)
        self.set_state((s0[0] + dt * k2,))

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        return self.nR

    def make_result_nodes(self) -> list[ResultNode]:
        nodes = []
        if self.config.expose_results:
            nodes.append(
                ResultNode(
                    "N_R",
                    compute_fn=lambda **ctx: self.nR,
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
                    compute_fn=lambda **ctx: float(self.nR.max()),
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
