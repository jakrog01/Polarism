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


class DoubleReservoir(AbstractReservoir, ResultProvider):
    config: ReservoirParameters
    R: float
    gamma_I: float
    gamma_A: float
    R_IA: float
    R_AI: float
    D_I: float
    D_A: float
    dx: float
    dy: float
    nI: Union[np.ndarray, cp.ndarray]
    nA: Union[np.ndarray, cp.ndarray]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_I = physics.gamma_I
        self.gamma_A = physics.gamma_A
        self.R_IA = physics.R_IA
        self.R_AI = physics.R_AI
        self.D_A = physics.D_A
        self.D_I = physics.D_I
        self.dx = grid.dx
        self.dy = grid.dy
        self.nI = self.xp.zeros((grid.ny, grid.nx), dtype=self.xp.float32)
        self.nA = self.xp.zeros((grid.ny, grid.nx), dtype=self.xp.float32)

    def get_state(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        return (self.nA, self.nI)

    def set_state(
        self, state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]
    ) -> None:
        self.nA = self.xp.maximum(state[0], 0)
        self.nI = self.xp.maximum(state[1], 0)

    def get_active_density(
        self,
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> Union[np.ndarray, cp.ndarray]:
        return state[0]

    def get_derivatives(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
        state: tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]],
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        nA, nI = state
        abs_psi2 = self.xp.abs(psi) ** 2

        dnI = Pxy - (self.gamma_I + self.R_IA) * nI + self.R_AI * nA
        dnA = self.R_IA * nI - (self.gamma_A + self.R_AI + self.R * abs_psi2) * nA

        if self.D_I != 0.0:
            lapI = (
                self.xp.roll(nI, -1, axis=0) - 2 * nI + self.xp.roll(nI, 1, axis=0)
            ) / (self.dy**2) + (
                self.xp.roll(nI, -1, axis=1) - 2 * nI + self.xp.roll(nI, 1, axis=1)
            ) / (
                self.dx**2
            )
            dnI = dnI + self.D_I * lapI

        if self.D_A != 0.0:
            lapA = (
                self.xp.roll(nA, -1, axis=0) - 2 * nA + self.xp.roll(nA, 1, axis=0)
            ) / (self.dy**2) + (
                self.xp.roll(nA, -1, axis=1) - 2 * nA + self.xp.roll(nA, 1, axis=1)
            ) / (
                self.dx**2
            )
            dnA = dnA + self.D_A * lapA

        return (dnA, dnI)

    def step(
        self,
        dt: float,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        s0 = self.get_state()
        k1 = self.get_derivatives(psi, Pxy, s0)
        s_mid = (s0[0] + 0.5 * dt * k1[0], s0[1] + 0.5 * dt * k1[1])
        k2 = self.get_derivatives(psi, Pxy, s_mid)
        self.set_state((s0[0] + dt * k2[0], s0[1] + dt * k2[1]))

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        return self.nA

    def get_reservoir_densities(
        self,
    ) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
        return self.nA, self.nI

    def make_result_nodes(self) -> list[ResultNode]:
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
