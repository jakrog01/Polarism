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
        self.nR = self.xp.zeros((grid.nx, grid.ny))

    def step(
        self,
        dt: float,
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        Gamma = self.R * self.xp.abs(psi) ** 2 + self.gamma_r
        self.nR = self.nR * self.xp.exp(-Gamma * dt) + (Pxy / Gamma) * (
            1 - self.xp.exp(-Gamma * dt)
        )

    def step_frozen_psi(self, dt, psi, pump) -> None:
        self.step(dt, psi, pump)

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        return self.nR

    def get_state(self) -> tuple:
        return (self.nR,)

    def set_state(self, state: tuple) -> None:
        (self.nR,) = state

    def get_active_density(self, state: tuple):
        (nR,) = state
        return nR

    def get_derivatives(self, psi, pump, state: tuple) -> tuple:
        (nR,) = state
        abs_psi2 = self.xp.abs(psi) ** 2
        dnR = pump - (self.gamma_r + self.R * abs_psi2) * nR
        return (dnR,)

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
