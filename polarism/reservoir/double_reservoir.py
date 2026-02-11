from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider
from polarism.simulation_grid_2D import SimulationGrid2D
from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp


class DoubleReservoir(AbstractReservoir, ResultProvider):
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
    ):
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_I = physics.gamma_I
        self.gamma_A = physics.gamma_A
        self.R_IA = physics.R_IA
        self.R_AI = physics.R_AI
        self.nI = self.xp.zeros((grid.nx, grid.ny))
        self.nA = self.xp.zeros((grid.nx, grid.ny))

    def step(self, dt: float, psi: Union[np.ndarray, cp.ndarray], Pxy: Union[np.ndarray, cp.ndarray]) -> None:
        abs_psi2 = self.xp.abs(psi) ** 2
        self.nI += dt * (
            Pxy - self.gamma_I * self.nI - self.R_IA * self.nI + self.R_AI * self.nA
        )
        self.nA += dt * (
            self.R_IA * self.nI
            - self.gamma_A * self.nA
            - self.R_AI * self.nA
            - self.R * abs_psi2 * self.nA
        )

    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        return self.nA

    def get_reservoir_densities(self) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
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
