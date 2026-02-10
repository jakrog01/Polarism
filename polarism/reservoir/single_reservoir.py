from __future__ import annotations

import numpy as np

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider
from polarism.simulation_grid_2D import SimulationGrid2D


class SingleReservoir(AbstractReservoir, ResultProvider):
    config: ReservoirParameters
    R: float
    gamma_r: float
    nR: np.ndarray

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_r = physics.gamma_R
        self.nR = np.zeros((grid.nx, grid.ny))

    def step(self, dt: float, psi: np.ndarray, Pxy: np.ndarray) -> None:
        Gamma = self.R * np.abs(psi) ** 2 + self.gamma_r
        self.nR = self.nR * np.exp(-Gamma * dt) + (Pxy / Gamma) * (
            1 - np.exp(-Gamma * dt)
        )

    def get_reservoir_density(self) -> np.ndarray:
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
