from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.results.result_node import ResultNode
from polarism.simulation_grid_2D import SimulationGrid2D


class AbstractReservoir(ABC):
    @abstractmethod
    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        pass

    @abstractmethod
    def step(self, dt: float, psi: np.ndarray, Pxy: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_reservoir_density(self) -> np.ndarray:
        pass

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        pass
