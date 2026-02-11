from __future__ import annotations

from typing import TYPE_CHECKING, Union

from abc import ABC, abstractmethod

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.results.result_node import ResultNode
from polarism.simulation_grid_2D import SimulationGrid2D

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp

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
    def step(self, dt: float, psi: Union[np.ndarray, cp.ndarray], Pxy: Union[np.ndarray, cp.ndarray]) -> None:
        pass

    @abstractmethod
    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        pass

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        pass
