from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.results.result_node import ResultNode

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


class AbstractReservoir(ABC):
    @abstractmethod
    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        dt: float,
        psi_mid: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_derivatives(self, psi, pump, state: tuple) -> tuple:
        raise NotImplementedError

    def get_active_density(self, state: tuple):
        return state[0]
