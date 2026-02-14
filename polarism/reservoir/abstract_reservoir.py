from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.results.result_node import ResultNode
from polarism.simulation_grid_2D import SimulationGrid2D

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
        psi: Union[np.ndarray, cp.ndarray],
        Pxy: Union[np.ndarray, cp.ndarray],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        raise NotImplementedError

    @abstractmethod
    def step_frozen_psi(self, dt, psi, pump) -> None:
        self.step(dt, psi, pump)

    @abstractmethod
    def get_state(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_derivatives(self, psi, pump, state: tuple) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def get_active_density(self, state: tuple):
        return state[0]
