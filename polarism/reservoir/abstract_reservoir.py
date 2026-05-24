"""Reservoir interfaces."""
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
    """Define the interface for reservoir models."""
    @abstractmethod
    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
    ):
        """Set up the abstract reservoir."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        dt: float,
        psi_mid: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
    ) -> None:
        """Advance the reservoir by one time step."""
        raise NotImplementedError

    @abstractmethod
    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the total reservoir density."""
        raise NotImplementedError

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> tuple:
        """Return the current state."""
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: tuple) -> None:
        """Set the current state."""
        raise NotImplementedError

    @abstractmethod
    def get_derivatives(self, psi, pump, state: tuple) -> tuple:
        """Return the reservoir time derivatives."""
        raise NotImplementedError

    def get_active_density(self, state: tuple):
        """Return the active reservoir density."""
        return state[0]

    def get_inactive_density(self, state: tuple):
        """Return the inactive reservoir density, or zero if the model has none."""
        return state[0] * 0.0
