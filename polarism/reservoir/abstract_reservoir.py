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

    def _public_density(self, field):
        """Return a public-shape view of a canonical reservoir field."""
        return field[:, 0] if self._batched else field[0, 0]

    def _check_state_shape(self, state, shape):
        """Require every reservoir field to share the canonical field shape."""
        if any(field.shape != shape for field in state):
            raise ValueError(f"reservoir state shape must be {shape}")

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
        """Advance canonical reservoir fields using a canonical condensate field."""
        raise NotImplementedError

    @abstractmethod
    def get_reservoir_density(self) -> Union[np.ndarray, cp.ndarray]:
        """Return total density in the public scalar shape."""
        raise NotImplementedError

    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> tuple:
        """Return canonical ``(B, C_res, ny, nx)`` state arrays."""
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: tuple) -> None:
        """Set canonical ``(B, C_res, ny, nx)`` state arrays."""
        raise NotImplementedError

    @abstractmethod
    def get_derivatives(self, psi, pump, state: tuple, rho=None) -> tuple:
        """Return canonical derivatives from canonical condensate and state arrays."""
        raise NotImplementedError

    def get_active_density(self, state: tuple):
        """Return the active reservoir density."""
        return state[0]

    def get_inactive_density(self, state: tuple):
        """Return the inactive reservoir density, or zero if the model has none."""
        return state[0] * 0.0
