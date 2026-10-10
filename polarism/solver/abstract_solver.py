"""Solver interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Union

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.simulation_state import SimulationState


class AbstractSolver(ABC):
    """Define the interface for solver backends."""
    @abstractmethod
    def __init__(self, config: Config):
        """Set up the abstract solver."""
        self.config = config
        self.physics = config.physics
        self.xp = compute_engine.xp

    @abstractmethod
    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Callable[[float], Union[np.ndarray, cp.ndarray]],
        reservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the fields from ``state.t`` to ``state.t + dt``.

        ``pump(t)`` returns the pump array at time ``t``. Evaluate it at the
        integration stage times; the caller updates ``state.t`` after the step.
        """
        pass
