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

        ``state.field``, ``state.field_k``, and reservoir state tuples use
        ``(B, 1, *S)``, where ``S`` is the spatial grid shape. Shared pump and potential arrays use ``grid.shape``;
        member-stacked arrays use ``(B, 1, *S)``. Evaluate ``pump(t)``
        at integration stage times; the caller updates ``state.t`` afterward.
        """
        pass

    def advance(
        self,
        potential,
        pump,
        reservoir,
        boundary_condition,
        state: SimulationState,
        start_step: int,
        n_steps: int,
    ) -> None:
        """Advance several steps, preserving controller stage times."""
        if hasattr(self, "batch_physics"):
            self.batch_physics.check(state.field)
        for k in range(n_steps):
            self.step(potential, pump, reservoir, boundary_condition, state)
            state.t = (start_step + k + 1) * self.config.solver.dt
