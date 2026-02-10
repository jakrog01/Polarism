from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from polarism.boundary_conditions.absorption.absorption_strategy import (
        AbsorptionStrategy,
    )
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.simulation_state import SimulationState


class AbstractSolver(ABC):
    @abstractmethod
    def __init__(self, config: Config):
        self.config = config
        self.physics = config.physics

    @abstractmethod
    def step(
        self,
        potential: AbsorptionStrategy,
        pump: np.ndarray,
        reservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        pass
