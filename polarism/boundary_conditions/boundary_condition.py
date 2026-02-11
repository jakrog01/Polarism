from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.boundary_conditions.absorption import create_absorption_strategy
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)
from polarism.config.simulation_parameters import PhysicsConstants

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        PhysicsConstants,
    )
    from polarism.simulation_grid_2D import SimulationGrid2D


class BoundaryCondition:
    absorption: AbsorptionStrategy

    def __init__(
        self,
        grid: SimulationGrid2D,
        boundary_conditions_config: BoundaryConditionParameters,
        physics_constants: PhysicsConstants,
    ):
        self.absorption = create_absorption_strategy(
            grid, boundary_conditions_config, physics_constants
        )

    def before_step_action(self) -> Union[np.ndarray, cp.ndarray]:
        return self.absorption.get_potential_distribution()

    def after_step_action(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        return self.absorption.apply_absorption(psi)
