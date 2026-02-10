from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.boundary_conditions.absorption.absorption_registry import (
    register_absorption,
)
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)

if TYPE_CHECKING:
    import numpy as np

    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        PhysicsConstants,
    )
    from polarism.simulation_grid_2D import SimulationGrid2D


@register_absorption("no-absorption")
class NoAbsorptionStrategy(AbsorptionStrategy):
    def __init__(
        self,
        grid: SimulationGrid2D | None,
        absorption_cfg: BoundaryConditionParameters | None,
        physics_constants: PhysicsConstants | None,
    ):
        super().__init__()

    def get_potential_distribution(self) -> float:
        return 0.0

    def apply_absorption(self, psi: np.ndarray) -> np.ndarray:
        return psi
