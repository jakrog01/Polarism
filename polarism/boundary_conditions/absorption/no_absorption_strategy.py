"""No-op absorption."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.boundary_conditions.absorption.absorption_registry import (
    register_absorption,
)
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        PhysicsConstants,
    )
    from polarism.grid.simulation_grid_2d import SimulationGrid2D


@register_absorption("no-absorption")
class NoAbsorptionStrategy(AbsorptionStrategy):
    """Leave the field unchanged at the boundary."""
    after_step_is_noop = True

    def __init__(
        self,
        grid: SimulationGrid2D | None,
        absorption_cfg: BoundaryConditionParameters | None,
        physics_constants: PhysicsConstants | None,
    ):
        """Set up the no absorption strategy."""
        super().__init__()

    def get_potential_distribution(self) -> Union[np.ndarray, cp.ndarray]:
        """Return potential distribution."""
        return 0.0

    def apply_absorption(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply absorption."""
        return psi
