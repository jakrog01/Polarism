from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.boundary_conditions.absorption.absorption_registry import (
    available_boundary_conditions,
)

if TYPE_CHECKING:
    from polarism.boundary_conditions.absorption.absorption_strategy import (
        AbsorptionStrategy,
    )
    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        PhysicsConstants,
    )
    from polarism.simulation_grid_2D import SimulationGrid2D


def create_absorption_strategy(
    grid: SimulationGrid2D,
    boundary_conditions_config: BoundaryConditionParameters,
    physics_constants: PhysicsConstants,
) -> AbsorptionStrategy:
    if boundary_conditions_config.absorption not in available_boundary_conditions:
        raise ValueError(
            f"Unknown boundary conditions: '{boundary_conditions_config.absorption}'. "
            f"Available: {list(available_boundary_conditions.keys())}"
        )

    return available_boundary_conditions[boundary_conditions_config.absorption](
        grid, boundary_conditions_config, physics_constants
    )
