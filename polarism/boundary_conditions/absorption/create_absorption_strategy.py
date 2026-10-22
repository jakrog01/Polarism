"""Absorption factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.boundary_conditions.absorption.absorption_registry import (
    available_absorptions,
)

if TYPE_CHECKING:
    from polarism.boundary_conditions.absorption.absorption_strategy import (
        AbsorptionStrategy,
    )
    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        PhysicsConstants,
    )
    from polarism.grid.simulation_grid_2d import SimulationGrid2D


def create_absorption_strategy(
    grid: SimulationGrid2D,
    boundary_conditions_config: BoundaryConditionParameters,
    physics_constants: PhysicsConstants,
) -> AbsorptionStrategy:
    """Create absorption strategy."""
    return available_absorptions.lookup(boundary_conditions_config.absorption)(
        grid, boundary_conditions_config, physics_constants
    )
