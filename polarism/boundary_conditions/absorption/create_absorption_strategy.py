from polarism.boundary_conditions.absorption.absorption_registry import (
    available_boundry_conditions,
)
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    PhysicsConstants,
)


def create_absorption_strategy(
    grid,
    boundry_conditions_config: BoundaryConditionParameters,
    physics_constants: PhysicsConstants,
):
    if boundry_conditions_config.absorption not in available_boundry_conditions:
        raise ValueError(
            f"Unknown boundry conditions: '{boundry_conditions_config.absorption}'. "
            f"Available: {list(available_boundry_conditions.keys())}"
        )

    return available_boundry_conditions[boundry_conditions_config.absorption](
        grid, boundry_conditions_config, physics_constants
    )
