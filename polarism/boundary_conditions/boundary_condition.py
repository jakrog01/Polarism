"""Boundary-condition helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.boundary_conditions.absorption import create_absorption_strategy
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)
from polarism.config.simulation_parameters import PhysicsConstants
from polarism.config.batch_physics import BatchPhysics
from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import BoundaryConditionParameters
    from polarism.grid.simulation_grid import SimulationGrid


class BoundaryCondition:
    """Apply boundary actions around each solver step."""
    absorption: AbsorptionStrategy

    def __init__(
        self,
        grid: SimulationGrid,
        boundary_conditions_config: BoundaryConditionParameters,
        physics_constants: PhysicsConstants,
        dt: float,
        batch_physics=None,
    ):
        """Set up the boundary absorption strategy."""
        self.absorption = create_absorption_strategy(
            grid, boundary_conditions_config, physics_constants
        )
        self.absorption.set_time_step(dt)
        if batch_physics:
            physics = BatchPhysics(physics_constants, batch_physics, compute_engine.xp, ndim=grid.ndim)
            if boundary_conditions_config.absorption == "cap" and "hbar" in physics.host:
                self.absorption.potential_dist = (
                    -0.5j * physics.get("hbar") * boundary_conditions_config.strength
                    * self.absorption.absorption_profile
                )

    def before_step_action(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the boundary potential term."""
        return self.absorption.get_potential_distribution()

    def after_step_action(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply the boundary action to psi."""
        return self.absorption.apply_absorption(psi)
