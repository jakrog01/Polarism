"""Mask-based absorption."""
from __future__ import annotations

from polarism.boundary_conditions.absorption.absorption_registry import (
    register_absorption,
)
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
    create_absorption_profile,
)
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    PhysicsConstants,
)
from polarism.grid.simulation_grid_2d import SimulationGrid2D



@register_absorption("mask")
class AbsorptionMaskStrategy(AbsorptionStrategy):
    """Apply boundary absorption with a mask."""
    grid: SimulationGrid2D
    absorption_cfg: BoundaryConditionParameters

    def __init__(
        self,
        grid: SimulationGrid2D,
        absorption_cfg: BoundaryConditionParameters,
        physics_constants: PhysicsConstants | None = None
    ):
        """Set up the absorption mask strategy."""
        self.absorption_profile = create_absorption_profile(grid.ny, grid.nx, absorption_cfg)
        self.mask = (1.0 - self.absorption_profile) ** absorption_cfg.strength

    def get_potential_distribution(self):
        """Return potential distribution."""
        return 0.0

    def apply_absorption(self, psi):
        """Apply absorption."""
        return psi * self.mask
