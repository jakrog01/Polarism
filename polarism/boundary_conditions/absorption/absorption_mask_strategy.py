"""Mask-based absorption."""
from __future__ import annotations

from polarism.boundary_conditions.absorption.absorption_registry import (
    register_absorption,
)
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
    create_absorption_profile,
)
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    PhysicsConstants,
)
from polarism.grid.simulation_grid_2d import SimulationGrid2D



@register_absorption("mask")
class AbsorptionMaskStrategy(AbsorptionStrategy):
    """Apply boundary absorption with a per-step mask.

    The mask is the exact per-step propagator of the ``cap`` absorbing
    potential with the same ``strength``: ``|psi|^2`` decays at the rate
    ``strength * profile``. Applying it after each step is a first-order
    splitting, so it limits the solver to first order in time; prefer ``cap``
    for quantitative runs.
    """
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
        self.strength = absorption_cfg.strength
        self.mask = None

    def set_time_step(self, dt: float) -> None:
        """Build the per-step mask for the solver time step."""
        xp = compute_engine.xp
        self.mask = xp.exp(-0.5 * self.strength * self.absorption_profile * dt)

    def get_potential_distribution(self):
        """Return potential distribution."""
        return 0.0

    def apply_absorption(self, psi):
        """Apply absorption."""
        if self.mask is None:
            raise RuntimeError("AbsorptionMaskStrategy.set_time_step must be called before apply_absorption")
        if self.mask.dtype != psi.real.dtype:
            self.mask = self.mask.astype(psi.real.dtype)
        return psi * self.mask
