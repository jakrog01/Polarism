from polarism.boundary_conditions.absorption import register_absorption
from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
    create_absorption_profile,
)
from polarism.compute_engine import compute_engine


@register_absorption("example-soft-edge")
class SoftEdgeAbsorption(AbsorptionStrategy):
    after_step_is_noop = False

    def __init__(self, grid, absorption_cfg, physics_constants):
        self.xp = compute_engine.xp
        self.profile = create_absorption_profile(grid.shape, absorption_cfg)
        self.strength = absorption_cfg.strength
        self.dt = None

    def set_time_step(self, dt):
        self.dt = dt

    def get_potential_distribution(self):
        return 0.0

    def apply_absorption(self, psi):
        return psi * self.xp.exp(-0.5 * self.strength * self.dt * self.profile)
