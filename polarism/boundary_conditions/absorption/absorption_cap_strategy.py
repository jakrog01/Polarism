from polarism.boundary_conditions.absorption.absorption_strategy import AbsorptionStrategy, create_absorption_profile
from polarism.boundary_conditions.absorption.absorption_registry import register_absorption
from polarism.simulation_grid_2D import SimulationGrid2D
from polarism.config.simulation_parameters import BoundaryConditionParameters, PhysicsConstants

@register_absorption("cap")
class AbsorptionCapStrategy(AbsorptionStrategy):
    def __init__(self, grid: SimulationGrid2D, absorption_cfg: BoundaryConditionParameters, physics_constants: PhysicsConstants):
        shape = (grid.nx, grid.ny)
        self.absorption_profile = create_absorption_profile(shape, absorption_cfg)
        self.potential_dist = -0.5j * physics_constants.hbar * absorption_cfg.strength * self.absorption_profile

    def get_potential_distribution(self):
        return self.potential_dist

    def apply_absorption(self, psi):
        return psi
