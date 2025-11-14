from polarism.boundary_conditions.absorption.absorption_strategy import AbsorptionStrategy
from polarism.boundary_conditions.absorption.absorption_registry import register_absorption
<<<<<<< HEAD

@register_absorption("no-absorption")
class NoAbsorptionStrategy(AbsorptionStrategy):
    def before_step_action(self):
        return 0.0

    def after_step_action(self, psi):
=======
from polarism.simulation_grid_2D import SimulationGrid2D
from polarism.config.simulation_parameters import BoundaryConditionParameters, PhysicsConstants

@register_absorption("no-absorption")
class NoAbsorptionStrategy(AbsorptionStrategy):
    def __init__(self, grid: SimulationGrid2D, absorption_cfg: BoundaryConditionParameters, physics_constants: PhysicsConstants):
        super().__init__(grid, absorption_cfg, physics_constants)

    def get_potential_distribution(self):
        return 0.0

    def apply_absorption(self, psi):
>>>>>>> b73a8ee (Add NoAbsorption strategy)
        return psi