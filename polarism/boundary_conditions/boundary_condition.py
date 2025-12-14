from polarism.boundary_conditions.absorption import create_absorption_strategy


class BoundaryCondition:
    def __init__(self, grid, boundary_conditions_config, physics_constants):
        self.absorption = create_absorption_strategy(
            grid, boundary_conditions_config, physics_constants
        )

    def before_step_action(self):
        return self.absorption.get_potential_distribution()

    def after_step_action(self, psi):
        return self.absorption.apply_absorption(psi)
