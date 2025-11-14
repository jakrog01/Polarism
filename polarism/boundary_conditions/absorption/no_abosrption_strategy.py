from polarism.boundary_conditions.absorption.absorption_strategy import AbsorptionStrategy
from polarism.boundary_conditions.absorption.absorption_registry import register_absorption

@register_absorption("no-absorption")
class NoAbsorptionStrategy(AbsorptionStrategy):
    def before_step_action(self):
        return 0.0

    def after_step_action(self, psi):
        return psi