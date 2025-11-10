class BoundryConditionFactory:
    @staticmethod
    def create_boundry_condition_strategy(grid, boundry_condition_cfg, physics_cfg):
        if boundry_condition_cfg.absorption == "absorbtion-mask":
            from simulation.boundry_conditions_strategies.absorption_mask_strategy import AbsorptionMaskStrategy
            return AbsorptionMaskStrategy(grid, boundry_condition_cfg, physics_cfg)
        elif boundry_condition_cfg.absorption == "absorbtion-perturbation":
            from simulation.boundry_conditions_strategies.absorption_perturbation_strategy import AbsorptionPerturbationStrategy
            return AbsorptionPerturbationStrategy(grid, boundry_condition_cfg, physics_cfg)
        else:
            raise ValueError(f"Unknown boundary condition type: {boundry_condition_cfg.absorption}")
