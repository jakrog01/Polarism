from .absorption_mask_strategy import AbsorptionMaskStrategy
from .absorption_perturbation_strategy import AbsorptionPerturbationStrategy

class BoundaryConditionFactory:
    @staticmethod
    def create_boundary_condition_strategy(grid, boundary_condition_cfg, physics_cfg):
        if boundary_condition_cfg.absorption == "absorption-mask":
            return AbsorptionMaskStrategy(grid, boundary_condition_cfg, physics_cfg)
        elif boundary_condition_cfg.absorption == "absorption-perturbation":
            return AbsorptionPerturbationStrategy(grid, boundary_condition_cfg, physics_cfg)
        else:
            raise ValueError(f"Unknown boundary condition type: {boundary_condition_cfg.absorption}")
