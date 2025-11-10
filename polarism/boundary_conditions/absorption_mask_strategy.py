import numpy as np
import matplotlib.pyplot as plt
from .absorption_strategy import AbsorptionStrategy, create_absorption_profile
from ..simulation_grid_2D import SimulationGrid2D
from ..config.simulation_parameters import AbsorptionParameters
from ..config.simulation_parameters import PhysicsConstants

class AbsorptionMaskStrategy(AbsorptionStrategy):
    def __init__(self, grid: SimulationGrid2D, absorption_cfg: AbsorptionParameters, physics_constants: PhysicsConstants):
        shape = (grid.nx, grid.ny)
        self.absorption_profile = create_absorption_profile(shape, absorption_cfg)
        self.mask = 1.0 - self.absorption_profile

    def get_potential_distribution(self):
        return 0.0
    
    def apply_absorption(self, psi):
        return psi * self.mask
    