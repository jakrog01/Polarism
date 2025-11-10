from simulation.boundry_conditions_strategies.absorption_strategy import AbsorptionStrategy, create_absorption_profile
from simulation.simulation_grid_2D import SimulationGrid2D
import numpy as np
from config.simulation_parameters import AbsorptionParameters
from config.simulation_parameters import PhysicsConstants
import matplotlib.pyplot as plt

class AbsorptionMaskStrategy(AbsorptionStrategy):
    def __init__(self, grid: SimulationGrid2D, absorption_cfg: AbsorptionParameters, physics_constants: PhysicsConstants):
        shape = (grid.nx, grid.ny)
        self.absorption_profile = create_absorption_profile(shape, absorption_cfg)
        self.mask = 1.0 - self.absorption_profile

    def get_potential_distribution(self):
        return 0.0
    
    def apply_absorption(self, psi):
        return psi * self.mask
    