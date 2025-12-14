from polarism.config.simulation_parameters import ReservoirParameters, PhysicsConstants
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.simulation_grid_2D import SimulationGrid2D
import numpy as np

class SingleReservoir(AbstractReservoir):
    def __init__(self, reservoir_config: ReservoirParameters, physics: PhysicsConstants, grid : SimulationGrid2D):
        self.R = physics.R
        self.gamma_r = physics.gamma_R
        self.nR = np.zeros((grid.nx, grid.ny))

    def step(self, dt, psi, Pxy):
        Gamma = self.R * np.abs(psi)**2 + self.gamma_r
        self.nR = self.nR*np.exp(-Gamma*dt) + (Pxy/Gamma)*(1 - np.exp(-Gamma*dt))
       
    def get_reservoir_density(self):
        return self.nR