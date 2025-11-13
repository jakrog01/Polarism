from polarism.config.simulation_parameters import ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.simulation_grid_2D import SimulationGrid2D
import numpy as np

class SingleReservoir(AbstractReservoir):
    def __init__(self, reservoir_config: ReservoirParameters, grid: SimulationGrid2D):
        self.R = reservoir_config.R
        self.gamma_r = reservoir_config.gamma_R
        self.n = np.zeros((grid.nx, grid.ny), dtype=np.float32)

    def step(self, dt, psi, Pxy):
        Gamma = self.R * np.abs(psi)**2 + self.gamma_r
        self.n = self.n*np.exp(-Gamma*dt) + (Pxy/Gamma)*(1 - np.exp(-Gamma*dt))
        return self.n
