from polarism.config.simulation_parameters import ReservoirParameters, PhysicsConstants
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.simulation_grid_2D import SimulationGrid2D
import numpy as np


class DoubleReservoir(AbstractReservoir):
    def __init__(self, reservoir_config, physics, grid):
        self.R = physics.R
        self.gamma_I = physics.gamma_I
        self.gamma_A = physics.gamma_A
        self.R_IA = physics.R_IA
        self.R_AI = physics.R_AI
        self.nI = np.zeros((grid.nx, grid.ny))
        self.nA = np.zeros((grid.nx, grid.ny))

    def step(self, dt, psi, Pxy):
        abs_psi2 = np.abs(psi)**2
        self.nI += dt * (Pxy - self.gamma_I * self.nI - self.R_IA * self.nI + self.R_AI * self.nA)
        self.nA += dt * (self.R_IA * self.nI - self.gamma_A * self.nA - self.R_AI * self.nA - self.R * abs_psi2 * self.nA)

    def get_reservoir_density(self):
        return self.nA
