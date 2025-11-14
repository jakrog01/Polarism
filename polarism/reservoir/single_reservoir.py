from polarism.config.simulation_parameters import ReservoirParameters, PhysicsConstants
from polarism.reservoir.abstract_reservoir import AbstractReservoir
import numpy as np

class SingleReservoir(AbstractReservoir):
    def __init__(self, reservoir_config: ReservoirParameters, physics: PhysicsConstants):
        self.R = physics.R
        self.gamma_r = physics.gamma_R

    def step(self, n, dt, psi, Pxy):
        Gamma = self.R * np.abs(psi)**2 + self.gamma_r
        n_new = n*np.exp(-Gamma*dt) + (Pxy/Gamma)*(1 - np.exp(-Gamma*dt))
        return n_new
