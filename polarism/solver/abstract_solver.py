from abc import ABC, abstractmethod

from polarism.config.simulation_parameters import Config


class AbstractSolver(ABC):
    @abstractmethod
    def __init__(self, config: Config):
        self.config = config
        self.physics = config.physics

    @abstractmethod
    def step(self, psi):
        pass
