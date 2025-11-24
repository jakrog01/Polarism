from abc import ABC, abstractmethod
from polarism.config.simulation_parameters import Config

class AbstractSolver(ABC):
    @abstractmethod
    def __init__(self, state, config: Config, grid, potential, laser, reservoir, boundary_condition, visualizer):
        self.config = config
        self.physics = config.physics
        self.grid = grid
        self.potential = potential
        self.laser = laser
        self.reservoir = reservoir
        self.boundary_condition = boundary_condition
        self.state = state
        self.visualizer = visualizer

    @abstractmethod
    def step(self, psi):
        pass
