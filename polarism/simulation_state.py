import numpy as np
rng = np.random.default_rng()

class SimulationState:
    def __init__(self, grid):
        self.psi = 1e-3 * (rng.random((grid.nx, grid.ny)) + 1j * rng.random((grid.nx, grid.ny)))
        self.t = 0.0
