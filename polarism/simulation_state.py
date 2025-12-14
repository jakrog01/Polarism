import numpy as np

rng = np.random.default_rng()


class SimulationState:
    def __init__(self, grid):
        self.psi = 1e-3 * (
            rng.random((grid.nx, grid.ny)) + 1j * rng.random((grid.nx, grid.ny))
        )
        self.t = 0.0

    def get_simulation_state(self):
        return self.psi, self.t
