import numpy as np
from polarism.solver.solver_registry import register_solver
from polarism.solver.abstract_solver import AbstractSolver

@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):    
    def __init__(self, state, config, grid, potential, laser, reservoir, boundary_condition):
        super().__init__(state, config, grid, potential, laser, reservoir, boundary_condition)
        
    def step(self):
        self._first_half_step_kinetic()
        self._full_step_fourier()
        self._first_half_step_kinetic()
        self.state.t += self.config.solver.dt
    
    def _first_half_step_kinetic(self):
        kinetic_propagator = np.exp(-1j * self.physics.hbar * self.grid.k_squared * (self.config.solver.dt / 2) / (2 * self.physics.m_eff))
        psi_k = np.fft.fft2(self.state.psi)
        psi_k *= kinetic_propagator
        self.state.psi = np.fft.ifft2(psi_k)
    
    def _full_step_fourier(self):
        self.state.n = self.reservoir.step(self.state.n, self.config.solver.dt, self.state.psi, self.laser.P(self.grid.X, self.grid.Y, self.state.t))

        eff_energy = self.potential + self.physics.g_C * np.abs(self.state.psi)**2 + self.physics.g_R * self.reservoir.n
        gain_loss = (self.physics.R * self.reservoir.n - self.physics.gamma_C) / 2.0

        self.state.psi = self.state.psi * np.exp(-1j * eff_energy * self.config.solver.dt / self.physics.hbar) * np.exp(gain_loss * self.config.solver.dt)