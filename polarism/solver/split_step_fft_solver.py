import numpy as np
from polarism.solver.solver_registry import register_solver
from polarism.solver.abstract_solver import AbstractSolver
import matplotlib.pyplot as plt

@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):    
    def __init__(self, state, config, grid, potential, lasers, reservoir, boundary_condition, visualizer):
        super().__init__(state, config, grid, potential, lasers, reservoir, boundary_condition, visualizer)
        self.potential += self.boundary_condition.before_step_action()
        self._kinetic_propagator = np.exp(-1j * self.physics.hbar * self.grid.k_squared * (self.config.solver.dt / 2) / (2 * self.physics.m_eff))
        self.steps_count = 0
        try:
            plt.ion()
            self._interactive_fig = None
        except Exception:
            self._interactive_fig = None

    def step(self):
        self._first_half_step_kinetic()
        P_total = np.zeros_like(self.grid.X)
        for laser in self.lasers:
            P_total += laser.P(self.grid.X, self.grid.Y, self.state.t)
        self._full_step_potential(P_total)
        self.state.psi = self.boundary_condition.after_step_action(self.state.psi)
        self._first_half_step_kinetic()

        if (self.steps_count % 100) == 0:
            self.visualizer.plot(P_total, self.state.psi, self.state.nR, self.grid)

        self.steps_count += 1
        self.state.t += self.config.solver.dt
    
    def _first_half_step_kinetic(self):
        psi_k = np.fft.fft2(self.state.psi)
        psi_k *= self._kinetic_propagator
        self.state.psi = np.fft.ifft2(psi_k)
    
    def _full_step_potential(self, P):
        self.state.nR = self.reservoir.step(self.state.nR, self.config.solver.dt, self.state.psi, P)
        eff_energy = self.potential + self.physics.g_C * np.abs(self.state.psi)**2 + self.physics.g_R * self.state.nR
        gain_loss = (self.physics.R * self.state.nR - self.physics.gamma_C) / 2.0
        self.state.psi = self.state.psi * np.exp(-1j * eff_energy * self.config.solver.dt / self.physics.hbar) * np.exp(gain_loss * self.config.solver.dt)
