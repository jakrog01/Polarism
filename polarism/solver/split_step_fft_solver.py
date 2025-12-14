import numpy as np

from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver


@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):
    def __init__(self, config, grid):
        super().__init__(config)
        self._kinetic_propagator = np.exp(
            -1j
            * self.config.physics.hbar
            * grid.k_squared
            * (self.config.solver.dt / 2)
            / (2 * self.config.physics.m_eff)
        )

    def step(self, potential, pump, reservoir, boundary_condition, state):
        self._half_step_kinetic(state)
        self._full_step_potential(pump, reservoir, potential, state)
        state.psi = boundary_condition.after_step_action(state.psi)
        self._half_step_kinetic(state)

    def _half_step_kinetic(self, state):
        psi_k = np.fft.fft2(state.psi)
        psi_k *= self._kinetic_propagator
        state.psi = np.fft.ifft2(psi_k)

    def _full_step_potential(self, P, reservoir, potential, state):
        reservoir.step(self.config.solver.dt, state.psi, P)
        eff_energy = (
            potential
            + self.config.physics.g_C * np.abs(state.psi) ** 2
            + self.config.physics.g_R * reservoir.get_reservoir_density()
        )
        gain_loss = (
            self.config.physics.R * reservoir.get_reservoir_density()
            - self.config.physics.gamma_C
        ) / 2.0
        state.psi = (
            state.psi
            * np.exp(
                -1j * eff_energy * self.config.solver.dt / self.config.physics.hbar
            )
            * np.exp(gain_loss * self.config.solver.dt)
        )
