"""Split-step FFT solver."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid_2d import SimulationGrid2D
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):
    """Solve the model with split-step FFT."""
    _kinetic_propagator_half: Union[np.ndarray, cp.ndarray]

    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the split step fft solver."""
        super().__init__(config)

        self._kinetic_propagator_half = self.xp.exp(
            -1j
            * self.config.physics.hbar
            * grid.k_squared
            * (self.config.solver.dt / 2)
            / (2 * self.config.physics.m_eff)
        )

    def _kinetic_half_step_fft(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply the FFT half-step of the kinetic term."""
        psi_k = self.xp.fft.fft2(psi)
        psi_k *= self._kinetic_propagator_half
        return self.xp.fft.ifft2(psi_k)

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        dt = self.config.solver.dt
        hbar = self.config.physics.hbar
        g_C = self.config.physics.g_C
        g_R = self.config.physics.g_R
        g_I = getattr(self.config.physics, "g_I", 0.0)
        R = self.config.physics.R
        gamma_C = self.config.physics.gamma_C

        res_state = reservoir.get_state()
        nR = reservoir.get_active_density(res_state)
        nI = reservoir.get_inactive_density(res_state)

        state.psi = self._kinetic_half_step_fft(state.psi)

        rho = self.xp.abs(state.psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR + g_I * nI
        gain_loss = (R * nR - gamma_C) / 2.0
        state.psi = state.psi * self.xp.exp((-1j * eff_energy / hbar + gain_loss) * dt)

        state.psi = self._kinetic_half_step_fft(state.psi)

        state.psi = boundary_condition.after_step_action(state.psi)
        reservoir.step(dt, state.psi, pump)
