"""Split-step FFT solver."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from polarism.config.batch_physics import BatchPhysics
from polarism.laser.pump_field import evaluate_pump
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

    def __init__(self, config: Config, grid: SimulationGrid2D, batch_physics=None):
        """Set up the split step fft solver."""
        super().__init__(config)
        self.batch_physics = BatchPhysics(config.physics, batch_physics, self.xp, getattr(config.solver, "precision", "double"))

        self._kinetic_propagator_half = self.xp.exp(
            -1j
            * self.batch_physics.get("hbar")
            * grid.k_squared
            * (self.config.solver.dt / 2)
            / (2 * self.batch_physics.get("m_eff"))
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
        pump: Callable[[float], Union[np.ndarray, cp.ndarray]],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        self.batch_physics.check(state.field)
        dt = self.config.solver.dt
        hbar = self.batch_physics.get("hbar")
        g_C = self.batch_physics.get("g_C")
        g_R = self.batch_physics.get("g_R")
        g_I = self.batch_physics.get("g_I")
        R = self.batch_physics.get("R")
        gamma_C = self.batch_physics.get("gamma_C")

        reservoir.step(dt / 2, state.field, evaluate_pump(pump, state.t + dt / 4))
        res_state = reservoir.get_state()
        nR = reservoir.get_active_density(res_state)
        nI = reservoir.get_inactive_density(res_state)

        state.field = self._kinetic_half_step_fft(state.field)

        rho = self.xp.abs(state.field) ** 2
        gain_loss = (R * nR - gamma_C) / 2.0 + self.xp.imag(potential) / hbar
        x = 2 * gain_loss * dt
        small = self.xp.abs(x) < 1e-6
        phase_time = dt * self.xp.where(
            small,
            1 + x / 2 + x * x / 6,
            self.xp.expm1(x) / self.xp.where(small, 1, x),
        )
        phase = (
            (self.xp.real(potential) + g_R * nR + g_I * nI) * dt
            + g_C * rho * phase_time
        ) / hbar
        state.field = state.field * self.xp.exp(gain_loss * dt - 1j * phase)

        state.field = self._kinetic_half_step_fft(state.field)

        reservoir.step(dt / 2, state.field, evaluate_pump(pump, state.t + 3 * dt / 4))
        state.field = boundary_condition.after_step_action(state.field)
