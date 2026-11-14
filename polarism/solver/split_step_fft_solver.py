"""Split-step FFT solver."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver, ACCEPTED, SolverCapabilities
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):
    """Solve the model with split-step FFT."""
    capabilities = SolverCapabilities(
        components=frozenset({1}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="quantitative", potential=True,
        kinetic_relaxation=False, dispersion=frozenset({"*"}),
        coherent_drive=False, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='Spectral Strang splitting, second order in time. Reservoir advances in two half-steps around the condensate step. FFT-only, periodic grid only. Does not evaluate kinetic_relaxation_eta.',
    )
    injects_noise = True
    _kinetic_propagator_half: Union[np.ndarray, cp.ndarray]

    def __init__(self, config: Config, grid: SimulationGrid, model):
        """Set up the split step fft solver."""
        super().__init__(config, grid, model)
        self._time_dependent_potential = model.time_dependent_potential
        self._fft = self.xp.fft.fft if grid.ndim == 1 else self.xp.fft.fft2
        self._ifft = self.xp.fft.ifft if grid.ndim == 1 else self.xp.fft.ifft2

        if model.dispersion.kind == "parabolic":
            self._kinetic_propagator_half = self.xp.exp(
                -1j
                * self.batch_physics.get("hbar")
                * grid.k_squared
                * (self.config.solver.dt / 2)
                / (2 * self.batch_physics.get("m_eff"))
            )
        else:
            self._kinetic_propagator_half = self.xp.exp(
                -1j
                * model.dispersion.energy(grid, self.batch_physics)
                * (self.config.solver.dt / 2)
                / self.batch_physics.get("hbar")
            )

    def _kinetic_half_step_fft(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply the FFT half-step of the kinetic term."""
        psi_k = self._fft(psi)
        psi_k *= self._kinetic_propagator_half
        return self._ifft(psi_k)

    def step(self, model, state, dt):
        """Advance the solver by one time step."""
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field)
        dt = self.config.solver.dt
        if self._time_dependent_potential:
            potential = model.potential_at(state.t + dt / 2)
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
        if model.terms:
            extra_energy, extra_rate = model.extra_coefficients(
                state.field, res_state, state.t + dt / 2
            )
            state.field = state.field * self.xp.exp((extra_rate - 1j * extra_energy / hbar) * dt)

        if model.noise is not None:
            model.noise.apply_arrays(state.field, nR, state.step)

        state.field = self._kinetic_half_step_fft(state.field)

        reservoir.step(dt / 2, state.field, evaluate_pump(pump, state.t + 3 * dt / 4))
        state.field = boundary_condition.after_step_action(state.field)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
