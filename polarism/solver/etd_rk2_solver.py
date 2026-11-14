"""ETD-RK2 solver."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Union

from polarism.config.dtype_utils import complex_dtype
from polarism.dispersion import apply_spinor_matrix, spinor_matrix_exponential
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

# The direct phi formulas lose about eps/|z|**2 to cancellation, so small |z|
# uses the Taylor series; ten terms keep its truncation error below 1e-16.
_PHI_TAYLOR_RADIUS = 0.1
_PHI_TAYLOR_TERMS = 10


@register_solver("etd-rk2")
class ETDRK2Solver(AbstractSolver):
    """Solve the model with ETD-RK2."""
    capabilities = SolverCapabilities(
        components=frozenset({1, 2}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="quantitative", potential=True,
        kinetic_relaxation=False, dispersion=frozenset({"*"}),
        coherent_drive=True, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='ETD-RK2 spectral solver. FFT-only, periodic grid only. Reservoir advanced between predictor and corrector using a midpoint condensate estimate (psi_n + a)/2, giving O(dt²) psi↔reservoir coupling (measured p≈2.00 on quadratic-double, per convergence study). Does not evaluate kinetic_relaxation_eta.',
    )
    def __init__(self, config: Config, grid: SimulationGrid, model):
        """Set up the etdrk 2 solver."""
        super().__init__(config, grid, model)
        self._has_drive = model.drive is not None
        self._time_dependent_potential = model.time_dependent_potential
        self._fft = self.xp.fft.fft if grid.ndim == 1 else self.xp.fft.fft2
        self._ifft = self.xp.fft.ifft if grid.ndim == 1 else self.xp.fft.ifft2
        dt = self.config.solver.dt
        hbar = self.batch_physics.get("hbar")
        m_eff = self.batch_physics.get("m_eff")

        grid_type = grid.boundary
        if grid_type != "periodic":
            import warnings

            warnings.warn(
                f"ETD-RK2 solver uses FFT-based spectral method which assumes periodic "
                f"boundaries, but grid_type='{grid_type}'. Results may be inaccurate. "
                f"Consider using 'rk4-fdm' solver for non-periodic grids.",
                UserWarning,
            )
        self._spinor = model.components == 2
        if self._spinor:
            matrix = model.dispersion.matrix(grid, self.batch_physics, 2)
            cdtype = complex_dtype(self.xp, getattr(config.solver, "precision", "double"))
            self._exp_L = spinor_matrix_exponential(matrix, hbar, dt, self.xp).astype(cdtype)
            self._phi1 = self._spinor_phi(matrix, hbar, dt, 1).astype(cdtype)
            self._phi2 = self._spinor_phi(matrix, hbar, dt, 2).astype(cdtype)
            return
        if model.dispersion.kind == "parabolic":
            self._L = -1j * hbar * grid.k_squared / (2 * m_eff)
        else:
            self._L = -1j * model.dispersion.energy(grid, self.batch_physics) / hbar
        L_dt = self._L * dt

        # Coefficients are evaluated in float64, then stored in solver precision.
        cdtype = complex_dtype(self.xp, getattr(config.solver, "precision", "double"))
        self._exp_L = self.xp.exp(L_dt).astype(cdtype)
        self._phi1 = self._compute_phi1(L_dt).astype(cdtype)
        self._phi2 = self._compute_phi2(L_dt).astype(cdtype)

    def _phi_taylor(self, z, offset):
        """Return sum_k z**k / (k + offset)! for small ``|z|``."""
        result = self.xp.zeros_like(z)
        for k in range(_PHI_TAYLOR_TERMS - 1, -1, -1):
            result = result * z + 1.0 / math.factorial(k + offset)
        return result

    def _compute_phi1(self, z):
        """Compute the phi1 coefficient."""
        small = self.xp.abs(z) < _PHI_TAYLOR_RADIUS
        z_safe = self.xp.where(small, 1.0, z)
        direct = (self.xp.exp(z_safe) - 1) / z_safe
        return self.xp.where(small, self._phi_taylor(z, 1), direct)

    def _compute_phi2(self, z):
        """Compute the phi2 coefficient."""
        small = self.xp.abs(z) < _PHI_TAYLOR_RADIUS
        z_safe = self.xp.where(small, 1.0, z)
        direct = (self.xp.exp(z_safe) - 1 - z_safe) / (z_safe**2)
        return self.xp.where(small, self._phi_taylor(z, 2), direct)

    def _spinor_phi(self, matrix, hbar, dt, order):
        """Evaluate phi on the two eigenvalues and reconstruct the 2×2 matrix."""
        if hasattr(hbar, "ndim") and hbar.ndim:
            hbar = hbar[:, 0]
        epsilon = (matrix[:, 0, 0] + matrix[:, 1, 1]) / 2
        z = (matrix[:, 0, 0] - matrix[:, 1, 1]) / 2
        upper = matrix[:, 0, 1]
        lower = matrix[:, 1, 0]
        omega = self.xp.sqrt(z * z + upper * lower)
        coefficient = self._compute_phi1 if order == 1 else self._compute_phi2
        plus = coefficient(-1j * (epsilon + omega) * dt / hbar)
        minus = coefficient(-1j * (epsilon - omega) * dt / hbar)
        average = (plus + minus) / 2
        ratio = (plus - minus) / (2 * self.xp.where(omega == 0, 1, omega))
        return self.xp.stack((self.xp.stack((average + ratio * z, ratio * upper), axis=1),
                              self.xp.stack((ratio * lower, average - ratio * z), axis=1)), axis=1)

    def _linear(self, matrix, field):
        """Apply the spinor spectral matrix."""
        return apply_spinor_matrix(matrix, field, self.xp)

    def _reservoir_step(self, model, dt, field, pump):
        """Advance component or shared reservoirs with the midpoint rule."""
        reservoir = model.reservoir
        if model.reservoir_components == 2:
            reservoir.step(dt, field, pump * model.pump_weights)
            return
        if not reservoir.get_state():
            reservoir.step(dt, field[:, :1], pump)
            return
        state = reservoir.get_state()
        density = (field.real * field.real + field.imag * field.imag).sum(axis=1, keepdims=True)
        first = reservoir.get_derivatives(field[:, :1], pump, state, rho=density)
        middle = tuple(value + 0.5 * dt * slope for value, slope in zip(state, first))
        second = reservoir.get_derivatives(field[:, :1], pump, middle, rho=density)
        reservoir.set_state(tuple(value + dt * slope for value, slope in zip(state, second)))

    def _nonlinear_rhs(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        nR: Union[np.ndarray, cp.ndarray],
        nI: Union[np.ndarray, cp.ndarray],
        potential: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Compute the nonlinear right-hand side."""
        hbar = self.batch_physics.get("hbar")
        g_C = self.batch_physics.get("g_C")
        g_R = self.batch_physics.get("g_R")
        g_I = self.batch_physics.get("g_I")
        R = self.batch_physics.get("R")
        gamma_C = self.batch_physics.get("gamma_C")

        rho = self.xp.abs(psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR + g_I * nI
        if self._spinor:
            eff_energy = eff_energy + self.batch_physics.get("alpha_2") * rho[:, ::-1]
        gain_loss = (R * nR - gamma_C) / 2.0

        return (-1j / hbar) * eff_energy * psi + gain_loss * psi

    def step(self, model, state, dt):
        """Advance the solver by one time step."""
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field, getattr(state, "components", 1))
        dt = self.config.solver.dt

        res_state = reservoir.get_state()
        nR = reservoir.get_active_density(res_state)
        nI = reservoir.get_inactive_density(res_state)

        psi_k = self._fft(state.field)

        if self._time_dependent_potential:
            potential = model.potential_at(state.t)
        N_n = self._nonlinear_rhs(state.field, nR, nI, potential)
        if model.terms:
            N_n = N_n + model.extra_rhs(state.field, res_state, state.t)
        if self._has_drive:
            N_n = N_n + model.drive_contribution(state.field, state.t)
        N_n_k = self._fft(N_n)

        if self._spinor:
            a_k = self._linear(self._exp_L, psi_k) + self._linear(self._phi1, N_n_k) * dt
        else:
            a_k = self._exp_L * psi_k + self._phi1 * N_n_k * dt
        a = self._ifft(a_k)

        psi_mid = (state.field + a) / 2
        if self._spinor:
            self._reservoir_step(model, dt, psi_mid, evaluate_pump(pump, state.t + 0.5 * dt))
        else:
            reservoir.step(dt, psi_mid, evaluate_pump(pump, state.t + 0.5 * dt))
        res_new = reservoir.get_state()
        nR_new = reservoir.get_active_density(res_new)
        nI_new = reservoir.get_inactive_density(res_new)

        if self._time_dependent_potential:
            potential = model.potential_at(state.t + dt)
        N_a = self._nonlinear_rhs(a, nR_new, nI_new, potential)
        if model.terms:
            N_a = N_a + model.extra_rhs(a, res_new, state.t + dt)
        if self._has_drive:
            N_a = N_a + model.drive_contribution(a, state.t + dt)
        N_a_k = self._fft(N_a)

        psi_new_k = (a_k + self._linear(self._phi2, N_a_k - N_n_k) * dt
                     if self._spinor else a_k + self._phi2 * (N_a_k - N_n_k) * dt)
        state.field = self._ifft(psi_new_k)

        state.field = boundary_condition.after_step_action(state.field)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
