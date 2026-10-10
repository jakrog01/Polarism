"""ETD-RK2 solver."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Union

from polarism.config.dtype_utils import complex_dtype
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

# The direct phi formulas lose about eps/|z|**2 to cancellation, so small |z|
# uses the Taylor series; ten terms keep its truncation error below 1e-16.
_PHI_TAYLOR_RADIUS = 0.1
_PHI_TAYLOR_TERMS = 10


@register_solver("etd-rk2")
class ETDRK2Solver(AbstractSolver):
    """Solve the model with ETD-RK2."""
    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the etdrk 2 solver."""
        super().__init__(config)
        dt = self.config.solver.dt
        hbar = self.config.physics.hbar
        m_eff = self.config.physics.m_eff

        grid_type = grid.boundary
        if grid_type != "periodic":
            import warnings

            warnings.warn(
                f"ETD-RK2 solver uses FFT-based spectral method which assumes periodic "
                f"boundaries, but grid_type='{grid_type}'. Results may be inaccurate. "
                f"Consider using 'rk4-fdm' solver for non-periodic grids.",
                UserWarning,
            )
        self._L = -1j * hbar * grid.k_squared / (2 * m_eff)
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

    def _nonlinear_rhs(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        nR: Union[np.ndarray, cp.ndarray],
        nI: Union[np.ndarray, cp.ndarray],
        potential: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        """Compute the nonlinear right-hand side."""
        hbar = self.config.physics.hbar
        g_C = self.config.physics.g_C
        g_R = self.config.physics.g_R
        g_I = getattr(self.config.physics, "g_I", 0.0)
        R = self.config.physics.R
        gamma_C = self.config.physics.gamma_C

        rho = self.xp.abs(psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR + g_I * nI
        gain_loss = (R * nR - gamma_C) / 2.0

        return (-1j / hbar) * eff_energy * psi + gain_loss * psi

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Callable[[float], Union[np.ndarray, cp.ndarray]],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        dt = self.config.solver.dt

        res_state = reservoir.get_state()
        nR = reservoir.get_active_density(res_state)
        nI = reservoir.get_inactive_density(res_state)

        psi_k = self.xp.fft.fft2(state.psi)

        N_n = self._nonlinear_rhs(state.psi, nR, nI, potential)
        N_n_k = self.xp.fft.fft2(N_n)

        a_k = self._exp_L * psi_k + self._phi1 * N_n_k * dt
        a = self.xp.fft.ifft2(a_k)

        psi_mid = (state.psi + a) / 2
        reservoir.step(dt, psi_mid, pump(state.t + 0.5 * dt))
        res_new = reservoir.get_state()
        nR_new = reservoir.get_active_density(res_new)
        nI_new = reservoir.get_inactive_density(res_new)

        N_a = self._nonlinear_rhs(a, nR_new, nI_new, potential)
        N_a_k = self.xp.fft.fft2(N_a)

        psi_new_k = a_k + self._phi2 * (N_a_k - N_n_k) * dt
        state.psi = self.xp.fft.ifft2(psi_new_k)

        state.psi = boundary_condition.after_step_action(state.psi)
