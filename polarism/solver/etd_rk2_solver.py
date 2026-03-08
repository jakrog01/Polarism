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


@register_solver("etd-rk2")
class ETDRK2Solver(AbstractSolver):
    def __init__(self, config: Config, grid: SimulationGrid2D):
        super().__init__(config)
        dt = self.config.solver.dt
        hbar = self.config.physics.hbar
        m_eff = self.config.physics.m_eff

        grid_type = getattr(config.grid, "grid_type", "periodic")
        if grid_type != "periodic":
            import warnings

            warnings.warn(
                f"ETD-RK2 solver uses FFT-based spectral method which assumes periodic "
                f"boundaries, but grid_type='{grid_type}'. Results may be inaccurate. "
                f"Consider using 'rk4-fdm' solver for non-periodic grids.",
                UserWarning,
            )
        self._L = 1j * hbar * grid.k_squared / (2 * m_eff)
        L_dt = self._L * dt

        self._exp_L = self.xp.exp(L_dt)
        self._phi1 = self._compute_phi1(L_dt)
        self._phi2 = self._compute_phi2(L_dt)

    def _compute_phi1(self, z):
        exp_z = self.xp.exp(z)
        z_safe = self.xp.where(self.xp.abs(z) > 1e-12, z, 1e-12)
        direct = (exp_z - 1) / z_safe
        taylor = 1 + z / 2 + z**2 / 6 + z**3 / 24
        result = self.xp.where(self.xp.abs(z) > 1e-10, direct, taylor)
        return result

    def _compute_phi2(self, z):
        exp_z = self.xp.exp(z)
        z_safe = self.xp.where(self.xp.abs(z) > 1e-12, z, 1e-12)
        direct = (exp_z - 1 - z_safe) / (z_safe**2)
        taylor = 0.5 + z / 6 + z**2 / 24 + z**3 / 120
        result = self.xp.where(self.xp.abs(z) > 1e-10, direct, taylor)
        return result

    def _nonlinear_rhs(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        nR: Union[np.ndarray, cp.ndarray],
        potential: Union[np.ndarray, cp.ndarray],
    ) -> Union[np.ndarray, cp.ndarray]:
        hbar = self.config.physics.hbar
        g_C = self.config.physics.g_C
        g_R = self.config.physics.g_R
        R = self.config.physics.R
        gamma_C = self.config.physics.gamma_C

        rho = self.xp.abs(psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR
        gain_loss = (R * nR - gamma_C) / 2.0

        return (-1j / hbar) * eff_energy * psi + gain_loss * psi

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        dt = self.config.solver.dt

        nR = reservoir.get_reservoir_density()

        psi_k = self.xp.fft.fft2(state.psi)

        N_n = self._nonlinear_rhs(state.psi, nR, potential)
        N_n_k = self.xp.fft.fft2(N_n)

        a_k = self._exp_L * psi_k + self._phi1 * N_n_k * dt
        a = self.xp.fft.ifft2(a_k)

        psi_mid = (state.psi + a) / 2
        reservoir.step(dt, psi_mid, pump)
        nR_new = reservoir.get_reservoir_density()

        N_a = self._nonlinear_rhs(a, nR_new, potential)
        N_a_k = self.xp.fft.fft2(N_a)

        psi_new_k = a_k + self._phi2 * (N_a_k - N_n_k) * dt
        state.psi = self.xp.fft.ifft2(psi_new_k)

        state.psi = boundary_condition.after_step_action(state.psi)
