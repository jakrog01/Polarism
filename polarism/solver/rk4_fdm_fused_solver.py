"""Fused finite-difference RK4 solver."""
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


@register_solver("rk4-fdm-fused")
class RK4FDMFusedSolver(AbstractSolver):

    """Solve the model with fused finite-difference RK4."""
    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the fused RK4 solver."""
        super().__init__(config)
        self.dx = grid.dx
        self.dy = grid.dy
        self.nx = grid.nx
        self.ny = grid.ny
        self.grid_type = getattr(config.grid, "grid_type", "periodic")

        self._hbar = config.physics.hbar
        self._m_eff = config.physics.m_eff
        self._g_C = config.physics.g_C
        self._g_R = config.physics.g_R
        self._g_I = getattr(config.physics, "g_I", 0.0)
        self._R = config.physics.R
        self._gamma_C = config.physics.gamma_C
        self._dt = config.solver.dt

        self._inv_dx2 = 1.0 / (self.dx**2)
        self._inv_dy2 = 1.0 / (self.dy**2)
        self._kinetic_coeff = -(self._hbar**2) / (2.0 * self._m_eff)
        self._minus_i_over_hbar = -1j / self._hbar
        self._kinetic_relaxation_eta = getattr(config.physics, "kinetic_relaxation_eta", 0.0)
        self._hbar_over_2m = self._hbar / (2.0 * self._m_eff)

        self._buf_initialized = False
        _empty = self.xp.empty(0, dtype=complex)
        self._lap = _empty
        self._k1_psi = _empty
        self._k2_psi = _empty
        self._k3_psi = _empty
        self._k4_psi = _empty
        self._tmp_psi = _empty

        self._fused_rhs = None
        self._fused_combine = None
        self._setup_fused_kernels()

    def _setup_fused_kernels(self):
        """Build fused array kernels when the backend supports them."""
        if not hasattr(self.xp, "fuse"):
            return

        xp = self.xp
        g_C = self._g_C
        g_R = self._g_R
        g_I = self._g_I
        R = self._R
        gamma_C = self._gamma_C
        minus_i_over_hbar = self._minus_i_over_hbar
        kinetic_coeff = self._kinetic_coeff
        eta = self._kinetic_relaxation_eta
        hbar_over_2m = self._hbar_over_2m

        @xp.fuse()
        def fused_rhs(psi, lap, potential, n_active, n_inactive):
            """Compute the fused right-hand side."""
            rho = xp.abs(psi) ** 2
            eff_energy = potential + g_C * rho + g_R * n_active + g_I * n_inactive
            gain_loss = (R * n_active - gamma_C) * 0.5
            kinetic_energy = kinetic_coeff * lap
            return (
                minus_i_over_hbar * (kinetic_energy + eff_energy * psi)
                + gain_loss * psi
                + eta * n_active * hbar_over_2m * lap
            )

        @xp.fuse()
        def fused_combine_rk4(psi0, k1, k2, k3, k4, dt_over_6):
            """Combine RK4 stages in one fused kernel."""
            return psi0 + dt_over_6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self._fused_rhs = fused_rhs
        self._fused_combine = fused_combine_rk4

    def _ensure_buffers(self, psi):
        """Allocate work buffers for the current field shape."""
        if self._buf_initialized and self._lap.shape == psi.shape:
            return
        shape = psi.shape
        dtype = psi.dtype
        xp = self.xp
        self._lap = xp.empty(shape, dtype=dtype)
        self._k1_psi = xp.empty(shape, dtype=dtype)
        self._k2_psi = xp.empty(shape, dtype=dtype)
        self._k3_psi = xp.empty(shape, dtype=dtype)
        self._k4_psi = xp.empty(shape, dtype=dtype)
        self._tmp_psi = xp.empty(shape, dtype=dtype)
        self._buf_initialized = True

    def _laplacian(self, psi, out) -> None:
        """Apply the Laplacian for the current grid."""
        if self.grid_type == "periodic":
            self._laplacian_periodic(psi, out)
        elif self.grid_type == "closed-interval":
            self._laplacian_closed_interval_neumann(psi, out)
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}")

    def _laplacian_periodic(self, psi, out) -> None:
        """Apply the periodic Laplacian stencil."""
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        xp = self.xp
        out[:] = (
            xp.roll(psi, -1, axis=0) - 2 * psi + xp.roll(psi, 1, axis=0)
        ) * inv_dy2 + (
            xp.roll(psi, -1, axis=1) - 2 * psi + xp.roll(psi, 1, axis=1)
        ) * inv_dx2

    def _laplacian_closed_interval_neumann(self, psi, out) -> None:
        """Apply the closed-interval Neumann Laplacian stencil."""
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        out.fill(0)

        if self.ny > 2 and self.nx > 2:
            out[1:-1, 1:-1] = (
                psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]
            ) * inv_dy2 + (
                psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]
            ) * inv_dx2

        if self.nx > 2:
            out[0, 1:-1] = (
                2 * (psi[1, 1:-1] - psi[0, 1:-1]) * inv_dy2
                + (psi[0, 2:] - 2 * psi[0, 1:-1] + psi[0, :-2]) * inv_dx2
            )
            out[-1, 1:-1] = (
                2 * (psi[-2, 1:-1] - psi[-1, 1:-1]) * inv_dy2
                + (psi[-1, 2:] - 2 * psi[-1, 1:-1] + psi[-1, :-2]) * inv_dx2
            )

        if self.ny > 2:
            out[1:-1, 0] = (
                psi[2:, 0] - 2 * psi[1:-1, 0] + psi[:-2, 0]
            ) * inv_dy2 + 2 * (psi[1:-1, 1] - psi[1:-1, 0]) * inv_dx2
            out[1:-1, -1] = (
                psi[2:, -1] - 2 * psi[1:-1, -1] + psi[:-2, -1]
            ) * inv_dy2 + 2 * (psi[1:-1, -2] - psi[1:-1, -1]) * inv_dx2

        out[0, 0] = (
            2 * (psi[1, 0] - psi[0, 0]) * inv_dy2
            + 2 * (psi[0, 1] - psi[0, 0]) * inv_dx2
        )
        out[0, -1] = (
            2 * (psi[1, -1] - psi[0, -1]) * inv_dy2
            + 2 * (psi[0, -2] - psi[0, -1]) * inv_dx2
        )
        out[-1, 0] = (
            2 * (psi[-2, 0] - psi[-1, 0]) * inv_dy2
            + 2 * (psi[-1, 1] - psi[-1, 0]) * inv_dx2
        )
        out[-1, -1] = (
            2 * (psi[-2, -1] - psi[-1, -1]) * inv_dy2
            + 2 * (psi[-1, -2] - psi[-1, -1]) * inv_dx2
        )

    def _rhs_into(self, psi, n_active, n_inactive, potential, out) -> None:
        """Write the right-hand side into the output buffer."""
        if self._fused_rhs is not None:
            self._laplacian(psi, self._lap)
            out[:] = self._fused_rhs(psi, self._lap, potential, n_active, n_inactive)
        else:
            self._laplacian(psi, self._lap)
            rho = self.xp.abs(psi) ** 2
            eff_energy = potential + self._g_C * rho + self._g_R * n_active + self._g_I * n_inactive
            gain_loss = (self._R * n_active - self._gamma_C) * 0.5
            kinetic = self._kinetic_coeff * self._lap
            out[:] = (
                self._minus_i_over_hbar * (kinetic + eff_energy * psi)
                + gain_loss * psi
                + self._kinetic_relaxation_eta * n_active * self._hbar_over_2m * self._lap
            )

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        dt = self._dt
        psi0 = state.psi
        self._ensure_buffers(psi0)

        res0 = reservoir.get_state()

        nR_0 = reservoir.get_active_density(res0)
        nI_0 = reservoir.get_inactive_density(res0)
        self._rhs_into(psi0, nR_0, nI_0, potential, self._k1_psi)
        k1_res = reservoir.get_derivatives(psi0, pump, res0)

        self._tmp_psi[:] = psi0
        self._tmp_psi += 0.5 * dt * self._k1_psi
        res2 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k1_res))
        nR_2 = reservoir.get_active_density(res2)
        nI_2 = reservoir.get_inactive_density(res2)
        self._rhs_into(self._tmp_psi, nR_2, nI_2, potential, self._k2_psi)
        k2_res = reservoir.get_derivatives(self._tmp_psi, pump, res2)

        self._tmp_psi[:] = psi0
        self._tmp_psi += 0.5 * dt * self._k2_psi
        res3 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k2_res))
        nR_3 = reservoir.get_active_density(res3)
        nI_3 = reservoir.get_inactive_density(res3)
        self._rhs_into(self._tmp_psi, nR_3, nI_3, potential, self._k3_psi)
        k3_res = reservoir.get_derivatives(self._tmp_psi, pump, res3)

        self._tmp_psi[:] = psi0
        self._tmp_psi += dt * self._k3_psi
        res4 = tuple(r + dt * k for r, k in zip(res0, k3_res))
        nR_4 = reservoir.get_active_density(res4)
        nI_4 = reservoir.get_inactive_density(res4)
        self._rhs_into(self._tmp_psi, nR_4, nI_4, potential, self._k4_psi)
        k4_res = reservoir.get_derivatives(self._tmp_psi, pump, res4)

        dt6 = dt / 6.0
        if self._fused_combine is not None:
            state.psi = self._fused_combine(
                psi0, self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi, dt6
            )
        else:
            state.psi = psi0 + dt6 * (
                self._k1_psi + 2.0 * self._k2_psi + 2.0 * self._k3_psi + self._k4_psi
            )

        state.psi = boundary_condition.after_step_action(state.psi)

        res_new = tuple(
            r0 + dt6 * (k1 + 2 * k2 + 2 * k3 + k4)
            for r0, k1, k2, k3, k4 in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )
        reservoir.set_state(res_new)
