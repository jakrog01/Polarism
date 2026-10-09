"""Finite-difference RK4 solver."""
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


@register_solver("rk4-fdm")
class RK4FDMSolver(AbstractSolver):
    """Solve the model with finite-difference RK4."""
    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the rk 4 fdm solver."""
        super().__init__(config)
        self.dx = grid.dx
        self.dy = grid.dy
        self.nx = grid.nx
        self.ny = grid.ny
        self.grid_type = getattr(config.grid, "grid_type", "periodic")
        self._laplacian_buf = None

    @staticmethod
    def _tuple_add(t1: tuple, t2: tuple, scalar: float) -> tuple:
        """Add two state tuples with a scale factor."""
        return tuple(a + scalar * b for a, b in zip(t1, t2))

    def _ensure_buffers(self, psi):
        """Allocate work buffers for the current field shape."""
        if (
            self._laplacian_buf is None
            or self._laplacian_buf.shape != psi.shape
            or self._laplacian_buf.dtype != psi.dtype
        ):
            self._laplacian_buf = self.xp.zeros_like(psi)

    def _laplacian_periodic(self, psi, out) -> None:
        """Apply the periodic Laplacian stencil."""
        out[:] = (
            self.xp.roll(psi, -1, axis=0) - 2 * psi + self.xp.roll(psi, 1, axis=0)
        ) / (self.dy**2) + (
            self.xp.roll(psi, -1, axis=1) - 2 * psi + self.xp.roll(psi, 1, axis=1)
        ) / (
            self.dx**2
        )

    def _laplacian_closed_interval_neumann(self, psi, out) -> None:
        """Apply the closed-interval Neumann Laplacian stencil."""
        out.fill(0)

        if self.ny > 2 and self.nx > 2:
            out[1:-1, 1:-1] = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (
                self.dy**2
            ) + (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (self.dx**2)

        if self.nx > 2:
            out[0, 1:-1] = 2 * (psi[1, 1:-1] - psi[0, 1:-1]) / (self.dy**2) + (
                psi[0, 2:] - 2 * psi[0, 1:-1] + psi[0, :-2]
            ) / (self.dx**2)
            out[-1, 1:-1] = 2 * (psi[-2, 1:-1] - psi[-1, 1:-1]) / (self.dy**2) + (
                psi[-1, 2:] - 2 * psi[-1, 1:-1] + psi[-1, :-2]
            ) / (self.dx**2)

        if self.ny > 2:
            out[1:-1, 0] = (psi[2:, 0] - 2 * psi[1:-1, 0] + psi[:-2, 0]) / (
                self.dy**2
            ) + 2 * (psi[1:-1, 1] - psi[1:-1, 0]) / (self.dx**2)
            out[1:-1, -1] = (psi[2:, -1] - 2 * psi[1:-1, -1] + psi[:-2, -1]) / (
                self.dy**2
            ) + 2 * (psi[1:-1, -2] - psi[1:-1, -1]) / (self.dx**2)

        out[0, 0] = 2 * (psi[1, 0] - psi[0, 0]) / (self.dy**2) + 2 * (
            psi[0, 1] - psi[0, 0]
        ) / (self.dx**2)
        out[0, -1] = 2 * (psi[1, -1] - psi[0, -1]) / (self.dy**2) + 2 * (
            psi[0, -2] - psi[0, -1]
        ) / (self.dx**2)
        out[-1, 0] = 2 * (psi[-2, 0] - psi[-1, 0]) / (self.dy**2) + 2 * (
            psi[-1, 1] - psi[-1, 0]
        ) / (self.dx**2)
        out[-1, -1] = 2 * (psi[-2, -1] - psi[-1, -1]) / (self.dy**2) + 2 * (
            psi[-1, -2] - psi[-1, -1]
        ) / (self.dx**2)

    def _laplacian(self, psi, out) -> None:
        """Apply the Laplacian for the current grid."""
        if self.grid_type == "periodic":
            self._laplacian_periodic(psi, out)
            return

        if self.grid_type == "closed-interval":
            self._laplacian_closed_interval_neumann(psi, out)
            return

        raise ValueError(f"Unknown grid_type: {self.grid_type}")

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

        psi0 = state.psi
        self._ensure_buffers(psi0)

        res0 = reservoir.get_state()

        k1_psi, k1_res = self._rhs(psi0, res0, potential, pump, reservoir)

        psi2 = psi0 + 0.5 * dt * k1_psi
        res2 = self._tuple_add(res0, k1_res, 0.5 * dt)
        k2_psi, k2_res = self._rhs(psi2, res2, potential, pump, reservoir)

        psi3 = psi0 + 0.5 * dt * k2_psi
        res3 = self._tuple_add(res0, k2_res, 0.5 * dt)
        k3_psi, k3_res = self._rhs(psi3, res3, potential, pump, reservoir)

        psi4 = psi0 + dt * k3_psi
        res4 = self._tuple_add(res0, k3_res, dt)
        k4_psi, k4_res = self._rhs(psi4, res4, potential, pump, reservoir)

        psi_new = psi0 + (dt / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
        psi_new = boundary_condition.after_step_action(psi_new)

        res_new = tuple(
            r0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            for r0, k1, k2, k3, k4 in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )

        state.psi = psi_new.astype(psi0.dtype, copy=False)
        reservoir.set_state(res_new)

    def _rhs(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        res_state: tuple,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
    ) -> tuple[Union[np.ndarray, cp.ndarray], tuple]:
        """Compute the right-hand side."""
        res_derivs = reservoir.get_derivatives(psi, pump, res_state)
        n_active = reservoir.get_active_density(res_state)
        n_inactive = reservoir.get_inactive_density(res_state)

        lap = self._laplacian_buf
        if lap is None:
            raise RuntimeError("Buffers not initialized")

        self._laplacian(psi, lap)

        kinetic = -(self.config.physics.hbar**2) / (2 * self.config.physics.m_eff) * lap

        eff_energy = (
            potential
            + self.config.physics.g_C * self.xp.abs(psi) ** 2
            + self.config.physics.g_R * n_active
            + getattr(self.config.physics, "g_I", 0.0) * n_inactive
        )
        gain_loss = (
            self.config.physics.R * n_active - self.config.physics.gamma_C
        ) / 2.0

        eta = getattr(self.config.physics, "kinetic_relaxation_eta", 0.0)
        hbar_over_2m = self.config.physics.hbar / (2.0 * self.config.physics.m_eff)
        dpsi_dt = (-1j / self.config.physics.hbar) * (
            kinetic + eff_energy * psi
        ) + gain_loss * psi + eta * n_active * hbar_over_2m * lap

        return dpsi_dt, res_derivs
