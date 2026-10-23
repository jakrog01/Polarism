"""Interaction-picture RK4 solver."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from polarism.config.dtype_utils import complex_dtype, real_dtype
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
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


@register_solver("ip-rk4")
class IPRK4Solver(AbstractSolver):
    """Solve the model with interaction-picture RK4."""
    def __init__(self, config: Config, grid: SimulationGrid, batch_physics=None):
        """Set up the iprk 4 solver."""
        super().__init__(config)
        self._fft = self.xp.fft.fft if grid.ndim == 1 else self.xp.fft.fft2
        self._ifft = self.xp.fft.ifft if grid.ndim == 1 else self.xp.fft.ifft2
        self.batch_physics = BatchPhysics(config.physics, batch_physics, self.xp, getattr(config.solver, "precision", "double"), ndim=grid.ndim)
        dt = self.config.solver.dt
        hbar = self.batch_physics.get("hbar")
        m_eff = self.batch_physics.get("m_eff")

        self._eta = self.batch_physics.get("kinetic_relaxation_eta")

        precision = getattr(config.solver, "precision", "double")
        cdtype = complex_dtype(self.xp, precision)
        self._L_eig = grid.k_squared.astype(real_dtype(self.xp, precision))
        self._L = -1j * hbar * grid.k_squared / (2 * m_eff)

        # Propagators follow the solver precision so float32 runs stay float32.
        self._exp_L_half = self.xp.exp(self._L * (dt / 2)).astype(cdtype)
        self._exp_L_full = self.xp.exp(self._L * dt).astype(cdtype)
        self._exp_L_neg_half = self.xp.exp(-self._L * (dt / 2)).astype(cdtype)
        self._exp_L_neg_full = self.xp.exp(-self._L * dt).astype(cdtype)
        self._hbar_over_2m = hbar / (2.0 * m_eff)

    def _forward(self, psi):
        """Transform psi to spectral space."""
        return self._fft(psi)

    def _inverse(self, psi_k):
        """Transform from spectral space back to real space."""
        return self._ifft(psi_k)

    def _nonlinear_rhs(self, psi, nR, nI, potential, psi_k=None):
        """Compute the nonlinear right-hand side."""
        hbar = self.batch_physics.get("hbar")
        g_C = self.batch_physics.get("g_C")
        g_R = self.batch_physics.get("g_R")
        g_I = self.batch_physics.get("g_I")
        R = self.batch_physics.get("R")
        gamma_C = self.batch_physics.get("gamma_C")

        rho = self.xp.abs(psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR + g_I * nI
        gain_loss = (R * nR - gamma_C) / 2.0
        rhs = (-1j / hbar) * eff_energy * psi + gain_loss * psi

        if self.batch_physics.any_nonzero("kinetic_relaxation_eta"):
            if psi_k is None:
                psi_k = self._forward(psi)
            lap_psi = self._inverse(-self._L_eig * psi_k)
            rhs = rhs + self._eta * nR * self._hbar_over_2m * lap_psi

        return rhs

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
        t = state.t
        p0 = evaluate_pump(pump, t)
        p_half = evaluate_pump(pump, t + 0.5 * dt)
        p1 = evaluate_pump(pump, t + dt)

        psi0 = state.field
        res0 = reservoir.get_state()

        psi0_k = self._forward(psi0)
        nR_0 = reservoir.get_active_density(res0)
        nI_0 = reservoir.get_inactive_density(res0)
        N1 = self._nonlinear_rhs(psi0, nR_0, nI_0, potential, psi0_k)
        N1_k = self._forward(N1)
        k1_u = N1_k
        k1_res = reservoir.get_derivatives(psi0, p0, res0)

        u_mid = psi0_k + (dt / 2) * k1_u
        psi_mid_k = self._exp_L_half * u_mid
        psi_mid = self._inverse(psi_mid_k)
        res_mid = tuple(r + 0.5 * dt * k for r, k in zip(res0, k1_res))
        nR_mid = reservoir.get_active_density(res_mid)
        nI_mid = reservoir.get_inactive_density(res_mid)
        N2 = self._nonlinear_rhs(psi_mid, nR_mid, nI_mid, potential, psi_mid_k)
        N2_k = self._forward(N2)
        k2_u = self._exp_L_neg_half * N2_k
        k2_res = reservoir.get_derivatives(psi_mid, p_half, res_mid)

        u_mid2 = psi0_k + (dt / 2) * k2_u
        psi_mid2_k = self._exp_L_half * u_mid2
        psi_mid2 = self._inverse(psi_mid2_k)
        res_mid2 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k2_res))
        nR_mid2 = reservoir.get_active_density(res_mid2)
        nI_mid2 = reservoir.get_inactive_density(res_mid2)
        N3 = self._nonlinear_rhs(psi_mid2, nR_mid2, nI_mid2, potential, psi_mid2_k)
        N3_k = self._forward(N3)
        k3_u = self._exp_L_neg_half * N3_k
        k3_res = reservoir.get_derivatives(psi_mid2, p_half, res_mid2)

        u_end = psi0_k + dt * k3_u
        psi_end_k = self._exp_L_full * u_end
        psi_end = self._inverse(psi_end_k)
        res_end = tuple(r + dt * k for r, k in zip(res0, k3_res))
        nR_end = reservoir.get_active_density(res_end)
        nI_end = reservoir.get_inactive_density(res_end)
        N4 = self._nonlinear_rhs(psi_end, nR_end, nI_end, potential, psi_end_k)
        N4_k = self._forward(N4)
        k4_u = self._exp_L_neg_full * N4_k
        k4_res = reservoir.get_derivatives(psi_end, p1, res_end)

        u_new = psi0_k + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        psi_new_k = self._exp_L_full * u_new
        state.field = self._inverse(psi_new_k)

        state.field = boundary_condition.after_step_action(state.field)

        res_new = tuple(
            r0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            for r0, k1, k2, k3, k4 in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )
        reservoir.set_state(tuple(self.xp.maximum(x, 0) for x in res_new))
