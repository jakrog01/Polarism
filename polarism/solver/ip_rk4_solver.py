"""Interaction-picture RK4 solver."""
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


@register_solver("ip-rk4")
class IPRK4Solver(AbstractSolver):
    """Solve the model with interaction-picture RK4."""
    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the iprk 4 solver."""
        super().__init__(config)
        dt = self.config.solver.dt
        hbar = self.config.physics.hbar
        m_eff = self.config.physics.m_eff

        grid_type = getattr(config.grid, "grid_type", "periodic")
        self._use_dct = grid_type == "closed-interval"

        if self._use_dct and hasattr(self.xp, "cuda"):
            import warnings

            warnings.warn(
                "ip-rk4: closed-interval grid uses SciPy DCT "
                "with host-device copies on every step. "
                "This is CPU-bound and not suitable for GPU production runs. "
                "For GPU-native spectral evolution use 'ifrk4-fft-cuda' "
                "with a periodic grid and CAP absorber.",
                UserWarning,
                stacklevel=2,
            )

        self._eta = getattr(config.physics, "kinetic_relaxation_eta", 0.0)

        if self._use_dct:
            nx, ny = grid.nx, grid.ny
            dx, dy = grid.dx, grid.dy
            nx_idx = self.xp.arange(nx, dtype=self.xp.float64)
            ny_idx = self.xp.arange(ny, dtype=self.xp.float64)
            lambda_x = 4.0 / (dx**2) * self.xp.sin(self.xp.pi * nx_idx / (2 * nx)) ** 2
            lambda_y = 4.0 / (dy**2) * self.xp.sin(self.xp.pi * ny_idx / (2 * ny)) ** 2
            LX, LY = self.xp.meshgrid(lambda_x, lambda_y, indexing="xy")
            self._L_eig = LX + LY
        else:
            self._L_eig = grid.k_squared

        self._L = -1j * hbar * self._L_eig / (2 * m_eff)

        self._exp_L_half = self.xp.exp(self._L * (dt / 2))
        self._exp_L_full = self.xp.exp(self._L * dt)
        self._exp_L_neg_half = self.xp.exp(-self._L * (dt / 2))
        self._exp_L_neg_full = self.xp.exp(-self._L * dt)
        self._hbar_over_2m = hbar / (2.0 * m_eff)

        self._dctn = None
        self._idctn = None

    def _load_dct(self):
        """Load DCT helpers for closed-interval grids."""
        if self._dctn is None:
            try:
                from scipy.fft import dctn, idctn

                self._dctn = dctn
                self._idctn = idctn
            except ImportError:
                raise ImportError(
                    "scipy is required for ip-rk4 solver with closed-interval grids."
                )

    def _forward(self, psi):
        """Transform psi to spectral space."""
        if self._use_dct:
            self._load_dct()
            if hasattr(self.xp, "asnumpy"):
                psi_np = self.xp.asnumpy(psi)
            else:
                psi_np = psi
            re_k = self._dctn(psi_np.real, type=2, norm="ortho")
            im_k = self._dctn(psi_np.imag, type=2, norm="ortho")
            result = re_k + 1j * im_k
            if hasattr(self.xp, "asnumpy"):
                return self.xp.asarray(result)
            return result
        return self.xp.fft.fft2(psi)

    def _inverse(self, psi_k):
        """Transform from spectral space back to real space."""
        if self._use_dct:
            self._load_dct()
            if hasattr(self.xp, "asnumpy"):
                psi_k_np = self.xp.asnumpy(psi_k)
            else:
                psi_k_np = psi_k
            re = self._idctn(psi_k_np.real, type=2, norm="ortho")
            im = self._idctn(psi_k_np.imag, type=2, norm="ortho")
            result = re + 1j * im
            if hasattr(self.xp, "asnumpy"):
                return self.xp.asarray(result)
            return result
        return self.xp.fft.ifft2(psi_k)

    def _nonlinear_rhs(self, psi, nR, nI, potential, psi_k=None):
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
        rhs = (-1j / hbar) * eff_energy * psi + gain_loss * psi

        if self._eta != 0.0:
            if psi_k is None:
                psi_k = self._forward(psi)
            lap_psi = self._inverse(-self._L_eig * psi_k)
            rhs = rhs + self._eta * nR * self._hbar_over_2m * lap_psi

        return rhs

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
        res0 = reservoir.get_state()

        psi0_k = self._forward(psi0)
        nR_0 = reservoir.get_active_density(res0)
        nI_0 = reservoir.get_inactive_density(res0)
        N1 = self._nonlinear_rhs(psi0, nR_0, nI_0, potential, psi0_k)
        N1_k = self._forward(N1)
        k1_u = N1_k
        k1_res = reservoir.get_derivatives(psi0, pump, res0)

        u_mid = psi0_k + (dt / 2) * k1_u
        psi_mid_k = self._exp_L_half * u_mid
        psi_mid = self._inverse(psi_mid_k)
        res_mid = tuple(r + 0.5 * dt * k for r, k in zip(res0, k1_res))
        nR_mid = reservoir.get_active_density(res_mid)
        nI_mid = reservoir.get_inactive_density(res_mid)
        N2 = self._nonlinear_rhs(psi_mid, nR_mid, nI_mid, potential, psi_mid_k)
        N2_k = self._forward(N2)
        k2_u = self._exp_L_neg_half * N2_k
        k2_res = reservoir.get_derivatives(psi_mid, pump, res_mid)

        u_mid2 = psi0_k + (dt / 2) * k2_u
        psi_mid2_k = self._exp_L_half * u_mid2
        psi_mid2 = self._inverse(psi_mid2_k)
        res_mid2 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k2_res))
        nR_mid2 = reservoir.get_active_density(res_mid2)
        nI_mid2 = reservoir.get_inactive_density(res_mid2)
        N3 = self._nonlinear_rhs(psi_mid2, nR_mid2, nI_mid2, potential, psi_mid2_k)
        N3_k = self._forward(N3)
        k3_u = self._exp_L_neg_half * N3_k
        k3_res = reservoir.get_derivatives(psi_mid2, pump, res_mid2)

        u_end = psi0_k + dt * k3_u
        psi_end_k = self._exp_L_full * u_end
        psi_end = self._inverse(psi_end_k)
        res_end = tuple(r + dt * k for r, k in zip(res0, k3_res))
        nR_end = reservoir.get_active_density(res_end)
        nI_end = reservoir.get_inactive_density(res_end)
        N4 = self._nonlinear_rhs(psi_end, nR_end, nI_end, potential, psi_end_k)
        N4_k = self._forward(N4)
        k4_u = self._exp_L_neg_full * N4_k
        k4_res = reservoir.get_derivatives(psi_end, pump, res_end)

        u_new = psi0_k + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        psi_new_k = self._exp_L_full * u_new
        state.psi = self._inverse(psi_new_k)

        state.psi = boundary_condition.after_step_action(state.psi)

        res_new = tuple(
            r0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            for r0, k1, k2, k3, k4 in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )
        reservoir.set_state(tuple(self.xp.maximum(x, 0) for x in res_new))
