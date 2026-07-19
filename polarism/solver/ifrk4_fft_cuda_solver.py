"""GPU-native interaction-picture RK4 solver with cuFFT spectral kinetics."""
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


@register_solver("ifrk4-fft-cuda")
class IFRK4FFTCudaSolver(AbstractSolver):
    """GPU-native FFT interaction-picture RK4 solver.

    Parameters
    ----------
    config : Config
        Full simulation configuration.
    grid : SimulationGrid2D
        Must have ``grid_type == 'periodic'``.

    Raises
    ------
    ValueError
        If ``grid_type`` is not ``'periodic'``.

    Notes
    -----
    Requires a periodic grid.  Open-boundary problems should use
    ``grid_type: periodic`` with a wide CAP absorber.

    The kinetic operator is handled exactly through the integrating factor
    ``exp(L * dt)`` where ``L = -i*hbar*k^2 / (2*m_eff)``.  The nonlinear
    right-hand side—including the variable-coefficient energy-relaxation
    term ``eta * nR * laplacian(psi)``—is evaluated in real space and
    transformed by FFT at each stage.

    The quadratic-double reservoir fields ``(nR, nI)`` are advanced in the
    same RK4 stages as ``psi``, eliminating the O(dt) operator-splitting
    error present in split-step-based approaches.
    """

    def __init__(self, config: Config, grid: SimulationGrid2D) -> None:
        """Initialise precomputed propagators and physics constants."""
        super().__init__(config)

        grid_type = getattr(config.grid, "grid_type", "periodic")
        if grid_type != "periodic":
            raise ValueError(
                f"ifrk4-fft-cuda requires grid_type='periodic', got '{grid_type}'. "
                "For open-boundary problems model the absorbing region as a wide CAP "
                "absorber on a periodic grid."
            )

        dt = config.solver.dt
        hbar = config.physics.hbar
        m_eff = config.physics.m_eff

        self._L = -1j * hbar * grid.k_squared / (2.0 * m_eff)

        self._exp_L_half = self.xp.exp(self._L * (dt / 2.0))
        self._exp_L_full = self.xp.exp(self._L * dt)
        self._exp_L_neg_half = self.xp.exp(-self._L * (dt / 2.0))
        self._exp_L_neg_full = self.xp.exp(-self._L * dt)
        self._minus_k_squared = -grid.k_squared

        self._hbar = hbar
        self._hbar_over_2m = hbar / (2.0 * m_eff)
        self._g_C = config.physics.g_C
        self._g_R = config.physics.g_R
        self._g_I = getattr(config.physics, "g_I", 0.0)
        self._R = config.physics.R
        self._gamma_C = config.physics.gamma_C
        self._eta = getattr(config.physics, "kinetic_relaxation_eta", 0.0)

    def _rhs_psi(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        res_state: tuple,
        potential: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        rho: Union[np.ndarray, cp.ndarray] | None = None,
        psi_k: Union[np.ndarray, cp.ndarray] | None = None,
    ) -> Union[np.ndarray, cp.ndarray]:
        """Evaluate the nonlinear GPE right-hand side in real space."""
        xp = self.xp
        n_active = reservoir.get_active_density(res_state)
        n_inactive = reservoir.get_inactive_density(res_state)
        density = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag

        eff_energy = potential + self._g_C * density + self._g_R * n_active + self._g_I * n_inactive
        gain_loss = 0.5 * (self._R * n_active - self._gamma_C)
        rhs = (-1j / self._hbar) * eff_energy * psi + gain_loss * psi

        if self._eta != 0.0:
            if psi_k is None:
                psi_k = xp.fft.fft2(psi, axes=(-2, -1))
            lap_psi = xp.fft.ifft2(self._minus_k_squared * psi_k, axes=(-2, -1))
            rhs = rhs + self._eta * n_active * self._hbar_over_2m * lap_psi

        return rhs

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the system by one timestep with Lawson/IP-RK4."""
        xp = self.xp
        dt = self.config.solver.dt

        psi0 = state.psi
        res0 = reservoir.get_state()
        cached_psi_k = getattr(state, "psi_k", None)
        psi0_k = cached_psi_k if cached_psi_k is not None else xp.fft.fft2(
            psi0, axes=(-2, -1)
        )
        rho0 = psi0.real * psi0.real + psi0.imag * psi0.imag

        N1 = self._rhs_psi(psi0, res0, potential, reservoir, rho0, psi0_k)
        k1_psi = xp.fft.fft2(N1, axes=(-2, -1))
        k1_res = reservoir.get_derivatives(psi0, pump, res0, rho0)

        psi2_k = self._exp_L_half * (psi0_k + 0.5 * dt * k1_psi)
        psi2 = xp.fft.ifft2(psi2_k, axes=(-2, -1))
        res2 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k1_res))
        rho2 = psi2.real * psi2.real + psi2.imag * psi2.imag
        N2 = self._rhs_psi(psi2, res2, potential, reservoir, rho2, psi2_k)
        k2_psi = self._exp_L_neg_half * xp.fft.fft2(N2, axes=(-2, -1))
        k2_res = reservoir.get_derivatives(psi2, pump, res2, rho2)

        psi3_k = self._exp_L_half * (psi0_k + 0.5 * dt * k2_psi)
        psi3 = xp.fft.ifft2(psi3_k, axes=(-2, -1))
        res3 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k2_res))
        rho3 = psi3.real * psi3.real + psi3.imag * psi3.imag
        N3 = self._rhs_psi(psi3, res3, potential, reservoir, rho3, psi3_k)
        k3_psi = self._exp_L_neg_half * xp.fft.fft2(N3, axes=(-2, -1))
        k3_res = reservoir.get_derivatives(psi3, pump, res3, rho3)

        psi4_k = self._exp_L_full * (psi0_k + dt * k3_psi)
        psi4 = xp.fft.ifft2(psi4_k, axes=(-2, -1))
        res4 = tuple(r + dt * k for r, k in zip(res0, k3_res))
        rho4 = psi4.real * psi4.real + psi4.imag * psi4.imag
        N4 = self._rhs_psi(psi4, res4, potential, reservoir, rho4, psi4_k)
        k4_psi = self._exp_L_neg_full * xp.fft.fft2(N4, axes=(-2, -1))
        k4_res = reservoir.get_derivatives(psi4, pump, res4, rho4)

        psi_new_k = self._exp_L_full * (
            psi0_k + (dt / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi)
        )
        state.psi_k = psi_new_k
        if self._can_skip_after_step(boundary_condition):
            state.psi = None
        else:
            state.psi = boundary_condition.after_step_action(
                xp.fft.ifft2(psi_new_k, axes=(-2, -1))
            )
            state.invalidate_psi_k()

        res_new = tuple(
            xp.maximum(r0 + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d), 0.0)
            for r0, a, b, c, d in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )
        reservoir.set_state(res_new)

    def _can_skip_after_step(self, boundary_condition: BoundaryCondition) -> bool:
        absorption = getattr(boundary_condition, "absorption", None)
        return bool(getattr(absorption, "after_step_is_noop", False))
