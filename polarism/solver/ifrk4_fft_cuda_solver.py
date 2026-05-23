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
        self._g_C = config.physics.g_C
        self._g_R = config.physics.g_R
        self._R = config.physics.R
        self._gamma_C = config.physics.gamma_C
        self._eta = getattr(config.physics, "kinetic_relaxation_eta", 0.0)

    def _rhs_psi(
        self,
        psi: Union[np.ndarray, cp.ndarray],
        res_state: tuple,
        potential: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
    ) -> Union[np.ndarray, cp.ndarray]:
        """Evaluate the nonlinear GPE right-hand side in real space."""
        xp = self.xp
        n_active = reservoir.get_active_density(res_state)
        rho = xp.abs(psi) ** 2

        eff_energy = potential + self._g_C * rho + self._g_R * n_active
        gain_loss = 0.5 * (self._R * n_active - self._gamma_C)
        rhs = (-1j / self._hbar) * eff_energy * psi + gain_loss * psi

        if self._eta != 0.0:
            lap_psi = xp.fft.ifft2(self._minus_k_squared * xp.fft.fft2(psi))
            rhs = rhs + self._eta * n_active * lap_psi

        return rhs

    def _clamp_reservoir_state(self, state: tuple) -> tuple:
        """Clamp all reservoir density fields to non-negative values."""
        return tuple(self.xp.maximum(x, 0) for x in state)

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
        psi0_k = xp.fft.fft2(psi0)

        N1 = self._rhs_psi(psi0, res0, potential, reservoir)
        k1_psi = xp.fft.fft2(N1)
        k1_res = reservoir.get_derivatives(psi0, pump, res0)

        psi2_k = self._exp_L_half * (psi0_k + 0.5 * dt * k1_psi)
        psi2 = xp.fft.ifft2(psi2_k)
        res2 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k1_res))
        N2 = self._rhs_psi(psi2, res2, potential, reservoir)
        k2_psi = self._exp_L_neg_half * xp.fft.fft2(N2)
        k2_res = reservoir.get_derivatives(psi2, pump, res2)

        psi3_k = self._exp_L_half * (psi0_k + 0.5 * dt * k2_psi)
        psi3 = xp.fft.ifft2(psi3_k)
        res3 = tuple(r + 0.5 * dt * k for r, k in zip(res0, k2_res))
        N3 = self._rhs_psi(psi3, res3, potential, reservoir)
        k3_psi = self._exp_L_neg_half * xp.fft.fft2(N3)
        k3_res = reservoir.get_derivatives(psi3, pump, res3)

        psi4_k = self._exp_L_full * (psi0_k + dt * k3_psi)
        psi4 = xp.fft.ifft2(psi4_k)
        res4 = tuple(r + dt * k for r, k in zip(res0, k3_res))
        N4 = self._rhs_psi(psi4, res4, potential, reservoir)
        k4_psi = self._exp_L_neg_full * xp.fft.fft2(N4)
        k4_res = reservoir.get_derivatives(psi4, pump, res4)

        psi_new_k = self._exp_L_full * (
            psi0_k + (dt / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi)
        )
        state.psi = xp.fft.ifft2(psi_new_k)
        state.psi = boundary_condition.after_step_action(state.psi)

        res_new = tuple(
            r0 + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
            for r0, a, b, c, d in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )
        reservoir.set_state(self._clamp_reservoir_state(res_new))
