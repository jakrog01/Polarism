"""Split-step FFT solver."""
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


@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):
    """Solve the model with split-step FFT."""
    _kinetic_propagator_half: Union[np.ndarray, cp.ndarray]
    _use_dct: bool

    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the split step fft solver."""
        super().__init__(config)

        grid_type = getattr(config.grid, "grid_type", "periodic")
        self._use_dct = grid_type == "closed-interval"

        if self._use_dct:
            nx, ny = grid.nx, grid.ny
            dx, dy = grid.dx, grid.dy

            nx_idx = self.xp.arange(nx, dtype=self.xp.float64)
            ny_idx = self.xp.arange(ny, dtype=self.xp.float64)

            lambda_x = 4.0 / (dx**2) * self.xp.sin(self.xp.pi * nx_idx / (2 * nx)) ** 2
            lambda_y = 4.0 / (dy**2) * self.xp.sin(self.xp.pi * ny_idx / (2 * ny)) ** 2
            LX, LY = self.xp.meshgrid(lambda_x, lambda_y, indexing="xy")
            laplacian_eigenvalues = LX + LY

            self._kinetic_propagator_half = self.xp.exp(
                -1j
                * self.config.physics.hbar
                * laplacian_eigenvalues
                * (self.config.solver.dt / 2)
                / (2 * self.config.physics.m_eff)
            )
        else:
            self._kinetic_propagator_half = self.xp.exp(
                -1j
                * self.config.physics.hbar
                * grid.k_squared
                * (self.config.solver.dt / 2)
                / (2 * self.config.physics.m_eff)
            )

    def _kinetic_half_step_fft(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply the FFT half-step of the kinetic term."""
        psi_k = self.xp.fft.fft2(psi)
        psi_k *= self._kinetic_propagator_half
        return self.xp.fft.ifft2(psi_k)

    def _kinetic_half_step_dct(
        self, psi: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        """Apply the DCT half-step of the kinetic term."""
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            raise ImportError(
                "scipy is required for split-step-fft solver with closed-interval grids. "
                "Install with: pip install scipy"
            )

        if hasattr(self.xp, "asnumpy"):
            psi_np = self.xp.asnumpy(psi)
            prop_np = self.xp.asnumpy(self._kinetic_propagator_half)
            use_gpu = True
        else:
            psi_np = psi
            prop_np = self._kinetic_propagator_half
            use_gpu = False

        psi_re_dct = dctn(psi_np.real, type=2, norm="ortho")
        psi_im_dct = dctn(psi_np.imag, type=2, norm="ortho")

        psi_dct = psi_re_dct + 1j * psi_im_dct

        psi_dct *= prop_np

        psi_new_re = idctn(psi_dct.real, type=2, norm="ortho")
        psi_new_im = idctn(psi_dct.imag, type=2, norm="ortho")
        psi_new = psi_new_re + 1j * psi_new_im

        if use_gpu:
            return self.xp.asarray(psi_new)
        return psi_new

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
        hbar = self.config.physics.hbar
        g_C = self.config.physics.g_C
        g_R = self.config.physics.g_R
        R = self.config.physics.R
        gamma_C = self.config.physics.gamma_C

        nR = reservoir.get_reservoir_density()

        if self._use_dct:
            state.psi = self._kinetic_half_step_dct(state.psi)
        else:
            state.psi = self._kinetic_half_step_fft(state.psi)

        rho = self.xp.abs(state.psi) ** 2
        eff_energy = potential + g_C * rho + g_R * nR
        gain_loss = (R * nR - gamma_C) / 2.0
        state.psi = state.psi * self.xp.exp((-1j * eff_energy / hbar + gain_loss) * dt)

        if self._use_dct:
            state.psi = self._kinetic_half_step_dct(state.psi)
        else:
            state.psi = self._kinetic_half_step_fft(state.psi)

        state.psi = boundary_condition.after_step_action(state.psi)
        reservoir.step(dt, state.psi, pump)
