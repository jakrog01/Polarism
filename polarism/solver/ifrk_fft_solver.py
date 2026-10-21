"""Fixed-step Lawson integrating-factor Runge-Kutta solver."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.config.batch_physics import BatchPhysics
from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver
from polarism.solver.tableaus import TABLEAUS

if TYPE_CHECKING:
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid_2d import SimulationGrid2D
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


@register_solver("ifrk-fft-cuda")
class IFRKFFTCudaSolver(AbstractSolver):
    """Stage-coupled spectral Lawson RK with a CPU and plugin-compatible path."""

    def __init__(self, config: Config, grid: SimulationGrid2D, batch_physics=None) -> None:
        super().__init__(config)
        self.batch_physics = BatchPhysics(config.physics, batch_physics, self.xp, getattr(config.solver, "precision", "double"))
        if grid.boundary != "periodic":
            raise ValueError(
                f"{config.solver.method} requires grid_type='periodic', got '{grid.boundary}'. "
                "For open boundaries use a wide CAP absorber on a periodic grid."
            )
        self.tableau = TABLEAUS[getattr(self, "tableau_name", config.solver.tableau)]
        self._precision = config.solver.precision
        self._real_dtype = real_dtype(self.xp, self._precision)
        self._complex_dtype = complex_dtype(self.xp, self._precision)
        self._scalar = np.float32 if self._precision == "single" else np.float64
        self._k_squared = self.xp.asarray(grid.k_squared, dtype=self._real_dtype)
        self._minus_k_squared = -self._k_squared
        phase_rate = -self.batch_physics.get("hbar") / (2.0 * self.batch_physics.get("m_eff"))
        self._phase_rate = (self._scalar(phase_rate) if self.batch_physics.size is None
                            else self.xp.asarray(phase_rate, dtype=self._real_dtype))
        self._hbar = self.batch_physics.get("hbar")
        self._hbar_over_2m = self.batch_physics.get("hbar") / (2.0 * self.batch_physics.get("m_eff"))
        self._g_C = self.batch_physics.get("g_C")
        self._g_R = self.batch_physics.get("g_R")
        self._g_I = self.batch_physics.get("g_I")
        self._R = self.batch_physics.get("R")
        self._gamma_C = self.batch_physics.get("gamma_C")
        self._eta = self.batch_physics.get("kinetic_relaxation_eta")
        self._a = tuple(tuple(float(v) for v in row) for row in self.tableau.A)
        self._b = tuple(float(v) for v in self.tableau.b)
        self._c = tuple(float(v) for v in self.tableau.c)
        self._grid = grid
        self._fused = None

    def _cuda(self, reservoir):
        if getattr(self.xp, "__name__", "") != "cupy":
            return None
        if self._fused is None:
            from polarism.solver.ifrk_cuda_fused import FusedIFRK
            self._fused = FusedIFRK(self, self._grid)
        return self._fused if self._fused.eligible(reservoir) else None

    def _phase(self, fraction: float, h: float):
        angle = self._phase_rate * self._scalar(h * fraction) * self._k_squared
        return self.xp.exp(1j * angle).astype(self._complex_dtype, copy=False)

    def _rhs_psi(self, psi, res_state, potential, reservoir, rho=None, psi_k=None):
        xp = self.xp
        n_active = reservoir.get_active_density(res_state)
        n_inactive = reservoir.get_inactive_density(res_state)
        density = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag
        eff_energy = potential + self._g_C * density + self._g_R * n_active + self._g_I * n_inactive
        gain_loss = 0.5 * (self._R * n_active - self._gamma_C)
        rhs = (-1j / self._hbar) * eff_energy * psi + gain_loss * psi
        if self.batch_physics.any_nonzero("kinetic_relaxation_eta"):
            if psi_k is None:
                psi_k = xp.fft.fft2(psi, axes=(-2, -1))
            lap_psi = xp.fft.ifft2(self._minus_k_squared * psi_k, axes=(-2, -1))
            rhs = rhs + self._eta * n_active * self._hbar_over_2m * lap_psi
        return rhs.astype(self._complex_dtype, copy=False)

    def _can_skip_after_step(self, boundary_condition: BoundaryCondition) -> bool:
        absorption = getattr(boundary_condition, "absorption", None)
        return bool(getattr(absorption, "after_step_is_noop", False))

    def _generic_step(self, potential, pump, reservoir, boundary_condition, state,
                      h: float | None = None) -> None:
        xp = self.xp
        h = self.config.solver.dt if h is None else h
        psi0_k = state.psi_k
        psi0 = state.psi if psi0_k is None else None
        if psi0_k is None:
            psi0_k = xp.fft.fft2(psi0, axes=(-2, -1)).astype(self._complex_dtype, copy=False)
        res0 = reservoir.get_state()
        pump_at = {}
        f = []
        derivatives = []
        for i, ci in enumerate(self._c):
            if i == 0 and psi0 is not None:
                psi_i = psi0
                psi_i_k = psi0_k
            else:
                psi_i_k = self._phase(ci, h) * psi0_k
                for j in range(i):
                    aij = self._a[i][j]
                    if aij:
                        psi_i_k += self._scalar(h * aij) * self._phase(ci - self._c[j], h) * f[j]
                psi_i_k = psi_i_k.astype(self._complex_dtype, copy=False)
                psi_i = xp.fft.ifft2(psi_i_k, axes=(-2, -1)).astype(self._complex_dtype, copy=False)
            res_i = []
            for field, r0 in enumerate(res0):
                value = r0.copy()
                for j in range(i):
                    aij = self._a[i][j]
                    if aij:
                        value += self._scalar(h * aij) * derivatives[j][field]
                res_i.append(value.astype(self._real_dtype, copy=False))
            res_i = tuple(res_i)
            rho = psi_i.real * psi_i.real + psi_i.imag * psi_i.imag
            if ci not in pump_at:
                pump_at[ci] = evaluate_pump(pump, state.t + ci * h)
            rhs = self._rhs_psi(psi_i, res_i, potential, reservoir, rho, psi_i_k)
            f.append(xp.fft.fft2(rhs, axes=(-2, -1)).astype(self._complex_dtype, copy=False))
            derivatives.append(reservoir.get_derivatives(psi_i, pump_at[ci], res_i, rho))
        psi_new_k = self._phase(1.0, h) * psi0_k
        for j, bj in enumerate(self._b):
            if bj:
                psi_new_k += self._scalar(h * bj) * self._phase(1.0 - self._c[j], h) * f[j]
        psi_new_k = psi_new_k.astype(self._complex_dtype, copy=False)
        if self._can_skip_after_step(boundary_condition):
            state.psi = None
            state.psi_k = psi_new_k
        else:
            state.psi = boundary_condition.after_step_action(
                xp.fft.ifft2(psi_new_k, axes=(-2, -1)).astype(self._complex_dtype, copy=False)
            )
            state.psi_k = xp.fft.fft2(state.psi, axes=(-2, -1)).astype(self._complex_dtype, copy=False)
        res_new = []
        for field, r0 in enumerate(res0):
            value = r0.copy()
            for j, bj in enumerate(self._b):
                if bj:
                    value += self._scalar(h * bj) * derivatives[j][field]
            res_new.append(xp.maximum(value, 0).astype(self._real_dtype, copy=False))
        reservoir.set_state(tuple(res_new))

    def step(self, potential, pump, reservoir: AbstractReservoir,
             boundary_condition: BoundaryCondition, state: SimulationState) -> None:
        self.batch_physics.check(state.psi_k if state._psi is None else state.psi)
        fused = self._cuda(reservoir)
        if fused is None:
            self._generic_step(potential, pump, reservoir, boundary_condition, state)
        else:
            fused.step(potential, pump, reservoir, boundary_condition, state)

    def advance(self, potential, pump, reservoir, boundary_condition, state,
                start_step: int, n_steps: int) -> None:
        self.batch_physics.check(state.psi_k if state._psi is None else state.psi)
        if n_steps <= 0:
            return
        fused = self._cuda(reservoir)
        if fused is None or not self._can_skip_after_step(boundary_condition):
            super().advance(potential, pump, reservoir, boundary_condition,
                            state, start_step, n_steps)
        else:
            fused.advance(potential, pump, reservoir, boundary_condition,
                          state, start_step, n_steps)
