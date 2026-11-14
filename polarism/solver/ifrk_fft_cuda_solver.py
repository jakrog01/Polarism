"""Fixed-step Lawson integrating-factor Runge-Kutta solver."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver, ACCEPTED, SolverCapabilities
from polarism.solver.solver_registry import register_solver
from polarism.solver.tableaus import TABLEAUS

if TYPE_CHECKING:
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


@register_solver("ifrk-fft-cuda")
class IFRKFFTCudaSolver(AbstractSolver):
    """Stage-coupled spectral Lawson RK with a CPU and plugin-compatible path."""
    capabilities = SolverCapabilities(
        components=frozenset({1}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="quantitative", potential=True,
        kinetic_relaxation=True, dispersion=frozenset({"parabolic"}), terms=True,
        coherent_drive=True, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='FFT interaction-picture RK solver with selectable tableau. Stage-coupled reservoir and cuFFT fused stages for supported GPU cases. Evaluates kinetic_relaxation_eta in nonlinear RHS. Spectral production path for periodic grids; use a wide CAP absorber for open boundaries.',
    )

    def __init__(self, config: Config, grid: SimulationGrid, model) -> None:
        super().__init__(config, grid, model)
        self._model_features = model
        self._ndim = grid.ndim
        self._fft = self.xp.fft.fft if grid.ndim == 1 else self.xp.fft.fft2
        self._ifft = self.xp.fft.ifft if grid.ndim == 1 else self.xp.fft.ifft2
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
        n_active = reservoir.get_active_density(res_state)
        n_inactive = reservoir.get_inactive_density(res_state)
        density = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag
        eff_energy = potential + self._g_C * density + self._g_R * n_active + self._g_I * n_inactive
        gain_loss = 0.5 * (self._R * n_active - self._gamma_C)
        rhs = (-1j / self._hbar) * eff_energy * psi + gain_loss * psi
        if self.batch_physics.any_nonzero("kinetic_relaxation_eta"):
            if psi_k is None:
                psi_k = self._fft(psi, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)}))
            lap_psi = self._ifft(self._minus_k_squared * psi_k, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)}))
            rhs = rhs + self._eta * n_active * self._hbar_over_2m * lap_psi
        return rhs.astype(self._complex_dtype, copy=False)

    def _can_skip_after_step(self, boundary_condition: BoundaryCondition) -> bool:
        absorption = getattr(boundary_condition, "absorption", None)
        return bool(getattr(absorption, "after_step_is_noop", False))

    def _generic_step(self, potential, pump, reservoir, boundary_condition, state,
                      h: float | None = None) -> None:
        xp = self.xp
        h = self.config.solver.dt if h is None else h
        psi0_k = state.field_k
        psi0 = state.field if psi0_k is None else None
        if psi0_k is None:
            psi0_k = self._fft(psi0, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)})).astype(self._complex_dtype, copy=False)
        res0 = reservoir.get_state()
        pump_at = {}
        f = []
        derivatives = []
        for i, ci in enumerate(self._c):
            stage_time = state.t + ci * h
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
                psi_i = self._ifft(psi_i_k, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)})).astype(self._complex_dtype, copy=False)
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
                pump_at[ci] = evaluate_pump(pump, stage_time)
            stage_potential = (self._model_features.potential_at(stage_time)
                               if self._model_features.time_dependent_potential else potential)
            rhs = self._rhs_psi(psi_i, res_i, stage_potential, reservoir, rho, psi_i_k)
            if self._model_features.terms:
                rhs = rhs + self._model_features.extra_rhs(psi_i, res_i, stage_time)
            if self._model_features.drive is not None:
                rhs = rhs + self._model_features.drive_contribution(psi_i, stage_time)
            f.append(self._fft(rhs, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)})).astype(self._complex_dtype, copy=False))
            derivatives.append(reservoir.get_derivatives(psi_i, pump_at[ci], res_i, rho))
        psi_new_k = self._phase(1.0, h) * psi0_k
        for j, bj in enumerate(self._b):
            if bj:
                psi_new_k += self._scalar(h * bj) * self._phase(1.0 - self._c[j], h) * f[j]
        psi_new_k = psi_new_k.astype(self._complex_dtype, copy=False)
        if self._can_skip_after_step(boundary_condition):
            state.field = None
            state.field_k = psi_new_k
        else:
            state.field = boundary_condition.after_step_action(
                self._ifft(psi_new_k, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)})).astype(self._complex_dtype, copy=False)
            )
            state.field_k = self._fft(state.field, **({"axis": -1} if self._ndim == 1 else {"axes": (-2, -1)})).astype(self._complex_dtype, copy=False)
        res_new = []
        for field, r0 in enumerate(res0):
            value = r0.copy()
            for j, bj in enumerate(self._b):
                if bj:
                    value += self._scalar(h * bj) * derivatives[j][field]
            res_new.append(xp.maximum(value, 0).astype(self._real_dtype, copy=False))
        reservoir.set_state(tuple(res_new))

    def step(self, model, state, dt):
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        fused = self._cuda(reservoir)
        if fused is None:
            self._generic_step(potential, pump, reservoir, boundary_condition, state)
        else:
            fused.step(model, state)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED

    def advance(self, model, state, n_steps: int) -> None:
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        start_step = state.step
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        if n_steps <= 0:
            return
        fused = self._cuda(reservoir)
        if fused is None or not self._can_skip_after_step(boundary_condition):
            super().advance(model, state, n_steps)
        else:
            fused.advance(model, state, n_steps)


@register_solver("ifrk4-fft-cuda")
class IFRK4FFTCudaSolver(IFRKFFTCudaSolver):
    """Use the RK4 tableau regardless of the generic solver's setting."""
    capabilities = IFRKFFTCudaSolver.capabilities
    tableau_name = "rk4"
