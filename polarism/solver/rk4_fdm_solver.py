"""Finite-difference RK4 solver."""
from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.solver.abstract_solver import AbstractSolver, ACCEPTED, SolverCapabilities
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid


@register_solver("rk4-fdm")
class RK4FDMSolver(AbstractSolver):
    """Solve the model with finite-difference RK4."""
    capabilities = SolverCapabilities(
        components=frozenset({1}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic", "closed-interval"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="quantitative", potential=True,
        kinetic_relaxation=True, dispersion=frozenset({"parabolic"}),
        coherent_drive=True, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='Reference FDM RK4. Stage-coupled reservoir (psi and reservoir advance through the same 4 RK4 stages). Works with all grid types.',
    )
    def __init__(self, config: Config, grid: SimulationGrid, model):
        """Set up the rk 4 fdm solver."""
        super().__init__(config, grid, model)

    @staticmethod
    def _tuple_add(t1: tuple, t2: tuple, scalar: float) -> tuple:
        """Add two state tuples with a scale factor."""
        return tuple(a + scalar * b for a, b in zip(t1, t2))

    def step(self, model, state, dt):
        """Advance the solver by one time step."""
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field)
        dt = self.config.solver.dt
        t = state.t

        psi0 = state.field
        if (model._laplacian_buf is None or model._laplacian_buf.shape != psi0.shape
                or model._laplacian_buf.dtype != psi0.dtype):
            model._laplacian_buf = self.xp.zeros_like(psi0)

        res0 = reservoir.get_state()

        k1_psi, k1_res = model.rhs(psi0, res0, t, None)

        psi2 = psi0 + 0.5 * dt * k1_psi
        res2 = self._tuple_add(res0, k1_res, 0.5 * dt)
        k2_psi, k2_res = model.rhs(psi2, res2, t + 0.5 * dt, None)

        psi3 = psi0 + 0.5 * dt * k2_psi
        res3 = self._tuple_add(res0, k2_res, 0.5 * dt)
        k3_psi, k3_res = model.rhs(psi3, res3, t + 0.5 * dt, None)

        psi4 = psi0 + dt * k3_psi
        res4 = self._tuple_add(res0, k3_res, dt)
        k4_psi, k4_res = model.rhs(psi4, res4, t + dt, None)

        psi_new = psi0 + (dt / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
        psi_new = boundary_condition.after_step_action(psi_new)

        res_new = tuple(
            r0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            for r0, k1, k2, k3, k4 in zip(res0, k1_res, k2_res, k3_res, k4_res)
        )

        state.field = psi_new.astype(psi0.dtype, copy=False)
        reservoir.set_state(res_new)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
