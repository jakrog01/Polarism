"""Solver decay tests."""
import numpy as np
import pytest

from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.solver.rk4_fdm_solver import RK4FDMSolver
from polarism.solver.split_step_fft_solver import SplitStepFFTSolver

from tests._helpers import (
    GridCfg,
    NoBoundaryCondition,
    Recorder,
    State,
    initial_uniform,
    make_config,
    make_grid,
    make_physics_default,
    make_reservoir_cfg,
    potential_zero,
    pump_zero,
)


@pytest.mark.parametrize("solver_cls", [RK4FDMSolver, SplitStepFFTSolver])
@pytest.mark.parametrize("reservoir_cls", [SingleReservoir, DoubleReservoir])
def test_zero_pump_exponential_decay(solver_cls, reservoir_cls, output_dir):
    """Test that zero pump exponential decay."""
    grid = make_grid(GridCfg(nx=64, ny=64, lx=200.0, ly=200.0))
    physics = make_physics_default(g_C=0.0, g_R=0.0, R=0.0)
    cfg = make_config(dt=0.02, physics=physics)

    solver = solver_cls(cfg, grid)
    reservoir = reservoir_cls(make_reservoir_cfg(False), physics, grid)
    bc = NoBoundaryCondition()

    psi0 = initial_uniform(grid, amp=1e-3)
    state = State(psi0.copy())

    V = potential_zero(grid.X)
    P = pump_zero(grid.X)

    rec = Recorder()
    t = 0.0
    norm0 = float(np.sum(np.abs(state.psi) ** 2))

    for _ in range(1200):
        solver.step(V, P, reservoir, bc, state)
        t += cfg.solver.dt

        norm = float(np.sum(np.abs(state.psi) ** 2))
        expected = norm0 * np.exp(-physics.gamma_C * t)
        rel_err = abs(norm - expected) / (abs(expected) + 1e-30)
        rec.add(t, norm=norm, expected_norm=expected, rel_err_norm=rel_err)

    summary = dict(
        solver=solver_cls.__name__,
        reservoir=reservoir_cls.__name__,
        final_rel_err=rec.metrics["rel_err_norm"][-1],
        max_rel_err=max(rec.metrics["rel_err_norm"]),
    )

    rec.plot(
        output_dir / "decay.png",
        title=f"{solver_cls.__name__} + {reservoir_cls.__name__}: decay vs analytic",
        y_keys=["norm", "expected_norm"],
        y_log=True,
    )
    rec.plot(
        output_dir / "decay_error.png",
        title=f"{solver_cls.__name__} + {reservoir_cls.__name__}: relative error",
        y_keys=["rel_err_norm"],
        y_log=True,
    )

    assert summary["max_rel_err"] < 0.05
