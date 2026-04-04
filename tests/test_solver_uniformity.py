"""Solver uniformity tests."""
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
    grad_central_roll,
    laplacian_roll,
    make_config,
    make_grid,
    make_physics_default,
    make_reservoir_cfg,
    max_abs,
    potential_zero,
    pump_uniform,
    uniformity_error,
)


@pytest.mark.parametrize("solver_cls", [RK4FDMSolver, SplitStepFFTSolver])
@pytest.mark.parametrize("reservoir_cls", [SingleReservoir, DoubleReservoir])
def test_uniform_pump_does_not_create_spatial_structure(
    solver_cls, reservoir_cls
):
    """Test that uniform pump does not create spatial structure."""
    grid = make_grid(GridCfg(nx=64, ny=64, lx=200.0, ly=200.0))
    physics = make_physics_default()
    cfg = make_config(dt=0.02, physics=physics)

    solver = solver_cls(cfg, grid)
    reservoir = reservoir_cls(make_reservoir_cfg(False), physics, grid)
    bc = NoBoundaryCondition()

    psi0 = (1e-6 + 0j) * np.ones((grid.nx, grid.ny), dtype=np.complex128)
    state = State(psi0.copy())

    V = potential_zero(grid.X)
    P = pump_uniform(0.01, grid.X)

    rec = Recorder()
    t = 0.0

    for _ in range(5000):
        solver.step(V, P, reservoir, bc, state)
        t += cfg.solver.dt

        ux = uniformity_error(state.psi)
        gx = max_abs(grad_central_roll(state.psi, grid.dx, axis=0))
        gy = max_abs(grad_central_roll(state.psi, grid.dy, axis=1))
        lap = max_abs(laplacian_roll(state.psi, grid.dx, grid.dy))

        if hasattr(reservoir, "get_reservoir_density"):
            nR = reservoir.get_reservoir_density()
            rec.add(
                t,
                psi_uniformity=ux,
                grad_x_max=gx,
                grad_y_max=gy,
                lap_max=lap,
                n_mean=float(np.mean(nR)),
            )
        else:
            nA, nI = reservoir.get_reservoir_densities()
            rec.add(
                t,
                psi_uniformity=ux,
                grad_x_max=gx,
                grad_y_max=gy,
                lap_max=lap,
                nA_mean=float(np.mean(nA)),
                nI_mean=float(np.mean(nI)),
            )

    summary = dict(
        solver=solver_cls.__name__,
        reservoir=reservoir_cls.__name__,
        max_uniformity=max(rec.metrics["psi_uniformity"]),
        max_grad_x=max(rec.metrics["grad_x_max"]),
        max_grad_y=max(rec.metrics["grad_y_max"]),
        max_lap=max(rec.metrics["lap_max"]),
    )

    assert summary["max_uniformity"] < 1e-10
    assert summary["max_grad_x"] < 1e-10
    assert summary["max_grad_y"] < 1e-10
    assert summary["max_lap"] < 1e-10
