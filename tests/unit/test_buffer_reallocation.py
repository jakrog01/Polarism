"""Work-buffer caches recover from failed allocations and batch changes."""
import numpy as np
import pytest

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters, Config
from polarism.grid.create_grid import create_grid
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver


def _case(method, batch, xp):
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 16
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.solver.method = method
    cfg.solver.cuda_graphs = False
    cfg.solver.dt = 0.01
    cfg.boundary_condition.absorption = "cap"
    grid = create_grid(cfg.grid)
    state = SimulationState(grid, eps=0.02, seed=7, batch_size=batch)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid, batch_size=batch)
    boundary = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt)
    potential = xp.zeros((grid.ny, grid.nx), dtype=xp.complex128)

    def pump(t):
        return xp.full((grid.ny, grid.nx), 0.1, dtype=xp.float64)

    return create_solver(cfg, grid), (potential, pump, reservoir, boundary, state)


def _run(solver, args, method):
    if method == "split-step-fft-cuda":
        solver.advance(*args, 0, 1)
    else:
        solver.step(*args)
    state = args[-1]
    reservoir = args[2]
    return (compute_engine.xp.asnumpy(state.psi) if compute_engine.use_gpu else state.psi.copy(),
            *(compute_engine.xp.asnumpy(field) if compute_engine.use_gpu else field.copy()
              for field in reservoir.get_state()))


def _assert_matches_fresh(solver, args, method, batch, xp):
    actual = _run(solver, args, method)
    fresh, fresh_args = _case(method, batch, xp)
    expected = _run(fresh, fresh_args, method)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=2e-12, atol=2e-12)


def _fail_third_allocation(monkeypatch, xp):
    original = xp.empty
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise MemoryError("injected allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(xp, "empty", fail_once)


def test_fused_fdm_recovers_after_allocation_failure_cpu(monkeypatch):
    solver, args = _case("rk4-fdm-fused", 4, np)
    _run(solver, args, "rk4-fdm-fused")
    _, small_args = _case("rk4-fdm-fused", 2, np)
    with monkeypatch.context() as patch:
        _fail_third_allocation(patch, np)
        with pytest.raises(MemoryError, match="injected allocation failure"):
            _run(solver, small_args, "rk4-fdm-fused")
    assert not solver._buf_initialized
    assert solver._lap is None
    _assert_matches_fresh(solver, small_args, "rk4-fdm-fused", 2, np)


def test_fused_fdm_batch_shape_changes_match_fresh_cpu():
    solver, args = _case("rk4-fdm-fused", 4, np)
    _assert_matches_fresh(solver, args, "rk4-fdm-fused", 4, np)
    for batch in (2, 4):
        _, args = _case("rk4-fdm-fused", batch, np)
        _assert_matches_fresh(solver, args, "rk4-fdm-fused", batch, np)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["ifrk-fft-cuda", "split-step-fft-cuda", "rk4-fdm-fused"])
def test_gpu_recovers_after_allocation_failure(method, monkeypatch):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    solver, args = _case(method, 4, cp)
    _run(solver, args, method)
    _, small_args = _case(method, 2, cp)
    with monkeypatch.context() as patch:
        _fail_third_allocation(patch, cp)
        with pytest.raises(MemoryError, match="injected allocation failure"):
            if method == "ifrk-fft-cuda":
                solver._fused._activate_shape(small_args[-1])
                solver._fused._ensure((0, 1))
            elif method == "split-step-fft-cuda":
                solver._activate_shape(small_args[-1])
                solver._ensure_advance_buffers(1)
            else:
                solver._ensure_buffers(small_args[-1].psi)
    if method == "ifrk-fft-cuda":
        assert solver._fused._buffers is None
        assert solver._fused._buffer_shape is None
    elif method == "split-step-fft-cuda":
        assert solver._advance_buffers is None
        assert solver._advance_shape is None
    else:
        assert not solver._buf_initialized
        assert solver._lap is None
    _assert_matches_fresh(solver, small_args, method, 2, cp)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["ifrk-fft-cuda", "split-step-fft-cuda", "rk4-fdm-fused"])
def test_gpu_batch_shape_changes_match_fresh(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    solver, args = _case(method, 4, cp)
    _assert_matches_fresh(solver, args, method, 4, cp)
    for batch in (2, 4):
        _, args = _case(method, batch, cp)
        _assert_matches_fresh(solver, args, method, batch, cp)
