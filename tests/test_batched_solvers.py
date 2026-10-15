"""A leading batch axis represents independent simulations for every solver."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters, Config
from polarism.grid.create_grid import create_grid
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver
from polarism.solver.solver_registry import available_solvers

CPU_SOLVERS = ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "split-step-fft",
               "ip-rk4", "etd-rk2", "ifrk-fft-cuda", "ifrk4-fft-cuda"]
GPU_SOLVERS = ["rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda"]


def _config(method, model, absorption, grid_type="periodic"):
    cfg = Config()
    cfg.grid.nx, cfg.grid.ny = 9, 8
    cfg.grid.lx, cfg.grid.ly = 9.0, 8.0
    cfg.grid.grid_type = grid_type
    cfg.solver.method = method
    cfg.solver.dt = 0.0125
    cfg.solver.precision = "double"
    cfg.solver.cuda_graphs = False
    cfg.reservoir.reservoir_type = model
    cfg.boundary_condition.absorption = absorption
    cfg.boundary_condition.strength = 0.4
    return cfg


def _run(cfg, amplitudes, batched, xp, *, advance=False, pump_style="array"):
    grid = create_grid(cfg.grid)
    batch_size = len(amplitudes) if batched else None
    state = SimulationState(grid, eps=0.05, seed=13, batch_size=batch_size)
    if batched:
        state.psi = state.psi * xp.asarray(amplitudes)[:, None, None]
    else:
        state.psi = state.psi * amplitudes[0]
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid,
                                 batch_size=batch_size)
    solver = create_solver(cfg, grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt)
    potential = (0.03 * xp.cos(grid.X * 0.4) + bc.before_step_action())
    profile = 1 + 0.1 * xp.sin(grid.Y * 0.6)
    amplitude = xp.asarray(amplitudes)[:, None, None] if batched else amplitudes[0]

    if pump_style == "array":
        def pump(t):
            return amplitude * profile * (0.4 + 0.1 * t)
    elif pump_style == "shared":
        def pump(t):
            return profile * (0.4 + 0.1 * t)
    else:
        class SeparablePump:
            def __init__(self):
                if pump_style == "multi":
                    self.terms = ((amplitude * profile, lambda t: 0.4 + 0.1 * t),
                                  (0.2 * amplitude * xp.cos(grid.X * 0.3),
                                   lambda t: 0.3 + 0.05 * t))
                elif pump_style == "vector":
                    self.terms = ((profile, lambda t: amplitude * (0.4 + 0.1 * t)),)
                else:
                    self.terms = ((profile, lambda t: 0.4 + 0.1 * t),)

            def __call__(self, t):
                result = 0
                for envelope, coefficient in self.terms:
                    value = coefficient(t)
                    result = result + envelope * value
                return result

            def separable_term(self):
                return self.terms[0] if len(self.terms) == 1 else None

            def separable_terms(self):
                return self.terms

        pump = SeparablePump()

    if advance:
        solver.advance(potential, pump, reservoir, bc, state, 0, 2)
    else:
        for k in range(2):
            solver.step(potential, pump, reservoir, bc, state)
            state.t = (k + 1) * cfg.solver.dt
    return xp.asarray(state.psi), tuple(xp.asarray(f) for f in reservoir.get_state())


def _compare(method, model, absorption, xp, *, advance=False, grid_type="periodic",
             pump_style="array", graphs=False, laplacian="five-point"):
    cfg = _config(method, model, absorption, grid_type)
    cfg.solver.cuda_graphs = graphs
    cfg.solver.laplacian = laplacian
    amplitudes = (0.7, 1.1, 1.8)
    actual_psi, actual_fields = _run(cfg, amplitudes, True, xp, advance=advance,
                                     pump_style=pump_style)
    references = [_run(cfg, (a,), False, xp, advance=advance, pump_style=pump_style)
                  for a in amplitudes]
    np.testing.assert_allclose(xp.asnumpy(actual_psi) if xp is not np else actual_psi,
                               np.stack([xp.asnumpy(r[0]) if xp is not np else r[0]
                                         for r in references]), rtol=2e-12, atol=2e-12)
    for field_index, field in enumerate(actual_fields):
        np.testing.assert_allclose(xp.asnumpy(field) if xp is not np else field,
                                   np.stack([xp.asnumpy(r[1][field_index]) if xp is not np
                                             else r[1][field_index] for r in references]),
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("method", CPU_SOLVERS)
@pytest.mark.parametrize("model", ["single", "double"])
@pytest.mark.parametrize("absorption", ["cap", "mask"])
def test_cpu_solver_batch_matches_individual(method, model, absorption):
    _compare(method, model, absorption, np)


@pytest.mark.parametrize("method", ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda"])
def test_closed_interval_batch(method):
    _compare(method, "double", "mask", np, grid_type="closed-interval")


def test_batch_construction_and_seed():
    cfg = _config("rk4-fdm", "double", "cap")
    grid = create_grid(cfg.grid)
    for mode in ("legacy_positive_uniform", "complex_gaussian_zero_mean",
                 "filtered_complex_gaussian"):
        kwargs = {"seed": 9, "mode": mode}
        if mode == "filtered_complex_gaussian":
            kwargs["k_cutoff_um"] = 2.0
        single = SimulationState(grid, **kwargs)
        batch = SimulationState(grid, batch_size=3, **kwargs)
        assert batch.psi.shape == (3, grid.ny, grid.nx)
        for member in batch.psi:
            np.testing.assert_array_equal(member, single.psi)
    for model in ("single", "double", "quadratic-double"):
        cfg.reservoir.reservoir_type = model
        reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid, batch_size=3)
        for field in reservoir.get_state():
            assert field.shape == (3, grid.ny, grid.nx)
            assert np.count_nonzero(field) == 0
        reservoir.reset()
        assert all(field.shape == (grid.ny, grid.nx) for field in reservoir.get_state())
    with pytest.raises(ValueError, match="batch_size"):
        SimulationState(grid, batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        create_reservoir(cfg.reservoir, cfg.physics, grid, batch_size=0)


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
@pytest.mark.parametrize("model", ["single", "double"])
@pytest.mark.parametrize("absorption", ["cap", "mask"])
def test_gpu_solver_batch_matches_individual(method, model, absorption):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare(method, model, absorption, cp)


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
def test_gpu_batched_advance(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare(method, "single", "cap", cp, advance=True)


def test_registered_solvers_are_covered():
    assert set(available_solvers) == set(CPU_SOLVERS) | {"split-step-fft-cuda"}


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda"])
@pytest.mark.parametrize("pump_style", ["shared", "scalar", "vector", "multi"])
def test_gpu_pump_modes_and_graphs(method, pump_style):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare(method, "double", "cap", cp, advance=True,
             pump_style=pump_style, graphs=True)


@pytest.mark.parametrize("method", CPU_SOLVERS)
def test_quadratic_reservoir_batch(method):
    _compare(method, "quadratic-double", "cap", np)


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
def test_gpu_quadratic_reservoir_batch(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare(method, "quadratic-double", "cap", cp)


@pytest.mark.gpu
def test_gpu_closed_interval_rk4_batch():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "double", "mask", cp, grid_type="closed-interval")


@pytest.mark.parametrize("grid_type", ["periodic", "closed-interval"])
def test_reservoir_laplacian_uses_spatial_axes(grid_type):
    from polarism.reservoir.laplacian import real_laplacian
    rng = np.random.default_rng(4)
    field = rng.random((3, 8, 9))
    actual = real_laplacian(field, np, 0.7, 1.1, grid_type)
    expected = np.stack([real_laplacian(member, np, 0.7, 1.1, grid_type)
                         for member in field])
    np.testing.assert_array_equal(actual, expected)


def test_rk4_cuda_nine_point_batch_fallback():
    _compare("rk4-cuda", "double", "mask", np,
             laplacian="isotropic-9pt")


@pytest.mark.gpu
def test_gpu_rk4_cuda_nine_point_batch_fallback():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "double", "mask", cp,
             laplacian="isotropic-9pt")


@pytest.mark.parametrize("method", CPU_SOLVERS)
def test_cpu_shared_pump_batch(method):
    _compare(method, "single", "cap", np, pump_style="shared")


@pytest.mark.gpu
def test_gpu_rk4_cuda_shared_pump_batch():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "single", "cap", cp, pump_style="shared")


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
def test_gpu_solver_accepts_strided_psi(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    cfg = _config(method, "single", "cap")
    grid = create_grid(cfg.grid)
    state = SimulationState(grid, eps=0.05, seed=13)
    state.psi = cp.asfortranarray(state.psi)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt)
    create_solver(cfg, grid).step(bc.before_step_action(), lambda t: 0.4 * grid.X ** 0,
                                  reservoir, bc, state)
    assert bool(cp.all(cp.isfinite(state.psi)))


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
def test_gpu_repeated_solver_instances_agree(method):
    """Fresh solver instances in one process must reproduce the same result."""
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    cfg = _config(method, "single", "cap")
    cfg.grid.nx, cfg.grid.ny = 64, 63
    cfg.grid.lx, cfg.grid.ly = 64.0, 63.0
    results = [cp.asnumpy(_run(cfg, (1.1,), False, cp)[0]) for _ in range(8)]
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
