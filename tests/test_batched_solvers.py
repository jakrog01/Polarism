"""A leading batch axis represents independent simulations for every solver."""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import fields
import hashlib

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters, Config
from polarism.config.simulation_parameters import PhysicsConstants
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


def _run(cfg, amplitudes, batched, xp, *, advance=False, pump_style="array",
         potential_style="shared"):
    grid = create_grid(cfg.grid)
    batch_size = len(amplitudes) if batched else None
    state = SimulationState(grid, eps=0.05, seed=13, batch_size=batch_size,
                            precision=cfg.solver.precision)
    if batched:
        state.psi = state.psi * xp.asarray(amplitudes)[:, None, None]
    else:
        state.psi = state.psi * amplitudes[0]
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid,
                                 batch_size=batch_size, precision=cfg.solver.precision)
    solver = create_solver(cfg, grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt)
    potential = (0.03 * xp.cos(grid.X * 0.4) + bc.before_step_action())
    if potential_style == "batched":
        scale = xp.asarray(amplitudes)[:, None, None] if batched else amplitudes[0]
        potential = scale * potential
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
                elif pump_style in ("vector", "vector_term_only"):
                    coefficient_amplitude = xp.asarray(amplitudes) if batched else amplitude
                    self.terms = ((profile, lambda t: coefficient_amplitude *
                                   (0.4 + 0.1 * t)),)
                elif pump_style == "vector_multi":
                    coefficient_amplitude = xp.asarray(amplitudes) if batched else amplitude
                    self.terms = ((profile, lambda t: coefficient_amplitude *
                                   (0.4 + 0.1 * t)),
                                  (0.2 * amplitude * xp.cos(grid.X * 0.3),
                                   lambda t: 0.3 + 0.05 * t))
                else:
                    self.terms = ((profile, lambda t: 0.4 + 0.1 * t),)

            def __call__(self, t):
                result = 0
                for envelope, coefficient in self.terms:
                    value = coefficient(t)
                    if xp.ndim(value):
                        value = value[:, None, None]
                    result = result + envelope * value
                return result

            def separable_term(self):
                return self.terms[0] if len(self.terms) == 1 else None

            def separable_terms(self):
                return self.terms

        pump = SeparablePump()
        if pump_style == "vector_term_only":
            full_pump = pump
            class TermOnlyPump:
                def __call__(self, t):
                    return full_pump(t)

                def separable_term(self):
                    return full_pump.separable_term()

            pump = TermOnlyPump()

    if advance:
        solver.advance(potential, pump, reservoir, bc, state, 0, 2)
    else:
        for k in range(2):
            solver.step(potential, pump, reservoir, bc, state)
            state.t = (k + 1) * cfg.solver.dt
    return xp.asarray(state.psi), tuple(xp.asarray(f) for f in reservoir.get_state())


def _compare(method, model, absorption, xp, *, advance=False, grid_type="periodic",
             pump_style="array", graphs=False, laplacian="five-point",
             potential_style="shared", precision="double"):
    cfg = _config(method, model, absorption, grid_type)
    cfg.solver.cuda_graphs = graphs
    cfg.solver.laplacian = laplacian
    cfg.solver.precision = precision
    amplitudes = (0.7, 1.1, 1.8)
    actual_psi, actual_fields = _run(cfg, amplitudes, True, xp, advance=advance,
                                     pump_style=pump_style,
                                     potential_style=potential_style)
    references = [_run(cfg, (a,), False, xp, advance=advance,
                       pump_style=pump_style, potential_style=potential_style)
                  for a in amplitudes]
    tolerance = 2e-5 if precision == "single" else 2e-12
    np.testing.assert_allclose(xp.asnumpy(actual_psi) if xp is not np else actual_psi,
                               np.stack([xp.asnumpy(r[0]) if xp is not np else r[0]
                                         for r in references]), rtol=tolerance, atol=tolerance)
    for field_index, field in enumerate(actual_fields):
        np.testing.assert_allclose(xp.asnumpy(field) if xp is not np else field,
                                   np.stack([xp.asnumpy(r[1][field_index]) if xp is not np
                                             else r[1][field_index] for r in references]),
                                   rtol=tolerance, atol=tolerance)


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


@pytest.mark.parametrize("laser_type", ["uniform", "continuous-gaussian",
                                        "continuous-exp", "continuous-exp-length",
                                        "pulse-gaussian"])
@pytest.mark.parametrize("definition", ["peak_amplitude", "pulse_energy"])
def test_builtin_batch_power_coefficient_matches_lasers(laser_type, definition):
    from polarism.config.simulation_parameters import LaserParameters
    from polarism.laser.abstract_laser import AbstractLaser
    from polarism.laser import LaserFactory
    from polarism.laser.laser_registy import available_lasers

    X, Y = np.meshgrid(np.arange(4.0), np.arange(4.0))
    lasers = [LaserFactory.create_laser(
        LaserParameters(laser_type=laser_type, delay=i * 0.03,
                        P0=0.4 + i * 0.2, Pmax=1.2 + i * 0.2,
                        sigma_time=0.1 + i * 0.01,
                        pulse_separation=0.6 + i * 0.1, n_pulses=i + 1,
                        power_definition=definition), X, Y)[0]
        for i in range(3)]
    times = np.array([0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0])
    actual = available_lasers[laser_type].batch_power_coefficient(lasers, times)
    expected = np.array([[laser.power_coefficient(t) for laser in lasers]
                         for t in times])
    assert actual.shape == (len(times), len(lasers))
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(AbstractLaser.batch_power_coefficient(lasers, times),
                                  expected)


def test_pump_field_separable_member_coefficient():
    from polarism.config.simulation_parameters import LaserParameters
    from polarism.laser.pump_field import PumpField
    from polarism.laser.uniform_laser import UniformLaser

    grid = create_grid(_config("rk4-fdm", "single", "cap").grid)
    laser = UniformLaser(LaserParameters(laser_type="uniform"), grid.X, grid.Y)
    laser.power_coefficient = lambda t: np.array([0.4 + t, 0.7 + t])
    pump = PumpField([laser], grid, "double")
    envelope, coefficient = pump.separable_terms()[0]
    assert coefficient(0.2).shape == (2,)
    np.testing.assert_array_equal(pump(0.2),
                                  envelope[None, :, :] * coefficient(0.2)[:, None, None])


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


def test_rk4_cuda_nine_point_cpu_batch():
    _compare("rk4-cuda", "double", "mask", np,
             laplacian="isotropic-9pt")


@pytest.mark.gpu
def test_gpu_rk4_cuda_nine_point_batch_kernel():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "double", "mask", cp,
             laplacian="isotropic-9pt")


@pytest.mark.parametrize("method", CPU_SOLVERS)
def test_cpu_shared_pump_batch(method):
    _compare(method, "single", "cap", np, pump_style="shared")


@pytest.mark.parametrize("method", CPU_SOLVERS)
@pytest.mark.parametrize("pump_style", ["vector", "vector_term_only",
                                        "vector_multi", "multi"])
def test_cpu_member_pump_coefficient_is_bitwise_separate(method, pump_style):
    cfg = _config(method, "double", "cap")
    amplitudes = (0.7, 1.1, 1.8)
    actual = _run(cfg, amplitudes, True, np, pump_style=pump_style)
    expected = [_run(cfg, (amplitude,), False, np, pump_style=pump_style)
                for amplitude in amplitudes]
    np.testing.assert_array_equal(actual[0], np.stack([item[0] for item in expected]))
    for index, field in enumerate(actual[1]):
        np.testing.assert_array_equal(field,
                                      np.stack([item[1][index] for item in expected]))


@pytest.mark.gpu
def test_gpu_rk4_cuda_shared_pump_batch():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "single", "cap", cp, pump_style="shared")


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("pump_style", ["vector", "vector_term_only",
                                        "vector_multi"])
def test_gpu_member_pump_coefficient_matches_separate(method, graphs, pump_style):
    """FFT batches allow rounding; rk4 uses member-indexed versions of the same stages."""
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare(method, "double", "cap", cp, advance=True,
             pump_style=pump_style, graphs=graphs)


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("graphs", [False, True])
def test_gpu_rk4_member_axis_potential_and_graph(model, graphs):
    """The staged and tiled variants use the same arithmetic as separate runs."""
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", model, "cap", cp, advance=True, graphs=graphs,
             pump_style="vector", potential_style="batched")


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_gpu_rk4_tiled_member_axis(graphs):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    _compare("rk4-cuda", "double", "cap", cp, advance=True, graphs=graphs,
             pump_style="vector_multi", potential_style="batched",
             precision="single")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_gpu_rk4_batch_uses_cuda_kernels(monkeypatch, precision):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    from polarism.solver.rk4_cuda_solver import RK4CudaSolver

    def fail(*args, **kwargs):
        raise AssertionError("rk4-cuda batched state entered CuPy stages")

    monkeypatch.setattr(RK4CudaSolver, "_step_cpu", fail)
    _compare("rk4-cuda", "double", "cap", cp, pump_style="vector",
             precision=precision)


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


_PHYSICS_FIELDS = tuple(field.name for field in fields(PhysicsConstants)
                        if field.type is float)


def _physics_values(cfg, *, diffusion=True):
    values = {}
    for name in _PHYSICS_FIELDS:
        base = getattr(cfg.physics, name)
        if name == "kinetic_relaxation_eta":
            values[name] = np.array([0.0, 0.008])
        elif name.startswith("reservoir_diffusion"):
            values[name] = np.array([0.0, 0.004 if diffusion else 0.0])
        elif name == "init_eps":
            values[name] = np.array([base, base * 1.2])
        else:
            values[name] = np.array([base * 0.9 + 0.0001, base * 1.1 + 0.0002])
    return values


def _run_physics(cfg, values, xp, batched, *, advance=False):
    grid = create_grid(cfg.grid)
    size = len(next(iter(values.values()))) if batched else None
    state = SimulationState(grid, eps=0.05, seed=13, batch_size=size,
                            precision=cfg.solver.precision)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid,
                                 precision=cfg.solver.precision, batch_size=size,
                                 batch_physics=values if batched else None)
    solver = create_solver(cfg, grid, batch_physics=values if batched else None)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt,
                           batch_physics=values if batched else None)
    potential = 0.03 * xp.cos(grid.X * 0.4) + bc.before_step_action()
    profile = 1 + 0.1 * xp.sin(grid.Y * 0.6)
    class Pump:
        def __call__(self, t):
            return profile * (0.4 + 0.1 * t)

        def separable_term(self):
            return profile, lambda t: 0.4 + 0.1 * t

    pump = Pump()
    if advance:
        solver.advance(potential, pump, reservoir, bc, state, 0, 2)
    else:
        for k in range(2):
            solver.step(potential, pump, reservoir, bc, state)
            state.t = (k + 1) * cfg.solver.dt
    return state.psi, reservoir.get_state()


@pytest.mark.parametrize("method", CPU_SOLVERS)
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
def test_cpu_per_member_physics_matches_separate(method, model):
    cfg = _config(method, model, "cap")
    values = _physics_values(cfg)
    actual = _run_physics(cfg, values, np, True)
    expected = []
    for member in range(2):
        member_cfg = _config(method, model, "cap")
        for name, sequence in values.items():
            setattr(member_cfg.physics, name, float(sequence[member]))
        expected.append(_run_physics(member_cfg, {}, np, False))
    np.testing.assert_array_equal(actual[0], np.stack([item[0] for item in expected]))
    for index, field in enumerate(actual[1]):
        np.testing.assert_array_equal(field, np.stack([item[1][index] for item in expected]))


def test_batch_physics_validation():
    cfg = _config("rk4-fdm", "single", "cap")
    grid = create_grid(cfg.grid)
    with pytest.raises(ValueError, match="non-float"):
        create_solver(cfg, grid, {"init_mode": [1, 2]})
    with pytest.raises(ValueError, match="length"):
        create_solver(cfg, grid, {"hbar": [0.6, 0.7], "m_eff": [0.2]})
    solver = create_solver(cfg, grid, {"hbar": [0.6, 0.7]})
    state = SimulationState(grid)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt)
    with pytest.raises(ValueError, match="state batch size"):
        solver.step(bc.before_step_action(), lambda t: grid.X * 0, reservoir, bc, state)


def test_plugin_without_batch_physics_rejected():
    from polarism.solver.solver_registry import available_solvers
    from polarism.reservoir.reservoir_registry import available_reservoirs
    cfg = _config("rk4-fdm", "single", "cap")
    grid = create_grid(cfg.grid)
    available_solvers["legacy-test"] = lambda config, grid: None
    available_reservoirs["legacy-test"] = lambda config, physics, grid, precision: None
    try:
        cfg.solver.method = "legacy-test"
        with pytest.raises(ValueError, match="Solver 'legacy-test'"):
            create_solver(cfg, grid, {"hbar": [0.6, 0.7]})
        cfg.reservoir.reservoir_type = "legacy-test"
        with pytest.raises(ValueError, match="Reservoir 'legacy-test'"):
            create_reservoir(cfg.reservoir, cfg.physics, grid,
                             batch_physics={"hbar": [0.6, 0.7]})
    finally:
        del available_solvers["legacy-test"]
        del available_reservoirs["legacy-test"]


def test_unbatched_cuda_kernel_sources_match_release():
    from polarism.solver.rk4_cuda_solver import build_kernel_source_1d
    from polarism.solver.rk4_cuda_solver import build_tiled_kernel_source
    from polarism.solver.split_step_fft_cuda_solver import kernel_source
    from polarism.solver.ifrk_cuda_fused import fused_source
    hashes = [hashlib.sha256(source.encode()).hexdigest() for source in
              build_kernel_source_1d("periodic", "double", "single")]
    assert hashes == [
        "aec7bb2786cf538bee940716e68dde05ff49eeba75a633d67d9dc21a6105eb14",
        "a1af56182692411167496467438de2d585ae5ccb57113db12572856bd5c07476",
        "0ccb24f0c4ce9d7cbcbbf667055a06d4ac9eb291dfc44e427b3dcf118232995a",
    ]
    tiled_hashes = [hashlib.sha256(build_tiled_kernel_source(
        "periodic", "double", "single", "five-point", 16, 16, True,
        graph=graph).encode()).hexdigest() for graph in (False, True)]
    assert tiled_hashes == [
        "5292865448d5633b27ed4f9d420bc81d67920de79e757159b1b4efa0904cc7eb",
        "c8137d0023569428c3f203678e71512d1397015098fb3dfa0e1f5b7844afa37c",
    ]
    assert hashlib.sha256(kernel_source("double", "single", physics_fields=None).encode()).hexdigest() == (
        "1f182a977c129097d13119453291930d551dd08b77abec66e52be9a80f733da8"
    )
    assert hashlib.sha256(fused_source("double", 0, 0, 4, 1, "k", "k", "sincos",
                                      physics_fields=None).encode()).hexdigest() == (
        "50016fa7eeb73ee973f60f4698625f75a1f10563c213470ca741e8c73ded8075"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("method", GPU_SOLVERS)
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
def test_gpu_per_member_physics_matches_separate(method, model):
    """FFT and rk4 kernel variants permit 2e-11 rounding against separate runs."""
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    import cupy as cp
    cfg = _config(method, model, "cap")
    cfg.solver.cuda_graphs = True
    values = _physics_values(cfg, diffusion=False)
    if method == "split-step-fft-cuda":
        values["kinetic_relaxation_eta"][:] = 0.0
    actual = _run_physics(cfg, values, cp, True, advance=True)
    expected = []
    for member in range(2):
        member_cfg = _config(method, model, "cap")
        member_cfg.solver.cuda_graphs = True
        for name, sequence in values.items():
            setattr(member_cfg.physics, name, float(sequence[member]))
        expected.append(_run_physics(member_cfg, {}, cp, False, advance=True))
    compare = (np.testing.assert_array_equal if method == "split-step-fft-cuda"
               else lambda left, right: np.testing.assert_allclose(
                   left, right, rtol=2e-11, atol=2e-11))
    compare(cp.asnumpy(actual[0]),
            np.stack([cp.asnumpy(item[0]) for item in expected]))
    for index, field in enumerate(actual[1]):
        compare(cp.asnumpy(field),
                np.stack([cp.asnumpy(item[1][index]) for item in expected]))
