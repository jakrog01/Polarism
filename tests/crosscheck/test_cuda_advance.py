"""CUDA chunk results must match repeated public step calls."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import has_cuda_device
from polarism.config.simulation_parameters import Config, LaserParameters
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.pump_field import PumpField
from polarism.simulation_controller import SimulationController


def _assert_matches_steps(actual, expected, method, precision, long_run=False):
    actual = actual.get()
    expected = expected.get()
    if method == "rk4-cuda":
        np.testing.assert_array_equal(actual, expected)
        return
    tolerance = ((2e-5 if long_run else 2e-6) if precision == "single"
                 else (5e-13 if long_run else 5e-14))
    np.testing.assert_allclose(actual, expected, rtol=0,
                               atol=tolerance * np.max(np.abs(expected)))


def _multi_lasers(controller, precision):
    X, Y = controller.grid.X, controller.grid.Y
    configs = [
        LaserParameters(laser_type="pulse-gaussian", P0=0.3, Pmax=0.3,
                        x0=-1.0, delay=delay, sigma_time=width)
        for delay, width in ((0.0, 0.02), (0.012, 0.03), (0.025, 0.04))
    ]
    configs.append(LaserParameters(laser_type="continuous-gaussian", P0=0.1, x0=1.0))
    controller.lasers = [LaserFactory.create_laser(cfg, X, Y, precision)[0]
                         for cfg in configs]
    controller.pump_field = PumpField(controller.lasers, controller.grid, precision)


def _assert_multi_matches(actual, expected, precision, long_run=False):
    tolerance = ((2e-5 if long_run else 2e-6) if precision == "single"
                 else (5e-13 if long_run else 5e-14))
    for field, reference in zip((actual.state.psi, *actual.reservoir.get_state()),
                                (expected.state.psi, *expected.reservoir.get_state())):
        np.testing.assert_allclose(field.get(), reference.get(), rtol=0,
                                   atol=tolerance * max(np.max(np.abs(reference.get())),
                                                        1e-30))


@pytest.mark.gpu
@pytest.mark.parametrize("method,tableau,path", [
    ("rk4-cuda", "rk4", "tiled"), ("rk4-cuda", "rk4", "staged"),
    ("split-step-fft-cuda", "rk4", "auto"),
    ("ifrk-fft-cuda", "rk4", "auto"), ("ifrk-fft-cuda", "dp5", "auto"),
])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("graphs", [True, False])
def test_multi_laser_advance_matches_steps(method, tableau, path, precision, graphs):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.tableau = tableau
    cfg.solver.precision = precision
    cfg.solver.dt = 0.01
    cfg.solver.cuda_graphs = graphs
    cfg.physics.init_seed = 11
    cfg.reservoir.reservoir_type = "single"
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    for controller in (fast, old):
        _multi_lasers(controller, precision)
    if path == "tiled":
        if fast.solver._tile_shape is None:
            pytest.skip("No supported CUDA tile on this device")
        fast.solver._use_tiled = True
    elif path == "staged":
        fast.solver._use_tiled = False
    args = (fast.potential, fast.pump_field, fast.reservoir,
            fast.boundary_condition, fast.state)
    for start, length in ((0, 1), (1, 5)):
        fast.solver.advance(*args, start, length)
        for k in range(start, start + length):
            old.solver.step(old.potential, old.pump_field, old.reservoir,
                            old.boundary_condition, old.state)
            old.state.t = (k + 1) * cfg.solver.dt
        _assert_multi_matches(fast, old, precision)
        assert fast.state.t == old.state.t
    owner = fast.solver._fused if method == "ifrk-fft-cuda" else fast.solver
    if graphs:
        assert owner._graph_cache
        assert not owner._graphs_failed


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda"])
def test_multi_laser_graph_chunks_match_steps(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.tableau = "rk4"
    cfg.solver.precision = "double"
    cfg.solver.dt = 0.01
    cfg.physics.init_seed = 12
    cfg.reservoir.reservoir_type = "single"
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    for controller in (fast, old):
        _multi_lasers(controller, "double")
    args = (fast.potential, fast.pump_field, fast.reservoir,
            fast.boundary_condition, fast.state)
    for start, length in ((0, 150), (150, 37)):
        fast.solver.advance(*args, start, length)
        for k in range(start, start + length):
            old.solver.step(old.potential, old.pump_field, old.reservoir,
                            old.boundary_condition, old.state)
            old.state.t = (k + 1) * cfg.solver.dt
        _assert_multi_matches(fast, old, "double", long_run=True)
    owner = fast.solver._fused if method == "ifrk-fft-cuda" else fast.solver
    assert owner._graph_cache
    assert not owner._graphs_failed


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
def test_advance_matches_steps(method, precision, model):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.precision = precision
    cfg.solver.dt = 0.01
    cfg.physics.init_seed = 7
    cfg.reservoir.reservoir_type = model
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.02
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    fast.pump_field.lasers = fast.lasers
    old.pump_field.lasers = old.lasers
    args = (fast.potential, fast.pump_field, fast.reservoir,
            fast.boundary_condition, fast.state)
    fast.solver.advance(*args, 0, 6)
    psi_reference = fast.state.psi
    field_references = fast.reservoir.get_state()
    saved_psi = psi_reference.copy()
    saved_fields = tuple(field.copy() for field in field_references)
    fast.solver.advance(*args, 6, 2)
    for k in range(8):
        old.solver.step(old.potential, old.pump_field, old.reservoir,
                        old.boundary_condition, old.state)
        old.state.t = (k + 1) * cfg.solver.dt
    _assert_matches_steps(fast.state.psi, old.state.psi, method, precision)
    for field, other in zip(fast.reservoir.get_state(), old.reservoir.get_state()):
        _assert_matches_steps(field, other, method, precision)
    assert fast.state.t == old.state.t
    np.testing.assert_array_equal(psi_reference.get(), saved_psi.get())
    for reference, saved in zip(field_references, saved_fields):
        np.testing.assert_array_equal(reference.get(), saved.get())


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_multiple_graph_lengths_and_odd_remainder_match_steps(method, precision):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.precision = precision
    cfg.solver.dt = 0.01
    cfg.physics.init_seed = 10
    cfg.reservoir.reservoir_type = "single"
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.02
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    for controller in (fast, old):
        controller.pump_field.lasers = controller.lasers

    args = (fast.potential, fast.pump_field, fast.reservoir,
            fast.boundary_condition, fast.state)
    for start, length in ((0, 150), (150, 37)):
        fast.solver.advance(*args, start, length)
        for k in range(start, start + length):
            old.solver.step(old.potential, old.pump_field, old.reservoir,
                            old.boundary_condition, old.state)
            old.state.t = (k + 1) * cfg.solver.dt
        _assert_matches_steps(fast.state.psi, old.state.psi, method, precision,
                              long_run=True)
        for field, other in zip(fast.reservoir.get_state(), old.reservoir.get_state()):
            _assert_matches_steps(field, other, method, precision, long_run=True)
        assert fast.state.t == old.state.t
    if not fast.solver._graphs_failed:
        assert {64, 22, 36}.issubset(fast.solver._graph_cache)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda"])
@pytest.mark.parametrize("case", ["nonseparable", "mask", "graphs_off"])
def test_advance_fallback_matches_steps(method, case):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.precision = "single"
    cfg.solver.dt = 0.01
    cfg.physics.init_seed = 8
    cfg.reservoir.reservoir_type = "single"
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.02
    if case == "mask":
        cfg.boundary_condition.absorption = "mask"
    if case == "graphs_off":
        cfg.solver.cuda_graphs = False
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    for controller in (fast, old):
        if case == "nonseparable":
            controller.lasers.append(LaserFactory.create_laser(
                cfg.laser, controller.grid.X, controller.grid.Y,
                cfg.solver.precision,
            )[0])
            controller.pump_field = PumpField(controller.lasers, controller.grid,
                                              cfg.solver.precision)
        controller.pump_field.lasers = controller.lasers
    fast.solver.advance(fast.potential, fast.pump_field, fast.reservoir,
                        fast.boundary_condition, fast.state, 0, 4)
    for k in range(4):
        old.solver.step(old.potential, old.pump_field, old.reservoir,
                        old.boundary_condition, old.state)
        old.state.t = (k + 1) * cfg.solver.dt
    _assert_matches_steps(fast.state.psi, old.state.psi, method, "single")
    for field, other in zip(fast.reservoir.get_state(), old.reservoir.get_state()):
        _assert_matches_steps(field, other, method, "single")


@pytest.mark.gpu
@pytest.mark.parametrize("method,case", [
    ("rk4-cuda", "closed-interval"),
    ("rk4-cuda", "isotropic-9pt"),
    ("rk4-cuda", "cap"),
    ("split-step-fft-cuda", "cap"),
    ("split-step-fft-cuda", "diffusion"),
])
def test_advance_boundary_and_diffusion_matches_steps(method, case):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = method
    cfg.solver.precision = "double"
    cfg.solver.dt = 0.01
    cfg.physics.init_seed = 9
    cfg.reservoir.reservoir_type = "single"
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.02
    if case == "closed-interval":
        cfg.grid.grid_type = "closed-interval"
    elif case == "isotropic-9pt":
        cfg.solver.laplacian = "isotropic-9pt"
    elif case == "cap":
        cfg.boundary_condition.absorption = "cap"
    elif case == "diffusion":
        cfg.physics.reservoir_diffusion_R = 0.01
    fast = SimulationController(cfg)
    old = SimulationController(cfg)
    for controller in (fast, old):
        controller.pump_field.lasers = controller.lasers
    fast.solver.advance(fast.potential, fast.pump_field, fast.reservoir,
                        fast.boundary_condition, fast.state, 0, 4)
    for k in range(4):
        old.solver.step(old.potential, old.pump_field, old.reservoir,
                        old.boundary_condition, old.state)
        old.state.t = (k + 1) * cfg.solver.dt
    _assert_matches_steps(fast.state.psi, old.state.psi, method, "double")
    for field, other in zip(fast.reservoir.get_state(), old.reservoir.get_state()):
        _assert_matches_steps(field, other, method, "double")
