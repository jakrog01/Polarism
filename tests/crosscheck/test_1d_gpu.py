"""One-dimensional CUDA agreement and batched layout checks."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import has_cuda_device
from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController


def configuration(method, reservoir, boundary, gpu):
    cfg = Config()
    cfg.compute_engine.use_gpu = gpu
    cfg.grid.ndim = 1
    cfg.grid.nx = 64
    cfg.grid.lx = 40.0
    cfg.grid.grid_type = boundary
    cfg.reservoir.reservoir_type = reservoir
    cfg.solver.method = method
    cfg.solver.precision = "double"
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.003
    cfg.physics.init_seed = 17
    cfg.physics.init_eps = 0.1
    if method in ("split-step-fft", "split-step-fft-cuda"):
        cfg.physics.R = 0.0
        cfg.physics.g_R = 0.0
        cfg.physics.g_I = 0.0
    cfg.laser.P0 = 0.02
    cfg.potential.potential_type = "double-well-supergaussian"
    cfg.result.save_results = False
    return cfg


def evolve(controller, steps=3):
    for _ in range(steps):
        controller.solver.step(controller.potential, controller.pump_field,
                               controller.reservoir, controller.boundary_condition,
                               controller.state)
        controller.state.t += controller.cfg.solver.dt
    return (controller.xp.asnumpy(controller.state.psi) if controller.cfg.compute_engine.use_gpu
            else np.asarray(controller.state.psi).copy())


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "ifrk-fft-cuda", "split-step-fft-cuda"])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
def test_1d_gpu_agrees_with_cpu(method, reservoir, boundary):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    if boundary == "closed-interval" and method != "rk4-cuda":
        pytest.skip("spectral solver requires periodic boundaries")
    cpu_method = "split-step-fft" if method == "split-step-fft-cuda" else method
    cpu = SimulationController(configuration(cpu_method, reservoir, boundary, False))
    expected = evolve(cpu)
    gpu = SimulationController(configuration(method, reservoir, boundary, True))
    actual = evolve(gpu)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
    for actual_field, expected_field in zip(gpu.reservoir.get_state(), cpu.reservoir.get_state()):
        np.testing.assert_allclose(gpu.xp.asnumpy(actual_field), expected_field,
                                   rtol=1e-9, atol=1e-11)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "ifrk-fft-cuda", "split-step-fft-cuda"])
def test_1d_gpu_batch_matches_separate(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = configuration(method, "single", "periodic", True)
    members = [{}, {}]
    batch = SimulationController(cfg, members=members)
    batch.solver.advance(batch.potential, batch.pump_field, batch.reservoir,
                         batch.boundary_condition, batch.state, 0, 3)
    actual = batch.xp.asnumpy(batch.state.psi)
    assert actual.shape == (2, 64)
    for i in range(2):
        single = SimulationController(configuration(method, "single", "periodic", True))
        single.solver.advance(single.potential, single.pump_field, single.reservoir,
                              single.boundary_condition, single.state, 0, 3)
        np.testing.assert_allclose(actual[i], single.xp.asnumpy(single.state.psi),
                                   rtol=1e-10, atol=1e-11)
