"""GPU parity checks for the rk4-cuda 4c features."""

from copy import deepcopy

import numpy as np
import pytest

from polarism import SimulationController
from polarism.compute_engine import has_cuda_device
from polarism.model_terms import CudaSnippet, Term, available_terms, register_term
from tests.unit.conftest import small_config


def _config(method="rk4-cuda", precision="double"):
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 8.0
    cfg.solver.method = method
    cfg.solver.precision = precision
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.01
    cfg.reservoir.reservoir_type = "none"
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.laser.P0 = 0.0
    cfg.noise.seed = 173
    cfg.noise.initial = "none"
    cfg.result.save_results = False
    return cfg


def _run(cfg, gpu, members=None):
    import cupy as cp

    current = deepcopy(cfg)
    current.compute_engine.use_gpu = gpu
    run = SimulationController(current, members=members)
    dtype = np.complex64 if current.solver.precision == "single" else np.complex128
    run.state.psi = (cp if gpu else np).full(run.grid.shape, 0.4 + 0.2j,
                                              dtype=dtype)
    run.run()
    field = cp.asnumpy(run.state.psi) if gpu else run.state.psi.copy()
    return run, field


@pytest.mark.gpu
@pytest.mark.parametrize("feature", ("none", "term", "potential", "drive"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_new_features_agree_with_fdm(feature, precision, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")

    class CubicLoss(Term):
        category = "rate"
        cuda = CudaSnippet("-alpha * rho", (("double", "alpha"),))

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return -self.params["alpha"] * abs(field) ** 2

    saved = dict(available_terms)
    try:
        register_term("rk4-test-loss")(CubicLoss)
        cfg = _config(precision=precision)
        if feature == "term":
            cfg.terms = {"rk4-test-loss": {"alpha": 0.2}}
        elif feature == "potential":
            cfg.potential.potential_type = "profile"
            cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
            cfg.potential.envelope = {"type": "step", "t0": 0.005}
            cfg.boundary_condition.absorption = "cap"
        elif feature == "drive":
            cfg.drive.enabled = True
            cfg.drive.amplitude = 0.2
            cfg.drive.energy = 0.3
            cfg.drive.kx = 2 * np.pi / cfg.grid.lx
            cfg.drive.envelope = {"type": "step", "t0": 0.005}
        reference = deepcopy(cfg)
        reference.solver.method = "rk4-fdm"
        _, expected = _run(reference, False)
        _, actual = _run(cfg, True)
        tolerance = 2e-5 if precision == "single" else 1e-10
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    finally:
        available_terms.clear()
        available_terms.update(saved)


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", (False, True))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_fused_wigner_increment_matches_numpy(graphs, precision, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = _config(precision=precision)
    cfg.reservoir.reservoir_type = "single"
    cfg.solver.total_time = 2 * cfg.solver.dt
    cfg.solver.cuda_graphs = graphs
    cfg.noise.stochastic = "wigner"
    cfg.physics.gamma_C = 0.0
    cfg.physics.R = 0.0
    cfg.noise.stochastic_strength = 1.0
    cfg.physics.gamma_C = 0.2
    cfg.compute_engine.use_gpu = True
    run = SimulationController(cfg)
    run.state.psi = cp.zeros(run.grid.shape, dtype=run.state.psi.dtype)
    run.run()
    reference = deepcopy(cfg)
    reference.solver.method = "rk4-fdm"
    reference.compute_engine.use_gpu = False
    cpu = SimulationController(reference)
    cpu.state.psi = np.zeros(cpu.grid.shape, dtype=cpu.state.psi.dtype)
    cpu.run()
    expected = cpu.state.psi.ravel()
    tolerance = 3e-6 if precision == "single" else 3e-13
    np.testing.assert_allclose(cp.asnumpy(run.state.psi).ravel(), expected,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
def test_drive_and_potential_graph_replay_matches_launches(use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")

    cfg = _config()
    cfg.solver.total_time = 0.004
    cfg.drive.enabled = True
    cfg.drive.amplitude = 0.2
    cfg.drive.energy = 0.3
    cfg.drive.envelope = {"type": "step", "t0": 0.002}
    cfg.potential.potential_type = "profile"
    cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
    cfg.potential.envelope = {"type": "step", "t0": 0.002}
    cfg.solver.cuda_graphs = False
    _, launches = _run(cfg, True)
    cfg.solver.cuda_graphs = True
    _, graphed = _run(cfg, True)
    np.testing.assert_allclose(graphed, launches, rtol=0, atol=1e-13)


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", (False, True))
def test_noisy_batch_and_checkpoint_resume(graphs, use_gpu, tmp_path):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = _config()
    cfg.reservoir.reservoir_type = "single"
    cfg.noise.stochastic = "wigner"
    cfg.solver.cuda_graphs = graphs
    cfg.solver.total_time = 0.004
    cfg.compute_engine.use_gpu = True
    members = [{"physics.gamma_C": 0.1}, {"physics.gamma_C": 0.2}]
    batch = SimulationController(cfg, members=members)
    batch.run()
    for index, override in enumerate(members):
        single = SimulationController(cfg, members=[override])
        single.run()
        np.testing.assert_array_equal(cp.asnumpy(batch.state.psi[index]),
                                      cp.asnumpy(single.state.psi))
    first = deepcopy(cfg)
    first.solver.total_time = 0.002
    first.checkpoint.path = str(tmp_path / "rk4_cuda_noise.h5")
    SimulationController(first).run()
    resume = deepcopy(cfg)
    resume.checkpoint.resume_from = first.checkpoint.path
    resumed = SimulationController(resume)
    resumed.run()
    full = SimulationController(cfg)
    full.run()
    np.testing.assert_array_equal(cp.asnumpy(resumed.state.psi),
                                  cp.asnumpy(full.state.psi))
