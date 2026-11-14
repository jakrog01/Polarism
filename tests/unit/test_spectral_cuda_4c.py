"""GPU parity for spectral CUDA term and stochastic paths."""
from copy import deepcopy

import numpy as np
import pytest

from polarism import SimulationController
from polarism.compute_engine import has_cuda_device
from polarism.model_terms import CudaSnippet, Term, available_terms, register_term
from polarism.noise.philox import counter_words, normal_pair, philox4x32
from tests.unit.conftest import small_config


def configuration(method, precision="double", graphs=False):
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 8.0
    cfg.solver.method = method
    cfg.solver.precision = precision
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.004
    cfg.solver.cuda_graphs = graphs
    cfg.reservoir.reservoir_type = "none"
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.physics.gamma_C = 0.1
    cfg.laser.P0 = 0.0
    cfg.noise.seed = 173
    cfg.noise.initial = "none"
    cfg.result.save_results = False
    return cfg


def run_case(cfg, gpu, members=None, initial=0.4 + 0.2j):
    import cupy as cp

    actual = deepcopy(cfg)
    actual.compute_engine.use_gpu = gpu
    run = SimulationController(actual, members=members)
    xp = cp if gpu else np
    dtype = np.complex64 if cfg.solver.precision == "single" else np.complex128
    shape = run.grid.shape if not members or len(members) == 1 else (len(members), *run.grid.shape)
    run.state.psi = xp.full(shape, initial, dtype=dtype)
    run.run()
    return run, cp.asnumpy(run.state.psi) if gpu else run.state.psi.copy()


@pytest.mark.gpu
@pytest.mark.parametrize("method,reference", (("split-step-fft-cuda", "split-step-fft"),
                                            ("ifrk-fft-cuda", "ip-rk4"),
                                            ("ifrk4-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("feature", ("none", "term", "potential", "drive"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_spectral_features_match_numpy(method, reference, feature, precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")

    class CubicLoss(Term):
        category = "rate"
        cuda = CudaSnippet("-alpha*rho", (("double", "alpha"),))

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return -self.params["alpha"] * abs(field) ** 2

    saved = dict(available_terms)
    try:
        register_term("spectral-test-loss")(CubicLoss)
        cfg = configuration(method, precision, graphs)
        if feature == "term":
            cfg.terms = {"spectral-test-loss": {"alpha": 0.2}}
        elif feature == "potential":
            cfg.potential.potential_type = "profile"
            cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
            cfg.potential.envelope = {"type": "step", "t0": 0.002}
            cfg.boundary_condition.absorption = "cap"
        elif feature == "drive":
            cfg.drive.enabled = True
            cfg.drive.amplitude = 0.2
            cfg.drive.energy = 0.3
            cfg.drive.envelope = {"type": "gaussian-pulses", "center": 0.002, "sigma": 0.001}
            if method == "split-step-fft-cuda":
                reference = "ip-rk4"
        expected_cfg = deepcopy(cfg)
        expected_cfg.solver.method = reference
        _, expected = run_case(expected_cfg, False)
        _, actual = run_case(cfg, True)
        tolerance = 2e-4 if precision == "single" else 3e-5
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    finally:
        available_terms.clear()
        available_terms.update(saved)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_split_step_noise_matches_cpu_solver(precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = configuration("split-step-fft-cuda", precision, graphs)
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    expected_cfg = deepcopy(cfg)
    expected_cfg.solver.method = "split-step-fft"
    _, expected = run_case(expected_cfg, False)
    _, actual = run_case(cfg, True)
    noiseless = deepcopy(cfg)
    noiseless.noise.stochastic_strength = 0.0
    _, reference = run_case(noiseless, True)
    assert np.max(np.abs(expected - reference)) > 1e-3
    tolerance = 3e-6 if precision == "single" else 3e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ("ifrk-fft-cuda",))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_fused_noise_matches_numpy_philox(method, precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = configuration(method, precision, graphs)
    cfg.solver.total_time = cfg.solver.dt
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    _, actual = run_case(cfg, True, initial=0)
    words = philox4x32(counter_words(np.arange(64), 0), cfg.noise.seed)
    re, im = normal_pair(words, precision)
    dtype = np.float32 if precision == "single" else np.float64
    sigma = np.sqrt(dtype(cfg.physics.gamma_C) * dtype(cfg.solver.dt / 4))
    expected = (sigma * (re + 1j * im)).reshape(8, 8)
    tolerance = 3e-6 if precision == "single" else 3e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ("split-step-fft-cuda", "ifrk-fft-cuda"))
@pytest.mark.parametrize("graphs", (False, True))
def test_noisy_batch_and_checkpoint_resume(method, graphs, use_gpu, tmp_path):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = configuration(method, graphs=graphs)
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    members = [{"physics.gamma_C": 0.1}, {"physics.gamma_C": 0.2}]
    batch, _ = run_case(cfg, True, members=members)
    for index, override in enumerate(members):
        single, _ = run_case(cfg, True, members=[override])
        np.testing.assert_allclose(cp.asnumpy(batch.state.psi[index]),
                                   cp.asnumpy(single.state.psi), rtol=0, atol=3e-13)
    first_cfg = deepcopy(cfg)
    first_cfg.solver.total_time = 0.002
    first_cfg.checkpoint.path = str(tmp_path / "spectral_noise.h5")
    first, _ = run_case(first_cfg, True)
    resumed_cfg = deepcopy(cfg)
    resumed_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed_cfg.compute_engine.use_gpu = True
    resumed = SimulationController(resumed_cfg)
    resumed.run()
    full, _ = run_case(cfg, True)
    np.testing.assert_allclose(cp.asnumpy(resumed.state.psi),
                               cp.asnumpy(full.state.psi), rtol=0, atol=3e-13)
