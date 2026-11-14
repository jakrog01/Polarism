"""CUDA spinor source and GPU parity checks."""
from copy import deepcopy

import numpy as np
import pytest

from polarism import SimulationController
from polarism.compute_engine import has_cuda_device
from polarism.model_terms import CudaSnippet, Term, available_terms, register_term
from polarism.solver.rk4_cuda_solver import build_spinor_tiled_rhs_source
from polarism.solver.split_step_fft_cuda_solver import spinor_matrix_kernel_source
from tests.unit.conftest import small_config


def _config(method, precision="double", graphs=False, reservoir_components=2):
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 8.0
    cfg.physics.spin = "spinor"
    cfg.physics.hbar = 1.0
    cfg.physics.m_eff = 1.0
    cfg.physics.g_C = 0.15
    cfg.physics.alpha_2 = -0.07
    cfg.physics.zeeman = 0.19
    cfg.physics.te_tm = 0.12
    cfg.physics.g_R = 0.03
    cfg.physics.g_I = 0.0
    cfg.physics.R = 0.1
    cfg.physics.gamma_C = 0.2
    cfg.reservoir.reservoir_type = "single"
    cfg.reservoir.components = reservoir_components
    cfg.laser.P0 = 0.3
    cfg.laser.circular_polarization = 0.4
    cfg.solver.method = method
    cfg.solver.precision = precision
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.003
    cfg.solver.cuda_graphs = graphs
    cfg.noise.enabled = False
    cfg.noise.initial = "none"
    cfg.noise.seed = 173
    cfg.result.save_results = False
    return cfg


def _initial(grid, dtype):
    x = grid.X.get() if hasattr(grid.X, "get") else grid.X
    y = grid.Y.get() if hasattr(grid.Y, "get") else grid.Y
    wave = np.exp(2j * np.pi * (x + 2 * y) / 8.0)
    return np.stack(((0.35 + 0.1j) * wave, (0.2 - 0.15j) * np.conj(wave))).astype(dtype)


def _run(cfg, gpu, initial=None, members=None):
    import cupy as cp

    current = deepcopy(cfg)
    current.compute_engine.use_gpu = gpu
    run = SimulationController(current, members=members)
    if initial is None:
        dtype = np.complex64 if cfg.solver.precision == "single" else np.complex128
        initial = _initial(run.grid, dtype)
    run.set_initial_state(cp.asarray(initial) if gpu else initial)
    run.run()
    field = cp.asnumpy(run.state.psi) if gpu else run.state.psi.copy()
    densities = tuple(cp.asnumpy(x) if gpu else x.copy()
                      for x in run.reservoir.get_state())
    return run, field, densities


@pytest.mark.parametrize("real", ("float", "double"))
@pytest.mark.parametrize("boundary", ("periodic", "closed-interval"))
def test_spinor_tiled_source_has_corner_halo(real, boundary):
    source = build_spinor_tiled_rhs_source(real, boundary, 8, 8)
    assert "scratch[(2*c)*area+p]" in source
    assert "p+pitch+1" in source
    assert "p-pitch-1" in source
    assert "cross_re" in source and "cross_im" in source
    assert "alpha2*densities[other]" in source
    assert "sign*zeeman" in source


def test_spinor_tiled_term_codegen():
    term = type("RateTerm", (), {
        "category": "rate",
        "cuda": CudaSnippet("-alpha * rho", (("double", "alpha"),)),
    })()
    source = build_spinor_tiled_rhs_source("double", "periodic", 8, 8, (term,))
    assert "double term_0_alpha" in source
    assert "real rho = densities[c]" in source
    assert "real extra_rate = (-term_0_alpha * rho);" in source


@pytest.mark.parametrize("precision", ("single", "double"))
def test_spinor_spectral_source_multiplies_two_spectra(precision):
    source = spinor_matrix_kernel_source(precision)
    assert "matrix_members == 1 ? 0 : member" in source
    assert "output[a]" in source and "output[b]" in source


@pytest.mark.parametrize("method,reference", (("rk4-cuda", "rk4-fdm"),
                                            ("ifrk-fft-cuda", "ip-rk4"),
                                            ("ifrk4-fft-cuda", "ip-rk4")))
def test_spinor_cpu_path_matches_reference(method, reference):
    cfg = _config(method)
    cfg.compute_engine.use_gpu = False
    expected_cfg = deepcopy(cfg)
    expected_cfg.solver.method = reference
    _, expected, expected_res = _run(expected_cfg, False)
    _, actual, actual_res = _run(cfg, False)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    for got, want in zip(actual_res, expected_res):
        np.testing.assert_allclose(got, want, rtol=0, atol=2e-8)


@pytest.mark.gpu
@pytest.mark.parametrize("method,reference", (("rk4-cuda", "rk4-fdm"),
                                            ("split-step-fft-cuda", "split-step-fft"),
                                            ("ifrk-fft-cuda", "ip-rk4"),
                                            ("ifrk4-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
@pytest.mark.parametrize("reservoir_components", (1, 2))
def test_spinor_gpu_matches_numpy(method, reference, precision, graphs,
                                  reservoir_components, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config(method, precision, graphs, reservoir_components)
    expected_cfg = deepcopy(cfg)
    expected_cfg.solver.method = reference
    _, expected, expected_res = _run(expected_cfg, False)
    _, actual, actual_res = _run(cfg, True)
    tolerance = 4e-5 if precision == "single" else 2e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for got, want in zip(actual_res, expected_res):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize("boundary", ("periodic", "closed-interval"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_spinor_tiled_boundary_matches_fdm(boundary, precision, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config("rk4-cuda", precision)
    cfg.grid.grid_type = boundary
    reference = deepcopy(cfg)
    reference.solver.method = "rk4-fdm"
    _, expected, _ = _run(reference, False)
    _, actual, _ = _run(cfg, True)
    tolerance = 4e-5 if precision == "single" else 2e-10
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
def test_spinor_tiled_term_matches_fdm(use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")

    class CubicLoss(Term):
        category = "rate"
        cuda = CudaSnippet("-alpha * rho", (("double", "alpha"),))

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return -self.params["alpha"] * abs(field) ** 2

    saved = dict(available_terms)
    try:
        register_term("spinor-cuda-loss")(CubicLoss)
        cfg = _config("rk4-cuda")
        cfg.terms = {"spinor-cuda-loss": {"alpha": 0.2}}
        reference = deepcopy(cfg)
        reference.solver.method = "rk4-fdm"
        _, expected, _ = _run(reference, False)
        _, actual, _ = _run(cfg, True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    finally:
        available_terms.clear()
        available_terms.update(saved)


@pytest.mark.gpu
@pytest.mark.parametrize("method,reference", (("rk4-cuda", "rk4-fdm"),
                                            ("split-step-fft-cuda", "split-step-fft"),
                                            ("ifrk-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_noise_streams_match_numpy(method, reference, precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config(method, precision, graphs)
    cfg.solver.total_time = cfg.solver.dt
    cfg.reservoir.reservoir_type = "none"
    cfg.physics.g_C = cfg.physics.alpha_2 = 0.0
    cfg.physics.zeeman = cfg.physics.te_tm = 0.0
    cfg.physics.R = 0.0
    cfg.laser.P0 = 0.0
    cfg.noise.enabled = True
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    initial = np.zeros((2, 8, 8), dtype=np.complex64 if precision == "single"
                       else np.complex128)
    expected_cfg = deepcopy(cfg)
    expected_cfg.solver.method = reference
    _, expected, _ = _run(expected_cfg, False, initial)
    _, actual, _ = _run(cfg, True, initial)
    tolerance = 3e-6 if precision == "single" else 3e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert not np.array_equal(actual[0], actual[1])


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_batch_and_checkpoint_resume(graphs, use_gpu, tmp_path):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = _config("rk4-cuda", graphs=graphs)
    cfg.noise.enabled = True
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    cfg.compute_engine.use_gpu = True
    members = [{"physics.gamma_C": 0.1}, {"physics.gamma_C": 0.2}]
    batch = SimulationController(deepcopy(cfg), members=members)
    initial = _initial(batch.grid, np.complex128)
    batch.set_initial_state(cp.asarray(np.stack((initial, initial))))
    batch.run()
    for index, override in enumerate(members):
        single = SimulationController(deepcopy(cfg), members=[override])
        single.set_initial_state(cp.asarray(initial))
        single.run()
        np.testing.assert_allclose(cp.asnumpy(batch.state.psi[index]),
                                   cp.asnumpy(single.state.psi), rtol=0, atol=1e-12)
    first_cfg = deepcopy(cfg)
    first_cfg.solver.total_time = 0.001
    first_cfg.checkpoint.path = str(tmp_path / "spinor_cuda.h5")
    first = SimulationController(first_cfg)
    first.set_initial_state(cp.asarray(initial))
    first.run()
    resumed_cfg = deepcopy(cfg)
    resumed_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed = SimulationController(resumed_cfg)
    resumed.run()
    full = SimulationController(deepcopy(cfg))
    full.set_initial_state(cp.asarray(initial))
    full.run()
    np.testing.assert_allclose(cp.asnumpy(resumed.state.psi), cp.asnumpy(full.state.psi),
                               rtol=0, atol=1e-12)
