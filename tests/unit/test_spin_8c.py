"""CUDA spinor source and GPU parity checks."""
from copy import deepcopy

import numpy as np
import pytest

from polarism import SimulationController
from polarism.compute_engine import has_cuda_device
from polarism.model_terms import CudaSnippet, Term, available_terms, register_term
from polarism.solver.ifrk_cuda_fused import spinor_feature_source, spinor_fused_source
from polarism.solver.rk4_cuda_solver import (build_spinor_step_source,
                                            build_spinor_tiled_rhs_source,
                                            build_spinor_tiled_stage_source)
from polarism.solver.split_step_fft_cuda_solver import (spinor_local_source,
                                                        spinor_matrix_kernel_source,
                                                        spinor_spectral_kernel_source)
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


@pytest.mark.parametrize("real", ("float", "double"))
def test_spinor_stage_sources_pair_fields_and_reservoirs(real):
    rk = build_spinor_tiled_stage_source(real, "periodic", 8, 8, graph=True)
    split = spinor_local_source("single" if real == "float" else "double",
                                "single", graph=True, noise=True)
    ifrk = spinor_fused_source(real, 0, 1, 4, 1, 2)
    assert "previous_field[2*source]" in rk
    assert "densities[0]+densities[1]" in rk
    assert "scales[scale_index]" in rk
    assert "alpha2*rho[1-c]*phase_time[1-c]" in split
    assert "unsigned int)c" in split
    assert "spinor_apply(matrix" in ifrk
    assert "alpha2*rho[1-component]" in ifrk


def test_spinor_feature_sources_inline_terms_and_drive():
    term = type("RateTerm", (), {
        "category": "rate",
        "cuda": CudaSnippet("-alpha*rho", (("double", "alpha"),)),
    })()
    split = spinor_local_source("double", "single", terms=(term,),
                                varying_potential=True, drive=True)
    ifrk = spinor_feature_source(spinor_fused_source("double", 0, 0, 4, 1, 2),
                                "double", (term,), True, True)
    rk = build_spinor_tiled_stage_source("double", "periodic", 8, 8, (term,),
                                         graph=True, varying_potential=True, drive=True)
    for source in (rk, split, ifrk):
        assert "term_0_alpha" in source
        assert "potential_scale" in source
        assert "drive_scale" in source
        assert "extra_rate" in source


@pytest.mark.parametrize("phase,stages", (("whole", "stage < 4"), ("first", "stage < 2"),
                                          ("second", "stage = 2")))
def test_spinor_step_source_covers_stage_range(phase, stages):
    source = build_spinor_step_source("float", 16, 8, True, 0, 2, phase=phase,
                                      relaxation=False)
    assert stages in source
    assert ("k_field" in source) == (phase != "whole")
    assert ("const real* __restrict__ k_field" in source) == (phase == "second")
    assert "__launch_bounds__(256, 2)" in source
    assert "relaxed" not in source
    relaxed = build_spinor_step_source("double", 16, 8, True, 1, 2, relaxation=True)
    assert "relaxed*lap_re" in relaxed and "gI*ni" in relaxed


def test_spinor_step_source_one_dimensional_has_no_cross_terms():
    source = build_spinor_step_source("float", 64, 1, False, 0, 1)
    assert "crs" not in source and "sums_y" not in source


@pytest.mark.parametrize("precision", ("single", "double"))
def test_spinor_spectral_source_builds_hermitian_propagator(precision):
    source = spinor_spectral_kernel_source(precision)
    assert "sincos(theta" in source and "sincos(-eps*tau" in source
    assert "output[b+1]" in source


@pytest.mark.parametrize("method,reference", (("rk4-cuda", "rk4-fdm"),
                                            ("ifrk-fft-cuda", "ip-rk4"),
                                            ("ifrk4-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_cpu_path_matches_reference(method, reference, graphs):
    cfg = _config(method, graphs=graphs)
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
                                            ("ifrk-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_spinor_one_dimensional_gpu_matches_numpy(method, reference, precision, use_gpu):
    import cupy as cp

    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config(method, precision, True)
    cfg.grid.ndim = 1
    cfg.grid.nx = 64
    cfg.grid.lx = 8.0
    results = []
    for solver, gpu in ((method, True), (reference, False)):
        current = deepcopy(cfg)
        current.solver.method = solver
        current.compute_engine.use_gpu = gpu
        run = SimulationController(current)
        wave = np.exp(2j * np.pi * np.asarray(run.grid.X.get() if gpu else run.grid.X) / 8.0)
        dtype = np.complex64 if precision == "single" else np.complex128
        initial = np.stack(((0.35 + 0.1j) * wave, (0.2 - 0.15j) * np.conj(wave))).astype(dtype)
        run.set_initial_state(cp.asarray(initial) if gpu else initial)
        run.run()
        results.append(cp.asnumpy(run.state.psi) if gpu else run.state.psi.copy())
    tolerance = 4e-5 if precision == "single" else 2e-8
    np.testing.assert_allclose(results[0], results[1], rtol=tolerance, atol=tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_rk_relaxation_matches_fdm(precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config("rk4-cuda", precision, graphs)
    cfg.physics.kinetic_relaxation_eta = 0.3
    reference = deepcopy(cfg)
    reference.solver.method = "rk4-fdm"
    _, expected, expected_res = _run(reference, False)
    _, actual, actual_res = _run(cfg, True)
    tolerance = 4e-5 if precision == "single" else 2e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for got, want in zip(actual_res, expected_res):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


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
    run, actual, actual_res = _run(cfg, True)
    tolerance = 4e-5 if precision == "single" else 2e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for got, want in zip(actual_res, expected_res):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    if graphs:
        cache = (run.solver._spinor_graphs if method == "rk4-cuda" else
                 run.solver._graph_cache if method == "split-step-fft-cuda" else
                 run.solver._fused._graph_cache)
        assert cache


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
@pytest.mark.parametrize("method,reference", (("split-step-fft-cuda", "split-step-fft"),
                                            ("ifrk-fft-cuda", "ip-rk4")))
@pytest.mark.parametrize("feature", ("term", "potential"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_spectral_features_match_numpy(method, reference, feature,
                                              precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")

    class CubicLoss(Term):
        category = "rate"
        cuda = CudaSnippet("-alpha*rho", (("double", "alpha"),))

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return -self.params["alpha"] * abs(field) ** 2

    saved = dict(available_terms)
    try:
        register_term("spinor-spectral-loss")(CubicLoss)
        cfg = _config(method, precision, graphs)
        if feature == "term":
            cfg.terms = {"spinor-spectral-loss": {"alpha": 0.2}}
        else:
            cfg.potential.potential_type = "profile"
            cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
            cfg.potential.envelope = {"type": "step", "t0": 0.002}
            cfg.boundary_condition.absorption = "cap"
        expected_cfg = deepcopy(cfg)
        expected_cfg.solver.method = reference
        _, expected, expected_res = _run(expected_cfg, False)
        run, actual, actual_res = _run(cfg, True)
        tolerance = 4e-5 if precision == "single" else 2e-8
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        for got, want in zip(actual_res, expected_res):
            np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
        if graphs:
            assert (run.solver._graph_cache if method == "split-step-fft-cuda"
                    else run.solver._fused._graph_cache)
    finally:
        available_terms.clear()
        available_terms.update(saved)


@pytest.mark.gpu
@pytest.mark.parametrize("feature", ("potential", "drive"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("graphs", (False, True))
def test_spinor_rk_stage_features_match_fdm(feature, precision, graphs, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config("rk4-cuda", precision, graphs)
    if feature == "potential":
        cfg.potential.potential_type = "profile"
        cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
        cfg.potential.envelope = {"type": "step", "t0": 0.002}
        cfg.boundary_condition.absorption = "cap"
    else:
        cfg.drive.enabled = True
        cfg.drive.amplitude = 0.2
        cfg.drive.energy = 0.3
        cfg.drive.envelope = {"type": "gaussian-pulses", "center": 0.002,
                              "sigma": 0.001}
    reference = deepcopy(cfg)
    reference.solver.method = "rk4-fdm"
    _, expected, expected_res = _run(reference, False)
    run, actual, actual_res = _run(cfg, True)
    tolerance = 4e-5 if precision == "single" else 2e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for got, want in zip(actual_res, expected_res):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    if graphs:
        assert run.solver._spinor_graphs


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
@pytest.mark.parametrize("method", ("rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda"))
def test_spinor_noise_graph_captures_multiple_steps(method, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = _config(method, graphs=True)
    cfg.noise.enabled = True
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 1.0
    run, _, _ = _run(cfg, True)
    cache = (run.solver._spinor_graphs if method == "rk4-cuda" else
             run.solver._graph_cache if method == "split-step-fft-cuda" else
             run.solver._fused._graph_cache)
    assert cache


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
    if graphs:
        assert batch.solver._spinor_graphs
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
