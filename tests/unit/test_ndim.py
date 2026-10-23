"""One-dimensional geometry and persistence contracts."""
from __future__ import annotations

from copy import deepcopy

import h5py
import numpy as np
import pytest

from polarism.config.simulation_parameters import Config, GridParameters
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.boundary_conditions.absorption.absorption_strategy import create_absorption_profile
from polarism.grid.create_grid import create_grid
from polarism.init_condition import make_initial_psi
from polarism.potential.create_potential import create_potential
from polarism.simulation_controller import SimulationController
from polarism.solver.rk4_cuda_solver import build_flat_1d_sources


def configuration(method="rk4-fdm", boundary="periodic", reservoir="single", ndim=1):
    cfg = Config()
    cfg.grid = GridParameters(ndim=ndim, nx=32, ny=4, lx=40.0, ly=7.0,
                              grid_type=boundary)
    cfg.physics.init_seed = 8
    cfg.physics.reservoir_diffusion_R = 0.02
    cfg.physics.reservoir_diffusion_A = 0.02
    cfg.physics.reservoir_diffusion_I = 0.02
    cfg.solver.method = method
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.004
    cfg.reservoir.reservoir_type = reservoir
    cfg.result.save_results = False
    return cfg


@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
def test_grid_1d_contract(boundary):
    grid = create_grid(GridParameters(ndim=1, nx=16, ny=-5, lx=10.0, ly=-1.0,
                                      grid_type=boundary))
    dx = 10.0 / (16 if boundary == "periodic" else 15)
    assert grid.ndim == 1
    assert grid.shape == (16,)
    assert grid.spacing == (dx,)
    assert grid.lengths == (10.0,)
    assert grid.X.shape == grid.KX.shape == grid.k_squared.shape == (16,)
    np.testing.assert_array_equal(grid.k_squared, grid.kx ** 2)
    assert not any(hasattr(grid, name) for name in ("ny", "ly", "dy", "Y", "ky", "KY"))


@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
def test_grid_2d_metadata(boundary):
    grid = create_grid(GridParameters(ndim=2, nx=16, ny=8, lx=10.0, ly=6.0,
                                      grid_type=boundary))
    assert grid.ndim == 2
    assert grid.shape == (8, 16)
    assert grid.spacing == (grid.dy, grid.dx)
    assert grid.lengths == (6.0, 10.0)
    np.testing.assert_array_equal(grid.k_squared, grid.KX ** 2 + grid.KY ** 2)


def test_1d_validation():
    cfg = configuration()
    cfg.grid.ny = -3
    cfg.grid.ly = -3
    validate_config(cfg)
    for bad in (0, 3, True):
        invalid = deepcopy(cfg)
        invalid.grid.ndim = bad
        with pytest.raises(ConfigValidationError, match="grid.ndim"):
            validate_config(invalid)
    cfg.solver.laplacian = "isotropic-9pt"
    with pytest.raises(ConfigValidationError, match="isotropic-9pt"):
        validate_config(cfg)
    cfg.solver.laplacian = "five-point"
    for field in ("animate", "real_time_view"):
        invalid = deepcopy(cfg)
        setattr(invalid.result, field, True)
        with pytest.raises(ConfigValidationError, match="visualization of 1D runs is not supported yet"):
            validate_config(invalid)
    cfg.solver.method = "custom-plugin-solver"
    with pytest.raises(ConfigValidationError, match="does not support grid.ndim"):
        validate_config(cfg)


@pytest.mark.parametrize("mode", ["legacy_positive_uniform", "complex_gaussian_zero_mean",
                                   "filtered_complex_gaussian"])
def test_1d_initial_conditions(mode):
    psi = make_initial_psi(np, (64,), (0.5,), 1e-3, mode=mode,
                           k_cutoff_um=2.0, seed=4)
    assert psi.shape == (64,)
    assert np.isfinite(psi).all()
    if mode == "filtered_complex_gaussian":
        k = 2 * np.pi * np.fft.fftfreq(64, d=0.5)
        assert np.max(np.abs(np.fft.fft(psi)[np.abs(k) > 2.0])) < 1e-12


def test_1d_absorption_and_potential_cut():
    cfg = configuration()
    cfg.boundary_condition.absorption = "cap"
    cfg.potential.potential_type = "double-well-supergaussian"
    grid = create_grid(cfg.grid)
    profile = create_absorption_profile(grid.shape, cfg.boundary_condition)
    assert profile.shape == (32,)
    assert profile[0] == profile[-1] == 1
    assert profile[16] == 0
    from polarism.potential.potential_registry import available_potentials
    expected = available_potentials.lookup(cfg.potential.potential_type)(
        grid.X, np.zeros_like(grid.X), cfg.potential)
    np.testing.assert_array_equal(create_potential(cfg.potential, grid), expected)
    run = SimulationController(cfg)
    assert run.potential.shape == (32,)
    assert run.boundary_condition.before_step_action().shape == (32,)


def test_1d_set_initial_state_batched():
    run = SimulationController(configuration())
    psi = np.ones((2, 32), dtype=np.complex128)
    reservoir = (np.full((2, 32), 0.2),)
    run.set_initial_state(psi, reservoir, t=0.002)
    assert run.state.field.shape == (2, 1, 32)
    assert run.reservoir.get_state()[0].shape == (2, 1, 32)
    assert run.state.t == 0.002


@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("method", ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda",
                                    "split-step-fft", "etd-rk2", "ip-rk4",
                                    "ifrk-fft-cuda", "ifrk4-fft-cuda"])
def test_1d_matches_y_uniform_2d(method, reservoir, boundary):
    if boundary == "closed-interval" and method not in ("rk4-fdm", "rk4-fdm-fused", "rk4-cuda"):
        pytest.skip("spectral solver requires periodic boundaries")
    controllers = []
    for ndim in (1, 2):
        controller = SimulationController(configuration(method, boundary, reservoir, ndim))
        x = np.linspace(-1, 1, 32)
        psi = np.exp(-5 * x * x) * (1 + 0.2j * x)
        controller.state.psi = np.broadcast_to(psi, controller.grid.shape).copy()
        controllers.append(controller)
    for controller in controllers:
        profile = 0.02 + np.linspace(0, 0.01, 32)
        pump = lambda t, shape=controller.grid.shape: np.broadcast_to(profile, shape)
        potential = np.broadcast_to(np.linspace(-0.001, 0.001, 32), controller.grid.shape)
        for _ in range(4):
            controller.solver.step(potential, pump, controller.reservoir,
                                   controller.boundary_condition, controller.state)
            controller.state.t += controller.cfg.solver.dt
    one, two = controllers
    np.testing.assert_allclose(one.state.psi, two.state.psi[0], rtol=1e-12, atol=1e-12)
    for a, b in zip(one.reservoir.get_state(), two.reservoir.get_state()):
        np.testing.assert_allclose(a[0, 0], b[0, 0, 0], rtol=1e-12, atol=1e-12)
    if method in ("rk4-fdm", "rk4-fdm-fused", "rk4-cuda"):
        assert np.array_equal(one.state.psi, two.state.psi[0])


@pytest.mark.parametrize("method", ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda",
                                    "split-step-fft", "etd-rk2", "ip-rk4",
                                    "ifrk-fft-cuda", "ifrk4-fft-cuda"])
def test_1d_batch_members_match_separate(method):
    cfg = configuration(method)
    members = [{"physics.g_C": 0.001}, {"physics.g_C": 0.002}]
    batch = SimulationController(cfg, members=members)
    separate = []
    for member in members:
        single_cfg = configuration(method)
        single_cfg.physics.g_C = member["physics.g_C"]
        separate.append(SimulationController(single_cfg))
    for run in (batch, *separate):
        for _ in range(3):
            run.solver.step(run.potential, run.pump_field, run.reservoir,
                            run.boundary_condition, run.state)
            run.state.t += run.cfg.solver.dt
    for i, run in enumerate(separate):
        np.testing.assert_allclose(batch.state.psi[i], run.state.psi, rtol=1e-12, atol=1e-12)


def test_1d_checkpoint_and_hdf5(tmp_path):
    cfg = configuration()
    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.save_npy = True
    cfg.result.save_psi = True
    cfg.result.output_directory = str(tmp_path)
    run = SimulationController(cfg)
    run.run()
    checkpoint = tmp_path / "checkpoint.h5"
    run.save_checkpoint(checkpoint)
    with h5py.File(run.run_directory / "results.h5") as h5:
        assert h5["fields/psi"].shape == (5, 1, 32)
        assert h5["grid/x"].shape == (32,)
        assert "grid/y" not in h5
    with h5py.File(checkpoint) as h5:
        assert h5["field"].shape == (1, 1, 32)
        assert h5["reservoir/nR"].shape == (1, 1, 32)
    with np.load(next(run.run_directory.glob("batch_*.npz"))) as npy:
        assert npy["field_psi"].shape == (5, 32)
    restored = SimulationController(cfg)
    restored.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.state.psi, run.state.psi)
    np.testing.assert_array_equal(restored.reservoir.get_state()[0], run.reservoir.get_state()[0])


@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("batched", [False, True])
def test_1d_cuda_source_is_flat(boundary, reservoir, batched):
    sources = build_flat_1d_sources(boundary, reservoir_type=reservoir, batched=batched,
                                    physics_fields=("hbar", "g_C") if batched else ())
    assert len(sources) == 3
    for source in sources:
        body = source.split(") {", 1)[1]
        assert "const int lt =" in body and "const int rt =" in body
        assert "const int up =" not in body and "const int dn =" not in body
        assert "* inv_dy2" not in body
        assert "__shared__" not in body
