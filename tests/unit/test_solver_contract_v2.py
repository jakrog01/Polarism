"""Solver contract and reservoir-free evolution checks."""
from __future__ import annotations

import sys

import h5py
import numpy as np
import pytest

from polarism import Config, SimulationController, StepReport, SolverCapabilities
from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.validation import validate_config
from polarism.plugins import load_plugins
from polarism.solver import ACCEPTED
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from polarism.solver.solver_registry import available_solvers

CPU_SOLVERS = (
    "rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "split-step-fft",
    "ip-rk4", "etd-rk2", "ifrk-fft-cuda", "ifrk4-fft-cuda",
)
NONE_SOLVERS = ("rk4-fdm", "rk4-cuda", "split-step-fft", "ip-rk4", "etd-rk2")
FUSED_SOLVERS = ("rk4-fdm-fused", "rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda")


def config(method="rk4-fdm", reservoir="single", gpu=False):
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 16
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.solver.method = method
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.02
    cfg.solver.cuda_graphs = False
    cfg.reservoir.reservoir_type = reservoir
    cfg.laser.P0 = 0.0
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.noise.seed = 12
    cfg.noise.initial = "none"
    cfg.result.save_results = False
    cfg.compute_engine.use_gpu = gpu
    return cfg


@pytest.mark.parametrize("method", CPU_SOLVERS)
def test_builtin_step_returns_shared_report_and_updates_time(method):
    run = SimulationController(config(method))
    report = run.solver.step(run.model, run.state, run.cfg.solver.dt)
    assert report is ACCEPTED
    assert isinstance(report, StepReport)
    assert run.state.step == 1
    assert run.state.t == run.state.t0 + run.cfg.solver.dt


@pytest.mark.parametrize("method", NONE_SOLVERS)
def test_no_reservoir_gaussian_free_decay(method):
    cfg = config(method, "none")
    run = SimulationController(cfg)
    x, y = run.grid.X, run.grid.Y
    run.state.psi = np.exp(-(x*x + y*y) / 8.0).astype(np.complex128)
    initial_norm = np.sum(np.abs(run.state.psi)**2)
    assert run.reservoir.get_state() == ()
    assert run.reservoir.get_active_density(()) == 0
    assert run.reservoir.make_result_nodes() == []
    run.run()
    final_norm = np.sum(np.abs(run.state.psi)**2)
    expected = initial_norm * np.exp(-cfg.physics.gamma_C * cfg.solver.total_time)
    assert np.isclose(final_norm, expected, rtol=5e-4)


@pytest.mark.parametrize("method", ("rk4-fdm-fused",))
def test_fused_solver_rejects_no_reservoir_at_validation(method):
    cfg = config(method, "none")
    with pytest.raises(SolverCompatibilityError, match="reservoir_type='none'"):
        check_solver_compatibility(cfg)


def test_no_reservoir_checkpoint_has_empty_group(tmp_path):
    cfg = config(reservoir="none")
    run = SimulationController(cfg)
    path = tmp_path / "none.h5"
    run.save_checkpoint(path)
    with h5py.File(path) as handle:
        assert list(handle["reservoir"]) == []
    restored = SimulationController(cfg)
    restored.load_checkpoint(path)
    assert restored.reservoir.get_state() == ()


def test_no_reservoir_wigner_variance_uses_condensate_decay():
    cfg = config(reservoir="none")
    cfg.noise.stochastic = "wigner"
    run = SimulationController(cfg)
    run.solver.advance(run.model, run.state, 1)
    measured = float(np.mean(np.abs(run.state.psi)**2))
    expected = cfg.physics.gamma_C * cfg.solver.dt / (2 * run.grid.dx * run.grid.dy)
    assert np.isclose(measured, expected, rtol=0.25)


def test_plugin_capabilities_are_checked_at_validation(monkeypatch):
    class Restricted:
        capabilities = SolverCapabilities(
            components=frozenset({1}), ndim=frozenset({2}),
            grid_types=frozenset({"periodic"}), reservoirs=frozenset({"none"}),
            custom_reservoirs="unsupported", potential=False,
            kinetic_relaxation=False, dispersion=frozenset({"parabolic"}),
            noise=frozenset({"none"}),
        )

    monkeypatch.setitem(available_solvers, "restricted-test", Restricted)
    cfg = config("restricted-test", "single")
    with pytest.raises(SolverCompatibilityError, match="reservoir_type='single'"):
        check_solver_compatibility(cfg)
    cfg.reservoir.reservoir_type = "none"
    validate_config(cfg)
    check_solver_compatibility(cfg)
    cfg.grid.ndim = 1
    with pytest.raises(ValueError, match="grid.ndim"):
        validate_config(cfg)


def test_plugin_without_capabilities_warns_and_uses_defaults(monkeypatch):
    class Undeclared:
        pass

    monkeypatch.setitem(available_solvers, "undeclared-test", Undeclared)
    cfg = config("undeclared-test", "none")
    with pytest.warns(UserWarning, match="does not declare capabilities"):
        check_solver_compatibility(cfg)
    cfg.grid.ndim = 1
    with pytest.warns(UserWarning, match="does not declare capabilities"):
        with pytest.raises(ValueError, match="grid.ndim"):
            validate_config(cfg)


def test_plugin_api_mismatch_rejected(tmp_path, monkeypatch):
    root = tmp_path / "polarism_plugins"
    root.mkdir()
    (root / "incompatible.py").write_text("POLARISM_PLUGIN_API = 1\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(RuntimeError, match="POLARISM_PLUGIN_API=1.*requires 2"):
        load_plugins(tmp_path)
    for name in list(sys.modules):
        if name == "polarism_plugins" or name.startswith("polarism_plugins."):
            monkeypatch.delitem(sys.modules, name)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ("split-step-fft-cuda",))
def test_gpu_step_report(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    run = SimulationController(config(method, gpu=True))
    assert run.solver.step(run.model, run.state, run.cfg.solver.dt) is ACCEPTED
    assert run.state.step == 1


@pytest.mark.gpu
@pytest.mark.parametrize("method", NONE_SOLVERS)
def test_gpu_no_reservoir_decay(method):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = config(method, "none", gpu=True)
    run = SimulationController(cfg)
    x, y = run.grid.X, run.grid.Y
    run.state.psi = compute_engine.xp.exp(-(x*x + y*y) / 8.0).astype(compute_engine.xp.complex128)
    before = float(compute_engine.to_cpu(compute_engine.xp.sum(compute_engine.xp.abs(run.state.psi)**2)))
    run.run()
    after = float(compute_engine.to_cpu(compute_engine.xp.sum(compute_engine.xp.abs(run.state.psi)**2)))
    assert np.isclose(after, before * np.exp(-cfg.physics.gamma_C * cfg.solver.total_time), rtol=5e-4)
