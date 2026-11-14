"""Coherent source, rotating frame, and capability checks."""
from __future__ import annotations

import numpy as np
import pytest

from polarism import SimulationController
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from polarism.sweep_config import apply_overrides, override_kind
from tests.unit.conftest import small_config


DRIVE_SOLVERS = ("rk4-fdm", "ip-rk4", "etd-rk2")
NO_DRIVE_SOLVERS = ("rk4-fdm-fused", "rk4-cuda", "split-step-fft", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda")


def config(method="rk4-fdm"):
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 8.0
    cfg.solver.method = method
    cfg.solver.dt = 0.02
    cfg.solver.total_time = 10.0
    cfg.reservoir.reservoir_type = "none"
    cfg.laser.P0 = 0.0
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.physics.gamma_C = 2.0
    cfg.noise.enabled = False
    cfg.drive.enabled = True
    cfg.drive.amplitude = 0.5
    cfg.drive.energy = 0.4
    return cfg


@pytest.mark.parametrize("method", DRIVE_SOLVERS)
def test_uniform_drive_reaches_detuned_steady_state(method):
    cfg = config(method)
    run = SimulationController(cfg)
    run.run()
    hbar = cfg.physics.hbar
    steady = (-1j * cfg.drive.amplitude / hbar) / (cfg.physics.gamma_C / 2 - 1j * cfg.drive.energy / hbar)
    np.testing.assert_allclose(np.abs(run.state.psi) ** 2, abs(steady) ** 2, rtol=3e-4)
    assert run.model.drive is not None
    nodes = {node.name: node for node in run._build_result_nodes()}
    assert nodes["drive"].static
    np.testing.assert_allclose(nodes["drive"].compute_field_device(), cfg.drive.amplitude)


@pytest.mark.parametrize("method", DRIVE_SOLVERS)
def test_plane_wave_drive_sets_spatial_phase(method):
    cfg = config(method)
    cfg.grid.nx = cfg.grid.ny = 16
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.1
    cfg.physics.gamma_C = 0.0
    cfg.drive.energy = 0.0
    cfg.drive.kx = 2 * np.pi / cfg.grid.lx
    run = SimulationController(cfg)
    run.run()
    expected = np.exp(1j * cfg.drive.kx * run.grid.X)
    field = run.state.psi
    np.testing.assert_allclose(field / field[0, 0], expected / expected[0, 0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("method", DRIVE_SOLVERS)
def test_disabled_drive_leaves_evolution_identical(method):
    cfg = config(method)
    cfg.drive.enabled = False
    cfg.solver.total_time = 0.1
    reference = SimulationController(cfg)
    reference.state.psi = np.ones(reference.grid.shape, dtype=np.complex128)
    reference.run()
    cfg.drive.amplitude = 4.0
    cfg.drive.energy = 3.0
    disabled = SimulationController(cfg)
    disabled.state.psi = np.ones(disabled.grid.shape, dtype=np.complex128)
    disabled.run()
    np.testing.assert_array_equal(disabled.state.psi, reference.state.psi)
    assert disabled.model.drive is None
    assert "drive" not in {node.name for node in disabled._build_result_nodes()}


@pytest.mark.parametrize("method", NO_DRIVE_SOLVERS)
def test_drive_is_rejected_by_unsupported_solvers(method):
    cfg = config(method)
    cfg.reservoir.reservoir_type = "single"
    with pytest.raises(SolverCompatibilityError, match="drive.enabled"):
        check_solver_compatibility(cfg)


def test_drive_validation_and_sweep_paths():
    cfg = config()
    cfg.drive.profile = {"type": "missing"}
    with pytest.raises(ConfigValidationError, match="drive profile/envelope"):
        validate_config(cfg)
    cfg.drive.profile = {"type": "uniform"}
    cfg.drive.envelope = {"type": "step", "t0": 1.0}
    assert override_kind("drive.enabled") == "group"
    assert override_kind("drive.amplitude") == "batch"
    assert override_kind("drive.energy") == "batch"
    assert override_kind("drive.kx") == "batch"
    assert override_kind("drive.ky") == "batch"
    assert override_kind("drive.profile.type") == "group"
    assert override_kind("drive.envelope.type") == "group"
    assert override_kind("drive.envelope.t0") == "batch"
    changed = apply_overrides(cfg, {"drive.amplitude": 0.8, "drive.envelope.t0": 2.0})
    assert changed.drive.amplitude == 0.8
    assert changed.drive.envelope["t0"] == 2.0
    assert cfg.drive.amplitude == 0.5


def test_batched_drive_spatial_energy_and_envelope():
    cfg = config()
    cfg.drive.envelope = {"type": "step", "t0": 1.0}
    run = SimulationController(cfg, members=[
        {"drive.amplitude": 0.5, "drive.energy": 0.2, "drive.envelope.t0": 1.0},
        {"drive.amplitude": 1.0, "drive.energy": 0.4, "drive.envelope.t0": 2.0},
    ])
    np.testing.assert_allclose(run.model.drive.spatial[:, 0, 0, 0], [0.5, 1.0])
    np.testing.assert_allclose(run.model.drive.energy[:, 0, 0, 0], [0.2, 0.4])
    np.testing.assert_allclose(run.model.drive.envelope(1.5)[:, 0, 0, 0], [1.0, 0.0])


def test_drive_step_envelope_switches_source():
    cfg = config()
    cfg.drive.envelope = {"type": "step", "t0": 0.05}
    run = SimulationController(cfg)
    np.testing.assert_allclose(run.model.drive_rhs(0.0), 0.0)
    np.testing.assert_allclose(run.model.drive_rhs(0.05), -1j * cfg.drive.amplitude / cfg.physics.hbar)
