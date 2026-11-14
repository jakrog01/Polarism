"""Temporal potential envelopes and an unmodulated absorber."""
from __future__ import annotations

import numpy as np
import pytest

from polarism import SimulationController
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from polarism.sweep_config import apply_overrides, override_kind
from tests.unit.conftest import small_config


POTENTIAL_SOLVERS = ("rk4-fdm", "ip-rk4", "etd-rk2", "split-step-fft")
NO_DYNAMIC_SOLVERS = ("rk4-fdm-fused", "rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda")


def config(method="rk4-fdm"):
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 8.0
    cfg.solver.method = method
    cfg.solver.dt = 0.0005
    cfg.solver.total_time = 0.2
    cfg.reservoir.reservoir_type = "none"
    cfg.laser.P0 = 0.0
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = cfg.physics.gamma_C = 0.0
    cfg.noise.enabled = False
    cfg.potential.potential_type = "profile"
    cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
    cfg.potential.envelope = {"type": "step", "t0": 0.05}
    return cfg


@pytest.mark.parametrize("method", POTENTIAL_SOLVERS)
def test_step_potential_rotates_uniform_field_after_onset(method):
    cfg = config(method)
    run = SimulationController(cfg)
    run.state.psi = np.ones(run.grid.shape, dtype=np.complex128)
    run.run()
    expected = np.exp(-1j * 0.4 * (cfg.solver.total_time - 0.05) / cfg.physics.hbar)
    np.testing.assert_allclose(run.state.psi, expected, rtol=0, atol=2e-4)
    np.testing.assert_allclose(run.model.potential_at(0.0), 0.0)
    np.testing.assert_allclose(run.model.potential_at(0.1), 0.4)


def test_cap_remains_active_when_configured_potential_is_off():
    cfg = config()
    cfg.grid.nx = cfg.grid.ny = 32
    cfg.grid.lx = cfg.grid.ly = 32.0
    cfg.boundary_condition.absorption = "cap"
    run = SimulationController(cfg)
    cap = run.boundary_condition.before_step_action()
    np.testing.assert_allclose(run.model.potential_at(0.0), cap)
    np.testing.assert_allclose(run.model.potential_at(0.1), cap + run.model.potential_static)
    assert np.any(np.imag(cap) != 0)


@pytest.mark.parametrize("method", NO_DYNAMIC_SOLVERS)
def test_dynamic_potential_is_rejected_by_unsupported_solvers(method):
    cfg = config(method)
    cfg.reservoir.reservoir_type = "single"
    with pytest.raises(SolverCompatibilityError, match="time-dependent potential"):
        check_solver_compatibility(cfg)


def test_envelope_validation_and_sweep_paths():
    cfg = config()
    cfg.potential.envelope = {"type": "missing"}
    with pytest.raises(ConfigValidationError, match="potential.envelope"):
        validate_config(cfg)
    cfg.potential.envelope = {"type": "step", "t0": 0.05}
    assert override_kind("potential.envelope.type") == "group"
    assert override_kind("potential.envelope.t0") == "batch"
    changed = apply_overrides(cfg, {"potential.envelope.t0": 0.08})
    assert changed.potential.envelope["t0"] == 0.08
    assert cfg.potential.envelope["t0"] == 0.05


def test_batched_envelopes_keep_each_members_potential():
    cfg = config()
    run = SimulationController(cfg, members=[
        {"potential.profile.amplitude": 0.4, "potential.envelope.t0": 0.05},
        {"potential.profile.amplitude": 0.8, "potential.envelope.t0": 0.1},
    ])
    np.testing.assert_allclose(run.model.potential_at(0.075)[:, 0, 0, 0], [0.4, 0.0])
    np.testing.assert_allclose(run.model.potential_at(0.15)[:, 0, 0, 0], [0.4, 0.8])


def test_constant_envelope_uses_legacy_potential_array():
    cfg = config()
    cfg.potential.envelope = {"type": "constant"}
    run = SimulationController(cfg)
    assert not run.model.time_dependent_potential
    assert run.model.potential_at(0.0) is run.model.potential
