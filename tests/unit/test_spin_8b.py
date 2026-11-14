"""CPU checks for spectral spinor propagation and reservoir coupling."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController


METHODS = ("split-step-fft", "ip-rk4", "etd-rk2")


def config(method, *, ndim=1, reservoir="none", dt=0.002, total_time=0.02):
    cfg = Config()
    cfg.grid.ndim = ndim
    cfg.grid.nx = 16
    cfg.grid.ny = 16
    cfg.grid.lx = 32.0
    cfg.grid.ly = 32.0
    cfg.physics.spin = "spinor"
    cfg.physics.hbar = 1.0
    cfg.physics.m_eff = 1.0
    cfg.physics.gamma_C = 0.0
    cfg.physics.g_C = 0.0
    cfg.physics.g_R = 0.0
    cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.physics.zeeman = 0.23
    cfg.physics.te_tm = 0.31
    cfg.noise.enabled = False
    cfg.reservoir.reservoir_type = reservoir
    cfg.laser.P0 = 0.0
    cfg.solver.method = method
    cfg.solver.dt = dt
    cfg.solver.total_time = total_time
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_linear_plane_wave_matches_two_by_two_exponential(method):
    cfg = config(method, ndim=2)
    run = SimulationController(cfg)
    kx = 2 * np.pi / cfg.grid.lx
    ky = 4 * np.pi / cfg.grid.ly
    wave = np.exp(1j * (kx * run.grid.X + ky * run.grid.Y))
    initial = np.array([0.7 + 0.2j, -0.3 + 0.4j])
    run.set_initial_state(initial[:, None, None] * wave)
    run.solver.advance(run.model, run.state, 10)
    epsilon = (kx * kx + ky * ky) / 2
    z = cfg.physics.zeeman / 2
    upper = cfg.physics.te_tm * (kx - 1j * ky) ** 2
    lower = np.conj(upper)
    omega = np.sqrt(z * z + upper * lower)
    t = cfg.solver.total_time
    expected = np.exp(-1j * epsilon * t) * (
        np.cos(omega * t) * initial
        - 1j * np.sin(omega * t) / omega
        * np.array([z * initial[0] + upper * initial[1],
                    lower * initial[0] - z * initial[1]])
    )
    np.testing.assert_allclose(run.state.psi, expected[:, None, None] * wave, rtol=0, atol=2e-14)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reservoir_components", (1, 2))
def test_nonlinear_reservoir_spinor_agrees_with_fdm_and_refines(method, reservoir_components):
    errors = []
    solutions = []
    for dt in (0.02, 0.01, 0.005):
        cfg = config(method, reservoir="single", dt=dt, total_time=0.1)
        cfg.reservoir.components = reservoir_components
        cfg.laser.laser_type = "uniform"
        cfg.laser.P0 = 0.3
        cfg.laser.circular_polarization = 0.4
        cfg.physics.g_C = 0.4
        cfg.physics.alpha_2 = -0.18
        cfg.physics.g_R = 0.25
        cfg.physics.R = 0.1
        cfg.physics.gamma_C = 0.05
        wave = np.exp(2j * np.pi * np.arange(16) / 16)
        initial = np.array([0.7 + 0.1j, 0.2 - 0.35j])[:, None] * wave
        spectral = SimulationController(cfg)
        spectral.set_initial_state(initial)
        spectral.solver.advance(spectral.model, spectral.state, round(0.1 / dt))
        solutions.append(spectral.state.psi.copy())
        cfg.solver.method = "rk4-fdm"
        fdm = SimulationController(cfg)
        fdm.set_initial_state(initial)
        fdm.solver.advance(fdm.model, fdm.state, round(0.1 / dt))
        errors.append(np.max(np.abs(spectral.state.psi - fdm.state.psi)))
        np.testing.assert_allclose(spectral.reservoir.get_state()[0],
                                   fdm.reservoir.get_state()[0], rtol=0, atol=2e-5)
    assert max(errors) < 5e-5
    assert errors[1] <= errors[0] + 2e-6
    coarse_change = np.max(np.abs(solutions[0] - solutions[2]))
    fine_change = np.max(np.abs(solutions[1] - solutions[2]))
    assert coarse_change > (8 if method == "ip-rk4" else 3) * fine_change


@pytest.mark.parametrize("method", METHODS)
def test_scalar_path_matches_reference_values(method):
    cfg = config(method)
    cfg.physics.spin = "scalar"
    cfg.physics.zeeman = 0.0
    cfg.physics.te_tm = 0.0
    run = SimulationController(cfg)
    wave = np.exp(2j * np.pi * np.arange(16) / 16)
    run.set_initial_state(wave)
    run.solver.advance(run.model, run.state, 10)
    k = 2 * np.pi / cfg.grid.lx
    np.testing.assert_allclose(run.state.psi, wave * np.exp(-1j * k * k * cfg.solver.total_time / 2),
                               rtol=0, atol=1e-14)
