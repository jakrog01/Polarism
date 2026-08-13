from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import grid, small_config

TOL_PHASE_ROT = 1e-10  # Exact algebraic identity for a uniform nonlinear field.
TOL_NORM_COND = 1e-10  # Conservative-limit norm preservation over many steps.
TOL_DECAY_FIT = 1e-4  # Least-squares fit of ln condensate norm.


@pytest.mark.compliance
def test_uniform_phase_rotation() -> None:
    """A uniform field evolves as psi(t)=psi(0) exp(-i g_C |psi|² t / hbar)."""
    compute_engine.xp = np
    cfg = small_config(
        physics__gamma_C=0.0,
        physics__gamma_R=0.0,
        physics__gamma_I=0.0,
        physics__gamma_A=0.0,
        physics__R=0.0,
        physics__R_IA=0.0,
        physics__R_AI=0.0,
        physics__g_C=0.2,
        physics__g_R=0.0,
        physics__g_I=0.0,
        laser__P0=0.0,
        physics__init_eps=1.0,
        physics__init_mode="legacy_positive_uniform",
        solver__total_time=0.02,
    )
    controller = SimulationController(cfg); controller.state.psi.fill(1.0 + 0j); controller.run()
    expected = np.exp(-1j * cfg.physics.g_C * cfg.solver.total_time / cfg.physics.hbar)
    assert np.max(np.abs(controller.state.psi - expected)) < TOL_PHASE_ROT


@pytest.mark.compliance
def test_pure_decay_rate_fit() -> None:
    """For zero pump and interactions N(t)=N(0) exp(-gamma_C t)."""
    compute_engine.xp = np
    cfg = small_config(physics__gamma_C=0.083, physics__g_C=0.0, physics__g_R=0.0, physics__R=0.0, solver__total_time=0.2)
    controller = SimulationController(cfg); controller.state.psi.fill(1.0 + 0j)
    norms, times = [], []
    for _ in range(20):
        controller.solver.step(controller.potential, np.zeros_like(controller.grid.X), controller.reservoir, controller.boundary_condition, controller.state)
        controller.state.t += cfg.solver.dt; times.append(controller.state.t); norms.append(float(np.sum(np.abs(controller.state.psi) ** 2)))
    slope = np.polyfit(times, np.log(norms), 1)[0]
    assert abs(slope + cfg.physics.gamma_C) < TOL_DECAY_FIT


@pytest.mark.compliance
def test_conservative_uniform_norm() -> None:
    """With gamma_C=R=P=0 the RK4-FDM conservative limit preserves norm."""
    compute_engine.xp = np
    cfg = small_config(physics__gamma_C=0.0, physics__R=0.0, physics__g_C=0.1, solver__dt=1e-3, solver__total_time=0.1)
    controller = SimulationController(cfg); controller.state.psi.fill(1.0 + 0j); initial = np.sum(np.abs(controller.state.psi) ** 2); controller.run()
    assert abs(np.sum(np.abs(controller.state.psi) ** 2) - initial) / initial < TOL_NORM_COND


@pytest.mark.compliance
def test_quadratic_double_zero_dimensional_stationary_state() -> None:
    """The zero-dimensional quadratic reservoir tends to its closed-form stationary state."""
    compute_engine.xp = np
    physics = PhysicsConstants(gamma_R=0.005, gamma_I=0.001, kappa=0.05, R=0.0)
    reservoir = QuadraticDoubleReservoir(ReservoirParameters(expose_results=False), physics, grid(1, 1))
    pump = np.full((1, 1), 0.03); psi = np.zeros((1, 1), complex)
    for _ in range(20_000):
        reservoir.step(0.1, psi, pump)
    n_i = (-physics.gamma_I + np.sqrt(physics.gamma_I**2 + 4 * physics.kappa * 0.03)) / (2 * physics.kappa)
    assert np.isclose(reservoir.nI[0, 0], n_i, rtol=1e-3)
    assert np.isclose(reservoir.nR[0, 0], physics.kappa * n_i**2 / physics.gamma_R, rtol=1e-3)


@pytest.mark.compliance
def test_free_wave_packet_broadens_analytically() -> None:
    compute_engine.xp = np
    sigma_0 = 4.0
    total_time = 2.0
    cfg = small_config(
        grid__nx=128,
        grid__ny=128,
        grid__lx=160.0,
        grid__ly=160.0,
        grid__grid_type="periodic",
        solver__method="rk4-fdm",
        solver__dt=1e-3,
        solver__total_time=total_time,
        physics__g_C=0.0,
        physics__g_R=0.0,
        physics__g_I=0.0,
        physics__R=0.0,
        physics__R_IA=0.0,
        physics__R_AI=0.0,
        physics__gamma_C=0.0,
        physics__gamma_R=0.0,
        physics__gamma_I=0.0,
        physics__gamma_A=0.0,
        laser__P0=0.0,
    )
    sigma_final = np.sqrt(
        sigma_0**2 + (cfg.physics.hbar * total_time / (2 * cfg.physics.m_eff * sigma_0)) ** 2
    )
    assert 6 * sigma_final < cfg.grid.lx / 2, (
        f"6*sigma_T={6 * sigma_final}, lx/2={cfg.grid.lx / 2}"
    )
    controller = SimulationController(cfg)
    psi = np.exp(-(controller.grid.X**2 + controller.grid.Y**2) / (2 * sigma_0**2))
    controller.state.psi[:] = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    def width_sq() -> float:
        rho = np.abs(controller.state.psi) ** 2
        total = np.sum(rho)
        xbar = np.sum(controller.grid.X * rho) / total
        ybar = np.sum(controller.grid.Y * rho) / total
        return float(
            np.sum(((controller.grid.X - xbar) ** 2 + (controller.grid.Y - ybar) ** 2) * rho)
            / (2 * total)
        )

    times = [0.0]
    widths = [width_sq()]
    pump = np.zeros_like(controller.grid.X)
    n_steps = int(total_time / cfg.solver.dt)
    for step in range(n_steps):
        controller.solver.step(
            controller.potential,
            pump,
            controller.reservoir,
            controller.boundary_condition,
            controller.state,
        )
        controller.state.t += cfg.solver.dt
        if (step + 1) % 30 == 0 or step + 1 == n_steps:
            times.append(controller.state.t)
            widths.append(width_sq())
    sigma_rho_0_sq = sigma_0**2 / 2
    expected = np.array([
        sigma_rho_0_sq
        + (cfg.physics.hbar * time / (2 * cfg.physics.m_eff * np.sqrt(sigma_rho_0_sq))) ** 2
        for time in times
    ])
    assert np.max(np.abs(np.asarray(widths) - expected) / expected) < 1e-2
    final_density = np.abs(controller.state.psi) ** 2
    boundary = np.concatenate((final_density[:, 0], final_density[0, :]))
    assert np.max(boundary) < 1e-8 * np.max(final_density)
