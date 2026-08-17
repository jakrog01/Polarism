from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.simulation_controller import SimulationController
from tests._reporting import write_validation_record
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


@pytest.mark.compliance
def test_closed_interval_neumann_eigenmode_phase() -> None:
    """A discrete Neumann cosine eigenmode rotates at its known phase rate.

    The calibration error is 1.89e-15; the 1e-10 limit admits accumulated
    float64 RK4 roundoff while remaining five orders above the measurement.
    """
    compute_engine.xp = np
    cfg = small_config(
        grid__nx=64,
        grid__ny=16,
        grid__lx=40.0,
        grid__ly=10.0,
        grid__grid_type="closed-interval",
        solver__method="rk4-fdm",
        solver__dt=1e-4,
        solver__total_time=0.05,
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
    controller = SimulationController(cfg)
    initial = np.cos(np.pi * (controller.grid.X + cfg.grid.lx / 2.0) / cfg.grid.lx).astype(
        np.complex128
    )
    controller.state.psi[:] = initial
    controller.run()
    discrete_wave_number_squared = (
        4.0
        * np.sin(np.pi / (2.0 * (cfg.grid.nx - 1))) ** 2
        / controller.grid.dx**2
    )
    energy = cfg.physics.hbar**2 * discrete_wave_number_squared / (2.0 * cfg.physics.m_eff)
    expected = initial * np.exp(-1j * energy * cfg.solver.total_time / cfg.physics.hbar)
    complex_error = float(
        np.linalg.norm((controller.state.psi - expected).ravel())
        / np.linalg.norm(expected.ravel())
    )
    modulus_error = float(np.max(np.abs(np.abs(controller.state.psi) - np.abs(initial))))
    measured = max(complex_error, modulus_error)
    threshold = 1e-10
    write_validation_record(
        Path("closed_interval_neumann_eigenmode.json"),
        error_norm="max(relative_l2_complex,absolute_modulus_linf)",
        measured_value=measured,
        threshold=threshold,
        passed=measured < threshold,
        precision="fp64",
        grid={
            "nx": cfg.grid.nx,
            "ny": cfg.grid.ny,
            "lx": cfg.grid.lx,
            "ly": cfg.grid.ly,
            "dx": controller.grid.dx,
            "grid_type": cfg.grid.grid_type,
        },
        dt=cfg.solver.dt,
        total_time=cfg.solver.total_time,
        n_steps=int(cfg.solver.total_time / cfg.solver.dt),
        solver_reference="discrete-neumann-eigenmode",
        solver_under_test="rk4-fdm",
        backend_reference="analytic",
        backend_under_test="cpu",
        reservoir_type="single",
        boundary="closed-interval-neumann",
        potential_type="zero",
        extra={"complex_error": complex_error, "modulus_error": modulus_error},
        artifact_root=Path("artifacts/reference"),
    )
    assert measured < threshold


@pytest.mark.compliance
def test_double_reservoir_zero_dimensional_closed_form() -> None:
    """The linear double reservoir matches its matrix-exponential solution.

    The measured combined stationary/transient error is 1.21e-8. The 1e-7
    limit provides an eightfold allowance for the midpoint time discretization.
    """
    compute_engine.xp = np
    physics = PhysicsConstants(
        gamma_I=0.03,
        gamma_A=0.08,
        R_IA=0.12,
        R_AI=0.02,
        R=0.0,
    )
    reservoir_cfg = ReservoirParameters(expose_results=False, reservoir_type="double")
    spatial_grid = grid(1, 1)
    pump_value = 0.03
    matrix = np.array(
        [
            [-(physics.gamma_A + physics.R_AI), physics.R_IA],
            [physics.R_AI, -(physics.gamma_I + physics.R_IA)],
        ],
        dtype=np.float64,
    )
    source = np.array([0.0, pump_value], dtype=np.float64)
    stationary = -np.linalg.solve(matrix, source)
    stationary_reservoir = DoubleReservoir(reservoir_cfg, physics, spatial_grid)
    stationary_reservoir.set_state(
        (np.array([[stationary[0]]]), np.array([[stationary[1]]]))
    )
    zero_psi = np.zeros((1, 1), dtype=np.complex128)
    pump = np.full((1, 1), pump_value)
    for _ in range(100):
        stationary_reservoir.step(0.01, zero_psi, pump)
    stationary_actual = np.array(
        [stationary_reservoir.nA[0, 0], stationary_reservoir.nI[0, 0]]
    )
    stationary_error = float(
        np.linalg.norm(stationary_actual - stationary) / np.linalg.norm(stationary)
    )

    transient = DoubleReservoir(reservoir_cfg, physics, spatial_grid)
    initial = np.array([0.04, 0.01], dtype=np.float64)
    transient.set_state((np.array([[initial[0]]]), np.array([[initial[1]]])))
    dt = 0.002
    n_steps = 500
    for _ in range(n_steps):
        transient.step(dt, zero_psi, pump)
    expected = stationary + expm(matrix * (dt * n_steps)) @ (initial - stationary)
    actual = np.array([transient.nA[0, 0], transient.nI[0, 0]])
    transient_error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    measured = max(stationary_error, transient_error)
    threshold = 1e-7
    write_validation_record(
        Path("double_reservoir_zero_dimensional.json"),
        error_norm="max(stationary_relative_l2,transient_relative_l2)",
        measured_value=measured,
        threshold=threshold,
        passed=measured < threshold,
        precision="fp64",
        grid={
            "nx": 1,
            "ny": 1,
            "lx": spatial_grid.lx,
            "ly": spatial_grid.ly,
            "dx": spatial_grid.dx,
            "grid_type": spatial_grid.grid_type,
        },
        dt=dt,
        total_time=dt * n_steps,
        n_steps=n_steps,
        solver_reference="scipy.linalg.expm",
        solver_under_test="DoubleReservoir.midpoint",
        backend_reference="analytic",
        backend_under_test="cpu",
        reservoir_type="double",
        boundary="zero-dimensional",
        potential_type="zero",
        extra={
            "stationary_relative_l2": stationary_error,
            "transient_relative_l2": transient_error,
        },
        artifact_root=Path("artifacts/reference"),
    )
    assert measured < threshold
