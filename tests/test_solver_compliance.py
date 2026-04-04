"""Solver compliance tests."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.solver.etd_rk2_solver import ETDRK2Solver
from polarism.solver.ip_rk4_solver import IPRK4Solver
from polarism.solver.rk4_cuda_solver import RK4CudaSolver
from polarism.solver.rk4_fdm_fused_solver import RK4FDMFusedSolver
from polarism.solver.rk4_fdm_solver import RK4FDMSolver
from polarism.solver.split_step_fft_solver import SplitStepFFTSolver
from tests._helpers import (
    GridCfg,
    NoBoundaryCondition,
    State,
    make_config,
    make_grid,
    make_physics_default,
    make_reservoir_cfg,
    potential_zero,
    pump_uniform,
)

pytestmark = pytest.mark.compliance

RESULTS_DIR = Path(__file__).resolve().parent / "test_results" / "solver_compliance"

PERIODIC_SOLVERS = [
    RK4FDMFusedSolver,
    RK4CudaSolver,
    SplitStepFFTSolver,
    IPRK4Solver,
    ETDRK2Solver,
]

CLOSED_INTERVAL_SOLVERS = [
    RK4FDMFusedSolver,
    RK4CudaSolver,
]


def _save_fig(outdir: Path, name: str) -> None:
    """Save the current figure."""
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / name, dpi=160, bbox_inches="tight")
    plt.close()


def _run_solver(solver_cls, grid_cfg, physics, dt, n_steps, P0):
    """Run solver."""
    grid = make_grid(grid_cfg)
    cfg = make_config(dt=dt, physics=physics, grid_type=grid_cfg.grid_type)
    solver = solver_cls(cfg, grid)
    reservoir = SingleReservoir(make_reservoir_cfg(False), physics, grid)
    bc = NoBoundaryCondition()

    psi0 = 1e-3 * np.ones((grid.ny, grid.nx), dtype=np.complex128)
    state = State(psi0)
    V = potential_zero(grid.X)
    P = pump_uniform(P0, grid.X)

    sample_stride = max(1, n_steps // 50)
    times = []
    rho_max = []
    for step in range(n_steps):
        solver.step(V, P, reservoir, bc, state)
        if (step + 1) % sample_stride == 0:
            times.append((step + 1) * dt)
            rho_max.append(float(np.max(np.abs(state.psi) ** 2)))

    return np.array(times), np.array(rho_max), state.psi


def _log_correlation(rho_a, rho_b):
    """Compute the log-scale correlation between two fields."""
    floor = 1e-30
    log_a = np.log10(np.maximum(rho_a, floor))
    log_b = np.log10(np.maximum(rho_b, floor))
    return float(np.corrcoef(log_a, log_b)[0, 1])


def _plot_rho_comparison(
    outdir: Path,
    name: str,
    t_ref, rho_ref, ref_label,
    t_test, rho_test, test_label,
    corr: float,
):
    """Plot rho comparison."""
    floor = 1e-30
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(t_ref, rho_ref, "b-", label=ref_label, linewidth=1.5)
    axes[0].semilogy(t_test, rho_test, "r--", label=test_label, linewidth=1.2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"max($|\psi|^2$)")
    axes[0].set_title(f"{name} (corr={corr:.6f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rho_test_interp = np.interp(t_ref, t_test, rho_test)
    rel_err = np.abs(rho_test_interp - rho_ref) / np.maximum(rho_ref, floor)
    axes[1].semilogy(t_ref, rel_err, "k-", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Relative Error vs Time")
    axes[1].grid(True, alpha=0.3)

    _save_fig(outdir, f"{name}.png")


@pytest.mark.parametrize(
    "solver_cls",
    PERIODIC_SOLVERS,
    ids=lambda cls: cls.__name__,
)
def test_periodic_matches_reference(solver_cls):
    """Test that periodic matches reference."""
    grid_cfg = GridCfg(nx=64, ny=64, lx=200.0, ly=200.0, grid_type="periodic")
    physics = make_physics_default()
    dt = 0.01
    n_steps = 5000
    P0 = 0.01

    t_ref, ref_rho, _ = _run_solver(RK4FDMSolver, grid_cfg, physics, dt, n_steps, P0)
    t_test, test_rho, _ = _run_solver(solver_cls, grid_cfg, physics, dt, n_steps, P0)

    corr = _log_correlation(ref_rho, test_rho)

    _plot_rho_comparison(
        RESULTS_DIR / "periodic",
        f"periodic_{solver_cls.__name__}_vs_RK4FDM",
        t_ref, ref_rho, "RK4FDMSolver",
        t_test, test_rho, solver_cls.__name__,
        corr,
    )

    assert corr > 0.999, (
        f"{solver_cls.__name__} correlation {corr:.6f} vs RK4FDMSolver"
    )


@pytest.mark.parametrize(
    "solver_cls",
    CLOSED_INTERVAL_SOLVERS,
    ids=lambda cls: cls.__name__,
)
def test_closed_interval_matches_reference(solver_cls):
    """Test that closed interval matches reference."""
    grid_cfg = GridCfg(
        nx=64, ny=64, lx=200.0, ly=200.0, grid_type="closed-interval"
    )
    physics = make_physics_default()
    dt = 0.01
    n_steps = 5000
    P0 = 0.01

    t_ref, ref_rho, _ = _run_solver(RK4FDMSolver, grid_cfg, physics, dt, n_steps, P0)
    t_test, test_rho, _ = _run_solver(solver_cls, grid_cfg, physics, dt, n_steps, P0)

    corr = _log_correlation(ref_rho, test_rho)

    _plot_rho_comparison(
        RESULTS_DIR / "closed_interval",
        f"closed_{solver_cls.__name__}_vs_RK4FDM",
        t_ref, ref_rho, "RK4FDMSolver",
        t_test, test_rho, solver_cls.__name__,
        corr,
    )

    assert corr > 0.999, (
        f"{solver_cls.__name__} correlation {corr:.6f} vs RK4FDMSolver"
    )


@pytest.mark.parametrize(
    "solver_cls",
    [RK4FDMSolver, RK4FDMFusedSolver, SplitStepFFTSolver, IPRK4Solver, ETDRK2Solver],
    ids=lambda cls: cls.__name__,
)
def test_double_reservoir_matches_single(solver_cls):
    """Test that double reservoir matches single."""
    grid_cfg = GridCfg(nx=64, ny=64, lx=200.0, ly=200.0, grid_type="periodic")
    grid = make_grid(grid_cfg)
    physics = make_physics_default()
    dt = 0.01
    n_steps = 3000
    P0 = 0.01

    cfg = make_config(dt=dt, physics=physics, grid_type="periodic")

    V = potential_zero(grid.X)
    P = pump_uniform(P0, grid.X)
    bc = NoBoundaryCondition()

    psi0 = 1e-3 * np.ones((grid.ny, grid.nx), dtype=np.complex128)
    sample_stride = max(1, n_steps // 50)

    solver_s = solver_cls(cfg, grid)
    res_s = SingleReservoir(make_reservoir_cfg(False), physics, grid)
    state_s = State(psi0.copy())

    solver_d = solver_cls(cfg, grid)
    res_d = DoubleReservoir(make_reservoir_cfg(False), physics, grid)
    state_d = State(psi0.copy())

    times = []
    rho_single = []
    rho_double = []
    for step in range(n_steps):
        solver_s.step(V, P, res_s, bc, state_s)
        solver_d.step(V, P, res_d, bc, state_d)
        if (step + 1) % sample_stride == 0:
            times.append((step + 1) * dt)
            rho_single.append(float(np.max(np.abs(state_s.psi) ** 2)))
            rho_double.append(float(np.max(np.abs(state_d.psi) ** 2)))

    times = np.array(times)
    rho_single = np.array(rho_single)
    rho_double = np.array(rho_double)
    corr = _log_correlation(rho_single, rho_double)

    _plot_rho_comparison(
        RESULTS_DIR / "reservoir",
        f"reservoir_{solver_cls.__name__}_single_vs_double",
        times, rho_single, "SingleReservoir",
        times, rho_double, "DoubleReservoir",
        corr,
    )

    assert corr > 0.99, (
        f"{solver_cls.__name__} single vs double reservoir correlation {corr:.6f}"
    )


@pytest.mark.parametrize(
    "solver_cls",
    [RK4FDMSolver, RK4FDMFusedSolver, RK4CudaSolver],
    ids=lambda cls: cls.__name__,
)
def test_fdm_solvers_bitwise_close(solver_cls):
    """Test that fdm solvers bitwise close."""
    if solver_cls is RK4FDMSolver:
        pytest.skip("reference solver")

    grid_cfg = GridCfg(nx=64, ny=64, lx=200.0, ly=200.0, grid_type="periodic")
    physics = make_physics_default()
    dt = 0.01
    n_steps = 2000
    P0 = 0.01

    t_ref, rho_ref, psi_ref = _run_solver(
        RK4FDMSolver, grid_cfg, physics, dt, n_steps, P0
    )
    t_test, rho_test, psi_test = _run_solver(
        solver_cls, grid_cfg, physics, dt, n_steps, P0
    )

    rel_err = np.linalg.norm(psi_test - psi_ref) / (
        np.linalg.norm(psi_ref) + 1e-30
    )

    outdir = RESULTS_DIR / "bitwise"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    diff = np.abs(psi_test - psi_ref)
    im = axes[0].imshow(diff, origin="lower", cmap="hot")
    axes[0].set_title(f"|psi_test - psi_ref| (L2 rel={rel_err:.2e})")
    fig.colorbar(im, ax=axes[0])

    axes[1].semilogy(t_ref, rho_ref, "b-", label="RK4FDMSolver", linewidth=1.5)
    axes[1].semilogy(t_test, rho_test, "r--", label=solver_cls.__name__, linewidth=1.2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"max($|\psi|^2$)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    _save_fig(outdir, f"bitwise_{solver_cls.__name__}.png")

    assert rel_err < 1e-8, (
        f"{solver_cls.__name__} relative L2 error {rel_err:.2e} vs RK4FDMSolver"
    )
