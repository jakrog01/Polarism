from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.grid.create_grid import create_grid
from polarism.reservoir.laplacian import real_laplacian
from polarism.solver.rk4_cuda_solver import RK4CudaSolver
from tests.unit.conftest import small_config

TOL_FDM_STENCIL = 3e-5  # Second-order stencil applied to a smooth field.
TOL_ORDER_SLOPE = 0.15  # Log-log convergence-slope fit tolerance.


def test_periodic_cosine_stencil() -> None:
    compute_engine.xp = np
    nx, length, mode = 512, 20.0, 2
    dx = length / nx
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    f = np.broadcast_to(np.cos(2 * np.pi * mode * x / length), (nx, nx)).copy()
    k = 2 * np.pi * mode / length
    assert np.max(np.abs(real_laplacian(f, np, dx, dx, "periodic") + k**2 * f)) < TOL_FDM_STENCIL


def test_periodic_stencil_second_order() -> None:
    compute_engine.xp = np
    length, mode = 20.0, 2
    errors = []
    for nx in (32, 64, 128, 256):
        dx = length / nx
        x = (np.arange(nx) - (nx - 1) / 2) * dx
        f = np.broadcast_to(np.cos(2 * np.pi * mode * x / length), (nx, nx)).copy()
        k = 2 * np.pi * mode / length
        errors.append(np.sqrt(np.mean((real_laplacian(f, np, dx, dx, "periodic") + k**2 * f) ** 2)))
    slope = np.polyfit(np.log(np.array([32, 64, 128, 256])), np.log(errors), 1)[0]
    assert abs(slope + 2.0) < TOL_ORDER_SLOPE


def test_closed_constant_neumann_laplacian_is_exactly_zero() -> None:
    compute_engine.xp = np
    assert np.max(np.abs(real_laplacian(np.ones((8, 8)), np, 1.0, 1.0, "closed-interval"))) == 0.0


def test_isotropic_nine_point_direction_independence() -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    cfg = small_config(
        grid__nx=128,
        grid__ny=128,
        grid__lx=20.0,
        grid__ly=20.0,
        solver__method="rk4-cuda",
        solver__laplacian="isotropic-9pt",
    )
    grid = create_grid(cfg.grid)
    solver = RK4CudaSolver(cfg, grid)
    x, y = grid.X, grid.Y
    k = 2.0 * np.pi / cfg.grid.lx
    axis = np.cos(k * x).astype(np.complex128)
    diagonal = np.cos(k * (x + y)).astype(np.complex128)
    axis_lap = np.empty_like(axis)
    diagonal_lap = np.empty_like(diagonal)
    solver._cpu_laplacian(axis, axis_lap)
    solver._cpu_laplacian(diagonal, diagonal_lap)
    axis_error = np.linalg.norm(axis_lap + k**2 * axis) / np.linalg.norm(k**2 * axis)
    diagonal_error = np.linalg.norm(diagonal_lap + 2.0 * k**2 * diagonal) / np.linalg.norm(2.0 * k**2 * diagonal)
    axis_scaled = axis_error / k**2
    diagonal_scaled = diagonal_error / (2.0 * k**2)
    assert abs(axis_scaled - diagonal_scaled) / max(axis_scaled, diagonal_scaled) < 0.03


def test_isotropic_nine_point_rejects_unequal_spacings() -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    cfg = small_config(
        grid__nx=64,
        grid__ny=64,
        grid__lx=20.0,
        grid__ly=10.0,
        solver__method="rk4-cuda",
        solver__laplacian="isotropic-9pt",
    )
    with np.testing.assert_raises_regex(ValueError, "requires dx == dy"):
        RK4CudaSolver(cfg, create_grid(cfg.grid))
