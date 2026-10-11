"""Whole-step CUDA RK4 and tile halo checks."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from polarism.compute_engine import has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.simulation_controller import SimulationController
from polarism.solver.rk4_cuda_solver import (
    _choose_tile_shape, _prefer_tiled, _tile_halo_indices, _tile_index,
    _tile_shared_bytes,
)
from tests.unit.conftest import small_config


@pytest.mark.parametrize("shape", [(6, 9), (37, 53)])
@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
def test_tile_halo_maps_staged_neighbors(shape, boundary):
    ny, nx = shape
    tile_x, tile_y = 16, 8
    for block_x, block_y in ((0, 0), (3, 2)):
        halo = _tile_halo_indices(nx, ny, tile_x, tile_y,
                                  block_x, block_y, boundary)
        assert halo.shape == (tile_y + 8, tile_x + 8)
        for stage in range(4):
            for y in range(stage + 1, tile_y + 7 - stage):
                for x in range(stage + 1, tile_x + 7 - stage):
                    vx = block_x * tile_x + x - 4
                    vy = block_y * tile_y + y - 4
                    gx = _tile_index(vx, nx, boundary)
                    gy = _tile_index(vy, ny, boundary)
                    up = gy - 1 if gy > 0 else (ny - 1 if boundary == "periodic" else 1)
                    down = gy + 1 if gy < ny - 1 else (0 if boundary == "periodic" else ny - 2)
                    left = gx - 1 if gx > 0 else (nx - 1 if boundary == "periodic" else 1)
                    right = gx + 1 if gx < nx - 1 else (0 if boundary == "periodic" else nx - 2)
                    if boundary == "closed-interval":
                        up_y = y - 1 if _tile_index(vy - 1, ny, boundary) == up else y + 1
                        down_y = y + 1 if _tile_index(vy + 1, ny, boundary) == down else y - 1
                        left_x = x - 1 if _tile_index(vx - 1, nx, boundary) == left else x + 1
                        right_x = x + 1 if _tile_index(vx + 1, nx, boundary) == right else x - 1
                        assert halo[up_y, x] == up * nx + gx
                        assert halo[down_y, x] == down * nx + gx
                        assert halo[y, left_x] == gy * nx + left
                        assert halo[y, right_x] == gy * nx + right
                    assert up * nx + gx in (halo[y - 1, x], halo[y + 1, x])
                    assert down * nx + gx in (halo[y - 1, x], halo[y + 1, x])
                    assert gy * nx + left in (halo[y, x - 1], halo[y, x + 1])
                    assert gy * nx + right in (halo[y, x - 1], halo[y, x + 1])
                    assert up * nx + left in (
                        halo[y - 1, x - 1], halo[y - 1, x + 1],
                        halo[y + 1, x - 1], halo[y + 1, x + 1],
                    )


def test_tile_selection_respects_shared_memory_limits():
    assert _choose_tile_shape("float", "single", 100 * 1024, 100 * 1024) == (32, 16)
    assert _choose_tile_shape("double", "quadratic-double", 48 * 1024,
                              96 * 1024) == (16, 8)
    assert _choose_tile_shape("double", "double", 1024, 2048) is None
    tile = _choose_tile_shape("double", "double", 48 * 1024, 96 * 1024)
    assert 2 * _tile_shared_bytes(*tile, "double", "double") <= 96 * 1024


def test_default_tile_policy():
    assert _prefer_tiled("single", 8, 9)
    assert not _prefer_tiled("double", 8, 9)
    assert _prefer_tiled("double", 7, 0)


def _assert_rounding_agreement(actual, reference, precision):
    actual = actual.get()
    reference = reference.get()
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(reference))
    tolerance = 2e-6 if precision == "single" else 5e-14
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(actual, reference, rtol=0,
                               atol=tolerance * scale, equal_nan=False)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("boundary", ["periodic", "closed-interval"])
@pytest.mark.parametrize("laplacian", ["five-point", "isotropic-9pt"])
@pytest.mark.parametrize("complex_potential", [False, True])
@pytest.mark.parametrize("separable", [False, True])
def test_tiled_matches_staged_to_rounding_level(
    precision, model, boundary, laplacian, complex_potential, separable,
):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    key = (precision, model, boundary, laplacian, complex_potential, separable)
    case = sum((i + 1) * ord(str(value)[0]) for i, value in enumerate(key))
    ny, nx = ((6, 9), (37, 53))[case % 2]
    spacing_x = nx if boundary == "periodic" else nx - 1
    spacing_y = ny if boundary == "periodic" else ny - 1
    cfg = small_config(
        grid__nx=nx, grid__ny=ny, grid__lx=float(spacing_x),
        grid__ly=float(spacing_y), grid__grid_type=boundary,
        solver__method="rk4-cuda", solver__precision=precision,
        solver__laplacian=laplacian, solver__dt=0.01,
        reservoir__reservoir_type=model,
        boundary_condition__absorption="no-absorption",
        laser__laser_type="pulse-gaussian", laser__P0=0.7,
        laser__Pmax=0.7, laser__sigma_time=0.08,
        physics__kinetic_relaxation_eta=0.02,
        physics__reservoir_diffusion_R=0.03,
        physics__reservoir_diffusion_A=0.03,
        physics__reservoir_diffusion_I=0.02,
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
    tiled = SimulationController(cfg)
    staged = SimulationController(cfg)
    tiled.solver._use_tiled = True
    staged.solver._use_tiled = False
    for controller in (tiled, staged):
        if not separable:
            second = replace(cfg.laser, x0=2.0, y0=-1.0, P0=0.4, Pmax=0.4)
            controller.lasers.append(PulseGaussian(
                second, controller.grid.X, controller.grid.Y, precision,
            ))
        controller.pump_field.lasers = controller.lasers
        y, x = np.indices((ny, nx))
        psi = (0.2 + 0.03 * np.cos(x / 4)) * np.exp(1j * (x / 8 + y / 9))
        controller.state.psi[...] = controller.xp.asarray(
            psi, dtype=controller.state.psi.dtype,
        )
        fields = tuple(controller.xp.asarray(
            0.15 + 0.04 * index + 0.02 * np.sin(x / 5 + index) * np.cos(y / 6),
            dtype=field.dtype,
        ) for index, field in enumerate(controller.reservoir.get_state()))
        controller.reservoir.set_state(fields)
    potential = tiled.xp.real(tiled.potential)
    if complex_potential:
        potential = potential + 0.05j
    for step in range(3):
        for controller in (tiled, staged):
            controller.state.t = step * cfg.solver.dt
            controller.solver.step(
                potential, controller.pump_field, controller.reservoir,
                controller.boundary_condition, controller.state,
            )
        _assert_rounding_agreement(tiled.state.psi, staged.state.psi, precision)
        for left, right in zip(tiled.reservoir.get_state(), staged.reservoir.get_state()):
            _assert_rounding_agreement(left, right, precision)


@pytest.mark.gpu
def test_tiled_large_square_matches_staged():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = small_config(
        grid__nx=256, grid__ny=256, solver__method="rk4-cuda",
        solver__precision="single", reservoir__reservoir_type="single",
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
    tiled = SimulationController(cfg)
    staged = SimulationController(cfg)
    tiled.solver._use_tiled = True
    staged.solver._use_tiled = False
    for controller in (tiled, staged):
        controller.solver.step(
            controller.potential, controller.pump_field, controller.reservoir,
            controller.boundary_condition, controller.state,
        )
    _assert_rounding_agreement(tiled.state.psi, staged.state.psi, "single")
    for left, right in zip(tiled.reservoir.get_state(), staged.reservoir.get_state()):
        _assert_rounding_agreement(left, right, "single")
