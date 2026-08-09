from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.reservoir.laplacian import real_laplacian

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


@pytest.mark.xfail(strict=True, reason="isotropic-9pt stencil not implemented (only 'five-point' available)")
def test_isotropic_nine_point_direction_independence() -> None:
    pytest.fail("isotropic-9pt stencil family is unavailable")


@pytest.mark.xfail(strict=True, reason="isotropic-9pt stencil not implemented (only 'five-point' available)")
def test_isotropic_nine_point_rejects_unequal_spacings() -> None:
    pytest.fail("isotropic-9pt stencil family is unavailable")
