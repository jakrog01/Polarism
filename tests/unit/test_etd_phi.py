from __future__ import annotations

import math

import numpy as np
import pytest

from polarism.grid.create_grid import create_grid
from polarism.solver.etd_rk2_solver import ETDRK2Solver
from tests.unit.conftest import small_config

TOL_PHI = 1e-13  # Relative accuracy of phi1/phi2 across all |z|.


def _series(z: complex, offset: int) -> complex:
    return sum(z**k / math.factorial(k + offset) for k in range(40))


@pytest.mark.parametrize("offset", [1, 2])
def test_phi_functions_are_accurate_for_small_and_large_arguments(offset: int) -> None:
    cfg = small_config(solver__method="etd-rk2")
    solver = ETDRK2Solver(cfg, create_grid(cfg.grid))
    z = np.concatenate([[0.0], -1j * np.logspace(-9, 0.3, 60)])
    phi = solver._compute_phi1(z) if offset == 1 else solver._compute_phi2(z)
    expected = np.array([_series(complex(value), offset) for value in z])
    assert np.max(np.abs(phi - expected) / np.abs(expected)) < TOL_PHI
