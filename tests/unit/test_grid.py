from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config, GridParameters
from polarism.config.validation import validate_config
from polarism.grid.closed_interval_simulation_grid_2d import ClosedIntervalSimulationGrid2D
from polarism.grid.create_grid import create_grid

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


@pytest.mark.parametrize("nx", [5, 8, 33, 64])
@pytest.mark.parametrize("grid_type", ["periodic", "closed-interval"])
def test_grid_spacing_and_nodes(nx: int, grid_type: str) -> None:
    compute_engine.xp = np
    lx = 20.0
    grid = create_grid(GridParameters(nx=nx, ny=nx, lx=lx, ly=lx, grid_type=grid_type))
    expected_dx = lx / (nx if grid_type == "periodic" else nx - 1)
    assert grid.dx == expected_dx
    if grid_type == "periodic":
        expected = (np.arange(nx) - (nx - 1) / 2) * expected_dx
        assert np.array_equal(grid.X[0], expected)
        assert np.allclose(expected, -expected[::-1])
        assert (0.0 in expected) is (nx % 2 == 1)
        assert np.array_equal(grid.kx, 2 * np.pi * np.fft.fftfreq(nx, d=expected_dx))
        if nx % 2 == 0:
            assert np.isclose(np.abs(grid.kx).max(), np.pi / expected_dx, rtol=TOL_MACHINE_F64)
    else:
        assert grid.X[0, 0] == -lx / 2
        assert grid.X[-1, -1] == lx / 2
        assert np.allclose(grid.X[0], -grid.X[0, ::-1], rtol=TOL_MACHINE_F64)
        assert np.array_equal(grid.kx, 2 * np.pi * np.fft.fftfreq(nx, d=expected_dx))


def test_closed_interval_requires_two_nodes_but_periodic_allows_one() -> None:
    compute_engine.xp = np
    with pytest.raises(ValueError, match="nx >= 2"):
        ClosedIntervalSimulationGrid2D(GridParameters(nx=1, ny=4, lx=4.0, ly=4.0, grid_type="closed-interval"))
    create_grid(GridParameters(nx=1, ny=4, lx=4.0, ly=4.0, grid_type="periodic"))


def test_validate_config_accepts_periodic_single_node() -> None:
    cfg = Config(grid=GridParameters(nx=1, ny=4, lx=4.0, ly=4.0))
    validate_config(cfg)
