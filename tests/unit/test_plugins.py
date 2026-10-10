from __future__ import annotations

import sys
import textwrap
import warnings

import numpy as np
import pytest

from polarism import SimulationController
from polarism.config.simulation_parameters import LaserParameters, PotentialParameters
from polarism.grid.grid_registry import available_grids, register_grid
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.laser_registy import available_lasers, register_laser
from polarism.plugins import load_plugins
from polarism.potential.create_potential import create_potential
from polarism.potential.potential_registy import available_potentials
from polarism.reservoir.reservoir_registry import available_reservoirs
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from tests.unit.conftest import grid, small_config

LASER_PLUGIN = """
from polarism.laser import register_laser
from polarism.laser.abstract_laser import AbstractLaser


@register_laser("plugin-ring")
class RingLaser(AbstractLaser):
    def __init__(self, laser_config, X, Y, precision="double"):
        super().__init__(laser_config, X, Y, precision)
        self.radius = laser_config.extra.get("ring_radius", 2.0)
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        r = self.xp.sqrt((X - self.x0) ** 2 + (Y - self.y0) ** 2)
        return self.xp.exp(-((r - self.radius) ** 2))

    def _P_time(self, t):
        return 1.0
"""

POTENTIAL_PLUGIN = """
from polarism.potential import register_potential


@register_potential("plugin-const")
def plugin_const(X, Y, potential_config):
    return 0.0 * X + 1.0
"""


@pytest.fixture
def plugin_root(tmp_path, monkeypatch):
    """Provide a project root and undo every side effect of loading plugins."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    lasers, potentials = dict(available_lasers), dict(available_potentials)
    yield tmp_path
    for name in [m for m in sys.modules if m == "polarism_plugins" or m.startswith("polarism_plugins.")]:
        del sys.modules[name]
    available_lasers.clear()
    available_lasers.update(lasers)
    available_potentials.clear()
    available_potentials.update(potentials)


def _write(root, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def test_missing_plugin_folder_loads_nothing(tmp_path) -> None:
    assert load_plugins(tmp_path) == []


def test_nested_plugins_without_init_files_are_registered(plugin_root) -> None:
    _write(plugin_root, "polarism_plugins/laser/ring_laser.py", LASER_PLUGIN)
    _write(plugin_root, "polarism_plugins/potential/const.py", POTENTIAL_PLUGIN)

    loaded = load_plugins(plugin_root)

    assert loaded == ["polarism_plugins.laser.ring_laser", "polarism_plugins.potential.const"]
    g = grid()
    cfg = LaserParameters(laser_type="plugin-ring", extra={"ring_radius": 3.0})
    laser = LaserFactory.create_laser(cfg, g.X, g.Y)[0]
    assert laser.radius == 3.0
    potential = create_potential(PotentialParameters(potential_type="plugin-const"), g)
    assert np.all(potential == 1.0)


def test_loading_twice_is_idempotent(plugin_root) -> None:
    _write(plugin_root, "polarism_plugins/__init__.py", "")
    _write(plugin_root, "polarism_plugins/ring_laser.py", LASER_PLUGIN)
    load_plugins(plugin_root)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_plugins(plugin_root)
    assert "plugin-ring" in available_lasers


def test_multiple_laser_yaml_routes_unknown_keys_to_extra(plugin_root) -> None:
    _write(plugin_root, "polarism_plugins/ring_laser.py", LASER_PLUGIN)
    load_plugins(plugin_root)
    config_file = plugin_root / "lasers.yaml"
    config_file.write_text("lasers:\n  - laser_type: plugin-ring\n    ring_radius: 4.5\n    P0: 2.0\n")

    g = grid()
    cfg = LaserParameters(mode="multiple", config_file=str(config_file))
    (laser,) = LaserFactory.create_laser(cfg, g.X, g.Y)

    assert laser.radius == 4.5
    assert laser.P0 == 2.0


def test_duplicate_registration_warns(plugin_root) -> None:
    with pytest.warns(UserWarning, match="already registered"):
        @register_laser("uniform")
        class OtherUniform:
            pass


RESERVOIR_PLUGIN = """
from polarism.reservoir import register_reservoir
from polarism.reservoir.single_reservoir import SingleReservoir


@register_reservoir("plugin-res")
class PluginReservoir(SingleReservoir):
    pass
"""

GRID_PLUGIN = """
from polarism.grid import register_grid
from polarism.grid.periodic_simulation_grid_2d import PeriodicSimulationGrid2D
from polarism.grid.closed_interval_simulation_grid_2d import ClosedIntervalSimulationGrid2D


@register_grid("plugin-periodic")
class PluginPeriodicGrid(PeriodicSimulationGrid2D):
    pass


@register_grid("plugin-closed")
class PluginClosedGrid(ClosedIntervalSimulationGrid2D):
    pass
"""


@pytest.fixture
def grid_and_reservoir_plugins(plugin_root):
    """Load grid and reservoir plugins and restore both registries afterwards."""
    grids, reservoirs = dict(available_grids), dict(available_reservoirs)
    _write(plugin_root, "polarism_plugins/grid.py", GRID_PLUGIN)
    _write(plugin_root, "polarism_plugins/reservoir.py", RESERVOIR_PLUGIN)
    load_plugins(plugin_root)
    yield plugin_root
    available_grids.clear()
    available_grids.update(grids)
    available_reservoirs.clear()
    available_reservoirs.update(reservoirs)


def test_plugin_grid_and_reservoir_run_with_generic_solver(grid_and_reservoir_plugins) -> None:
    for grid_type in ("plugin-periodic", "plugin-closed"):
        cfg = small_config(
            grid__grid_type=grid_type,
            reservoir__reservoir_type="plugin-res",
            solver__method="rk4-fdm",
            solver__total_time=0.02,
            solver__dt=0.01,
            result__save_results=False,
        )
        controller = SimulationController(cfg)
        assert controller.grid.boundary == available_grids[grid_type].boundary
        controller.run()
        assert np.all(np.isfinite(controller.state.psi))


def test_plugin_grid_boundary_drives_solver_compatibility(grid_and_reservoir_plugins) -> None:
    check_solver_compatibility(small_config(grid__grid_type="plugin-periodic", solver__method="ip-rk4"))
    with pytest.raises(SolverCompatibilityError, match="boundary 'closed-interval'"):
        check_solver_compatibility(small_config(grid__grid_type="plugin-closed", solver__method="ip-rk4"))


def test_plugin_reservoir_compatibility(grid_and_reservoir_plugins) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_solver_compatibility(small_config(reservoir__reservoir_type="plugin-res", solver__method="rk4-fdm"))
    with pytest.raises(SolverCompatibilityError, match="plugin reservoir_type"):
        check_solver_compatibility(small_config(reservoir__reservoir_type="plugin-res", solver__method="rk4-cuda"))
    with pytest.warns(UserWarning, match="Lie splitting"):
        check_solver_compatibility(small_config(reservoir__reservoir_type="plugin-res", solver__method="split-step-fft"))


def test_register_grid_requires_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        @register_grid("plugin-no-boundary")
        class NoBoundaryGrid:
            pass
