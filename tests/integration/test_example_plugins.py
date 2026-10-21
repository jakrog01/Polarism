"""Exercise the documented plugin examples through the public controller."""

from __future__ import annotations

import sys
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from polarism import Config, SimulationController
from polarism.boundary_conditions.absorption.absorption_registry import available_boundary_conditions
from polarism.grid.grid_registry import available_grids
from polarism.laser.laser_registy import available_lasers
from polarism.plugins import load_plugins
from polarism.potential.potential_registy import available_potentials
from polarism.reservoir.reservoir_registry import available_reservoirs
from polarism.results.observable_registry import available_observables
from polarism.results.storage import available_storages
from polarism.solver.solver_registry import available_solvers


@pytest.fixture
def example_plugins(tmp_path, monkeypatch):
    """Load every example and restore Python import and registry state."""
    root = Path(__file__).resolve().parents[2] / "examples" / "plugins"
    shutil.copytree(root / "polarism_plugins", tmp_path / "polarism_plugins")
    monkeypatch.setattr(sys, "path", list(sys.path))
    registries = (
        available_boundary_conditions,
        available_grids,
        available_lasers,
        available_potentials,
        available_reservoirs,
        available_solvers,
        available_storages,
        available_observables,
    )
    snapshots = [dict(registry) for registry in registries]
    loaded = load_plugins(tmp_path)
    assert len(loaded) == 8
    yield
    for name in list(sys.modules):
        if name == "polarism_plugins" or name.startswith("polarism_plugins."):
            del sys.modules[name]
    for registry, snapshot in zip(registries, snapshots):
        registry.clear()
        registry.update(snapshot)


@pytest.mark.parametrize("component", [
    "laser", "potential", "reservoir", "solver", "grid", "absorption",
])
def test_example_component_runs_on_cpu(example_plugins, component):
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.P0 = 0.02
    cfg.laser.sigma_space = 3.0
    cfg.reservoir.reservoir_type = "single"
    cfg.solver.method = "rk4-fdm"
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.002
    cfg.result.save_results = False
    cfg.physics.init_seed = 42

    if component == "laser":
        cfg.laser.laser_type = "example-ring"
        cfg.laser.extra = {"ring_radius": 4.0}
    elif component == "potential":
        cfg.potential.potential_type = "example-parabolic-trap"
        cfg.potential.V1 = 0.001
    elif component == "reservoir":
        cfg.reservoir.reservoir_type = "example-slow-single"
        cfg.reservoir.extra = {"decay_factor": 0.5}
    elif component == "solver":
        cfg.solver.method = "example-midpoint-fdm"
    elif component == "grid":
        cfg.grid.grid_type = "example-labelled-periodic"
        cfg.grid.extra = {"label": "test"}
    else:
        cfg.boundary_condition.absorption = "example-soft-edge"

    controller = SimulationController(cfg)
    controller.run()
    assert np.all(np.isfinite(controller.state.psi))
    assert np.all(np.isfinite(controller.reservoir.get_reservoir_density()))
    if component == "grid":
        assert controller.grid.label == "test"


def _comparison_config():
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 12
    cfg.grid.lx = cfg.grid.ly = 24.0
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.P0 = 0.02
    cfg.laser.sigma_space = 4.0
    cfg.reservoir.reservoir_type = "single"
    cfg.solver.method = "rk4-fdm"
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.005
    cfg.physics.init_seed = 42
    return cfg


def test_midpoint_solver_agrees_with_rk4_for_short_run(example_plugins):
    reference = _comparison_config()
    candidate = deepcopy(reference)
    candidate.solver.method = "example-midpoint-fdm"

    rk4 = SimulationController(reference)
    rk4.run()
    midpoint = SimulationController(candidate)
    midpoint.run()

    np.testing.assert_allclose(midpoint.state.psi, rk4.state.psi, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(
        midpoint.reservoir.get_reservoir_density(),
        rk4.reservoir.get_reservoir_density(), rtol=1e-3, atol=1e-6,
    )


@pytest.mark.parametrize("method", ["rk4-fdm", "split-step-fft"])
def test_neutral_plugin_reservoir_matches_builtin(example_plugins, method):
    reference = _comparison_config()
    reference.solver.method = method
    candidate = deepcopy(reference)
    candidate.reservoir.reservoir_type = "example-slow-single"
    candidate.reservoir.extra = {"decay_factor": 1.0}

    builtin = SimulationController(reference)
    builtin.run()
    plugin = SimulationController(candidate)
    plugin.run()

    np.testing.assert_allclose(plugin.state.psi, builtin.state.psi, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        plugin.reservoir.get_reservoir_density(),
        builtin.reservoir.get_reservoir_density(), rtol=1e-10, atol=1e-12,
    )


def test_example_storage_and_observable_run(example_plugins, tmp_path):
    import csv

    cfg = _comparison_config()
    cfg.result.save_results = True
    cfg.result.formats = ("example-scalar-csv",)
    cfg.result.observables = ("example-center-x",)
    cfg.result.output_directory = str(tmp_path)
    cfg.result.batch_size = 2
    controller = SimulationController(cfg)
    controller.run()

    path = controller.storage_visitor.output_dir / "scalars.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert "center_x" in rows[0]
    assert all(np.isfinite(float(row["center_x"])) for row in rows)
