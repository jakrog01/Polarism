from __future__ import annotations

import pytest

from polarism.boundary_conditions.absorption.absorption_registry import available_boundary_conditions, register_absorption
from polarism.boundary_conditions.absorption.create_absorption_strategy import create_absorption_strategy
from polarism.config.simulation_parameters import BoundaryConditionParameters, LaserParameters, PhysicsConstants, PotentialParameters, ReservoirParameters
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.laser_registy import available_lasers, register_laser
from polarism.potential.create_potential import create_potential
from polarism.potential.potential_registy import available_potentials, register_potential
from polarism.solver.create_solver import create_solver
from polarism.solver.solver_registry import available_solvers, register_solver
from tests.unit.conftest import grid, small_config


@pytest.mark.parametrize("family", ["solver", "laser", "potential", "boundary"])
def test_unknown_registry_identifier_lists_available_entries(family: str) -> None:
    g = grid()
    with pytest.raises(ValueError, match="Available") as exc:
        if family == "solver":
            registry = available_solvers
            create_solver(small_config(solver__method="missing"), g)
        elif family == "laser":
            registry = available_lasers
            LaserFactory.create_laser(LaserParameters(laser_type="missing"), g.X, g.Y)
        elif family == "potential":
            registry = available_potentials
            create_potential(PotentialParameters(potential_type="missing"), g)
        else:
            registry = available_boundary_conditions
            create_absorption_strategy(g, BoundaryConditionParameters(absorption="missing"), small_config().physics)
    assert str(list(registry.keys())) in str(exc.value)


def test_registry_decorators_register_and_cleanup(monkeypatch) -> None:
    @register_laser("dummy_registry")
    class DummyLaser:
        pass
    @register_potential("dummy_registry")
    def dummy_potential(*args):
        return 0
    @register_absorption("dummy_registry")
    class DummyAbsorption:
        pass
    @register_solver("dummy_registry")
    class DummySolver:
        pass
    assert all("dummy_registry" in registry for registry in (available_lasers, available_potentials, available_boundary_conditions, available_solvers))
    for registry in (available_lasers, available_potentials, available_boundary_conditions, available_solvers):
        monkeypatch.delitem(registry, "dummy_registry")


def test_reservoir_factory_supports_builtin_models() -> None:
    """Create each built-in reservoir through the public factory."""
    g = grid()
    physics = PhysicsConstants()
    for name in ("single", "double", "quadratic-double"):
        reservoir = create_reservoir(ReservoirParameters(reservoir_type=name), physics, g)
        assert reservoir.get_reservoir_density().shape == (g.ny, g.nx)
