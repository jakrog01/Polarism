"""Checks for shared registries and YAML configuration."""
from __future__ import annotations

import sys

import pytest

from polarism.boundary_conditions.absorption.absorption_registry import available_absorptions, register_absorption
from polarism.config import Config
from polarism.grid.grid_registry import available_grids, register_grid
from polarism.laser.laser_registry import available_lasers, register_laser
from polarism.potential.potential_registry import available_potentials, register_potential
from polarism.reservoir.reservoir_registry import available_reservoirs, register_reservoir
from polarism.results.observable_registry import available_observables, register_observable
from polarism.results.storage.storage_registry import available_storages, register_storage
from polarism.solver.solver_registry import available_solvers, register_solver


REGISTRIES = [
    ("absorption", available_absorptions, register_absorption),
    ("grid", available_grids, register_grid),
    ("laser", available_lasers, register_laser),
    ("potential", available_potentials, register_potential),
    ("reservoir", available_reservoirs, register_reservoir),
    ("solver", available_solvers, register_solver),
    ("storage", available_storages, register_storage),
    ("observable", available_observables, register_observable),
]


@pytest.mark.parametrize("kind,registry,register", REGISTRIES)
def test_registry_duplicate_warning_and_unknown_name(kind, registry, register, monkeypatch) -> None:
    first = type("First", (), {"boundary": "periodic"})
    second = type("Second", (), {"boundary": "periodic"})
    monkeypatch.setitem(registry, "structure-test", first)
    with pytest.warns(UserWarning, match="already registered"):
        assert register("structure-test")(second) is second
    assert registry["structure-test"] is second
    with pytest.raises(ValueError) as exc:
        registry.lookup("structure-missing")
    assert str(exc.value) == f"Unknown {kind} 'structure-missing'. Available: {sorted(registry)}"
    monkeypatch.delitem(registry, "structure-test")


def test_config_yaml_round_trip(tmp_path) -> None:
    cfg = Config()
    cfg.grid.nx = 64
    cfg.laser.extra = {"ring_radius": 12.0}
    cfg.result.formats = ("hdf5", "json")
    cfg.result.animation_fields = ("density",)
    cfg.sweep.batch_size = 2
    path = cfg.to_yaml(tmp_path / "run.yaml")
    assert Config.from_yaml(path) == cfg


@pytest.mark.parametrize("content,path", [
    ("unknown: {}\n", "unknown"),
    ("grid:\n  typo: 12\n", "grid.typo"),
])
def test_config_yaml_rejects_unknown_key(tmp_path, content, path) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text(content)
    with pytest.raises(ValueError, match=path):
        Config.from_yaml(source)


def test_cli_config_flag_override(tmp_path, monkeypatch) -> None:
    from polarism import cli

    cfg = Config()
    cfg.grid.nx = 48
    path = cfg.to_yaml(tmp_path / "run.yaml")
    captured = []

    class Controller:
        def __init__(self, config):
            captured.append(config)

        def run(self):
            pass

    monkeypatch.setattr(cli, "SimulationController", Controller)
    monkeypatch.setattr(cli.compute_engine, "configure", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["polarism", "--config", str(path), "--grid.nx", "32"])
    cli.main()
    assert captured[0].grid.nx == 32
    assert captured[0].grid.ny == cfg.grid.ny
