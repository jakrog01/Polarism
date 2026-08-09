from __future__ import annotations

import numpy as np
import pytest

from polarism.boundary_conditions.absorption.absorption_registry import available_boundary_conditions, register_absorption
from polarism.boundary_conditions.absorption.absorption_strategy import AbsorptionStrategy
from polarism.compute_engine import compute_engine
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import available_lasers, register_laser
from polarism.potential.potential_registy import available_potentials, register_potential
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_registry_extensions_run_without_solver_edits(monkeypatch) -> None:
    """Reservoir dispatch is hardcoded in `create_reservoir` — a reservoir plugin requires a library change and is out of scope for this test."""
    compute_engine.xp = np
    @register_laser("dummy_test")
    class DummyLaser(AbstractLaser):
        def __init__(self, cfg, X, Y, precision="double"):
            super().__init__(cfg, X, Y, precision); self._finalize_spatial_envelope(X, Y)
        def _amplitude(self, t): return self.P0
        def _P_space(self, X, Y): return self.xp.ones_like(X)
        def _P_time(self, t): return 1.0
    @register_potential("dummy_test")
    def dummy_potential(X, Y, cfg): return np.zeros_like(X)
    @register_absorption("dummy_test")
    class DummyAbsorption(AbsorptionStrategy):
        after_step_is_noop = True
        def __init__(self, *args): pass
        def get_potential_distribution(self): return 0.0
        def apply_absorption(self, psi): return psi
    cfg = small_config(laser__laser_type="dummy_test", potential__potential_type="dummy_test", boundary_condition__absorption="dummy_test", solver__total_time=0.02)
    SimulationController(cfg).run()
    for registry in (available_lasers, available_potentials, available_boundary_conditions):
        monkeypatch.delitem(registry, "dummy_test")
