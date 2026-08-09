from __future__ import annotations

import numpy as np
import pytest

from polarism.boundary_conditions.absorption.absorption_registry import available_boundary_conditions
from polarism.boundary_conditions.absorption.absorption_strategy import create_absorption_profile
from polarism.boundary_conditions.absorption.create_absorption_strategy import create_absorption_strategy
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import BoundaryConditionParameters, PhysicsConstants
from tests.unit.conftest import grid

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


def test_absorption_profiles_are_bounded_and_distinct() -> None:
    compute_engine.xp = np
    base = BoundaryConditionParameters(absorption="mask", mask_width_percent=0.2, profile_type="sin2")
    sin2 = create_absorption_profile(20, 20, base)
    parabolic = create_absorption_profile(20, 20, BoundaryConditionParameters(absorption="mask", mask_width_percent=0.2, profile_type="parabolic"))
    wider = create_absorption_profile(20, 20, BoundaryConditionParameters(absorption="mask", mask_width_percent=0.4, profile_type="sin2"))
    assert np.all((0 <= sin2) & (sin2 <= 1))
    assert sin2[10, 10] == 0.0 and sin2[0, 0] == 1.0
    assert np.count_nonzero(wider) >= np.count_nonzero(sin2)
    assert np.max(np.abs(sin2 - parabolic)) > 0.01


@pytest.mark.parametrize("absorption", ["cap", "mask", "no-absorption"])
def test_absorption_strategies(absorption: str) -> None:
    compute_engine.xp = np
    g = grid()
    strategy = create_absorption_strategy(g, BoundaryConditionParameters(absorption=absorption), PhysicsConstants())
    psi = np.ones(g.X.shape, dtype=np.complex128)
    if absorption == "cap":
        potential = strategy.get_potential_distribution()
        assert np.all(potential.real == 0.0) and np.all(potential.imag <= 0.0)
        assert strategy.apply_absorption(psi) is psi and strategy.after_step_is_noop is True
    elif absorption == "mask":
        assert strategy.get_potential_distribution() == 0.0
        assert np.allclose(strategy.apply_absorption(psi), 1.0 - strategy.absorption_profile, rtol=TOL_MACHINE_F64)
        assert strategy.after_step_is_noop is False
    else:
        assert strategy.get_potential_distribution() == 0.0
        assert strategy.apply_absorption(psi) is psi and strategy.after_step_is_noop is True


def test_unknown_absorption_configuration_errors() -> None:
    compute_engine.xp = np
    with pytest.raises(ValueError, match="Unsupported profile type"):
        create_absorption_profile(8, 8, BoundaryConditionParameters(absorption="mask", profile_type="bad"))
    with pytest.raises(ValueError, match="Available") as exc:
        create_absorption_strategy(grid(), BoundaryConditionParameters(absorption="bad"), PhysicsConstants())
    assert str(list(available_boundary_conditions.keys())) in str(exc.value)
