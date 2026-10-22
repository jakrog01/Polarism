from __future__ import annotations

import numpy as np
import pytest

from polarism.boundary_conditions.absorption.absorption_registry import available_absorptions
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
        dt = 0.01
        strategy.set_time_step(dt)
        expected = np.exp(-0.5 * strategy.strength * strategy.absorption_profile * dt)
        assert strategy.get_potential_distribution() == 0.0
        assert np.allclose(strategy.apply_absorption(psi), expected, rtol=TOL_MACHINE_F64)
        assert strategy.after_step_is_noop is False
    else:
        assert strategy.get_potential_distribution() == 0.0
        assert strategy.apply_absorption(psi) is psi and strategy.after_step_is_noop is True


def test_mask_strength_controls_boundary_damping() -> None:
    compute_engine.xp = np
    g = grid()
    weak = create_absorption_strategy(
        g,
        BoundaryConditionParameters(
            absorption="mask", mask_width_percent=0.2, strength=0.25
        ),
        PhysicsConstants(),
    )
    strong = create_absorption_strategy(
        g,
        BoundaryConditionParameters(
            absorption="mask", mask_width_percent=0.2, strength=2.0
        ),
        PhysicsConstants(),
    )
    disabled = create_absorption_strategy(
        g,
        BoundaryConditionParameters(
            absorption="mask", mask_width_percent=0.2, strength=0.0
        ),
        PhysicsConstants(),
    )
    for strategy in (weak, strong, disabled):
        strategy.set_time_step(0.01)
    interior = weak.absorption_profile == 0.0
    transition = (weak.absorption_profile > 0.0) & (weak.absorption_profile < 1.0)
    assert np.allclose(disabled.mask, 1.0, rtol=TOL_MACHINE_F64)
    assert np.allclose(weak.mask[interior], 1.0, rtol=TOL_MACHINE_F64)
    assert np.all(strong.mask[transition] < weak.mask[transition])


def test_mask_requires_time_step_and_keeps_field_dtype() -> None:
    compute_engine.xp = np
    strategy = create_absorption_strategy(grid(), BoundaryConditionParameters(absorption="mask"), PhysicsConstants())
    psi = np.ones(strategy.absorption_profile.shape, dtype=np.complex64)
    with pytest.raises(RuntimeError, match="set_time_step"):
        strategy.apply_absorption(psi)
    strategy.set_time_step(0.01)
    assert strategy.apply_absorption(psi).dtype == np.complex64


def test_unknown_absorption_configuration_errors() -> None:
    compute_engine.xp = np
    with pytest.raises(ValueError, match="Unsupported profile type"):
        create_absorption_profile(8, 8, BoundaryConditionParameters(absorption="mask", profile_type="bad"))
    with pytest.raises(ValueError, match="Available") as exc:
        create_absorption_strategy(grid(), BoundaryConditionParameters(absorption="bad"), PhysicsConstants())
    assert str(sorted(available_absorptions)) in str(exc.value)
