from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.create_potential import create_potential
from polarism.potential.potential_registy import available_potentials
from tests.unit.conftest import grid

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


def _expected(g, cfg):
    return sum(
        v * np.exp(-((((g.X - x) ** 2 + (g.Y - y) ** 2) / w**2) ** cfg.order))
        for x, y, v, w in ((cfg.x1, cfg.y1, cfg.V1, cfg.w1), (cfg.x2, cfg.y2, cfg.V2, cfg.w2))
    )


def test_double_well_matches_hand_expression() -> None:
    compute_engine.xp = np
    g = grid(31, 31)
    cfg = PotentialParameters(potential_type="double-well-supergaussian")
    assert np.max(np.abs(create_potential(cfg, g) - _expected(g, cfg))) < TOL_MACHINE_F64


def test_order_one_is_sum_of_gaussians() -> None:
    compute_engine.xp = np
    g = grid(31, 31)
    cfg = PotentialParameters(potential_type="double-well-supergaussian", order=1.0)
    assert np.max(np.abs(create_potential(cfg, g) - _expected(g, cfg))) < TOL_MACHINE_F64


def test_moving_one_well_changes_its_contribution_only() -> None:
    compute_engine.xp = np
    g = grid(41, 41)
    original = PotentialParameters(potential_type="double-well-supergaussian")
    moved = PotentialParameters(potential_type="double-well-supergaussian", x1=7.0, y1=5.0)
    contribution_two = original.V2 * np.exp(-((((g.X - original.x2) ** 2 + (g.Y - original.y2) ** 2) / original.w2**2) ** original.order))
    moved_contribution_two = moved.V2 * np.exp(-((((g.X - moved.x2) ** 2 + (g.Y - moved.y2) ** 2) / moved.w2**2) ** moved.order))
    assert np.array_equal(contribution_two, moved_contribution_two)
    assert not np.array_equal(create_potential(original, g), create_potential(moved, g))


def test_zero_and_unknown_potentials() -> None:
    compute_engine.xp = np
    g = grid()
    assert np.array_equal(create_potential(PotentialParameters(potential_type="zero"), g), np.zeros_like(g.X))
    with pytest.raises(ValueError, match="Available") as exc:
        create_potential(PotentialParameters(potential_type="missing"), g)
    assert str(list(available_potentials.keys())) in str(exc.value)
