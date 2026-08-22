from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from polarism.analysis.condensation import (
    CONDENSATION_PSI_SQ_FLOOR,
    CrossingResult,
    classify,
    count_upward_crossings,
    critical_reservoir_density,
    gain_loss_signal,
    psi_sq_floor,
    validate_sampling,
)


def _physics() -> SimpleNamespace:
    return SimpleNamespace(
        R=0.023,
        gamma_C=0.1,
        gamma_R=0.15,
        gamma_I=0.001,
        kappa=0.05,
        init_eps=1.0e-3,
    )


def test_gain_threshold_primitives() -> None:
    physics = _physics()
    critical = critical_reservoir_density(physics)
    assert critical == physics.gamma_C / physics.R
    np.testing.assert_allclose(
        gain_loss_signal(np.array((critical * 0.5, critical * 1.5)), physics),
        np.array((-0.05, 0.05)),
    )


def test_schmitt_crossings_and_sampling_validation() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    result = count_upward_crossings(time, np.sin(2.0 * np.pi * time), hysteresis=0.02)
    assert result.n_crossings == 10
    chatter = count_upward_crossings(time, 0.01 * np.sin(100.0 * time), hysteresis=0.02)
    assert chatter.n_crossings == 0
    validate_sampling(0.15, 1.5)
    with pytest.raises(ValueError, match="too sparse"):
        validate_sampling(0.151, 1.5)


@pytest.mark.parametrize(
    ("crossings", "psi_sq_max", "klass"),
    ((0, 1.0, "dark"), (1, 0.0, "gain_only"), (1, 1.0, "latched"), (2, 1.0, "spiking")),
)
def test_classification(crossings: int, psi_sq_max: float, klass: str) -> None:
    result = CrossingResult(crossings, (), 5.0, 0.0, None, 5.0 / critical_reservoir_density(_physics()))
    assert classify(result, psi_sq_max, 0.05).klass == klass


def test_psi_sq_floors() -> None:
    physics = _physics()
    assert CONDENSATION_PSI_SQ_FLOOR == 5.0e-2
    assert psi_sq_floor(physics) == CONDENSATION_PSI_SQ_FLOOR
    assert psi_sq_floor(physics, mode="seed_relative", decades=3.0) == 1.0e-3
