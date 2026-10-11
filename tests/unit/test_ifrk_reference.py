"""CPU regression against trajectories saved from the original IFRK4 solver."""
from pathlib import Path

import numpy as np

from tests.data.ifrk4_reference.generate import run_cases
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config
import pytest


def test_ifrk4_matches_old_cpu_implementation():
    path = Path(__file__).parents[1] / "data/ifrk4_reference/reference.npz"
    actual = run_cases()
    with np.load(path) as reference:
        assert set(actual) == set(reference.files)
        for key, value in actual.items():
            tolerance = 1e-5 if key.startswith("single") else 1e-12
            np.testing.assert_allclose(value, reference[key], rtol=tolerance,
                                       atol=tolerance * max(1e-3, np.max(abs(reference[key]))))


def test_invalid_tableau_is_rejected():
    cfg = small_config(solver__tableau="unknown")
    with pytest.raises(ConfigValidationError, match="solver.tableau"):
        validate_config(cfg)


def test_ifrk4_alias_forces_rk4_tableau():
    cfg = small_config(solver__method="ifrk4-fft-cuda", solver__tableau="vern9")
    assert SimulationController(cfg).solver.tableau.name == "rk4"
