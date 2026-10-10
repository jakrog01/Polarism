from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


@pytest.mark.parametrize("reservoir_type", ["single", "double", "quadratic-double"])
def test_ifrk4_single_precision_stays_float32(reservoir_type: str) -> None:
    cfg = small_config(
        solver__method="ifrk4-fft-cuda",
        solver__precision="single",
        solver__total_time=0.05,
        reservoir__reservoir_type=reservoir_type,
        compute_engine=ComputeEngineParameters(use_gpu=False),
    )
    controller = SimulationController(cfg)
    controller.run()
    assert controller.state.psi.dtype == np.complex64
    for field in controller.reservoir.get_state():
        assert field.dtype == np.float32
