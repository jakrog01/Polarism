from __future__ import annotations

import numpy as np
import pytest

from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


@pytest.mark.parametrize("total_time,dt,expected_steps", [(0.3, 0.1, 3), (0.7, 0.1, 7), (0.1, 0.01, 10)])
def test_run_does_not_truncate_step_count(total_time: float, dt: float, expected_steps: int) -> None:
    cfg = small_config(solver__total_time=total_time, solver__dt=dt)
    controller = SimulationController(cfg)
    controller.run()
    assert controller.state.t == pytest.approx(expected_steps * dt)


def test_assigning_psi_discards_cached_spectrum() -> None:
    cfg = small_config(
        solver__method="ifrk4-fft-cuda",
        solver__dt=1e-3,
        boundary_condition__absorption="cap",
        laser__P0=0.0,
    )
    controller = SimulationController(cfg)
    step_args = (controller.potential, controller._compute_total_pump(0.0), controller.reservoir, controller.boundary_condition, controller.state)
    controller.solver.step(*step_args)
    assert controller.state.psi_k is not None

    controller.state.psi = np.zeros_like(controller.state.psi)
    assert controller.state.psi_k is None

    controller.solver.step(*step_args)
    assert np.abs(controller.state.psi).max() == 0.0
