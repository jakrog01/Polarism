from __future__ import annotations

import pytest

from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


@pytest.mark.parametrize("total_time,dt,expected_steps", [(0.3, 0.1, 3), (0.7, 0.1, 7), (0.1, 0.01, 10)])
def test_run_does_not_truncate_step_count(total_time: float, dt: float, expected_steps: int) -> None:
    cfg = small_config(solver__total_time=total_time, solver__dt=dt)
    controller = SimulationController(cfg)
    controller.run()
    assert controller.state.t == pytest.approx(expected_steps * dt)
