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
    step_args = (controller.potential, lambda t: controller._compute_total_pump(0.0), controller.reservoir, controller.boundary_condition, controller.state)
    controller.solver.step(*step_args)
    assert controller.state.psi_k is not None

    controller.state.psi = np.zeros_like(controller.state.psi)
    assert controller.state.psi_k is None

    controller.solver.step(*step_args)
    assert np.abs(controller.state.psi).max() == 0.0


_FLOAT64_PROPAGATORS = pytest.mark.xfail(strict=True, reason="spectral propagators are built in float64")


@pytest.mark.parametrize("method", [
    "rk4-fdm",
    "rk4-fdm-fused",
    "rk4-cuda",
    "split-step-fft",
    pytest.param("ip-rk4", marks=_FLOAT64_PROPAGATORS),
    pytest.param("etd-rk2", marks=_FLOAT64_PROPAGATORS),
    "ifrk4-fft-cuda",
])
def test_mask_absorption_keeps_single_precision(method: str) -> None:
    cfg = small_config(solver__method=method, solver__precision="single", boundary_condition__absorption="mask")
    controller = SimulationController(cfg)
    controller.run()
    assert controller.state.psi.dtype == np.complex64


def test_mask_absorption_converges_to_cap() -> None:
    def final_psi(absorption: str, dt: float) -> np.ndarray:
        cfg = small_config(
            solver__dt=dt,
            solver__total_time=0.5,
            boundary_condition__absorption=absorption,
            boundary_condition__strength=2.0,
            laser__P0=0.0,
        )
        controller = SimulationController(cfg)
        controller.state.psi = np.ones_like(controller.state.psi)
        controller.run()
        return controller.state.psi

    cap = final_psi("cap", 1e-3)
    errors = [np.linalg.norm(final_psi("mask", dt) - cap) for dt in (4e-3, 2e-3, 1e-3)]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.1)
