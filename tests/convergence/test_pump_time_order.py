"""Temporal self-convergence with a pulsed, stage-dependent reservoir source.

The wide ``cutoff_sigma`` makes the pulse cutoff jumps negligible (~1e-14),
so the pump is smooth and every solver should reach its nominal order.
"""
from __future__ import annotations

import numpy as np
import pytest

from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def _run(method: str, dt: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = small_config(
        grid__nx=8,
        grid__ny=8,
        grid__lx=20.0,
        grid__ly=20.0,
        grid__grid_type="periodic",
        solver__method=method,
        solver__precision="double",
        solver__dt=dt,
        solver__total_time=1.0,
        compute_engine__use_gpu=False,
        boundary_condition__absorption="no-absorption",
        reservoir__reservoir_type="single",
        physics__hbar=1.0,
        physics__m_eff=1.0,
        physics__g_C=0.1,
        physics__g_R=0.2,
        physics__R=1.0,
        physics__gamma_C=0.2,
        physics__gamma_R=0.5,
        laser__laser_type="pulse-gaussian",
        laser__power_definition="peak_amplitude",
        laser__P0=2.0,
        laser__Pmax=2.0,
        laser__sigma_space=4.0,
        laser__sigma_time=0.1,
        laser__cutoff_sigma=8.0,
        laser__n_pulses=1,
    )
    controller = SimulationController(cfg)
    controller.state.psi.fill(0.3 + 0.1j)
    controller.run()
    return controller.state.psi.copy(), controller.reservoir.nR.copy()


@pytest.mark.parametrize("method,min_order", [
    ("rk4-fdm", 3.7),
    ("rk4-fdm-fused", 3.7),
    ("rk4-cuda", 3.7),
    ("ip-rk4", 3.7),
    ("ifrk4-fft-cuda", 3.7),
    ("etd-rk2", 1.8),
])
def test_pump_time_order(method: str, min_order: float) -> None:
    fields = [_run(method, dt) for dt in (0.02, 0.01, 0.005)]
    for index, name in enumerate(("psi", "nR")):
        coarse_error = np.linalg.norm(fields[0][index] - fields[1][index])
        fine_error = np.linalg.norm(fields[1][index] - fields[2][index])
        order = np.log2(coarse_error / fine_error)
        assert np.isfinite(order)
        assert order >= min_order, f"{method} {name}: order={order:.3f}"
