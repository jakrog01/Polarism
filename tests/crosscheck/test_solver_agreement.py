from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config

TOL_STAGE_COUPLED = 1e-3  # Stage-coupled solver agreement threshold.
TOL_SPLIT_COUPLED = 5e-2  # Split-coupled solver agreement threshold.
TOL_DIAGNOSTIC_SPLIT_COUPLED = 2e-1


def _metrics(solver: str, dt: float, reservoir: str):
    cfg = small_config(
        grid__nx=64,
        grid__ny=64,
        grid__lx=30.0,
        grid__ly=30.0,
        solver__method=solver,
        solver__dt=dt,
        solver__total_time=0.5,
        reservoir__reservoir_type=reservoir,
        laser__P0=0.2,
    )
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.sigma_space = 4.0
    controller = SimulationController(cfg)
    seed = 0.05 * np.exp(
        -(controller.grid.X**2 + controller.grid.Y**2) / (2.0 * 4.0**2)
    ) * np.exp(0.2j * controller.grid.X)
    controller.state.psi[...] = seed.astype(controller.state.psi.dtype)
    controller.run()
    rho = np.abs(np.asarray(controller.state.psi)) ** 2
    return float(rho.sum()), float(rho.max())


@pytest.mark.compliance
@pytest.mark.parametrize("solver", ["rk4-fdm-fused", "ip-rk4", "split-step-fft", "etd-rk2"])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
def test_solver_end_state_agreement(solver: str, reservoir: str) -> None:
    compute_engine.xp = np
    reference = np.array(_metrics("rk4-fdm", 5e-3, reservoir))
    actual = np.array(_metrics(solver, 2.5e-3 if solver in {"split-step-fft", "etd-rk2"} else 5e-3, reservoir))
    diagnostic = reservoir == "quadratic-double" and solver in {"split-step-fft", "etd-rk2"}
    tolerance = TOL_DIAGNOSTIC_SPLIT_COUPLED if diagnostic else TOL_SPLIT_COUPLED if solver in {"split-step-fft", "etd-rk2"} else TOL_STAGE_COUPLED
    assert np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-30)) < tolerance
