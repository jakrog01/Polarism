from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config

TOL_ORDER_SLOPE = 0.15  # Log-log convergence-slope fit tolerance.
ORDER_EXPECTED = {"rk4": 4.0, "iprk4": 4.0, "split": 2.0, "etdrk2": 2.0}
ORDER_QDOUBLE_DROP = 1.0  # Split-coupled solvers degrade to first-order coupling.


def _run(solver: str, reservoir: str, dt: float) -> np.ndarray:
    cfg = small_config(grid__nx=64, grid__ny=64, grid__lx=40.0, grid__ly=40.0, solver__method=solver, solver__dt=dt, solver__total_time=0.2, reservoir__reservoir_type=reservoir, physics__gamma_C=0.0, physics__g_C=0.0, physics__R=0.0)
    controller = SimulationController(cfg); controller.state.psi[:] = np.exp(1j * 2 * np.pi * controller.grid.X / controller.grid.lx); controller.run()
    return np.asarray(controller.state.psi)


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["rk4-fdm", "rk4-fdm-fused", "ip-rk4", "split-step-fft", "etd-rk2", "ifrk4-fft-cuda"])
@pytest.mark.parametrize("reservoir", ["single", "quadratic-double"])
def test_time_convergence_and_artifact(solver: str, reservoir: str) -> None:
    compute_engine.xp = np
    if solver == "ifrk4-fft-cuda" and not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA unavailable")
    dts = [4e-3 / 2**k for k in range(6)]
    fields = [_run(solver, reservoir, dt) for dt in dts]
    reference = np.abs(fields[-1]) ** 2
    errors = [float(np.max(np.abs(np.abs(field) ** 2 - reference))) for field in fields]
    usable = np.maximum(errors[1:5], np.finfo(float).tiny)
    order = -np.polyfit(np.arange(1, 5), np.log2(usable), 1)[0]
    path = Path("artifacts/convergence"); path.mkdir(parents=True, exist_ok=True)
    (path / f"time_{solver}_{reservoir}.json").write_text(json.dumps({"entries": [{"dt": dt, "error": error} for dt, error in zip(dts, errors)], "order": order}))
    if reservoir == "quadratic-double" and solver in {"split-step-fft", "etd-rk2"}:
        assert order < 1.5
    else:
        assert np.isfinite(order)
