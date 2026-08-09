from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config

TOL_ORDER_SLOPE = 0.15  # Log-log convergence-slope fit tolerance.


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["rk4-fdm", "rk4-fdm-fused", "ip-rk4", "split-step-fft", "ifrk4-fft-cuda", "etd-rk2"])
def test_space_convergence_artifact(solver: str) -> None:
    compute_engine.xp = np
    if solver == "ifrk4-fft-cuda" and not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA unavailable")
    norms = []
    for nx in (64, 128, 256, 512):
        cfg = small_config(grid__nx=nx, grid__ny=nx, grid__lx=40.0, grid__ly=40.0, solver__method=solver, solver__dt=1e-4, solver__total_time=0.05, physics__gamma_C=0.0, physics__R=0.0)
        controller = SimulationController(cfg); controller.run(); norms.append(float(np.sum(np.abs(controller.state.psi) ** 2)))
    path = Path("artifacts/convergence"); path.mkdir(parents=True, exist_ok=True)
    (path / f"space_{solver}.json").write_text(json.dumps({"nx": [64, 128, 256, 512], "norm": norms}))
    assert np.isfinite(norms).all()
