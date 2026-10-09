from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests._helpers import initial_packet, rel_l2_err
from tests.unit.conftest import small_config

TOL_ORDER_SLOPE = 0.15  # Log-log convergence-slope fit tolerance.
ORDER_EXPECTED = {"rk4": 4.0, "iprk4": 4.0, "split": 2.0, "etdrk2": 2.0}
ORDER_QDOUBLE_DROP = 1.0  # Split-coupled solvers degrade to first-order coupling.


def _run(solver: str, reservoir: str, dt: float) -> np.ndarray:
    use_gpu = solver == "ifrk4-fft-cuda"
    reservoir_overrides = (
        {
            "physics__gamma_C": 0.0,
            "physics__g_R": 0.0,
            "physics__R": 0.0,
            "laser__P0": 0.0,
        }
        if reservoir == "single"
        else {
            "physics__g_R": 0.05,
            "physics__R": 0.1,
            "laser__P0": 1.0,
        }
    )
    cfg = small_config(
        grid__nx=64,
        grid__ny=64,
        grid__lx=40.0,
        grid__ly=40.0,
        solver__method=solver,
        solver__dt=dt,
        solver__total_time=1.0,
        reservoir__reservoir_type=reservoir,
        physics__g_C=5e-3,
        laser__laser_type="continuous-gaussian",
        laser__sigma_space=8.0,
        compute_engine=ComputeEngineParameters(use_gpu=use_gpu),
        **reservoir_overrides,
    )
    controller = SimulationController(cfg)
    packet_scale = 1000.0 if solver in {"rk4-fdm", "rk4-fdm-fused", "ip-rk4", "ifrk4-fft-cuda"} else 100.0
    controller.state.psi[:] = packet_scale * initial_packet(controller.grid, sigma=4.0, kx=3.0)
    controller.run()
    return compute_engine.to_cpu(controller.state.psi)


def _richardson_reference(fields: list[np.ndarray]) -> np.ndarray:
    coarse, fine = (np.abs(field) ** 2 for field in fields)
    return (4.0 * fine - coarse) / 3.0


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["rk4-fdm", "rk4-fdm-fused", "ip-rk4", "split-step-fft", "etd-rk2", "ifrk4-fft-cuda"])
@pytest.mark.parametrize("reservoir", ["single", "quadratic-double"])
def test_time_convergence_and_artifact(solver: str, reservoir: str) -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=solver == "ifrk4-fft-cuda"))
    dts = [4e-3 / 2**k for k in range(6)]
    fields = [_run(solver, reservoir, dt) for dt in dts]
    reference = _richardson_reference([
        _run(solver, reservoir, dts[-1] / 2),
        _run(solver, reservoir, dts[-1] / 4),
    ])
    errors = [
        rel_l2_err(np.abs(field) ** 2, reference)
        for field in fields
    ]
    usable = np.maximum(errors[1:5], np.finfo(float).tiny)
    order = -np.polyfit(np.arange(1, 5), np.log2(usable), 1)[0]
    path = Path("artifacts/convergence"); path.mkdir(parents=True, exist_ok=True)
    (path / f"time_{solver}_{reservoir}.json").write_text(json.dumps({"entries": [{"dt": dt, "error": error} for dt, error in zip(dts, errors)], "order": order}))
    assert np.isfinite(order)
    if solver in {"rk4-fdm", "rk4-fdm-fused", "ip-rk4", "ifrk4-fft-cuda"}:
        assert max(errors) / min(errors) >= 1e4
