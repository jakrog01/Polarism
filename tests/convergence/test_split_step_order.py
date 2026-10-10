"""Temporal self-convergence of the reservoir-coupled Strang split step."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def _run(dt: float, overrides: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    options = dict(
        grid__nx=8,
        grid__ny=8,
        grid__lx=16.0,
        grid__ly=16.0,
        grid__grid_type="periodic",
        solver__method="split-step-fft",
        solver__precision="double",
        solver__dt=dt,
        solver__total_time=1.0,
        compute_engine__use_gpu=False,
        boundary_condition__absorption="no-absorption",
        reservoir__reservoir_type="single",
        physics__hbar=1.0,
        physics__m_eff=1.0,
        physics__g_C=0.8,
        physics__g_R=0.3,
        physics__g_I=0.2,
        physics__R=0.6,
        physics__gamma_C=0.6,
        physics__gamma_R=0.5,
        physics__gamma_I=0.4,
        physics__kappa=0.5,
        laser__laser_type="continuous-gaussian",
        laser__P0=2.0,
        laser__sigma_space=5.0,
    )
    cfg = small_config(**(options | overrides))
    controller = SimulationController(cfg)
    x, y = controller.grid.X, controller.grid.Y
    controller.state.psi[:] = (
        0.5 + 0.3 * np.exp(-(x * x + y * y) / 20)
    ) * np.exp(0.15j * x)
    controller.run()
    return controller.state.psi.copy(), controller.reservoir.get_state()[0].copy()


@pytest.mark.parametrize("case,overrides", [
    ("loss without reservoir coupling", {
        "physics__R": 0.0,
        "physics__g_R": 0.0,
        "physics__g_I": 0.0,
        "laser__P0": 0.0,
    }),
    ("single reservoir", {}),
    ("quadratic-double reservoir", {"reservoir__reservoir_type": "quadratic-double"}),
    ("CAP and potential", {
        "boundary_condition__absorption": "cap",
        "boundary_condition__strength": 1.0,
        "potential__potential_type": "double-well-supergaussian",
        "potential__V1": -0.6,
        "potential__V2": -0.4,
    }),
])
def test_split_step_temporal_order(case: str, overrides: dict[str, object]) -> None:
    fields = [_run(dt, overrides) for dt in (0.04, 0.02, 0.01)]
    for index, name in enumerate(("psi", "nR")):
        if case == "loss without reservoir coupling" and name == "nR":
            assert all(not np.any(field[index]) for field in fields)
            continue
        coarse_error = np.linalg.norm(fields[0][index] - fields[1][index])
        fine_error = np.linalg.norm(fields[1][index] - fields[2][index])
        order = np.log2(coarse_error / fine_error)
        assert np.isfinite(order)
        assert order >= 1.8, f"{case} {name}: order={order:.3f}"
