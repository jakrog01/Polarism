"""Observed fixed-step orders for a smooth, periodic coupled problem."""
import numpy as np
import pytest

from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from polarism.solver.tableaus import TABLEAUS
from tests.unit.conftest import small_config


def _trajectory(tableau, dt):
    cfg = small_config(
        grid__nx=8, grid__ny=8, solver__method="ifrk-fft-cuda",
        solver__tableau=tableau, solver__precision="double",
        solver__dt=dt, solver__total_time=1.6,
        reservoir__reservoir_type="quadratic-double",
        physics__g_C=1.1, physics__g_R=0.8, physics__g_I=0.3,
        physics__R=0.4, physics__kinetic_relaxation_eta=0.03,
        compute_engine=ComputeEngineParameters(use_gpu=False),
    )
    controller = SimulationController(cfg)
    x, y = controller.grid.X, controller.grid.Y
    controller.state.psi = np.asarray(
        0.35 * (1 + 0.12 * np.cos(2 * np.pi * x / cfg.grid.lx))
        * np.exp(0.3j * np.sin(2 * np.pi * y / cfg.grid.ly)), dtype=np.complex128)
    controller.reservoir.set_state((
        np.asarray(0.7 + 0.04 * np.cos(2 * np.pi * x / cfg.grid.lx)),
        np.asarray(0.6 + 0.04 * np.sin(2 * np.pi * y / cfg.grid.ly)),
    ))
    potential = 0.6 * np.cos(2 * np.pi * x / cfg.grid.lx) - 0.07j * (
        1 + 0.1 * np.sin(2 * np.pi * y / cfg.grid.ly))
    envelope = 1 + 0.1 * np.cos(2 * np.pi * x / cfg.grid.lx)
    pump = lambda t: envelope * (1.5 + 0.7 * np.sin(3 * t) + 0.4 * np.cos(5 * t))
    steps = round(cfg.solver.total_time / dt)
    controller.solver.advance(potential, pump, controller.reservoir,
                              controller.boundary_condition, controller.state, 0, steps)
    return np.concatenate([
        controller.state.psi.ravel(),
        *(field.ravel() for field in controller.reservoir.get_state()),
    ])


@pytest.mark.parametrize("name", TABLEAUS)
def test_ifrk_observed_order(name):
    reference = _trajectory("vern9", 0.003125)
    coarse = _trajectory(name, 0.4)
    fine = _trajectory(name, 0.2)
    error_coarse = np.linalg.norm(coarse - reference)
    error_fine = np.linalg.norm(fine - reference)
    observed = np.log2(error_coarse / error_fine)
    assert observed >= TABLEAUS[name].order - 0.4, (name, observed, error_coarse, error_fine)
