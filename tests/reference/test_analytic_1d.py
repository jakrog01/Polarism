"""One-dimensional free-packet broadening."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController


@pytest.mark.compliance
@pytest.mark.parametrize("method,tolerance", [("rk4-fdm", 0.003), ("split-step-fft", 1e-8)])
def test_free_gaussian_packet_width(method, tolerance):
    cfg = Config()
    cfg.grid.ndim = 1
    cfg.grid.nx = 512
    cfg.grid.lx = 64.0
    cfg.solver.method = method
    cfg.solver.dt = 0.002
    cfg.solver.total_time = 1.0
    cfg.result.save_results = False
    cfg.physics.hbar = 1.0
    cfg.physics.m_eff = 1.0
    cfg.physics.g_C = 0.0
    cfg.physics.g_R = 0.0
    cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.physics.gamma_C = 0.0
    cfg.physics.gamma_R = 0.0
    cfg.laser.P0 = 0.0
    run = SimulationController(cfg)
    sigma = 1.0
    x = run.grid.X
    run.state.psi = np.exp(-x * x / (4 * sigma * sigma)).astype(np.complex128)
    run.run()
    probability = np.abs(run.state.psi) ** 2
    mean = np.sum(x * probability) / np.sum(probability)
    variance = np.sum((x - mean) ** 2 * probability) / np.sum(probability)
    expected = sigma ** 2 + (cfg.physics.hbar * cfg.solver.total_time
                             / (2 * cfg.physics.m_eff * sigma)) ** 2
    assert abs(variance - expected) / expected < tolerance
