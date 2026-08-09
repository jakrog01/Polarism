from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_factory import LaserFactory
from tests.unit.conftest import grid

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.
TOL_LASER_INTEG = 1e-6  # Numerical integration of a Gaussian.


def _laser(cfg: LaserParameters):
    compute_engine.xp = np
    g = grid(40, 40)
    return LaserFactory.create_laser(cfg, g.X, g.Y)[0], g


def test_uniform_and_continuous_gaussian_peaks() -> None:
    uniform, g = _laser(LaserParameters(laser_type="uniform", P0=1.0))
    assert np.max(np.abs(uniform.get_power(g.X, g.Y, 0.0) - 1.0)) < TOL_MACHINE_F64
    cfg = LaserParameters(laser_type="continuous-gaussian", P0=2.0, x0=0.25, y0=-0.25, sigma_space=1.0)
    laser, g = _laser(cfg)
    power = laser.get_power(g.X, g.Y, 0.0)
    iy, ix = np.where((g.X == cfg.x0) & (g.Y == cfg.y0))
    iy1, ix1 = np.where((g.X == cfg.x0 + cfg.sigma_space) & (g.Y == cfg.y0))
    assert np.isclose(power[iy[0], ix[0]], cfg.P0, rtol=TOL_MACHINE_F64)
    assert np.isclose(power[iy1[0], ix1[0]], cfg.P0 * np.exp(-0.5), rtol=TOL_LASER_INTEG)


def test_pulse_timing_cutoff_ramp_and_count() -> None:
    cfg = LaserParameters(laser_type="pulse-gaussian", P0=0.4, Pmax=1.2, x0=0.25, y0=-0.25, sigma_time=0.1, pulse_separation=1.0, n_pulses=0)
    laser, g = _laser(cfg)
    assert laser._pulse_index(0.0) == 0
    assert laser._pulse_index(cfg.pulse_separation + laser.phase) == 1
    assert laser.get_power(g.X, g.Y, laser.phase + cfg.cutoff_sigma * cfg.sigma_time + 1e-9).max() == 0.0
    iy, ix = np.where((g.X == cfg.x0) & (g.Y == cfg.y0))
    for k in range(6):
        got = laser.get_power(g.X, g.Y, laser.phase + k * cfg.pulse_separation)[iy[0], ix[0]]
        assert np.isclose(got, min(cfg.P0 + k * (cfg.Pmax - cfg.P0), cfg.Pmax))
    limited, _ = _laser(LaserParameters(**{**vars(cfg), "n_pulses": 3}))
    assert limited.get_power(g.X, g.Y, limited.phase + 3 * cfg.pulse_separation).max() == 0.0


def test_pulse_energy_normalisation_and_multiple_mode(tmp_path) -> None:
    cfg = LaserParameters(laser_type="pulse-gaussian", power_definition="pulse_energy", P0=0.4, sigma_space=1.5, sigma_time=0.1)
    laser, g = _laser(cfg)
    dt = cfg.sigma_time / 50
    times = np.arange(laser.phase - cfg.cutoff_sigma * cfg.sigma_time, laser.phase + cfg.cutoff_sigma * cfg.sigma_time + dt / 2, dt)
    energy = sum(np.sum(laser.get_power(g.X, g.Y, t)) * g.dx * g.dy * dt for t in times)
    assert np.isclose(energy, cfg.P0, rtol=1e-4)
    cfg2 = LaserParameters(**{**vars(cfg), "sigma_space": 3.0})
    laser2, _ = _laser(cfg2)
    energy2 = sum(np.sum(laser2.get_power(g.X, g.Y, t)) * g.dx * g.dy * dt for t in times)
    assert np.isclose(energy2, energy, rtol=1e-4)
    yaml_path = tmp_path / "lasers.yaml"
    yaml_path.write_text("lasers:\n  - laser_type: uniform\n    P0: 1.0\n  - laser_type: uniform\n    P0: 2.0\n")
    multiple = LaserFactory.create_laser(LaserParameters(mode="multiple", config_file=str(yaml_path)), g.X, g.Y)
    assert len(multiple) == 2
    assert np.array_equal(sum(item.get_power(g.X, g.Y, 0.0) for item in multiple), np.full_like(g.X, 3.0))
