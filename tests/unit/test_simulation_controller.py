from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.pump_field import PumpField
from polarism.laser.uniform_laser import UniformLaser
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_total_pump_matches_zero_initialized_sum_without_mutating_lasers() -> None:
    controller = SimulationController(small_config())
    X, Y = controller.grid.X, controller.grid.Y
    configs = [
        LaserParameters(laser_type="continuous-gaussian", P0=0.7, sigma_space=2.0),
        LaserParameters(laser_type="uniform", P0=0.3),
        LaserParameters(laser_type="pulse-gaussian", P0=0.5, Pmax=0.5,
                        sigma_space=3.0, sigma_time=0.05, pulse_separation=1.0,
                        cutoff_sigma=3.0, delay=0.2),
        LaserParameters(laser_type="uniform", P0=0.15),
    ]
    controller.lasers = [LaserFactory.create_laser(cfg, X, Y)[0] for cfg in configs]
    t = 0.1
    fields = [laser.get_power(X, Y, t) for laser in controller.lasers]
    snapshots = [field.copy() for field in fields]
    expected = np.zeros(X.shape, dtype=np.float64)
    for field in fields:
        expected += field
    zero = fields[2]

    actual = controller._compute_total_pump(t)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    for field, snapshot in zip(fields, snapshots):
        np.testing.assert_array_equal(field, snapshot)
    assert all(actual is not field for field in fields)
    assert controller.lasers[2].P is zero
    assert np.count_nonzero(zero) == 0


def test_pump_field_matches_laser_sum_and_only_exposes_one_standard_laser() -> None:
    controller = SimulationController(small_config())
    X, Y = controller.grid.X, controller.grid.Y
    first = LaserFactory.create_laser(
        LaserParameters(laser_type="continuous-gaussian", P0=0.7), X, Y
    )[0]
    second = LaserFactory.create_laser(
        LaserParameters(laser_type="uniform", P0=0.3), X, Y
    )[0]
    pump = PumpField([], controller.grid, "double")
    np.testing.assert_array_equal(pump(0.2), np.zeros_like(X))
    assert pump.separable_term() is None

    pump.lasers = [first]
    envelope, coefficient = pump.separable_term()
    assert envelope is first._spatial_envelope
    assert coefficient(0.2) == first.power_coefficient(0.2)
    np.testing.assert_array_equal(pump(0.2), first.get_power(X, Y, 0.2))

    pump.lasers = [first, second]
    assert pump.separable_term() is None
    expected = first.get_power(X, Y, 0.2) + second.get_power(X, Y, 0.2)
    np.testing.assert_array_equal(pump(0.2), expected)

    class PluginLaser(UniformLaser):
        def get_power(self, X, Y, t):
            return super().get_power(X, Y, t)

    plugin = PluginLaser(LaserParameters(laser_type="uniform", P0=1.0), X, Y)
    pump.lasers = [plugin]
    assert pump.separable_term() is None
    np.testing.assert_array_equal(pump(0.2), plugin.get_power(X, Y, 0.2))


def test_single_laser_returns_its_field_directly() -> None:
    controller = SimulationController(small_config())
    field = controller._compute_total_pump(0.0)
    assert field is controller.lasers[0].P


def test_pulse_zero_field_is_reused_before_delay_and_in_gaps() -> None:
    controller = SimulationController(small_config())
    X, Y = controller.grid.X, controller.grid.Y
    cfg = LaserParameters(laser_type="pulse-gaussian", P0=0.5, Pmax=0.5,
                          sigma_time=0.05, pulse_separation=1.0,
                          cutoff_sigma=3.0, delay=0.2)
    pulse = LaserFactory.create_laser(cfg, X, Y)[0]
    controller.lasers = [pulse]
    zero = controller._compute_total_pump(0.0)
    assert controller._compute_total_pump(0.1) is zero

    active = controller._compute_total_pump(cfg.delay + pulse.phase)
    assert active is not zero
    assert np.max(active) > 0
    gap = cfg.delay + pulse.phase + 0.2
    assert controller._compute_total_pump(gap) is zero
    assert pulse.P is zero
    assert np.count_nonzero(zero) == 0

    uniform = LaserFactory.create_laser(LaserParameters(laser_type="uniform", P0=1.0), X, Y)[0]
    controller.lasers = [pulse, uniform]
    summed = controller._compute_total_pump(gap)
    assert summed is not zero
    assert np.count_nonzero(zero) == 0
    np.testing.assert_array_equal(summed, uniform.P)


def test_single_precision_pump_fields_stay_float32() -> None:
    controller = SimulationController(small_config(solver__precision="single"))
    X, Y = controller.grid.X, controller.grid.Y
    cfg = LaserParameters(laser_type="pulse-gaussian", power_definition="pulse_energy",
                          P0=0.5, Pmax=0.5, sigma_time=0.05)
    pulse = LaserFactory.create_laser(cfg, X, Y, precision="single")[0]
    uniform = LaserFactory.create_laser(
        LaserParameters(laser_type="uniform", P0=0.2), X, Y, precision="single"
    )[0]
    controller.lasers = [pulse]
    assert controller._compute_total_pump(-1.0).dtype == np.float32
    assert controller._compute_total_pump(pulse.phase).dtype == np.float32
    controller.lasers = [pulse, uniform]
    assert controller._compute_total_pump(pulse.phase).dtype == np.float32


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
