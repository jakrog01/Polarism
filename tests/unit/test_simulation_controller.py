from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.pump_field import PumpField
from polarism.laser.uniform_laser import UniformLaser
from polarism.simulation_controller import SimulationController
from polarism.results import ResultNode, ResultVisitor, register_observable
from polarism.results.observable_registry import available_observables
from polarism.results.visitors.visualization_visitor import VisualizationVisitor
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


def test_custom_visitor_receives_frames_and_finalizes() -> None:
    cfg = small_config()
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.004
    cfg.result.save_results = False
    cfg.result.save_interval = 2
    controller = SimulationController(cfg)

    class Collector(ResultVisitor):
        def __init__(self):
            self.times = []
            self.finalized = False

        def needs(self, nodes):
            return {"norm"}

        def visit_all(self, nodes, **context):
            self.times.append((context["t"], float(context["cached"]["norm"][0])))

        def finalize(self):
            self.finalized = True

    visitor = Collector()
    controller.add_visitor(visitor)
    controller.run()
    assert [t for t, _ in visitor.times] == pytest.approx([0, 0.002, 0.004])
    assert all(np.isfinite(value) for _, value in visitor.times)
    assert visitor.finalized


def test_observable_factory_selected(monkeypatch) -> None:
    @register_observable("test-observable")
    def factory(controller):
        return ResultNode(
            name="test_scalar", compute_fn=lambda **context: np.array(2.0),
            reduce_dim_fn=lambda value: value, cmap=None, scaling=None,
            clim=None, expose=False, save=True, cut=None, is_field=False,
        )

    cfg = small_config()
    cfg.result.observables = ("test-observable",)
    controller = SimulationController(cfg)

    class Collector(ResultVisitor):
        def needs(self, nodes):
            return {"test_scalar"}

        def visit_all(self, nodes, **context):
            self.value = float(context["cached"]["test_scalar"][0])

    visitor = Collector()
    controller.add_visitor(visitor)
    controller.run()
    assert visitor.value == 2.0
    monkeypatch.delitem(available_observables, "test-observable")


def test_storage_formats_merge_with_legacy_flags_once(tmp_path) -> None:
    cfg = small_config()
    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.formats = ("hdf5", "json")
    cfg.result.output_directory = str(tmp_path)
    controller = SimulationController(cfg)
    assert [type(storage).__name__ for storage in controller.storage_visitor.storages] == [
        "HDF5Storage", "JSONStorage",
    ]
    controller.run()


def test_pump_field_matches_laser_sum_and_exposes_standard_terms() -> None:
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
    assert pump.separable_terms() == []

    pump.lasers = [first]
    envelope, coefficient = pump.separable_term()
    assert envelope is first._spatial_envelope
    assert coefficient(0.2) == first.power_coefficient(0.2)
    assert pump.separable_terms()[0][0] is first._spatial_envelope
    np.testing.assert_array_equal(pump(0.2), first.get_power(X, Y, 0.2))

    pump.lasers = [first, second]
    assert pump.separable_term() is None
    terms = pump.separable_terms()
    assert len(terms) == 2
    assert terms[0][0] is first._spatial_envelope
    assert terms[1][0] is second._spatial_envelope
    assert [term[1](0.2) for term in terms] == [first.power_coefficient(0.2),
                                                second.power_coefficient(0.2)]
    expected = first.get_power(X, Y, 0.2) + second.get_power(X, Y, 0.2)
    np.testing.assert_array_equal(pump(0.2), expected)

    class InheritedPlugin(UniformLaser):
        pass

    inherited = InheritedPlugin(LaserParameters(laser_type="uniform", P0=1.0), X, Y)
    pump.lasers = [first, inherited]
    assert pump.separable_terms()[1][0] is inherited._spatial_envelope
    del inherited._spatial_envelope
    assert pump.separable_terms() is None

    class PluginLaser(UniformLaser):
        def get_power(self, X, Y, t):
            return super().get_power(X, Y, t)

    plugin = PluginLaser(LaserParameters(laser_type="uniform", P0=1.0), X, Y)
    pump.lasers = [plugin]
    assert pump.separable_term() is None
    assert pump.separable_terms() is None
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


@pytest.mark.parametrize("method", [
    "rk4-fdm",
    "rk4-fdm-fused",
    "rk4-cuda",
    "split-step-fft",
    "ip-rk4",
    "etd-rk2",
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


def test_visualization_visitor_alone_does_not_trigger_per_step_recording() -> None:
    class Recorder:
        updates = 0

        def update(self, **kwargs):
            self.updates += 1

    cfg = small_config()
    cfg.result.save_interval = 1
    controller = SimulationController(cfg)
    recorder = Recorder()
    controller.add_visitor(VisualizationVisitor(recorder))
    controller.run()
    assert recorder.updates == 1
