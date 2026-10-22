"""Sweep construction and CPU execution tests."""
from __future__ import annotations

import numpy as np
import pytest

from polarism import Config, SimulationController, Sweep
from polarism.sweep import _cache_file, _write_cache, autotune_batch_size


CPU_METHODS = ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "split-step-fft",
               "etd-rk2", "ip-rk4", "ifrk-fft-cuda", "ifrk4-fft-cuda"]


def small_config(method="rk4-fdm"):
    cfg = Config()
    cfg.grid.nx = 8
    cfg.grid.ny = 8
    cfg.solver.total_time = 0.003
    cfg.solver.method = method
    cfg.physics.init_seed = 7
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.potential.potential_type = "double-well-supergaussian"
    return cfg


def assert_cpu_matches_separate(cfg, members, batched=None):
    if batched is None:
        batched = SimulationController(cfg, members=members)
    batched.run()
    for index, overrides in enumerate(members):
        separate = SimulationController(cfg, members=[overrides])
        separate.run()
        assert np.array_equal(batched.state.psi[index], separate.state.psi)
        for whole, single in zip(batched.reservoir.get_state(), separate.reservoir.get_state()):
            assert np.array_equal(whole[index], single)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("method", CPU_METHODS)
def test_batched_members_equal_separate(method, precision):
    cfg = small_config(method)
    cfg.solver.precision = precision
    members = [
        {"physics.g_C": 0.002, "physics.init_seed": 11, "laser.P0": 0.3,
         "laser.sigma_space": 8.0, "potential.V1": -0.002},
        {"physics.g_C": 0.003, "physics.init_seed": 12, "laser.P0": 0.5,
         "laser.sigma_space": 10.0, "potential.V1": -0.003},
    ]
    batched = SimulationController(cfg, members=members)
    assert batched.state.psi.shape == (2, 8, 8)
    assert_cpu_matches_separate(cfg, members, batched)


@pytest.mark.parametrize("method", CPU_METHODS)
def test_cap_with_batched_hbar_matches_separate(method):
    cfg = small_config(method)
    cfg.boundary_condition.absorption = "cap"
    members = [{"physics.hbar": 0.55, "physics.g_C": 0.002},
               {"physics.hbar": 0.75, "physics.g_C": 0.003}]
    batched = SimulationController(cfg, members=members)
    assert batched.boundary_condition.before_step_action().shape == (2, 8, 8)
    assert_cpu_matches_separate(cfg, members, batched)


def pulse_config(method):
    cfg = small_config(method)
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.02
    cfg.solver.dt = 0.01
    cfg.solver.total_time = 0.12
    return cfg


@pytest.mark.parametrize("method", CPU_METHODS)
def test_pulse_member_coefficients_match_separate(method):
    cfg = pulse_config(method)
    members = [{"laser.delay": 0.0, "laser.sigma_time": 0.02},
               {"laser.delay": 0.02, "laser.sigma_time": 0.03}]
    batched = SimulationController(cfg, members=members)
    envelope, coefficient = batched.pump_field.separable_terms()[0]
    assert envelope.ndim == 2
    assert np.asarray(coefficient(0.08)).shape == (2,)
    assert_cpu_matches_separate(cfg, members, batched)


def test_product_zip_and_groups():
    cfg = small_config()
    parameters = {"solver.dt": [0.001, 0.0005], "laser.P0": [0.1, 0.2]}
    sweep = Sweep(cfg, parameters)
    assert sweep.members == [
        {"solver.dt": 0.001, "laser.P0": 0.1},
        {"solver.dt": 0.001, "laser.P0": 0.2},
        {"solver.dt": 0.0005, "laser.P0": 0.1},
        {"solver.dt": 0.0005, "laser.P0": 0.2},
    ]
    assert [group["indices"] for group in sweep.groups] == [[0, 1], [2, 3]]
    assert len(sweep.run()) == 4
    assert len(Sweep(cfg, parameters, mode="zip").members) == 2


def test_sweep_errors():
    cfg = small_config()
    with pytest.raises(ValueError, match="solver.dt"):
        SimulationController(cfg, members=[{"solver.dt": 0.002}])
    with pytest.raises(ValueError, match="Unknown sweep path"):
        Sweep(cfg, {"physics.unknown": [1]})
    with pytest.raises(ValueError, match="equal lengths"):
        Sweep(cfg, {"laser.P0": [1, 2], "physics.g_C": [0.1]}, mode="zip")
    cfg.result.save_results = True
    controller = SimulationController(cfg, members=[{}, {}])
    assert controller.state.psi.shape == (2, 8, 8)


def test_autotune_chooses_peak_and_stops_after_two_drops():
    called = []
    elapsed = {1: 1.0, 2: 1.0, 4: 1.0, 8: 3.0, 16: 9.0, 32: 1.0}

    def timer(size, steps):
        called.append((size, steps))
        return elapsed[size]

    assert autotune_batch_size(32, timer, max_steps=8) == 4
    assert called == [(size, 8) for size in [1, 2, 4, 8, 16]]


def test_autotune_adapts_timing_steps():
    called = []

    def timer(size, steps):
        called.append((size, steps))
        return steps / 1000 / size

    assert autotune_batch_size(2, timer, max_steps=128) == 2
    assert called == [(1, 8), (1, 16), (1, 32), (1, 64),
                      (2, 8), (2, 16), (2, 32), (2, 64), (2, 128)]


def test_cache_location_and_atomic_write(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = _cache_file()
    assert path == tmp_path / "polarism" / "sweep_batches.json"
    _write_cache(path, {"one": {"choice": 2, "limit": 4}})
    assert path.read_text().find('"choice": 2') >= 0
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("method", CPU_METHODS)
def test_multiple_laser_entry_override(tmp_path, method):
    cfg = pulse_config(method)
    cfg.laser.mode = "multiple"
    path = tmp_path / "lasers.yaml"
    path.write_text("lasers:\n  - laser_type: continuous-gaussian\n    P0: 0.2\n  - laser_type: pulse-gaussian\n    delay: 0.0\n")
    cfg.laser.config_file = str(path)
    members = [{"lasers[1].delay": 0.0, "lasers[0].P0": 0.2},
               {"lasers[1].delay": 0.02, "lasers[0].P0": 0.3}]
    batched = SimulationController(cfg, members=members)
    assert batched.lasers[1].delay == 0.0
    assert batched.pump_field.member_lasers[1][1].delay == 0.02
    assert batched.pump_field.member_lasers[1][0].P0 == 0.3
    assert_cpu_matches_separate(cfg, members, batched)


def test_shared_laser_and_potential_arrays():
    cfg = small_config()
    shared = SimulationController(cfg, members=[{}, {}])
    envelope, coefficient = shared.pump_field.separable_terms()[0]
    assert envelope.ndim == 2
    assert np.ndim(coefficient(0.0)) == 0
    assert shared.potential.ndim == 2
    varying = SimulationController(cfg, members=[{"potential.V1": -0.002},
                                                 {"potential.V1": -0.003}])
    assert varying.potential.shape == (2, 8, 8)
    assert_cpu_matches_separate(cfg, [{}, {}], shared)
    assert_cpu_matches_separate(cfg, [{"potential.V1": -0.002},
                                      {"potential.V1": -0.003}], varying)


def test_plugin_laser_extra_key(monkeypatch):
    from polarism.laser.abstract_laser import AbstractLaser
    from polarism.laser.laser_registry import available_lasers

    class RingLaser(AbstractLaser):
        def __init__(self, laser_config, X, Y, precision="double"):
            super().__init__(laser_config, X, Y, precision)
            self.radius = laser_config.extra["ring_radius"]
            self._finalize_spatial_envelope(X, Y)

        def _amplitude(self, t):
            return self.P0

        def _P_space(self, X, Y):
            return self.xp.exp(-((self.xp.sqrt(X ** 2 + Y ** 2) - self.radius) ** 2))

        def _P_time(self, t):
            return 1.0

    monkeypatch.setitem(available_lasers, "sweep-ring-test", RingLaser)
    cfg = small_config()
    cfg.laser.laser_type = "sweep-ring-test"
    cfg.laser.extra = {"ring_radius": 2.0}
    members = [{"laser.extra.ring_radius": 2.0},
               {"laser.extra.ring_radius": 3.0}]
    batched = SimulationController(cfg, members=members)
    batched.run()
    for index, overrides in enumerate(members):
        separate = SimulationController(cfg, members=[overrides])
        separate.run()
        assert np.array_equal(batched.state.psi[index], separate.state.psi)


def test_cli_sweep_parsing(tmp_path, monkeypatch):
    import sys
    from polarism import cli

    cfg = small_config()
    path = tmp_path / "sweep.yaml"
    path.write_text("parameters:\n  laser.P0: [0.2, 0.3]\nmode: zip\nsweep:\n  batch_size: 2\n")
    seen = {}

    def fake_cli(model, args):
        seen["args"] = args
        return cfg

    class FakeSweep:
        def __init__(self, config, parameters, mode):
            seen["parameters"] = parameters
            seen["mode"] = mode

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(cli.tyro, "cli", fake_cli)
    monkeypatch.setattr(cli, "Sweep", FakeSweep)
    monkeypatch.setattr(sys, "argv", ["polarism", "--solver.dt", "0.001", "--sweep", str(path)])
    cli.main()
    assert seen == {"args": ["--solver.dt", "0.001"],
                    "parameters": {"laser.P0": [0.2, 0.3]},
                    "mode": "zip", "ran": True}
    assert cfg.sweep.batch_size == 2


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda"])
def test_gpu_controller_sweep_equals_separate(method):
    from polarism.compute_engine import has_cuda_device

    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = small_config(method)
    cfg.compute_engine.use_gpu = True
    cfg.solver.precision = "single"
    members = [{"physics.g_C": 0.002, "laser.P0": 0.2, "laser.sigma_space": 8.0},
               {"physics.g_C": 0.003, "laser.P0": 0.3, "laser.sigma_space": 10.0}]
    batch = SimulationController(cfg, members=members)
    batch.run()
    for index, overrides in enumerate(members):
        separate = SimulationController(cfg, members=[overrides])
        separate.run()
        np.testing.assert_allclose(cp.asnumpy(batch.state.psi[index]),
                                   cp.asnumpy(separate.state.psi), rtol=2e-5, atol=2e-5)
        for whole, single in zip(batch.reservoir.get_state(), separate.reservoir.get_state()):
            np.testing.assert_allclose(cp.asnumpy(whole[index]), cp.asnumpy(single),
                                       rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda"])
def test_gpu_pulse_member_coefficients_with_graphs(method):
    from polarism.compute_engine import has_cuda_device

    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = pulse_config(method)
    cfg.compute_engine.use_gpu = True
    cfg.solver.precision = "single"
    cfg.solver.cuda_graphs = True
    members = [{"laser.delay": 0.0, "laser.sigma_time": 0.02},
               {"laser.delay": 0.02, "laser.sigma_time": 0.03}]
    batch = SimulationController(cfg, members=members)
    envelope, coefficient = batch.pump_field.separable_terms()[0]
    assert envelope.ndim == 2
    assert np.asarray(coefficient(0.08)).shape == (2,)
    batch.run()
    for index, overrides in enumerate(members):
        separate = SimulationController(cfg, members=[overrides])
        separate.run()
        np.testing.assert_allclose(cp.asnumpy(batch.state.psi[index]),
                                   cp.asnumpy(separate.state.psi), rtol=2e-5, atol=2e-5)
        for whole, single in zip(batch.reservoir.get_state(), separate.reservoir.get_state()):
            np.testing.assert_allclose(cp.asnumpy(whole[index]), cp.asnumpy(single),
                                       rtol=2e-5, atol=2e-5)
