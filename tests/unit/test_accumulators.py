"""Accumulated observables, persistence, and restart contracts."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from polarism.analysis import g1, g2
from polarism.compute_engine import has_cuda_device
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.results import load_results
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def config(**overrides):
    cfg = small_config(noise__enabled=False, reservoir__reservoir_type="none",
                       laser__P0=0.0, solver__total_time=0.04)
    cfg.result.accumulators = {"momentum-distribution": {"every": 0.02},
                               "g2": {"every": 0.02}}
    cfg.initial_condition.type = "plane-wave"
    cfg.initial_condition.amplitude = 0.2
    cfg.checkpoint.on_signal = False
    for name, value in overrides.items():
        section, field = name.split("__", 1)
        setattr(getattr(cfg, section), field, value)
    return cfg


def sampled(controller, fields):
    for index, field in enumerate(fields):
        controller.state.psi = field
        for accumulator in controller._accumulators.values():
            accumulator.update(controller.state, index * controller.cfg.solver.dt)
    return {result.name: np.asarray(result.value)
            for accumulator in controller._accumulators.values() for result in accumulator.results()}


def test_plane_wave_sum_rule_and_correlations():
    controller = SimulationController(config())
    grid = controller.grid
    kx = 2 * np.pi / grid.lx
    field = 1.2 * np.exp(1j * kx * grid.X)
    values = sampled(controller, [field])
    momentum = values["momentum_distribution"]
    expected_norm = np.sum(np.abs(field) ** 2) * np.prod(grid.spacing)
    np.testing.assert_allclose(momentum.sum(), expected_norm, rtol=1e-12)
    assert np.count_nonzero(momentum > momentum.max() * 1e-12) == 1
    np.testing.assert_allclose(np.abs(values["g1"]), 1, rtol=1e-12)
    np.testing.assert_allclose(values["g2"], 1, rtol=1e-12)


def test_complex_gaussian_ensemble_g2_and_g1():
    controller = SimulationController(config())
    rng = np.random.default_rng(12)
    fields = [(rng.normal(size=controller.grid.shape) +
               1j * rng.normal(size=controller.grid.shape)) / np.sqrt(2)
              for _ in range(3000)]
    values = sampled(controller, fields)
    assert abs(np.mean(values["g2"]) - 2) < 0.1
    center = tuple(size // 2 for size in controller.grid.shape)
    assert np.abs(values["g1"][0, 0]) < 0.1
    np.testing.assert_allclose(values["g1"][center], 1, atol=1e-12)


def test_wigner_vacuum_subtraction():
    cfg = config()
    cfg.noise.enabled = True
    cfg.noise.stochastic = "wigner"
    cfg.noise.stochastic_strength = 0.4
    controller = SimulationController(cfg)
    variance = 0.4 / (2 * np.prod(controller.grid.spacing))
    rng = np.random.default_rng(7)
    fields = [(rng.normal(size=controller.grid.shape) +
               1j * rng.normal(size=controller.grid.shape)) * np.sqrt(variance / 2)
              for _ in range(4000)]
    values = sampled(controller, fields)
    assert abs(np.mean(values["density_mean"])) < 0.02
    assert abs(np.mean(values["momentum_distribution"])) < 0.02
    assert all(result.wigner_corrected for accumulator in controller._accumulators.values()
               for result in accumulator.results())


def test_batched_matches_single_updates_bitwise():
    cfg = config()
    fields = [np.full((16, 16), amplitude, dtype=np.complex128) for amplitude in (1.0, 2.0)]
    batch = SimulationController(cfg, members=[{"noise.seed": 1}, {"noise.seed": 2}])
    combined = sampled(batch, [np.stack(fields)] * 2)
    for index, field in enumerate(fields):
        single = sampled(SimulationController(config()), [field] * 2)
        for name, value in single.items():
            np.testing.assert_array_equal(combined[name][index], value)


def test_sample_schedule_and_default_chunk_path():
    configured = SimulationController(config())
    chunks = []
    advance = configured.solver.advance

    def tracked(model, state, count):
        chunks.append(count)
        advance(model, state, count)

    configured.solver.advance = tracked
    configured.run()
    assert chunks == [2, 2]
    assert configured._accumulators["g2"].samples == 3
    assert configured._accumulators["g2"].t_first == 0.0
    assert configured._accumulators["g2"].t_last == 0.04
    plain = config()
    plain.result.accumulators = {}
    controller = SimulationController(plain)
    counts = []
    advance = controller.solver.advance

    def plain_tracked(model, state, count):
        counts.append(count)
        advance(model, state, count)

    controller.solver.advance = plain_tracked
    controller.run()
    assert counts == [4]
    np.testing.assert_array_equal(controller.state.psi, configured.state.psi)


def test_start_and_stop_select_only_the_requested_step():
    cfg = config()
    cfg.result.accumulators = {"g2": {"start": 0.02, "stop": 0.02, "every": 0.01}}
    controller = SimulationController(cfg)
    controller.run()
    accumulator = controller._accumulators["g2"]
    assert accumulator.samples == 1
    assert accumulator.t_first == accumulator.t_last == 0.02


def test_checkpoint_resume_is_bitwise_identical(tmp_path):
    full_cfg = config()
    full_cfg.result.save_results = True
    full_cfg.result.formats = ("hdf5",)
    full = SimulationController(full_cfg, output_dir=tmp_path / "full")
    full.run()
    first_cfg = config()
    first_cfg.solver.total_time = 0.02
    first_cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    first = SimulationController(first_cfg, output_dir=tmp_path / "split")
    first.run()
    with h5py.File(first_cfg.checkpoint.path) as handle:
        assert set(handle["accumulators"]) == {"momentum-distribution", "g2"}
        assert handle["accumulators/g2"].attrs["samples"] == 2
    resumed_cfg = config()
    resumed_cfg.result.save_results = True
    resumed_cfg.result.formats = ("hdf5",)
    resumed_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed = SimulationController(resumed_cfg, output_dir=tmp_path / "split")
    resumed.run()
    for name in full.accumulated:
        np.testing.assert_array_equal(resumed.accumulated[name].value, full.accumulated[name].value)
    with h5py.File(tmp_path / "split" / "results.h5") as handle:
        assert handle["accumulated/g2/value"].attrs["dims"][-1] == "x"


@pytest.mark.parametrize("fmt", ["hdf5", "npy", "json"])
def test_storage_round_trip_and_reader_axes(tmp_path, fmt):
    cfg = config()
    cfg.result.save_results = True
    cfg.result.formats = (fmt,)
    controller = SimulationController(cfg, output_dir=tmp_path)
    controller.run()
    with load_results(tmp_path) as results:
        for name, result in controller.accumulated.items():
            np.testing.assert_array_equal(np.asarray(results.accumulated[name]), result.value)
            assert results.accumulated_meta[name]["samples"] == 3
        assert len(results.kx) == cfg.grid.nx
        assert len(results.ky) == cfg.grid.ny
    if fmt == "npy":
        assert (tmp_path / "accumulated_g2.npy").exists()
    if fmt == "json":
        assert "accumulated" in json.loads((tmp_path / "metadata.json").read_text())


def test_sweep_member_and_weighted_mean(tmp_path):
    from polarism.sweep import Sweep

    cfg = config()
    cfg.noise.enabled = True
    cfg.noise.initial_amplitude = 0.1
    cfg.result.save_results = True
    cfg.result.formats = ("hdf5",)
    cfg.result.output_directory = str(tmp_path)
    cfg.sweep.batch_size = 1
    sweep = Sweep(cfg, {"noise.seed": [1, 2]})
    sweep.run()
    with load_results(sweep.run_directory) as results:
        values = [np.asarray(results.member(i).accumulated["density_mean"]) for i in range(2)]
        np.testing.assert_allclose(results.mean("density_mean"), np.mean(values, axis=0))
        np.testing.assert_allclose(results.mean("density_mean", members=[1]), values[1])
        results._results[1].accumulated_meta["density_mean"]["samples"] = 1
        np.testing.assert_allclose(results.mean("density_mean"), (values[0] * 3 + values[1]) / 4)
        results._results[1].accumulated_meta["density_mean"]["samples"] = 3
        np.testing.assert_allclose(g2(results.mean("density_mean"),
                                      results.mean("density_squared_mean")),
                                   g2(np.mean(values, axis=0), results.mean("density_squared_mean")))
        assert g1(results.mean("momentum_distribution")).shape == values[0].shape


@pytest.mark.parametrize("name,params", [
    ("missing", {}), ("g2", {"unknown": 1}), ("g2", {"every": 0.015}),
    ("g2", {"start": 0.015}),
])
def test_validation_rejects_unknown_or_off_grid(name, params):
    cfg = config()
    cfg.result.accumulators = {name: params}
    with pytest.raises(ConfigValidationError, match="accumulator"):
        validate_config(cfg)


@pytest.mark.gpu
@pytest.mark.skipif(not has_cuda_device(), reason="CUDA unavailable")
def test_gpu_parity_and_graph_capture():
    from polarism.config.simulation_parameters import ComputeEngineParameters

    cpu_cfg = config()
    cpu_cfg.solver.method = "ifrk-fft-cuda"
    cpu_cfg.solver.total_time = 0.08
    cpu_cfg.result.accumulators = {"momentum-distribution": {"every": 0.04},
                                   "g2": {"every": 0.04}}
    cpu = SimulationController(cpu_cfg)
    cpu.run()
    gpu_cfg = config()
    gpu_cfg.solver.method = "ifrk-fft-cuda"
    gpu_cfg.solver.total_time = 0.08
    gpu_cfg.result.accumulators = {"momentum-distribution": {"every": 0.04},
                                   "g2": {"every": 0.04}}
    gpu_cfg.compute_engine = ComputeEngineParameters(use_gpu=True)
    gpu = SimulationController(gpu_cfg)
    gpu.run()
    for name in cpu.accumulated:
        np.testing.assert_allclose(gpu.accumulated[name].value,
                                   cpu.accumulated[name].value, rtol=1e-12, atol=1e-12)
    assert gpu.solver._fused._graph_cache
