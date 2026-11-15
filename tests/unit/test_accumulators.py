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
def test_format_round_trip_and_reader_axes(tmp_path, fmt):
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


def spectrum_config(axis="k", ndim=2):
    cfg = config()
    cfg.grid.ndim = ndim
    cfg.solver.total_time = 0.15
    cfg.result.accumulators = {f"spectrum-{axis}": {"every": 0.01, "segment": 8}}
    return cfg


def test_spectrum_k_plane_wave_peak_and_sum_rule():
    cfg = spectrum_config()
    controller = SimulationController(cfg)
    grid = controller.grid
    accumulator = controller._accumulators["spectrum-k"]
    every = 0.01
    energy_bin = 2 * np.pi * cfg.physics.hbar / (8 * every)
    energy = 2 * energy_bin
    amplitude = 0.7
    spatial = amplitude * np.exp(2j * np.pi * grid.X / grid.lx)
    for index in range(16):
        controller.state.psi = spatial * np.exp(-1j * energy * index * every / cfg.physics.hbar)
        accumulator.update(controller.state, index * every)
    spectrum, energy_axis = accumulator.results()
    value = np.asarray(spectrum.value)
    energies = np.asarray(energy_axis.value)
    assert accumulator.segments == 3
    peak = np.unravel_index(np.argmax(value), value.shape)
    assert abs(energies[peak[0]] - energy) <= energy_bin
    np.testing.assert_allclose(controller.grid.kx[1], 2 * np.pi / grid.lx)
    assert peak[1] == grid.nx // 2 + 1
    expected = amplitude ** 2 * grid.ny ** 2 * grid.nx ** 2 * accumulator.cell / accumulator.cells
    np.testing.assert_allclose(value[:, peak[1]].sum() * energy_bin, expected, rtol=1e-12)
    assert spectrum.linear and not spectrum.wigner_corrected


def test_spectrum_drive_shift():
    cfg = spectrum_config()
    cfg.drive.enabled = True
    cfg.drive.energy = 1.5
    controller = SimulationController(cfg)
    accumulator = controller._accumulators["spectrum-k"]
    every = 0.01
    energy_bin = 2 * np.pi * cfg.physics.hbar / (8 * every)
    spatial = np.exp(2j * np.pi * controller.grid.X / controller.grid.lx)
    for index in range(8):
        controller.state.psi = spatial * np.exp(-1j * energy_bin * index * every / cfg.physics.hbar)
        accumulator.update(controller.state, index * every)
    spectrum, energy_axis = accumulator.results()
    peak = np.unravel_index(np.argmax(spectrum.value), spectrum.value.shape)
    np.testing.assert_allclose(energy_axis.value[peak[0]], energy_bin + cfg.drive.energy)


def test_spectrum_drive_shift_per_member():
    cfg = spectrum_config()
    cfg.drive.enabled = True
    controller = SimulationController(cfg, members=[{"drive.energy": 0.5}, {"drive.energy": 1.5}])
    accumulator = controller._accumulators["spectrum-k"]
    for index in range(8):
        accumulator.update(controller.state, index * 0.01)
    spectrum, energy = accumulator.results()
    assert spectrum.dims == ("member", "E", "kx")
    assert energy.dims == ("member", "E")
    np.testing.assert_allclose(np.asarray(energy.value)[1] - np.asarray(energy.value)[0], 1.0)


@pytest.mark.parametrize("ndim", [1, 2])
def test_spectrum_x_row_and_shape(ndim):
    cfg = spectrum_config("x", ndim)
    if ndim == 2:
        cfg.result.accumulators["spectrum-x"]["y"] = 1.0
    controller = SimulationController(cfg)
    accumulator = controller._accumulators["spectrum-x"]
    if ndim == 2:
        assert accumulator.row == np.argmin(np.abs(controller.grid.Y[:, 0] - 1.0))
    field = np.arange(np.prod(controller.grid.shape)).reshape(controller.grid.shape).astype(complex)
    for index in range(8):
        controller.state.psi = field
        accumulator.update(controller.state, index * 0.01)
    spectrum, energy = accumulator.results()
    assert spectrum.value.shape == (8, cfg.grid.nx)
    assert energy.value.shape == (8,)
    expected = field[accumulator.row] if ndim == 2 else field
    delta = 2 * np.pi * cfg.physics.hbar / 0.08
    np.testing.assert_allclose(np.asarray(spectrum.value).sum(axis=0) * delta,
                               np.abs(expected) ** 2, rtol=1e-12, atol=1e-12)


def test_spectrum_overlap_count():
    controller = SimulationController(spectrum_config())
    accumulator = controller._accumulators["spectrum-k"]
    for index in range(8 + 3 * 4):
        accumulator.update(controller.state, index * 0.01)
    assert accumulator.segments == 4


def test_spectrum_without_completed_segment_warns():
    controller = SimulationController(spectrum_config())
    accumulator = controller._accumulators["spectrum-k"]
    with pytest.warns(UserWarning, match="no completed segments"):
        assert accumulator.results() == []


def test_spectrum_checkpoint_resume_bitwise(tmp_path):
    full_cfg = spectrum_config()
    full_cfg.result.save_results = True
    full_cfg.result.formats = ("hdf5",)
    full = SimulationController(full_cfg, output_dir=tmp_path / "full")
    full.run()
    first_cfg = spectrum_config()
    first_cfg.solver.total_time = 0.10
    first_cfg.checkpoint.path = str(tmp_path / "spectrum_checkpoint.h5")
    first = SimulationController(first_cfg, output_dir=tmp_path / "split")
    first.run()
    with h5py.File(first_cfg.checkpoint.path) as handle:
        assert handle["accumulators/spectrum-k/since_segment"][0] == 3
    resumed_cfg = spectrum_config()
    resumed_cfg.result.save_results = True
    resumed_cfg.result.formats = ("hdf5",)
    resumed_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed = SimulationController(resumed_cfg, output_dir=tmp_path / "split")
    resumed.run()
    for name in full.accumulated:
        np.testing.assert_array_equal(resumed.accumulated[name].value, full.accumulated[name].value)


@pytest.mark.parametrize("fmt", ["hdf5", "npy", "json"])
def test_spectrum_format_and_linear_mean(tmp_path, fmt):
    cfg = spectrum_config()
    cfg.result.save_results = True
    cfg.result.formats = (fmt,)
    controller = SimulationController(cfg, output_dir=tmp_path)
    controller.run()
    with load_results(tmp_path) as results:
        for name, result in controller.accumulated.items():
            np.testing.assert_array_equal(np.asarray(results.accumulated[name]), result.value)
            assert bool(results.accumulated_meta[name]["linear"]) == result.linear
    from polarism.sweep import Sweep

    cfg.result.output_directory = str(tmp_path / "sweep")
    cfg.sweep.batch_size = 1
    cfg.noise.enabled = True
    cfg.noise.initial_amplitude = 0.1
    cfg.result.accumulators["g2"] = {"every": 0.01}
    sweep = Sweep(cfg, {"noise.seed": [1, 2]})
    sweep.run()
    with load_results(sweep.run_directory) as results:
        values = [np.asarray(results.member(i).accumulated["spectrum_k"]) for i in range(2)]
        np.testing.assert_allclose(results.mean("spectrum_k"), np.mean(values, axis=0))
        with pytest.raises(ValueError, match="linear"):
            results.mean("g2")


@pytest.mark.parametrize("axis,ndim,params", [
    ("k", 2, {"segment": 7}),
    ("k", 2, {"segment": 6}),
    ("k", 2, {"segment": 20}),
    ("x", 1, {"y": 0.0}),
])
def test_spectrum_validation(axis, ndim, params):
    cfg = spectrum_config(axis, ndim)
    cfg.result.accumulators[f"spectrum-{axis}"].update(params)
    with pytest.raises(ConfigValidationError, match="accumulator"):
        validate_config(cfg)


@pytest.mark.gpu
@pytest.mark.skipif(not has_cuda_device(), reason="CUDA unavailable")
def test_spectrum_gpu_parity():
    import cupy as cp
    from polarism.config.simulation_parameters import ComputeEngineParameters

    cpu = SimulationController(spectrum_config())
    cpu_acc = cpu._accumulators["spectrum-k"]
    grid = cpu.grid
    fields = [np.exp(2j * np.pi * grid.X / grid.lx - 0.3j * index)
              for index in range(8)]
    for index, field in enumerate(fields):
        cpu.state.psi = field
        cpu_acc.update(cpu.state, index * 0.01)
    cpu_value = cpu_acc.results()[0].value
    gpu_cfg = spectrum_config()
    gpu_cfg.compute_engine = ComputeEngineParameters(use_gpu=True)
    gpu = SimulationController(gpu_cfg)
    gpu_acc = gpu._accumulators["spectrum-k"]
    for index, field in enumerate(fields):
        gpu.state.psi = cp.asarray(field)
        gpu_acc.update(gpu.state, index * 0.01)
    np.testing.assert_allclose(cp.asnumpy(gpu_acc.results()[0].value), cpu_value,
                               rtol=1e-12, atol=1e-12)
