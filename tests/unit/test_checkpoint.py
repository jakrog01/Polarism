"""Checkpoint and restart behavior at solver step boundaries."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import signal
import subprocess
import sys

import h5py
import numpy as np
import pytest

from polarism.simulation_controller import SimulationController
from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.results.visitors.result_visitor import ResultVisitor
from tests.unit.conftest import small_config


def configuration(tmp_path, method="rk4-fdm", reservoir="single"):
    """Build a small run with a pulse that spans the restart boundary."""
    cfg = small_config(solver__method=method, reservoir__reservoir_type=reservoir)
    cfg.solver.dt = 0.01
    cfg.solver.total_time = 0.08
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = 0.012
    cfg.laser.pulse_separation = 0.045
    cfg.laser.cutoff_sigma = 2.0
    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.save_psi = True
    cfg.result.save_every = 2 * cfg.solver.dt
    cfg.result.output_directory = str(tmp_path)
    return cfg


def frames(controller):
    """Read the arrays needed to compare two stored trajectories."""
    with h5py.File(controller.run_directory / "results.h5") as h5:
        result = {}

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and (name == "time" or name.startswith(
                    ("fields/", "scalars/", "scalar_groups/"))):
                result[name] = obj[:]

        h5.visititems(collect)
        return result


@pytest.mark.parametrize("method", [
    "rk4-fdm", "rk4-fdm-fused", "split-step-fft", "etd-rk2", "ip-rk4",
    "ifrk-fft-cuda", "ifrk4-fft-cuda",
])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
def test_split_run_is_bit_exact(tmp_path, method, reservoir):
    cfg = configuration(tmp_path / "full", method, reservoir)
    full = SimulationController(cfg)
    full.run()
    expected = frames(full)

    first_cfg = configuration(tmp_path / "split", method, reservoir)
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 3 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    path = first.run_directory / "checkpoint.h5"
    assert path.exists()

    second_cfg = configuration(tmp_path / "split", method, reservoir)
    second_cfg.checkpoint.resume_from = str(path)
    second = SimulationController(second_cfg)
    second.run()
    assert second.run_directory != first.run_directory
    assert second.state.t == full.state.t
    assert np.array_equal(second.state.psi, full.state.psi)
    for actual, wanted in zip(second.reservoir.get_state(), full.reservoir.get_state()):
        assert np.array_equal(actual, wanted)
    before, after = frames(first), frames(second)
    assert before.keys() == after.keys() == expected.keys()
    for name, wanted in expected.items():
        assert np.array_equal(np.concatenate((before[name], after[name])), wanted)


def test_signal_saves_and_returns_normally(tmp_path):
    cfg = configuration(tmp_path / "signal")
    full_cfg = configuration(tmp_path / "full")
    full = SimulationController(full_cfg)
    full.run()

    class Interrupt(ResultVisitor):
        def needs(self, nodes):
            return set()

        def visit_all(self, nodes, **context):
            if context["t"] == 0.04:
                os.kill(os.getpid(), signal.SIGUSR1)

        def finalize(self):
            self.finalized = True

    stopped = SimulationController(cfg)
    visitor = Interrupt()
    stopped.add_visitor(visitor)
    previous = signal.getsignal(signal.SIGUSR1)
    stopped.run()
    assert visitor.finalized
    assert signal.getsignal(signal.SIGUSR1) is previous
    assert stopped.state.t == 0.04
    path = stopped.run_directory / "checkpoint.h5"
    assert path.exists()
    resumed_cfg = configuration(tmp_path / "signal")
    resumed_cfg.checkpoint.resume_from = str(path)
    resumed = SimulationController(resumed_cfg)
    resumed.run()
    assert np.array_equal(resumed.state.psi, full.state.psi)
    for actual, wanted in zip(resumed.reservoir.get_state(), full.reservoir.get_state()):
        assert np.array_equal(actual, wanted)


def test_resume_from_step_zero_does_not_repeat_initial_frame(tmp_path):
    full = SimulationController(configuration(tmp_path / "full"))
    full.run()

    class StopAtInitialFrame(ResultVisitor):
        def needs(self, nodes):
            return set()

        def visit_all(self, nodes, **context):
            if context["t"] == 0.0:
                os.kill(os.getpid(), signal.SIGUSR1)

    first = SimulationController(configuration(tmp_path / "split"))
    first.add_visitor(StopAtInitialFrame())
    first.run()
    assert first.state.t == 0.0
    second_cfg = configuration(tmp_path / "split")
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    before, after, expected = frames(first), frames(second), frames(full)
    for name, wanted in expected.items():
        assert np.array_equal(np.concatenate((before[name], after[name])), wanted)


def test_checkpoint_validation_and_precision_cast(tmp_path):
    cfg = configuration(tmp_path / "source")
    cfg.result.save_results = False
    cfg.checkpoint.every = 1 * cfg.solver.dt
    source = SimulationController(cfg)
    source.run()
    path = source.run_directory / "checkpoint.h5"

    bad_grid = replace(cfg, grid=replace(cfg.grid, nx=8))
    with pytest.raises(ValueError, match="grid shape"):
        SimulationController(bad_grid).load_checkpoint(path)
    bad_reservoir = replace(cfg, reservoir=replace(cfg.reservoir, reservoir_type="double"))
    with pytest.raises(ValueError, match="reservoir type"):
        SimulationController(bad_reservoir).load_checkpoint(path)
    batch = SimulationController(cfg)
    batch.set_initial_state(np.zeros((2, 16, 16), dtype=np.complex128))
    with pytest.raises(ValueError, match="batch shape"):
        batch.load_checkpoint(path)

    single = replace(cfg, solver=replace(cfg.solver, precision="single"))
    with pytest.warns(UserWarning, match="precision"):
        cast = SimulationController(single)
        cast.load_checkpoint(path)
    assert cast.state.psi.dtype == np.complex64
    assert cast.reservoir.get_state()[0].dtype == np.float32


def test_batched_checkpoint_roundtrip(tmp_path):
    cfg = configuration(tmp_path)
    cfg.result.save_results = False
    cfg.checkpoint.every = 2 * cfg.solver.dt
    first = SimulationController(cfg)
    first.set_initial_state(np.ones((2, 16, 16), dtype=np.complex128))
    first.run()
    path = first.run_directory / "checkpoint.h5"
    restored = SimulationController(cfg)
    restored.load_checkpoint(path)
    assert np.array_equal(restored.state.psi, first.state.psi)
    assert np.array_equal(restored.reservoir.get_state()[0], first.reservoir.get_state()[0])


def test_batched_spectral_restart_is_bit_exact(tmp_path):
    cfg = configuration(tmp_path / "full", "ifrk-fft-cuda", "double")
    cfg.result.save_results = False
    initial = np.ones((2, 16, 16), dtype=np.complex128)
    full = SimulationController(cfg)
    full.set_initial_state(initial)
    full.run()

    first_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda", "double")
    first_cfg.result.save_results = False
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 4 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.set_initial_state(initial)
    first.run()
    second_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda", "double")
    second_cfg.result.save_results = False
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    assert np.array_equal(second.state.psi_k, full.state.psi_k)
    for actual, wanted in zip(second.reservoir.get_state(), full.reservoir.get_state()):
        assert np.array_equal(actual, wanted)


def test_checkpoint_options_accept_yaml_and_cli():
    import tyro
    import yaml

    from polarism.config.simulation_parameters import CheckpointParameters, Config

    data = yaml.safe_load("every: 0.07\npath: checkpoint.h5\nresume_from: old.h5\non_signal: false\n")
    options = CheckpointParameters(**data)
    assert options.every == 0.07
    assert options.resume_from == "old.h5"
    assert not options.on_signal
    parsed = tyro.cli(Config, args=["--checkpoint.resume-from", "old.h5",
                                    "--checkpoint.every", "0.07"])
    assert parsed.checkpoint.resume_from == "old.h5"
    assert parsed.checkpoint.every == 0.07


def test_cli_resume_from_file(tmp_path):
    cfg = configuration(tmp_path / "first")
    cfg.result.save_results = False
    cfg.solver.total_time = 0.04
    cfg.checkpoint.every = 4 * cfg.solver.dt
    first = SimulationController(cfg)
    first.run()
    destination = tmp_path / "resumed"
    command = [
        str(Path(sys.executable).with_name("polarism")),
        "--checkpoint.resume-from", str(first.run_directory / "checkpoint.h5"),
        "--solver.total-time", "0.08",
        "--solver.dt", "0.01",
        "--grid.nx", "16",
        "--grid.ny", "16",
        "--grid.lx", "20",
        "--grid.ly", "20",
        "--reservoir.reservoir-type", "single",
        "--result.output-directory", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    paths = list(destination.glob("*/checkpoint.h5"))
    assert len(paths) == 1
    with h5py.File(paths[0]) as h5:
        assert h5.attrs["step"] == 8


@pytest.mark.parametrize("field,value", [
    ("every", -1), ("every", True), ("path", ""),
    ("resume_from", ""), ("on_signal", 1),
])
def test_checkpoint_config_rejects_invalid_values(field, value):
    from polarism.config.validation import ConfigValidationError, validate_config

    cfg = small_config()
    setattr(cfg.checkpoint, field, value)
    with pytest.raises(ConfigValidationError, match=f"checkpoint.{field}"):
        validate_config(cfg)


def test_set_initial_state_continues_from_absolute_time(tmp_path):
    cfg = configuration(tmp_path / "full")
    cfg.result.save_results = False
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "first")
    first_cfg.result.save_results = False
    first_cfg.solver.total_time = 0.04
    first = SimulationController(first_cfg)
    first.run()
    second_cfg = configuration(tmp_path / "second")
    second_cfg.result.save_results = False
    second_cfg.solver.total_time = 0.04
    second = SimulationController(second_cfg)
    second.set_initial_state(first.state.psi,
                             tuple(field[0, 0] for field in first.reservoir.get_state()),
                             first.state.t)
    second.run()
    assert np.array_equal(second.state.psi, full.state.psi)
    assert np.array_equal(second.reservoir.get_state()[0], full.reservoir.get_state()[0])


def test_checkpoint_without_storage_creates_run_directory(tmp_path):
    cfg = configuration(tmp_path)
    cfg.result.save_results = False
    cfg.checkpoint.every = 3 * cfg.solver.dt
    controller = SimulationController(cfg)
    controller.run()
    path = controller.run_directory / "checkpoint.h5"
    assert path.exists()
    assert not list(controller.run_directory.glob(".checkpoint.h5.*.tmp"))
    with h5py.File(path) as h5:
        assert h5.attrs["checkpoint_version"] == 2
        assert h5.attrs["step"] == 8
        assert h5.attrs["t"] == 0.08
        assert h5.attrs["dt"] == 0.01
        assert "config_json" in h5.attrs
        assert "polarism_version" in h5.attrs


@pytest.mark.parametrize("absorption", ["mask", "cap"])
def test_absorption_restart_is_bit_exact(tmp_path, absorption):
    cfg = configuration(tmp_path / "full", "ifrk-fft-cuda", "double")
    cfg.boundary_condition.absorption = absorption
    cfg.result.save_results = False
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda", "double")
    first_cfg.boundary_condition.absorption = absorption
    first_cfg.result.save_results = False
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 4 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    second_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda", "double")
    second_cfg.boundary_condition.absorption = absorption
    second_cfg.result.save_results = False
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    assert np.array_equal(second.state.psi, full.state.psi)
    for actual, wanted in zip(second.reservoir.get_state(), full.reservoir.get_state()):
        assert np.array_equal(actual, wanted)


@pytest.mark.parametrize("method", ["ifrk-fft-cuda", "ifrk4-fft-cuda"])
def test_spectral_state_survives_checkpoint_without_result_materialization(tmp_path, method):
    cfg = configuration(tmp_path / "full", method)
    cfg.result.save_results = False
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "split", method)
    first_cfg.result.save_results = False
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 4 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    assert first.state.psi_k is not None
    second_cfg = configuration(tmp_path / "split", method)
    second_cfg.result.save_results = False
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    assert second.state.psi_k is not None
    second.run()
    assert np.array_equal(second.state.psi_k, full.state.psi_k)
    assert np.array_equal(second.state.psi, full.state.psi)


@pytest.mark.parametrize("tableau", ["dp5", "dop853", "vern7", "vern9"])
def test_ifrk_tableau_restart(tmp_path, tableau):
    cfg = configuration(tmp_path / "full", "ifrk-fft-cuda")
    cfg.solver.tableau = tableau
    cfg.result.save_results = False
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda")
    first_cfg.solver.tableau = tableau
    first_cfg.solver.total_time = 0.04
    first_cfg.result.save_results = False
    first_cfg.checkpoint.every = 4 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    second_cfg = configuration(tmp_path / "split", "ifrk-fft-cuda")
    second_cfg.solver.tableau = tableau
    second_cfg.result.save_results = False
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    assert np.array_equal(second.state.psi_k, full.state.psi_k)


@pytest.mark.parametrize("method", ["rk4-fdm", "split-step-fft", "etd-rk2", "ip-rk4", "ifrk-fft-cuda"])
def test_single_precision_cpu_restart_is_bit_exact(tmp_path, method):
    cfg = configuration(tmp_path / "full", method)
    cfg.solver.precision = "single"
    cfg.result.save_results = False
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "split", method)
    first_cfg.solver.precision = "single"
    first_cfg.result.save_results = False
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 4 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    second_cfg = configuration(tmp_path / "split", method)
    second_cfg.solver.precision = "single"
    second_cfg.result.save_results = False
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    assert np.array_equal(second.state.psi, full.state.psi)
    for actual, wanted in zip(second.reservoir.get_state(), full.reservoir.get_state()):
        assert np.array_equal(actual, wanted)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_gpu_restart_agrees_with_full_run(tmp_path, method, precision, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("requires --use-gpu and a CUDA device")
    cfg = configuration(tmp_path / "full", method)
    cfg.compute_engine.use_gpu = True
    cfg.solver.precision = precision
    full = SimulationController(cfg)
    full.run()

    first_cfg = configuration(tmp_path / "split", method)
    first_cfg.compute_engine.use_gpu = True
    first_cfg.solver.precision = precision
    first_cfg.solver.total_time = 0.04
    first_cfg.checkpoint.every = 3 * first_cfg.solver.dt
    first = SimulationController(first_cfg)
    first.run()
    second_cfg = configuration(tmp_path / "split", method)
    second_cfg.compute_engine.use_gpu = True
    second_cfg.solver.precision = precision
    second_cfg.checkpoint.resume_from = str(first.run_directory / "checkpoint.h5")
    second = SimulationController(second_cfg)
    second.run()
    tolerance = 2e-6 if precision == "single" else 2e-13
    np.testing.assert_allclose(compute_engine.to_cpu(second.state.psi),
                               compute_engine.to_cpu(full.state.psi), rtol=tolerance, atol=tolerance)
    for actual, wanted in zip(second.reservoir.get_state(), full.reservoir.get_state()):
        np.testing.assert_allclose(compute_engine.to_cpu(actual), compute_engine.to_cpu(wanted),
                                   rtol=tolerance, atol=tolerance)
    before, after, expected = frames(first), frames(second), frames(full)
    assert before.keys() == after.keys() == expected.keys()
    for name, wanted in expected.items():
        np.testing.assert_allclose(np.concatenate((before[name], after[name])), wanted,
                                   rtol=tolerance, atol=tolerance)
