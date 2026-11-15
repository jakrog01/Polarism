"""Embedded IFRK trial control, scheduling, and restart behavior."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from polarism.compute_engine import has_cuda_device
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def configuration(tableau="dp5", rtol=1e-6, dt=1.0, total_time=2.0):
    """Construct a uniform nonlinear decay case."""
    cfg = small_config(
        grid__nx=4, grid__ny=4, solver__method="ifrk-fft-cuda",
        solver__tableau=tableau, solver__adaptive=True, solver__dt=dt,
        solver__total_time=total_time, solver__rtol=rtol,
        solver__atol=rtol * 0.01, reservoir__reservoir_type="none",
        noise__enabled=False, laser__P0=0.0, physics__g_C=2.0,
        physics__gamma_C=0.7,
    )
    return cfg


def uniform_run(cfg):
    controller = SimulationController(cfg)
    controller.state.psi = np.ones(cfg.grid.nx * cfg.grid.ny, dtype=complex).reshape(cfg.grid.ny, cfg.grid.nx)
    controller.run()
    return controller


def exact_uniform(cfg):
    t = cfg.solver.total_time
    gamma = cfg.physics.gamma_C
    phase = -cfg.physics.g_C / (cfg.physics.hbar * gamma) * (1 - np.exp(-gamma * t))
    return np.exp(-gamma * t / 2 + 1j * phase)


@pytest.mark.parametrize("change,pattern", [
    ({"solver.tableau": "rk4"}, "embedded"),
    ({"solver.method": "ifrk4-fft-cuda"}, "adaptive"),
    ({"solver.method": "rk4-fdm"}, "adaptive"),
    ({"noise.enabled": True, "noise.stochastic": "wigner"}, "stochastic"),
    ({"boundary_condition.absorption": "mask"}, "absorption"),
    ({"solver.precision": "single"}, "rtol"),
])
def test_validation(change, pattern):
    cfg = configuration()
    for path, value in change.items():
        section, key = path.split(".")
        setattr(getattr(cfg, section), key, value)
    with pytest.raises(ConfigValidationError, match=pattern):
        validate_config(cfg)


@pytest.mark.parametrize("key,value", [("rtol", 0), ("atol", -1), ("dt_max", 0)])
def test_tolerance_validation(key, value):
    cfg = configuration()
    setattr(cfg.solver, key, value)
    with pytest.raises(ConfigValidationError, match=key):
        validate_config(cfg)


def test_dt_max_caps_first_trial():
    cfg = configuration(dt=0.5, total_time=0.5)
    cfg.solver.dt_max = 0.05
    controller = uniform_run(cfg)
    assert controller.step_statistics["dt_max"] <= 0.05


@pytest.mark.parametrize("tableau", ["dp5", "dop853", "vern7", "vern9"])
def test_uniform_analytic_and_tolerance(tableau):
    loose = uniform_run(configuration(tableau, 1e-6))
    tight = uniform_run(configuration(tableau, 1e-9))
    for controller in (loose, tight):
        error = np.max(np.abs(controller.state.psi - exact_uniform(controller.cfg)))
        assert error <= 50 * controller.cfg.solver.rtol
    assert loose.step_statistics["accepted"] < tight.step_statistics["accepted"]


def test_rejection_and_tick_schedule():
    cfg = configuration(dt=5.0, total_time=5.0)
    controller = uniform_run(cfg)
    assert controller.step_statistics["rejected"] > 0
    assert np.max(np.abs(controller.state.psi - exact_uniform(cfg))) <= 50 * cfg.solver.rtol
    assert controller.state.step == 1
    assert controller.state.t == controller.state.t0 + cfg.solver.dt


def test_rejected_trial_does_not_commit():
    cfg = configuration(dt=5.0, total_time=5.0)
    controller = SimulationController(cfg)
    controller.state.psi = np.ones((4, 4), dtype=complex)
    before = controller.state.psi.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        report = controller.solver.step(controller.model, controller.state, 5.0)
    assert not report.accepted
    assert controller.state.t == 0
    assert controller.state.step == 0
    np.testing.assert_array_equal(controller.state.psi, before)


def test_zero_field_with_zero_absolute_tolerance():
    cfg = configuration(dt=0.1, total_time=0.2)
    cfg.solver.atol = 0.0
    controller = SimulationController(cfg)
    controller.run()
    assert controller.step_statistics["rejected"] == 0
    np.testing.assert_array_equal(controller.state.psi, np.zeros((4, 4)))


def test_checkpoint_resume_bitwise_and_mode_mismatch(tmp_path):
    cfg = configuration(dt=0.25, total_time=2.0)
    cfg.checkpoint.every = 1.0
    cfg.checkpoint.path = str(tmp_path / "full.h5")
    full = uniform_run(cfg)
    first_cfg = configuration(dt=0.25, total_time=1.0)
    first_cfg.checkpoint.every = 1.0
    first_cfg.checkpoint.path = str(tmp_path / "split.h5")
    uniform_run(first_cfg)
    resumed_cfg = configuration(dt=0.25, total_time=2.0)
    resumed_cfg.checkpoint.every = 1.0
    resumed_cfg.checkpoint.resume_from = str(tmp_path / "split.h5")
    resumed = SimulationController(resumed_cfg)
    resumed.run()
    np.testing.assert_array_equal(resumed.state.psi, full.state.psi)
    assert resumed.step_statistics == full.step_statistics
    with h5py.File(tmp_path / "split.h5") as h5:
        assert "step_controller" in h5
    fixed_cfg = configuration(dt=0.25, total_time=2.0)
    fixed_cfg.solver.adaptive = False
    fixed_cfg.checkpoint.resume_from = str(tmp_path / "split.h5")
    with pytest.raises(ValueError, match="adaptive mode"):
        SimulationController(fixed_cfg)
    fixed_source = configuration(dt=0.25, total_time=1.0)
    fixed_source.solver.adaptive = False
    fixed_source.checkpoint.path = str(tmp_path / "fixed.h5")
    fixed = SimulationController(fixed_source)
    fixed.save_checkpoint(fixed_source.checkpoint.path)
    adaptive_target = configuration(dt=0.25, total_time=2.0)
    adaptive_target.checkpoint.resume_from = fixed_source.checkpoint.path
    with pytest.raises(ValueError, match="adaptive mode"):
        SimulationController(adaptive_target)


def test_accumulator_samples_and_result_metadata(tmp_path):
    cfg = configuration(dt=0.1, total_time=0.4)
    cfg.result.accumulators = {"momentum-distribution": {"every": 0.2}}
    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.save_every = 0.1
    cfg.result.output_directory = str(tmp_path)
    controller = uniform_run(cfg)
    assert controller._accumulators["momentum-distribution"].samples == 3
    with h5py.File(controller.run_directory / "results.h5") as h5:
        for name, value in controller.step_statistics.items():
            assert h5.attrs[f"adaptive_{name}"] == value
        expected = np.array([controller.state.t0 + step * cfg.solver.dt for step in range(5)])
        np.testing.assert_array_equal(h5["time"][:], expected)


def test_batched_common_step_size():
    cfg = configuration(dt=1.0)
    batch = SimulationController(cfg, members=[{"physics.g_C": 1.0}, {"physics.g_C": 3.0}])
    batch.state.psi = np.ones((2, 4, 4), dtype=complex)
    batch.run()
    for index, g_c in enumerate((1.0, 3.0)):
        member_cfg = configuration(dt=1.0)
        member_cfg.physics.g_C = g_c
        single = uniform_run(member_cfg)
        expected = exact_uniform(member_cfg)
        assert np.max(np.abs(batch.state.psi[index] - expected)) <= 50 * cfg.solver.rtol
        if index == 1:
            assert batch.step_statistics["accepted"] >= single.step_statistics["accepted"]


def test_spinor_drive_and_dynamic_potential_matches_reference():
    cfg = configuration("dop853", dt=0.05, total_time=0.1)
    cfg.physics.spin = "spinor"
    cfg.physics.te_tm = 0.12
    cfg.physics.zeeman = 0.19
    cfg.drive.enabled = True
    cfg.drive.amplitude = 0.2
    cfg.drive.energy = 0.3
    cfg.potential.potential_type = "profile"
    cfg.potential.profile = {"type": "uniform", "amplitude": 0.4}
    cfg.potential.envelope = {"type": "gaussian-pulses", "center": 0.05, "sigma": 0.02}
    adaptive = SimulationController(cfg)
    adaptive.state.psi = np.stack((np.ones((4, 4)), 0.7 * np.ones((4, 4)))).astype(complex)
    adaptive.run()
    reference_cfg = configuration("dop853", dt=0.0005, total_time=0.1)
    reference_cfg.physics = cfg.physics
    reference_cfg.drive = cfg.drive
    reference_cfg.potential = cfg.potential
    reference_cfg.solver.adaptive = False
    reference = SimulationController(reference_cfg)
    reference.state.psi = np.stack((np.ones((4, 4)), 0.7 * np.ones((4, 4)))).astype(complex)
    reference.run()
    np.testing.assert_allclose(adaptive.state.psi, reference.state.psi,
                               rtol=10 * cfg.solver.rtol, atol=10 * cfg.solver.rtol)


@pytest.mark.gpu
def test_gpu_matches_cpu():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cpu_cfg = configuration()
    cpu_psi = np.array(uniform_run(cpu_cfg).state.psi)
    gpu_cfg = configuration()
    gpu_cfg.compute_engine.use_gpu = True
    gpu = SimulationController(gpu_cfg)
    gpu.state.psi = cp.ones((4, 4), dtype=cp.complex128)
    gpu.run()
    np.testing.assert_allclose(cp.asnumpy(gpu.state.psi), cpu_psi,
                               rtol=100 * gpu_cfg.solver.rtol,
                               atol=100 * gpu_cfg.solver.rtol)
    fixed_cpu_cfg = configuration()
    fixed_cpu_cfg.solver.adaptive = False
    fixed_cpu_psi = np.array(uniform_run(fixed_cpu_cfg).state.psi)
    fixed_gpu_cfg = configuration()
    fixed_gpu_cfg.solver.adaptive = False
    fixed_gpu_cfg.compute_engine.use_gpu = True
    fixed_gpu = SimulationController(fixed_gpu_cfg)
    fixed_gpu.state.psi = cp.ones((4, 4), dtype=cp.complex128)
    fixed_gpu.run()
    np.testing.assert_allclose(cp.asnumpy(fixed_gpu.state.psi), fixed_cpu_psi,
                               rtol=1e-12, atol=1e-12)
