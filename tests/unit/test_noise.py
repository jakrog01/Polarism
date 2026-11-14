"""Counter-based noise and integration contracts."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.noise.philox import counter_words, gpu_philox4x32, philox4x32
from polarism.noise.stochastic import NoiseApplier
from polarism.simulation_controller import SimulationController
from polarism.sweep import Sweep
from polarism.sweep_config import override_kind
from tests.unit.conftest import small_config


CPU_SOLVERS = (
    "rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "split-step-fft",
    "ip-rk4", "etd-rk2", "ifrk-fft-cuda", "ifrk4-fft-cuda",
)
GPU_SOLVERS = CPU_SOLVERS + ("split-step-fft-cuda",)


def noise_config(**changes):
    """Build a small run with no pump and an explicit counter seed."""
    cfg = small_config()
    cfg.grid.nx = cfg.grid.ny = 32
    cfg.grid.lx = cfg.grid.ly = 32.0
    cfg.laser.P0 = 0.0
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.noise.seed = 123456
    cfg.noise.initial = "none"
    cfg.noise.stochastic = "wigner"
    cfg.solver.total_time = cfg.solver.dt
    for path, value in changes.items():
        section, name = path.split("__", 1)
        setattr(getattr(cfg, section), name, value)
    return cfg


def run_state(cfg, members=None, member_indices=None):
    """Return copied final condensate and reservoir arrays."""
    controller = SimulationController(cfg, members=members, member_indices=member_indices)
    controller.run()
    return controller, controller.state.psi.copy(), tuple(
        field.copy() for field in controller.reservoir.get_state())


@pytest.mark.parametrize("key,counter,expected", [
    (0, (0, 0, 0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (0xFFFFFFFFFFFFFFFF, (0xFFFFFFFF,) * 4,
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (0x299F31D0A4093822, (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
])
def test_philox_random123_vectors(key, counter, expected):
    assert tuple(philox4x32(np.asarray(counter, dtype=np.uint32), key)) == expected


def test_philox_step_and_stream_change_words():
    cells = np.arange(1024)
    first = philox4x32(counter_words(cells, 0), 42)
    second = philox4x32(counter_words(cells, 1), 42)
    another_stream = philox4x32(counter_words(cells, 0, 2), 42)
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, another_stream)


@pytest.mark.gpu
def test_gpu_philox_matches_cpu(use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    counters = counter_words(np.arange(4096), (1 << 40) + 7)
    expected = philox4x32(counters, 0x123456789ABCDEF0)
    actual = cp.asnumpy(gpu_philox4x32(cp.asarray(counters), 0x123456789ABCDEF0))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ("single", "double"))
def test_gpu_wigner_normal_pair_agrees_with_cpu(precision, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = noise_config(solver__precision=precision)
    grid = SimpleNamespace(shape=(256,), spacing=(1.0,))
    real = np.float32 if precision == "single" else np.float64
    complex_type = np.complex64 if precision == "single" else np.complex128
    compute_engine.xp = np
    compute_engine.use_gpu = False
    cpu = np.zeros((1, 1, 256), dtype=complex_type)
    active = np.zeros((1, 1, 256), dtype=real)
    NoiseApplier(cfg, grid).apply_arrays(cpu, active, 7)
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    gpu = cp.zeros_like(cp.asarray(cpu))
    NoiseApplier(cfg, grid).apply_arrays(gpu, cp.asarray(active), 7)
    tolerance = 2e-6 if precision == "single" else 2e-14
    np.testing.assert_allclose(cp.asnumpy(gpu), cpu, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("strength", (1.0, 2.0))
def test_one_step_wigner_variance(strength):
    cfg = noise_config(noise__stochastic_strength=strength)
    cfg.grid.nx = cfg.grid.ny = 64
    cfg.grid.lx = cfg.grid.ly = 64.0
    _, psi, _ = run_state(cfg)
    expected = strength * cfg.physics.gamma_C * cfg.solver.dt / 2.0
    assert abs(np.mean(abs(psi) ** 2) - expected) / expected < 0.06


def test_stationary_damped_wigner_variance_1d():
    cfg = noise_config(grid__ndim=1, grid__nx=256, grid__lx=256.0,
                       physics__gamma_C=1.0, solver__total_time=10.0)
    _, psi, _ = run_state(cfg)
    expected = 1.0 / (2.0 * cfg.grid.lx / cfg.grid.nx)
    assert abs(np.mean(abs(psi) ** 2) - expected) / expected < 0.22


def test_disabled_and_none_initial_fields():
    disabled = noise_config(noise__enabled=False)
    controller, psi, _ = run_state(disabled)
    assert np.array_equal(psi, np.zeros_like(psi))
    assert not hasattr(controller.solver, "noise")
    initial = noise_config(noise__stochastic="none")
    controller = SimulationController(initial)
    assert np.array_equal(controller.state.psi, np.zeros_like(controller.state.psi))
    zero_strength = noise_config(noise__stochastic_strength=0.0)
    controller, psi, _ = run_state(zero_strength)
    assert np.array_equal(psi, np.zeros_like(psi))
    assert not hasattr(controller.solver, "noise")


def test_seed_reproducibility_and_resolved_config(tmp_path):
    first = run_state(noise_config())[1]
    second = run_state(noise_config())[1]
    different = run_state(noise_config(noise__seed=123457))[1]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    cfg = noise_config(noise__seed=None)
    cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    controller, _, _ = run_state(cfg)
    assert type(controller.cfg.noise.seed) is int
    with h5py.File(cfg.checkpoint.path) as saved:
        assert json.loads(saved.attrs["config_json"])["noise"]["seed"] == controller.cfg.noise.seed


def test_noisy_checkpoint_resume_is_exact(tmp_path):
    full_cfg = noise_config(solver__total_time=0.1)
    full = run_state(full_cfg)[1:]
    first_cfg = noise_config(solver__total_time=0.05)
    first_cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    run_state(first_cfg)
    resume_cfg = noise_config(solver__total_time=0.1)
    resume_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed = run_state(resume_cfg)[1:]
    np.testing.assert_array_equal(resumed[0], full[0])
    for actual, expected in zip(resumed[1], full[1]):
        np.testing.assert_array_equal(actual, expected)


def test_null_seed_checkpoint_resume_uses_saved_seed(tmp_path):
    first_cfg = noise_config(noise__seed=None, solver__total_time=0.05)
    first_cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    first = run_state(first_cfg)[0]
    full_cfg = noise_config(noise__seed=first.cfg.noise.seed,
                            solver__total_time=0.1)
    full = run_state(full_cfg)[1:]
    resumed_cfg = noise_config(noise__seed=None, solver__total_time=0.1)
    resumed_cfg.checkpoint.resume_from = first_cfg.checkpoint.path
    resumed = run_state(resumed_cfg)[1:]
    np.testing.assert_array_equal(resumed[0], full[0])
    for actual, expected in zip(resumed[1], full[1]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("solver", CPU_SOLVERS)
def test_noisy_batch_member_matches_single(solver):
    cfg = noise_config(solver__method=solver, solver__total_time=0.03)
    members = [{"physics.gamma_C": gamma} for gamma in (0.05, 0.08, 0.11)]
    _, batch_psi, batch_res = run_state(cfg, members=members)
    for index, overrides in enumerate(members):
        _, psi, reservoir = run_state(cfg, members=[overrides])
        np.testing.assert_array_equal(batch_psi[index], psi)
        for actual, expected in zip(batch_res, reservoir):
            np.testing.assert_array_equal(actual[index], expected[0])


def test_null_sweep_seeds_are_batch_size_independent(tmp_path):
    cfg = noise_config(noise__seed=None, solver__total_time=0.02)
    cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    cfg.result.output_directory = str(tmp_path / "runs")
    values = {"physics.gamma_C": [0.05, 0.08, 0.11]}
    first = Sweep(cfg, values)
    first.cfg.sweep.batch_size = 1
    separate = first.run()
    second_cfg = deepcopy(cfg)
    second_cfg.noise.seed = first.cfg.noise.seed
    second_cfg.sweep.batch_size = 3
    second = Sweep(second_cfg, values)
    batched = second.run()
    for left, right in zip(separate, batched):
        np.testing.assert_array_equal(left["psi"], right["psi"])
        for a, b in zip(left["reservoir"], right["reservoir"]):
            np.testing.assert_array_equal(a, b)
    manifest = json.loads((first.run_directory / "sweep.json").read_text())
    assert cfg.noise.seed is None
    assert manifest["seed"] == first.cfg.noise.seed
    assert manifest["member_seeds"] == [first.cfg.noise.seed] * 3
    with h5py.File(first.run_directory / "batch_001" / "checkpoint.h5") as saved:
        members = json.loads(saved.attrs["member_configs_json"])
        assert members[0]["noise"]["seed"] == first.cfg.noise.seed


def test_null_seed_draws_fresh_noise_per_run():
    cfg = noise_config(noise__seed=None)
    first, psi_first, _ = run_state(cfg)
    second, psi_second, _ = run_state(cfg)
    assert cfg.noise.seed is None
    assert first.cfg.noise.seed != second.cfg.noise.seed
    assert not np.array_equal(psi_first, psi_second)
    replay = run_state(noise_config(noise__seed=first.cfg.noise.seed))[1]
    np.testing.assert_array_equal(psi_first, replay)


@pytest.mark.parametrize("solver", CPU_SOLVERS)
def test_cpu_solver_accepts_wigner(solver):
    cfg = noise_config(solver__method=solver)
    run_state(cfg)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", GPU_SOLVERS)
def test_gpu_solver_accepts_wigner(solver, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = noise_config(solver__method=solver, compute_engine__use_gpu=True)
    run_state(cfg)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ("rk4-cuda", "split-step-fft-cuda", "ifrk4-fft-cuda"))
def test_gpu_noisy_batch_member_matches_single(solver, use_gpu):
    if not use_gpu or not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    cfg = noise_config(solver__method=solver, compute_engine__use_gpu=True,
                       solver__total_time=0.03)
    members = [{"physics.gamma_C": gamma} for gamma in (0.05, 0.08, 0.11)]
    _, batch_psi, batch_res = run_state(cfg, members=members)
    for index, overrides in enumerate(members):
        _, psi, reservoir = run_state(cfg, members=[overrides])
        np.testing.assert_array_equal(cp.asnumpy(batch_psi[index]), cp.asnumpy(psi))
        for actual, expected in zip(batch_res, reservoir):
            np.testing.assert_array_equal(cp.asnumpy(actual[index]), cp.asnumpy(expected[0]))


@pytest.mark.parametrize("path,value", [
    ("noise__seed", -1), ("noise__seed", 1.5), ("noise__seed", True),
    ("noise__initial", "bad"), ("noise__initial_amplitude", -1.0),
    ("noise__stochastic", "bad"), ("noise__stochastic_strength", -1.0),
])
def test_invalid_noise_config(path, value):
    cfg = noise_config(**{path: value})
    with pytest.raises(ConfigValidationError, match=path.replace("__", ".")):
        validate_config(cfg)


@pytest.mark.parametrize("field,kind", [
    ("enabled", "group"), ("stochastic", "group"), ("seed", "batch"),
    ("initial", "batch"), ("initial_amplitude", "batch"),
    ("initial_k_cutoff_um", "batch"), ("stochastic_strength", "batch"),
])
def test_noise_sweep_path_kind(field, kind):
    assert override_kind("noise." + field) == kind
