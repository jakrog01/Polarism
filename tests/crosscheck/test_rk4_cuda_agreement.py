"""Crosscheck the CUDA RK4 kernels against independent CPU integration."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.simulation_controller import SimulationController
from tests._helpers import rel_l2_err
from tests.unit.conftest import small_config


def _require_cuda() -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    if not compute_engine.use_gpu:
        pytest.skip("CUDA device unavailable")


def _run(
    method: str,
    reservoir_type: str,
    grid_type: str,
    precision: str,
    use_gpu: bool,
    laplacian: str = "five-point",
    steps: int = 50,
    n: int = 32,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    cfg = small_config(
        grid__nx=n,
        grid__ny=n,
        grid__lx=20.0 * n / 32,
        grid__ly=20.0 * n / 32,
        grid__grid_type=grid_type,
        solver__method=method,
        solver__precision=precision,
        solver__laplacian=laplacian,
        solver__dt=0.01,
        solver__total_time=steps * 0.01,
        reservoir__reservoir_type=reservoir_type,
        boundary_condition__absorption="cap" if grid_type == "periodic" else "no-absorption",
        boundary_condition__strength=0.3,
        potential__potential_type="double-well-supergaussian",
        laser__laser_type="pulse-gaussian",
        laser__P0=1.0,
        laser__Pmax=1.0,
        laser__sigma_space=4.0,
        laser__sigma_time=0.08,
        laser__cutoff_sigma=4.0,
        laser__n_pulses=1,
        physics__kinetic_relaxation_eta=0.02,
        physics__reservoir_diffusion_R=0.03,
        physics__reservoir_diffusion_A=0.03,
        physics__reservoir_diffusion_I=0.02,
        compute_engine=ComputeEngineParameters(use_gpu=use_gpu),
    )
    controller = SimulationController(cfg)
    second_laser = replace(cfg.laser, x0=3.0, y0=-2.0, P0=0.7, Pmax=0.7)
    controller.lasers.append(
        PulseGaussian(second_laser, controller.grid.X, controller.grid.Y, precision)
    )
    assert controller.lasers[0]._P_time(0.3) > 0
    if grid_type == "periodic":
        assert np.any(np.imag(compute_engine.to_cpu(controller.potential)) != 0)

    y, x = np.indices((n, n))
    initial_psi = (0.2 + 0.03 * np.cos(x / 4)) * np.exp(1j * (x / 8 + y / 9))
    controller.state.psi[...] = controller.xp.asarray(initial_psi, dtype=controller.state.psi.dtype)
    initial_reservoir = tuple(
        controller.xp.asarray(
            0.15 + 0.04 * index + 0.02 * np.sin(x / 5 + index) * np.cos(y / 6),
            dtype=field.dtype,
        )
        for index, field in enumerate(controller.reservoir.get_state())
    )
    controller.reservoir.set_state(initial_reservoir)
    controller.run()
    return (
        compute_engine.to_cpu(controller.state.psi),
        tuple(compute_engine.to_cpu(field) for field in controller.reservoir.get_state()),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("reservoir_type", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("grid_type", ["periodic", "closed-interval"])
@pytest.mark.parametrize("precision,tolerance", [("double", 1e-10), ("single", 1e-5)])
def test_rk4_cuda_matches_cpu_fdm(
    reservoir_type: str, grid_type: str, precision: str, tolerance: float
) -> None:
    _require_cuda()
    reference = _run("rk4-fdm", reservoir_type, grid_type, precision, False)
    actual = _run("rk4-cuda", reservoir_type, grid_type, precision, True)
    assert rel_l2_err(actual[0], reference[0]) < tolerance
    for gpu_field, cpu_field in zip(actual[1], reference[1]):
        assert rel_l2_err(gpu_field, cpu_field) < tolerance


@pytest.mark.gpu
def test_isotropic_9pt_matches_cpu_fallback() -> None:
    _require_cuda()
    reference = _run("rk4-cuda", "single", "periodic", "double", False, "isotropic-9pt")
    actual = _run("rk4-cuda", "single", "periodic", "double", True, "isotropic-9pt")
    assert rel_l2_err(actual[0], reference[0]) < 1e-10
    assert rel_l2_err(actual[1][0], reference[1][0]) < 1e-10


@pytest.mark.gpu
def test_rk4_cuda_is_bitwise_deterministic() -> None:
    _require_cuda()
    first = _run("rk4-cuda", "double", "periodic", "double", True, steps=200, n=512)
    second = _run("rk4-cuda", "double", "periodic", "double", True, steps=200, n=512)
    assert np.all(np.isfinite(first[0]))
    assert np.array_equal(first[0], second[0])
    for left, right in zip(first[1], second[1]):
        assert np.array_equal(left, right)


@pytest.mark.gpu
@pytest.mark.parametrize("reservoir_type", ["single", "double", "quadratic-double"])
def test_rk4_cuda_does_not_alias_previous_state(reservoir_type: str) -> None:
    _require_cuda()
    cfg = small_config(
        grid__nx=32, grid__ny=32,
        solver__method="rk4-cuda", solver__precision="single",
        reservoir__reservoir_type=reservoir_type,
        laser__laser_type="pulse-gaussian",
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
    controller = SimulationController(cfg)
    controller.solver.step(
        controller.potential, controller._compute_total_pump,
        controller.reservoir, controller.boundary_condition, controller.state,
    )
    before = controller.state.psi
    saved = before.copy()
    reservoir_before = controller.reservoir.get_state()
    reservoir_saved = tuple(field.copy() for field in reservoir_before)
    controller.state.t += cfg.solver.dt
    controller.solver.step(
        controller.potential, controller._compute_total_pump,
        controller.reservoir, controller.boundary_condition, controller.state,
    )
    assert np.array_equal(compute_engine.to_cpu(before), compute_engine.to_cpu(saved))
    for previous, copy in zip(reservoir_before, reservoir_saved):
        assert np.array_equal(compute_engine.to_cpu(previous), compute_engine.to_cpu(copy))
    assert controller.state.psi.shape == (32, 32)
    assert controller.state.psi.dtype == np.complex64
    assert all(field.dtype == np.float32 for field in controller.reservoir.get_state())
