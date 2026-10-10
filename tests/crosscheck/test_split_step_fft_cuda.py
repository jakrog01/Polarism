"""Independent NumPy composition checks for the two-transform CUDA solver."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters, Config
from polarism.grid.create_grid import create_grid
from polarism.laser.pump_field import PumpField
from polarism.simulation_controller import SimulationController
from polarism.solver.split_step_fft_cuda_solver import SplitStepFFTCudaSolver


def _lap(field, dx, dy):
    return ((np.roll(field, -1, 0) - 2 * field + np.roll(field, 1, 0)) / dy**2
            + (np.roll(field, -1, 1) - 2 * field + np.roll(field, 1, 1)) / dx**2)


def _derivatives(model, fields, rho, pump, p, dx, dy):
    a = fields[0]
    if model == "single":
        da = pump - (p.gamma_R + p.R * rho) * a
        return (da + p.reservoir_diffusion_R * _lap(a, dx, dy),)
    b = fields[1]
    if model == "double":
        da = p.R_IA * b - (p.gamma_A + p.R_AI + p.R * rho) * a
        db = pump - (p.gamma_I + p.R_IA) * b + p.R_AI * a
        return (da + p.reservoir_diffusion_A * _lap(a, dx, dy),
                db + p.reservoir_diffusion_I * _lap(b, dx, dy))
    transfer = p.kappa * b * b
    da = transfer - p.gamma_R * a - p.R * a * rho
    db = pump - transfer - p.gamma_I * b
    return (da + p.reservoir_diffusion_R * _lap(a, dx, dy),
            db + p.reservoir_diffusion_I * _lap(b, dx, dy))


def _reservoir_half(model, fields, psi, pump, half_dt, p, dx, dy):
    rho = psi.real**2 + psi.imag**2
    k1 = _derivatives(model, fields, rho, pump, p, dx, dy)
    mid = tuple(x + half_dt * 0.5 * k for x, k in zip(fields, k1))
    k2 = _derivatives(model, mid, rho, pump, p, dx, dy)
    return tuple(np.maximum(x + half_dt * k, 0) for x, k in zip(fields, k2))


def reference_step(psi, fields, potential, pump1, pump2, cfg, grid, mask=None):
    """K/2, RK2 reservoir/2, exact local N, RK2 reservoir/2, K/2."""
    p, dt = cfg.physics, cfg.solver.dt
    k_squared = grid.k_squared.get() if hasattr(grid.k_squared, "get") else np.asarray(grid.k_squared)
    half = np.exp(-1j * p.hbar * k_squared * dt / (4 * p.m_eff))
    psi = np.fft.ifft2(np.fft.fft2(psi) * half)
    fields = _reservoir_half(cfg.reservoir.reservoir_type, fields, psi, pump1,
                             dt / 2, p, grid.dx, grid.dy)
    a = fields[0]
    b = fields[1] if len(fields) > 1 else 0
    rho = np.abs(psi)**2
    gain = (p.R * a - p.gamma_C) / 2 + np.imag(potential) / p.hbar
    x = 2 * gain * dt
    small = np.abs(x) < 1e-6
    phase_time = dt * np.where(small, 1 + x/2 + x*x/6,
                               np.expm1(x) / np.where(small, 1, x))
    phase = ((np.real(potential) + p.g_R * a + p.g_I * b) * dt
             + p.g_C * rho * phase_time) / p.hbar
    psi = psi * np.exp(gain * dt - 1j * phase)
    fields = _reservoir_half(cfg.reservoir.reservoir_type, fields, psi, pump2,
                             dt / 2, p, grid.dx, grid.dy)
    psi = np.fft.ifft2(np.fft.fft2(psi) * half)
    if mask is not None:
        psi *= mask
    return psi, fields


def _controller(model="single", potential="zero", lasers=1, precision="double",
                diffusion=False, absorption="no-absorption", dt=0.02):
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx, cfg.grid.ny = 12, 10
    cfg.grid.lx, cfg.grid.ly = 12.0, 10.0
    cfg.solver.method = "split-step-fft-cuda"
    cfg.solver.precision = precision
    cfg.solver.dt = dt
    cfg.reservoir.reservoir_type = model
    cfg.potential.potential_type = "zero"
    cfg.boundary_condition.absorption = "cap" if potential == "cap" else absorption
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.P0 = 0.4
    cfg.laser.Pmax = 1.2
    cfg.laser.sigma_time = 0.3
    cfg.laser.delay = 0.0
    cfg.physics.init_seed = 7
    cfg.physics.init_eps = 0.2
    if diffusion:
        cfg.physics.reservoir_diffusion_R = 0.03
        cfg.physics.reservoir_diffusion_A = 0.02
        cfg.physics.reservoir_diffusion_I = 0.01
    ctl = SimulationController(cfg)
    if lasers == 2:
        # A second real laser forces PumpField's array-valued path.
        ctl.lasers.append(type(ctl.lasers[0])(cfg.laser, ctl.grid.X, ctl.grid.Y, precision))
        ctl.pump_field = PumpField(ctl.lasers, ctl.grid, precision)
    assert (ctl.pump_field.separable_term() is None) == (lasers == 2)
    return ctl


def _run_crosscheck(ctl, steps=3):
    import cupy as cp

    cfg = ctl.cfg
    psi = cp.asnumpy(ctl.state.psi)
    fields = tuple(cp.asnumpy(x) for x in ctl.reservoir.get_state())
    potential = cp.asnumpy(ctl.potential)
    mask = getattr(ctl.boundary_condition.absorption, "mask", None)
    mask = cp.asnumpy(mask) if mask is not None else None
    for n in range(steps):
        t = n * cfg.solver.dt
        p1 = cp.asnumpy(ctl.pump_field(t + cfg.solver.dt / 4))
        p2 = cp.asnumpy(ctl.pump_field(t + 3 * cfg.solver.dt / 4))
        psi, fields = reference_step(psi, fields, potential, p1, p2,
                                     cfg, ctl.grid, mask)
        ctl.solver.step(ctl.potential, ctl.pump_field, ctl.reservoir,
                        ctl.boundary_condition, ctl.state)
        ctl.state.t += cfg.solver.dt
    tolerance = 1e-5 if cfg.solver.precision == "single" else 1e-10
    actual = cp.asnumpy(ctl.state.psi)
    assert np.linalg.norm(actual - psi) / np.linalg.norm(psi) < tolerance
    for got, expected in zip(ctl.reservoir.get_state(), fields):
        assert np.linalg.norm(cp.asnumpy(got) - expected) / (np.linalg.norm(expected) + 1e-30) < tolerance
    if cfg.solver.precision == "single":
        assert ctl.state.psi.dtype == cp.complex64
        assert ctl.state.psi_k.dtype == cp.complex64
        assert all(x.dtype == cp.float32 for x in ctl.reservoir.get_state())


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("potential", ["zero", "cap"])
@pytest.mark.parametrize("lasers", [1, 2])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_numpy_reference(model, potential, lasers, precision):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    _run_crosscheck(_controller(model, potential, lasers, precision))


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["single", "double", "quadratic-double"])
def test_diffusion(model):
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    _run_crosscheck(_controller(model, diffusion=True))


@pytest.mark.gpu
def test_mask_absorption():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    _run_crosscheck(_controller(absorption="mask"))


@pytest.mark.gpu
def test_custom_reservoir_fallback():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    from polarism.reservoir.single_reservoir import SingleReservoir

    class CustomSingleReservoir(SingleReservoir):
        pass

    ctl = _controller()
    ctl.reservoir = CustomSingleReservoir(ctl.cfg.reservoir, ctl.cfg.physics,
                                           ctl.grid, ctl.cfg.solver.precision)
    _run_crosscheck(ctl)


@pytest.mark.gpu
def test_bitwise_determinism():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp
    results = []
    for _ in range(2):
        ctl = _controller("quadratic-double", "cap", 1)
        for n in range(4):
            ctl.solver.step(ctl.potential, ctl.pump_field, ctl.reservoir,
                            ctl.boundary_condition, ctl.state)
            ctl.state.t += ctl.cfg.solver.dt
        results.append((cp.asnumpy(ctl.state.psi_k),
                        tuple(cp.asnumpy(x) for x in ctl.reservoir.get_state())))
    assert np.array_equal(results[0][0], results[1][0])
    assert all(np.array_equal(a, b) for a, b in zip(results[0][1], results[1][1]))


@pytest.mark.gpu
def test_small_dt_agreement_with_existing_split_step():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp
    from polarism.solver.split_step_fft_solver import SplitStepFFTSolver

    def run(method: str, dt: float) -> np.ndarray:
        cfg = Config()
        cfg.compute_engine.use_gpu = True
        cfg.grid.nx = cfg.grid.ny = 64
        cfg.grid.lx = cfg.grid.ly = 40.0
        cfg.solver.method = "split-step-fft-cuda"
        cfg.solver.dt = dt
        cfg.reservoir.reservoir_type = "quadratic-double"
        cfg.laser.laser_type = "continuous-gaussian"
        cfg.laser.P0 = 0.2
        cfg.laser.sigma_space = 8.0
        ctl = SimulationController(cfg)
        if method == "split-step-fft":
            ctl.solver = SplitStepFFTSolver(ctl.cfg, ctl.grid)
        X, Y = cp.asarray(ctl.grid.X), cp.asarray(ctl.grid.Y)
        ctl.state.psi = (20.0 * cp.exp(-0.5 * (X**2 + Y**2) / 6.0**2)
                         * cp.exp(1j * 1.5 * X)).astype(ctl.state.psi.dtype)
        for _ in range(round(0.8 / dt)):
            ctl.solver.step(ctl.potential, ctl.pump_field, ctl.reservoir,
                            ctl.boundary_condition, ctl.state)
            ctl.state.t += dt
        return cp.asnumpy(ctl.state.psi)

    # Both schemes are second-order Strang splittings with different error
    # constants, so their difference must itself shrink as dt**2.
    differences = [np.linalg.norm(run("split-step-fft-cuda", dt) - run("split-step-fft", dt))
                   for dt in (0.04, 0.02, 0.01)]
    assert differences[-1] > 1e-10
    assert 3.2 < differences[0] / differences[1] < 4.8
    assert 3.2 < differences[1] / differences[2] < 4.8


def test_requires_gpu():
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    cfg = Config()
    cfg.solver.method = "split-step-fft-cuda"
    with pytest.raises(RuntimeError, match="requires CuPy and an active CUDA GPU"):
        SplitStepFFTCudaSolver(cfg, create_grid(cfg.grid))
