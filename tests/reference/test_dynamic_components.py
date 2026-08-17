from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController
from tests._reporting import write_validation_record
from tests.unit.conftest import small_config


def _linear_configuration(**overrides):
    values = {
        "physics__g_C": 0.0,
        "physics__g_R": 0.0,
        "physics__g_I": 0.0,
        "physics__R": 0.0,
        "physics__R_IA": 0.0,
        "physics__R_AI": 0.0,
        "physics__gamma_C": 0.0,
        "physics__gamma_R": 0.0,
        "physics__gamma_I": 0.0,
        "physics__gamma_A": 0.0,
        "laser__P0": 0.0,
    }
    values.update(overrides)
    return small_config(**values)


def _closed_form_pulse_response(
    cfg: Config,
    controller: SimulationController,
    time: float,
) -> np.ndarray:
    sigma_time = cfg.laser.sigma_time
    cutoff_sigma = cfg.laser.cutoff_sigma
    gamma = cfg.physics.gamma_R
    pulse_center = cutoff_sigma * sigma_time
    pulse_end = 2.0 * pulse_center
    upper = min(max(time, 0.0), pulse_end)
    if upper <= 0.0:
        return np.zeros_like(controller.grid.X)

    radius_squared = (
        (controller.grid.X - cfg.laser.x0) ** 2
        + (controller.grid.Y - cfg.laser.y0) ** 2
    )
    raw_spatial = np.exp(-0.5 * radius_squared / cfg.laser.sigma_space**2)
    spatial_integral = (
        float(np.sum(raw_spatial)) * controller.grid.dx * controller.grid.dy
    )
    spatial_envelope = raw_spatial / spatial_integral
    temporal_integral = (
        math.sqrt(2.0 * math.pi)
        * sigma_time
        * math.erf(cutoff_sigma / math.sqrt(2.0))
    )
    shifted_center = pulse_center + gamma * sigma_time**2
    exponential_factor = math.exp(
        gamma * pulse_center + 0.5 * (gamma * sigma_time) ** 2
    )
    integrated_gaussian = (
        sigma_time
        * math.sqrt(math.pi / 2.0)
        * exponential_factor
        * (
            math.erf(
                (upper - shifted_center) / (math.sqrt(2.0) * sigma_time)
            )
            - math.erf(-shifted_center / (math.sqrt(2.0) * sigma_time))
        )
    )
    return (
        cfg.laser.P0
        * spatial_envelope
        * integrated_gaussian
        * math.exp(-gamma * time)
        / temporal_integral
    )


def _pulse_response_error(dt: float) -> tuple[float, SimulationController]:
    cfg = _linear_configuration(
        grid__nx=16,
        grid__ny=16,
        grid__lx=20.0,
        grid__ly=20.0,
        solver__dt=dt,
        solver__total_time=0.2,
        physics__gamma_R=0.5,
        laser__laser_type="pulse-gaussian",
        laser__power_definition="pulse_energy",
        laser__P0=0.2,
        laser__Pmax=0.2,
        laser__sigma_space=3.0,
        laser__sigma_time=0.02,
        laser__cutoff_sigma=3.0,
        laser__pulse_separation=1.0,
        laser__n_pulses=1,
        reservoir__reservoir_type="single",
    )
    controller = SimulationController(cfg)
    controller.state.psi.fill(0.0)
    squared_error = 0.0
    squared_reference = 0.0
    for step in range(int(cfg.solver.total_time / dt)):
        time = step * dt
        pump = np.asarray(controller._compute_total_pump(time))
        controller.solver.step(
            controller.potential,
            pump,
            controller.reservoir,
            controller.boundary_condition,
            controller.state,
        )
        expected = _closed_form_pulse_response(cfg, controller, time + dt)
        squared_error += float(np.linalg.norm(controller.reservoir.nR - expected) ** 2)
        squared_reference += float(np.linalg.norm(expected) ** 2)
    return math.sqrt(squared_error / squared_reference), controller


@pytest.mark.compliance
def test_pulse_gaussian_drives_linear_reservoir_reference() -> None:
    """Validate pulse-energy encoding against an analytic Gaussian convolution.

    The pump in Polarism drives the incoherent reservoir rather than adding a
    coherent source term to psi. The independent reference evaluates the
    closed-form convolution of the truncated, energy-normalized Gaussian with
    exponential reservoir decay. The measured trajectory errors are 5.28e-3
    at dt=1e-3 and 2.64e-3 at dt=5e-4, consistent with first-order sampling of
    the time-dependent source at the left edge of each solver step.
    """
    compute_engine.xp = np
    coarse_dt = 1e-3
    fine_dt = 5e-4
    coarse_error, _ = _pulse_response_error(coarse_dt)
    measured, controller = _pulse_response_error(fine_dt)
    observed_order = math.log(coarse_error / measured) / math.log(coarse_dt / fine_dt)
    cfg = controller.cfg
    threshold = 3.5e-3
    write_validation_record(
        Path("pulse_gaussian_linear_response.json"),
        error_norm="maximum_transient_relative_l2",
        measured_value=measured,
        threshold=threshold,
        passed=measured < threshold,
        precision="fp64",
        grid={
            "nx": cfg.grid.nx,
            "ny": cfg.grid.ny,
            "lx": cfg.grid.lx,
            "ly": cfg.grid.ly,
            "dx": controller.grid.dx,
            "grid_type": cfg.grid.grid_type,
        },
        dt=fine_dt,
        total_time=cfg.solver.total_time,
        n_steps=int(cfg.solver.total_time / fine_dt),
        solver_reference="closed-form-truncated-gaussian-convolution",
        solver_under_test="rk4-fdm",
        backend_reference="analytic",
        backend_under_test="cpu",
        reservoir_type="single",
        boundary="periodic",
        potential_type="zero",
        extra={
            "laser_type": "pulse-gaussian",
            "power_definition": "pulse_energy",
            "coarse_dt": coarse_dt,
            "coarse_error": coarse_error,
            "observed_order": observed_order,
            "cutoff_sigma": cfg.laser.cutoff_sigma,
        },
        artifact_root=Path("artifacts/reference"),
    )
    assert 0.9 < observed_order < 1.1
    assert measured < threshold


def _reflected_fraction(
    absorption: str,
    width: float,
    lx: float,
    nx: int,
    total_time: float,
    interior_half_width: float | None = None,
) -> float:
    cfg = _linear_configuration(
        grid__nx=nx,
        grid__ny=8,
        grid__lx=lx,
        grid__ly=4.0,
        grid__grid_type="closed-interval",
        solver__method="rk4-fdm",
        solver__dt=0.005,
        solver__total_time=total_time,
        boundary_condition__absorption=absorption,
        boundary_condition__mask_width_percent=width,
        boundary_condition__strength=0.2,
    )
    controller = SimulationController(cfg)
    envelope = np.exp(-0.5 * ((controller.grid.X + 20.0) / 4.0) ** 2)
    initial = (envelope * np.exp(1j * controller.grid.X)).astype(np.complex128)
    initial /= np.linalg.norm(initial.ravel())
    controller.state.psi[:] = initial
    zero_pump = np.zeros_like(controller.grid.X)
    for _ in range(int(total_time / cfg.solver.dt)):
        controller.solver.step(
            controller.potential,
            zero_pump,
            controller.reservoir,
            controller.boundary_condition,
            controller.state,
        )
    density = np.abs(controller.state.psi) ** 2
    if interior_half_width is not None:
        density = np.where(np.abs(controller.grid.X) <= interior_half_width, density, 0.0)
    return float(np.sum(density))


@pytest.mark.compliance
def test_absorber_wave_packet_reflection() -> None:
    """Measure reflected packet power for CAP and mask boundary absorbers.

    The calibration run measured 0.03242 for CAP, below numerical resolution
    for mask, and 0.97555 without absorption at width 0.30. The 0.20 limit
    retains a factor-six margin over the larger absorbed result, while
    monotonic improvement with absorber width is enforced independently.
    """
    compute_engine.xp = np
    compute_engine.use_gpu = False
    widths = (0.10, 0.20, 0.30)
    total_time = 40.0
    wide_reference = _reflected_fraction(
        "no-absorption", 0.0, 160.0, 511, total_time, interior_half_width=40.0
    )
    no_absorption = max(
        0.0,
        _reflected_fraction("no-absorption", 0.0, 80.0, 256, total_time)
        - wide_reference,
    )
    reflections = {
        strategy: [
            max(
                0.0,
                _reflected_fraction(strategy, width, 80.0, 256, total_time)
                - wide_reference,
            )
            for width in widths
        ]
        for strategy in ("cap", "mask")
    }
    threshold = 0.20
    measured = max(values[-1] for values in reflections.values())
    monotonic = all(
        all(later < earlier for earlier, later in zip(values, values[1:]))
        for values in reflections.values()
    )
    passed = measured < threshold and no_absorption > 0.8 and monotonic
    write_validation_record(
        Path("absorber_wave_packet_reflection.json"),
        error_norm="interior_norm_reflection_fraction",
        measured_value=measured,
        threshold=threshold,
        passed=passed,
        precision="fp64",
        grid={
            "nx": 256,
            "ny": 8,
            "lx": 80.0,
            "ly": 4.0,
            "dx": 80.0 / 255.0,
            "grid_type": "closed-interval",
        },
        dt=0.005,
        total_time=total_time,
        n_steps=int(total_time / 0.005),
        solver_reference="double-width-no-absorption",
        solver_under_test="rk4-fdm",
        backend_reference="cpu",
        backend_under_test="cpu",
        reservoir_type="single",
        boundary="closed-interval-with-absorber",
        potential_type="zero",
        extra={
            "mask_width_percent": list(widths),
            "reflection_by_strategy": reflections,
            "no_absorption_reflection": no_absorption,
            "wide_reference_interior_norm": wide_reference,
            "monotonic": monotonic,
        },
        artifact_root=Path("artifacts/reference"),
    )
    assert no_absorption > 0.8
    assert monotonic
    assert measured < threshold
