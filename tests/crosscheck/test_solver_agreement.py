from __future__ import annotations


import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests._helpers import rel_l2_err
from tests.unit.conftest import small_config

TOL_STAGE_COUPLED = 1e-3
TOL_SPLIT_COUPLED = 5e-2
TOL_DIAGNOSTIC_SPLIT_COUPLED = 2e-1
TOL_IFRK4_INDEPENDENT = 5e-4
MINIMUM_ACTIVE_RESERVOIR_DENSITY = 1e-3
MINIMUM_RESERVOIR_VARIANT_SEPARATION = 1e-3
MINIMUM_ERROR_VARIANT_SEPARATION = 1e-2


def _metrics(solver: str, dt: float, reservoir: str):
    cfg = small_config(
        grid__nx=64,
        grid__ny=64,
        grid__lx=30.0,
        grid__ly=30.0,
        solver__method=solver,
        solver__dt=dt,
        solver__total_time=0.5,
        reservoir__reservoir_type=reservoir,
        laser__P0=0.2,
    )
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.sigma_space = 4.0
    controller = SimulationController(cfg)
    seed = 0.05 * np.exp(
        -(controller.grid.X**2 + controller.grid.Y**2) / (2.0 * 4.0**2)
    ) * np.exp(0.2j * controller.grid.X)
    controller.state.psi[...] = seed.astype(controller.state.psi.dtype)
    controller.run()
    rho = np.abs(np.asarray(controller.state.psi)) ** 2
    return float(rho.sum()), float(rho.max())


@pytest.mark.compliance
@pytest.mark.parametrize("solver", ["rk4-fdm-fused", "ip-rk4", "split-step-fft", "etd-rk2"])
@pytest.mark.parametrize("reservoir", ["single", "double", "quadratic-double"])
def test_solver_end_state_agreement(solver: str, reservoir: str) -> None:
    compute_engine.xp = np
    reference = np.array(_metrics("rk4-fdm", 5e-3, reservoir))
    actual = np.array(_metrics(solver, 2.5e-3 if solver in {"split-step-fft", "etd-rk2"} else 5e-3, reservoir))
    diagnostic = reservoir == "quadratic-double" and solver in {"split-step-fft", "etd-rk2"}
    tolerance = TOL_DIAGNOSTIC_SPLIT_COUPLED if diagnostic else TOL_SPLIT_COUPLED if solver in {"split-step-fft", "etd-rk2"} else TOL_STAGE_COUPLED
    measured = float(
        np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-30))
    )
    assert measured < tolerance


def _independent_solver_result(
    solver: str,
    reservoir: str,
    use_gpu: bool,
) -> tuple[np.ndarray, float]:
    compute_engine.configure(ComputeEngineParameters(use_gpu=use_gpu))
    if use_gpu and not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA device unavailable for ifrk4-fft-cuda independent crosscheck")
    cfg = small_config(
        grid__nx=128,
        grid__ny=128,
        grid__lx=40.0,
        grid__ly=40.0,
        solver__method=solver,
        solver__dt=1e-3,
        solver__total_time=1.0,
        solver__precision="double",
        reservoir__reservoir_type=reservoir,
        laser__P0=5.0,
        laser__sigma_space=8.0,
        potential__potential_type="double-well-supergaussian",
        physics__R=0.1,
        physics__g_R=0.02,
    )
    cfg.compute_engine.use_gpu = use_gpu
    cfg.laser.laser_type = "continuous-gaussian"
    controller = SimulationController(cfg)
    x_coordinates = compute_engine.to_cpu(controller.grid.X)
    initial = (
        0.02
        * (1.0 + 0.1 * np.cos(2.0 * np.pi * x_coordinates / cfg.grid.lx))
        * np.exp(2j * np.pi * x_coordinates / cfg.grid.lx)
    ).astype(np.complex128)
    controller.state.psi[...] = compute_engine.xp.asarray(initial)
    controller.run()
    field = compute_engine.to_cpu(controller.state.psi)
    active_density = compute_engine.to_cpu(
        controller.reservoir.get_reservoir_density()
    )
    return field, float(np.max(active_density))


@pytest.mark.gpu
def test_ifrk4_fft_cuda_matches_rk4_fdm() -> None:
    """Compare IFRK4 CUDA with an independently discretized CPU solver.

    The pump P0=5 is over one thousand times the threshold estimate
    gamma_C*gamma_R/R=0.00415. With R=0.1, g_R=0.02, a spatial Gaussian pump,
    and 1 ps evolution, the active-density maxima reach 4.99 and 0.378 for the
    single and quadratic models. Their fields differ by 5.02e-2. The measured
    cross-solver errors are 6.82e-6 and 5.91e-6, a 13.2 percent distinction
    that explicitly exercises stage-coupled reservoir evolution while staying
    well below the independently established 5e-4 agreement threshold.
    """
    results: dict[str, tuple[np.ndarray, np.ndarray, float, float, float]] = {}
    for reservoir in ("single", "quadratic-double"):
        reference, reference_active = _independent_solver_result(
            "rk4-fdm", reservoir, False
        )
        actual, actual_active = _independent_solver_result(
            "ifrk4-fft-cuda", reservoir, True
        )
        measured = rel_l2_err(actual, reference)
        results[reservoir] = (
            reference,
            actual,
            reference_active,
            actual_active,
            measured,
        )

    cpu_variant_separation = rel_l2_err(
        results["quadratic-double"][0], results["single"][0]
    )
    gpu_variant_separation = rel_l2_err(
        results["quadratic-double"][1], results["single"][1]
    )
    error_variant_separation = abs(
        results["single"][4] - results["quadratic-double"][4]
    ) / results["single"][4]

    for reservoir, result in results.items():
        reference_active = result[2]
        actual_active = result[3]
        measured = result[4]
        assert reference_active > MINIMUM_ACTIVE_RESERVOIR_DENSITY
        assert actual_active > MINIMUM_ACTIVE_RESERVOIR_DENSITY
        assert measured < TOL_IFRK4_INDEPENDENT

    assert cpu_variant_separation > MINIMUM_RESERVOIR_VARIANT_SEPARATION
    assert gpu_variant_separation > MINIMUM_RESERVOIR_VARIANT_SEPARATION
    assert error_variant_separation > MINIMUM_ERROR_VARIANT_SEPARATION
