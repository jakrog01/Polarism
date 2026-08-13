from __future__ import annotations

import pytest
import warnings

from polarism.config.validation import ConfigValidationError, validate_config
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from tests.unit.conftest import small_config


@pytest.mark.parametrize("path,value", [("solver__dt", 0.0), ("solver__total_time", 0.0), ("solver__precision", "quad"), ("physics__gamma_C", -1.0), ("physics__init_mode", "foobar"), ("boundary_condition__mask_width_percent", 0.6), ("laser__power_definition", "foo")])
def test_validate_config_rejects_invalid_values(path: str, value) -> None:
    cfg = small_config(**{path: value})
    with pytest.raises(ConfigValidationError, match=path.replace("__", ".").split(".")[-1]):
        validate_config(cfg)


def test_filtered_mode_requires_cutoff() -> None:
    cfg = small_config(physics__init_mode="filtered_complex_gaussian", physics__init_k_cutoff_um=None)
    with pytest.raises(ConfigValidationError, match="init_k_cutoff_um"):
        validate_config(cfg)


def test_solver_compatibility_errors_and_warnings() -> None:
    cfg = small_config(solver__method="etd-rk2", grid__grid_type="closed-interval")
    with pytest.raises(SolverCompatibilityError, match="grid_type"):
        check_solver_compatibility(cfg)
    cfg.solver.method = "ifrk4-fft-cuda"
    with pytest.raises(SolverCompatibilityError, match="grid_type"):
        check_solver_compatibility(cfg)
    cfg.solver.method = "split-step-fft"
    with pytest.warns(UserWarning, match="spectral"):
        check_solver_compatibility(cfg)
    cfg.grid.grid_type = "periodic"
    cfg.potential.potential_type = "double-well-supergaussian"
    with pytest.warns(UserWarning, match="split-step-fft"):
        check_solver_compatibility(cfg)
    cfg.potential.potential_type = "zero"
    cfg.physics.kinetic_relaxation_eta = 0.1
    with pytest.warns(UserWarning, match="kinetic_relaxation_eta"):
        check_solver_compatibility(cfg)
    cfg.physics.kinetic_relaxation_eta = 0.0
    cfg.reservoir.reservoir_type = "quadratic-double"
    with pytest.warns(UserWarning, match="Lie splitting"):
        check_solver_compatibility(cfg)


def test_split_step_fft_quadratic_double_warns_lie_splitting() -> None:
    cfg = small_config(
        solver__method="split-step-fft",
        grid__grid_type="periodic",
        reservoir__reservoir_type="quadratic-double",
    )
    with pytest.warns(UserWarning, match="Lie splitting"):
        check_solver_compatibility(cfg)


def test_etd_rk2_quadratic_double_is_silent() -> None:
    cfg = small_config(
        solver__method="etd-rk2",
        grid__grid_type="periodic",
        reservoir__reservoir_type="quadratic-double",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_solver_compatibility(cfg)
    assert not [warning for warning in caught if "coupling" in str(warning.message).lower()]


@pytest.mark.parametrize(
    ("use_gpu", "cuda_available", "warning_match"),
    [
        (True, False, "no CUDA device is available"),
        (False, True, "compute_engine.use_gpu=False"),
    ],
)
def test_gpu_solver_compatibility_warning_causes(
    monkeypatch: pytest.MonkeyPatch,
    use_gpu: bool,
    cuda_available: bool,
    warning_match: str,
) -> None:
    cfg = small_config(solver__method="ifrk4-fft-cuda")
    cfg.compute_engine.use_gpu = use_gpu
    monkeypatch.setattr(
        "polarism.solver.solver_compatibility.has_cuda_device", lambda: cuda_available
    )
    with pytest.warns(UserWarning, match=warning_match):
        check_solver_compatibility(cfg)


@pytest.mark.parametrize("use_gpu,cuda_available", [(True, True), (False, False)])
def test_gpu_solver_compatibility_silent_when_configuration_matches_device(
    monkeypatch: pytest.MonkeyPatch, use_gpu: bool, cuda_available: bool
) -> None:
    """Keep CPU benchmark halves silent when a GPU host is intentionally unused.

    This prevents `compute_engine.use_gpu=False` benchmark configurations from cluttering logs.
    """
    cfg = small_config(solver__method="ifrk4-fft-cuda")
    cfg.compute_engine.use_gpu = use_gpu
    monkeypatch.setattr(
        "polarism.solver.solver_compatibility.has_cuda_device", lambda: cuda_available
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_solver_compatibility(cfg)
    assert not caught


def test_iprk4_supports_kinetic_relaxation() -> None:
    cfg = small_config(solver__method="ip-rk4", physics__kinetic_relaxation_eta=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_solver_compatibility(cfg)
    assert not caught
