"""Solver compatibility checks."""
from __future__ import annotations

import warnings

from polarism.compute_engine import has_cuda_device
from polarism.grid.create_grid import grid_boundary
from polarism.solver.abstract_solver import DEFAULT_CAPABILITIES
from polarism.solver.solver_registry import available_solvers

_ALL_RESERVOIR_TYPES = frozenset({"single", "double", "quadratic-double", "none"})


class SolverCompatibilityError(RuntimeError):
    """Raised when the solver is fundamentally incompatible with the config."""


def solver_capabilities(name):
    """Return declared capabilities or a conservative plugin default."""
    component = available_solvers.lookup(name)
    if getattr(component, "capabilities", DEFAULT_CAPABILITIES) is DEFAULT_CAPABILITIES:
        warnings.warn(
            f"Solver '{name}' does not declare capabilities; using conservative defaults.",
            UserWarning, stacklevel=2,
        )
        return DEFAULT_CAPABILITIES
    return component.capabilities


def check_solver_compatibility(cfg) -> None:
    """Check whether the solver matches the config."""
    solver = cfg.solver.method
    caps = solver_capabilities(solver)
    grid_type = getattr(cfg.grid, "grid_type", "periodic")
    boundary = grid_boundary(grid_type)
    reservoir_type = getattr(cfg.reservoir, "reservoir_type", "single")
    if cfg.grid.ndim not in caps.ndim:
        raise SolverCompatibilityError(
            f"solver.method={solver!r} does not support grid.ndim={cfg.grid.ndim}"
        )
    if 1 not in caps.components:
        raise SolverCompatibilityError(f"Solver '{solver}' does not support one component")
    if boundary not in caps.grid_types:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support grid_type='{grid_type}' "
            f"(boundary '{boundary}'). Supported: {sorted(caps.grid_types)}. "
            f"Hint: {caps.description}"
        )
    custom_reservoir = reservoir_type not in _ALL_RESERVOIR_TYPES
    if custom_reservoir and caps.custom_reservoirs == "unsupported":
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support plugin reservoir_type='{reservoir_type}'. "
            f"Supported: {sorted(caps.reservoirs)}. Hint: {caps.description}"
        )
    if not custom_reservoir and reservoir_type not in caps.reservoirs:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support reservoir_type='{reservoir_type}'. "
            f"Supported: {sorted(caps.reservoirs)}. Hint: {caps.description}"
        )
    dispersion = cfg.physics.dispersion
    if "*" not in caps.dispersion and dispersion not in caps.dispersion:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support physics.dispersion='{dispersion}'. "
            f"Supported: {sorted(caps.dispersion)}. Hint: {caps.description}"
        )
    if cfg.terms and not caps.terms:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support extra terms {sorted(cfg.terms)}. "
            f"Hint: {caps.description}"
        )
    if solver in {"rk4-cuda", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda"} and cfg.terms:
        from polarism.model_terms import CudaSnippet, available_terms

        missing = [name for name in cfg.terms
                   if not isinstance(available_terms.lookup(name).cuda, CudaSnippet)]
        if missing:
            raise SolverCompatibilityError(
                f"Solver '{solver}' requires a CUDA snippet for terms {missing}."
            )
    if cfg.drive.enabled and not caps.coherent_drive:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support drive.enabled=True. Hint: {caps.description}"
        )
    if cfg.potential.envelope.get("type") != "constant" and not caps.time_dependent_potential:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support time-dependent potential.envelope. Hint: {caps.description}"
        )
    if cfg.noise.enabled and cfg.noise.stochastic not in caps.noise:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support noise.stochastic='{cfg.noise.stochastic}'"
        )
    eta = cfg.physics.kinetic_relaxation_eta
    if eta != 0.0 and not caps.kinetic_relaxation:
        warnings.warn(
            f"Solver '{solver}' does not evaluate kinetic_relaxation_eta "
            f"(eta={eta}). The energy-relaxation term eta*n_active*lap(psi) "
            f"is silently ignored, making results incomparable with solvers "
            f"that implement it (rk4-fdm, rk4-fdm-fused, rk4-cuda, ip-rk4, "
            f"ifrk4-fft-cuda). Set kinetic_relaxation_eta=0 or switch solver.",
            UserWarning, stacklevel=2,
        )
    if cfg.potential.potential_type != "zero" and not caps.potential:
        warnings.warn(
            f"Solver '{solver}' has known accuracy issues with non-zero potential.",
            UserWarning, stacklevel=2,
        )
    quantitative = (caps.custom_reservoirs == "quantitative" if custom_reservoir
                    else reservoir_type in caps.quantitative_reservoirs)
    if not quantitative and reservoir_type != "none":
        warnings.warn(
            f"Solver '{solver}' with reservoir_type='{reservoir_type}': "
            "the reservoir is advanced sequentially after the full condensate step "
            "(Lie splitting of psi and reservoir), giving a global O(dt) coupling "
            "error whenever the reservoir dynamics are non-trivial. Acceptable for "
            "diagnostic use; not recommended for quantitative threshold or amplitude "
            "comparisons. For production reservoir dynamics use 'rk4-cuda' (FDM), "
            "'ifrk-fft-cuda' (spectral periodic), or 'etd-rk2' (spectral, "
            "second-order coupling via predictor-corrector midpoint reservoir sample).",
            UserWarning, stacklevel=2,
        )
    if solver in {"rk4-cuda", "ifrk4-fft-cuda", "ifrk-fft-cuda"}:
        use_gpu = getattr(cfg.compute_engine, "use_gpu", False)
        cuda_available = has_cuda_device()
        if use_gpu and not cuda_available:
            warnings.warn(
                f"Solver '{solver}' is optimized for GPU but no CUDA device is available "
                "(CuPy missing or `cuda.runtime.getDeviceCount()==0`); falling back to "
                "CPU (NumPy). For CPU-only runs, 'rk4-fdm-fused' may be faster.",
                UserWarning, stacklevel=2,
            )
        elif not use_gpu and cuda_available:
            warnings.warn(
                f"Solver '{solver}' is GPU-optimised and a CUDA device is available, but "
                "`compute_engine.use_gpu=False` in the config; running on CPU by explicit "
                "request.", UserWarning, stacklevel=2,
            )
