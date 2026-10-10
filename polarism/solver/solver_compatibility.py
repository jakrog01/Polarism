"""Solver compatibility checks."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, TypedDict

from polarism.compute_engine import has_cuda_device
from polarism.grid.create_grid import grid_boundary

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config

_ALL_RESERVOIR_TYPES = {"single", "double", "quadratic-double"}
_STAGE_COUPLED_RESERVOIR_TYPES = {"single", "double", "quadratic-double"}


class _SolverCaps(TypedDict):
    """Store capability flags for each solver.

    Fields
    ------
    grid_types
        Grid boundary semantics the solver supports (``grid.boundary``).
    reservoir_types
        All reservoir models the solver can technically run without crashing.
    reservoir_types_quantitative
        Reservoir models for which the solver is suitable for production
        threshold/amplitude comparisons. Stage-coupled and symmetric-split
        solvers support all built-in reservoir types quantitatively.
    custom_reservoirs
        How reservoirs registered by plugins are handled: ``"quantitative"``
        for solvers that only use the AbstractReservoir interface,
        ``"diagnostic"`` for Lie-split solvers, ``"unsupported"``
        for solvers with hard-coded reservoir kernels.
    supports_kinetic_relaxation
        True if the solver evaluates kinetic_relaxation_eta*n_active*lap(psi)
        in the time loop.  False solvers silently ignore a non-zero eta,
        producing results incomparable with solvers that implement it.
    """
    grid_types: set[str]
    supports_potential: bool
    reservoir_types: set[str]
    reservoir_types_quantitative: set[str]
    custom_reservoirs: Literal["quantitative", "diagnostic", "unsupported"]
    supports_kinetic_relaxation: bool
    description: str


_SOLVER_CAPABILITIES: dict[str, _SolverCaps] = {
    "rk4-fdm": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": True,
        "description": (
            "Reference FDM RK4. Stage-coupled reservoir (psi and reservoir "
            "advance through the same 4 RK4 stages). Works with all grid types."
        ),
    },
    "rk4-fdm-fused": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": True,
        "description": (
            "Optimized FDM RK4 with pre-allocated buffers. "
            "Stage-coupled reservoir; numerically equivalent to rk4-fdm."
        ),
    },
    "rk4-cuda": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "unsupported",
        "supports_kinetic_relaxation": True,
        "description": (
            "GPU-fused CUDA RK4 solver. Stage-coupled reservoir. "
            "FDM production/reference path for periodic and closed-interval grids."
        ),
    },
    "split-step-fft": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _ALL_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": False,
        "description": (
            "Spectral Strang splitting, second order in time. Reservoir advances "
            "in two half-steps around the condensate step. FFT-only, periodic "
            "grid only. Does not evaluate kinetic_relaxation_eta."
        ),
    },
    "split-step-fft-cuda": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _ALL_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": False,
        "description": (
            "GPU-only, two-transform spectral Strang splitting with symmetric "
            "reservoir half-steps. Periodic grids only; no kinetic relaxation."
        ),
    },
    "ip-rk4": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": True,
        "description": (
            "Interaction-picture RK4. Stage-coupled reservoir. FFT-only, "
            "periodic grid only. Evaluates kinetic_relaxation_eta spectrally "
            "via _forward/_inverse. For GPU-native spectral production use "
            "'ifrk4-fft-cuda'."
        ),
    },
    "etd-rk2": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": False,
        "description": (
            "ETD-RK2 spectral solver. FFT-only, periodic grid only. Reservoir "
            "advanced between predictor and corrector using a midpoint condensate "
            "estimate (psi_n + a)/2, giving O(dt²) psi↔reservoir coupling (measured "
            "p≈2.00 on quadratic-double, per convergence study). Does not evaluate "
            "kinetic_relaxation_eta."
        ),
    },
    "ifrk4-fft-cuda": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "custom_reservoirs": "quantitative",
        "supports_kinetic_relaxation": True,
        "description": (
            "GPU-native FFT interaction-picture RK4 solver. Stage-coupled reservoir. "
            "cuFFT kinetics, no CPU transfers in the time loop. "
            "Evaluates kinetic_relaxation_eta in nonlinear RHS. "
            "Spectral production path for periodic grids; "
            "use a wide CAP absorber for open boundaries."
        ),
    },
}


class SolverCompatibilityError(RuntimeError):
    """Raised when the solver is fundamentally incompatible with the config."""


def check_solver_compatibility(cfg: Config) -> None:
    """Check whether the solver matches the config."""
    solver = cfg.solver.method
    grid_type = getattr(cfg.grid, "grid_type", "periodic")
    boundary = grid_boundary(grid_type)
    potential_type = getattr(cfg.potential, "potential_type", "zero")
    reservoir_type = getattr(cfg.reservoir, "reservoir_type", "single")

    caps = _SOLVER_CAPABILITIES.get(solver)
    if caps is None:
        return

    if boundary not in caps["grid_types"]:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support grid_type='{grid_type}' "
            f"(boundary '{boundary}'). "
            f"Supported: {sorted(caps['grid_types'])}. "
            f"Hint: {caps['description']}"
        )

    custom_reservoir = reservoir_type not in _ALL_RESERVOIR_TYPES
    if custom_reservoir and caps["custom_reservoirs"] == "unsupported":
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support plugin reservoir_type='{reservoir_type}'. "
            f"Supported: {sorted(caps['reservoir_types'])}. "
            f"Hint: {caps['description']}"
        )

    if not custom_reservoir and reservoir_type not in caps["reservoir_types"]:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support reservoir_type='{reservoir_type}'. "
            f"Supported: {sorted(caps['reservoir_types'])}. "
            f"Hint: {caps['description']}"
        )

    has_potential = potential_type != "zero"
    eta = getattr(getattr(cfg, "physics", None), "kinetic_relaxation_eta", 0.0)

    if eta != 0.0 and not caps["supports_kinetic_relaxation"]:
        warnings.warn(
            f"Solver '{solver}' does not evaluate kinetic_relaxation_eta "
            f"(eta={eta}). The energy-relaxation term eta*n_active*lap(psi) "
            f"is silently ignored, making results incomparable with solvers "
            f"that implement it (rk4-fdm, rk4-fdm-fused, rk4-cuda, ip-rk4, "
            f"ifrk4-fft-cuda). Set kinetic_relaxation_eta=0 or switch solver.",
            UserWarning,
            stacklevel=2,
        )

    if has_potential and not caps["supports_potential"]:
        warnings.warn(
            f"Solver '{solver}' has known accuracy issues with non-zero "
            f"potential (potential_type='{potential_type}'). "
            f"Split-step error will degrade results. "
            f"Consider 'rk4-fdm', 'rk4-fdm-fused', 'rk4-cuda', or 'ifrk4-fft-cuda'.",
            UserWarning,
            stacklevel=2,
        )

    if custom_reservoir:
        quantitative = caps["custom_reservoirs"] == "quantitative"
    else:
        quantitative = reservoir_type in caps["reservoir_types_quantitative"]
    if not quantitative:
        warnings.warn(
            f"Solver '{solver}' with reservoir_type='{reservoir_type}': "
            f"the reservoir is advanced sequentially after the full condensate step "
            f"(Lie splitting of psi and reservoir), giving a global O(dt) coupling "
            f"error whenever the reservoir dynamics are non-trivial. Acceptable for "
            f"diagnostic use; not recommended for quantitative threshold or amplitude "
            f"comparisons. For production reservoir dynamics use 'rk4-cuda' (FDM), "
            f"'ifrk4-fft-cuda' (spectral periodic), or 'etd-rk2' (spectral, "
            f"second-order coupling via predictor-corrector midpoint reservoir sample).",
            UserWarning,
            stacklevel=2,
        )

    if solver in {"rk4-cuda", "ifrk4-fft-cuda"}:
        use_gpu = getattr(cfg.compute_engine, "use_gpu", False)
        cuda_available = has_cuda_device()
        if use_gpu and not cuda_available:
            warnings.warn(
                f"Solver '{solver}' is optimized for GPU but no CUDA device is available "
                "(CuPy missing or `cuda.runtime.getDeviceCount()==0`); falling back to "
                "CPU (NumPy). For CPU-only runs, 'rk4-fdm-fused' may be faster.",
                UserWarning,
                stacklevel=2,
            )
        elif not use_gpu and cuda_available:
            warnings.warn(
                f"Solver '{solver}' is GPU-optimised and a CUDA device is available, but "
                "`compute_engine.use_gpu=False` in the config; running on CPU by explicit "
                "request.",
                UserWarning,
                stacklevel=2,
            )
