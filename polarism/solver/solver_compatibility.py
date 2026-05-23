"""Solver compatibility checks."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config

_SPECTRAL_SOLVERS = {"split-step-fft", "ip-rk4", "etd-rk2"}

_ALL_RESERVOIR_TYPES = {"single", "double", "quadratic-double"}
_STAGE_COUPLED_RESERVOIR_TYPES = {"single", "double", "quadratic-double"}
_SPLIT_STEP_RESERVOIR_TYPES = {"single", "double"}


class _SolverCaps(TypedDict):
    """Store capability flags for each solver.

    Fields
    ------
    reservoir_types
        All reservoir models the solver can technically run without crashing.
    reservoir_types_quantitative
        Reservoir models for which the solver is suitable for production
        threshold/amplitude comparisons.  For stage-coupled solvers this equals
        reservoir_types.  For split-step solvers (split-step-fft, etd-rk2)
        this is {"single", "double"}: acceptable O(dt) first-order coupling for
        simple reservoirs, but not stage-coupled — do not use for quantitative
        quadratic-double studies.
    supports_kinetic_relaxation
        True if the solver evaluates kinetic_relaxation_eta*n_active*lap(psi)
        in the time loop.  False solvers silently ignore a non-zero eta,
        producing results incomparable with solvers that implement it.
    """
    grid_types: set[str]
    supports_potential: bool
    boundary_types: set[str]
    reservoir_types: set[str]
    reservoir_types_quantitative: set[str]
    supports_kinetic_relaxation: bool
    description: str


_SOLVER_CAPABILITIES: dict[str, _SolverCaps] = {
    "rk4-fdm": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": True,
        "description": (
            "Reference FDM RK4. Stage-coupled reservoir (psi and reservoir "
            "advance through the same 4 RK4 stages). Works with all grid types."
        ),
    },
    "rk4-fdm-fused": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": True,
        "description": (
            "Optimized FDM RK4 with pre-allocated buffers. "
            "Stage-coupled reservoir; numerically equivalent to rk4-fdm."
        ),
    },
    "rk4-cuda": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": True,
        "description": (
            "GPU-fused CUDA RK4 solver. Stage-coupled reservoir. "
            "FDM production/reference path for periodic and closed-interval grids."
        ),
    },
    "rk4-cuda-v100": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": True,
        "description": (
            "V100-specialised CUDA RK4 solver. "
            "2-D block geometry + __launch_bounds__(256,4). "
            "Numerically identical to rk4-cuda; stage-coupled reservoir."
        ),
    },
    "split-step-fft": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": False,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _SPLIT_STEP_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": False,
        "description": (
            "Spectral split-step solver. Reservoir advanced via a separate "
            "midpoint RK2 step — acceptable O(dt) coupling for single/double, "
            "diagnostic path only for quadratic-double. "
            "Does not evaluate kinetic_relaxation_eta. "
            "closed-interval path uses CPU DCT with host-device copies every step."
        ),
    },
    "ip-rk4": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": True,
        "description": (
            "Interaction-picture RK4. Stage-coupled reservoir. "
            "Evaluates kinetic_relaxation_eta spectrally via _forward/_inverse. "
            "closed-interval path uses SciPy DCT with host-device copies every "
            "step — CPU-bound on GPU; not suitable for GPU production runs. "
            "For GPU-native spectral production use 'ifrk4-fft-cuda'."
        ),
    },
    "etd-rk2": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _SPLIT_STEP_RESERVOIR_TYPES,
        "supports_kinetic_relaxation": False,
        "description": (
            "ETD-RK2 spectral solver. FFT-only, periodic grid only. "
            "Reservoir advanced via a separate midpoint step — not stage-coupled. "
            "Does not evaluate kinetic_relaxation_eta. "
            "Diagnostic path for quadratic-double."
        ),
    },
    "ifrk4-fft-cuda": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": _ALL_RESERVOIR_TYPES,
        "reservoir_types_quantitative": _STAGE_COUPLED_RESERVOIR_TYPES,
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
    potential_type = getattr(cfg.potential, "potential_type", "zero")
    reservoir_type = getattr(cfg.reservoir, "reservoir_type", "single")

    caps = _SOLVER_CAPABILITIES.get(solver)
    if caps is None:
        return

    if grid_type not in caps["grid_types"]:
        raise SolverCompatibilityError(
            f"Solver '{solver}' does not support grid_type='{grid_type}'. "
            f"Supported: {sorted(caps['grid_types'])}. "
            f"Hint: {caps['description']}"
        )

    if reservoir_type not in caps["reservoir_types"]:
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
            f"Operator splitting error will degrade results. "
            f"Consider 'rk4-fdm', 'rk4-fdm-fused', 'rk4-cuda', or 'ifrk4-fft-cuda'.",
            UserWarning,
            stacklevel=2,
        )

    if reservoir_type not in caps["reservoir_types_quantitative"]:
        warnings.warn(
            f"Solver '{solver}' with reservoir_type='{reservoir_type}': "
            f"the reservoir is advanced with a separate sub-step (operator splitting), "
            f"introducing an O(dt) coupling error between psi and the reservoir. "
            f"Suitable for diagnostics; not recommended for quantitative "
            f"threshold or amplitude comparisons. "
            f"For production reservoir dynamics use 'rk4-cuda' (FDM) or "
            f"'ifrk4-fft-cuda' (spectral periodic).",
            UserWarning,
            stacklevel=2,
        )

    if solver in _SPECTRAL_SOLVERS and grid_type == "closed-interval":
        warnings.warn(
            f"Solver '{solver}' uses spectral methods (FFT/DCT) on a "
            f"closed-interval grid. The DCT cannot exactly diagonalize the "
            f"finite-difference Neumann Laplacian, so results may differ "
            f"from FDM-based solvers. For best accuracy on closed-interval "
            f"grids, use 'rk4-fdm', 'rk4-fdm-fused', or 'rk4-cuda'.",
            UserWarning,
            stacklevel=2,
        )

    if solver in {"rk4-cuda", "rk4-cuda-v100", "ifrk4-fft-cuda"} and not _gpu_available():
        warnings.warn(
            f"Solver '{solver}' is optimized for GPU but no GPU is detected. "
            "Falling back to CPU (numpy). For CPU-only runs, 'rk4-fdm-fused' "
            "may be faster.",
            UserWarning,
            stacklevel=2,
        )


def _gpu_available() -> bool:
    """Return whether the active backend exposes CUDA."""
    try:
        from polarism.compute_engine import compute_engine

        return hasattr(compute_engine.xp, "cuda")
    except Exception:
        return False
