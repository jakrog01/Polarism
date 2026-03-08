from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config


_SPECTRAL_SOLVERS = {"split-step-fft", "ip-rk4", "etd-rk2"}


class _SolverCaps(TypedDict):
    grid_types: set[str]
    supports_potential: bool
    boundary_types: set[str]
    reservoir_types: set[str]
    description: str


_SOLVER_CAPABILITIES: dict[str, _SolverCaps] = {
    "rk4-fdm": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": "Reference FDM solver. Works with all configurations.",
    },
    "rk4-fdm-fused": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": "Optimized FDM solver with pre-allocated buffers.",
    },
    "rk4-cuda": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": "GPU-fused CUDA RK4 solver. Supports single and double reservoir.",
    },
    "split-step-fft": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": False,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": (
            "Spectral split-step solver. Operator splitting error [K, V] "
            "makes it inaccurate with non-zero external potentials."
        ),
    },
    "ip-rk4": {
        "grid_types": {"periodic", "closed-interval"},
        "supports_potential": False,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": (
            "Interaction Picture RK4. DCT cannot diagonalize the FDM "
            "Neumann Laplacian, so accuracy degrades on closed-interval "
            "grids with non-trivial spatial structure."
        ),
    },
    "etd-rk2": {
        "grid_types": {"periodic"},
        "supports_potential": True,
        "boundary_types": {"no-absorption", "mask", "cap"},
        "reservoir_types": {"single", "double"},
        "description": ("ETD-RK2 spectral solver. FFT-only, requires periodic grid."),
    },
}


class SolverCompatibilityError(RuntimeError):
    """Raised when the solver is fundamentally incompatible with the config."""


def check_solver_compatibility(cfg: Config) -> None:
    solver = cfg.solver.method
    grid_type = getattr(cfg.grid, "grid_type", "periodic")
    potential_type = getattr(cfg.potential, "potential_type", "zero")
    absorption = getattr(cfg.boundary_condition, "absorption", "no-absorption")
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

    if has_potential and not caps["supports_potential"]:
        warnings.warn(
            f"Solver '{solver}' has known accuracy issues with non-zero "
            f"potential (potential_type='{potential_type}'). "
            f"Operator splitting or spectral approximation errors will degrade "
            f"results. Consider using 'rk4-fdm', 'rk4-fdm-fused', or "
            f"'rk4-cuda' for simulations with external potentials.",
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

    if solver == "rk4-cuda" and not _gpu_available():
        warnings.warn(
            "Solver 'rk4-cuda' is optimized for GPU but no GPU is detected. "
            "Falling back to CPU (numpy). For CPU-only runs, 'rk4-fdm-fused' "
            "may be faster.",
            UserWarning,
            stacklevel=2,
        )


def _gpu_available() -> bool:
    try:
        from polarism.compute_engine import compute_engine

        return hasattr(compute_engine.xp, "cuda")
    except Exception:
        return False
