"""Config validation and stability heuristics."""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config

_EXPLICIT_FDM_SOLVERS = {"rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "rk4-cuda-v100"}
_VALID_PRECISIONS = {"single", "double"}


class ConfigValidationError(ValueError):
    """Raised when a required config constraint is violated."""


def validate_config(cfg: Config) -> None:
    """Raise ConfigValidationError for invalid values; warn for instability risk."""
    _validate_grid(cfg)
    _validate_physics(cfg)
    _validate_solver(cfg)
    _validate_result(cfg)
    _warn_explicit_stability(cfg)


def _validate_grid(cfg: Config) -> None:
    g = cfg.grid
    if g.nx < 1:
        raise ConfigValidationError(f"grid.nx must be >= 1, got {g.nx}")
    if g.ny < 1:
        raise ConfigValidationError(f"grid.ny must be >= 1, got {g.ny}")
    if g.lx <= 0.0:
        raise ConfigValidationError(f"grid.lx must be > 0, got {g.lx}")
    if g.ly <= 0.0:
        raise ConfigValidationError(f"grid.ly must be > 0, got {g.ly}")


def _validate_physics(cfg: Config) -> None:
    p = cfg.physics
    if p.hbar <= 0.0:
        raise ConfigValidationError(f"physics.hbar must be > 0, got {p.hbar}")
    if p.m_eff <= 0.0:
        raise ConfigValidationError(f"physics.m_eff must be > 0, got {p.m_eff}")


def _validate_solver(cfg: Config) -> None:
    s = cfg.solver
    if s.dt <= 0.0:
        raise ConfigValidationError(f"solver.dt must be > 0, got {s.dt}")
    if s.total_time <= 0.0:
        raise ConfigValidationError(f"solver.total_time must be > 0, got {s.total_time}")
    if s.precision not in _VALID_PRECISIONS:
        raise ConfigValidationError(
            f"solver.precision must be one of {_VALID_PRECISIONS}, got '{s.precision}'"
        )


def _validate_result(cfg: Config) -> None:
    r = cfg.result
    if r.save_interval < 1:
        raise ConfigValidationError(f"result.save_interval must be >= 1, got {r.save_interval}")
    if r.batch_size < 1:
        raise ConfigValidationError(f"result.batch_size must be >= 1, got {r.batch_size}")


def _warn_explicit_stability(cfg: Config) -> None:
    if cfg.solver.method not in _EXPLICIT_FDM_SOLVERS:
        return

    g = cfg.grid
    if g.grid_type == "periodic":
        dx = g.lx / g.nx
        dy = g.ly / g.ny
    else:
        dx = g.lx / max(g.nx - 1, 1)
        dy = g.ly / max(g.ny - 1, 1)
    dx_min = min(dx, dy)

    p = cfg.physics
    dt_stable = math.sqrt(2.0) * p.m_eff * dx_min ** 2 / (2.0 * p.hbar)

    if cfg.solver.dt > dt_stable:
        warnings.warn(
            f"solver.dt={cfg.solver.dt:.3g} may exceed the RK4 kinetic-term stability "
            f"threshold (~{dt_stable:.3g} for dx_min={dx_min:.3g}, "
            f"hbar={p.hbar}, m_eff={p.m_eff}). "
            "Consider reducing dt or switching to an implicit/spectral solver.",
            UserWarning,
            stacklevel=3,
        )
