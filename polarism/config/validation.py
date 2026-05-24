"""Config validation and stability heuristics."""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config

_EXPLICIT_FDM_SOLVERS = {"rk4-fdm", "rk4-fdm-fused", "rk4-cuda", "rk4-cuda-v100"}
_VALID_PRECISIONS = {"single", "double"}
_VALID_INIT_MODES = {"legacy_positive_uniform", "complex_gaussian_zero_mean", "filtered_complex_gaussian"}
_VALID_PROFILE_TYPES = {"sin2", "parabolic"}
_VALID_POWER_DEFINITIONS = {"peak_amplitude", "pulse_energy"}
_RK4_NEG_REAL_AXIS_LIMIT = 2.7852935634052816


class ConfigValidationError(ValueError):
    """Raised when a required config constraint is violated."""


def validate_config(cfg: Config) -> None:
    """Raise ConfigValidationError for invalid values; warn for instability risk."""
    _validate_grid(cfg)
    _validate_physics(cfg)
    _validate_solver(cfg)
    _validate_result(cfg)
    _validate_boundary(cfg)
    _validate_laser(cfg)
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

    for name in ("gamma_C", "gamma_R", "gamma_I", "gamma_A", "R", "R_IA", "R_AI", "kappa"):
        val = getattr(p, name, None)
        if val is not None and val < 0.0:
            raise ConfigValidationError(
                f"physics.{name} must be >= 0 (decay/scattering/transfer rates cannot be negative), "
                f"got {val}"
            )

    for name in ("g_C", "g_R", "g_I", "kinetic_relaxation_eta", "reservoir_diffusion_I", "reservoir_diffusion_A", "reservoir_diffusion_R"):
        val = getattr(p, name, None)
        if val is not None and val < 0.0:
            raise ConfigValidationError(
                f"physics.{name} must be >= 0 (interaction/relaxation/diffusion coefficients cannot be negative), "
                f"got {val}"
            )

    if p.init_eps < 0.0:
        raise ConfigValidationError(
            f"physics.init_eps must be >= 0, got {p.init_eps}"
        )

    if p.init_mode not in _VALID_INIT_MODES:
        raise ConfigValidationError(
            f"physics.init_mode='{p.init_mode}' is not recognised. "
            f"Valid values: {sorted(_VALID_INIT_MODES)}."
        )

    if p.init_mode == "filtered_complex_gaussian":
        k_cut = getattr(p, "init_k_cutoff_um", None)
        if k_cut is None or k_cut <= 0.0:
            raise ConfigValidationError(
                f"physics.init_k_cutoff_um must be > 0 when "
                f"init_mode='filtered_complex_gaussian', got {k_cut!r}."
            )


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


def _validate_boundary(cfg: Config) -> None:
    bc = cfg.boundary_condition
    if bc.strength < 0.0:
        raise ConfigValidationError(
            f"boundary_condition.strength must be >= 0, got {bc.strength}"
        )
    if not (0.0 <= bc.mask_width_percent <= 0.5):
        raise ConfigValidationError(
            f"boundary_condition.mask_width_percent must be in [0, 0.5], "
            f"got {bc.mask_width_percent}"
        )
    if bc.absorption not in {"no-absorption", "mask", "cap"}:
        raise ConfigValidationError(
            f"boundary_condition.absorption='{bc.absorption}' is not recognised. "
            "Valid values: 'no-absorption', 'mask', 'cap'."
        )
    if bc.absorption != "no-absorption" and bc.profile_type not in _VALID_PROFILE_TYPES:
        raise ConfigValidationError(
            f"boundary_condition.profile_type='{bc.profile_type}' is not recognised. "
            f"Valid values: {sorted(_VALID_PROFILE_TYPES)}."
        )


def _validate_laser(cfg: Config) -> None:
    laser = cfg.laser
    if laser.sigma_space <= 0.0:
        raise ConfigValidationError(
            f"laser.sigma_space must be > 0, got {laser.sigma_space}"
        )
    if laser.laser_type == "pulse-gaussian":
        if laser.sigma_time <= 0.0:
            raise ConfigValidationError(
                f"laser.sigma_time must be > 0 for laser_type='pulse-gaussian', "
                f"got {laser.sigma_time}"
            )
        if laser.cutoff_sigma <= 0.0:
            raise ConfigValidationError(
                f"laser.cutoff_sigma must be > 0, got {laser.cutoff_sigma}"
            )
        if laser.n_pulses < 0:
            raise ConfigValidationError(
                f"laser.n_pulses must be >= 0, got {laser.n_pulses}"
            )
        if laser.n_pulses > 1 and laser.pulse_separation <= 0.0:
            raise ConfigValidationError(
                f"laser.pulse_separation must be > 0 when n_pulses > 1, "
                f"got {laser.pulse_separation}"
            )
    if laser.power_definition not in _VALID_POWER_DEFINITIONS:
        raise ConfigValidationError(
            f"laser.power_definition='{laser.power_definition}' is not recognised. "
            f"Valid values: {sorted(_VALID_POWER_DEFINITIONS)}."
        )


def _dx_min(cfg: Config) -> float:
    g = cfg.grid
    if g.grid_type == "periodic":
        dx = g.lx / g.nx
        dy = g.ly / g.ny
    else:
        dx = g.lx / max(g.nx - 1, 1)
        dy = g.ly / max(g.ny - 1, 1)
    return min(dx, dy)


def _diffusion_dt_limit(dx_min: float, diffusion_coeff: float) -> float:
    """Conservative RK4 limit for 2D explicit diffusion with a five-point stencil."""
    return (_RK4_NEG_REAL_AXIS_LIMIT / 8.0) * dx_min ** 2 / diffusion_coeff


def _warn_explicit_stability(cfg: Config) -> None:
    if cfg.solver.method not in _EXPLICIT_FDM_SOLVERS:
        return

    dx_min = _dx_min(cfg)
    p = cfg.physics
    dt = cfg.solver.dt

    dt_kinetic = math.sqrt(2.0) * p.m_eff * dx_min ** 2 / (2.0 * p.hbar)
    if dt > dt_kinetic:
        warnings.warn(
            f"solver.dt={dt:.3g} may exceed the RK4 kinetic-term stability "
            f"threshold (~{dt_kinetic:.3g} for dx_min={dx_min:.3g}, "
            f"hbar={p.hbar}, m_eff={p.m_eff}). "
            "Consider reducing dt or switching to an implicit/spectral solver.",
            UserWarning,
            stacklevel=3,
        )

    diff_coeffs = {
        "reservoir_diffusion_I": getattr(p, "reservoir_diffusion_I", 0.0),
        "reservoir_diffusion_A": getattr(p, "reservoir_diffusion_A", 0.0),
        "reservoir_diffusion_R": getattr(p, "reservoir_diffusion_R", 0.0),
    }
    d_max = max(diff_coeffs.values())
    if d_max > 0.0:
        dt_diff = _diffusion_dt_limit(dx_min, d_max)
        if dt > dt_diff:
            active = ", ".join(f"{k}={v:.3g}" for k, v in diff_coeffs.items() if v > 0.0)
            warnings.warn(
                f"solver.dt={dt:.3g} exceeds the explicit reservoir-diffusion stability "
                f"threshold (~{dt_diff:.3g} for dx_min={dx_min:.3g}, D_max={d_max:.3g}). "
                f"Active diffusion coefficients: {active}. "
                "Reduce dt or disable diffusion terms for this solver.",
                UserWarning,
                stacklevel=3,
            )

    eta = getattr(p, "kinetic_relaxation_eta", 0.0)
    if eta > 0.0:
        r = getattr(p, "R", 0.0)
        gamma_c = getattr(p, "gamma_C", 0.0)
        n_threshold = (gamma_c / r) if r > 0.0 else 1.0
        n_active_ref = max(1.0, n_threshold)
        d_eta = eta * n_active_ref * p.hbar / (2.0 * p.m_eff)
        dt_eta = _diffusion_dt_limit(dx_min, d_eta)
        if dt > dt_eta:
            density_note = (
                f"max(1, gamma_C/R)={n_active_ref:.3g}"
                if r > 0.0
                else "1 because R=0"
            )
            warnings.warn(
                f"solver.dt={dt:.3g} exceeds the stability threshold imposed by "
                f"kinetic_relaxation_eta={eta:.3g} (~{dt_eta:.3g} for "
                f"dx_min={dx_min:.3g}, D_eta={d_eta:.3g}). "
                f"D_eta uses conservative n_active_ref={density_note} "
                f"(condensation-threshold reservoir density); actual instability onset "
                f"scales with max(n_active) at runtime. "
                "Reduce dt or eta.",
                UserWarning,
                stacklevel=3,
            )
