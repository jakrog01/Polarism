"""Pre-submit config and slurm.env validation for the power sweep.

All checks are CPU-only and run on the login node before any Slurm
allocation is consumed.

CLI usage:
    python -m threshold_finder.config.validator --config config.yaml --slurm-env slurm.env
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of error strings; empty means valid.

    Parameters
    ----------
    cfg : dict
        Output of :func:`threshold_finder.config.loader.load_config`.
    """
    errors: list[str] = []

    if "global" not in cfg:
        errors.append("Config missing 'global' section")
        return errors

    g = cfg["global"]

    grid = g.get("grid", {})
    for key in ("nx", "ny", "lx", "ly"):
        val = grid.get(key)
        if val is None:
            errors.append(f"global.grid.{key} is missing")
        elif not (isinstance(val, (int, float)) and float(val) > 0):
            errors.append(f"global.grid.{key}={val!r} must be positive")

    nx, ny = int(grid.get("nx", 0)), int(grid.get("ny", 0))
    if nx > 0 and ny > 0 and (nx & (nx - 1) != 0 or ny & (ny - 1) != 0):
        errors.append(
            f"global.grid nx={nx}, ny={ny}: non-power-of-2 grid degrades FFT performance"
        )

    solver = g.get("solver", {})
    dt = solver.get("dt")
    total_time = solver.get("total_time")

    if dt is None or not (isinstance(dt, (int, float)) and float(dt) > 0):
        errors.append(f"global.solver.dt={dt!r} must be a positive number")
    if total_time is None or not (isinstance(total_time, (int, float)) and float(total_time) > 0):
        errors.append(f"global.solver.total_time={total_time!r} must be positive")

    if isinstance(dt, (int, float)) and isinstance(total_time, (int, float)):
        dt_f, tt_f = float(dt), float(total_time)
        if dt_f > tt_f:
            errors.append(f"global.solver.dt={dt_f} > total_time={tt_f}")
        if int(tt_f / dt_f) > 100_000_000:
            errors.append(
                f"Estimated n_steps={int(tt_f / dt_f):,} is very large — check walltime."
            )

    physics = g.get("physics", {})
    for key in ("hbar", "m_eff", "gamma_R", "gamma_C", "R"):
        val = physics.get(key)
        if val is not None:
            try:
                fv = float(val)
            except (TypeError, ValueError):
                errors.append(f"global.physics.{key}={val!r} is not a number")
                continue
            if not (math.isfinite(fv) and fv > 0):
                errors.append(f"global.physics.{key}={val!r} must be finite and positive")

    sweep = cfg.get("sweep", {})
    p_min = sweep.get("P_min")
    p_max = sweep.get("P_max")
    p_step = sweep.get("P_step")

    for name, val in (("P_min", p_min), ("P_max", p_max), ("P_step", p_step)):
        if val is None:
            errors.append(f"sweep.{name} is missing")
        elif not (isinstance(val, (int, float)) and float(val) > 0):
            errors.append(f"sweep.{name}={val!r} must be a positive number")

    if isinstance(p_min, (int, float)) and isinstance(p_max, (int, float)):
        if float(p_min) >= float(p_max):
            errors.append(f"sweep.P_min={p_min} must be < sweep.P_max={p_max}")

    laser = cfg.get("laser", {})
    power_definition = str(laser.get("power_definition", "peak_amplitude"))

    if (
        isinstance(p_min, (int, float))
        and isinstance(p_max, (int, float))
        and isinstance(p_step, (int, float))
        and isinstance(dt, (int, float))
        and isinstance(total_time, (int, float))
    ):
        R_val = float(physics.get("R", 0.02))
        p_max_f = _estimate_peak_pump_density(float(p_max), laser, power_definition)
        dt_f = float(dt)
        if R_val * p_max_f * dt_f > 0.5:
            errors.append(
                f"Pump-growth guard: R*P_peak_max*dt = {R_val * p_max_f * dt_f:.4f} > 0.5 — "
                f"high-power points (local peak P ≈ {p_max_f}) may drive fast reservoir growth "
                "that RK4 cannot track accurately. Consider reducing dt or P_max."
            )

        import math as _math
        nx_val = grid.get("nx")
        ny_val = grid.get("ny")
        lx_val = grid.get("lx")
        ly_val = grid.get("ly")
        hbar_val = physics.get("hbar")
        m_eff_val = physics.get("m_eff")
        _kin_inputs = (nx_val, ny_val, lx_val, ly_val, hbar_val, m_eff_val)
        if all(isinstance(v, (int, float)) and float(v) > 0 for v in _kin_inputs):
            dx = float(lx_val) / float(nx_val)
            dy = float(ly_val) / float(ny_val)
            k_max = _math.pi / min(dx, dy)
            kinetic_dt_limit = float(m_eff_val) / (float(hbar_val) * k_max ** 2)
            method = str(solver.get("method", "rk4-cuda"))
            spectral_methods = {"split-step-fft", "ip-rk4", "ifrk4-fft-cuda"}
            if method not in spectral_methods and dt_f > 0.5 * kinetic_dt_limit:
                finer_axis = "x" if dx <= dy else "y"
                errors.append(
                    f"Kinetic stability warning: dt={dt_f} exceeds "
                    f"0.5 * m_eff/(hbar*k_max^2) = {0.5 * kinetic_dt_limit:.6f} ps "
                    f"(finer axis: {finer_axis}, d{finer_axis}={min(dx,dy):.3f}, "
                    f"k_max={k_max:.2f}). "
                    "Consider reducing dt or increasing lx/nx (ly/ny)."
                )

    scalar_check_every = sweep.get("scalar_check_every")
    if scalar_check_every is not None:
        try:
            sce = int(scalar_check_every)
            if sce < 1:
                errors.append(
                    f"sweep.scalar_check_every={scalar_check_every!r} must be >= 1"
                )
        except (TypeError, ValueError):
            errors.append(
                f"sweep.scalar_check_every={scalar_check_every!r} must be a positive integer"
            )

    max_concurrent = sweep.get("max_concurrent")
    if max_concurrent is not None:
        try:
            mc = int(max_concurrent)
            if mc < 1:
                errors.append(f"sweep.max_concurrent={max_concurrent!r} must be >= 1")
        except (TypeError, ValueError):
            errors.append(f"sweep.max_concurrent={max_concurrent!r} must be a positive integer")

    for key in ("sigma_space", "sigma_time", "pulse_separation"):
        val = laser.get(key)
        if val is not None:
            try:
                if float(val) <= 0:
                    errors.append(f"laser.{key}={val!r} must be positive")
            except (TypeError, ValueError):
                errors.append(f"laser.{key}={val!r} is not a number")

    if laser.get("laser_type") is not None and laser["laser_type"] != "pulse-gaussian":
        errors.append(
            f"laser.laser_type={laser['laser_type']!r} — only 'pulse-gaussian' is supported"
        )
    if power_definition not in {"peak_amplitude", "pulse_energy"}:
        errors.append(
            f"laser.power_definition={power_definition!r} must be 'peak_amplitude' "
            "or 'pulse_energy'"
        )
    n_pulses = laser.get("n_pulses")
    if n_pulses is not None:
        try:
            if int(n_pulses) < 0:
                errors.append(f"laser.n_pulses={n_pulses!r} must be non-negative")
        except (TypeError, ValueError):
            errors.append(f"laser.n_pulses={n_pulses!r} must be an integer")
    try:
        sigma_time = float(laser.get("sigma_time", 1.0))
        pulse_sep = float(laser.get("pulse_separation", 10.0))
        cutoff_sigma = float(laser.get("cutoff_sigma", 3.0))
        if cutoff_sigma <= 0:
            errors.append(f"laser.cutoff_sigma={cutoff_sigma!r} must be positive")
        if cutoff_sigma * sigma_time >= pulse_sep / 2.0:
            errors.append(
                "laser pulse separation must be at least one full Gaussian support: "
                f"pulse_separation={pulse_sep:.3g}, sigma_time={sigma_time:.3g}, "
                f"cutoff_sigma={cutoff_sigma:.3g}"
            )
    except (TypeError, ValueError):
        pass

    return errors


def _estimate_peak_pump_density(
    power: float,
    laser: dict[str, Any],
    power_definition: str,
) -> float:
    """Estimate local pump peak for the timestep guard."""
    if power_definition != "pulse_energy":
        return power
    sigma_space = float(laser.get("sigma_space", 5.0))
    sigma_time = float(laser.get("sigma_time", 1.0))
    cutoff_sigma = float(laser.get("cutoff_sigma", 3.0))
    spatial_integral = 2.0 * math.pi * sigma_space**2
    temporal_integral = (
        math.sqrt(2.0 * math.pi)
        * sigma_time
        * math.erf(cutoff_sigma / math.sqrt(2.0))
    )
    return power / (spatial_integral * temporal_integral)


def validate_slurm_env(env_path: str) -> list[str]:
    """Check that slurm.env exists and exports all variables required by this pipeline.

    Parameters
    ----------
    env_path : str
        Absolute path to slurm.env.
    """
    errors: list[str] = []
    required = {
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "SLURM_MEM",
        "SLURM_GPUS",
        "SLURM_CPUS",
        "SLURM_TIME",
        "TETYDA_RUNS_BASE",
        "FINALIZE_MEM",
        "FINALIZE_CPUS",
        "FINALIZE_TIME",
    }

    if not os.path.isfile(env_path):
        errors.append(f"slurm.env not found: {env_path}")
        return errors

    defined: set[str] = set()
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=", line)
            if m:
                defined.add(m.group(1))

    for var in sorted(required - defined):
        errors.append(f"slurm.env missing required variable: {var}")

    return errors


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate threshold_finder config and Slurm env before submission"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slurm-env", default=None, help="Path to slurm.env")
    args = parser.parse_args()

    from threshold_finder.config.loader import load_config

    cfg = load_config(args.config)
    errors = validate_config(cfg)

    if args.slurm_env:
        errors.extend(validate_slurm_env(args.slurm_env))

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Config valid: {args.config}")
    if args.slurm_env:
        print(f"Slurm env valid: {args.slurm_env}")


if __name__ == "__main__":
    main()
