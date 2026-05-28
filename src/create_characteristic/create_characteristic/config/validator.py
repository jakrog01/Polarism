"""Pre-submit config and slurm.env validation for the 2D characteristic map.

All checks are CPU-only and run on the login node before any Slurm
allocation is consumed.

CLI usage:
    python -m create_characteristic.config.validator --config config.yaml --slurm-env slurm.env
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
        Output of :func:`create_characteristic.config.loader.load_config`.
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
    if dt is None or not (isinstance(dt, (int, float)) and float(dt) > 0):
        errors.append(f"global.solver.dt={dt!r} must be a positive number")

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
    required_sweep_keys = (
        "energy_min", "energy_max", "energy_step",
        "separation_min", "separation_max", "separation_step",
    )
    for name in required_sweep_keys:
        val = sweep.get(name)
        if val is None:
            errors.append(f"sweep.{name} is missing")
        elif not (isinstance(val, (int, float)) and float(val) > 0):
            errors.append(f"sweep.{name}={val!r} must be a positive number")

    e_min = sweep.get("energy_min")
    e_max = sweep.get("energy_max")
    if isinstance(e_min, (int, float)) and isinstance(e_max, (int, float)):
        if float(e_min) >= float(e_max):
            errors.append(f"sweep.energy_min={e_min} must be < sweep.energy_max={e_max}")

    s_min = sweep.get("separation_min")
    s_max = sweep.get("separation_max")
    if isinstance(s_min, (int, float)) and isinstance(s_max, (int, float)):
        if float(s_min) >= float(s_max):
            errors.append(
                f"sweep.separation_min={s_min} must be < sweep.separation_max={s_max}"
            )

    laser = cfg.get("laser", {})
    power_definition = str(laser.get("power_definition", "peak_amplitude"))
    if power_definition not in {"peak_amplitude", "pulse_energy"}:
        errors.append(
            f"laser.power_definition={power_definition!r} must be 'peak_amplitude' "
            "or 'pulse_energy'"
        )

    if laser.get("laser_type") is not None and laser["laser_type"] != "pulse-gaussian":
        errors.append(
            f"laser.laser_type={laser['laser_type']!r} — only 'pulse-gaussian' is supported"
        )

    n_pulses = laser.get("n_pulses")
    if n_pulses is not None:
        try:
            if int(n_pulses) < 0:
                errors.append(f"laser.n_pulses={n_pulses!r} must be non-negative")
        except (TypeError, ValueError):
            errors.append(f"laser.n_pulses={n_pulses!r} must be an integer")

    sigma_time_val = laser.get("sigma_time")
    cutoff_sigma_val = laser.get("cutoff_sigma")
    if sigma_time_val is not None:
        try:
            if float(sigma_time_val) <= 0:
                errors.append(f"laser.sigma_time={sigma_time_val!r} must be positive")
        except (TypeError, ValueError):
            errors.append(f"laser.sigma_time={sigma_time_val!r} is not a number")
    if cutoff_sigma_val is not None:
        try:
            if float(cutoff_sigma_val) <= 0:
                errors.append(f"laser.cutoff_sigma={cutoff_sigma_val!r} must be positive")
        except (TypeError, ValueError):
            errors.append(f"laser.cutoff_sigma={cutoff_sigma_val!r} is not a number")

    if (
        isinstance(s_min, (int, float))
        and isinstance(sigma_time_val, (int, float))
        and isinstance(cutoff_sigma_val, (int, float))
    ):
        if float(cutoff_sigma_val) * float(sigma_time_val) >= float(s_min) / 2.0:
            errors.append(
                "laser pulse separation_min must be at least one full Gaussian support: "
                f"separation_min={s_min}, sigma_time={sigma_time_val}, "
                f"cutoff_sigma={cutoff_sigma_val}"
            )

    if isinstance(dt, (int, float)):
        dt_f = float(dt)
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
            k_max = math.pi / min(dx, dy)
            kinetic_dt_limit = float(m_eff_val) / (float(hbar_val) * k_max ** 2)
            method = str(solver.get("method", "rk4-cuda"))
            spectral_methods = {"split-step-fft", "ip-rk4", "ifrk4-fft-cuda"}
            if method not in spectral_methods and dt_f > 0.5 * kinetic_dt_limit:
                errors.append(
                    f"Kinetic stability warning: dt={dt_f} exceeds "
                    f"0.5 * m_eff/(hbar*k_max^2) = {0.5 * kinetic_dt_limit:.6f} ps. "
                    "Consider reducing dt or using a spectral method."
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

    threshold_criterion = cfg.get("output", {}).get("threshold_criterion")
    if threshold_criterion is not None:
        try:
            tc = float(threshold_criterion)
            if tc <= 0:
                errors.append(f"output.threshold_criterion={threshold_criterion!r} must be positive")
        except (TypeError, ValueError):
            errors.append(f"output.threshold_criterion={threshold_criterion!r} is not a number")

    return errors


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
        description="Validate create_characteristic config and Slurm env before submission"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slurm-env", default=None, help="Path to slurm.env")
    args = parser.parse_args()

    from create_characteristic.config.loader import load_config

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
