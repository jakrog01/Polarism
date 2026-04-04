"""Pre-submit config and cluster-env validation.

All checks are cheap (no GPU, no network, only config parsing) and are run
on the login node before any Slurm allocation is consumed.

Return convention: every public function returns ``list[str]`` of error
messages.  An empty list means valid.

Usage (CLI):
    python -m pipeline.config.validator --config config.yaml --slurm-env slurm.env
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Validate the full parsed config dict.

    Parameters
    ----------
    cfg : dict
        Output of :func:`pipeline.config.loader.load_config`.

    Returns
    -------
    list[str]
        Error messages; empty means valid.
    """
    errors: list[str] = []

    if "global" not in cfg:
        errors.append("Config missing 'global' section")
        return errors

    g = cfg["global"]

    # ── Grid ──────────────────────────────────────────────────────────────────
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
            f"global.grid nx={nx}, ny={ny}: non-power-of-2 degrades FFT performance"
        )

    # ── Solver ────────────────────────────────────────────────────────────────
    solver = g.get("solver", {})
    dt = solver.get("dt")
    total_time = solver.get("total_time")

    if dt is None or not (isinstance(dt, (int, float)) and float(dt) > 0):
        errors.append(f"global.solver.dt={dt!r} must be a positive number")
    if total_time is None or not (
        isinstance(total_time, (int, float)) and float(total_time) > 0
    ):
        errors.append(f"global.solver.total_time={total_time!r} must be positive")

    if dt and total_time:
        dt_f, tt_f = float(dt), float(total_time)
        if dt_f > tt_f:
            errors.append(f"global.solver.dt={dt_f} > total_time={tt_f}")
        if int(tt_f / dt_f) > 100_000_000:
            errors.append(
                f"Estimated n_steps={int(tt_f / dt_f):,} is extremely large — check walltime."
            )

    # ── Physics positivity ────────────────────────────────────────────────────
    physics = g.get("physics", {})
    for key in ("hbar", "m_eff", "gamma_R", "gamma_C"):
        val = physics.get(key)
        if val is not None:
            try:
                fv = float(val)
            except (TypeError, ValueError):
                errors.append(f"global.physics.{key}={val!r} is not a number")
                continue
            if not (math.isfinite(fv) and fv > 0):
                errors.append(f"global.physics.{key}={val!r} must be finite and positive")

    # ── Threshold search ──────────────────────────────────────────────────────
    ts = g.get("threshold_search", {})
    power_values: list = ts.get("power_values", [])
    sigma_time_values: list = ts.get("sigma_time_values", [])
    pulse_sep_values: list = ts.get("pulse_separation_values", [])
    cutoff_sigma: float = float(g.get("laser_defaults", {}).get("cutoff_sigma", 3.0))

    if not power_values:
        errors.append("threshold_search.power_values is empty")
    if not sigma_time_values:
        errors.append("threshold_search.sigma_time_values is empty")
    if not pulse_sep_values:
        errors.append("threshold_search.pulse_separation_values is empty")

    if power_values and not all(
        isinstance(v, (int, float)) and float(v) > 0 for v in power_values
    ):
        errors.append("threshold_search.power_values must all be positive numbers")

    # Pulse-overlap constraint: at least one (sigma_time, pulse_sep) pair must be valid.
    if sigma_time_values and pulse_sep_values:
        valid_pairs = sum(
            1
            for st in sigma_time_values
            for ps in pulse_sep_values
            if cutoff_sigma * float(st) < float(ps) / 2.0
        )
        if valid_pairs == 0:
            errors.append(
                "All (sigma_time, pulse_separation) pairs violate the pulse-overlap "
                "constraint (cutoff_sigma * sigma_time < pulse_sep/2).  "
                "No valid threshold search point exists."
            )

    max_runtime = ts.get("max_runtime_minutes", 0)
    try:
        if float(max_runtime) <= 0:
            errors.append(
                f"threshold_search.max_runtime_minutes={max_runtime!r} must be positive"
            )
    except (TypeError, ValueError):
        errors.append(
            f"threshold_search.max_runtime_minutes={max_runtime!r} is not a number"
        )

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenarios: list = cfg.get("scenarios", [])
    if not scenarios:
        errors.append("No scenarios defined in config")
    names_seen: set[str] = set()
    for sc in scenarios:
        name = sc.get("name", "<unnamed>")
        if name in names_seen:
            errors.append(f"Duplicate scenario name: '{name}'")
        names_seen.add(name)
        if not sc.get("lasers"):
            errors.append(f"Scenario '{name}' has no lasers")
        for i, ldef in enumerate(sc.get("lasers", [])):
            for key in ("x0", "y0"):
                val = ldef.get(key, 0.0)
                try:
                    if not math.isfinite(float(val)):
                        errors.append(
                            f"Scenario '{name}' laser[{i}].{key}={val!r} is not finite"
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"Scenario '{name}' laser[{i}].{key}={val!r} is not a number"
                    )

    return errors


def validate_slurm_env(env_path: str) -> list[str]:
    """Check that slurm.env exists and exports all required variables.

    Parameters
    ----------
    env_path : str
        Absolute path to slurm.env.

    Returns
    -------
    list[str]
        Error messages; empty means valid.
    """
    errors: list[str] = []
    required = {
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "SLURM_MEM",
        "SLURM_GPUS",
        "SLURM_CPUS",
        "SLURM_TIME",
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
        description="Validate pipeline config and Slurm env before submission"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slurm-env", default=None, help="Path to slurm.env")
    args = parser.parse_args()

    from pipeline.config.loader import load_config

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
