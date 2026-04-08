"""Pre-submit config validation for the dot-response-fit pipeline.

All checks are cheap (no GPU, no ODE, only config parsing) and are run
before any Slurm allocation is consumed.

Usage (CLI)::

    python -m dot_response_fit.config.validator --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Validate the full parsed dot-response-fit config dict.

    Parameters
    ----------
    cfg : dict
        Output of :func:`dot_response_fit.config.loader.load_config`.

    Returns
    -------
    list[str]
        Error messages; an empty list means valid.
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

    solver = g.get("solver", {})
    dt = solver.get("dt")
    total_time = solver.get("total_time")
    if dt is None or not (isinstance(dt, (int, float)) and float(dt) > 0):
        errors.append(f"global.solver.dt={dt!r} must be a positive number")
    if total_time is None or not (isinstance(total_time, (int, float)) and float(total_time) > 0):
        errors.append(f"global.solver.total_time={total_time!r} must be positive")

    if "time_response" not in cfg:
        errors.append("Config missing 'time_response' section")
    else:
        tr = cfg["time_response"]
        for key in ("t_start", "t_end", "n_points"):
            if key not in tr:
                errors.append(f"time_response.{key} is missing")
        amps = tr.get("amplitudes", {})
        for key in ("start", "end", "count"):
            if key not in amps:
                errors.append(f"time_response.amplitudes.{key} is missing")
        pulse = tr.get("pulse", {})
        for key in ("center", "width_fwhm"):
            if key not in pulse:
                errors.append(f"time_response.pulse.{key} is missing")

    if "fit" not in cfg:
        errors.append("Config missing 'fit' section")
    else:
        fit = cfg["fit"]
        if not fit.get("sigma_space_values"):
            errors.append("fit.sigma_space_values is empty or missing")
        if "observable" not in fit:
            errors.append("fit.observable is missing")
        valid_observables = {"psi_sq_max", "integrated_psi_sq", "center_psi_sq"}
        obs = fit.get("observable")
        if obs is not None and obs not in valid_observables:
            errors.append(
                f"fit.observable={obs!r} is not recognised; "
                f"valid values: {sorted(valid_observables)}"
            )

    out = cfg.get("output", {})
    for _stride_key in ("field_record_stride", "scalar_record_stride"):
        _val = out.get(_stride_key)
        if _val is not None:
            try:
                if int(_val) < 1:
                    errors.append(f"output.{_stride_key}={_val!r} must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"output.{_stride_key}={_val!r} must be a positive integer")

    scenarios = cfg.get("scenarios", [])
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

    return errors


def validate_slurm_env(env_path: str) -> list[str]:
    """Check that slurm.env exists and exports only supported variables."""
    errors: list[str] = []
    required = {
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "SLURM_MEM",
        "SLURM_GPUS",
        "SLURM_CPUS",
        "SLURM_TIME",
        "NVME_GB",
        "TETYDA_RUNS_BASE",
        "MAX_CONCURRENT_SCENARIOS",
        "FINALIZE_MEM",
        "FINALIZE_CPUS",
        "FINALIZE_TIME",
        "TIME_RESPONSE_MEM",
        "TIME_RESPONSE_CPUS",
        "TIME_RESPONSE_TIME",
    }
    optional_known: frozenset[str] = frozenset({
        "SLURM_QOS",
        "THRESHOLD_TIME",
        "SCENARIO_TIME",
        "FIT_TIME",
    })

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

    unknown = defined - required - optional_known
    for var in sorted(unknown):
        errors.append(
            f"slurm.env defines unrecognised variable: {var!r}  "
            "(remove stale settings or fix the typo)"
        )

    return errors


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate dot-response-fit pipeline config"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slurm-env", default=None, help="Path to slurm.env")
    args = parser.parse_args()

    from dot_response_fit.config.loader import load_config

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


if __name__ == "__main__":
    main()
