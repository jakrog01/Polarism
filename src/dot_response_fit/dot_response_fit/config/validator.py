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


def validate_config(cfg: dict[str, Any], check_files: bool = False) -> list[str]:
    """Validate the full parsed dot-response-fit config dict.

    Parameters
    ----------
    cfg : dict
        Output of :func:`dot_response_fit.config.loader.load_config`.
    check_files : bool
        If true, also verify that local input files referenced by the config
        are readable.  This is intended for submit-time validation on Rysy.

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

    physics = g.get("physics", {})
    gamma_C = physics.get("gamma_C")
    if gamma_C is not None:
        try:
            gc_val = float(gamma_C)
            if abs(gc_val - 0.83) < 0.05:
                errors.append(
                    f"global.physics.gamma_C={gamma_C!r} looks like a typo.  "
                    "Expected ~0.0833 (≈ 1/12), not 0.83."
                )
        except (TypeError, ValueError):
            errors.append(f"global.physics.gamma_C={gamma_C!r} must be a float")

    if "mnist" not in cfg:
        errors.append("Config missing 'mnist' section")
    else:
        mnist = cfg["mnist"]
        data_path = mnist.get("data_path")
        if not data_path:
            errors.append("mnist.data_path is missing or empty")
        elif check_files:
            expanded_path = os.path.expanduser(str(data_path))
            if not os.path.isfile(expanded_path):
                errors.append(
                    f"mnist.data_path does not exist or is not a file on this node: "
                    f"{expanded_path}"
                )
            elif not os.access(expanded_path, os.R_OK):
                errors.append(f"mnist.data_path is not readable: {expanded_path}")
        max_pixels = mnist.get("max_pixels")
        if max_pixels is not None:
            try:
                if int(max_pixels) < 1:
                    errors.append("mnist.max_pixels must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"mnist.max_pixels={max_pixels!r} must be a positive integer")

        n_images = mnist.get("n_images")
        if n_images is not None:
            try:
                if int(n_images) < 1:
                    errors.append("mnist.n_images must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"mnist.n_images={n_images!r} must be a positive integer")

        sample_indices = mnist.get("sample_indices")
        if sample_indices is not None:
            if not isinstance(sample_indices, list):
                errors.append("mnist.sample_indices must be a list of non-negative integers")
            elif len(sample_indices) == 0:
                errors.append("mnist.sample_indices must not be empty (use null to disable)")
            else:
                for i, si in enumerate(sample_indices):
                    try:
                        if int(si) < 0:
                            errors.append(f"mnist.sample_indices[{i}]={si!r} must be >= 0")
                    except (TypeError, ValueError):
                        errors.append(f"mnist.sample_indices[{i}]={si!r} must be an integer")

    if "encoding" not in cfg:
        errors.append("Config missing 'encoding' section")
    else:
        enc = cfg["encoding"]
        for key in ("min_amp", "max_amp", "pulse_width_fwhm", "separation"):
            val = enc.get(key)
            if val is None:
                errors.append(f"encoding.{key} is missing")
            elif not (isinstance(val, (int, float)) and float(val) > 0):
                errors.append(f"encoding.{key}={val!r} must be a positive number")
        min_amp = enc.get("min_amp")
        max_amp = enc.get("max_amp")
        if (
            isinstance(min_amp, (int, float))
            and isinstance(max_amp, (int, float))
            and float(max_amp) <= float(min_amp)
        ):
            errors.append(
                f"encoding.max_amp ({max_amp}) must be strictly greater than "
                f"encoding.min_amp ({min_amp})"
            )

    if "reference" not in cfg:
        errors.append("Config missing 'reference' section")
    else:
        ref = cfg["reference"]
        n_points = ref.get("n_points")
        if n_points is None:
            errors.append("reference.n_points is missing")
        else:
            try:
                if int(n_points) < 10:
                    errors.append(f"reference.n_points={n_points!r} must be >= 10")
            except (TypeError, ValueError):
                errors.append(f"reference.n_points={n_points!r} must be a positive integer")
        nc_source = ref.get("nc_source")
        if nc_source is not None:
            try:
                if float(nc_source) < 0:
                    errors.append(f"reference.nc_source={nc_source!r} must be >= 0")
            except (TypeError, ValueError):
                errors.append(f"reference.nc_source={nc_source!r} must be a non-negative number")

    if "fit" not in cfg:
        errors.append("Config missing 'fit' section")
    else:
        fit = cfg["fit"]
        if not fit.get("sigma_space_values"):
            errors.append("fit.sigma_space_values is empty or missing")
        else:
            for i, v in enumerate(fit["sigma_space_values"]):
                if not (isinstance(v, (int, float)) and float(v) > 0):
                    errors.append(f"fit.sigma_space_values[{i}]={v!r} must be positive")

        for int_key in ("dt_factor", "scalar_stride", "n_ref_points"):
            val = fit.get(int_key)
            if val is not None:
                try:
                    if int(val) < 1:
                        errors.append(f"fit.{int_key}={val!r} must be >= 1")
                except (TypeError, ValueError):
                    errors.append(f"fit.{int_key}={val!r} must be a positive integer")

        n_fit_pixels = fit.get("n_fit_pixels")
        if n_fit_pixels is not None:
            try:
                if int(n_fit_pixels) < 1:
                    errors.append("fit.n_fit_pixels must be >= 1 or null (use all pixels)")
            except (TypeError, ValueError):
                errors.append(f"fit.n_fit_pixels={n_fit_pixels!r} must be a positive integer or null")

        aggregate = fit.get("aggregate", "mean_rmse")
        _supported_aggregates = {"mean_rmse"}
        if aggregate not in _supported_aggregates:
            errors.append(
                f"fit.aggregate={aggregate!r} is not supported. "
                f"Allowed: {sorted(_supported_aggregates)}"
            )

        _supported_observables = {"psi_sq_max", "integrated_psi_sq"}
        observable = fit.get("observable", "psi_sq_max")
        if observable not in _supported_observables:
            errors.append(
                f"fit.observable={observable!r} is not supported. "
                f"Allowed: {sorted(_supported_observables)}"
            )

    out = cfg.get("output", {})
    for stride_key in ("field_record_stride", "scalar_record_stride"):
        val = out.get(stride_key)
        if val is not None:
            try:
                if int(val) < 1:
                    errors.append(f"output.{stride_key}={val!r} must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"output.{stride_key}={val!r} must be a positive integer")

    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        errors.append("No scenarios defined in config")
    names_seen: set[str] = set()
    for sc in scenarios:
        name = sc.get("name", "<unnamed>")
        if name in names_seen:
            errors.append(f"Duplicate scenario name: '{name}'")
        names_seen.add(name)

    return errors


def validate_slurm_env(env_path: str) -> list[str]:
    """Check that slurm.env exists and defines all required variables."""
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
        "PREPARE_REF_MEM",
        "PREPARE_REF_CPUS",
        "PREPARE_REF_TIME",
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
        description="Validate dot-response-fit pipeline config"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slurm-env", default=None, help="Path to slurm.env")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also verify that local input files referenced by the config exist.",
    )
    args = parser.parse_args()

    from dot_response_fit.config.loader import load_config

    cfg = load_config(args.config)
    errors = validate_config(cfg, check_files=args.check_files)
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
