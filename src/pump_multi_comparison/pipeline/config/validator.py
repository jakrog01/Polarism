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

from pipeline.config.loader import build_timing_namespace, resolve_delay, resolve_power

_BASE_TIMING_NAMES: frozenset[str] = frozenset(
    {"sigma_time", "pulse_separation", "cutoff_sigma"}
)


def _resolve_threshold_pulse_sep_values(
    sigma_time_values: list[Any],
    pulse_sep_values: list[Any],
    pulse_sep_formula: Any,
    cutoff_sigma: float,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    if pulse_sep_formula is not None:
        for sigma_time in sigma_time_values:
            st = float(sigma_time)
            pulse_sep = resolve_delay(
                pulse_sep_formula,
                {
                    "sigma_time": st,
                    "cutoff_sigma": cutoff_sigma,
                    "pulse_separation": 0.0,
                },
            )
            pairs.append((st, pulse_sep))
        return pairs

    return [
        (float(st), float(ps))
        for st in sigma_time_values
        for ps in pulse_sep_values
    ]


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

    ts = g.get("threshold_search", {})
    power_values: list = ts.get("power_values", [])
    sigma_time_values: list = ts.get("sigma_time_values", [])
    pulse_sep_values: list = ts.get("pulse_separation_values", [])
    pulse_sep_formula = ts.get("pulse_separation_formula")
    cutoff_sigma: float = float(g.get("laser_defaults", {}).get("cutoff_sigma", 3.0))
    n_pulses = ts.get("n_pulses")

    if not power_values:
        errors.append("threshold_search.power_values is empty")
    if not sigma_time_values:
        errors.append("threshold_search.sigma_time_values is empty")
    if pulse_sep_formula is None and not pulse_sep_values:
        errors.append("threshold_search.pulse_separation_values is empty")

    if power_values and not all(
        isinstance(v, (int, float)) and float(v) > 0 for v in power_values
    ):
        errors.append("threshold_search.power_values must all be positive numbers")

    if pulse_sep_formula is not None:
        try:
            resolve_delay(
                pulse_sep_formula,
                {
                    "sigma_time": 1.0,
                    "cutoff_sigma": cutoff_sigma,
                    "pulse_separation": 1.0,
                },
            )
        except ValueError as exc:
            errors.append(f"threshold_search.pulse_separation_formula={pulse_sep_formula!r}: {exc}")

    if n_pulses is not None and not (isinstance(n_pulses, int) and n_pulses > 0):
        errors.append("threshold_search.n_pulses must be a positive integer when set")

    if sigma_time_values and (pulse_sep_values or pulse_sep_formula is not None):
        valid_pairs = sum(
            1
            for st, ps in _resolve_threshold_pulse_sep_values(
                sigma_time_values,
                pulse_sep_values,
                pulse_sep_formula,
                cutoff_sigma,
            )
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

    dt_mult = ts.get("dt_multiplier")
    if dt_mult is not None:
        try:
            if float(dt_mult) <= 0:
                errors.append(
                    f"threshold_search.dt_multiplier={dt_mult!r} must be positive"
                )
        except (TypeError, ValueError):
            errors.append(
                f"threshold_search.dt_multiplier={dt_mult!r} is not a number"
            )

    cond_frac = ts.get("condensation_fraction")
    if cond_frac is None:
        errors.append("threshold_search.condensation_fraction is missing")
    else:
        try:
            cf = float(cond_frac)
            if not (0 < cf <= 1):
                errors.append(
                    f"threshold_search.condensation_fraction={cond_frac!r} must be in (0, 1]"
                )
        except (TypeError, ValueError):
            errors.append(
                f"threshold_search.condensation_fraction={cond_frac!r} is not a number"
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

        laser_defs: list = sc.get("lasers", [])
        laser_ids: list[str] = [
            str(ldef.get("id", f"laser_{i}")) for i, ldef in enumerate(laser_defs)
        ]
        id_set: set[str] = set(laser_ids)

        if len(id_set) != len(laser_ids):
            errors.append(f"Scenario '{name}' has duplicate laser ids")

        timing_vars_raw = sc.get("timing_vars")
        if timing_vars_raw is not None and not isinstance(timing_vars_raw, dict):
            errors.append(
                f"Scenario '{name}' timing_vars must be a mapping, "
                f"got {type(timing_vars_raw).__name__!r}"
            )
            timing_vars_raw = {}
        timing_vars: dict = timing_vars_raw or {}
        errors.extend(_validate_timing_vars(name, timing_vars))
        allowed_delay_names = _BASE_TIMING_NAMES | frozenset(timing_vars.keys())

        for i, ldef in enumerate(laser_defs):
            lid = laser_ids[i]
            for key in ("x0", "y0"):
                val = ldef.get(key, 0.0)
                try:
                    if not math.isfinite(float(val)):
                        errors.append(
                            f"Scenario '{name}' laser '{lid}'.{key}={val!r} is not finite"
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"Scenario '{name}' laser '{lid}'.{key}={val!r} is not a number"
                    )

            power = ldef.get("power")
            if power is not None and not _valid_power_expr(power):
                errors.append(
                    f"Scenario '{name}' laser '{lid}'.power={power!r} is not a valid "
                    "power expression (use a number, 'P', '1.0P', etc.)"
                )

            if "timing" in ldef:
                errors.append(
                    f"Scenario '{name}' laser '{lid}' uses a 'timing' block which is "
                    "no longer supported. Use a top-level 'delay' field instead "
                    "(e.g. delay: 0.0 or delay: \"pulse_duration\")."
                )

            delay_expr = ldef.get("delay")
            if delay_expr is not None:
                errors.extend(_validate_delay_expr(name, lid, delay_expr, allowed_delay_names))

            n_pulses = ldef.get("n_pulses")
            if n_pulses is not None and not (isinstance(n_pulses, int) and n_pulses > 0):
                errors.append(
                    f"Scenario '{name}' laser '{lid}'.n_pulses={n_pulses!r} must be a positive integer"
                )

        for mod in sc.get("power_modifiers", []):
            mod_power = mod.get("power")
            if mod_power is not None and not _valid_power_expr(mod_power):
                errors.append(
                    f"Scenario '{name}' power_modifiers entry power={mod_power!r} "
                    "is not a valid power expression"
                )
            for mid in mod.get("ids", []):
                if str(mid) not in id_set:
                    errors.append(
                        f"Scenario '{name}' power_modifiers ids references unknown "
                        f"laser id '{mid}'"
                    )

        if (
            isinstance(total_time, (int, float))
            and float(total_time) > 0
            and sigma_time_values
            and (pulse_sep_values or pulse_sep_formula is not None)
        ):
            errors.extend(
                _validate_scenario_timing_budget(
                    scenario_name=name,
                    laser_defs=laser_defs,
                    timing_vars=timing_vars,
                    defaults=g.get("laser_defaults", {}),
                    total_time=float(total_time),
                    sigma_time_values=sigma_time_values,
                    pulse_sep_values=pulse_sep_values,
                    pulse_sep_formula=pulse_sep_formula,
                )
            )

    return errors


def _valid_power_expr(expr: Any) -> bool:
    if expr is None:
        return False
    try:
        resolve_power(expr, 1.0)
        return True
    except (ValueError, TypeError):
        return False


def _validate_delay_expr(
    scenario_name: str,
    laser_id: str,
    expr: Any,
    allowed_names: frozenset[str],
) -> list[str]:
    """Validate a single delay expression against the given allowed names.

    Checks: type, syntax, allowed variable names, and structural validity
    (evaluates with unit dummy values to detect division-by-zero, etc.).
    Non-negativity is verified at build time with actual threshold values.
    """
    if isinstance(expr, (int, float)):
        return []
    if not isinstance(expr, str):
        return [
            f"Scenario '{scenario_name}' laser '{laser_id}' delay={expr!r} "
            f"must be a number or expression string, got {type(expr).__name__!r}"
        ]
    dummy_ns = {name: 1.0 for name in allowed_names}
    try:
        resolve_delay(expr, dummy_ns)
    except ValueError as exc:
        return [
            f"Scenario '{scenario_name}' laser '{laser_id}' delay={expr!r}: {exc}"
        ]
    return []


def _validate_timing_vars(
    scenario_name: str,
    timing_vars: dict[str, Any],
) -> list[str]:
    """Validate a scenario-level ``timing_vars`` block.

    Each variable is evaluated in document order; later variables may
    reference earlier ones plus the base timing names.
    """
    if not isinstance(timing_vars, dict):
        return [
            f"Scenario '{scenario_name}' timing_vars must be a mapping, "
            f"got {type(timing_vars).__name__!r}"
        ]
    errors: list[str] = []
    defined: set[str] = set()
    for var_name, expr in timing_vars.items():
        allowed = _BASE_TIMING_NAMES | frozenset(defined)
        dummy_ns = {name: 1.0 for name in allowed}
        try:
            resolve_delay(expr, dummy_ns)
        except ValueError as exc:
            errors.append(
                f"Scenario '{scenario_name}' timing_vars.{var_name}={expr!r}: {exc}"
            )
        defined.add(var_name)
    return errors


def _validate_scenario_timing_budget(
    scenario_name: str,
    laser_defs: list[dict[str, Any]],
    timing_vars: dict[str, Any],
    defaults: dict[str, Any],
    total_time: float,
    sigma_time_values: list[Any],
    pulse_sep_values: list[Any],
    pulse_sep_formula: Any,
) -> list[str]:
    """Ensure the full finite pulse train of every laser fits inside total_time.

    For each valid (sigma_time, pulse_separation) pair from the threshold
    search grid, computes the worst-case end time across all lasers as:
        delay + (n_pulses - 1) * pulse_separation + 2 * cutoff_sigma * sigma_time
    and checks it does not exceed total_time.
    """
    errors: list[str] = []
    cutoff_default = float(defaults.get("cutoff_sigma", 3.0))

    valid_pairs = [
        (float(st), float(ps))
        for st, ps in _resolve_threshold_pulse_sep_values(
            sigma_time_values,
            pulse_sep_values,
            pulse_sep_formula,
            cutoff_default,
        )
        if cutoff_default * float(st) < float(ps) / 2.0
    ]
    if not valid_pairs:
        return errors

    worst_required = 0.0
    worst_pair: tuple[float, float] | None = None
    for sigma_time, pulse_sep in valid_pairs:
        timing_ns = build_timing_namespace(
            {"sigma_time": sigma_time, "pulse_separation": pulse_sep},
            defaults,
            timing_vars,
        )
        required_time = 0.0
        for ldef in laser_defs:
            merged = {**defaults, **ldef}
            laser_sigma = float(merged.get("sigma_time", sigma_time))
            raw_sep = merged.get("pulse_separation", None)
            if raw_sep is None:
                laser_sep = pulse_sep
            elif isinstance(raw_sep, str):
                laser_sep = resolve_delay(raw_sep, timing_ns)
            else:
                laser_sep = float(raw_sep)
            laser_cutoff = float(merged.get("cutoff_sigma", cutoff_default))
            per_laser_ns = {
                **timing_ns,
                "sigma_time": laser_sigma,
                "pulse_separation": laser_sep,
                "cutoff_sigma": laser_cutoff,
            }
            delay = resolve_delay(ldef.get("delay"), per_laser_ns)
            n_pulses = ldef.get("n_pulses")
            if isinstance(n_pulses, int) and n_pulses > 0:
                pulse_span = (n_pulses - 1) * laser_sep
            else:
                pulse_span = 0.0
            required_time = max(
                required_time,
                delay + pulse_span + 2.0 * laser_cutoff * laser_sigma,
            )
        if required_time > worst_required:
            worst_required = required_time
            worst_pair = (sigma_time, pulse_sep)

    if worst_pair is not None and worst_required > total_time:
        sigma_time, pulse_sep = worst_pair
        errors.append(
            f"Scenario '{scenario_name}' requires at least {worst_required:.1f} ps "
            f"to contain the first full pulse train event, but global.solver.total_time="
            f"{total_time:.1f} ps. Worst valid threshold pair: sigma_time={sigma_time:.3g}, "
            f"pulse_separation={pulse_sep:.3g}."
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
        "NVME_GB",
        "TETYDA_RUNS_BASE",
        "MAX_CONCURRENT_SCENARIOS",
        "FINALIZE_MEM",
        "FINALIZE_CPUS",
        "FINALIZE_TIME",
    }
    _OPTIONAL_KNOWN: frozenset[str] = frozenset({
        "SLURM_QOS",
        "THRESHOLD_TIME",
        "SCENARIO_TIME",
        "FIT_TIME",
        "TIME_RESPONSE_MEM", "TIME_RESPONSE_CPUS", "TIME_RESPONSE_TIME",
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

    unknown = defined - required - _OPTIONAL_KNOWN
    for var in sorted(unknown):
        errors.append(
            f"slurm.env defines unrecognised variable: {var!r}  "
            "(typo in a stage override?)"
        )

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
