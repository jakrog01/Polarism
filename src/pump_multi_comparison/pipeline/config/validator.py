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
from pipeline.config.sweep import parameter_sweep_enabled

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

    errors.extend(_validate_physics_init_keys("global.physics", physics))
    errors.extend(_validate_merged_physics_init("global", physics))

    global_laplacian = solver.get("laplacian")
    if global_laplacian is not None:
        errors.extend(_validate_laplacian("global.solver", global_laplacian))

    sweep_enabled = parameter_sweep_enabled(cfg)
    ts = g.get("threshold_search", {})
    sweep_cfg = g.get("parameter_sweep", {})
    power_values: list = ts.get("power_values", [])
    sigma_time_values: list = ts.get("sigma_time_values", [])
    pulse_sep_values: list = ts.get("pulse_separation_values", [])
    pulse_sep_formula = ts.get("pulse_separation_formula")
    cutoff_sigma: float = float(g.get("laser_defaults", {}).get("cutoff_sigma", 3.0))
    n_pulses = ts.get("n_pulses")

    if sweep_enabled:
        power_values = sweep_cfg.get("power_values", [])
        pulse_sep_values = sweep_cfg.get("pulse_separation_values", [])
        sigma_time_values = sweep_cfg.get(
            "sigma_time_values",
            ts.get("sigma_time_values", [g.get("laser_defaults", {}).get("sigma_time", 1.0)]),
        )
        pulse_sep_formula = None
        if not power_values:
            errors.append("parameter_sweep.power_values is empty")
        if not pulse_sep_values:
            errors.append("parameter_sweep.pulse_separation_values is empty")
        if not sigma_time_values:
            errors.append("parameter_sweep.sigma_time_values is empty")
        base_scenario_filter = sweep_cfg.get("base_scenarios")
        if base_scenario_filter is not None:
            known_scenario_names = {str(sc.get("name", "")) for sc in cfg.get("scenarios", [])}
            for bname in base_scenario_filter:
                if str(bname) not in known_scenario_names:
                    errors.append(
                        f"parameter_sweep.base_scenarios: unknown scenario name '{bname}'"
                    )
        if power_values and not all(
            isinstance(v, (int, float)) and float(v) > 0 for v in power_values
        ):
            errors.append("parameter_sweep.power_values must all be positive numbers")
        if pulse_sep_values and not all(
            isinstance(v, (int, float)) and float(v) > 0 for v in pulse_sep_values
        ):
            errors.append("parameter_sweep.pulse_separation_values must all be positive numbers")
        if sigma_time_values and not all(
            isinstance(v, (int, float)) and float(v) > 0 for v in sigma_time_values
        ):
            errors.append("parameter_sweep.sigma_time_values must all be positive numbers")
        for st in sigma_time_values or []:
            for ps in pulse_sep_values or []:
                if float(ps) < 2.0 * cutoff_sigma * float(st):
                    errors.append(
                        "parameter_sweep pulse separation must be at least one full "
                        f"Gaussian support: pulse_separation={float(ps):.3g}, "
                        f"sigma_time={float(st):.3g}, cutoff_sigma={cutoff_sigma:.3g}"
                    )
    else:
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

        if n_pulses is not None and not (isinstance(n_pulses, int) and n_pulses >= 0):
            errors.append("threshold_search.n_pulses must be a non-negative integer when set")

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
    roi_cfg = out.get("roi_metrics", {})
    if roi_cfg.get("rois") is not None:
        errors.extend(_validate_rois("output.roi_metrics", roi_cfg.get("rois")))

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

        scenario_grid = sc.get("grid")
        if scenario_grid is not None:
            errors.extend(_validate_scenario_grid(name, scenario_grid))

        scenario_solver = sc.get("solver")
        if scenario_solver is not None:
            errors.extend(_validate_scenario_solver(name, scenario_solver))

        scenario_physics = sc.get("physics")
        if scenario_physics is not None:
            merged_physics = {**physics, **scenario_physics}
            errors.extend(_validate_physics_init_keys(f"scenario '{name}' physics", scenario_physics))
            errors.extend(_validate_merged_physics_init(f"scenario '{name}'", merged_physics))

        if sc.get("rois") is not None:
            errors.extend(_validate_rois(name, sc.get("rois")))

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
            if n_pulses is not None and not (isinstance(n_pulses, int) and n_pulses >= 0):
                errors.append(
                    f"Scenario '{name}' laser '{lid}'.n_pulses={n_pulses!r} must be a non-negative integer"
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
            not sweep_enabled
            and isinstance(total_time, (int, float))
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
        elif (
            sweep_enabled
            and isinstance(total_time, (int, float))
            and float(total_time) > 0
        ):
            errors.extend(
                _validate_sweep_timing_budget(
                    scenario_name=name,
                    laser_defs=laser_defs,
                    timing_vars=timing_vars,
                    defaults=g.get("laser_defaults", {}),
                    total_time=float(total_time),
                    sigma_time_values=sigma_time_values,
                    pulse_sep_values=pulse_sep_values,
                )
            )

    return errors


def _validate_scenario_grid(scenario_name: str, grid: Any) -> list[str]:
    """Validate optional per-scenario grid overrides."""
    if not isinstance(grid, dict):
        return [
            f"Scenario '{scenario_name}' grid override must be a mapping, "
            f"got {type(grid).__name__!r}"
        ]
    errors: list[str] = []
    for key in ("nx", "ny"):
        if key in grid:
            try:
                val = int(grid[key])
            except (TypeError, ValueError):
                errors.append(f"Scenario '{scenario_name}' grid.{key}={grid[key]!r} must be an integer")
                continue
            if val < 2:
                errors.append(f"Scenario '{scenario_name}' grid.{key}={grid[key]!r} must be >= 2")
    for key in ("lx", "ly"):
        if key in grid:
            try:
                val = float(grid[key])
            except (TypeError, ValueError):
                errors.append(f"Scenario '{scenario_name}' grid.{key}={grid[key]!r} must be numeric")
                continue
            if not (math.isfinite(val) and val > 0):
                errors.append(f"Scenario '{scenario_name}' grid.{key}={grid[key]!r} must be positive")
    if "grid_type" in grid and grid["grid_type"] not in {"periodic", "closed-interval"}:
        errors.append(
            f"Scenario '{scenario_name}' grid.grid_type={grid['grid_type']!r} "
            "must be 'periodic' or 'closed-interval'"
        )
    return errors


def _validate_rois(scenario_name: str, rois: Any) -> list[str]:
    """Validate scenario-level ROI definitions cheaply."""
    if isinstance(rois, dict):
        rois = [rois]
    if not isinstance(rois, list):
        return [
            f"Scenario '{scenario_name}' rois must be a mapping or list of mappings, "
            f"got {type(rois).__name__!r}"
        ]
    errors: list[str] = []
    dummy_ns = {
        "sigma_space": 1.0,
        "sigma_time": 1.0,
        "pulse_separation": 1.0,
        "cutoff_sigma": 1.0,
        "D": 4.0,
    }
    for i, roi in enumerate(rois):
        if not isinstance(roi, dict):
            errors.append(f"Scenario '{scenario_name}' rois[{i}] must be a mapping")
            continue
        shape = roi.get("shape", "circle")
        if shape != "circle":
            errors.append(
                f"Scenario '{scenario_name}' rois[{i}].shape={shape!r}: only 'circle' is supported"
            )
        if "radius" not in roi:
            errors.append(f"Scenario '{scenario_name}' rois[{i}] is missing required radius")
            continue
        for key in ("x0", "y0", "radius"):
            if key == "radius" and key not in roi:
                continue
            value = roi.get(key, 0.0)
            try:
                resolved = resolve_delay(value, dummy_ns) if isinstance(value, str) else float(value)
            except (TypeError, ValueError) as exc:
                errors.append(f"Scenario '{scenario_name}' rois[{i}].{key}={value!r}: {exc}")
                continue
            if key == "radius" and resolved <= 0.0:
                errors.append(
                    f"Scenario '{scenario_name}' rois[{i}].radius={value!r} must resolve positive"
                )
    return errors


def _validate_laplacian(location: str, laplacian: Any) -> list[str]:
    valid = {"five-point", "isotropic-9pt"}
    if laplacian not in valid:
        return [f"{location}.laplacian={laplacian!r} must be one of {sorted(valid)}"]
    return []


def _validate_physics_init_keys(location: str, physics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    init_mode = physics.get("init_mode")
    if init_mode is not None:
        valid_modes = {"legacy_positive_uniform", "complex_gaussian_zero_mean", "filtered_complex_gaussian"}
        if init_mode not in valid_modes:
            errors.append(
                f"{location}.init_mode={init_mode!r} must be one of {sorted(valid_modes)}"
            )

    init_k_cutoff_um = physics.get("init_k_cutoff_um")
    if init_k_cutoff_um is not None:
        try:
            fv = float(init_k_cutoff_um)
            if not (math.isfinite(fv) and fv > 0):
                errors.append(
                    f"{location}.init_k_cutoff_um={init_k_cutoff_um!r} must be a finite positive number"
                )
        except (TypeError, ValueError):
            errors.append(f"{location}.init_k_cutoff_um={init_k_cutoff_um!r} is not a number")

    init_seed = physics.get("init_seed")
    if init_seed is not None and not isinstance(init_seed, int):
        errors.append(f"{location}.init_seed={init_seed!r} must be an integer")

    return errors


def _validate_merged_physics_init(location: str, physics: dict[str, Any]) -> list[str]:
    init_mode = physics.get("init_mode")
    init_k_cutoff_um = physics.get("init_k_cutoff_um")
    if init_mode == "filtered_complex_gaussian" and init_k_cutoff_um is None:
        return [
            f"{location}: init_mode='filtered_complex_gaussian' requires init_k_cutoff_um to be set"
        ]
    return []


def _validate_scenario_solver(scenario_name: str, solver: Any) -> list[str]:
    """Validate an optional per-scenario solver override block."""
    if not isinstance(solver, dict):
        return [
            f"Scenario '{scenario_name}' solver override must be a mapping, "
            f"got {type(solver).__name__!r}"
        ]
    errors: list[str] = []
    laplacian = solver.get("laplacian")
    if laplacian is not None:
        errors.extend(_validate_laplacian(f"scenario '{scenario_name}' solver", laplacian))

    method = solver.get("method")
    if method is None:
        return errors
    if not isinstance(method, str) or not method.strip():
        errors.append(
            f"Scenario '{scenario_name}' solver.method={method!r} must be a non-empty string"
        )
        return errors
    try:
        from polarism.solver.solver_registry import available_solvers
        import polarism.solver  # noqa: F401 — triggers @register_solver decorators
        known = set(available_solvers.keys())
        if known and method not in known:
            errors.append(
                f"Scenario '{scenario_name}' solver.method={method!r} is not a known solver. "
                f"Known: {sorted(known)}"
            )
    except Exception:
        pass
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


def _validate_sweep_timing_budget(
    scenario_name: str,
    laser_defs: list[dict[str, Any]],
    timing_vars: dict[str, Any],
    defaults: dict[str, Any],
    total_time: float,
    sigma_time_values: list[Any],
    pulse_sep_values: list[Any],
) -> list[str]:
    """Check that each expanded sweep scenario fits inside total_time.

    In sweep mode, the expansion directly sets each laser's pulse_separation to
    the sweep value and resolves cycle_duration = pulse_sep.  This function
    replicates that logic: it builds a timing namespace where cycle_duration is
    replaced by the sweep pulse_sep before evaluating delays.
    """
    errors: list[str] = []
    cutoff_default = float(defaults.get("cutoff_sigma", 3.0))

    for sigma_time in sigma_time_values or []:
        st = float(sigma_time)
        for pulse_sep in pulse_sep_values or []:
            ps = float(pulse_sep)
            if ps < 2.0 * cutoff_default * st:
                continue

            base_ns: dict[str, float] = {
                "sigma_time": st,
                "pulse_separation": ps,
                "cutoff_sigma": cutoff_default,
            }
            resolved_vars: dict[str, float] = {}
            timing_var_errors: list[str] = []
            for var, expr in (timing_vars or {}).items():
                try:
                    resolved_vars[var] = resolve_delay(expr, {**base_ns, **resolved_vars})
                except (ValueError, KeyError) as exc:
                    timing_var_errors.append(
                        f"Scenario '{scenario_name}' timing_vars.{var}={expr!r} "
                        f"failed at sweep point sigma_time={st:.3g}, pulse_sep={ps:.3g}: {exc}"
                    )
            errors.extend(timing_var_errors)
            if timing_var_errors:
                continue
            if "cycle_duration" in resolved_vars:
                resolved_vars["cycle_duration"] = ps
            timing_ns = {**base_ns, **resolved_vars}

            required_time = 0.0
            for ldef in laser_defs:
                merged = {**defaults, **ldef}
                laser_sigma = float(merged.get("sigma_time", st))
                laser_cutoff = float(merged.get("cutoff_sigma", cutoff_default))
                per_laser_ns = {
                    **timing_ns,
                    "sigma_time": laser_sigma,
                    "pulse_separation": ps,
                    "cutoff_sigma": laser_cutoff,
                }
                delay_expr = ldef.get("delay")
                try:
                    delay = resolve_delay(delay_expr, per_laser_ns)
                except (ValueError, KeyError) as exc:
                    errors.append(
                        f"Scenario '{scenario_name}' laser delay={delay_expr!r} "
                        f"failed at sigma_time={st:.3g}, pulse_sep={ps:.3g}: {exc}"
                    )
                    delay = 0.0
                n_pulses_l = ldef.get("n_pulses")
                pulse_span = (int(n_pulses_l) - 1) * ps if isinstance(n_pulses_l, int) and n_pulses_l > 1 else 0.0
                required_time = max(required_time, delay + pulse_span + 2.0 * laser_cutoff * laser_sigma)

            if required_time > total_time:
                errors.append(
                    f"Scenario '{scenario_name}' sweep point sigma_time={st:.3g}, "
                    f"pulse_sep={ps:.3g} requires {required_time:.1f} ps but "
                    f"global.solver.total_time={total_time:.1f} ps"
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
        "PREPARE_REF_MEM", "PREPARE_REF_CPUS", "PREPARE_REF_TIME",
        "SCRATCH",
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
