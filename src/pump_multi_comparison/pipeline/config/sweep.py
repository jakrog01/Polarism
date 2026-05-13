"""Parameter-sweep expansion for pump_multi_comparison runs."""
from __future__ import annotations

import copy
import re
from typing import Any

from pipeline.config.loader import resolve_delay, resolve_power


def parameter_sweep_enabled(cfg: dict[str, Any]) -> bool:
    """Return whether the config requests direct parameter-sweep mode."""
    return bool(cfg.get("global", {}).get("parameter_sweep", {}).get("enabled", False))


def _fmt_value(value: float) -> str:
    text = f"{float(value):.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _selected_base_scenarios(
    scenarios: list[dict[str, Any]],
    sweep_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = sweep_cfg.get("base_scenarios")
    if not selected:
        return scenarios
    selected_set = {str(name) for name in selected}
    return [sc for sc in scenarios if sc.get("name") in selected_set]


def _apply_absolute_power(
    scenario: dict[str, Any],
    p_reference: float,
) -> None:
    """Resolve all P-relative laser powers to absolute numeric values."""
    laser_defs: list[dict[str, Any]] = scenario.get("lasers", [])
    modifiers: list[dict[str, Any]] = scenario.get("power_modifiers", [])
    ids = [str(ldef.get("id", f"laser_{i}")) for i, ldef in enumerate(laser_defs)]

    for i, ldef in enumerate(laser_defs):
        tags = ldef.get("tags") or []
        power = resolve_power(ldef.get("power"), p_reference)
        for mod in modifiers:
            if ids[i] in mod.get("ids", []) or any(t in tags for t in mod.get("tags", [])):
                power = resolve_power(mod.get("power"), p_reference)
        ldef["power"] = float(power)

    scenario.pop("power_modifiers", None)


def _apply_timing_grid_point(
    scenario: dict[str, Any],
    pulse_separation: float,
    sigma_time: float,
    cutoff_sigma: float,
) -> None:
    """Inject one temporal grid point into a copied scenario.

    All timing_vars expressions are resolved to floats using the given
    sigma_time/pulse_separation/cutoff_sigma, so that build_timing_namespace
    in the builder does not re-evaluate them with wrong threshold values.
    """
    base_ns: dict[str, float] = {
        "sigma_time": sigma_time,
        "pulse_separation": pulse_separation,
        "cutoff_sigma": cutoff_sigma,
    }
    timing_vars = copy.deepcopy(scenario.get("timing_vars", {}))
    resolved: dict[str, float] = {}
    for var, expr in timing_vars.items():
        try:
            resolved[var] = resolve_delay(expr, {**base_ns, **resolved})
        except Exception:
            resolved[var] = expr
    if "cycle_duration" in resolved:
        resolved["cycle_duration"] = pulse_separation
    scenario["timing_vars"] = resolved

    for ldef in scenario.get("lasers", []):
        ldef["sigma_time"] = float(sigma_time)
        ldef["pulse_separation"] = float(pulse_separation)


def expand_parameter_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Expand base scenarios over power and pulse-separation values.

    Returns an expanded config snapshot, expanded scenario names, and a synthetic
    threshold result that lets the existing scenario stage run without invoking
    threshold search.
    """
    if not parameter_sweep_enabled(cfg):
        names = [sc["name"] for sc in cfg.get("scenarios", [])]
        return copy.deepcopy(cfg), names, {}

    expanded_cfg = copy.deepcopy(cfg)
    global_cfg = expanded_cfg["global"]
    sweep_cfg = global_cfg["parameter_sweep"]
    defaults = global_cfg.get("laser_defaults", {})
    threshold_cfg = global_cfg.get("threshold_search", {})

    cutoff_sigma: float = float(defaults.get("cutoff_sigma", 3.0))
    powers = [float(v) for v in sweep_cfg["power_values"]]
    separations = [float(v) for v in sweep_cfg["pulse_separation_values"]]
    sigma_times = [
        float(v)
        for v in sweep_cfg.get(
            "sigma_time_values",
            threshold_cfg.get("sigma_time_values", [defaults.get("sigma_time", 1.0)]),
        )
    ]
    sigma_spaces = [
        float(v)
        for v in sweep_cfg.get(
            "sigma_space_values",
            [defaults.get("sigma_space", 5.0)],
        )
    ]

    base_scenarios = _selected_base_scenarios(expanded_cfg.get("scenarios", []), sweep_cfg)
    expanded_scenarios: list[dict[str, Any]] = []

    multiple_sigma = len(sigma_times) > 1
    multiple_sigma_space = len(sigma_spaces) > 1
    for base in base_scenarios:
        base_name = str(base["name"])
        first_laser_n_pulses = int(
            base.get("lasers", [{}])[0].get("n_pulses", 0)
        ) if base.get("lasers") else 0
        for sigma_time in sigma_times:
            for pulse_sep in separations:
                for sigma_space in sigma_spaces:
                    for power in powers:
                        sc = copy.deepcopy(base)
                        suffix = f"P{_fmt_value(power)}_sep{_fmt_value(pulse_sep)}"
                        if multiple_sigma:
                            suffix += f"_sig{_fmt_value(sigma_time)}"
                        if multiple_sigma_space:
                            suffix += f"_sp{_fmt_value(sigma_space)}"
                        sc["name"] = f"{base_name}_{suffix}"
                        _apply_absolute_power(sc, power)
                        _apply_timing_grid_point(sc, pulse_sep, sigma_time, cutoff_sigma)
                        for ldef in sc.get("lasers", []):
                            ldef["sigma_space"] = float(sigma_space)
                        sc["sweep"] = {
                            "base_scenario": base_name,
                            "power": float(power),
                            "pulse_separation": float(pulse_sep),
                            "sigma_time": float(sigma_time),
                            "sigma_space": float(sigma_space),
                            "n_pulses": first_laser_n_pulses,
                        }
                        expanded_scenarios.append(sc)

    expanded_cfg["scenarios"] = expanded_scenarios
    names = [sc["name"] for sc in expanded_scenarios]

    grid = global_cfg.get("grid", {})
    threshold_stub = {
        "search_completed": True,
        "mode": "parameter_sweep",
        "P_threshold": 1.0,
        "sigma_time": sigma_times[0],
        "pulse_separation": separations[0],
        "cutoff_sigma": cutoff_sigma,
        "sigma_space": sigma_spaces[0],
        "lx": float(grid.get("lx", 100.0)),
        "ly": float(grid.get("ly", 100.0)),
    }
    return expanded_cfg, names, threshold_stub

