"""Parameter-sweep expansion for the polariton experiment pipeline.

Generic expansion logic lives here.  Experiment-specific modes (probe5, etc.)
are implemented in pipeline/experiments/ and dispatched via the registry.
"""
from __future__ import annotations

import copy
from typing import Any

from pipeline.config.sweep_utils import (
    PULSE_LASER_TYPES,
    apply_absolute_power,
    apply_square_side,
    apply_timing_grid_point,
    fmt_label,
    fmt_value,
    is_reference_power,
    power_reference_expr,
    power_suffix_prefix,
    representative_resolved_power,
    resolve_laser_type,
    selected_base_scenarios,
    uses_absolute_power_label,
)

_PULSE_LASER_TYPES = PULSE_LASER_TYPES
_apply_absolute_power = apply_absolute_power
_apply_square_side = apply_square_side
_apply_timing_grid_point = apply_timing_grid_point
_fmt_label = fmt_label
_fmt_value = fmt_value
_is_reference_power = is_reference_power
_p_expr = power_reference_expr
_power_suffix_prefix = power_suffix_prefix
_representative_resolved_power = representative_resolved_power
_selected_base_scenarios = selected_base_scenarios
_uses_absolute_power_label = uses_absolute_power_label


def parameter_sweep_enabled(cfg: dict[str, Any]) -> bool:
    """Return whether the config requests direct parameter-sweep mode."""
    return bool(cfg.get("global", {}).get("parameter_sweep", {}).get("enabled", False))


def probe5_sweep_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when the config requests probe5 mode."""
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool(ps.get("enabled", False)) and str(ps.get("mode", "")) == "probe5"


def probe5_ring_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should run a ring-only calibration first."""
    if not probe5_sweep_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool((ps.get("ring_calibration") or {}).get("enabled", False))


def probe5_probe_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should also calibrate the center probe alone."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool((ps.get("probe_calibration") or {}).get("enabled", False))


def probe5_assisted_probe_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should calibrate the probe inside a ring trap."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool((ps.get("assisted_probe_calibration") or {}).get("enabled", False))


def probe5_assisted_ring_calibration_enabled(cfg: dict[str, Any]) -> bool:
    """Return True when probe5 should calibrate ring strength for a fixed probe."""
    if not probe5_ring_calibration_enabled(cfg):
        return False
    ps = cfg.get("global", {}).get("parameter_sweep", {})
    return bool((ps.get("assisted_ring_calibration") or {}).get("enabled", False))


def expand_probe5_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Compatibility wrapper for the probe5 experiment-specific expansion."""
    from pipeline.experiments.probe5_trap_gate import Probe5TrapGate
    return Probe5TrapGate().expand_parameter_sweep(cfg)


def expand_generic(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Generic sweep expansion: power × pulse-separation × sigma × square-side axes.

    Used by GenericExperiment and Square4FringeExperiment.  Probe5 expansion
    lives in pipeline/experiments/probe5_trap_gate.py.
    """
    if not parameter_sweep_enabled(cfg):
        names = [sc["name"] for sc in cfg.get("scenarios", [])]
        return copy.deepcopy(cfg), names, {}

    expanded_cfg = copy.deepcopy(cfg)
    global_cfg = expanded_cfg["global"]
    sweep_cfg = global_cfg["parameter_sweep"]
    defaults = global_cfg.get("laser_defaults", {})
    threshold_cfg = global_cfg.get("threshold_search", {})
    power_definition = str(defaults.get("power_definition", "peak_amplitude"))
    power_prefix = power_suffix_prefix(power_definition)

    cutoff_sigma: float = float(defaults.get("cutoff_sigma", 3.0))
    powers = [float(v) for v in sweep_cfg["power_values"]]
    separations: list[float] = [float(v) for v in sweep_cfg.get("pulse_separation_values", [])]
    square_side_values: list[float] = [
        float(v) for v in sweep_cfg.get("square_side_values", [])
    ]
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

    base_scenarios = selected_base_scenarios(expanded_cfg.get("scenarios", []), sweep_cfg)
    expanded_scenarios: list[dict[str, Any]] = []

    multiple_sigma = len(sigma_times) > 1
    multiple_sigma_space = len(sigma_spaces) > 1
    any_pulse_scenario = False

    for base in base_scenarios:
        base_name = str(base["name"])
        is_square4 = base.get("geometry") == "square4"
        scenario_has_pulse = any(
            resolve_laser_type(ldef, defaults) in PULSE_LASER_TYPES
            for ldef in base.get("lasers", [])
        )
        if scenario_has_pulse:
            any_pulse_scenario = True
        use_resolved_power_label = uses_absolute_power_label(base)
        first_laser_n_pulses = int(
            base.get("lasers", [{}])[0].get("n_pulses", 0)
        ) if base.get("lasers") else 0

        if is_square4:
            sep_loop: list[float | None] = [separations[0] if separations else None]
        elif scenario_has_pulse:
            sep_loop = separations if separations else [None]
        else:
            sep_loop = [None]

        side_loop: list[float | None] = square_side_values if is_square4 else [None]

        for sigma_time in sigma_times:
            for square_side in side_loop:
                for pulse_sep in sep_loop:
                    for sigma_space in sigma_spaces:
                        for power in powers:
                            sc = copy.deepcopy(base)
                            apply_absolute_power(sc, power)
                            sweep_power = (
                                representative_resolved_power(sc, power)
                                if use_resolved_power_label
                                else float(power)
                            )
                            side_prefix = (
                                f"a{fmt_value(square_side)}_"
                                if is_square4 and square_side is not None
                                else ""
                            )
                            suffix = f"{side_prefix}{power_prefix}{fmt_value(sweep_power)}"
                            if scenario_has_pulse and pulse_sep is not None and not is_square4:
                                suffix += f"_sep{fmt_value(pulse_sep)}"
                            if multiple_sigma:
                                suffix += f"_sig{fmt_value(sigma_time)}"
                            if multiple_sigma_space:
                                suffix += f"_sp{fmt_value(sigma_space)}"
                            sc["name"] = f"{base_name}_{suffix}"
                            if scenario_has_pulse and pulse_sep is not None:
                                apply_timing_grid_point(sc, pulse_sep, sigma_time, cutoff_sigma)
                            if is_square4 and square_side is not None:
                                apply_square_side(sc, square_side)
                            for ldef in sc.get("lasers", []):
                                ldef["sigma_space"] = float(sigma_space)
                            sweep_meta: dict[str, Any] = {
                                "base_scenario": base_name,
                                "power": sweep_power,
                                "sigma_space": float(sigma_space),
                                "power_definition": power_definition,
                                "power_label": power_prefix,
                            }
                            if is_square4 and square_side is not None:
                                sweep_meta["square_side"] = float(square_side)
                            if scenario_has_pulse:
                                sweep_meta["n_pulses"] = first_laser_n_pulses
                                sweep_meta["sigma_time"] = float(sigma_time)
                                if pulse_sep is not None:
                                    sweep_meta["pulse_separation"] = float(pulse_sep)
                            sc["sweep"] = sweep_meta
                            expanded_scenarios.append(sc)

    expanded_cfg["scenarios"] = expanded_scenarios
    names = [sc["name"] for sc in expanded_scenarios]

    grid = global_cfg.get("grid", {})
    threshold_stub: dict[str, Any] = {
        "search_completed": True,
        "mode": "parameter_sweep",
        "P_threshold": 1.0,
        "power_definition": power_definition,
        "sigma_time": sigma_times[0],
        "cutoff_sigma": cutoff_sigma,
        "sigma_space": sigma_spaces[0],
        "lx": float(grid.get("lx", 100.0)),
        "ly": float(grid.get("ly", 100.0)),
    }
    if any_pulse_scenario and separations:
        threshold_stub["pulse_separation"] = separations[0]
    return expanded_cfg, names, threshold_stub


_expand_generic = expand_generic


def expand_parameter_sweep(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Route to the appropriate experiment's expansion logic via the registry."""
    from pipeline.experiments.registry import get_experiment
    return get_experiment(cfg).expand_parameter_sweep(cfg)
